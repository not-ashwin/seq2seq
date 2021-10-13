import random
import operator
from queue import PriorityQueue

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from seq2seq.encoder import Encoder
from seq2seq.decoder import Decoder


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''Node object for storing info about trg word
        @param hiddenstate:
        @param previousNode:
        @param wordId:
        @param logProb:
        @param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        #TODO: find a suitable reward function

        return self.logp / float(self.length - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.length < other.length

    def __gt__(self, other):
        return self.length > other.length


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        """The seq2seq model for consolidating the Enocder Decoder into a single class
        @param encoder (Encoder): Encoder class for encoding sentences 
        @param decoder (Decoder): Decoder class for decoding sentences
        @src_pad_idx (int): the index used for padding
        @device (str): The device on which the model needs to run (cuda or cpu)"""

        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.src_pad_idx = src_pad_idx

        self.device = device

    def create_masks(self, src):
        return (src != self.src_pad_idx).permute(1, 0)

    def forward(self, src, src_len, trg, teacher_forcing=0.5):
        """Forward function for the seq2seq model
        @param src (torch tensor): the source sentences
        @param src_len (torch tensor): the lengths of the source sentences
        @param trg (torch tensor): the target sentences
        @param teacher_forcing (float): ratio for using target input for faster training. Needs to be between 0 and 1
        returns outputs (torch tensor): the tensor of the predicted target sentence"""

        # src --> [src len, batch size]
        # src_len --> [batch size]
        # trg --> [trg len, batch size]
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store the output values
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, (hidden, cell) = self.encoder(src, src_len)

        # First token for the trg is always <sos>
        dec_inputs = trg[0, :]

        masks = self.create_masks(src)

        for t in range(1, trg_len):
            output, (hidden, cell), _ = self.decoder(dec_inputs, hidden, cell, encoder_outputs, masks)

            outputs[t] = output

            teacher_force = random.random() < teacher_forcing 
            
            top1 = output.argmax(1)

            dec_inputs = trg[t] if teacher_force else top1

        return outputs

    def predict(self, src, src_len, trg_pad_token, max_len=50):
        """Predict the decoder after the training is complete
        @param src (Pytorch tensor): input sentence for encoder
        @param src_len (Pytorch tensor): lneghts of the input sentences
        @param trg_pad_token (int): the padding token index for target"""

        # src --> [src len, batch size]
        # src_len --> [batch size]

        with torch.no_grad():
            enc_output, (hidden, cell) = self.encoder(src, src_len)

        masks = self.create_masks(src)

        trg_indexes = [[1,] * src.shape[1]]  # Target <SOS> token idx

        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes)[-1, :].to(self.device)

            with torch.no_grad():
                output, (hidden, cell), _ = self.decoder(trg_tensor, hidden, cell, enc_output, masks)

            _, pred_token = output.data.topk(1)

            trg_indexes.append([_.item() for _ in pred_token])

        trgs = np.array(trg_indexes)
        trgs = [trgs[1:,i] for i in range(trgs.shape[1])]

        return trgs

    def beam_decode(self, src, src_len, trg_sos_token, trg_pad_token, max_len=50):
        """Using beam search algo for decoding"""

        beam_width = 10 #TODO: add in params
        topk = 1 #TODO: add in params

        decoded_batch = []

        with torch.no_grad():
            enc_output, (hidden, cell) = self.encoder(src, src_len)

        masks = self.create_masks(src)

        for i in range(src.shape[1]):
            dec_hidden = hidden[i,:].unsqueeze(0)
            dec_cell = cell[i,:].unsqueeze(0)
            encoder_output = enc_output[:,i,:].unsqueeze(1)

            dec_input = torch.LongTensor([1]).to(self.device) # SOS token

            mask = masks[i,:].unsqueeze(0)

            end_nodes = []
            number_required = min(topk + 1, topk - len(end_nodes))

            node = BeamSearchNode((dec_hidden, dec_cell), None, dec_input, 0, 1)
            nodes = PriorityQueue()

            nodes.put((-node.eval(), node))
            qsize = 1

            while True:
                if qsize > 100: break # MAX length for beam search

                score, n = nodes.get()

                decoder_input = n.wordid
                dec_hidden, dec_cell = n.h

                if n.wordid.item() == 2 and n.prevNode != None:
                    end_nodes.append((score, n))
            
                    if len(end_nodes) >= number_required:
                        break
                    else:
                        continue

                dec_output, (dec_hidden, dec_cell), _ = self.decoder(decoder_input, dec_hidden, dec_cell, encoder_output, mask)
                dec_output = F.log_softmax(dec_output, dim=1)
                # dec_output --> [1, output dim]

                score, indexes = torch.topk(dec_output, beam_width)

                for newk in range(beam_width):
                    decoded_t = indexes[0][newk].view(-1, 1)[0]
                    sc = score[0][newk].item()

                    node = BeamSearchNode((dec_hidden, dec_cell), n, decoded_t, n.logp + sc, n.length + 1)
                    sc = -node.eval()
                    nodes.put((sc, node))

                qsize += 1


            if len(end_nodes) == 0:
                end_nodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for sc, n in sorted(end_nodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid.item())

                # back trace queue
                while n.prevNode:
                    n = n.prevNode
                    utterance.append(n.wordid.item())

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.extend(utterances)

        return decoded_batch


if __name__ == "__main__":
    enc = Encoder(10, 5, 3, 3)
    dec = Decoder(10, 5, 3, 3)

    input = torch.randint(0, 9, (10, 100))
    inp_len = torch.randint(1, 11, (100,))

    out, (hidden, cell) = enc(input, inp_len)

    model = Seq2Seq(enc, dec, -1, 'cpu')

    outputs = model(input, inp_len, input)

    print(outputs.shape)