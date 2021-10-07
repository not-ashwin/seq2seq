import random

import torch
from torch import nn

from encoder import Encoder
from decoder import Decoder


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
        # trg --> [trg_len, batch size]
        
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





enc = Encoder(10, 5, 3, 3)
dec = Decoder(10, 5, 3, 3)

input = torch.randint(0, 9, (10, 100))
inp_len = torch.randint(1, 11, (100,))

out, (hidden, cell) = enc(input, inp_len)

model = Seq2Seq(enc, dec, -1, 'cpu')

outputs = model(input, inp_len, input)

print(outputs.shape)