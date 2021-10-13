import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        """Attention layer for seq2seq
        @param enc_hidden_dim (int): The hidden dimension for encoder
        @param dec_hidden_dim (int): The hidden dimension for decoder"""

        super(Attention, self).__init__()

        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim) 
        # enc_hidden_dim * 2 because of bidirectional in Encoder.

        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, enc_outputs, masks):
        """Forward function for Attentional layer
        @param hidden (torch tensor): The hidden layer of the previous decoder output
        @param enc_outputs (torch tensor): The output from the encoder layer
        @param masks (torch tensor): Tensor to mask out the padded values in enc output
        returns attention (torch tensor): The attention values of the encoder outputs"""

        # hidden --> [batch size, dec hidden dim]
        # enc_outputs --> [src_len, batch_size, enc hidden dim * 2]

        batch_size = enc_outputs.shape[1]
        src_len = enc_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) # Repeat the hidden state T times to match that of enc_output
        # hidden --> [batch size, src len, dec hidden dim]

        encoder_outputs = enc_outputs.permute(1, 0, 2) # Change enc_outputs as batch first to match with hidden
        # encoder_outputs --> [batch size, src len, enc hidden dim * 2]

        concat = torch.cat((hidden, encoder_outputs), dim=2) # concat the prev hidden state and current enc state
        # concat --> [batch size, src len, enc hidden dim * 2 + dec hidden dim]

        energy = torch.tanh(self.attn(concat))
        # energy --> [batch size, src len, dec hidden dim]

        attention = self.v(energy)
        # attention = [batch size, src len, 1]
        attention = attention.squeeze(2)
        # attention --> [batch size, src len]

        attention = attention.masked_fill(masks == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, tgt_vocab, emb_dim, enc_hidden_dim, dec_hidden_dim, dropout=0.5):
        """Decoder layer for seq2seq model
        @param tgt_vocab (int): the vocabulary of the target
        @param emb_dim (int): the dimension of the embedding layer
        @param enc_hidden_dim (int): the hidden dimension of the encoder layer
        @param dec_hidden_dim (int): the hidden dimension of the decoder layer
        @param dropout (int): the value of the dropout must be between 0 and 1"""
        super(Decoder, self).__init__()

        self.output_dim = tgt_vocab
        
        self.hidden_attention = Attention(enc_hidden_dim, dec_hidden_dim)
        self.cell_attention = Attention(enc_hidden_dim, dec_hidden_dim)

        self.embedding = nn.Embedding(tgt_vocab, emb_dim)

        self.rnn = nn.LSTM((enc_hidden_dim * 4) + emb_dim, dec_hidden_dim, bidirectional=False)

        self.fc = nn.Linear((enc_hidden_dim * 4) + emb_dim + dec_hidden_dim, self.output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_inputs, hidden, cell, enc_outputs, masks):
        """Forward function for the decoder layer
        @param dec_inputs (torch tensor): The value of the previous decoder outputs
        @param hidden (torch tensor): The hidden state of the previous decoder value
        @param cell (torch tensor): The cell state of the previous decoder value
        @param enc_outputs (torch tensor): The output value from the encoder
        @param masks (torch tensor): To mask out the padded values from encoder output
        returns decoder_output, (hidden, cell), attention : the decoder outputs along with the hidden state and cell state and attention for debugging"""

        # dec_inputs --> [batch size] (single word)
        # hidden --> [batch size, dec hidden dim]
        # cell --> [batch size, dec hidden dim]
        # enc_outputs --> [src len, batch size, enc hidden dim * 2]

        dec_inputs = dec_inputs.unsqueeze(0)    # as sequence length is of 1 (LSTM accepts sequence first)
        # dec_inputs --> [1, batch size] 

        embedded = self.embedding(dec_inputs)
        #embedded --> [1, batch size, emb dim]

        hidden_a = self.hidden_attention(hidden, enc_outputs, masks)
        cell_a = self.cell_attention(cell, enc_outputs, masks)

        # hidden_a --> [batch size, src len]
        # cell_a --> [batch size, src len]

        hidden_a = hidden_a.unsqueeze(1)
        cell_a = cell_a.unsqueeze(1)
        #hidden_a --> [batch size, 1, src len]
        #cell_a --> [batch size, 1, src len]

        enc_outputs =enc_outputs.permute(1, 0, 2)
        # enc_outputs --> [batch size, src len, enc hidden dim * 2]

        hidden_weighted = torch.bmm(hidden_a, enc_outputs) # The reshaping was done so that the matrix multiplication could be done 
        cell_weighted = torch.bmm(cell_a, enc_outputs)
        # hidden_weighted --> [batch size, 1, enc hidden dim * 2]
        # cell_weighted --> [batch size, 1, enc hidden dim * 2]
        
        hidden_weighted = hidden_weighted.permute(1, 0, 2)
        cell_weighted = cell_weighted.permute(1, 0, 2)
        # hidden_weighted --> [1, batch size, enc hidden dim * 2]
        # cell_weighted --> [1, batch size, enc hidden dim * 2]

        rnn_input = torch.cat((embedded, hidden_weighted, cell_weighted), dim=2)
        # rnn_input = torch.cat((embedded, hidden_weighted), dim=2)
        # rnn_input --> [1, batch size, (enc hidden dim * 2) + emb dim]

        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))

        # output --> [1, batch size, dec hidden dim]
        # hidden --> [1, batch size, dec hidden dim]
        # cell --> [1, batch size, dec hidden dim]

        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        hidden_weighted = hidden_weighted.squeeze(0)
        cell_weighted = cell_weighted.squeeze(0)

        prediction = self.fc(torch.cat((embedded, output, hidden_weighted, cell_weighted), dim=1))
        # prediction = self.fc(torch.cat((embedded, output, hidden_weighted), dim=1))
        # prediction --> [batch size, output dim]

        return prediction, (hidden.squeeze(0), cell.squeeze(0)), hidden_a.squeeze(1)
