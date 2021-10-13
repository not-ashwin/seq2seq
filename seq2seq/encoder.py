import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, enc_hidden_dim, dec_hidden_dim, pad_idx=0.0, dropout=0.5):
        """Encoder class for seq2seq using RNN
        @param vocab_dim (int): vocabulary size of the input data
        @param embedding_dim (int): the dimension for the embedding layer of input
        @param enc_hidden_dim (int): hidden size for RNN for encoders
        @param dec_hidden_dim (int): hidden size for decoder RNN
        @param pad_idx (int): index for padding
        @param dropout (float): value of dropout between 0 to 1
        """

        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_dim, embedding_dim)

        self.rnn = nn.LSTM(embedding_dim, enc_hidden_dim, bidirectional=True)

        self.cell_resize = nn.Linear(enc_hidden_dim*2, dec_hidden_dim)

        self.hidden_resize = nn.Linear(enc_hidden_dim*2, dec_hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.pad_idx = pad_idx

    def forward(self, src, src_len):
        """Forward function for encoder
        @param src (torch tensor): input for model
        @param src_len (torch tensor): actual lengths of the sentences in the batch
        returns output, (hidden, cell): The enc output and the hidden states of the encoders"""

        # src --> [src len, batch size]
        # src_len --> [batch size]

        embedded = self.dropout(self.embedding(src))
        #embedded --> [src len, batch size, embedding dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'), enforce_sorted=False)
        # need to put src len on cpu

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output, padding_value=self.pad_idx) # returns unpacked tensors and lengths in tuple

        # outputs --> [src len, batch size, enc_hidden_dim * 2]
        # hidden --> [n layers * num of directions, batch size, dec_hidden_dim]
        # cell --> [n layers * num of directions, batch size, enc_hidden_dim]

        hidden = self.hidden_resize(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        hidden = self.dropout(torch.tanh(hidden))
        # hidden --> [batch size, dec_hidden_dim]

        cell = self.dropout(torch.tanh(self.cell_resize(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim = 1))))
        # cell --> [batch size, dec_hidden_dim]

        return outputs, (hidden, cell)
