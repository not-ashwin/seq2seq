import torch
from torch.utils.data import Dataset

class Lang(Dataset):
    def __init__(self, expressions, tokenizer=None, init_token='<sos>', eos_token='<eos>', pad_token='<pad>'):
        """The Lang dataset to load the training expressions in a pytorch compatible manner
        @param expressions (iterable): Iterable of the training expressions
        @tokenizer (Function): the function to break sentences into tokens"""

        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        if tokenizer:
            self.tokenizer = tokenizer
            
        else:
            self.tokenizer = lambda x: x.split()

        self.build_vocab(expressions)

    def build_vocab(self, expressions):
        """Build the vocabulary"""
        self.vocab = {'<unk>': 1, self.init_token: 1, self.eos_token: 1, self.pad_token: 1}

        corpus = map(self.tokenizer, expressions)

        for sentence in corpus:
            for token in sentence:
                if self.vocab.get(token):
                    self.vocab[token] += 1
                else:
                    self.vocab[token] = 1

        self.word2idx = {word: idx for idx, word in enumerate(self.vocab.keys())}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.pad_idx = self.word2idx[self.pad_token]

    def encode(self, sentence, return_pt=False):
        """Encode sentences into idx
        @param sentence (str or iterable): sentence(s) to encode
        @param return_pt (bool): whether to return as pytorch tensor or list"""

        if isinstance(sentence, str):
            tokens = [[self.word2idx.get(tok, 0) for tok in [self.init_token] + self.tokenizer(sentence) + [self.eos_token]]]
            token_lens = [len(tokens[0])]
        else:
            tokens = []
            token_lens = []
            for sent in sentence:
                tokens.append([self.word2idx.get(tok, 0) for tok in [self.init_token] + self.tokenizer(sent) + [self.eos_token]])
                token_lens.append(len(tokens[-1]))
            
            max_len = max(token_lens)

            tokens = [tok + [self.pad_idx] * (max_len - len(tok)) for tok in tokens]

        if return_pt:
            return torch.tensor(tokens), torch.LongTensor(token_lens)

        return tokens, token_lens

    def decode(self, idx):
        """Decode a list of indexes
        @param idx (iterable): iterable of indexes of tokens
        returns tokens (list of str): list of the tokens from idx"""
        return [self.idx2word.get(i, '<unk>') for i in idx if i not in (self.word2idx[self.init_token], self.word2idx[self.eos_token], self.pad_idx)]


class TrainingData(Dataset):
    def __init__(self, src, trg, src_vocab, trg_vocab):
        """Dataset to be used for training"""

        self.src = src
        self.trg = trg

        assert len(src) == len(trg), "The size of input and target should be the same"

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_tokens = [self.src_vocab.init_token] + self.src_vocab.tokenizer(self.src[idx]) + [self.src_vocab.eos_token]
        trg_tokens = [self.trg_vocab.init_token] + self.trg_vocab.tokenizer(self.trg[idx]) + [self.trg_vocab.eos_token]

        return src_tokens, len(src_tokens), trg_tokens, len(trg_tokens)

    def collate_fn(self, data):
        _, src_lengths, trg, trg_lengths = zip(*data)
        tokens = torch.tensor([token + [self.src_vocab.pad_idx] * (max(src_lengths) - length) for token, length, _, _ in data])
        tokens = tokens.permute(1, 0)
        trg = torch.tensor([token + [self.trg_vocab.pad_idx] * (max(trg_lengths) - length) for _, _, token, length in data])
        trg = trg.permute(1, 0)
        return tokens, torch.tensor(src_lengths), trg