from os import sep
import random
import numpy as np

from nltk import tokenize
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import torch

from seq2seq import Encoder, Decoder, Seq2Seq
from utils.trainer import train, evaluate
from utils.metrics import calculate_bleu
from utils.datasets import Lang, TrainingData

import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

torch.backends.cudnn.deterministic = True
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

df = pd.read_csv(r"Sentence pairs in English-Tagalog - 2021-10-11.tsv", sep='\t', header=None)

df[1] = df[1].str.lower()
df[3] = df[3].str.lower()

train_expressions, test = train_test_split(df, test_size=0.3, random_state=42)

en = Lang(train_expressions[1], tokenizer=word_tokenize)

tl = Lang(train_expressions[3].to_list(), tokenizer=word_tokenize)


training_data = TrainingData(train_expressions[1], train_expressions[3], en, tl)
validation_data = TrainingData(test[1], test[3], en, tl)

params_train = {'batch_size': 32,
          'shuffle': True,
          'collate_fn': training_data.collate_fn}

params_val = {'batch_size': 128,
          'shuffle': True,
          'collate_fn': training_data.collate_fn}

training_generator = DataLoader(training_data, **params_train)
validation_generator = DataLoader(validation_data, **params_val)

INPUT_VOCAB = len(en.vocab)
OUTPUT_VOCAB = len(tl.vocab)
ENC_EMB_DIM = 258
DEC_EMB_DIM = 258
ENC_HIDDEN_DIM = 512
DEC_HIDDEN_DIM = 512
TRG_PAD_IDX = tl.pad_idx


print("starting to train now")

enc = Encoder(INPUT_VOCAB, ENC_EMB_DIM, ENC_HIDDEN_DIM, DEC_HIDDEN_DIM, pad_idx=en.pad_idx, dropout=0.5)
dec = Decoder(OUTPUT_VOCAB, DEC_EMB_DIM, ENC_HIDDEN_DIM, DEC_HIDDEN_DIM, dropout=0.5)
model = Seq2Seq(enc, dec, en.pad_idx, device='cuda:0')

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

model = model.to('cuda:0')

best_val_loss = -float('inf')

for _ in range(100):
    loss = train(model, training_generator, optimizer, criterion, 1, 'cuda:0')
    torch.cuda.empty_cache()
    # val_loss = evaluate(model, validation_generator, criterion, 'cuda:0' )
    score = calculate_bleu(validation_generator, model, tl, 'cuda:0')
    print("Training loss: ", loss, "Eval BLEU: ", score)

    if score >= best_val_loss:
        best_val_loss = score
        torch.save(model.state_dict(), 'conll_ner.pt')


model.load_state_dict(torch.load('conll_ner.pt'))

score = calculate_bleu(validation_generator, model, tl, 'cuda:0')

print(score)

text = ["she likes music.", "he likes music"]
print(text)
input_tensor, tensor_len = en.encode(text, return_pt=True)
input_tensor = input_tensor.to('cuda:0')
x = model.beam_decode(input_tensor.permute(1, 0), tensor_len, None, tl.pad_idx)

print([" ".join(tl.decode(_)) for _ in x])