from torchtext.data.metrics import bleu_score
import numpy as np

def calculate_bleu(iterator, model, lang, device='cpu'):
    """Function to calculate the bleu score of translations
    @param iterator (DataLoader iterator): Iterator for testing bleu score
    @param model (Pytorch model): The seq2seq model to check bleu score
    @param lang (Lang object): the trg lang object for decoding
    @param device (str): either cpu or cuda"""
    from tqdm import tqdm
    pred_trgs = []
    reference = []

    model = model.eval()
    model = model.to(device)
    model.device = device

    for batch in tqdm(iterator, desc='Calc Bleu', leave=False):
        src, src_len, trg = batch
        src = src.to(device)

        pred_tokens = model.beam_decode(src, src_len.to('cpu'), lang.pad_idx, lang.pad_idx)

        pred_trgs.extend([lang.decode(np.array(_)) for _ in pred_tokens])

        reference.extend([[lang.decode(np.array(_))] for _ in trg.permute(1, 0)])


    return bleu_score(pred_trgs, reference)





