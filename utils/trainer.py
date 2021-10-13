from torch.nn.utils import clip_grad_norm_


def train(model, training_generator, optimizer, criterion, clip, device='cpu'):
    """Function to train a single epoch of training data
    @param model (Pytorch model): The model we want to train
    @param training_generator (iterable): the training data
    @param optimizer (Pytorch optim object): Optimizing the gradient descent
    @param criterion (Pytorch loss object): The loss function to be minimized
    @param clip (float): the value above which to clip the gradient descent
    returns epoch loss"""
    from tqdm import tqdm

    model.train() #to activate the dropouts defined in the model
    model = model.to(device)
    epoch_loss = 0

    for batch in tqdm(training_generator, desc='Training', leave=False):
        src, src_len, trg = batch
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()

        output = model(src, src_len, trg)

        # trg --> [trg_len, batch_size]
        # output --> [trg_len, batch_size, trg_vocab_size]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim) # as first one is always the <sos> token
        trg = trg[1:].reshape(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]
        # For the outputs to be fed into a crossentropy function

        loss = criterion(output, trg)

        loss.backward()

        clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(training_generator)


def evaluate(model, iterator, criterion, device='cpu'):
    """Function to evaluate the model on evaluation dataset
    @param model (Pytorch model): The model to evaluate
    @param iterator (iterator): the evalutation dataset
    @param criterion (Pytorch Loss object): the loss to be calcluated
    returns evaluation loss"""
    from tqdm import tqdm

    model.eval() # to ignore the dropout values
    model = model.to(device)
    eval_loss = 0

    for batch in tqdm(iterator, desc='Evaluation', leave=False):
        src, src_len, trg = batch
        src, trg = src.to(device), trg.to(device)

        output = model(src, src_len, trg, teacher_forcing=0)

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim) # as first one is always the <sos> token
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)

        eval_loss += loss.item()

    return eval_loss / len(iterator)
