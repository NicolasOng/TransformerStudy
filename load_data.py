import torch

def load_char_data(filename, block_size, batch_size, device):
    # get the text from the text file
    # "./data/tinyshakespeare.txt"
    # "./data/2023-07-03_knowledge.txt"
    input_file_path = filename
    with open(input_file_path, 'r') as f:
        data = f.read()
    print(f"num characters in dataset: {len(data):,}")

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # create the train and test splits from the text file
    encoded_data = torch.tensor(encode(data), dtype=torch.long)
    n = int(len(encoded_data)*0.9)
    train_data = encoded_data[:n]
    val_data = encoded_data[n:]

    print(f"train has {len(train_data):,} tokens")
    print(f"val has {len(val_data):,} tokens")

    def get_batch(split):
        '''
        Generates two tensors. Each of size (batch, block).
        x is [batch] character sequences of length [block], each character encoded as a token/int.
        y is the targets of the character sequences in x.
        For example, for x=[abcde,fghij], y=[bcdef,ghijk],
        given the sequence x[n][:m], you want to predict y[n][m]
        '''
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
    return encode, decode, get_batch, vocab_size

@torch.no_grad()
def estimate_loss(model, eval_iters, get_batch):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
