import torch
from torch.nn import functional as F

from tqdm import tqdm

from transformer import DecoderOnlyTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 16
block_size = 64
eval_iters = 200

# get the text from the text file
input_file_path = "./data/tinyshakespeare.txt"
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

@torch.no_grad()
def estimate_loss():
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

def generate(model, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in tqdm(range(max_new_tokens)):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        # get the predictions
        logits, loss = model.forward(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

model = DecoderOnlyTransformer(vocab_size, block_size, 128, 4, 4, 0)
model = model.to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
learning_rate = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

max_iters = 2000
eval_interval = 500
for iter in tqdm(range(max_iters)):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
model.eval()
print(decode(generate(model, context, max_new_tokens=2000)[0].tolist()))

torch.save(model.state_dict(), 'model_state.pth')
#model.load_state_dict(torch.load('model_state.pth'))