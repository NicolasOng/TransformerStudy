import argparse

import torch
from torch.nn import functional as F

from tqdm import tqdm

from transformer import DecoderOnlyTransformer
from load_data import load_char_data, estimate_loss
from generate import generate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 64
block_size = 512
eval_iters = 256

parser = argparse.ArgumentParser(description="This file trains a transformer model and produces sample output.")
parser.add_argument('-ms', type=str, default="tiny", help='Model Size')
parser.add_argument('-ds', type=str, default="shakespeare", help='Dataset')
args = parser.parse_args()
model_size = args.ms
dataset = args.ds

if (dataset == "shakespeare"):
    dataset_filename = "./data/tinyshakespeare.txt"
elif (dataset == "personal"):
    dataset_filename = "./data/2023-07-03_knowledge.txt"

encode, decode, get_batch, vocab_size = load_char_data(dataset_filename, block_size, batch_size, device)

if (model_size == "tiny"):
    #Transformer (custom; tiny)
    model = DecoderOnlyTransformer(vocab_size, block_size, 128, 6, 2, 0)
elif (model_size == "small"):
    #Transformer (custom; small)
    model = DecoderOnlyTransformer(vocab_size, block_size, 256, 6, 4, 0)
elif (model_size == "base"):
    #Transformer (base model)
    model = DecoderOnlyTransformer(vocab_size, block_size, 512, 6, 8, 0.1)
elif (model_size == "big"):
    #Transformer (big)
    model = DecoderOnlyTransformer(vocab_size, block_size, 1024, 6, 16, 0.3)

model = model.to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters in the model.')

# create a PyTorch optimizer
learning_rate = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

max_iters = 5000
eval_interval = 500
for iter in tqdm(range(max_iters)):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model, eval_iters, get_batch)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        torch.save(model.state_dict(), 'model_state_' + model_size + "_" + str(iter) + '.pth')

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

#torch.save(model.state_dict(), 'model_state.pth')
#model.load_state_dict(torch.load('model_state.pth'))