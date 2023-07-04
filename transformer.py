'''
Implements the transformer according to the
"All You Need Is Attention" paper.
'''

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


# Token Embeddings

def positional_encoding(seq_len, d_model):
    # Create a positional encoding matrix of shape (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        '''
        vocab_size: number of symbols in the vocabulary
        d_model: the dimension for the token+position embedding
        '''
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        # x: (batch, seq_len)
        # output: (batch, seq_len, d_model)
        _, seq_len = x.shape
        return self.token_embeddings(x) + positional_encoding(seq_len, self.d_model).to(x.device)
    

# Scaled Dot-Product Attention

class Attention(nn.Module):
    def __init__(self, seq_len_q, seq_len_kv, dropout, masked=True):
        '''
        seq_len_q: the amount of tokens in Q
        seq_len_kv: the amount of tokens in K&V
        dropout: percentage of parameters to dropout during training
        '''
        super().__init__()
        self.register_buffer('tril', torch.tril(torch.ones(seq_len_q, seq_len_kv)))
        self.dropout = nn.Dropout(dropout)
        self.masked = masked

    def forward(self, Q, K, V):
        # Q: (batch_size, seq_len_q, d_qk)
        # K: (batch_size, seq_len_kv, d_qk)
        # V: (batch_size, seq_len_kv, d_v)
        batch_size, seq_len_q, d_qk = Q.shape
        _, seq_len_kv, d_v = V.shape

        # compare the queries to the keys,
        # so there's a DP for each pair.
        # wei: (batch_size, seq_len_q, seq_len_kv)
        wei = Q @ K.transpose(-2,-1) * d_qk**-0.5

        # mask the weights so queries at position i
        # don't look at keys in position i + 1
        if (self.masked):
            wei = wei.masked_fill(self.tril[:seq_len_q, :seq_len_kv] == 0, float('-inf'))

        # softmax so the sum of the weights (for each query) goes to 1.
        wei = F.softmax(wei, dim=-1)

        # dropout some weights during training can be a good idea.
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        # out: (batch, seq_len_q, d_v)
        out = wei @ V

        # returns a tensor where, for each given query,
        # a weighted sum of the values is created,
        # where the weights are determined by
        # comparing queries to keys.
        return out


# Multi-Head Attention

class MultiHeadAttention(nn.Module):

    def __init__(self, h, masked, seq_len_q, seq_len_kv, d_model_q, d_model_k, d_model_v, d_qk, d_v, d_model, dropout):
        super().__init__()
        '''
        h: number of attention heads
        masked: if tokens can "look ahead"
        seq_len_q: number of tokens in Q
        seq_len_kv: number of tokens in K&V
        d_model_q: the embedding dim of Q
        d_model_k: the embedding dim of K
        d_model_v: the embedding dim of V
        d_qk: the embedding dim to project Q&K to
        d_v: the embedding dim to project V to
        d_model: the embedding dim of the output
        dropout percentage of parameters to dropout during training
        '''
        self.heads = nn.ModuleList([Attention(seq_len_q, seq_len_kv, dropout, masked) for _ in range(h)])
        # WQ, WK, WV: Martices that project the given QKVs
        # to appropriate dimensions
        self.WQs = nn.ModuleList([nn.Linear(d_model_q, d_qk, bias=False) for _ in range(h)])
        self.WKs = nn.ModuleList([nn.Linear(d_model_k, d_qk, bias=False) for _ in range(h)])
        self.WVs = nn.ModuleList([nn.Linear(d_model_v, d_v, bias=False) for _ in range(h)])
        # WO: Matrix that scales the concat'd attention outputs
        # to the output dimension.
        self.WO = nn.Linear(h * d_v, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        h_outs = []
        for i, h in enumerate(self.heads):
            h_outs.append(h(self.WQs[i](Q), self.WKs[i](K), self.WVs[i](V)))
        out = torch.cat(h_outs, dim=-1)
        out = self.dropout(self.WO(out))
        return out


# Feed-Forward

class FeedFoward(nn.Module):
    def __init__(self, d_io, d_h, dropout):
        '''
        d_io: the input and output embedding dim
        d_h: the hidden embedding dim
        dropout percentage of parameters to dropout during training
        '''
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_io, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_io),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Transformer Blocks

class DecoderOnlyBlock(nn.Module):

    def __init__(self, seq_len, h, d_model, dropout):
        '''
        seq_len: how many tokens the input x will be.
        h: the number of heads for the multi-head attention
        d_model: the input/output embedding dim for this block
        dropout: percentage of parameters to dropout during training.
        '''
        super().__init__()
        head_size = d_model // h
        self.mmha = MultiHeadAttention(h, True, seq_len, seq_len, d_model, d_model, d_model, head_size, head_size, d_model, dropout)
        self.ffwd = FeedFoward(d_model, d_model * 4, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.mmha(x, x, x))
        x = self.ln2(x + self.ffwd(x))
        return x

class EncoderBlock(nn.Module):

    def __init__(self, seq_len_e, h, d_model, dropout):
        '''
        seq_len_e: how many tokens the input of the encoder will be.
        h: the number of heads for the multi-head attention
        d_model: the input/output embedding dim for this block.
            same as the decoder (to be compatible)
        dropout: percentage of parameters to dropout during training.
        '''
        super().__init__()
        head_size = d_model // h
        self.mha = MultiHeadAttention(h, False, seq_len_e, seq_len_e, d_model, d_model, d_model, head_size, head_size, d_model, dropout)
        self.ffwd = FeedFoward(d_model, d_model * 4, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.mha(x, x, x))
        x = self.ln2(x + self.ffwd(x))
        return x
    
class DecoderBlock(nn.Module):

    def __init__(self, seq_len_d, seq_len_e, h1, h2, d_model, dropout):
        '''
        seq_len_d: how many tokens the input of the decoder will be
        seq_len_e: how many tokens the input of the encoder will be
        h1: the number of heads for the masked multi-head attention
        h2: the number of heads for the multi-head attention
        d_model: the input/output embedding dim for this block
            same as the encoder (to be compatible)
        dropout: percentage of parameters to dropout during training.
        '''
        super().__init__()
        head_size1 = d_model // h1
        head_size2 = d_model // h2
        self.mmha = MultiHeadAttention(h1, True, seq_len_d, seq_len_d, d_model, d_model, d_model, head_size1, head_size1, d_model, dropout)
        self.mha = MultiHeadAttention(h2, False, seq_len_d, seq_len_e, d_model, d_model, d_model, head_size2, head_size2, d_model, dropout)
        self.ffwd = FeedFoward(d_model, d_model * 4, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, di, eo):
        '''
        di: decoder input
        eo: encoder output
        '''
        x = self.ln1(di + self.mmha(di, di, di))
        x = self.ln2(x + self.mha(x, eo, eo))
        x = self.ln3(x + self.ffwd(x))
        return x


# Models

class DecoderOnlyTransformer(nn.Module):

    def __init__(self, vocab_size, seq_len, d_model, n, h, dropout):
        '''
        vocab_size: number of symbols in the vocab
        seq_len: length of the input sequence
        d_model: input embedding dimension for this model
        n: number of transformer layers
        h: number of heads in each transformer layer
        dropout: percentage of parameters to dropout during training
        '''
        super().__init__()
        self.embedding_layer = TokenEmbedding(vocab_size, d_model)
        self.blocks = nn.Sequential(*[DecoderOnlyBlock(seq_len, h, d_model, dropout) for _ in range(n)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets=None):
        '''
        x: the list of encoded tokens
        targets: the actual next token for each input token
        logits: logits for each token predicting the next token
        loss: loss of the predicted logits compared to the targets
        '''
        '''
        For every token:
        - create an embedding based on its symbol/position
        - pass it through n layers, where in each layer it
          adds all the (projected) prior tokens together,
          paying more attention to some. It does this multiple
          times, then concats all the weighted sums. Then it
          reprojects this vector to a new representation. This
          vector gets fed into a feed-forward network, and then
          into the next layer.
        - after n layers, each token is represented by a vector
          that has "payed attention" to all the prior vectors
          in different ways (weights).
        - these tokens are then each used to predict the next
          token by applying a linear layer from the token's dim
          to the vocab dim.
        - use softmax + sampling to predict the next token.
        - or use calculated loss w/ target preds to update parameters.
        '''
        '''
        to use for inference:
        - input a list of tokens
        - every token will consider all the previous tokens
          and make a next-token prediction over the vocab
        - sample from the last next-token prediction,
          append that to the list of tokens.
        - feed this new list back into the model.
        '''
        x = self.embedding_layer(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            batch, seq_len, embed_dim = logits.shape
            logits = logits.view(batch*seq_len, embed_dim)
            targets = targets.view(batch*seq_len)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

class EncoderDecoderTransformer(nn.Module):

    def __init__(self, vocab_size_e, vocab_size_d, seq_len_e, seq_len_d, d_model, h_e, h_d1, h_d2, n_e, n_d, dropout):
        '''
        vocab_size_e: size of vocab for the encoder's language
        vocab_size_d: size of vocab for the decoder's language
        seq_len_e: sequence length for the encoder's input/output
        seq_len_d: sequence length for the decoder's input/output
        d_model: embedding dimension for the model [512]
        h_e: number of heads in the encoder's multi-head attention [8]
        h_d1: number of heads in the decoder's first mha [8]
        h_d2: number of heads in the decoder's second mha [8]
        n_e: number of transformer layers in the encoder [6]
        n_d: number of transformer layers in the decoder [6]
        dropout: percentage of parameters to dropout during training [0.1]
        []: the value used in the paper
        '''
        super().__init__()
        # encoder stuff
        self.embedding_layer_e = TokenEmbedding(vocab_size_e, d_model)
        self.blocks_e = nn.Sequential(*[EncoderBlock(seq_len_e, h_e, d_model, dropout) for _ in range(n_e)])
        # decoder stuff
        self.embedding_layer_d = TokenEmbedding(vocab_size_d, d_model)
        self.blocks_d = nn.Sequential(*[DecoderBlock(seq_len_d, seq_len_e, h_d1, h_d2, d_model, dropout) for _ in range(n_d)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size_d) 

    def forward(self, x_e, x_d, targets=None):
        '''
        x_e: the input sentence to the encoder,
            to be translated
        x_d: the input sentence to the decoder,
            to predict the next token for all tokens
        targets: the actual next token for each input x_d token
        logits: logits for each token predicting the next token
        loss: loss of the predicted logits compared to the targets
        '''
        '''
        for every token in the encoder's sentence,
        - look at all previous tokens and combine them to
          form a new representation.
        - do this many times to get a final representation.
        for every token in the decoder's sentence,
        - look at all previous tokens and combine them to
          form a new representation.
        - look at all tokens from the encoder's final output.
          make a new representation by combining them.
        - feed this new representation back into the decoder,
          until you get the final representation.
        - with the final representation, use a linear layer
          to get predictions over the decoder's vocab.
        - softmax + sample to predict the next tokens, feed
          this back into the decoder to predict the next token.
        - or backprop loss to update the params.
        '''
        # get the encoder's final output
        # TODO: This only needs to be run once, while the decoder
        # needs to be run for each token generated. Save the output
        # somewhere.
        x_e = self.embedding_layer_e(x_e)
        x_e = self.blocks_e(x_e)

        # get the decoder's final output using the encoder's output
        x_d = self.embedding_layer_d(x_d)
        x_d = self.blocks_d(x_d, x_e)

        # get the next token predictions
        x_d = self.ln_f(x_d)
        logits = self.lm_head(x_d)

        if targets is None:
            loss = None
        else:
            batch, seq_len_d, embed_dim = logits.shape
            logits = logits.view(batch*seq_len_d, embed_dim)
            targets = targets.view(batch*seq_len_d)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
