# this will store all of the layers/class definitions for building a small GPT
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
import math

# custom attention layer
class MaskedMHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, d_k: int = None, d_v: int = None):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        
        # 1. FIX: Correctly set d_k and d_v
        self.d_k = d_k if d_k is not None else d_model // n_heads
        self.d_v = d_v if d_v is not None else d_model // n_heads

        # Projections
        self.w_q = nn.Linear(d_model, n_heads * self.d_k)
        self.w_k = nn.Linear(d_model, n_heads * self.d_k)
        self.w_v = nn.Linear(d_model, n_heads * self.d_v)
        
        self.w_o = nn.Linear(n_heads * self.d_v, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def resize_input(self, x, d_size):
        # (B, S, H * D) -> (B, S, H, D)
        xi_resize = x.view(x.shape[0], x.shape[1], self.n_heads, d_size)
        # (B, S, H, D) -> (B, H, S, D)
        xi_resize = xi_resize.permute(0, 2, 1, 3)
        return xi_resize
    
    def resize_output(self, x):
        # (B, H, S, D) -> (B, S, H, D)
        xo_resize = x.permute(0, 2, 1, 3)
        # (B, S, H, D) -> (B, S, H*D)
        xo_resize = xo_resize.contiguous().view(xo_resize.shape[0], xo_resize.shape[1], -1)
        return xo_resize

    def forward(self, x, mask=None):
        q = x
        k = x
        v = x
        S = q.shape[1]

        # Ideally, move mask to the same device as q
        if mask is None:
            mask = torch.triu(torch.ones(S, S, device=q.device), diagonal=1).bool()

        qi = self.w_q(q)
        ki = self.w_k(k)
        vi = self.w_v(v)

        # Resize with explicit dimensions
        qi_r = self.resize_input(qi, self.d_k)
        ki_r = self.resize_input(ki, self.d_k)
        vi_r = self.resize_input(vi, self.d_v) # Uses d_v now

        # Scaled Dot Product
        scaled = (qi_r @ ki_r.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask
        # Standard PyTorch way: masked_fill(mask_condition, value)
        scaled = scaled.masked_fill(mask == 1, float("-inf"))
        
        attn = torch.softmax(scaled, dim=3)
        attn = self.attn_dropout(attn)

        outputs = attn @ vi_r
        
        outputs_r = self.resize_output(outputs)

        return self.residual_dropout(self.w_o(outputs_r))
    
    # def forward(self, q, k, v, mask=None):
    #     # shape
    #     B = q.shape[0]
    #     S = q.shape[1]

    #     # in projection
    #     Q = self.in_proj(q) # (B, S, d_model)
    #     K = self.in_proj(k) # (B, S, d_model)
    #     V = self.in_proj(v) # (B, S, d_model)

    #     # separate into heads
    #     Q = Q.view(B, S, -1, self.n_heads)  # (B, S, d_k, n_heads)
    #     Q = Q.permute(0, 1, 3 ,2)           # (B, n_heads, s, d_k)
    #     K = K.view(B, S, -1, self.n_heads)  # (B, S, d_k, n_heads)
    #     K = K.permute(0, 1, 3 ,2)           # (B, n_heads, S, d_k)

    #     # calculate
    #     a = (Q @ K.T)/torch.sqrt(self.d_k)  # (B, n_heads, S, S)
    #     A = F.softmax(a, dim=-1)            # (B, n_heads, S, S)


class GELU(nn.Module):
    # GELU layer as implemented in GPT-2
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class FeedForward(nn.Module):
    # feedforward layer as implemented in GPT-2
    # GPT-1
    #   3072 dimensional inner states
    # 
    # GPT-2
    #   not explicitly mentioned in paper, assuming 3072
    # 
    # Attention is All You Need
    #   FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2
    def __init__(self, d_model: int, d_inner: int):
        super().__init__()

        # model parameters
        self.d_model = d_model
        self.d_inner = d_inner

        # define layers
        self.w_in = Linear(d_model, d_inner)
        self.gelu = GELU()
        self.w_out = Linear(d_inner, d_model)

    def forward(self, x):
        x = self.gelu(self.w_in(x))
        x = self.w_out(x)
        return x


class Norm(nn.Module):
    # norm function as implemented in GPT-2
    def __init__(self):
        super().__init__()

        # affine transformation code4
        self.g = nn.Parameter(1)
        self.u = nn.Parameter(0)

    def forward(self, x, axis=-1, epsilon=1e-5):
        # input shape: (B, S, d_model)

        # find the mean and std across the last axis
        m = torch.mean(x, dim=axis)  # (B, S)
        s = torch.std(x, dim=axis)   # (B, S)

        # shift and scale by mean and std
        x = (x - m)/(s + epsilon)

        # apply diagonal affine transformation 
        x = x*self.g + self.u   


# full decoder layer
class DecoderLayer(nn.Module):
    # the decoder layer is formed according to the GPT 2 paper, which is based off of GPT 1
    # GPT-1:
    #   MHA + x
    #   Norm
    #   FeedForward + x
    #   Norm
    # 
    # GPT-2:
    #   Norm
    #   MHA + x
    #   Norm
    #   FeedForward + x
    #   Dropout 
    # 
    #   "We scale the weights of residual layers at initialization by a factor of 1/sqrt(N), where
    #   N is the number of residual layers"
    #       This is not exactly what happens in the released source code for GPT-2.
    #       I have defined a function above that should work, but I think for the initial test runs,
    #       I should use the PyTorch implementation of normalization and see how it does. 
    def __init__(self, d_model: int, d_inner: int, n_heads: int, dropout: float):
        super().__init__()

        # norm1
        self.norm1 = nn.LayerNorm(d_model)

        # MHA
        self.mha = MaskedMHA(d_model, n_heads, dropout)

        # norm2
        self.norm2 = nn.LayerNorm(d_model)

        # feedforwad
        self.ff = FeedForward(d_model, d_inner)

        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mha(self.norm1(x)) + x
        x = self.dropout(self.ff(self.norm2(x))) + x
        return x

# main model class
class GPT(nn.Module):
    # Architecture:
    #   input -> (B, S)
    #   embedding -> (B, S, E)
    #   positional encoding -> (B, S, E)
    #   N x decoder blocks
    #   Linear Layer -> (B, S) - (shifted token outputs?)
    #
    # Notes:
    #   
    def __init__(self, vocab_size: int, seq_len: int, n_layers: int, d_model: int, d_inner:int, n_heads: int, dropout: float):
        super().__init__()

        # embedding layer
        self.embed = nn.Embedding(vocab_size, d_model)  # (B, S, d_model)
        self.seq_len = seq_len

        # positional encoding
        self.positional = nn.Embedding(seq_len, d_model)

        # transformer layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_heads, dropout) for l in range(n_layers)
        ])

        # final layers
        # out proj
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.ln_f = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, targets=None):
        device = x.device
        B = x.shape[0]
        S = x.shape[1]
        pos = torch.arange(0, S, dtype=torch.long, device=device).unsqueeze(0)

        # embedding and positional encoding
        emb = self.embed(x)
        tok_pos = self.positional(pos)
        x1 = self.dropout(emb + tok_pos)
        
        # go through transformer layers
        for layer in self.layers:
            x1 = layer(x1)
        x1 = self.ln_f(x1)
        # final layers - (B, S, D)
        logits = self.out_proj(x1) # (B, S, vocab_size)

        # how is this supposed to work?
        # my guess is that each of the sequence tokens above are represented by
        # one-hot encoded vectors. 
        # so, for the input, the below operation squeezes the input to: (B*S, vocab_size).
        # basically, it just concats the input of every example on top of each other.
        # now we have a square matrix in which the rows represent a token in sequence
        # and the columns represent a vector like this [0, 0,..., 0, 1, 0, ..., 0, 0],
        # or something similar.
        # 
        # now, what is happening on the right with the outputs?
        # the targets are going to look something like this right? -> (B, S)
        # each row contains the correct sequence of tokens.
        # now, what happens when we squeeze it? we have a vector that
        # looks like this -> (B*S). Each element is the correct token number.
        #  
        # so, let's recap, for each row in (B*S), on the input we have some logits 
        # ([0, 0,..., 0, 1, 0, ..., 0, 0]) and in the input we have the correct
        # token, in other words, the index of these logits. 
        # 
        # my guess is that this cross entropy converts these indices to logit vectors
        # that it can use to compute the loss as the proper formula dictates. 
        # it looks like this is how it works in the documentation. 
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt, tokenizer, max_new_tokens=100, temperature=0.6, top_k=10):
        # Source: minGPT (Karpathy)
        idx = torch.tensor(tokenizer.tokenize(prompt), dtype=torch.long).unsqueeze(0).to('cuda')
        for _ in range(max_new_tokens):
            # 2. Crop context if it gets too long (block_size)
            idx_cond = idx[:, -self.seq_len:]
            
            # 3. Get predictions
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature  # Apply Temperature
            
            # 4. Optional: Top-K Sampling (The Fix!)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Mask out everything that isn't in the top_k
                logits[logits < v[:, [-1]]] = -float('Inf')

            # 5. Apply Softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # 6. Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 7. Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        # Decode
        return tokenizer.detokenize(idx[0].tolist())
