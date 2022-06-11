import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim import Optimizer

import random
import numpy as np

# %% id="k0XGXhzRTsqB"
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        x = self.encode(src, src_mask)
        return self.decode(x, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        #print(src.shape) # [32, 72]
        #print(src_mask.shape) # [32, 1, 72]
        embed = self.src_embed(src)
        #print(embed.shape) # [32, 72, 512]
        enc = self.encoder(embed, src_mask)
        #print(enc.shape) # [32, 72, 512]
        return enc

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# %% id="NKGoH2RsTsqC"
class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

# %% id="2gxTApUYTsqD"
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# %% id="xqVTz9MkTsqD"
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        #self.norm = BatchNorm1d(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for i, layer in enumerate(self.layers):
            x = layer(x, mask) # [32, 72, 512]
        #x = self.norm(x.transpose(1,2)).transpose(1,2)
        return self.norm(x)


# %% id="3jKa_prZTsqE"
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        self.features = features

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# %% id="U1P7zI0eTsqE"
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        #self.norm = BatchNorm1d(size)
        self.dropout = nn.Dropout(dropout)
        #self.fc = nn.Linear(size*2, size)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        #x = self.norm(x.transpose(1,2)).transpose(1,2)
        return x + self.dropout(sublayer(self.norm(x)))
        #return self.fc(torch.cat(( x, self.dropout(sublayer(self.norm(x))) ), dim=-1))


# %% id="qYkUFr6GTsqE"
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# %% [markdown] id="7ecOQIhkTsqF"
# ### Decoder
#
# The decoder is also composed of a stack of $N=6$ identical layers.
#

# %%
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        #self.norm = BatchNorm1d(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        #print(x.shape) # [32, 71, 512]
        #print(memory.shape) # [32, 72, 512]
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        #print(x.shape) # [32, 71, 512]
        #x = self.norm(x.transpose(1,2)).transpose(1,2)
        return self.norm(x)


# %% id="M2hA1xFQTsqF"
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size # 512
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        #print("m:", m.shape) # [32, 72, 512]
        #print("x:", x.shape) # [32, 71, 512]
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# %% id="QN98O2l3TsqF"
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


# %% id="qsoVxS5yTsqG"
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # [32, 4, 72, 128]
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # [32, 4, 72, 72]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1) # [32, 4, 72, 72]
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# %% id="D2LBMKCQTsqH"
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        #print(d_model, h, self.d_k) # 512, 8, 64
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # "Implements Figure 2"
        #if query.size(1) != key.size(1):
        #    verbose = True
        verbose = False
        
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        if verbose:
            print(query.shape) # [32, 71, 512]
            print(key.shape) # [32, 72, 512]
            print(value.shape) # [32, 72, 512]
        
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        if verbose:
            print(query.shape) # [32, 4, 71, 128]
            print(key.shape) # [32, 4, 72, 128]
            print(value.shape) # [32, 4, 72, 128]

        #print(query.shape) # [32, 4, 72, 128]
        #print(key.shape) # [32, 4, 72, 128]
        #print(value.shape) # [32, 4, 72, 128]
        #print(mask.shape) # [32, 1, 1, 72]
 
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        ) # [32, 4, 72, 128]

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


# %% id="6HHCemCxTsqH"
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


# %% id="pyrChq9qTsqH"
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# %% id="zaHGD4yJTsqH"
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


# %% id="mPe1ES0UTsqI"
def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=4, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# %% id="zUz3PdAnVg4o"
def rate(step, model_size, factor, warmup, scale, shift, c_steps, rate_decay):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    if step <= warmup:
        return factor * (model_size**(-0.5) * step * warmup ** (-1.5))
    
    #if c_steps <= step <= int(c_steps * 1.5):
    #    p = step // c_steps
    #    factor /= 2 ** p ###
    #elif step > int(c_steps * 1.5):
    #    p = step // (c_steps // 2)
    #    factor /= 2 ** (p-1)
    if step >= c_steps:
        p = step / c_steps
        factor /= rate_decay ** p
        #p = step // c_steps
        #factor /= rate_decay ** p

    #if step < warmup:
    #    scale = 1.
    return factor * (
        model_size ** (-0.5) * (step-shift) ** (-0.5*scale)
        #model_size ** (-0.5) * min(step ** (-0.5 * scale), step * warmup ** (-1.5))
    )

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, eta_min = 0., num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            current_step = max(current_step, 1)
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, eta_min + (1 - eta_min) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# %% id="shU2GyiETsqK"
class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

