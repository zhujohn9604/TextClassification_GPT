#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:15:09 2020

@author: zhuchen
"""

import copy
import json
import math
import re
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)

# activation functions
ACT_FNS = {
    'relu': nn.ReLU,
    'swish': swish,
    'gelu': gelu
}


class LayerNorm(nn.Module):
    # sample-wise
    def __init__(self, n_state, e=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(n_state))  # gamma
        self.b = nn.Parameter(torch.zeros(n_state))  # beta
        self.e = e

    def forward(self, x):  # x: [batch_size x n_ctx x n_state]
        mu = x.mean(-1, keepdim=True)  # x: [batch_size x len x 1]
        var = (x - mu).pow(2).mean(-1, keepdim=True)  # x: [batch_size x len x 1]
        x = (x - mu) / torch.sqrt(var + self.e)
        return self.g * x + self.b


class Conv1D(nn.Module):
    """
    nf: out_size
    nx: in_size
    """

    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        self.rf = rf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)  # w: [nx, nf]
            self.b = Parameter(torch.zeros(nf))
        else:
            raise NotImplementedError

    def forward(self, x):  # x: [batch_size x n_ctx x n_embd] n_ctx <==> len_seq
        if self.rf == 1:
            return x @ self.w + self.b
        else:
            raise NotImplementedError
            
            
class Attention(nn.Module):
    # Input size: [batch_size x n_ctx x n_embd]
    def __init__(self, nx, n_ctx, cfg, scale=False):
        super().__init__()
        n_state = nx # n_embd
        assert n_state % cfg.n_head == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        # register_buffer: saved in the state_dict(), but will not be trained
        self.n_head = cfg.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

    def _attn(self, q, k, v):
        """
        q: [batch_size x n_head x n_ctx x d]
        k: [batch_size x n_head x n_ctx x d]
        v: [batch_size x n_head x n_ctx x d]
        """
        w = torch.matmul(q, k.transpose(-1, -2))  # [batch_size x n_head x n_ctx x n_ctx]
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        b = self.b[:, :, :w.size(-2), :w.size(-1)]  # crop it as the same with w
        w = w * b + -1e9 * (1 - b) # mask
        w = nn.Softmax(dim=-1)(w)  # [batch_size x n_head x n_ctx(Q) x n_ctx(K)]
        w = self.attn_dropout(w)
        return torch.matmul(w, v)  # [batch_size x n_head x n_ctx(Q) x d(V)]

    def merge_heads(self, x):
        """
        merge heads
        """
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch_size x n_ctx(Q) x n_head x d(V)]
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # [batch_size x n_ctx(Q) x (n_head * d(V))]

    def split_heads(self, x):
        new_x_shape = x.size()[:-1] + (self.n_head, -1)  # [batch_size x n_ctx x n_head x dv] 
        # n_state = n_head x dv
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # [batch_size x n_head x n_ctx x d]

    def forward(self, x):  # [batch_size x n_ctx x n_embd]
        x = self.c_attn(x)  # [batch_size x n_ctx x (n_state * 3)]
        query, key, value = x.split(self.split_size, dim=2)
        # query: [batch_size x n_ctx x n_state]
        # key: [batch_size x n_ctx x n_state]
        # value: [batch_size x n_ctx x n_state]
        query = self.split_heads(query)  # query: [batch_size x n_head x n_ctx x d] (n_state = n_head x d)
        key = self.split_heads(key)  # key: [batch_size x n_head x n_ctx x d]
        value = self.split_heads(value)  # value: [batch_size x n_head x n_ctx x d]
        attn = self._attn(query, key, value)  # [batch_size x n_head x n_ctx(Q) x d(V)]
        attn = self.merge_heads(attn)  # [batch_size x n_ctx x n_state(n_head * d)]
        attn = self.c_proj(attn)  # [batch_size x n_ctx x n_state]
        attn = self.resid_dropout(attn)
        return attn


class MLP(nn.Module):
    """
    Input: [batch_size x len x nx]
    Output: [batch_size x len x nx]
    """

    def __init__(self, n_state, cfg):
        super().__init__()
        nx = cfg.n_embd
        self.c_fc = Conv1D(n_state, 1, nx)  # W: [nx x n_state] nx -- > n_state
        self.c_proj = Conv1D(nx, 1, n_state)  # W: [n_state x nx]
        self.act = ACT_FNS[cfg.afn]  # self.act = gelu
        self.dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h = self.c_proj(h)
        return self.dropout(h)


class Block(nn.Module):
    """
    Input: [batch_size x n_ctx x nx]
    Output: [batch_size x n_ctx x n_state]
    n_embd = n_state = nx for the ease of skip-connection
    """

    def __init__(self, n_ctx, cfg, scale=False):
        super().__init__()
        nx = cfg.n_embd
        self.attn = Attention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)  # Position-wise Feed-Forward Networks
        self.ln_2 = LayerNorm(nx)

    def forward(self, x): # [batch_size x n_ctx x n_embd]
        a = self.attn(x)  # [batch_size x n_ctx x n_state]
        n = self.ln_1(x + a)  # [batch_size x n_ctx x n_state]
        m = self.mlp(n)  # [batch_size x n_ctx x n_state]
        h = self.ln_2(m + n)
        return h


class TransformerModel(nn.Module):
    """ Transformer model """
    # Input size: [bs x 1 x n_ctx x 2] bs:batch_size & n_ctx: seq_len
    def __init__(self, cfg, vocab=40990, n_ctx=512):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.embd_pdrop)
        block = Block(n_ctx, cfg, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(cfg.n_layer)])
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x):
        x = x.view(-1, x.size(-2), x.size(-1)) # x: [batch_size x n_ctx x 2]
        e = self.dropout(self.embed(x))
        # [batch_size x n_ctx x 2 x n_embd] ==> 1 x n_embd: word_embd ; 1 x n_embd: position embd
        # Add the position information to the input embeddings
        h = e.sum(dim=2)  # [batch_size x n_ctx x n_embd]
        for block in self.h:
            h = block(h)
        return h  # [batch_size x n_ctx x n_state] (n_state = n_embd)


class ClfHead(nn.Module):
    """ Classifier Head for the transformer"""

    def __init__(self, clf_token, cfg, n_class):
        super(ClfHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.clf_token = clf_token
        self.dropout = nn.Dropout(cfg.clf_pdrop)
        self.linear = nn.Linear(cfg.n_embd, n_class)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, h, x):
        # h: [batch_size x n_ctx x n_embd]
        clf_h = h.view(-1, self.n_embd)  # [(batch_size*n_ctx) x n_embd]
        # x: [batch_size, n_ctx, 2]
        flat = x[..., 0].contiguous().view(-1)  # [batch_size * n_ctx]
        clf_h = clf_h[flat == self.clf_token, :]  # [batch_size x n_embd]
        clf_h = self.dropout(clf_h)
        clf_logits = self.linear(clf_h)  # [batch_size x n_class]

        return clf_logits


class LMHead(nn.Module):
    def __init__(self, model, cfg, trunc_and_reshape=True):
        super().__init__()
        self.n_embd = cfg.n_embd
        embed_shape = model.embed.weight.shape  # [vocab x n_embd]
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model.embed.weight
        self.trunc_and_reshape = trunc_and_reshape

    def forward(self, h):  # h: [batch_size x n_ctx x n_embd]
        #  h: [batch_sizex n_ctx x n_embd] ==> [batch_size x (n_ctx-1) x n_embd]
        h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd) if self.trunc_and_reshape else h
        # h_trunc: [(batch_size * (n_ctx - 1)) x n_embd]  or not h_trunc: [batch_size x n_ctx x n_embd]
        lm_logits = self.decoder(h_trunc)  # h_trunc: [(batch_size * (n_ctx - 1)) x vocab]
        return lm_logits
    
    

class DoubleHeadModel(nn.Module):
    """Transformer with language model and Classification head ONLY"""

    def __init__(self, cfg, clf_token, n_class, vocab=40990, n_ctx=512):
        super(DoubleHeadModel, self).__init__()
        self.transformer = TransformerModel(cfg, vocab=vocab, n_ctx=n_ctx)
        self.lm_head = LMHead(self.transformer, cfg)  # truncate = True
        self.clf_head = ClfHead(clf_token, cfg, n_class)

    def forward(self, x):
        h = self.transformer(x)  # x: [batch_size x n_ctx x 2] ==> [batch_size x n_ctx x n_embd]
        lm_logits = self.lm_head(h)  # [(batch_size * (n_ctx - 1)) x vocab]
        clf_logits = self.clf_head(h, x)  # [batch_size x n_class]
        
        return lm_logits, clf_logits
    


def load_openai_pretrained_model(model, n_ctx=-1, n_special=-1, n_transfer=12, n_embd=768,
                                 path='./model/', path_name='./'):
    # Load weights from TF model
    print('loading weights...')
    names = json.load(open(path_name + 'parameters_names.json'))
    shapes = json.load(open(path + 'params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(path + 'params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    if n_ctx > 0:
        init_params[0] = init_params[0][:n_ctx]  # init_params[0]: n_ctx x n_embd
    if n_special > 0:
        init_params[0] = np.concatenate(
            [init_params[1],  # 40478 x 768
             (np.random.randn(n_special, n_embd) * 0.02).astype(np.float32),  # n_special x 768
             init_params[0]  # 512 x 768
             ], 0)  # [(n_vocab + n_special + n_ctx) x 768]
    else:
        init_params[0] = np.concatenate(
            [init_params[1], init_params[0]], 0
        )  # [(n_vocab + n_ctx) x 768]

    del init_params[1]
    if n_transfer == -1:
        n_transfer = 0
    else:
        n_transfer = 1 + n_transfer * 12
    init_params = [arr.squeeze() for arr in init_params]

    assert model.embed.weight.shape == init_params[0].shape

    model.embed.weight.data = torch.from_numpy(init_params[0])
    
    for name, ip in zip(names[1:n_transfer], init_params[1:n_transfer]):
        name = name[6:]  # skip "model/"
        assert name[-2:] == ":0"
        name = name[:-2]
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]
            pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == ip.shape
        except AssertionError as e:
            e.args += (pointer.shape, ip.shape)
            raise
        pointer.data = torch.from_numpy(ip)

        

        
        
        
        
        
        
        
        
        












