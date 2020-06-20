#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:23:03 2020

@author: zhuchen
"""


import os
from datasets import patents
from text_utils import TextEncoder
import numpy as np
import torch
import torch.nn as nn
from utils import encode_dataset, iter_data, ResultLogger, make_path
from model_pytorch import DoubleHeadModel, load_openai_pretrained_model
from loss import ClassificationLossCompute
from opt import OpenAIAdam
from argparse import Namespace
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

def transform_text(X):
    n_batch = len(X)
    xmb = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.int32)
    start = encoder['_start_']
    for i, x in enumerate(X):
        # X - trX: 23894 - list type
        x_trans = [start] + x[:max_len] + [clf_token]
        l = len(x_trans)
        xmb[i, :l, 0] = x_trans #sentence
        mmb[i, :l] = 1 #mask
    xmb[..., 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    return xmb, mmb
       

def iter_apply(Xs, Ms, Ys):
    logits = []
    cost = 0
    with torch.no_grad():
        dh_model.eval()
        for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False):
            n = len(xmb)
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            _, clf_logits = dh_model(XMB)
            clf_logits *= n  #clf_logits: [batch_size x n_class]
            clf_losses = compute_loss_fct(XMB, YMB, MMB, clf_logits, only_return_losses=True)
            clf_losses *= n
            logits.append(clf_logits.to('cpu').numpy())
            cost += clf_losses.sum().item()
        logits = np.concatenate(logits, 0)
        
    return logits, cost



def log(save_dir, desc):
    global best_score
    print('Logging')
    tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trYt[:n_valid])
    va_logits, va_cost = iter_apply(vaX, vaM, vaYt)  # va_logits: [batch_size x n_class]
    tr_cost = tr_cost / n_valid
    va_cost = va_cost / n_valid
    tr_acc = accuracy_score(trYt[:n_valid], np.argmax(tr_logits, 1)) * 100
    va_acc = accuracy_score(vaYt, np.argmax(va_logits, 1)) * 100
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print(f'{n_epochs} {n_updates} {tr_cost} {va_cost} {tr_acc} {va_acc}')
    if va_acc > best_score:
        best_score = va_acc
        path = os.path.join(save_dir, desc, 'best_params')
        torch.save(dh_model.state_dict(), make_path(path))
    
    
    
def predict(dataset, submission_dir):
    pass
    
def run_epoch():
    for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trYt, random_state=np.random),
                                   n_batch=n_batch_train, truncate=True):
        global n_updates
        dh_model.train()
        XMB = torch.tensor(xmb, dtype=torch.long).to(device)
        MMB = torch.tensor(mmb).to(device)
        YMB = torch.tensor(ymb, dtype=torch.long).to(device)
        lm_logits, clf_logits = dh_model(XMB)
        compute_loss_fct(XMB, YMB, MMB, clf_logits, lm_logits)
        n_updates += 1
        if n_updates in [1000, 2000, 4000, 8000, 16000, 32000] and n_epochs == 0:
            log(args.save_dir, desc)




args = Namespace(afn='gelu', analysis=False, attn_pdrop=0.1, b1=0.9, b2=0.999, 
          bpe_path='model/vocab_40000.bpe', clf_pdrop=0.1, data_dir='data/', 
          dataset=None, desc=None, e=1e-08, embd_pdrop=0.1, 
          encoder_path='model/encoder_bpe_40000.json', l2=0.01, lm_coef=0.9, 
          log_dir='log/', lr=6.25e-05, lr_schedule='warmup_linear', 
          lr_warmup=0.002, max_grad_norm=1, n_batch=8, n_ctx=512, n_embd=768, 
          n_head=12, n_iter=3, n_layer=12, n_transfer=12, n_valid=5974, opt='adam', 
          resid_pdrop=0.1, save_dir='save/', seed=42, submission_dir='submission/',
          submit=False, vector_l2=False)



# Constants
n_ctx = args.n_ctx
desc = 'model_log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()

logger = ResultLogger(path=os.path.join(args.log_dir, f'{desc}.jsonl'), **args.__dict__)
text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
encoder = text_encoder.encoder
n_vocab = len(text_encoder.encoder)

((trainX, trainY), (validX, validY)) = encode_dataset(*patents('./'), encoder=text_encoder)

encoder['_start_'] = len(encoder)
encoder['_classify_'] = len(encoder)
clf_token = encoder['_classify_']

n_special = 2
max_len = n_ctx // 2 - 2
n_ctx = min(max(
    [len(x[:max_len]) for x in trainX]
    + [len(x[:max_len]) for x in validX]
    ) + n_special, n_ctx) # 256

vocab = n_vocab + n_special + n_ctx

trX, trM = transform_text(trainX)
# trX: [n_data x n_ctx x 2]
# trM: [n_data x n_ctx]

vaX, vaM = transform_text(validX)

n_train = len(trainY)
n_valid = len(validY)
n_batch_train = args.n_batch * max(n_gpu, 1)
n_updates_total = (n_train // n_batch_train) * args.n_iter

dh_model = DoubleHeadModel(args, clf_token, 4, vocab, n_ctx)

criterion = nn.CrossEntropyLoss(reduction='none')

model_opt = OpenAIAdam(dh_model.parameters(),
                           lr=args.lr,
                           schedule=args.lr_schedule,
                           warmup=args.lr_warmup,
                           t_total=n_updates_total,
                           b1=args.b1,
                           b2=args.b2,
                           e=args.e,
                           l2=args.l2,
                           vector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)


compute_loss_fct = ClassificationLossCompute(criterion, criterion, args.lm_coef, model_opt)

#%%
load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special)

dh_model.to(device)


if args.dataset != 'stsb':
    trYt = trainY
    vaYt = validY

n_updates = 0
n_epochs = 0
best_score = 0
#%%
for num, p in enumerate(dh_model.parameters()):
    if num <= (147 - 2 - 12):
        p.requires_grad=False
#%%
a = list(dh_model.parameters())

#%%
for i in range(args.n_iter):
    print('running epoch', i)
    run_epoch()
    n_epochs += 1
    log(args.save_dir, desc) # if best score are generated, log it.


# prediction
#path = os.path.join(args.save_dir, desc, 'best_params')
#dh_model.load_state_dict(torch.load(path))
#predict(dataset)
    
    

















