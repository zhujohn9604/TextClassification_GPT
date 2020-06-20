#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:44:31 2020

@author: zhuchen
"""

import os
import sys
import json
import time
from tqdm import tqdm


def encode_dataset(*splits, encoder):
    encoded_splits = []
    # splits: (train_text, train_label), (test_text, test_label)
    for split in splits:
        fields = []
        for field in split:
            if isinstance(field[0], str):
                field = encoder.encode(field)
            fields.append(field)
        encoded_splits.append(fields)
    return encoded_splits



def iter_data(*datas, n_batch=128, truncate=False, max_batches=float('inf')):
    n = len(datas[0])
    if truncate:
        n = (n//n_batch) * n_batch
    n = min(n, max_batches * n_batch)
    n_batches = 0
    # generate
    for i in tqdm(range(0, n, n_batch), total=n//n_batch, ncols=80):
        if n_batches >= max_batches: raise StopIteration
        if len(datas) == 1:
            yield datas[0][i: i+n_batch]
        else:
            yield  (d[i: i+n_batch] for d in datas)
        n_batches += 1
        
  
def make_path(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return f

      
class ResultLogger(object):
    def __init__(self, path, *args, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log = open(make_path(path), 'w')
        self.f_log.write(json.dumps(kwargs) + '\n')
        
    def log(self, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log.write(json.dumps(kwargs) + '\n')
        self.f_log.flush()  # buffer it up
        
    def close(self):
        self.f_log.close()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        


