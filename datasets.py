#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:29:23 2020

@author: zhuchen
"""

import os
import csv
from sklearn.model_selection import train_test_split
import numpy as np

seed = 1218

def _patents(path):
    ipc_dict = {'3':0, '5':1, '7':2, '99':3}
    with open(path, encoding='utf-8') as f:
        f = csv.reader(f)
        Inputabstract = []
        IpcCode = []
        for i, line in enumerate(list(f)):
            if i > 0:  #header ['abstract', 'title', 'IPC']
                Inputabstract.append(line[0])
                IpcCode.append(line[2])
        IpcCodeLabel = [ipc_dict[i] for i in IpcCode]
        return Inputabstract, IpcCodeLabel


def patents(data_dir, test_percentage=0.2):
    Inputabstract, IpcCodeLabel = _patents(os.path.join(data_dir, 'Data.csv'))
    n_test = round(len(Inputabstract) * test_percentage)
    train_text, test_text, train_label, test_label = train_test_split(Inputabstract, IpcCodeLabel, test_size=n_test,
                                                                      random_state=seed)
    train_label = np.array(train_label, dtype=np.int32)
    test_label = np.array(test_label, dtype=np.int32)
    return (train_text, train_label), (test_text, test_label)


if __name__ == '__main__':
    (train_text, train_label), (test_text, test_label) = patents('./')


    
