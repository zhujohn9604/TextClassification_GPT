#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:43:20 2020

@author: zhuchen
"""

import math
import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_


def warmup_cosine(x, warmup=0.002):
    # possible reference: https://arxiv.org/pdf/1608.03983.pdf
    s = 1 if x <= warmup else 0
    return s * (x / warmup) + (1 - s) * (0.5 * (1 + torch.cos(math.pi * x)))


def warmup_constant(x, warmup=0.002):
    s = 1 if x <= warmup else 0
    return s * (x / warmup) + (1 - s) * 1


def warmup_linear(x, warmup=0.002):
    s = 1 if x <= warmup else 0
    return (s * (x / warmup) + (1 - s)) * (1 - x)


SCHEDULES = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear
}


class OpenAIAdam(Optimizer):
    """
    Source: https://huggingface.co/transformers/_modules/transformers/optimization.html#AdamW
    AdamW
    Also see deep learning book (GoodFellow) 8.5.4 p306
    """
    def __init__(self, params, lr, schedule, warmup, t_total,
                 b1=0.9, b2=0.999, e=1e-8, l2=0, vector_l2=False,
                 max_grad_norm=-1, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate : {}".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0 <= warmup:
            raise ValueError("Invalid warmup: {}".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {}".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {}".format(b2))
        if not 0.0 <= e:
            raise ValueError("Invalid epsilon value: {}".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, l2=l2, vector_l2=vector_l2, max_grad_norm=max_grad_norm)
        super(OpenAIAdam, self).__init__(params, defaults)
        
        
    def step(self, closure=None):
        loss = None
        
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
        
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['b1'], group['b2']
                
                state['step'] += 1
                
                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)  # v <- beta1 * v + (1 - beta1) * g
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)  # r <- beta2 * r + (1 - beta2 ) * (g . g)
                denom = exp_avg_sq.sqrt().add_(group['e'])  # sqrt(r) + e

                # fix the initial phase
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                schedule_fct = SCHEDULES[group['schedule']]
                lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
                
                # Add weight decay at the end (fixed version)
                if (len(p.size()) > 1 or group['vector_l2']) and group['l2'] > 0:
                    p.data.add_(-lr_scheduled * group['l2'], p.data)

        return loss
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

