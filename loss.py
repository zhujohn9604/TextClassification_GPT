#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:15:55 2020

@author: zhuchen
"""

import torch


class ClassificationLossCompute:
    def __init__(self, lm_criterion, clf_criterion, lm_coef, opt=None):
        self.lm_criterion = lm_criterion
        self.clf_criterion = clf_criterion
        self.lm_coef = lm_coef
        self.opt = opt

    def __call__(self, X, Y, M, clf_logits, lm_logits=None, only_return_losses=False):
        # Language model loss
        if lm_logits is not None:
            # lm_logits: [(batch_size * (n_ctx - 1)) x vocab]
            # X: [batch_size x n_ctx x 2] targets for lm
            # M: [batch_size x n_ctx]
            M = M.view(-1, M.size(-1))  # [bs x n_ctx]
            x_shifted = X[:, 1:, 0].contiguous().view(-1)  # [bs x (n_ctx-1)]
            lm_losses = self.lm_criterion(lm_logits, x_shifted)  # [bs * (n_ctx-1)]
            lm_losses = lm_losses.view(X.size(0), -1)  # [bs x (n_ctx-1)]
            lm_losses = lm_losses * M[:, 1:]
            lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)  # batch_size

        # Classification loss
        # clf_logits: [batch_size x n_class]
        # Y: batch_size
        clf_losses = self.clf_criterion(clf_logits, Y)
        if only_return_losses:
            return (clf_losses, lm_losses) if lm_logits is not None else clf_losses

        if self.lm_coef > 0 and lm_logits is not None:
            train_loss = clf_losses.sum() + self.lm_coef * lm_losses.sum()
        else:
            train_loss = clf_losses.sum()

        train_loss.backward()

        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()

        return train_loss.item()
