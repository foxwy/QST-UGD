# -*- coding: utf-8 -*-
# @Author: foxwy
# @Function: Provide some loss functions
# @Paper: Efficient factored gradient descent algorithm for quantum state tomography

import torch


def MLE_loss(out, target):
    """Negative log-likelihood function"""
    #loss = torch.sum((out - target)**2 * (torch.sqrt(target)))
    out_idx = out > 1e-16
    loss = -target[out_idx].dot(torch.log(out[out_idx]))
    return loss


def LS_loss(out, target):
    loss = torch.sum((out - target)**2)
    return loss


def CF_loss(out, target):  # classical fidelity
    """As a loss function for testing, the combined function is used here"""
    # squared Hellinger distance
    p = 0.5
    loss = 1 - (target**p).dot(out**p)
    return loss
