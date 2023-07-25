# -*- coding: utf-8 -*-
# @Author: foxwy
# @Function: Provide some loss functions
# @Paper: Unifying the factored and projected gradient descent for quantum state tomography

import torch


def S(rho):
    """von Neumann entropy"""
    S_rho_tmp = torch.matmul(rho, torch.log(rho))
    S_rho = -torch.trace(S_rho_tmp)
    return S_rho


def MLE_loss(out, target):
    """Negative log-likelihood function"""
    #loss = torch.sum((out - target)**2 * (torch.sqrt(target)))
    out_idx = out > 1e-12
    loss = -target[out_idx].dot(torch.log(out[out_idx]))
    return loss


def CF_loss(out, target):  # classical fidelity
    """As a loss function for testing, the combined function is used here"""
    # squared Hellinger distance
    #p = 0.5
    #loss = 1 - (target**p).dot(out**p)

    # Negative log-likelihood function
    loss1 = MLE_loss(out, target)

    #loss = torch.sum(torch.abs(target - out) / (torch.abs(target) + torch.abs(out)))
    #loss = 1 - target.dot(out) / (torch.norm(target) * torch.norm(out))
    #loss = torch.sum(torch.abs(out - target)**2)

    loss2 = 1 - (target - torch.mean(target)).dot(out - torch.mean(out)) / (torch.norm(target - torch.mean(target)) * torch.norm(out - torch.mean(out)))

    loss = loss1 * 0.1 + loss2
    return loss
