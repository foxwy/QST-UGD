# -*- coding: utf-8 -*-
# @Author: yong
# @Date:   2023-07-10 14:01:40
# @Last Modified by:   yong
# @Last Modified time: 2024-07-30 15:14:52
# @Paper: Efficient factored gradient descent algorithm for quantum state tomography

import torch
from torch.optim.optimizer import Optimizer
import copy

__all__ = ["cg"]

def Armijo(func, 
           x, 
           g, 
           d, 
           lr, 
           rho, 
           c1, 
           iter):
    """
    func: (closure i.e loss) from conjugate gradient method
    x: parameter of loss
    g: grad.data of x
    d: data of direction vector
    lr: initialized stepsize
    rho: contraction factor
    c1: sufficient decrease constant
    iter: maximum step permitted
    """
    for _ in range(iter):
        F_o = float(func())

        x.data = x.data + lr * d

        if not float(func()) <= float(F_o + c1 * lr * torch.dot(g.reshape(-1), d.reshape(-1))):
            alpha = lr
            lr = lr * rho
        else:
            alpha = lr
            x.data = x.data - alpha * d
            break    
        
        x.data = x.data - lr * d     

    return alpha


def BB(func, 
       x, 
       g, 
       d, 
       ls, 
       a1,
       a2,
       eps):

    opts = {'step_adjust': 2}

    x.data = x.data + a1 * d
    ls1 = float(func())

    x.data = x.data + (a2 - a1) * d
    ls2 = float(func())

    x.data = x.data - a2 * d

    alphaprod = 0.5 * ((ls2 - ls) * a1**2 - (ls1 - ls) * a2**2) / ((ls2 - ls) * a1 - (ls1 - ls) * a2)

    if torch.isnan(alphaprod) or alphaprod > 1 / eps or alphaprod < 0:
        candidates = [0, a1, a2]
        l_list = [ls, ls1, ls2]
        index = l_list.index(min(l_list))
        if opts['step_adjust'] > 1:
            if torch.isnan(alphaprod) or alphaprod > 1 / eps:
                # curvature too small to estimate properly
                a1 = opts['step_adjust'] * a1
                a2 = opts['step_adjust'] * a2
            elif alphaprod < 0:
                # curvature too large, so steps overshoot parabola
                a1 = a1 / opts['step_adjust']
                a2 = a2 / opts['step_adjust']

        alphaprod = candidates[index]

    return alphaprod, a1, a2


class cg(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        ls=1e-8,
        a1=0.1,
        a2=0.2,
        eps=1e-16,
        method='PRP',
        line_search='Armijo'
        ):

        defaults = dict(
            lr=lr,
            ls=ls,
            a1=a1,
            a2=a2,
            eps=eps,
            method=method,
            line_search=line_search,
            )
        super().__init__(params, defaults)

    def step(self, closure):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad

                state = self.state[p]

                lr = group['lr']
                ls = group['ls']
                a1 = group['a1']
                a2 = group['a2']
                eps = group['eps']
                method = group['method']
                line_search = group['line_search']

                if len(state) == 0:
                    state['g'] = copy.deepcopy(d_p.data)

                    if torch.norm(state['g']) < group['eps']:
                        # Stop condition
                        return loss

                    # Direction vector
                    state['d'] = copy.deepcopy(-d_p.data)
                    # Determine whether to calculate A
                    state['index'] = True
                    # Step of Conjugate Gradient
                    state['step'] = 0
                    # initialized step length
                    state['alpha'] = lr
                else:
                    # Parameters that make gradient steps
                    if method == 'FR':
                        state['beta'] = torch.norm(d_p.data) / torch.norm(state['g'])
                    
                    elif method == 'PRP':
                        state['beta'] = torch.maximum(torch.dot(
                            d_p.data.reshape(-1), 
                            (d_p.data.reshape(-1) - 0.5 * state['g'].reshape(-1))) / torch.norm(state['g'])**2, torch.tensor(0))
                    
                    elif method == 'HS':
                        state['beta'] = torch.dot(
                            d_p.data.reshape(-1), 
                            (d_p.data.reshape(-1) - state['g'].reshape(-1))) \
                            / torch.dot(state['d'].data.reshape(-1), 
                                (d_p.data.reshape(-1) - state['g'].reshape(-1)))
                    
                    elif method == 'CD':
                        state['beta'] = -torch.norm(d_p.data) \
                            / torch.dot(state['d'].data.reshape(-1), state['g'].reshape(-1))
                    
                    elif method == 'DY':
                        state['beta'] = torch.norm(d_p.data) \
                            / torch.dot(state['d'].data.reshape(-1), 
                                (d_p.data.reshape(-1) - state['g'].reshape(-1)))
                    
                    elif method =='LS':
                        state['beta'] = -torch.dot(
                            d_p.data.reshape(-1), 
                            (d_p.data.reshape(-1) - state['g'].reshape(-1))) \
                            / torch.dot(state['d'].data.reshape(-1), state['g'].reshape(-1))
                    
                    elif method =='HZ':
                        Q = d_p.data - state['g']
                        M = Q - 2 *  torch.norm(Q) \
                            / torch.dot(state['d'].reshape(-1), Q.reshape(-1)) * state['d']
                        N = d_p.data / torch.dot(state['d'].reshape(-1), Q.reshape(-1))
                        state['beta'] = torch.dot(M.reshape(-1), N.reshape(-1))
                    
                    elif method =='HS-DY':
                        state['beta'] = max(0, 
                            min(torch.dot(d_p.data.reshape(-1), 
                                    (d_p.data.reshape(-1) - state['g'].reshape(-1))) \
                            / torch.dot(state['d'].data.reshape(-1), 
                                    (d_p.data.reshape(-1) - state['g'].reshape(-1))),
                            torch.norm(d_p.data) \
                            / torch.dot(state['d'].data.reshape(-1), 
                                    (d_p.data.reshape(-1) - state['g'].reshape(-1)))))

                    state['g'] = copy.deepcopy(d_p.data)
                    if torch.norm(state['g']) < group['eps']:
                        return loss

                    state['d'] = -state['g'] + state['beta'] * state['d']
                    state['index'] = False

                if line_search == 'Armijo':
                    state['alpha'] = Armijo(closure, p, state['g'], state['d'], state['alpha'], 0.5, 1e-4, 2000)
                else:
                    state['alpha'], group['a1'], group['a2'] = BB(closure, p, state['g'], state['d'], ls, a1, a2, eps)


                p.data.add_(state['d'], alpha=state['alpha'])

        return loss



