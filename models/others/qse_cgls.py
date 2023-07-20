# -*- coding: utf-8 -*-
# @Author: yong
# @Date:   2022-12-07 09:46:43
# @Last Modified by:   yong
# @Last Modified time: 2023-07-20 16:35:08
# @Author: foxwy
# @Method: conjugate-gradient algorithm
# @Paper: Unified factored and projected gradient descent for quantum state tomography

import sys
import numpy as np
import torch
from time import perf_counter
from tqdm import tqdm

sys.path.append('../..')
from Basis.Basic_Function import (qmt_torch, 
                                  qmt_torch_pure, 
                                  qmt_matrix, 
                                  qmt_matrix_torch, 
                                  get_default_device, 
                                  proj_spectrahedron_torch)
from evaluation.Fidelity import Fid
from Basis.Basis_State import Mea_basis, State


def qse_cgls(M, n_qubits, P_data, epochs, fid, map_method, P_proj, result_save, device='cpu'):
    """
    conjugate-gradient algorithm, see paper
    ``Superfast maximum-likelihood reconstruction for quantum tomography``.
    
    Args:
        M (tensor): The POVM, size (K, 2, 2).
        n_qubits (int): The number of qubits.
        P_data (tensor): The probability distribution obtained from the experimental measurements.
        epochs (int): Maximum number of iterations.
        fid (Fid): Class for calculating fidelity.
        map_method (str): State-mapping method, include ['chol', 'chol_h', 'proj_F', 'proj_S', 'proj_A'].
        P_proj (float): P order.
        result_save (set): A collection that holds process data.
        device (torch.device): GPU or CPU. 

    Stops:
        Reach the maximum number of iterations or quantum fidelity greater than or equal to 0.99 or other conditions.

    Examples::
        see ``FNN/FNN_learn`` or main.
    """
    # parameter
    opts = {'adjustment': 0.5, 'mincondchange': -torch.inf, 'step_adjust': 2, 'a2': 0.1, 'a3': 0.2}

    d = torch.tensor(2**n_qubits)
    M = M.to(torch.complex128)
    P_data = P_data.to(torch.double)
    eps = np.finfo(np.complex128).eps
    threshold_step = torch.sqrt(d) * d * eps
    threshold_fval = -P_data[P_data > 0].dot(torch.log(P_data[P_data > 0]))

    # rho init
    #A = (torch.eye(d) / torch.sqrt(d)).to(torch.complex128).to(device)
    #rho = (torch.eye(d) / d).to(torch.complex128).to(device)

    if map_method == 'chol':
        T = torch.randn(d, d).to(torch.complex128).to(device)
        T_triu = torch.triu(T, 1)
        A = torch.tril(T) + 1j * T_triu.T
        rho = torch.matmul(A.T.conj(), A)
        rho = rho / torch.trace(rho)
    elif map_method == 'chol_h':
        T = torch.randn(d, d).to(torch.complex128).to(device)
        T_triu = torch.triu(T, 1)
        A = torch.tril(T) + 1j * T_triu.T
        A += torch.tril(A, -1).T.conj()
        rho = torch.matmul(A.T.conj(), A)
        rho = rho / torch.trace(rho)

    # -----line search stuff-----
    a2 = opts['a2']
    a3 = opts['a3']

    # discard zero-vauled frequencies
    probs = qmt_torch(rho, [M] * n_qubits)
    adj = P_data / probs
    adj[P_data == 0] = 0
    rmatrix = qmt_matrix_torch(adj.to(torch.complex128), [M] * n_qubits)

    condchange = torch.inf

    if opts['mincondchange'] > 0:
        hessian_proxy = P_data / probs**2
        hessian_proxy[P_data == 0] = 0
        old_hessian_proxy = hessian_proxy

    fval = -P_data[P_data > 0].dot(torch.log(probs[P_data > 0]))

    # iterative
    pbar = tqdm(range(epochs))
    time_all = 0
    stop_i = -1
    for i in pbar:
        time_b = perf_counter()

        curvature_too_large = False

        if opts['mincondchange'] > 0:
            if i > 0:
                condchange = torch.real(torch.acos(torch.real(old_hessian_proxy.dot(hessian_proxy)) / torch.norm(old_hessian_proxy) / torch.norm(hessian_proxy)))

        # the gradient
        if i == 0:
            # gradient
            G = torch.matmul(A, rmatrix - torch.eye(d).to(device))
            # conjugate-gradient
            H = G
        else:
            G_next = torch.matmul(A, rmatrix - torch.eye(d).to(device))
            polakribiere = torch.real(torch.matmul(G_next.reshape(-1, 1).T.conj(), (G_next.reshape(-1, 1) - opts['adjustment'] * G.reshape(-1, 1)))) / torch.norm(G)**2
            gamma = torch.maximum(polakribiere, torch.tensor(0))
            # conjugate-gradient
            H = G_next + gamma * H
            # gradient
            G = G_next

        # line search
        A2 = A + a2 * H
        A3 = A + a3 * H
        rho2 = torch.matmul(A2.T.conj(), A2)
        rho2 = rho2 / torch.trace(rho2)
        rho3 = torch.matmul(A3.T.conj(), A3)
        rho3 = rho3 / torch.trace(rho3)
        probs2 = qmt_torch(rho2, [M] * n_qubits)
        probs3 = qmt_torch(rho3, [M] * n_qubits)

        l1 = fval
        l2 = -P_data[P_data > 0].dot(torch.log(probs2[P_data > 0]))
        l3 = -P_data[P_data > 0].dot(torch.log(probs3[P_data > 0]))
        alphaprod = 0.5 * ((l3 - l1) * a2**2 - (l2 - l1) * a3**2) / ((l3 - l1) * a2 - (l2 - l1) * a3)
        
        if torch.isnan(alphaprod) or alphaprod > 1 / eps or alphaprod < 0:
            candidates = [0, a2, a3]
            l_list = [l1, l2, l3]
            index = l_list.index(min(l_list))
            if opts['step_adjust'] > 1:
                if torch.isnan(alphaprod) or alphaprod > 1 / eps:
                    # curvature too small to estimate properly
                    a2 = opts['step_adjust'] * a2
                    a3 = opts['step_adjust'] * a3
                elif alphaprod < 0:
                    # curvature too large, so steps overshoot parabola
                    a2 = a2 / opts['step_adjust']
                    a3 = a3 / opts['step_adjust']
                    curvature_too_large = True

            alphaprod = candidates[index]

        # update
        A = A + alphaprod * H
        A = A / torch.norm(A)
        old_rho = rho

        # map
        if map_method == 'chol':
            T = torch.tril(A)
            T_temp = torch.matmul(T.T.conj(), T)
            rho = T_temp / torch.trace(T_temp)
        elif map_method == 'chol_h':
            T_temp = torch.matmul(A.T.conj(), A)
            rho = T_temp / torch.trace(T_temp)
        else:
            rho = proj_spectrahedron_torch(A, device, map_method, P_proj)

        probs = qmt_torch(rho, [M] * n_qubits)
        fval = -P_data[P_data > 0].dot(torch.log(probs[P_data > 0]))

        # check threshold
        steps_i = 0.5 * torch.sqrt(d) * torch.norm(rho - old_rho)
        satisfied_step = steps_i <= threshold_step
        satisfied_fval = fval <= threshold_fval

        if i < epochs - 1:
            adj = P_data / probs
            adj[P_data == 0] = 0
            rmatrix = qmt_matrix_torch(adj.to(torch.complex128), [M] * n_qubits)

            if opts['mincondchange'] > 0:
                old_hessian_proxy = hessian_proxy
                hessian_proxy = P_data / probs**2
                hessian_proxy[P_data == 0] = 0

        time_e = perf_counter()
        time_all += time_e - time_b

        # evalution
        Fq = 0
        if (i + 1) % 20 == 0:
            Fc = fid.cFidelity_rho(rho.to(torch.complex64))
            Fq = fid.Fidelity(rho.to(torch.complex64))

            result_save['time'].append(time_all)
            result_save['epoch'].append(i + 1)
            result_save['Fc'].append(Fc)
            result_save['Fq'].append(Fq)
            pbar.set_description("CGL --Fc {:.8f} | Fq {:.8f} | time {:.4f} | epochs {:d}".format(Fc, Fq, time_all, i + 1))

        if (not curvature_too_large and satisfied_step) or satisfied_fval or condchange < opts['mincondchange']:
            stop_i = i
            break

        if Fq >= 0.99:
            break

    pbar.close()

    if stop_i == -1:
        stop_i = epochs - 1
        
    return rho, stop_i, time_all


if __name__ == '__main__':
    n_qubits = 2
    POVM = 'Tetra4'
    ty_state = 'mixed'
    na_state = 'GHZi_P'
    P_state = 0.5
    device = get_default_device()

    M = Mea_basis(POVM).M
    M = torch.from_numpy(M).to(device)

    state_star, rho_star = State().Get_state_rho(na_state, n_qubits, P_state)
    rho_star = torch.from_numpy(rho_star).to(torch.complex64).to(device)

    fid = Fid(basis=POVM, n_qubits=n_qubits, ty_state=ty_state, rho_star=rho_star, M=M, device=device)
    
    P_data = qmt_torch(rho_star, [M] * n_qubits)

    result_save = {'time': [],
                   'epoch': [],
                   'Fc': [],
                   'Fq': []}
    qse_cgls(M, n_qubits, P_data, 200, fid, 'chol_h', 2, result_save, device)
