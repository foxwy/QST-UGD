# -*- coding: utf-8 -*-
# @Author: yong
# @Date:   2022-12-07 09:46:43
# @Last Modified by:   yong
# @Last Modified time: 2023-07-20 16:34:55
# @Author: foxwy
# @Method: iterative maximum likelihood estimation
# @Paper: Unified factored and projected gradient descent for quantum state tomography

import sys
import torch
from time import perf_counter
from tqdm import tqdm

#torch.set_default_dtype(torch.double)

sys.path.append('../..')
from Basis.Basic_Function import qmt_torch, qmt_matrix_torch, get_default_device
from evaluation.Fidelity import Fid
from Basis.Basis_State import Mea_basis, State


def iMLE(M, n_qubits, P_data, epochs, fid, result_save, device='cpu'):
    """
    Iterative maximum likelihood estimation, R\rhoR algorithm, see paper
    ``Iterative maximum-likelihood reconstruction in quantum homodyne tomography``.
    
    Args:
        M (tensor): The POVM, size (K, 2, 2).
        n_qubits (int): The number of qubits.
        P_data (tensor): The probability distribution obtained from the experimental measurements.
        epochs (int): Maximum number of iterations.
        fid (Fid): Class for calculating fidelity.
        result_save (set): A collection that holds process data.
        device (torch.device): GPU or CPU. 

    Stops:
        Reach the maximum number of iterations or quantum fidelity greater than or equal to 0.99.

    Examples::
        see ``FNN/FNN_learn`` or main.
    """
    d = 2**n_qubits

    # rho random init
    rho_t = torch.randn(d, d).to(torch.complex64).to(device)
    rho_t = torch.matmul(rho_t, rho_t.T.conj())
    rho =  rho_t / torch.trace(rho_t)

    # iterative
    pbar = tqdm(range(epochs))
    time_all = 0
    for i in pbar:
        time_b = perf_counter()

        P_out = qmt_torch(rho, [M] * n_qubits)
        adj = P_data / P_out
        adj[P_data == 0] = 0
        rmatrix = qmt_matrix_torch(adj.to(torch.complex64), [M] * n_qubits)  # R matrix

        rho = torch.matmul(rmatrix, torch.matmul(rho, rmatrix))  # R\rhoR
        rho = rho / torch.trace(rho)
        rho = 0.5 * (rho + rho.T.conj())

        time_e = perf_counter()
        time_all += time_e - time_b

        if (i + 1) % 20 == 0:
            Fc = fid.cFidelity_rho(rho)
            Fq = fid.Fidelity(rho)

            result_save['time'].append(time_all)
            result_save['epoch'].append(i + 1)
            result_save['Fc'].append(Fc)
            result_save['Fq'].append(Fq)
            pbar.set_description("iMLE --Fc {:.8f} | Fq {:.8f} | time {:.4f} | epochs {:d}".format(Fc, Fq, time_all, i + 1))

            if Fq >= 0.99:
                break

    pbar.close()


if __name__ == '__main__':
    n_qubits = 10
    POVM = 'Tetra4'
    ty_state = 'mixed'
    na_state = 'real_random'
    P_state = 0.1
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
    iMLE(M, n_qubits, P_data, 500, fid, result_save, device)
