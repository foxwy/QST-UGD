# -*- coding: utf-8 -*-
# @Author: yong
# @Date:   2022-12-07 09:46:43
# @Last Modified by:   yong
# @Last Modified time: 2024-07-30 15:42:29
# @Author: foxwy
# @Method: linear regression estimation, recursive LRE algorithm
# @Paper: Efficient factored gradient descent algorithm for quantum state tomography

import sys
import numpy as np
import torch
from time import perf_counter
from tqdm import tqdm

#torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=8)

sys.path.append('../..')
from Basis.Basic_Function import (qmt_torch, 
                                  qmt_matrix_torch, 
                                  get_default_device, 
                                  proj_spectrahedron_torch, 
                                  qmt_product_torch, 
                                  ten_to_k)
from evaluation.Fidelity import Fid
from Basis.Basis_State import Mea_basis, State


def cal_para(X, Y, n_qubits):
    """Using the product structure of POVM to speed up"""
    N = 7
    if n_qubits <= N:  # faster, further expansion will cause the memory to explode
        X_t = X
        for i in range(n_qubits - 1):
            X_t = torch.kron(X_t, X)
        return X_t @ Y
    else:
        Y = Y.reshape(-1, 4**N)
        n_qubits_t = n_qubits - N
        N_choice = 4**(n_qubits_t)
        num_choice = np.arange(N_choice)
        theta = 0

        X_t = X
        for i in range(N - 1):
            X_t = torch.kron(X_t, X)

        for num in num_choice:
            samples = ten_to_k(num, 4, n_qubits_t)
            theta_n = X[:, samples[0]]
            for sample in samples[1:]:
                theta_n = torch.kron(theta_n, X[:, sample])
            theta_n = torch.kron(theta_n, X_t @ Y[num, :])
            theta += theta_n

        return theta


def LRE(M, n_qubits, P_data, fid, map_method, P_proj, result_save, device='cpu'):
    """
    linear regression estimation, see paper
    ``Full reconstruction of a 14-qubit state within four hours``.
    
    Args:
        M (tensor): The POVM, size (K, 2, 2).
        n_qubits (int): The number of qubits.
        P_data (tensor): The probability distribution obtained from the experimental measurements.
        fid (Fid): Class for calculating fidelity.
        map_method (str): State-mapping method, include ['chol', 'chol_h', 'proj_F', 'proj_S', 'proj_A'].
        P_proj (float): P order.
        result_save (set): A collection that holds process data.
        device (torch.device): GPU or CPU. 

    Examples::
        see ``FNN/FNN_learn`` or main.
    """
    time_b = perf_counter()

    d = 2**n_qubits

    M_basis = Mea_basis('Pauli_normal').M
    M_basis = torch.from_numpy(M_basis).to(device)

    X = qmt_product_torch([M], [M_basis])
    X_t = torch.linalg.pinv(X.T @ X, hermitian=True)
    X_t = X_t @ X.T

    '''
    X_all = X_t
    for i in range(n_qubits - 1):
        X_all = torch.kron(X_all, X_t)
    theta = X_all @ P_data'''

    # method2
    theta = cal_para(X_t, P_data, n_qubits)
    rho = qmt_matrix_torch(theta.to(torch.complex64), [M_basis] * n_qubits)

    #eigenvalues, _ = torch.linalg.eigh(rho)
    #print('-'*20, eigenvalues)

    # state-mapping
    if map_method == 'fac_t':
        T = torch.tril(rho)
        rho = torch.matmul(T.T.conj(), T)
    elif map_method == 'fac_h':
        rho = torch.matmul(rho.T.conj(), rho)
    else:
        rho = proj_spectrahedron_torch(rho, device, map_method, P_proj)

    rho = rho / torch.trace(rho)

    #eigenvalues, _ = torch.linalg.eigh(rho)
    #print('-'*20, eigenvalues)

    time_e = perf_counter()
    time_all = time_e - time_b

    # show and save
    Fq = fid.Fidelity(rho)

    result_save['time'].append(time_all)
    result_save['Fq'].append(Fq)
    print("Fq {:.16f} | time {:.4f}".format(Fq, time_all))


def RLRE(M, n_qubits, P_data, fid, device='cpu'):
    """
    recursive LRE algorithm, see paper
    ``Adaptive quantum state tomography via linear regression estimation: 
    Theory and two-qubit experiment``.
    
    Args:
        M (tensor): The POVM, size (K, 2, 2).
        n_qubits (int): The number of qubits.
        P_data (tensor): The probability distribution obtained from the experimental measurements.
        fid (Fid): Class for calculating fidelity.
        device (torch.device): GPU or CPU. 
    """
    time_b = perf_counter()

    d = 2**n_qubits

    M_basis = Mea_basis('Pauli_normal').M
    M_basis = torch.from_numpy(M_basis).to(device)

    X = qmt_product_torch([M] * n_qubits, [M_basis] * n_qubits)

    psi_0 = X[0, :].reshape(-1, 1)
    Q_i_1 = torch.linalg.pinv(psi_0 @ psi_0.T + 1e-7 * torch.eye(len(psi_0)).to(device), hermitian=True)  # 矩阵奇异，无法直接求逆，需要加上一个小的正定矩阵
    theta_i_1 = Q_i_1 @ psi_0 * P_data[0]

    for i in range(1, d**2):
        psi_i = X[i, :].reshape(-1, 1)
        a_i = 1 / (1 + psi_i.T @ Q_i_1 @ psi_i)
        theta_i_1 += a_i * Q_i_1 @ psi_i * (P_data[i] - psi_i.T @ theta_i_1)

        Q_i_1 += -a_i * Q_i_1 @ psi_i @ psi_i.T @ Q_i_1

        if i % 16 == 0:
            print('iter:', i) 

    rho = qmt_matrix_torch(theta_i_1, [M_basis] * n_qubits)
    rho = proj_spectrahedron_torch(rho, device, 'proj_S', 2, 1)

    time_e = perf_counter()
    time_all = time_e - time_b

    Fc = fid.cFidelity_rho(rho)
    Fq = fid.Fidelity(rho)
    print("Fc {:.12f} | Fq {:.16f} | time {:.4f}".format(Fc, Fq, time_all))


if __name__ == '__main__':
    n_qubits = 8
    POVM = 'Tetra4'
    ty_state = 'mixed'
    na_state = 'random_P'
    P_state = 0.4
    device = get_default_device()

    M = Mea_basis(POVM).M
    M = torch.from_numpy(M).to(device)

    state_star, rho_star = State().Get_state_rho(na_state, n_qubits, P_state)
    rho_star = torch.from_numpy(rho_star).to(torch.complex64).to(device)

    fid = Fid(basis=POVM, n_qubits=n_qubits, ty_state=ty_state, rho_star=rho_star, M=M, device=device)
    
    P_data = qmt_torch(rho_star, [M] * n_qubits)
    
    result_save = {'time': [],
                   'Fc': [],
                   'Fq': []}
    LRE(M, n_qubits, P_data, fid, 'proj_A', 1, result_save, device)
