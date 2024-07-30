# -*- coding: utf-8 -*-
# @Author: yong
# @Date:   2022-12-07 09:46:43
# @Last Modified by:   yong
# @Last Modified time: 2024-07-30 15:14:20
# @Author: foxwy
# @Method: "CG-APG" algorithm that combines an accelerated projected-gradient (APG) approach with the existing conjugate-gradient (CG) algorithm
# @Paper: Efficient factored gradient descent algorithm for quantum state tomography

import sys
import numpy as np
import torch
from time import perf_counter
from tqdm import tqdm

torch.set_printoptions(precision=16)
torch.set_default_dtype(torch.double)

sys.path.append('../..')

from Basis.Basic_Function import (qmt_torch,
                                  qmt_matrix_torch,
                                  get_default_device,
                                  proj_spectrahedron_torch)
from models.others.qse_cgls import qse_cgls
from Basis.Basis_State import Mea_basis, State
from evaluation.Fidelity import Fid


def qse_apg(M, n_qubits, P_data, epochs, fid, map_method, P_proj, result_save, device='cpu'):
    """
    "CG-APG" algorithm that combines an accelerated projected-gradient (APG) approach with 
    the existing conjugate-gradient (CG) algorithm, see paper
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
        Reach the maximum number of iterations or quantum fidelity greater than or equal to 0.99.

    Examples::
        see ``FNN/FNN_learn`` or main.
    """
    # parameter
    stop_i = -1
    eps = np.finfo(float).eps
    opts = {'rho0': 'bootstrap', 'guard': True, 'bb': True, 'afactor': 1.1, 'bfactor': 0.5,
            'minimum_t': eps, 'restart_grad_param': 0.01, 'bootstrap_threshold': 0.01, 'accel': 'fista_tfocs'}

    d = torch.tensor(2**n_qubits)
    K = 4**n_qubits
    M = M.to(torch.complex128)

    # discard zero-vauled frequencies
    P_data = P_data.to(torch.double)
    fmap = P_data > 0
    P_data = P_data[fmap]

    threshold_step = torch.sqrt(d) * d * eps
    threshold_fval = -P_data.dot(torch.log(P_data))
    coeff_temp = torch.zeros(K, dtype=torch.double).to(device)

    # rho init
    if isinstance(opts['rho0'], str):
        if opts['rho0'] == 'white':
            rho = (torch.eye(d) / d).to(torch.complex128).to(device)
        elif opts['rho0'] == 'bootstrap':
            # run CG with line search until "condition number" of
            # adjustment vector stops changing by much
            coeff_temp[fmap] = P_data
            rho, stop_i, time_all = qse_cgls(
                M, n_qubits, coeff_temp, epochs, fid, result_save, device)
        else:
            print('unknown initializer specification: ', opts['rho0'])
    else:
        rho = opts['rho0']

    varrho = rho
    old_varrho = varrho
    varrho_changed = True
    gradient = None
    theta = 1
    t = None
    bb_okey = False  # don't do two-step Barzilai-Borwein on first step

    # compute initial probabilities
    probs_rho = qmt_torch(rho, [M] * n_qubits)
    probs_rho = probs_rho[fmap]
    probs_varrho = probs_rho

    # -----main loop-----
    epochs = epochs - stop_i - 1
    pbar = tqdm(range(epochs))
    time_all = 0
    for i in pbar:
        time_b = perf_counter()

        # compute new gradient if varrho has changed
        t_last = t
        if varrho_changed:
            if opts['bb'] and i > 0:
                old_gradient = gradient
            coeff_temp[fmap] = P_data / probs_varrho

            gradient = - \
                qmt_matrix_torch(coeff_temp.to(
                    torch.complex128), [M] * n_qubits)

            fval_varrho = -P_data.dot(torch.log(probs_varrho))

            if opts['bb'] and bb_okey:
                varrho_diff = varrho.reshape(-1, 1) - old_varrho.reshape(-1, 1)
                gradient_diff = gradient.reshape(-1, 1) - \
                    old_gradient.reshape(-1, 1)
                denominator = torch.norm(gradient_diff)**2
                if denominator > 0:
                    t = torch.abs(torch.real(torch.matmul(
                        varrho_diff.T.conj(), gradient_diff))) / denominator

        if t is None:
            # compute local Lipschitz constant from derivatives
            probs_gradient = qmt_torch(-gradient, [M] * n_qubits, True)
            probs_gradient = probs_gradient[fmap]
            first_deriv = -P_data.dot(probs_gradient / probs_varrho)
            second_deriv = P_data.dot(probs_gradient**2 / probs_varrho**2)
            t = -first_deriv / second_deriv
        else:
            if not (opts['bb'] and bb_okey):
                t = t * opts['afactor']

        # backtrack for finding a good value of t
        t_good = False
        fval_new = None
        first_order = None
        second_order = None
        t_threshold = opts['minimum_t'] / torch.norm(gradient)
        while not t_good:
            if fval_new is not None:
                new_t_estimate = second_order / \
                    torch.maximum(torch.tensor(0), fval_new -
                                  fval_varrho - first_order)
                # this comparison is false if any term is NaN (namely
                # new_t_estimate)
                if new_t_estimate > t_threshold:
                    t = torch.minimum(t * opts['bfactor'], new_t_estimate)
                else:
                    # if new_t_estimate is NaN or less than or equal to t_threshold
                    t = t * opts['bfactor']

            # map
            #map_method = 'proj_S'
            rho_new = proj_spectrahedron_torch(
                    varrho - t * gradient, device, map_method, P_proj, 0)

            # rho_new = varrho - t * gradient
            probs_rho_new = qmt_torch(rho_new, [M] * n_qubits, True)
            probs_rho_new = probs_rho_new[fmap]
            fval_new = -P_data.dot(torch.log(probs_rho_new))

            delta = rho_new.reshape(-1, 1) - varrho.reshape(-1, 1)
            first_order = torch.real(torch.matmul(
                gradient.reshape(-1, 1).T.conj(), delta))
            second_order = 0.5 * torch.norm(delta)**2

            # multiplied by t so that we don't get NaN or Inf if t is too small
            # not greater than catches NaNs
            t_good = not (t * fval_new > (t * fval_varrho +
                          t * first_order + 0.9 * second_order))

        # check threshold
        steps_i = 0.5 * torch.sqrt(d) * torch.norm(rho_new - rho)

        satisfied_step = steps_i <= threshold_step
        satisfied_fval = fval_new <= threshold_fval

        if t < t_threshold or satisfied_step or satisfied_fval:
            rho = rho_new
            # break

        # record previous value of varrho for Barzilai-Borwein
        if opts['bb']:
            old_varrho = varrho

        # check whether to do adaptive restart
        vec1 = varrho.reshape(-1, 1) - rho_new.reshape(-1, 1)
        vec2 = rho_new.reshape(-1, 1) - rho.reshape(-1, 1)
        do_restart = torch.real(torch.matmul(vec1.T.conj(
        ), vec2)) > -opts['restart_grad_param'] * torch.norm(vec1) * torch.norm(vec2)

        # enable Barzilai-Borwein for next iteration if no restart
        bb_okey = not do_restart

        # perform the restart if needed
        if do_restart:
            varrho = rho
            probs_varrho = probs_rho
            varrho_changed = theta > 1
            theta = 1
            continue

        # acceleration
        if i > 0 and t > eps:
            Lfactor = (t_last / t).item()
        else:
            Lfactor = 1

        if opts['accel'] == 'fista':
            theta_new = (1 + np.sqrt(1 + 4 * theta**2)) / 2
            beta = (theta - 1) / theta_new
            theta = theta_new
        elif opts['accel'] == 'fista_tfocs':
            theta_hat = np.sqrt(Lfactor) * theta
            theta_new = (1 + np.sqrt(1 + 4 * theta_hat**2)) / 2
            beta = (theta_hat - 1) / theta_new
            theta = theta_new
        elif opts['accel'] is None:
            beta = 0
        else:
            print('unknown acceleration scheme')

        # update
        varrho = rho_new + beta * (rho_new - rho)
        probs_varrho = probs_rho_new + beta * (probs_rho_new - probs_rho)
        if opts['guard'] and (torch.min(probs_varrho) <= 0):
            # discard momentum if momentum causes varrho to become infeasible
            # retain theta to keep estimate of current condition number
            varrho = rho_new
            probs_varrho = probs_rho_new

        varrho_changed = True
        rho = rho_new
        probs_rho = probs_rho_new

        time_e = perf_counter()
        time_all += time_e - time_b

        if (i + stop_i) % 20 == 0:
            Fq = fid.Fidelity(rho.to(torch.complex64))

            result_save['time'].append(time_all)
            result_save['epoch'].append(i + stop_i)
            result_save['Fq'].append(Fq)
            pbar.set_description(
                "APG Fq {:.8f} | time {:.4f} | epochs {:d}".format(Fq, time_all, i + stop_i))

            if Fq >= 0.99999:
                break

    pbar.close()


if __name__ == '__main__':
    n_qubits = 8
    POVM = 'Tetra4'
    ty_state = 'mixed'
    na_state = 'GHZi_P'
    P_state = 0.5
    device = get_default_device()

    M = Mea_basis(POVM).M
    M = torch.from_numpy(M).to(device)

    state_star, rho_star = State().Get_state_rho(na_state, n_qubits, P_state)
    rho_star = torch.from_numpy(rho_star).to(torch.complex64).to(device)

    fid = Fid(basis=POVM, n_qubits=n_qubits, ty_state=ty_state,
              rho_star=rho_star, M=M, device=device)

    P_data = qmt_torch(rho_star, [M] * n_qubits)

    result_save = {'time': [],
                   'epoch': [],
                   'Fc': [],
                   'Fq': []}
    qse_apg(M, n_qubits, P_data, 1000, fid, 'proj_S', 1, result_save, device)
