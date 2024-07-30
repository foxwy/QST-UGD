# -*- coding: utf-8 -*-
# @Author: yong
# @Date:   2022-12-07 09:46:43
# @Last Modified by:   yong
# @Last Modified time: 2024-07-30 15:39:02
# @Function: single-layer network, use same optimizer and others setups as neural network
# @Paper: Efficient factored gradient descent algorithm for quantum state tomography

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from time import perf_counter
from tqdm import tqdm

sys.path.append('../..')

from models.UGD.rprop import Rprop
from models.UGD.cg_optim import cg
from Basis.Basis_State import Mea_basis, State
from evaluation.Fidelity import Fid
from Basis.Loss_Function import MLE_loss, LS_loss, CF_loss
from Basis.Basic_Function import qmt_torch, get_default_device, proj_spectrahedron_torch, qmt_matrix_torch

# torch.set_default_dtype(torch.double)

class UGD_nn(nn.Module):
    """
    The single-layer network is used to perform the quantum state tomography task by directly 
    optimizing the layer parameters and mapping them to the density matrix and measuring the 
    distance from the probability distribution to optimize the network parameters, 
    see paper ``Efficient factored gradient descent algorithm for quantum state tomography``.
    """

    def __init__(self, na_state, 
                 n_qubits,
                 P_idxs,
                 M,
                 map_method='fac_h',
                 P_proj=2):
        """
        Args:
            n_qubits (int): The number of qubits.
            P_idxs (tensor): Index of the POVM used for measurement, Not all measurements 
                are necessarily used.
            M (tensor): The POVM, size (K, 2, 2).
            rho_init (tensor): If None, initialize the parameters randomly, and vice versa with rho.
            map_method (str): State-mapping method, include ['fac_t', 'fac_h', 'fac_a', 'proj_M', 'proj_S', 'proj_A'].
            P_proj (float): P order.
        """
        super().__init__()

        self.N = n_qubits
        self.P_idxs = P_idxs
        self.M = M
        self.device = M.device
        self.map_method = map_method
        self.P_proj = P_proj

        d = 2**n_qubits
        params = torch.randn(d, d, requires_grad=True).to(torch.float32)
        self.params = nn.Parameter(params)

        if self.map_method == "fac_a":
            params = torch.randn(d, 2*d, requires_grad=True).to(torch.float32)
            self.params = nn.Parameter(params)
        else:
            params = torch.randn(d, d, requires_grad=True).to(torch.float32)
            self.params = nn.Parameter(params)

    def forward(self):
        if 'fac' in self.map_method:
            self.rho = self.Rho_T()  # factorization

        elif 'proj_A' in self.map_method or (self.map_method in ("proj_S", "proj_M") and self.P_proj):  # gradient algorithm
            self.rho = self.Rho_proj()

        elif self.map_method in ("proj_S", "proj_M") and self.P_proj == 0:  # projection gradient algorithm
            with torch.no_grad():
                rho = self.Rho_proj()  
                params = self.H_to_T(rho)
                self.params.copy_(params)
            self.rho = self.T_to_H(self.params)

        P_out = self.Measure_rho()  # perfect measurement
        return P_out

    def Rho_T(self):
        """factorization"""
        if self.map_method in ('fac_t', 'fac_h'):
            H = torch.tril(self.params) + 1j * torch.triu(self.params, 1).T

            if self.map_method == 'fac_h':
                H += torch.tril(H, -1).T.conj()

        elif self.map_method == "fac_a":  # fac_a
            H = self.params[:, :2**self.N] + 1j * self.params[:, 2**self.N:]

        rho = torch.matmul(H.T.conj(), H)
        rho = rho / torch.trace(rho)
        return rho

    def Rho_proj(self):
        """projection"""
        H = self.T_to_H(self.params)
        rho = proj_spectrahedron_torch(
            H, self.device, self.map_method, self.P_proj)
        return rho

    @staticmethod
    def T_to_H(T):
        H = torch.tril(T) + 1j * torch.triu(T, 1).T  
        H += torch.tril(H, -1).T.conj()
        return H

    @staticmethod
    def H_to_T(H):
        T = torch.tril(H).real + torch.tril(H, -1).imag.T
        return T

    def Measure_rho(self):
        """perfect measurement"""
        self.rho = self.rho.to(torch.complex64)
        P_all = qmt_torch(self.rho, [self.M] * self.N)

        P_real = P_all[self.P_idxs]
        return P_real


class UGD():
    def __init__(self, na_state, map_method, generator, P_star, learning_rate=0.01, optim_f="M"):
        """
        Args:
            generator (generator): The network used for training.
            P_star (tensor): Probability distribution data from experimental measurements.
            learning_rate (float): Learning rate of the optimizer.
            optim_f: the flag of optimizer, if "M": Rprop, else: SGD 

        Net setups:
            Optimizer: Rpop.
            Loss: CF_loss in ``Basis/Loss_Function``.
        """
        super().__init__

        # torch.compile(generator, mode="max-autotune")
        self.generator = generator
        self.P_star = P_star
        self.map_method = map_method

        self.criterion = MLE_loss

        if optim_f == "M":
            self.optim = Rprop(self.generator.parameters(), lr=learning_rate, etas=(0.7, 1.1), momentum=1e-2)  # 0.65, 1.1 1e-2
        else:
            self.optim = optim.SGD(self.generator.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=1e-3)  # 0.5  0.9, nesterov=True, weight_decay=1e-3

    def train(self, epochs, fid, result_save):
        """Net training"""
        self.sche = optim.lr_scheduler.StepLR(self.optim, step_size=1500, gamma=0.2)

        pbar = tqdm(range(epochs), mininterval=0.01)
        epoch = 0
        time_all = 0
        for i in pbar:
            epoch += 1
            time_b = perf_counter()

            self.generator.train()

            def closure():
                self.optim.zero_grad()
                data = self.P_star
                P_out = self.generator()
                loss = self.criterion(P_out, data)
                assert torch.isnan(loss) == 0, print('loss is nan', loss)
                loss.backward()
                return loss

            self.optim.step(closure)
            self.sche.step()

            time_e = perf_counter()
            time_all += time_e - time_b

            # show and save
            if epoch % 10 == 0 or epoch == 1:
                loss = closure().item()
                self.generator.eval()
                with torch.no_grad():
                    rho = self.generator.rho
                    rho /= torch.trace(rho)

                    Fq = fid.Fidelity(rho)

                    result_save['time'].append(time_all)
                    result_save['epoch'].append(epoch)
                    result_save['Fq'].append(Fq)
                    pbar.set_description(
                        "UGD loss {:.10f} | Fq {:.8f} | time {:.5f}".format(loss, Fq, time_all))

                if Fq >= 0.9999:
                    break

        pbar.close()
