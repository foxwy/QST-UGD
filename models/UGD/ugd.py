# -*- coding: utf-8 -*-
# @Author: yong
# @Date:   2022-12-07 09:46:43
# @Last Modified by:   yong
# @Last Modified time: 2023-07-20 16:52:19
# @Function: single-layer network, use same optimizer and others setups as neural network
# @Paper: Unified factored and projected gradient descent for quantum state tomography

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from time import perf_counter
from tqdm import tqdm

#torch.set_default_dtype(torch.double)

sys.path.append('../..')
from Basis.Basic_Function import qmt_torch, get_default_device, proj_spectrahedron_torch, qmt_matrix_torch
from Basis.Loss_Function import MLE_loss, CF_loss
from evaluation.Fidelity import Fid
from Basis.Basis_State import Mea_basis, State
from models.UGD.cg_optim import cg
from models.UGD.rprop import Rprop


class UGD_nn(nn.Module):
    """
    The single-layer network is used to perform the quantum state tomography task by directly 
    optimizing the layer parameters and mapping them to the density matrix and measuring the 
    distance from the probability distribution to optimize the network parameters, 
    see paper ``Unified factored and projected gradient descent for quantum state tomography``.

    Examples::
        see ``FNN/FNN_learn``.
    """
    def __init__(self, n_qubits, 
                       P_idxs, 
                       M, 
                       map_method='chol_h', 
                       P_proj=1.5):
        """
        Args:
            n_qubits (int): The number of qubits.
            P_idxs (tensor): Index of the POVM used for measurement, Not all measurements 
                are necessarily used.
            M (tensor): The POVM, size (K, 2, 2).
            rho_init (tensor): If None, initialize the parameters randomly, and vice versa with rho.
            map_method (str): State-mapping method, include ['chol', 'chol_h', 'proj_F', 'proj_S', 'proj_A'].
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
        params = torch.randn(d, d, requires_grad=True).to(torch.complex64)
        self.params = nn.Parameter(params)

    def forward(self):
        if 'chol' in self.map_method:
            self.rho = self.Rho_T()  # decomposition
        elif 'proj' in self.map_method:
            self.rho = self.Rho_proj()  # projection

        P_out = self.Measure_rho()  # perfect measurement
        return P_out

    def Rho_T(self):
        """decomposition"""
        T_m = self.params  #.view(2**self.N, -1)
        T_triu = torch.triu(T_m, 1)
        T = torch.tril(T_m) + 1j * T_triu.T

        if self.map_method == 'chol_h':
            T += torch.tril(T, -1).T.conj()
        T_temp = torch.matmul(T.T.conj(), T)

        rho = T_temp / torch.trace(T_temp)
        return rho

    def Rho_proj(self):
        """projection"""
        T_m = self.params  #.view(2**self.N, -1)
        T_triu = torch.triu(T_m, 1)
        T = torch.tril(T_m) + 1j * T_triu.T
        #T += torch.tril(T, -1).T.conj()  # cause torch.linalg.eigh only use the lower triangular part of the matrix
        rho = proj_spectrahedron_torch(T, self.device, self.map_method, self.P_proj)
        return rho

    def Measure_rho(self):
        """perfect measurement"""
        self.rho = self.rho.to(torch.complex64)
        P_all = qmt_torch(self.rho, [self.M] * self.N)

        P_real = P_all[self.P_idxs]
        return P_real


class UGD():
    """
    For network training for SNN.

    Examples::
        see ``FNN/FNN_learn``.
    """
    def __init__(self, generator, P_star, learning_rate=0.01):
        """
        Args:
            generator (generator): The network used for training.
            P_star (tensor): Probability distribution data from experimental measurements.
            learning_rate (float): Learning rate of the optimizer.

        Net setups:
            Optimizer: Rpop.
            Loss: CF_loss in ``Basis/Loss_Function``.
        """
        super().__init__

        self.generator = generator  # torch.compile(generator, mode="max-autotune")
        self.P_star = P_star

        self.criterion = MLE_loss  #nn.MSELoss()
        self.ls = self.criterion(self.P_star, self.P_star)
        #self.optim = cg(self.generator.parameters(), lr=1e2, ls=self.ls, a1=0.1, a2=0.2, line_search='BB')
        self.optim = Rprop(self.generator.parameters(), lr=learning_rate, etas=(0.66, 1.14), momentum=1e-4)
        #self.optim = optim.Rprop(self.generator.parameters(), lr=learning_rate)
        #self.optim = optim.LBFGS(self.generator.parameters(), lr=learning_rate)

    def train(self, epochs, fid, result_save):
        """Net training"""
        #self.sche = optim.lr_scheduler.LinearLR(self.optim, start_factor=0.1, total_iters=epochs)

        pbar = tqdm(range(epochs), mininterval=0.001)
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
            #self.sche.step()

            time_e = perf_counter()
            time_all += time_e - time_b
           
            # show and save
            if epoch % 20 == 0:
                loss = closure().item()
                self.generator.eval()
                with torch.no_grad():
                    Fc = fid.cFidelity_rho(self.generator.rho)
                    Fq = fid.Fidelity(self.generator.rho)

                    result_save['time'].append(time_all)
                    result_save['epoch'].append(epoch)
                    result_save['Fc'].append(Fc)
                    result_save['Fq'].append(Fq)
                    pbar.set_description("UGD loss {:.8f} | Fc {:.8f} | Fq {:.8f} | time {:.4f} | epochs {:d}".format(loss, Fc, Fq, time_all, epoch))

                if Fq >= 0.99:
                    break
        pbar.close()
