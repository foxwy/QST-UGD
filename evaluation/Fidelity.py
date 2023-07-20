# -*- coding: utf-8 -*-
# @Author: foxwy
# @Function: Provide some loss functions
# @Paper: Unified factored and projected gradient descent for quantum state tomography

import sys
import numpy as np
import torch
from scipy.linalg import eigh
import time
#import tensorly as tl
#from tensorly.decomposition import matrix_product_state

#torch.set_default_dtype(torch.double)

sys.path.append('..')
from Basis.Basic_Function import qmt, qmt_pure, qmt_torch, qmt_torch_pure
from Basis.Basis_State import Mea_basis, State


class Fid(Mea_basis):
    """
    Calculating Classical Fidelity Drinking Quantum Fidelity.

    Examples::
        >>> _, rho_star = State().Get_state_rho('GHZ_P', 2, 0.2)
        >>> rho_star = torch.tensor(rho_star)
        >>> M = Mea_basis('Tetra4').M
        >>> M = torch.tensor(M)
        >>> fid = Fid('Tetra4', 2, 'mixed', rho_star, M, device='cpu'')
        >>> _, rho_hat = State().Get_state_rho('GHZ_P', 2, 0.1)
        >>> rho_hat = torch.tensor(rho_hat)
        >>> Fq = fid.Fidelity(rho_hat)
    """
    def __init__(self, basis, n_qubits, ty_state, rho_star, M=None, device='cpu'):
        """
        Args:
            basis (str): The name of measurement, asis: ['Tetra'], ['Tetra4'], 
                ['6Pauli'], ['4Pauli'], ['Pauli'], ['Pauli_rebit'], [Pauli_6'], 
                ['Trine'], ['Psi2'], ['Pauli_normal'].
            n_qubits (int): The number of qubits.
            ty_state (str): The type of state, include 'mixed' and 'pure'.
            rho_star (tensor): Desired quantum state, or quantum states that want to tomography.
            M (tensor): The POVM, size (K, 2, 2).
            device (torch.device): GPU or CPU. 
        """
        super().__init__(basis)
        self.N = n_qubits
        self.ty_state = ty_state

        if M is not None:
            self.M = M

        if isinstance(rho_star, list):
            self.rho_p = rho_star[0]
            self.rho_star = rho_star[1]
            self.pure_flag = 1
        else:
            self.rho_star = rho_star
            self.pure_flag = 0

        self.P_all = self.get_real_p(self.rho_star) 

    def Fidelity(self, rho):
        """
        Quantum fidelity, ``rho`` is the calculated quantum state.

        When it is a pure state, calculated as defined, when it is a mixed state, 
        calculated as decomposed, the details can be found in ``Ultrafast quantum 
        state tomography with feed-forward neural networks``.
        """
        Fq = torch.tensor(0)
        if self.ty_state == 'pure':
            rho = rho / torch.norm(rho)
            tmp = torch.matmul(self.rho_star.T.conj(), rho)[0, 0]
            Fq = (tmp * tmp.conj()).real
        else:
            rho = rho / torch.trace(rho)
            if self.pure_flag == 1:
                Fq = torch.matmul(torch.matmul(self.rho_p.T.conj(), rho), self.rho_p)[0, 0].real
            else:
                eigenvalues, eigenvecs = torch.linalg.eigh(self.rho_star)
                eigenvalues = torch.abs(eigenvalues)
                sqrt_rho = torch.matmul(eigenvecs * torch.sqrt(eigenvalues), eigenvecs.T.conj())  # sqrtm(self.rho_star)
                rho_tmp = torch.matmul(torch.matmul(sqrt_rho, rho), sqrt_rho)  # sqrtm(self.rho_star).dot(rho).dot(sqrtm(self.rho_star))

                try:
                    eigenvalues = torch.linalg.eigvalsh(rho_tmp)  # fast, in some special cases, an error will be reported
                except Exception:
                    print('error')
                    eigenvalues = torch.linalg.eigvals(rho_tmp)  # low

                sqrt_eigvals = torch.sqrt(torch.abs(eigenvalues))
                Fq = torch.sum(sqrt_eigvals)**2  # trace(sqrtm(sqrtm(self.rho_star).dot(rho).dot(sqrtm(self.rho_star))))**2

        if Fq > 1:
            Fq = 1  # precision error
        return Fq

    def cFidelity_rho(self, rho):
        """classical fidelity, ``rho`` is the calculated quantum state"""
        P_f = self.get_real_p(rho)
        Fc = self.cFidelity(P_f, self.P_all)

        return Fc

    def get_real_p(self, rho):
        """Calculating the probability distribution of quantum measurements"""
        if type(rho) is np.ndarray:  # numpy array
            rho = rho / np.trace(rho)
            if self.ty_state == 'pure':
                P_all = qmt_pure(rho, [self.M] * self.N)
            else:
                P_all = qmt(rho, [self.M] * self.N)

        elif type(rho) is torch.Tensor:  # torch tensor
            rho = rho / torch.trace(rho)
            if self.ty_state == 'pure':
                P_all = qmt_torch_pure(rho, [self.M] * self.N)
            else:
                P_all = qmt_torch(rho, [self.M] * self.N)
        return P_all

    def cFidelity_S_product(self, P_idxs, P_f, unique_flag=1):  # fast!!!
        """The classical fidelity is calculated for a given evaluated probability"""
        P_real = self.P_all[P_idxs]
        Fc = self.cFidelity(P_f, P_real, unique_flag)
        return Fc

    @staticmethod
    def cFidelity(P_f, P_real, unique_flag=1):
        """
        Calculated according to the definition of classical fidelity (sum_k sqrt{P_k*Q_K})^2.

        Args:
            P_f (array, tensor): Probability distribution of the evaluation.
            P_real (array, tensor): Probability distribution that want to approximate.
            unique_flag (1, !=1): If there is a value on each measurement, it is 1.
        """
        if type(P_f) is np.ndarray:
            if unique_flag == 1:  # has not repeat elements
                Fc = np.sqrt(P_real).dot(np.sqrt(P_f))
            else:
                Fc = np.sqrt(P_real).dot(1 / np.sqrt(P_f)) / len(P_real)
            return Fc**2
            
        elif type(P_f) is torch.Tensor:
            Fc = torch.sqrt(P_f).dot(torch.sqrt(P_real))
            return Fc.item()**2


# --------------------main--------------------
if __name__ == '__main__':
    pass
