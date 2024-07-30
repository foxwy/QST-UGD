# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2021-01-19 15:38:06
# @Last Modified by:   yong
# @Last Modified time: 2024-07-30 15:37:58
# @Function: Quantum state and quantum measurment
# @Paper: Efficient factored gradient descent algorithm for quantum state tomography

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append('..')
from Basis.Basic_Function import semidefinite_adjust


class State():
    """
    Some basic quantum states and matrices, including |0>, |1>, Pauli matrices, GHZ_P,
    GHZi_P, Product_P, W_P, and some random states.

    Examples::
        >>> st = State()
        >>> GHZ_state = st.Get_GHZ_P(1, 0.3)
        >>> (matrix([[0.70710678], [0.70710678]]), 
              array([[0.5 , 0.15], 
                     [0.15, 0.5 ]]))
        >>> GHZ_state = st.Get_state_rho('GHZ_P', 1, 0.3)
    """
    def __init__(self):
        # Pauli matrices
        self.I = np.array([[1, 0], [0, 1]])
        self.X = np.array([[0, 1], [1, 0]])
        self.s1 = self.X
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.s2 = self.Y
        self.Z = np.array([[1, 0], [0, -1]])
        self.s3 = self.Z

        # state
        self.state0 = np.matrix([[1], [0]])
        self.state1 = np.matrix([[0], [1]])
        self.state01 = 1 / np.sqrt(2) * (self.state0 + self.state1)

    def Get_state_rho(self, state_name, N, p=1, r=1):
        """
        Obtain the corresponding quantum state based on the input.

        Args:
            state_name (str): The name of quantum states, include [GHZ_P], [GHZi_P],
                [Product_P], [W_P], [random_P], [real_random].
            N (int): The number of qubits.
            p (int): The P of Werner state, pure state when p == 1, identity matrix when p == 0.
            r (int): The rank of target state, pure state when r == 1.

        Returns:
            matrix: Pure state.
            array: Rho, mixed state.

        Examples:
            >>> st = State()
            >>> GHZ_state = st.Get_state_rho('GHZ_P', 1, 0.3)
            >>> (matrix([[0.70710678], [0.70710678]]), 
                  array([[0.5 , 0.15], 
                         [0.15, 0.5 ]]))
        """
        if state_name == 'GHZ_P':
            state, rho = self.Get_GHZ_P(N, p)
        elif state_name == 'GHZi_P':
            state, rho = self.Get_GHZi_P(N, p)
        elif state_name == 'Product_P':
            state, rho = self.Get_Product_P(N, p)
        elif state_name == 'W_P':
            state, rho = self.Get_W_P(N, p)
        elif state_name == 'random_P':
            state, rho = self.Get_random_state(N, p)
        elif state_name == 'real_random_pur':
            state, rho = self.Get_real_random_state_purity(N, p)
        elif state_name == 'real_random_rank':
            state, rho = self.Get_real_random_state_rank(N, r)
        else:
            print('sorry, we have not yet achieved other quantum states!!!')
        return state, rho

    def Get_GHZ_P(self, N, p):
        """
        N_qubit Werner state with GHZ state,
        rho = p * |GHZ><GHZ| + (1 - P) / d * I.

        Args:
            N (int): The number of qubits.
            p (float): 0 <= p <= 1, |GHZ> when p == 1, I when p == 0.

        Returns:
            matrix: |GHZ>.
            array: rho.
        """
        assert p >= 0 and p <= 1, print('please input ``p`` of [0, 1]')

        state0 = self.state0
        state1 = self.state1
        for _ in range(N - 1):
            state0 = np.kron(state0, self.state0)
            state1 = np.kron(state1, self.state1)
        GHZ_state = 1 / np.sqrt(2) * (state0 + state1)
        GHZ_rho = GHZ_state.dot(GHZ_state.T.conjugate())

        # GHZ_P
        GHZ_P_rho = p * GHZ_rho + (1 - p) / 2**N * np.matrix(np.eye(2**N))
        return GHZ_state, np.array(GHZ_P_rho)

    def Get_GHZi_P(self, N, p):
        """
        N_qubit Werner state with GHZi state,
        rho = p * |GHZi><GHZi| + (1 - P) / d * I.

        Args:
            N (int): The number of qubits.
            p (float): 0 <= p <= 1, |GHZi> when p == 1, I when p == 0.

        Returns:
            matrix: |GHZi>.
            array: rho.
        """
        assert p >= 0 and p <= 1, print('please input ``p`` of [0, 1]')

        state0 = self.state0
        state1 = self.state1
        for _ in range(N - 1):
            state0 = np.kron(state0, self.state0)
            state1 = np.kron(state1, self.state1)
        GHZi_state = 1 / np.sqrt(2) * (state0 + 1j * state1)
        GHZi_rho = GHZi_state.dot(GHZi_state.T.conjugate())

        # GHZi_p
        GHZi_P_rho = p * GHZi_rho + (1 - p) / 2**N * np.matrix(np.eye(2**N))
        return GHZi_state, np.array(GHZi_P_rho)

    def Get_Product_P(self, N, p):
        """
        N_qubit Werner state with Product state,
        rho = p * |Product><Product| + (1 - P) / d * I.

        Args:
            N (int): The number of qubits.
            p (float): 0 <= p <= 1, |Product> when p == 1, I when p == 0.

        Returns:
            matrix: |Product>.
            array: rho.
        """
        assert p >= 0 and p <= 1, print('please input ``p`` of [0, 1]')

        Product_state = self.state01
        for _ in range(N - 1):
            Product_state = np.kron(Product_state, self.state01)
        Product_rho = Product_state.dot(Product_state.T.conjugate())

        # Product_P
        Product_P_rho = p * Product_rho + (1 - p) / 2**N * np.matrix(np.eye(2**N))
        return Product_state, np.array(Product_P_rho)

    def Get_state_from_array(self, array):
        """Calculate the corresponding pure state according to the given array"""
        st = {0: self.state0, 1: self.state1}
        State = st[array[0]]
        for i in array[1:]:
            State = np.kron(State, st[i])
        return State

    def Get_W_P(self, N, p):
        """
        N_qubit Werner state with W state,
        rho = p * |W><W| + (1 - P) / d * I.

        Args:
            N (int): The number of qubits.
            p (float): 0 <= p <= 1, |W> when p == 1, I when p == 0.

        Returns:
            matrix: |W>.
            array: rho.
        """
        assert p >= 0 and p <= 1, print('please input ``p`` of [0, 1]')

        I_array = np.identity(N)
        W_state = 0
        for row in I_array:
            W_state += self.Get_state_from_array(row)
        W_state = 1 / np.sqrt(N) * W_state
        W_rho = W_state.dot(W_state.T.conjugate())

        # W_P
        W_P_rho = p * W_rho + (1 - p) / 2**N * np.matrix(np.eye(2**N))
        return W_state, np.array(W_P_rho)

    def Get_random_state(self, N, purity):
        """
        Generate ``N``-qubit mixed states of corresponding ``purity`` with exponential 
        decay of eigenvalues, see paper ``Projected gradient descent algorithms for 
        quantum state tomography``.

        Args:
            N (int): The number of qubits.
            purity (float): The purity of mixed states, 0 <= purity <= 1.

        Returns:
            rho, if purity != 1, otherwise, pure state.
            rho, mixed state.
        """
        assert purity >= 0 and purity <= 1, print('please input purity of [0, 1]')
        
        if purity != 1:
            lambda_t = 0
            purity_t = 0
            x = np.arange(1, 2**N + 1)

            while purity_t < purity:
                lambda_t += 0.001
                lam = np.exp(-lambda_t * x)
                lamb = lam / np.sum(lam)
                purity_t = np.sum(lamb**2)

            randM = np.random.uniform(size=(2**N, 2**N)) * \
                    np.exp(1j * 2 * np.pi * np.random.uniform(size=(2**N, 2**N)))

            Q, _ = np.linalg.qr(randM)
            lamb_tmp = np.abs(lamb) / np.sum(np.abs(lamb))
            rho = (Q * lamb_tmp).dot(Q.T.conj())
            return rho, rho
        else:
            x_r = np.random.uniform(-1, 1, size=(2**N, 1))
            x_i = np.random.uniform(-1, 1, size=(2**N, 1))
            x = x_r + 1j * x_i
            x /= np.linalg.norm(x)

            rho = x.dot(x.T.conjugate())
            return x, rho

    def Get_random_pure_state(self, N):
        psi = np.random.choice(np.array([0, 1, 1j, -1, -1j]), (2**N, 1), replace=True)
        psi = psi.astype(np.complex64)
        psi /= np.linalg.norm(psi)  # pure state
        return psi

    def Get_real_random_state_purity(self, N, p=None):
        """
        Random Werner state with weight p, where the pure state has only 0 or 1 or 1i on each base. See paper 
        ``Efficient factored gradient descent algorithm for quantum state tomography``.
        """
        psi = self.Get_random_pure_state(N)
        rho = psi.dot(psi.T.conjugate())

        if p is None:  # random
            p = np.random.uniform(size=1)[0]
        else:  # given
            assert p >= 0 and p <= 1, print('please input ``p`` of [0, 1]')

        rho = p * rho + (1 - p) / 2**N * np.eye(2**N)
        return psi, rho

    def Get_real_random_state_rank(self, N, r=None):
        """
        Random mixed state with rank r, where the pure state has only 0 or 1 or 1i on each base. See paper 
        ``Efficient factored gradient descent algorithm for quantum state tomography``.
        """
        rho = 0
        rt = 0

        while rt < r:
            psi = np.random.choice(np.array([0, 1, 1j, -1, -1j]), (2**N, r), replace=True)
            rho = psi.dot(psi.T.conjugate())
            rho /= np.trace(rho).real

            rt = np.linalg.matrix_rank(rho, hermitian=True)
            #eigenvalues, eigenvecs = np.linalg.eigh(rho)
            #rs = min(eigenvalues)
            #print(rt, rs)
        rho /= np.trace(rho).real

        return psi, rho

    @staticmethod
    def Rho_info(rho):
        """
        Print out some values of the density matrix.
        """
        print('rho:\n', rho)
        print('rho+:\n', rho.T.conjugate())
        print('rho+=rho\n', rho.T.conjugate() == rho)
        print('Tr(rho)\n', np.trace(rho))

    @staticmethod
    def Is_rho(rho):
        """
        Determine if ``rho`` is a density matrix.

        Density matrix properties:
            1. unit trace.
            2. semi-definite positive.
            3. Hermitian.
        """
        if abs(np.trace(rho) - 1) < 1e-6 and np.all(np.abs(rho - rho.T.conjugate())) < 1e-7 and semidefinite_adjust(rho):
            return 1
        else:
            if abs(np.trace(rho) - 1) > 1e-6:
                print('trace of rho is not 1')
            if np.all(np.abs(rho - rho.T.conjugate())) > 1e-6:
                print('rho is not Hermitian')
            if not semidefinite_adjust(rho):
                print('rho is not positive semidefine')
            return 0


class Mea_basis(State):
    """
    Defining Quantum Measurement, include POVM and Pauli measurement.

    Examples::
        >>> Me = Mea_basis(basis='Tetra4')
        >>> M = Me.M
        >>> [[[ 0.39433756+0.j          0.14433756-0.14433756j]
              [ 0.14433756+0.14433756j  0.10566244+0.j        ]]

             [[ 0.39433756+0.j         -0.14433756+0.14433756j]
              [-0.14433756-0.14433756j  0.10566244+0.j        ]]

             [[ 0.10566244+0.j         -0.14433756-0.14433756j]
              [-0.14433756+0.14433756j  0.39433756+0.j        ]]

             [[ 0.10566244+0.j          0.14433756+0.14433756j]
              [ 0.14433756-0.14433756j  0.39433756+0.j        ]]]
    """
    def __init__(self, basis='Tetra'):
        """
        Selection of different measurement bases.

        Args:
            basis: ['Tetra'], ['Tetra4'], ['6Pauli'], ['4Pauli'], ['Pauli'], 
                   ['Pauli_rebit'], [Pauli_6'], ['Trine'], ['Psi2'], ['Pauli_normal'].

        Variables: 
            self.K: The number of POVM elements.
            slef.M: POVM or Pauli operators.
        """
        super().__init__()
        self.basis = basis
        self.Get_basis()

    def Get_basis(self):
        """POVM and Pauli operators"""
        if self.basis == 'Tetra':
            self.K = 4

            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)

            v1 = np.array([0, 0, 1.0])
            self.M[0, :, :] = 1.0 / 4.0 * \
                (self.I + v1[0] * self.s1 + v1[1] * self.s2 + v1[2] * self.s3);

            v2 = np.array([2.0 * np.sqrt(2.0) / 3.0, 0.0, -1.0 / 3.0])
            self.M[1, :, :] = 1.0 / 4.0 * \
                (self.I + v2[0] * self.s1 + v2[1] * self.s2 + v2[2] * self.s3);

            v3 = np.array(
                [-np.sqrt(2.0) / 3.0, np.sqrt(2.0 / 3.0), -1.0 / 3.0])
            self.M[2, :, :] = 1.0 / 4.0 * \
                (self.I + v3[0] * self.s1 + v3[1] * self.s2 + v3[2] * self.s3);

            v4 = np.array(
                [-np.sqrt(2.0) / 3.0, -np.sqrt(2.0 / 3.0), -1.0 / 3.0])
            self.M[3, :, :] = 1.0 / 4.0 * \
                (self.I + v4[0] * self.s1 + v4[1] * self.s2 + v4[2] * self.s3);

        elif self.basis == 'Tetra4':
            self.K = 4

            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)

            self.M[0, :, :] = 0.25 * (self.I + (self.X + self.Y + self.Z) / np.sqrt(3))
            self.M[1, :, :] = 0.25 * (self.I + (-self.X - self.Y + self.Z) / np.sqrt(3))
            self.M[2, :, :] = 0.25 * (self.I + (-self.X + self.Y - self.Z) / np.sqrt(3))
            self.M[3, :, :] = 0.25 * (self.I + (self.X - self.Y - self.Z) / np.sqrt(3))

        elif self.basis == '6Pauli':
            self.K = 6

            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)

            self.M[0, :, :] = (self.I + self.X) / 6
            self.M[1, :, :] = (self.I - self.X) / 6
            self.M[2, :, :] = (self.I + self.Y) / 6
            self.M[3, :, :] = (self.I - self.Y) / 6
            self.M[4, :, :] = (self.I + self.Z) / 6
            self.M[5, :, :] = (self.I - self.Z) / 6

        elif self.basis == '4Pauli':  # different from paper
            self.K = 4

            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)

            self.M[0, :, :] = 1.0 / 3.0 * np.array([[1, 0], [0, 0]])
            self.M[1, :, :] = 1.0 / 6.0 * np.array([[1, 1], [1, 1]])
            self.M[2, :, :] = 1.0 / 6.0 * np.array([[1, -1j], [1j, 1]])
            self.M[3, :, :] = 1.0 / 3.0 * (np.array([[0, 0], [0, 1]]) +
                                           0.5 * np.array([[1, -1], [-1, 1]])
                                           + 0.5 * np.array([[1, 1j], [-1j, 1]]))

        elif self.basis == 'Pauli':
            self.K = 6
            Ps = np.array([1. / 3., 1. / 3., 1. / 3.,
                           1. / 3., 1. / 3., 1. / 3.])
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            theta = np.pi / 2.0
            self.M[0, :, :] = Ps[0] * self.pXp(theta, 0.0)
            self.M[1, :, :] = Ps[1] * self.mXm(theta, 0.0)
            self.M[2, :, :] = Ps[2] * self.pXp(theta, np.pi / 2.0)
            self.M[3, :, :] = Ps[3] * self.mXm(theta, np.pi / 2.0)
            self.M[4, :, :] = Ps[4] * self.pXp(0.0, 0.0)
            self.M[5, :, :] = Ps[5] * self.mXm(0, 0.0)

        elif self.basis == 'Pauli_rebit':  # X
            self.K = 4
            Ps = np.array([1. / 2., 1. / 2., 1. / 2., 1. / 2.])
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            theta = np.pi / 2.0
            self.M[0, :, :] = Ps[0] * self.pXp(theta, 0.0)
            self.M[1, :, :] = Ps[1] * self.mXm(theta, 0.0)
            self.M[2, :, :] = Ps[2] * self.pXp(0.0, 0.0)
            self.M[3, :, :] = Ps[3] * self.mXm(0, 0.0)
            self.M = self.M.real

        elif self.basis == 'Pauli_6':
            self.K = 6
            Ps = np.array([1. / 3., 1. / 6., 1. / 2.])
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            self.M[0, :, :] = Ps[0] * np.array([[1, 0], [0, 0]])
            self.M[1, :, :] = Ps[0] * np.array([[0, 0], [0, 1]])
            self.M[2, :, :] = Ps[1] / 2 * np.array([[1, 1], [1, 1]])
            self.M[3, :, :] = Ps[1] / 2 * np.array([[1, -1], [-1, 1]])
            self.M[4, :, :] = Ps[2] / 2 * np.array([[1, -1j], [1j, 1]])
            self.M[5, :, :] = Ps[2] / 2 * np.array([[1, 1j], [-1j, 1]])

        elif self.basis == 'Trine':
            self.K = 3
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            phi0 = 0.0
            for k in range(self.K):
                phi = phi0 + (k) * 2 * np.pi / 3.0
                self.M[k, :, :] = 0.5 * (self.I + np.cos(phi)
                                         * self.Z + np.sin(phi) * self.X) * 2 / 3.0
            self.M = self.M.real

        elif self.basis == 'Psi2':
            self.K = 2
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            self.M[0, 0, 0] = 1
            self.M[1, 1, 1] = 1

        elif self.basis == 'Pauli_normal':  # projective measurement, not POVM
            self.K = 4
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            self.M[0, :, :] = 1 / np.sqrt(2) * np.array([[1.0, 0.0], [0.0, 1.0]])
            self.M[1, :, :] = 1 / np.sqrt(2) * np.array([[0.0, 1.0], [1.0, 0.0]])
            self.M[2, :, :] = 1 / np.sqrt(2) * np.array([[0.0, -1j], [1j, 0.0]])
            self.M[3, :, :] = 1 / np.sqrt(2) * np.array([[1.0, 0.0], [0.0, -1.0]])

        else:
            print(self.basis, 'does not exist!')

    def Basis_info(self):
        """
        Print out some values of the quantum measurement.
        """
        print('K:\n', self.K)
        print('M:\n', self.M)
        print('M+:\n', [self.M[i].T.conjugate() for i in range(len(self.M))])
        print('M+=M:\n', [self.M[i].T.conjugate() == self.M[i]
                          for i in range(len(self.M))])
        print('sum(M):\n', sum([self.M[i] for i in range(len(self.M))]))

        for M in self.M:
            semidefinite_adjust(M, 'M')

    @staticmethod
    def pXp(theta, phi):
        return np.array([[np.cos(theta / 2.0)**2, np.cos(theta / 2.0) * np.sin(theta / 2.0) * np.exp(-1j * phi)],
                         [np.cos(theta / 2.0) * np.sin(theta / 2.0) * np.exp(1j * phi), np.sin(theta / 2.0)**2]])

    @staticmethod
    def mXm(theta, phi):
        return np.array([[np.sin(theta / 2.0)**2, -np.cos(theta / 2.0) * np.sin(theta / 2.0) * np.exp(-1j * phi)],
                         [-np.cos(theta / 2.0) * np.sin(theta / 2.0) * np.exp(1j * phi), np.cos(theta / 2.0)**2]])


#--------------------main--------------------
if __name__ == '__main__':
    '''
    B = Mea_basis('Tetra')  # Tetra, 4Pauli, Pauli, Trine, Psi2
    #print(B.K, B.M)
    B.Basis_info()'''

    #print(State().Get_state_rho('GHZi_P', 2, 0.5))

    #MPS_state().MPS_info()

    '''
    for n in range(2, 11):
        rho_all, _ = State().Get_uniform_state(n, 200)
        np.save('../datasets/uniform_state/N'+str(n)+'.npy', rho_all)'''

    #print(Mea_basis(basis='Tetra4').M)
    r = 10
    _, rho = State().Get_real_random_state_rank(10, r)
    print(np.linalg.matrix_rank(rho, hermitian=True), np.trace(rho).real, np.trace(rho @ rho).real, 1/r)
    print(State().Is_rho(rho))