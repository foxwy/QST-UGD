# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2021-04-30 09:48:23
# @Last Modified by:   yong
# @Last Modified time: 2024-07-30 15:11:55
# @Function: Provide some of the most basic functions
# @Paper: Efficient factored gradient descent algorithm for quantum state tomography

import os
import sys
from time import perf_counter
import random
import numpy as np
from scipy.linalg import eigh
import torch
from torch.nn.functional import softmax

# environment
#torch.set_default_dtype(torch.double)

filepath = os.path.abspath(os.path.join(os.getcwd(), '../..'))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append('..')
from evaluation.ncon import ncon


def get_default_device():
    """
    Detects whether it is a CPU device or a GPU device in Pytorch.

    Returns:
        device('cuda') if GPU exists, otherwise device('cpu').
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def clamp(n, minn, maxn):  # limit num to [minn, maxn]
    """
    Restrict a number ``n`` to be between ``minn`` and ``maxn``

    Args:
        n: Any number.
        minn: Restricted range lower bound.
        maxn: Restricted range upper bound.

    Returns:
        n,    if minn <= n <= maxn;
        maxn, if maxn <= n;
        minn, if n <= minn. 
    """
    if minn > maxn:
        minn, maxn = maxn, minn
    return max(min(maxn, n), minn)


def num_to_groups(num, divisor) -> list:
    """
    Grouping ``num`` by ``divisor``

    Examples::
        >>> num_to_groups(12, 4)
        >>> [4, 4, 4]
        >>> num_to_groups(36, 32)
        >>> [32, 4]
        >>> num_to_groups(12.3, 4)
        >>> [4, 4, 4, 0.3]
    """
    groups = int(num // divisor)
    remainder = num - divisor * groups
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def ten_to_k(num, k, N) -> list:
    """
    Convert decimal ``num`` to ``k`` decimal and complementary

    Args:
        num: Decimal numbers.
        k: k decimal.
        N: Total number of digits.

    Returns:
        Converted k decimal list.

    Examples::
        >>> ten_to_k(10, 2, 5)
        >>> [0, 1, 0, 1, 0]
        >>> ten_to_k(10, 4, 5)
        >>> [0, 0, 0, 2, 2]
    """
    transfer_num = []
    if num > k**N - 1:  # error num
        print('please input the right number!')
    else:
        while num != 0:
            num, a = divmod(num, k)
            transfer_num.append(a)
        transfer_num = transfer_num[::-1]
        if len(transfer_num) != N:
            transfer_num = [0] * (N - len(transfer_num)) + transfer_num
    return transfer_num


def data_combination(N, k, p=1, seed=1):
    """
    Randomly select ``p`` proportion of the natural numbers from [0, k^N - 1] and 
    convert these numbers to ``k`` decimal with ``N`` complement.

    Args:
        N: Total number of digits.
        k: k decimal.
        p: Percentage of Acquired Number.
        seed: random seed.

    Returns:
        Converted data matrix.
    """
    samples_unique = []
    N_choice = k**N

    if p < 1 and p >= 0:  # random sample
        random.seed(seed)
        N_choice = int(N_choice * p)
        num_choice = random.sample(range(k**N), N_choice)
    else:  # all number
        num_choice = range(N_choice)

    for num in num_choice:
        # Convert to k decimal
        samples_unique.append(ten_to_k(num, k, N))
    return np.array(samples_unique)


def data_combination_M(M, N, k, p=1, seed=1):
    """
    Similarly ``data_combination`` select part of the data and calculate 
    the corresponding measurements ``M``.

    Args:
        M (array): Single-qubit measurement, size = (k, 2, 2).
        N (int): The number of qubits.
        k (int): The number of single-qubit measurement elements.
        p (float): Percentage of Acquired Number.
        seed (float): random seed.

    Returns:
        Array: Acquired samples.
        Array: multi-qubit measurements
    """
    samples_unique = []
    M_all = []

    if p < 1 and p >= 0:  # random sample
        random.seed(seed)
        N_choice = int(k**N * p)
        num_choice = random.sample(range(k**N), N_choice)
    else:  # all number
        num_choice = np.arange(N_choice)

    for num in num_choice:
        sample = ten_to_k(num, k, N)
        samples_unique.append(sample)

        # Kron operator
        M_temp = M[sample[0]]
        for i in sample[1:]:
            M_temp = np.kron(M_temp, M[i])
        M_all.append(M_temp)
    return np.array(samples_unique), np.array(M_all)


def cal_P(M, N, k, rho):
    """
    Calculate the probability of all measurements of the density matrix ``rho``.

    Args:
        M (tensor): Single-qubit measurement, size = (k, 2, 2).
        N (int): The number of qubits.
        k (int): The number of single-qubit measurement elements.
        rho (tensor): density matrix.

    Returns:
        Tensor (cuda): The calculated probability.
    """
    M = M.cpu().numpy()
    rho = rho.cpu().numpy()

    _, M_all = data_combination_M(M, N, k)
    P = np.zeros(len(M_all))
    for i in range(len(M_all)):
        P[i] = np.real(np.trace(M_all[i].dot(rho)))
    return torch.tensor(P).cuda()


def cal_R(X, M, N, k):
    """
    The combined matrix is obtained by multiplying all measurements {``M``} and ``X`` data of N-qubit.

    Args:
        X (tensor): A set of data, size = k*N.
        M (tensor): Single-qubit measurement, size = (k, 2, 2).
        N (int): The number of qubits.
        k (int): The number of single-qubit measurement elements.

    Returns:
        Tensor (cuda): The combined matrix.
    """
    X = X.cpu().numpy()
    M = M.cpu().numpy()
    _, M_all = data_combination_M(M, N, k)
    R = 0
    for i in range(len(M_all)):
        R += M_all[i] * X[i]
    return torch.tensor(R).cuda()


def onehot(data, k):
    """
    Onehot encoding.

    Args:
        data (array): Data matrix waiting for encoding.
        k (int): Number of bits of the code.

    Examples::
        >>> d = numpy.array([[3, 2], [1, 0]])
        >>> onehot(d, 4)
        >>> [[0 0 0 1 0 0 1 0]
             [0 1 0 0 1 0 0 0]]
    """
    data_onehot = []
    N = len(data[0])
    for i in range(len(data)):
        one_hot = np.squeeze(np.reshape(np.eye(k)[data[i]], [1, N * k]).astype(np.uint8)).tolist()
        data_onehot.append(one_hot)
    return np.array(data_onehot)


def ati_onehot(data_onehot, k):
    """Reverse onehot encoding"""
    N = int(data_onehot.shape[1] // k)
    data_onehot_reshape = np.reshape(data_onehot, [data_onehot.shape[0] * N, k])
    data_ati_onehot = np.argmax(data_onehot_reshape, axis=1)
    data_ati_onehot = np.reshape(np.array(data_ati_onehot), [data_onehot.shape[0], N])
    return data_ati_onehot


def array_posibility(a):
    """Counting the frequency of elements in ``a``"""
    x, cnts = np.unique(a, axis=0, return_counts=True)

    b = np.zeros((len(a), 1))
    for i in range(len(a)):
        b[i, 0] = cnts[np.where((x == a[i]).all(1))[0][0]]
    b = b / len(a)
    return b


def array_posibility_unique(a):
    """
    Statistics of non-repeating elements in ``a`` and their frequency.

    Returns:
        array: non-repeating elements.
        array: frequency of non-repeating elements.
    """
    x, cnts = np.unique(a, axis=0, return_counts=True)
    return x, cnts / len(a)


def semidefinite_adjust(M, eps=1e-08):
    """
    Determine whether the matrix ``M`` is a semi-positive definite matrix.

    Returns:
        bool: True if ``M`` is a semi-positive definite matrix, otherwise False.
    """
    M_vals, M_vecs = eigh(M)
    if np.all(M_vals > -eps):
        return True
    else:
        return False


def factorization(num):
    """Factoring a number"""
    factor = []
    while num > 1:
        for i in range(num - 1):
            k = i + 2
            if num % k == 0:
                factor.append(k)
                num = int(num / k)
                break
    return factor


def is_integer(number):
    """Determine if a number is an integer"""
    if int(number) == number:
        return True
    else:
        return False


def crack(integer):
    """Decompose a number as the product of the two closest values"""
    start = int(np.sqrt(integer))
    factor = integer / start
    while not is_integer(factor):
        start += 1
        factor = integer / start
    return int(factor), start


def Find_x(x, b):
    """Find the first n sums of ``x`` greater than ``b`` and return n"""
    if len(x) == 1:
        return 0
    else:
        mid_idx = len(x) // 2
        if sum(x[:mid_idx]) > b:
            x = x[:mid_idx]
            return Find_x(x, b)
        else:
            y = x[mid_idx:]
            y[0] += sum(x[:mid_idx])
            return mid_idx + Find_x(y, b)


def torch_sqrtm(rho):
    """the sqrt of a Hermitian matrix"""
    eigenvalues, eigenvecs = torch.linalg.eigh(rho)  # eigenvalues[i], eigenvecs[:, i]

    A = eigenvecs * torch.sqrt(torch.abs(eigenvalues))
    rho_sqrt = torch.matmul(A, eigenvecs.T.conj())

    return rho_sqrt


def shuffle_forward(rho, dims):
    """
    To transpose the density matrix, see
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [numpy] version we implemented.
    """
    N = len(dims)
    rho = rho.T
    rho = rho.reshape(np.concatenate((dims, dims), 0))
    ordering = np.reshape(np.arange(2*N).reshape(2, -1).T, -1)
    rho = np.transpose(rho, ordering)
    return rho


def qmt(X, operators, allow_negative=False):
    """
    Simplifying the computational complexity of mixed state measurements using the product structure of POVM, see 
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [numpy] version we implemented.

    Args:
        X (array): Density matrix.
        operators (list, [array]): The set of single-qubit measurement, such as [M1, M2, ....], the size of M is (k, 2, 2).
        allow_negative (bool): Flag for whether to set the calculated value negative to zero.
    """
    if not isinstance(operators, list):
        operators = [operators]

    N = len(operators)  # qubits number
    Ks = np.zeros(N, dtype=int)
    Ds = np.zeros(N, dtype=int)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = dims[0]
        Ds[i] = dims[1]
        operators[i] = operators[i].reshape((Ks[i], Ds[i]**2))

    X = shuffle_forward(X, Ds[::-1])
    for i in range(N - 1, -1, -1):
        P = operators[i]
        X = X.reshape(-1, Ds[i]**2)
        X = np.matmul(P, X.T)

    P_all = np.real(X.reshape(-1))
    if not allow_negative:
        P_all = np.maximum(P_all, 0)
        P_all /= np.sum(P_all)
    return P_all


def qmt_pure(X, operators, allow_negative=False):  # operators = [M1, M2, ....], two loops
    """>>>Awaiting further testing<<<
    Simplifying the computational complexity of pure state measurements using 
    the product structure of POVM, this is the [numpy] version we implemented.

    Args:
        X (array): Pure state matrix (column vector).
        operators (list, [array]): The set of single-qubit measurement, such as [M1, M2, ....], the size of M is (k, 2, 2).
        allow_negative (bool): Flag for whether to set the calculated value negative to zero.
    """
    if not isinstance(operators, list):
        operators = [operators]

    X = np.array(X)
    N = len(operators)  # qubits number
    Ks = np.zeros(N, dtype=int)
    Ds = np.zeros(N, dtype=int)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = dims[0]
        Ds[i] = dims[1]
        if i < N - 1:
            operators[i] = operators[i].reshape((Ks[i], Ds[i]**2))

    P_all = np.zeros(np.prod(Ks))
    X = X.reshape(2, -1)
    X_T = X.T.conjugate()
    for k in range(Ks[-1]):
        X_k = X_T.dot(operators[-1][k]).dot(X)

        X_k = shuffle_forward(X_k, Ds[:-1])
        for i in range(N - 2, -1, -1):
            P = operators[i]
            X_k = X_k.reshape(-1, Ds[i]**2).T
            X_k = P.dot(X_k)

        P_all[k * Ks[0]**(N - 1): ((k + 1) * Ks[0]**(N - 1))] = np.real(X_k.reshape(-1))

    if not allow_negative:
        P_all = np.maximum(P_all, 0)
        P_all /= np.sum(P_all)
    return P_all


def shuffle_adjoint(R, dims):
    """
    To transpose the density matrix, see
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [numpy] version we implemented.
    """
    N = len(dims)
    R = R.reshape(np.concatenate((dims, dims), 0))
    ordering = np.arange(2 * N).reshape(-1, 2).T.reshape(-1)
    R = np.transpose(R, ordering)
    R = R.reshape(np.prod(dims), np.prod(dims))
    return R


def qmt_matrix(coeffs, operators):
    """
    Simplifying the computational complexity of mixed state measurement operator mixing using the product structure of POVM, see 
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [numpy] version we implemented.

    Args:
        coeffs (array): Density matrix.
        operators (list, [array]): The set of single-qubit measurement, such as [M1, M2, ....], the size of M is (k, 2, 2).
        allow_negative (bool): Flag for whether to set the calculated value negative to zero.

    Examples::
        >>> M = numpy.array([a, b, c, d])
        >>> qmt_matrix([1, 2, 3, 4], [M])
        >>> 1 * a + 2 * b + 3 * c + 4 * d
        a, b, c, d is a matrix.
    """
    if not isinstance(operators, list):
        operators = [operators]

    X = np.array(coeffs)
    N = len(operators)  # qubits number
    Ks = np.zeros(N, dtype=int)
    Ds = np.zeros(N, dtype=int)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = dims[0]
        Ds[i] = dims[1]
        operators[i] = operators[i].reshape((Ks[i], Ds[i]**2))

    for i in range(N - 1, -1, -1):
        P = operators[i]
        X = X.reshape(-1, Ks[i])
        X = X.dot(P)
        X = X.T

    X = shuffle_adjoint(X, Ds[::-1])
    X = 0.5 * (X + X.T.conj())
    return X


def shuffle_forward_torch(rho, dims):
    """
    To transpose the density matrix, see
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [torch] version we implemented.
    """
    N = len(dims)
    rho = rho.T
    rho = rho.reshape(tuple(torch.cat([dims, dims], 0)))
    ordering = torch.reshape(torch.arange(2*N).reshape(2, -1).T, (1, -1))[0]
    rho = rho.permute(tuple(ordering))
    return rho


def qmt_torch(X, operators, allow_negative=False):
    """
    Simplifying the computational complexity of mixed state measurements using the product structure of POVM, see 
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [torch] version we implemented.

    Args:
        X (tensor): Density matrix.
        operators (list, [tensor]): The set of single-qubit measurement, such as [M1, M2, ....], the size of M is (k, 2, 2).
        allow_negative (bool): Flag for whether to set the calculated value negative to zero.
    """
    if not isinstance(operators, list):
        operators = [operators]

    N = len(operators)  # qubits number
    Ks = torch.zeros(N, dtype=torch.int)
    Ds = torch.zeros(N, dtype=torch.int)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = dims[0]
        Ds[i] = dims[1]
        operators[i] = operators[i].reshape((Ks[i], Ds[i]**2))

    if N > 12:  # torch does not support more dimensional operations
        X = X.cpu()
    X = shuffle_forward_torch(X, Ds)
    X = X.reshape(-1, Ds[-1]**2)
    if N > 12:
        X = X.to(operators[0].device)

    for i in range(N - 1, -1, -1):
        P = operators[i]
        X = torch.matmul(P, X.T)

        if i > 0:
            X = X.reshape(-1, Ds[i]**2)

    P_all = torch.real(X.reshape(-1))
    if not allow_negative:
        P_all = torch.maximum(P_all, torch.tensor(0))
        P_all /= torch.sum(P_all)
    return P_all


def qmt_torch_pure(X, operators, allow_negative=False):
    """>>>Awaiting further testing<<<
    Simplifying the computational complexity of pure state measurements using 
    the product structure of POVM, this is the [torch] version we implemented.

    Args:
        X (tensor): Pure state matrix (column vector).
        operators (list, [tensor]): The set of single-qubit measurement, such as [M1, M2, ....], the size of M is (k, 2, 2).
        allow_negative (bool): Flag for whether to set the calculated value negative to zero.
    """
    if not isinstance(operators, list):
        operators = [operators]

    N = len(operators)  # qubits number
    Ks = torch.zeros(N, dtype=torch.int32)
    Ds = torch.zeros(N, dtype=torch.int32)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = dims[0]
        Ds[i] = dims[1]
        if i < N - 1:
            operators[i] = operators[i].reshape((Ks[i], Ds[i]**2))

    P_all = torch.zeros(torch.prod(Ks)).float().to(X.device)
    X = X.reshape(2, -1).to(torch.complex64)
    X_T = X.T.conj()
    for k in range(Ks[-1]):
        X_k = torch.matmul(X_T, torch.matmul(operators[-1][k], X))

        if N > 13:  # torch does not support more dimensional operations
            X_k = X_k.cpu()
        X_k = shuffle_forward_torch(X_k, Ds[:-1])
        X_k = X_k.reshape(-1, Ds[-1]**2)
        if N > 13:
            X_k = X_k.to(operators[0].device)

        for i in range(N - 2, -1, -1):
            P = operators[i]
            X_k = X_k.permute(*torch.arange(X_k.ndim - 1, -1, -1))
            X_k = torch.matmul(P, X_k)

            if i > 0:
                X_k = X_k.reshape(-1, Ds[i]**2)

        P_all[k * Ks[0]**(N - 1): ((k + 1) * Ks[0]**(N - 1))] = torch.real(X_k.reshape(-1))

    if not allow_negative:
        P_all = torch.maximum(P_all, torch.tensor(0))
        P_all /= torch.sum(P_all)
    return P_all


def shuffle_adjoint_torch(R, dims):
    """
    To transpose the density matrix, see
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [torch] version we implemented.
    """
    N = len(dims)
    R = R.reshape(tuple(torch.cat([dims, dims], 0)))
    ordering = torch.arange(2 * N).reshape(-1, 2).T.reshape(-1)
    R = R.permute(tuple(ordering))
    R = R.reshape(torch.prod(dims), torch.prod(dims))

    return R


def qmt_matrix_torch(X, operators):
    """
    Simplifying the computational complexity of mixed state measurement operator mixing using the product structure of POVM, see 
    the matlab version in paper ``Superfast maximum likelihood reconstruction for quantum tomography``,
    this is the [torch] version we implemented.

    Args:
        coeffs (tensor): Density matrix.
        operators (list, [tensor]): The set of single-qubit measurement, such as [M1, M2, ....], the size of M is (k, 2, 2).
        allow_negative (bool): Flag for whether to set the calculated value negative to zero.

    Examples::
        >>> M = torch.tensor([a, b, c, d])
        >>> qmt_matrix(torch.tensor([1, 2, 3, 4]), [M])
        >>> 1 * a + 2 * b + 3 * c + 4 * d
        a, b, c, d is a matrix.
    """
    if not isinstance(operators, list):
        operators = [operators]

    N = len(operators)  # qubits number
    Ks = torch.zeros(N, dtype=torch.int)
    Ds = torch.zeros(N, dtype=torch.int)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = dims[0]
        Ds[i] = dims[1]
        operators[i] = operators[i].reshape((Ks[i], Ds[i]**2))

    for i in range(N - 1, -1, -1):
        P = operators[i]
        X = X.reshape(-1, Ks[i])
        X = torch.matmul(X, P)
        X = X.T

    X = shuffle_adjoint_torch(X, Ds.flip(dims=[0]))
    X = 0.5 * (X + X.T.conj())
    return X


def qmt_product_torch(operators_1, operators_2):
    """
    To calculate the X matrix in the LRE algorithm, see paper ``Full reconstruction of a 
    14-qubit state within four hours```.
    """
    if not isinstance(operators_1, list):
        operators_1 = [operators_1]

    if not isinstance(operators_2, list):
        operators_2 = [operators_2]

    N = len(operators_1)  # qubits number
    Ks_1 = torch.zeros(N, dtype=torch.int)
    Ds_1 = torch.zeros(N, dtype=torch.int)
    Ks_2 = torch.zeros(N, dtype=torch.int)
    Ds_2 = torch.zeros(N, dtype=torch.int)
    for i in range(N):
        dims = operators_1[i].shape
        Ks_1[i] = dims[0]
        Ds_1[i] = dims[1]
        dims = operators_2[i].shape
        Ks_2[i] = dims[0]
        Ds_2[i] = dims[1]

    operators_t = torch.einsum('...ij->...ji', [operators_1[0]])
    P_single = torch.real(torch.matmul(operators_t.reshape(Ks_1[0], Ds_1[0]**2), operators_2[0].reshape(Ks_2[0], Ds_2[0]**2).T))
    X_t = P_single
    for i in range(N - 1):
        X_t = torch.kron(X_t, P_single)
    return X_t


def proj_spectrahedron(rho):
    """
    Transformation of Hermitian matrix to nearest density matrix, F projection state-mapping method, 
    see paper ``Efficient method for computing the maximum-likelihood quantum state from 
    measurements with additive gaussian noise``,
    this is [numpy] version we implemented.
    """
    eigenvalues, eigenvecs = np.linalg.eigh(rho)  #eigenvalues[i], eigenvecs[:, i]
    eigenvalues = np.real(eigenvalues)
    u = -np.sort(-eigenvalues)
    csu = np.cumsum(u)
    t = (csu - 1) / np.arange(1, len(u) + 1)
    idx_max = np.flatnonzero(u > t)[-1]
    eigenvalues = np.maximum(eigenvalues - t[idx_max], 0)

    print(eigenvecs.shape, eigenvalues.shape)
    A = eigenvecs * np.sqrt(eigenvalues)
    rho = A.dot(A.T.conjugate())
    return rho


def proj_to_sum_one(x, fl):
    """
    project x in to a unit vector according to fl
    """
    if fl == 1:
        x = x / torch.sum(x)

    elif fl == 2:
        x = x + (1 - torch.sum(x)) / len(x)

    return x


def eigenvalues_trans_F(eigenvalues, device, fl):
    """
    Transformation of Hermitian matrix to nearest density matrix, F projection state-mapping method, 
    see paper ``Efficient method for computing the maximum-likelihood quantum state from 
    measurements with additive gaussian noise``,
    this is [torch] version we implemented.
    """
    eigenvalues = proj_to_sum_one(eigenvalues, fl)  # unit sum

    u, _ = torch.sort(eigenvalues)
    csu = torch.cumsum(u, 0)
    csu0 = torch.zeros_like(csu).to(device)
    csu0[1:] = csu[:-1]
    t = csu0 / torch.arange(len(u), 0, -1).to(device)
    idx = torch.nonzero(u + t > 0)[0, 0]
    eigenvalues = torch.maximum(eigenvalues + t[idx], torch.tensor(0))

    return eigenvalues


def eigenvalues_trans_S(eigenvalues, device):
    """
    Transformation of Hermitian matrix to nearest density matrix, S projection state-mapping method,
    see paper ``A projected gradient method for optimization over density matrices``,
    this is [torch] version we implemented.
    """
    idxs = np.arange(len(eigenvalues))
    for i in range(len(eigenvalues)):
        eigenvalues[idxs] = proj_to_sum_one(eigenvalues[idxs], 2)
        if min(eigenvalues) >= 0:
            break
        else:
            eigenvalues[eigenvalues < 0] = 0
            idxs = eigenvalues > 0

    return eigenvalues


def eigenvalues_trans_M(eigenvalues, device):
    eigenvalues = torch.maximum(eigenvalues, torch.tensor(0))
    eigenvalues = eigenvalues_trans_F(eigenvalues, device, 2)

    return eigenvalues


def eigenvalues_trans_abs(eigenvalues, P_proj):
    """
    Transformation of Hermitian matrix to nearest density matrix, P-order absolute 
    projection state-mapping method, see our paper ``Ultrafast quantum state tomography 
    with feed-forward neural networks``.

    Args:
        eigenvalues (tensor): Eigenvalues to be transformed.
        P_proj (int): P order.

    Returns:
        Normalized eigenvalues after mapping.
    """
    eigenvalues_abs = torch.abs(eigenvalues)
    return eigenvalues_abs**P_proj / torch.sum(eigenvalues_abs**P_proj)


def proj_spectrahedron_torch(rho, device, map_method, P_proj=2, trace_flag=1):
    """
    Select the state-mapping method according to the given parameters ``map_method`` and ``P_proj``.

    Args:
        rho (tensor): Matrix that does not satisfy the density matrix property.
        device (torch.device): GPU or CPU.
        map_method (str): State-mapping method, include ['proj_F', 'proj_S', 'proj_A'].
        P_proj (float): P order.
        trace_flag (1, not 1): 1 is required for the density matrix to satisfy a diagonal 
            sum of one, otherwise it is not required.

    Returns:
        The real density matrix.
    """
    eigenvalues, eigenvecs = torch.linalg.eigh(rho)  # eigenvalues[i], eigenvecs[:, i]

    if map_method == 'proj_F':
        eigenvalues = eigenvalues_trans_F(eigenvalues, device, 1)
    elif map_method == 'proj_S':
        eigenvalues = eigenvalues_trans_F(eigenvalues, device, 2) 
    elif map_method == 'proj_M':
        eigenvalues = eigenvalues_trans_M(eigenvalues, device) 
    elif map_method == 'proj_A':
        eigenvalues = eigenvalues_trans_abs(eigenvalues, P_proj)
    else:
        print('we have not this map method! please check setting!!!')

    A = eigenvecs * abs(eigenvalues)
    rho = torch.matmul(A, eigenvecs.T.conj())
    #rho = 0.5 * (rho + rho.T.conj())
    if trace_flag == 1:
        rho /= torch.trace(rho)  # prevent errors caused by computing accuracy
    return rho


def samples_mp(param):
    """
    Quantum sampling, already discarded, too slow!!!
    """
    P_all = param[0]
    group_N = param[1]
    K = param[2]
    N = param[3]

    counts = np.random.multinomial(1, P_all, group_N)
    idxs = np.argmax(counts, 1)
    S_all = []
    S_one_hot_all = []
    for n in range(group_N):
        ii = idxs[n]
        S = np.array(ten_to_k(ii, K, N))
        S_all.append(S)
        S_one_hot_all.append(np.squeeze(np.reshape(np.eye(K)[S], [1, N * K]).astype(np.uint8)).tolist())

    return [np.array(S_all), np.array(S_one_hot_all)]


#--------------------main--------------------
if __name__ == '__main__':
    #a = np.array([[1, 2], [3, 4], [1, 2], [1, 3], [1, 3], [3, 4], [1, 3], [3, 4], [1, 2]])
    # print(array_posibility(a))

    '''
    data = data_combination(2, 4, 0.5)
    print(data)
    data_onehot = onehot(data, 4)
    print(data_onehot)
    print(ati_onehot(data_onehot, 4))'''

    '''
    a, b = crack(164)
    print(a, b)'''

    '''
    a = np.array([[1, 2], [3, 4], [1, 2], [1, 3], [1, 3], [3, 4], [1, 3], [3, 4], [1, 2]])
    data_unique, p = array_posibility_unique(a)
    print(data_unique, p)'''

    '''
    a = np.array([[1, 2, 0, 0.5], [0, 3, 0, 0], [4, 0, 4, 0], [0.5, 0, 0, 0.5]])
    print(a)
    b = shuffle_forward(a, [2, 2])
    print(b)
    c = b.reshape(-1, 4).T
    print(c)'''

    '''
    rho = np.array([[1, 3+2j], [3-2j, 2]])
    rho_t = torch.tensor(rho)
    V, W = torch.linalg.eigh(rho_t)
    print(torch.matmul(W, W.T.conj()))
    print(proj_spectrahedron(rho))'''

    '''
    rho_t = torch.tensor(rho)
    print(proj_spectrahedron_torch(rho_t, 'imag'))'''

    '''
    coeffs = torch.tensor([1, 3])
    operators = torch.tensor([[[1, 2j], [1, 1j]], [[2, 3], [2, 2]]])
    print(qmt_matrix_torch(coeffs, operators))'''

    #print(onehot(np.array([[3, 2], [1, 0]]), 4))

    x = torch.tensor(np.random.normal(size=2**4))
    print('---x', x)

    y = proj_to_sum_one(x, 1)
    print(y, sum(y))

    y = proj_to_sum_one(x, 2)
    print(y, sum(y))

    t1 = perf_counter()
    print('F---', eigenvalues_trans_F(x, 'cpu', 1), perf_counter() - t1)

    t1 = perf_counter()
    print('S1---', eigenvalues_trans_F(x, 'cpu', 2), perf_counter() - t1)
    t1 = perf_counter()
    print('S2---', eigenvalues_trans_S(x, 'cpu'), perf_counter() - t1)

    t1 = perf_counter()
    print('soft---', softmax(x, 0), sum(softmax(x, 0)), perf_counter() - t1)