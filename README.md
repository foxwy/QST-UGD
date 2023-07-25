# **Unifying the factored and projected gradient descent for quantum state tomography**

The official Pytorch implementation of the paper named [`Unifying the factored and projected gradient descent for quantum state tomography`](https://arxiv.org/abs/2207.05341), under review.

### **Abstract**

Reconstructing the state of many-body quantum systems is of fundamental importance in quantum information tasks, but extremely challenging due to the curse of dimensionality. In this work, we present an efficient quantum tomography approach that unifies the state factored and projected methods to tackle the rank-deficient issue and incorporates a momentum-accelerated Rprop gradient algorithm to speed up the optimization process. In particular, the techniques of state decomposition and P-order absolute projection are jointly introduced to ensure both the positivity and rank of state matrices learned in the maximum likelihood function. Further, the proposed state-mapping method can substantially improve the tomography accuracy of other QST algorithms. Finally, numerical experiments demonstrate that the unified strategy is able to tackle the rank-deficient problem and admit a faster convergence and excellent purity robustness. We find that our method can accomplish the task of full tomography of random 11-qubit mixed states within one minute.

## Getting started

This code was tested on the computer with a single Intel(R) Core(TM) i7-12700KF CPU @ 3.60GHz with 64GB RAM and a single NVIDIA GeForce RTX 3090 Ti GPU with 24.0GB RAM, and requires:

- Python 3.9
- conda3
- torch==2.0.1+cu118
- h5py==3.1.0
- matplotlib==3.5.2
- numpy==1.23.4
- openpyxl==3.0.10
- SciencePlots==2.0.1
- scipy==1.9.1tqdm==4.64.1

## Runs QST algorithms

```bash
python main.py
```

### 1. Initial Parameters (`main`)

```python
parser = argparse.ArgumentParser()
parser.add_argument("--POVM", type=str, default="Tetra4", help="type of POVM")
parser.add_argument("--K", type=int, default=4, help='number of operators in single-qubit POVM')

parser.add_argument("--na_state", type=str, default="real_random", help="name of state in library")
parser.add_argument("--P_state", type=float, default=0.6, help="P of mixed state")
parser.add_argument("--ty_state", type=str, default="mixed", help="type of state (pure, mixed)")
parser.add_argument("--n_qubits", type=int, default=8, help="number of qubits")

parser.add_argument("--noise", type=str, default="no_noise", help="have or have not sample noise (noise, no_noise, depolar_noise)")
parser.add_argument("--n_samples", type=int, default=1000000, help="number of samples")
parser.add_argument("--P_povm", type=float, default=1, help="possbility of sampling POVM operators")
parser.add_argument("--seed_povm", type=float, default=1.0, help="seed of sampling POVM operators")
parser.add_argument("--read_data", type=bool, default=False, help="read data from text in computer")

parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--lr", type=float, default=0.001, help="optim: learning rate")

parser.add_argument("--map_method", type=str, default="chol_h", help="map method for output vector to density matrix (chol, chol_h, proj_F, proj_S, proj_A)")
parser.add_argument("--P_proj", type=float, default="2", help="coefficient for proj method")
```

### 2. Run UGD algorithm (`Net_train`)

```python
print('\n'+'-'*20+'UGD'+'-'*20)
gen_net = UGD_nn(opt.n_qubits, P_idxs, M, 
                 map_method=opt.map_method, P_proj=opt.P_proj).to(torch.float32).to(device)

net = UGD(gen_net, data, opt.lr)
result_save = {'parser': opt,
               'time': [], 
               'epoch': [],
               'Fc': [],
               'Fq': []}
net.train(opt.n_epochs, fid, result_save)
result_saves['UGD'] = result_save
```

### 3. Run iMLE algorithm (`Net_train`)

```python
print('\n'+'-'*20+'iMLE'+'-'*20)
result_save = {'parser': opt,
               'time': [], 
               'epoch': [],
               'Fc': [],
               'Fq': []}
iMLE(M, opt.n_qubits, data_all, opt.n_epochs, fid, result_save, device)
result_saves['iMLE'] = result_save
```

### 4. Run CG-APG algorithm (`Net_train`)

```python
print('\n'+'-'*20+'QSE APG'+'-'*20)
result_save = {'parser': opt,
               'time': [], 
               'epoch': [],
               'Fc': [],
               'Fq': []}
qse_apg(M, opt.n_qubits, data_all, opt.n_epochs, fid, 'chol_h', 2, result_save, device)
result_saves['APG'] = result_save
```

### 5. Run LRE algorithm (`Net_train`)

```python
print('\n'+'-'*20+'LRE'+'-'*20)
result_save = {'parser': opt,
               'time': [],
               'Fc': [],
               'Fq': []}
LRE(M, opt.n_qubits, data_all, fid, 'proj_F', 1, result_save, device)
result_saves['LRE'] = result_save
```

### 6. Run LRE algorithm with ProjA_1 (`Net_train`)

```python
print('\n'+'-'*20+'LRE proj'+'-'*20)
result_save = {'parser': opt,
               'time': [],
               'Fc': [],
               'Fq': []}
LRE(M, opt.n_qubits, data_all, fid, 'proj_A', 1, result_save, device)
result_saves['LRE_projA'] = result_save
```

#### **Acknowledgments**

This code is standing on the shoulders of giants. We want to thank the following contributors that our code is based on: [POVM_GENMODEL](https://github.com/carrasqu/POVM_GENMODEL), [qMLE](https://github.com/qMLE/qMLE).

## **License**

This code is distributed under an [Mozilla Public License Version 2.0](LICENSE).

Note that our code depends on other libraries, including POVM_GENMODEL, qMLE, and uses algorithms that each have their own respective licenses that must also be followed.
