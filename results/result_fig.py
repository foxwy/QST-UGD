# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import scipy.io as scio
import openpyxl
import scienceplots
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

import sys
sys.path.append(".")

plt.style.use(['science', 'no-latex'])
plt.rcParams["font.family"] = 'Arial'

font_size = 28  # 34, 28, 38
font = {'size': font_size, 'weight': 'normal'}

colors = ['#ef4343', '#7a1b6d', '#ad66d5', '#e9c469', '#16048a', '#73b8d5', '#6b705c', '#e16849', '#299c8e', '#cc997e']
markers = ['s', 's', 's', 'o', 'o', 'd', 'd', '^', '^', '^']
lines = ['-', '-', '-', '--', '--', ':', ':', '-.', '-.', '-.']


def Plt_set(ax, xlabel, ylabel, savepath, log_flag=0, loc=4, ncol=1, f_size=18.5, xlabel_f=True, ylabel_f=True, lf=True):  # 27
    ax.tick_params(labelsize=font_size, length=4, width=3)
    ax.xaxis.set_tick_params(pad=8)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    font2 = {'size': f_size, 'weight': 'normal'}
    if lf:
        ax.legend(prop=font2, loc=loc, frameon=True, ncol=ncol, framealpha=0.75, borderpad=0.1, columnspacing=0.2)
        '''
        if "samples" in xlabel:
            ax.legend(bbox_to_anchor=((-0.2, 1.03)), prop=font2, loc=loc, frameon=False, ncol=10, framealpha=0.75, borderpad=0.1, columnspacing=0.5)
        else:
            ax.legend(bbox_to_anchor=((3.515, 1.03)), prop=font2, loc=loc, frameon=False, ncol=10, framealpha=0.75, borderpad=0.1, columnspacing=0.5)'''

    if xlabel_f:
        ax.set_xlabel(xlabel, font)  # fontweight='bold'
    if ylabel_f:
        ax.set_ylabel(ylabel, font)

    if log_flag == 1:
        ax.set_xscale('log')
    if log_flag == 2:
        ax.set_yscale('log')
    if log_flag == 3:
        ax.set_xscale('log')
        ax.set_yscale('log')

    if 'Time' in ylabel:
        ax.set_yticks([0.01, 0.1, 1, 10, 100])
    '''
    if 'Time' in xlabel:
        ax.set_xticks([0.1, 1, 10, 100])'''

    if 'samples' in xlabel:
        ax.set_xticks([7, 8, 9, 10, 11])
        ax.set_xticklabels(['$10^7$', '$10^8$', '$10^9$', '$10^{10}$', '$10^{11}$'])
        #ax.set_xticks([5, 6, 7, 8, 9, 11])
        #ax.set_xticklabels(['$10^5$', '$10^6$', '$10^7$', '$10^8$', '$10^{9}$', '$10^{11}$'])

    if 'samples' in ylabel:
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        tx = ax.yaxis.get_offset_text()
        tx.set_fontsize(24)

    if "Rank" in xlabel:
        ax.set_xticklabels(['$1$', '$2^1$', '$2^2$', '$2^3$', '$2^4$', '$2^5$', '$2^6$', '$2^7$', '$2^8$'])

    ax.grid(linewidth=0.5, linestyle=(0, (3, 10, 3, 10)))
    #ax.set_xticks([1, 10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000])
    #ax.set_xticks([100, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000])
    ax.set_yticks(1/10**np.arange(0, 5))

    plt.savefig(savepath + '.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)


def Plt_set2(ax, ax2, xlabel, ylabel1, ylabel2, savepath, log_flag=0, loc=4, ncol=1, f_size=20):
    ax.tick_params(labelsize=font_size, length=4, width=3)
    ax2.tick_params(labelsize=font_size, length=4, width=3)
    ax.xaxis.set_tick_params(pad=8)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    font2 = {'size': 20, 'weight': 'normal'}
    ax.legend(bbox_to_anchor=(-0.4, 1.02), prop=font2, loc=3, frameon=False, ncol=1)
    ax2.legend(bbox_to_anchor=(0.4, 1.02), prop=font2, loc=3, frameon=False, ncol=1)
    #ax.legend(bbox_to_anchor=(1.18, 0.95), prop=font2, loc=2, frameon=False, ncol=1)
    #ax2.legend(bbox_to_anchor=(1.18, 0.05), prop=font2, loc=3, frameon=False, ncol=1)
    #ax.legend(prop=font2, loc=3, frameon=False, ncol=1)
    #ax2.legend(prop=font2, loc=4, frameon=False, ncol=1)
    ax.set_xlabel(xlabel, font)  # fontweight='bold'
    ax.set_ylabel(ylabel1, font)
    ax2.set_ylabel(ylabel2, font)

    if log_flag == 1:
        ax.set_xscale('log')
    if log_flag == 2:
        ax.set_yscale('log')
        ax2.set_yscale('log')
    if log_flag == 3:
        ax.set_xscale('log')
        ax.set_yscale('log')

    if 'time' in ylabel1:
        ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])

    if 'iterations' in ylabel2:
        ax2.set_yticks([1, 10, 100, 1000, 10000])
        #ax2.set_yticks([600, 700, 800, 900, 1000])

    if 'samples' in xlabel:
        ax.set_xticks([7, 8, 9, 10, 11])
        ax.set_xticklabels(['$10^7$', '$10^8$', '$10^9$', '$10^{10}$', '$10^{11}$'])

    ax.grid(linewidth=0.5, linestyle=(0, (3, 10, 3, 10)))

    plt.savefig(savepath + '.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)


def Plt_set3(ax, xlabel, ylabel, savepath, log_flag=0, loc=4, ncol=1, f_size=23, xlabel_f=True, ylabel_f=True, lf=True):
    ax.tick_params(labelsize=28, length=4, width=3)
    ax.xaxis.set_tick_params(pad=5)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    font2 = {'size': f_size, 'weight': 'normal'}
    if lf:
        ax.legend(bbox_to_anchor=((-0.2, 1.18)), prop=font2, loc=loc, frameon=False, ncol=ncol, framealpha=0.75, borderpad=0.1, columnspacing=0.5)
    if xlabel_f:
        ax.set_xlabel(xlabel, font)  # fontweight='bold'
    if ylabel_f:
        ax.set_ylabel(ylabel, font)

    if log_flag == 1:
        ax.set_xscale('log')
    if log_flag == 2:
        ax.set_yscale('log')
    if log_flag == 3:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.grid(linewidth=0.5, linestyle=(0, (3, 10, 3, 10)))

    #plt.savefig(savepath + '.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)


'''
def Get_r(V):
    V_avg = np.mean(V, 0)
    V_std = np.std(V, 0) / 2
    r1 = list(map(lambda x: x[0] - x[1], zip(V_avg, V_std)))
    r2 = list(map(lambda x: x[0] + x[1], zip(V_avg, V_std)))

    return r1, r2'''


def Get_r(V):
    r1 = np.percentile(V, 75, axis=0)
    r2 = np.percentile(V, 25, axis=0)

    return r1, r2


#-----ex: 1 (Random State Convergence Experiments for Different Mapping Methods and Fixed Samples)-----
def Map_conv_fig(na_state, r_path):
    Algo = 'UGD'
    m_methods = ['fac_h_1', 'fac_t_1', 'fac_a_1', 'proj_S_0', 'proj_M_0', 'proj_S_1', 'proj_M_1', 'proj_A_1', 'proj_A_3', 'proj_A_4']
    labels = ['Fac$\\rm _H$', 'Fac$\\rm _T$', 'Fac$\\rm _A$', \
              '$\mathcal{S}$(proj)', '$\mathcal{M}$(proj)', '$\mathcal{S}$', '$\mathcal{M}$', \
              '$\mathcal{A}_1$', '$\mathcal{A}_3$', '$\mathcal{A}_4$']

    #-----sample, Fq-----
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))
    N_s = 10
    for i in range(len(m_methods)):
        print(Algo, m_methods[i])

        savePath = r_path + Algo + '_' + m_methods[i] + '_' + str(N_s)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        Fq = []
        iterations = [1, 10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000]
        for p in results:
            Fq_t = list(np.minimum(torch.tensor(results[p][Algo]['Fq']).numpy(), 1))
            epoch = list(results[p][Algo]['epoch'])

            Fq_s = []
            for it in iterations:
                if it not in epoch:
                    Fq_s.append(np.maximum(1 - Fq_t[-1], 1e-4))
                else:
                    idx = epoch.index(it)
                    Fq_s.append(np.maximum(1 - Fq_t[idx], 1e-4))
            Fq.append(Fq_s)

        Fq = np.array(Fq)
        Fq_avg = np.median(Fq, 0)

        r1, r2 = Get_r(Fq)
        r1 = np.maximum(r1, 0)
        r2 = np.minimum(r2, 1)

        ax.plot(iterations, Fq_avg, linewidth=3, label=labels[i], color=colors[i], marker=markers[i], linestyle=lines[i], markersize=8)
        ax.fill_between(iterations, r1, r2, alpha=0.1, color=colors[i])

    ax.set_xlim(0.9, 1000+120)
    ax.set_ylim(8.5 * 1e-4, 1.2)
    ax.text(0.018, 0.5, "MLE-MRprop", transform=ax.transAxes, size=23)
    Plt_set(ax, "Number of iterations", "Infidelity", 'fig/'+na_state+'/map/'+Algo+'_'+na_state+'_epoch_fq_'+str(N_s), 3, ncol=2, loc=3)
    plt.show()


#-----ex: 2 (Random State Convergence Experiments for Different Mapping Methods and Samples)-----
def Map_ex_fig_sample(ax, na_state, r_path, Algo, m_idxs=range(10), xlabel_f=True, ylabel_f=True, lf=True):
    N_Samples = np.array([7, 8, 9, 10, 11])
    m_methods = ['fac_h_1', 'fac_t_1', 'fac_a_1', 'proj_S_0', 'proj_M_0', 'proj_S_1', 'proj_M_1', 'proj_A_1', 'proj_A_3', 'proj_A_4']
    labels = ['Fac$\\rm _H$', 'Fac$\\rm _T$', 'Fac$\\rm _A$', \
              '$\mathcal{S}$(proj)', '$\mathcal{M}$(proj)', '$\mathcal{S}$', '$\mathcal{M}$', \
              '$\mathcal{A}_1$', '$\mathcal{A}_3$', '$\mathcal{A}_4$']

    #-----sample, Fq-----
    for i in m_idxs:
        print(Algo, m_methods[i])

        Fq_m = []
        for N_s in N_Samples:
            savePath = r_path + Algo + '_' + m_methods[i] + '_' + str(N_s)

            results = np.load(savePath + '.npy', allow_pickle=True).item()
            Fq = []
            for p in results:
                Fq_t = np.minimum(torch.tensor(results[p][Algo]['Fq']).numpy(), 0.9999)
                Fq.append(Fq_t[-1])

            Fq_m.append(Fq)

        Fq_m = 1 - np.array(Fq_m).T

        Fq_avg = np.median(Fq_m, 0)
        r1, r2 = Get_r(Fq_m)
        r1 = np.maximum(r1, 0)
        r2 = np.minimum(r2, 1)

        ax.plot(N_Samples, Fq_avg, linewidth=3, label=labels[i], color=colors[i], marker=markers[i], linestyle=lines[i], markersize=8)
        ax.fill_between(N_Samples, r1, r2, alpha=0.08, color=colors[i])

    ax.set_xlim(6.93, 11.07)
    ax.set_ylim(7 * 1e-5, 1.1)
    Plt_set(ax, "Number of samples", "Infidelity", 'fig/'+na_state+'/map/'+Algo+'_'+na_state+'_sample_fq', 2, ncol=2, loc=3, xlabel_f=xlabel_f, ylabel_f=ylabel_f, lf=lf)


def Map_ex_fig_samples(na_state, r_path):
    plt.figure(figsize=(24, 5.7))

    ax = plt.subplot(1, 3, 1)
    Algo = 'UGD'
    Map_ex_fig_sample(ax, na_state, r_path, Algo, xlabel_f=True, ylabel_f=True, lf=True)
    ax.text(-0.23, 0.95, "(a)", transform=ax.transAxes, size=30, font={'family': 'Times New Roman'})
    ax.text(0.06, 0.1, "MLE-MRprop", transform=ax.transAxes, size=26)

    ax = plt.subplot(1, 3, 2)
    Algo = 'MGD'
    Map_ex_fig_sample(ax, na_state, r_path, Algo, xlabel_f=True, ylabel_f=False, lf=False)
    ax.text(-0.23, 0.95, "(b)", transform=ax.transAxes, size=30, font={'family': 'Times New Roman'})
    ax.text(0.06, 0.1, "MLE-MGD", transform=ax.transAxes, size=26)

    ax = plt.subplot(1, 3, 3)
    Algo = 'LRE'
    Map_ex_fig_sample(ax, na_state, r_path, Algo, m_idxs=[0, 1, 5, 6, 7, 8, 9], xlabel_f=True, ylabel_f=False, lf=False)
    ax.text(-0.23, 0.95, "(c)", transform=ax.transAxes, size=30, font={'family': 'Times New Roman'})
    ax.text(0.06, 0.1, "LRE", transform=ax.transAxes, size=26)

    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=0.25, hspace=0)
    file_name = 'fig/'+na_state+'/map/'+na_state+'_sample_fq_'+str(10)
    plt.savefig(file_name + '.pdf', bbox_inches='tight', pad_inches=0.05)


def Map_ex_fig_purity(ax, na_state, r_path, Algo, m_idxs=range(10), xlabel_f=True, ylabel_f=True, lf=True):
    N_Samples = np.array([7, 8, 9, 10, 11])
    m_methods = ['fac_h_1', 'fac_t_1', 'fac_a_1', 'proj_S_0', 'proj_M_0', 'proj_S_1', 'proj_M_1', 'proj_A_1', 'proj_A_3', 'proj_A_4']
    labels = ['Fac$\\rm _H$', 'Fac$\\rm _T$', 'Fac$\\rm _A$', \
              '$\mathcal{S}$(proj)', '$\mathcal{M}$(proj)', '$\mathcal{S}$', '$\mathcal{M}$', \
              '$\mathcal{A}_1$', '$\mathcal{A}_3$', '$\mathcal{A}_4$']

    #-----purity, Fq-----
    Ns = 10
    for i in m_idxs:
        print(Algo, m_methods[i])
        savePath = r_path + Algo + '_' + m_methods[i] + '_' + str(Ns)

        results = np.load(savePath + '.npy', allow_pickle=True).item()

        Fq_NN = []
        P_NN = []
        pi = 0
        for p in results:
            P_NN.append(float(p))
            pi += 1
            if pi >= 11:
                break
        P_NN.sort()
        for p in P_NN:
            p = str(p)
            Fq_t = np.minimum(torch.tensor(results[p][Algo]['Fq']).numpy(), 1)
            Fq_NN.append(Fq_t[-1])

        ax.plot(np.array(P_NN)**2*(1-1/2**8)+1/2**8, np.maximum(1 - np.array(Fq_NN), 1e-4), linewidth=3, label=labels[i], color=colors[i], marker=markers[i], linestyle=lines[i], markersize=8)

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(7 * 1e-5, 1.2)

    Plt_set(ax, "Purity", "Infidelity", 'fig/'+na_state+'/map/'+Algo+'_'+na_state+'_purity_fq_'+str(Ns), 2, ncol=3, loc=4, xlabel_f=xlabel_f, ylabel_f=ylabel_f, lf=lf)


def Map_ex_fig_puritys(na_state, r_path):
    plt.figure(figsize=(24, 5.7))

    ax = plt.subplot(1, 3, 1)
    Algo = 'UGD'
    Map_ex_fig_purity(ax, na_state, r_path, Algo, xlabel_f=True, ylabel_f=True, lf=True)
    ax.text(-0.23, 0.95, "(a)", transform=ax.transAxes, size=30, font={'family': 'Times New Roman'})
    ax.text(0.06, 0.1, "MLE-MRprop", transform=ax.transAxes, size=26)

    ax = plt.subplot(1, 3, 2)
    Algo = 'MGD'
    Map_ex_fig_purity(ax, na_state, r_path, Algo, xlabel_f=True, ylabel_f=False, lf=False)
    ax.text(-0.23, 0.95, "(b)", transform=ax.transAxes, size=30, font={'family': 'Times New Roman'})
    ax.text(0.06, 0.1, "MLE-MGD", transform=ax.transAxes, size=26)

    ax = plt.subplot(1, 3, 3)
    Algo = 'LRE'
    Map_ex_fig_purity(ax, na_state, r_path, Algo, m_idxs=[0, 1, 5, 6, 7, 8, 9], xlabel_f=True, ylabel_f=False, lf=False)
    ax.text(-0.23, 0.95, "(c)", transform=ax.transAxes, size=30, font={'family': 'Times New Roman'})
    ax.text(0.06, 0.1, "LRE", transform=ax.transAxes, size=26)

    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=0.25, hspace=0)
    file_name = 'fig/'+na_state+'/map/'+na_state+'_purity_fq_'+str(10)
    plt.savefig(file_name + '.pdf', bbox_inches='tight', pad_inches=0.05)


def Map_ex_fig_rank(ax, na_state, r_path, Algo, m_idxs=range(10), xlabel_f=True, ylabel_f=True, lf=True):
    m_methods = ['fac_h_1', 'fac_t_1', 'fac_a_1', 'proj_S_0', 'proj_M_0', 'proj_S_1', 'proj_M_1', 'proj_A_1', 'proj_A_3', 'proj_A_4']
    labels = ['Fac$\\rm _H$', 'Fac$\\rm _T$', 'Fac$\\rm _A$', \
              '$\mathcal{S}$(proj)', '$\mathcal{M}$(proj)', '$\mathcal{S}$', '$\mathcal{M}$', \
              '$\mathcal{A}_1$', '$\mathcal{A}_3$', '$\mathcal{A}_4$']

    #-----rank, Fq-----
    Ns = 10
    for i in m_idxs:
        print(Algo, m_methods[i])
        savePath = r_path + Algo + '_' + m_methods[i] + '_' + str(Ns)

        results = np.load(savePath + '.npy', allow_pickle=True).item()

        Fq_NN = []
        P_NN = []
        pi = 0
        for p in results:
            pi += 1
            if pi > 11:
                P_NN.append(int(p))
        P_NN.sort()
        for p in P_NN:
            p = str(p)
            Fq_t = np.minimum(torch.tensor(results[p][Algo]['Fq']).numpy(), 1)
            Fq_NN.append(Fq_t[-1])

        ax.plot(np.log2(np.array(P_NN)), np.maximum(1 - np.array(Fq_NN), 1e-4), linewidth=3, label=labels[i], color=colors[i], marker=markers[i], linestyle=lines[i], markersize=8)

    ax.set_xticks(np.arange(9))
    ax.set_xlim(-0.2, 8.2)
    ax.set_ylim(7 * 1e-5, 1.2)

    Plt_set(ax, "Rank", "Infidelity", 'fig/'+na_state+'/map/'+Algo+'_'+na_state+'_rank_fq_'+str(Ns), 2, ncol=3, loc=4, xlabel_f=xlabel_f, ylabel_f=ylabel_f, lf=lf)


def Map_ex_fig_ranks(na_state, r_path):
    plt.figure(figsize=(24, 5.7))

    ax = plt.subplot(1, 3, 1)
    Algo = 'UGD'
    Map_ex_fig_rank(ax, na_state, r_path, Algo, xlabel_f=True, ylabel_f=True, lf=True)
    ax.text(-0.23, 0.95, "(a)", transform=ax.transAxes, size=30, font={'family': 'Times New Roman'})
    ax.text(0.575, 0.1, "MLE-MRprop", transform=ax.transAxes, size=26)

    ax = plt.subplot(1, 3, 2)
    Algo = 'MGD'
    Map_ex_fig_rank(ax, na_state, r_path, Algo, xlabel_f=True, ylabel_f=False, lf=False)
    ax.text(-0.23, 0.95, "(b)", transform=ax.transAxes, size=30, font={'family': 'Times New Roman'})
    ax.text(0.66, 0.1, "MLE-MGD", transform=ax.transAxes, size=26)

    ax = plt.subplot(1, 3, 3)
    Algo = 'LRE'
    Map_ex_fig_rank(ax, na_state, r_path, Algo, m_idxs=[0, 1, 5, 6, 7, 8, 9], xlabel_f=True, ylabel_f=False, lf=False)
    ax.text(-0.23, 0.95, "(c)", transform=ax.transAxes, size=30, font={'family': 'Times New Roman'})
    ax.text(0.85, 0.1, "LRE", transform=ax.transAxes, size=26)

    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=0.25, hspace=0)
    file_name = 'fig/'+na_state+'/map/'+na_state+'_rank_fq_'+str(10)
    plt.savefig(file_name + '.pdf', bbox_inches='tight', pad_inches=0.05)


#-----ex: 3 (Eigenvalues analysis of different mapping methods)-----
def Eigen_fig_pur(ax, na_state, r_path, P_s, xlabel_f=True, ylabel_f=True, lf=True, ylim1=0.08):
    Algo = 'UGD'
    colors_t = ['black', '#ef4343', '#7a1b6d', '#ad66d5', '#e9c469', '#16048a', '#73b8d5', '#6b705c', '#e16849', '#cc997e']
    markers_t = ['', 's', 's', 's', 'o', 'o', 'd', 'd', '^', '^']
    lines_t = ['-', '-', '-', '-', '--', '--', ':', ':', '-.', '-.']

    m_methods = ['actual', 'fac_h1', 'fac_t1', 'fac_a1', 'proj_S0', 'proj_M0', 'proj_S1', 'proj_M1', 'proj_A1', 'proj_A4']
    labels = ['Actual', 'Fac$\\rm _H$', 'Fac$\\rm _T$', 'Fac$\\rm _A$', \
              '$\mathcal{S}$(proj)', '$\mathcal{M}$(proj)', '$\mathcal{S}$', '$\mathcal{M}$', \
              '$\mathcal{A}_1$', '$\mathcal{A}_4$']

    #-----sample, Fq-----
    savePath = r_path + na_state + "_" + str(P_s)
    results = np.load(savePath + '.npy', allow_pickle=True).item()
    axins = ax.inset_axes((0.2, 0.23, 0.7, 0.7))

    for i in range(len(m_methods)):
        rho = results[m_methods[i]]
        eigenvalues = torch.linalg.eigvalsh(torch.tensor(rho))
        eigenvalues = np.array(torch.maximum(eigenvalues.to('cpu'), torch.tensor(0)))
        eigenvalues = np.array(list(eigenvalues[:8]) + list(eigenvalues[-8:]))
        print(m_methods[i], min(eigenvalues))

        ax.plot(range(len(eigenvalues)), eigenvalues[::-1], linewidth=2, label=labels[i], color=colors_t[i], marker=markers_t[i], linestyle=lines_t[i], markersize=4)
        axins.plot(range(len(eigenvalues)), eigenvalues[::-1], linewidth=2, label=labels[i], color=colors_t[i], marker=markers_t[i], linestyle=lines_t[i], markersize=3)

    xlim0 = 0
    xlim1 = 15.1
    ylim0 = -0.0005
    #ylim1 = 0.008
    ax.plot([xlim0, xlim1, xlim1, xlim0, xlim0], [ylim0, ylim0, ylim1, ylim1, ylim0], '--', color='black', linewidth=1)
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)
    xy = (xlim0, ylim0)
    xy2 = (xlim0, ylim0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)

    xy = (xlim1, ylim1)
    xy2 = (xlim1, ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)
    axins.tick_params(labelsize=18, length=2, width=2)

    a = list(range(0, 16, 4))
    a.append(15)
    ax.set_xticks(a)
    axins.set_xticks(a)
    ax.set_xlim(0-0.3, len(eigenvalues)-1+0.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    axins.text(0.58, 0.8, "MLE-MRprop", transform=ax.transAxes, size=24)
    axins.text(0.58, 0.7, "Purity$\\approx$"+str(round(P_s, 3)), transform=ax.transAxes, size=24)

    Plt_set3(ax, "index", "Eigenvalues", 'fig/'+na_state+'/eigen/'+Algo+'_'+na_state+'_eigen'+str(P_s), 0, loc=2, ncol=10, xlabel_f=xlabel_f, ylabel_f=ylabel_f, lf=lf)

def Eigen_fig_rank(ax, na_state, r_path, P_s, xlabel_f=True, ylabel_f=True, lf=True, ylim1=0.08):
    Algo = 'UGD'
    colors_t = ['black', '#ef4343', '#7a1b6d', '#ad66d5', '#e9c469', '#16048a', '#73b8d5', '#6b705c', '#e16849', '#cc997e']
    markers_t = ['', 's', 's', 's', 'o', 'o', 'd', 'd', '^', '^']
    lines_t = ['-', '-', '-', '-', '--', '--', ':', ':', '-.', '-.']

    m_methods = ['actual', 'fac_h1', 'fac_t1', 'fac_a1', 'proj_S0', 'proj_M0', 'proj_S1', 'proj_M1', 'proj_A1', 'proj_A4']
    labels = ['Actual', 'Fac$\\rm _H$', 'Fac$\\rm _T$', 'Fac$\\rm _A$', \
              '$\mathcal{S}$(proj)', '$\mathcal{M}$(proj)', '$\mathcal{S}$', '$\mathcal{M}$', \
              '$\mathcal{A}_1$', '$\mathcal{A}_4$']

    #-----sample, Fq-----
    savePath = r_path + na_state + "_" + str(P_s)
    results = np.load(savePath + '.npy', allow_pickle=True).item()
    axins = ax.inset_axes((0.2, 0.23, 0.7, 0.7))

    for i in range(len(m_methods)):
        rho = results[m_methods[i]]
        eigenvalues = torch.linalg.eigvalsh(torch.tensor(rho))
        eigenvalues = np.array(torch.maximum(eigenvalues.to('cpu'), torch.tensor(0)))
        eigenvalues = np.array(list(eigenvalues[:8]) + list(eigenvalues[-8:]))
        print(m_methods[i], min(eigenvalues))

        ax.plot(range(len(eigenvalues)), eigenvalues[::-1], linewidth=2, label=labels[i], color=colors_t[i], marker=markers_t[i], linestyle=lines_t[i], markersize=4)
        axins.plot(range(len(eigenvalues)), eigenvalues[::-1], linewidth=2, label=labels[i], color=colors_t[i], marker=markers_t[i], linestyle=lines_t[i], markersize=3)

    xlim0 = 0
    xlim1 = 15.1
    ylim0 = -0.001
    #ylim1 = ylim1
    ax.plot([xlim0, xlim1, xlim1, xlim0, xlim0], [ylim0, ylim0, ylim1, ylim1, ylim0], '--', color='black', linewidth=1)
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)
    xy = (xlim0, ylim0)
    xy2 = (xlim0, ylim0)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)

    xy = (xlim1, ylim1)
    xy2 = (xlim1, ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)
    axins.tick_params(labelsize=18, length=2, width=2)

    a = list(range(0, 16, 4))
    a.append(15)
    ax.set_xticks(a)
    axins.set_xticks(a)
    ax.set_xlim(0-0.3, len(eigenvalues)-1+0.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    axins.text(0.58, 0.8, "MLE-MRprop", transform=ax.transAxes, size=24)
    axins.text(0.58, 0.7, "Rank$=$"+str(P_s), transform=ax.transAxes, size=24)

    Plt_set3(ax, "index", "Eigenvalues", 'fig/'+na_state+'/eigen/'+Algo+'_'+na_state+'_eigen'+str(P_s), 0, loc=2, ncol=10, xlabel_f=xlabel_f, ylabel_f=ylabel_f, lf=lf)


def Eigen_figs(na_state, r_path):
    plt.figure(figsize=(20, 12))

    ax = plt.subplot(2, 2, 1)
    P_s = 0.0039
    Eigen_fig_pur(ax, na_state, r_path, P_s, xlabel_f=False, ylabel_f=True, lf=True, ylim1=0.008)
    ax.text(-0.18, 0.95, "(a)", transform=ax.transAxes, size=30, font={'family': 'Times New Roman'})

    ax = plt.subplot(2, 2, 2)
    P_s = 0.5956
    Eigen_fig_pur(ax, na_state, r_path, P_s, xlabel_f=False, ylabel_f=False, lf=False, ylim1=0.003)
    ax.text(-0.18, 0.95, "(b)", transform=ax.transAxes, size=30, font={'family': 'Times New Roman'})

    ax = plt.subplot(2, 2, 3)
    P_s = 2**5
    Eigen_fig_rank(ax, na_state, r_path, P_s, xlabel_f=True, ylabel_f=True, lf=False, ylim1=0.06)
    ax.text(-0.18, 0.95, "(c)", transform=ax.transAxes, size=30, font={'family': 'Times New Roman'})

    ax = plt.subplot(2, 2, 4)
    P_s = 2**8
    Eigen_fig_rank(ax, na_state, r_path, P_s, xlabel_f=True, ylabel_f=False, lf=False, ylim1=0.02)
    ax.text(-0.18, 0.95, "(d)", transform=ax.transAxes, size=30, font={'family': 'Times New Roman'})

    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=0.2, hspace=0.15)
    file_name = 'fig/'+na_state+'/eigen/'+na_state+'_eigen_fq'
    plt.savefig(file_name + '.pdf', bbox_inches='tight', pad_inches=0.05)

#-----ex: 4 (Random State Convergence Experiments of Different QST Algorithms for Different samples)-----
def Conv_sample_ex_fig(na_state, r_path, Algo):
    N_Samples = np.array([7, 8, 9, 10, 11])
    m_methods = ['UGD', 'UGD_SGD', 'iMLE', 'APG', 'LRE', 'LRE_projA']
    labels = ['MLE-MRprop-Fac$\\rm _H$', 'MLE-MGD-Fac$\\rm _H$', 'iMLE', 'MLE-(CG-APG)-(Fac$\\rm _A$-$\mathcal{S}$)', 'LRE-$\mathcal{S}$', 'LRE-$\mathcal{A}_1$']
    #colors = ['#ef4343', '#7a1b6d', '#ad66d5', '#e9c469', '#16048a', '#73b8d5', '#6b705c', '#e16849', '#299c8e', '#cc997e']
    colors_t = ['#ef4343', '#299c8e', '#ad66d5', '#16048a', '#cc997e', '#e9c469']
    markers_t = ['s', 's', 'o', 'o', 'd', 'd']
    lines_t = ['-', '--', '--', '--', ':', ':']

    #-----sample, Fq-----
    '''
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))
    for i in range(len(m_methods)):
        print(m_methods[i])

        Fq_m = []
        for N_s in N_Samples:
            savePath = r_path + str(N_s) + 's_all_2'

            results = np.load(savePath + '.npy', allow_pickle=True).item()
            Fq = []
            for p in results:
                Fq_t = np.minimum(torch.tensor(results[p][m_methods[i]]['Fq']).numpy(), 1)
                Fq.append(Fq_t[-1])

            Fq_m.append(Fq)

        Fq_m = 1 - np.array(Fq_m).T

        Fq_avg = np.median(Fq_m, 0)
        r1, r2 = Get_r(Fq_m)
        r1 = np.maximum(r1, 0)
        r2 = np.minimum(r2, 1)

        ax.plot(N_Samples, Fq_avg, linewidth=3, label=labels[i], color=colors_t[i], marker=markers_t[i], linestyle=lines_t[i], markersize=8)
        ax.fill_between(N_Samples, r1, r2, alpha=0.1, color=colors[i])

    #ax.plot([N_Samples[0], N_Samples[-1]], [0.99, 0.99], 'k--', linewidth=3)
    #ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xlim(N_Samples[0]-0.07, N_Samples[-1]+0.07)
    ax.set_ylim(7 * 1e-5, 1)
    Plt_set(ax, "Number of samples", "Infidelity", 'fig/'+na_state+'/sample/'+na_state+'_sample_fq', 2, ncol=1, loc=3)'''


    #-----sample, time-----
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))
    labels = ['MLE-MRprop-Fac$\\rm _H$', 'MLE-MGD-Fac$\\rm _H$', 'iMLE', 'MLE-(CG-APG)-(Fac$\\rm _A$-$\mathcal{S}$)', 'LRE-$\mathcal{S}$', 'LRE-$\mathcal{A}_1$']
    labels_t = ['Time of ' + l for l in labels]
    labels_e = ['Iterations of ' + l for l in labels]
    ax2 = ax.twinx()
    for i in range(len(m_methods)):
        print(m_methods[i])

        Time_m = []
        Epoch_m = []
        for N_s in N_Samples:
            savePath = r_path + str(N_s) + 's_all'

            results = np.load(savePath + '.npy', allow_pickle=True).item()
            
            Time = []
            Epoch = []
            for p in results:
                Fq_t = np.minimum(torch.tensor(results[p][m_methods[i]]['Fq']).numpy(), 1)

                for j in range(len(Fq_t)):
                    if Fq_t[j] >= 0.99:
                        Time.append(results[p][m_methods[i]]['time'][j])
                        if m_methods[i] == 'LRE' or m_methods[i] == 'LRE_projA':
                            Epoch.append(1)
                        else:
                            epoch_j = results[p][m_methods[i]]['epoch'][j]
                            Epoch.append(epoch_j)
                        break

                    if j == len(Fq_t) - 1:
                        Time.append(results[p][m_methods[i]]['time'][j])
                        if m_methods[i] == 'LRE' or m_methods[i] == 'LRE_projA':
                            Epoch.append(1)
                        else:
                            epoch_j = results[p][m_methods[i]]['epoch'][j]
                            Epoch.append(epoch_j)

            Time_m.append(Time)
            Epoch_m.append(Epoch)

        Time_m = np.array(Time_m).T
        Time_avg = np.median(Time_m, 0)

        Epoch_m = np.array(Epoch_m).T
        Epoch_avg = np.median(Epoch_m, 0)

        ax.plot(N_Samples, Time_avg, linewidth=3, label=labels_t[i], color=colors_t[i], marker='o', markersize=8)
        ax2.plot(N_Samples, Epoch_avg, ':', linewidth=3, label=labels_e[i], color=colors_t[i], marker='d', markersize=8)

    ax.set_xlim(N_Samples[0]-0.07, N_Samples[-1]+0.07)
    ax.set_ylim(1e-2, 1e3)
    ax2.set_ylim(0.4, 13000)
    Plt_set2(ax, ax2, "Number of samples", "Runtime ($s$)", 'Number of iterations', 'fig/'+na_state+'/sample/'+na_state+'_sample_time', 2, ncol=2)
    plt.show()

#-----ex: 5 (Convergence Experiment of Random Mixed States for Different Qubits)-----
def Conv_qubit_ex_fig(na_state, r_path, Algo):
    m_methods = ['UGD', 'iMLE', 'APG', 'LRE', 'LRE_projA']
    colors_t = ['#ef4343', '#ad66d5', '#16048a', '#cc997e', '#e9c469']

    #-----sample, time-----
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))
    labels = ['MLE-MRprop-Fac$\\rm _H$', 'iMLE', 'MLE-(CG-APG)-(Fac$\\rm _A$-$\mathcal{S}$)', 'LRE-$\mathcal{S}$', 'LRE-$\mathcal{A}_1$']
    labels_t = ['Time of ' + l for l in labels]
    labels_e = ['Iterations of ' + l for l in labels]

    ax2 = ax.twinx()
    for i in range(len(m_methods)):
        print(m_methods[i])

        Time_m = []
        Epoch_m = []
        if m_methods[i] == 'iMLE':
            N_qubits = np.arange(2, 9)
        elif m_methods[i] == 'APG':
            N_qubits = np.arange(2, 11)
        else:
            N_qubits = np.arange(2, 12)

        for N_q in N_qubits:
            savePath = r_path + str(N_q)

            results = np.load(savePath + 'q_all.npy', allow_pickle=True).item()
            Time = []
            Epoch = []
            p_n = 0
            for p in results:
                p_n += 1
                if p_n > 20:
                    break

                Fq_t = np.minimum(torch.tensor(results[p][m_methods[i]]['Fq']).numpy(), 1)

                for j in range(len(Fq_t)):
                    if Fq_t[j] >= 0.99:
                        Time.append(results[p][m_methods[i]]['time'][j])
                        if 'LRE' in m_methods[i]:
                            Epoch.append(1)
                        else:
                            Epoch.append(results[p][m_methods[i]]['epoch'][j])
                        break

                    if j == len(Fq_t) - 1:
                        Time.append(results[p][m_methods[i]]['time'][j])
                        if 'LRE' in m_methods[i]:
                            Epoch.append(1)
                        else:
                            Epoch.append(results[p][m_methods[i]]['epoch'][j])

            Time_m.append(Time)
            Epoch_m.append(Epoch)

        Time_m = np.array(Time_m).T
        Time_avg = np.median(Time_m, 0)

        Epoch_m = np.array(Epoch_m).T
        Epoch_avg = np.median(Epoch_m, 0)

        ax.plot(N_qubits, Time_avg, linewidth=3, label=labels_t[i], color=colors_t[i], marker='o', markersize=8)
        ax2.plot(N_qubits, Epoch_avg, ':', linewidth=3, label=labels_e[i], color=colors_t[i], marker='d', markersize=8)
        #print(Time_avg, Epoch_avg)

    ax.set_xticks(np.arange(2, 12))
    ax.set_xlim(2-0.15, 11+0.15)
    ax.set_ylim(8 * 1e-4, 1e3)
    ax2.set_ylim(0.4, 13000)
    Plt_set2(ax, ax2, "Number of qubits", "Runtime ($s$)", 'Number of iterations', 'fig/'+na_state+'/qubit/'+na_state+'_qubit_time', log_flag=2, ncol=2)
    
#-----ex: 6 (Convergence Experiment of Random Mixed States for Different Samples)-----
def Conv_depolar_ex_fig(na_state, r_path, Alpha):
    N_Samples = np.array([6, 7, 9])
    m_methods = ['UGD', 'iMLE', 'APG', 'LRE', 'LRE_projA']
    labels = ['UGD', 'iMLE', 'CG_APG', 'LRE', 'LRE_$\\rm \mathcal{A}[\cdot]_1$']
    color = ['#ef4343', '#73b8d5', '#0d4c6d', '#e66f51', '#35b777']
    marker = ['s', 'o', 'o', 'd', 'd']
    line = ['-', '0.:', ':', '-.', '-.']
    #-----purity, Fq-----
    for N_s in N_Samples:
        print(N_s)
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        for i in range(len(m_methods)):
            savePath = r_path + str(N_s)

            results = np.load(savePath + '_depolar.npy', allow_pickle=True).item()

            Fq_NN = []
            P_NN = []
            for p in results:
                P_NN.append(float(p))
            P_NN.sort()
            for p in P_NN:
                p = str(p)
                Fq_t = np.minimum(torch.tensor(results[p][m_methods[i]]['Fq']).numpy(), 1)
                Fq_NN.append(Fq_t[-1])

            ax.plot(np.array(P_NN), Fq_NN, linewidth=4, label=labels[i], color=color[i], marker=marker[i], linestyle=line[i], markersize=8)

        ax.plot(np.array(P_NN), 1 - (1 - 1 / 2**10) * np.array(P_NN), 'k--', linewidth=4)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)

        Plt_set(ax, "$\lambda$", "Quantum fidelity", 'fig/'+na_state+'/depolar/'+na_state+'_'+str(N_s)+'_depolar_lambda_fq', 2, loc=1, ncol=1)
        #plt.show()


if __name__ == '__main__':
    na_state = 'real_random_pur_rank'

    print('-----state:', na_state)
    r_path = 'result/' + na_state + '/'
    #Map_ex_fig_puritys(na_state, r_path)
    #Map_ex_fig_ranks(na_state, r_path)
    #Map_ex_fig_samples(na_state, r_path)
    #Map_conv_fig(na_state, r_path)
    Eigen_figs(na_state, r_path)
    #Conv_sample_ex_fig(na_state, r_path, "UGD")