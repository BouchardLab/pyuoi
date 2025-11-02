#!/usr/bin/env python3

'''

IMG=nersc/causal-net:v4   # May 13
export OMP_NUM_THREADS=2
salloc -q interactive -C cpu --image=$IMG -t 4:00:00 -A m2043 -N 4

time srun -n 128 --distribution=block:block shifter python  uoi_var_poisson_fdr_test.py  --samples 10_000

Existing samples @ /pscratch/sd/y/yxu2/data:
 --dataName  daleM20_746c4b      4 min 100k samp @ N=4
 --dataName  daleM40_e33e89      9 min  100k samp @ N=4
 --dataName  daleM80_285c84      75 min  300k samp @ N=4
 --dataName  daleM150_448b86    62 min  300k samp @ N=4

Existing samples @ dataPath=/pscratch/sd/b/balewski/2025_causalNet_tmp/
 --dataName  daleM300_aa61f5
# hard case
--dataName daleM130_6eb245  

# Yao:  
n_process should be multiple of n_admm, and at MOST n_admm*n_boot*n_reg_param
 Example run command on NERSC interactive compute node session:
 srun -n 768 --ntasks-per-node=192 --distribution=block:block python uoi_var_poisson_test.py


UoI_Poisson parameters:
  dt = 0.01 - sample time bin size for Poisson process
  n_admm = 32 - number of ADMM processes
  n_lambdas = 4 - number of L1 penalty values to test
  manual_l1_range = [6e-7, 3e-6] - Hardcoded L1 penalty range
  ??? imbalance_tolerance = 10 - tolerance for load imbalance 
  n_boots_sel = 6 - number of bootstrap samples for selection
  n_boots_est = 6 - number of bootstrap samples for estimation
  max_iter = 1000 - maximum number of iterations
  selection_frac = 0.9 - Fraction of total data used for each bootstrap
  lag = 1 - VAR lag order, must be 1
  rho_scaler = 1.0 - ADMM rho scaling factor
  l1_suppression = 0 - L1 penalty suppression factor
  seed = 22 - random state seed

'''

import pdb, h5py, os,sys
import numpy as np
import scipy.sparse as sparse
from numpy.linalg import norm
import importlib
import argparse
import secrets

from mpi4py import MPI
from time import time

sys.path.append("/global/homes/b/balewski/prjs/2025_UoI-VAR/")
from examples.var_utils import *
from src.pyuoi.linear_model import *
sys.path.append("/global/homes/b/balewski/prjs/2025_UoI-VAR/src/pyuoi/linear_model")
from sparse_comm_util import build_bootstrap_comm
from var_utils import *

#...!...!..................
def qa_Bfit(B_truth, B_fit):
    assert B_truth.shape == B_fit.shape
    bterm_res = B_fit - B_truth
    bterm_mean = np.mean(bterm_res)
    bterm_std = np.std(bterm_res)
    N = bterm_res.shape[0]
    bterm_se_s = 0.
    if N > 1:
        bterm_se_s = bterm_std / np.sqrt(2 * (N - 1))
    print('B term: mean=%.2f    std=%.3f +/- %.3f' % (bterm_mean, bterm_std, bterm_se_s))

    out = {
        'tval': B_truth,
        'fval': B_fit,
        'res_mean': bterm_mean,
        'res_std': bterm_std
    }
    return out
    
#...!...!..................
def qa_Afit(A_truth, A_fit):
    assert A_truth.shape == A_fit.shape
    assert A_truth.shape[0] == A_truth.shape[1]
    
    M = A_truth.shape[0]
    diag_m = np.eye(M, dtype=bool)
    out = {}
    
    # --- Diagonal elements
    diag_tval = A_truth[diag_m]
    diag_fval = A_fit[diag_m]
    res_diag = diag_fval - diag_tval
    diag_mean = np.mean(res_diag) if res_diag.size > 0 else 0
    diag_std = np.std(res_diag) if res_diag.size > 0 else 0
    out['diag'] = {
        'tval': diag_tval,
        'fval': diag_fval,
        'res_mean': diag_mean,
        'res_std': diag_std
    }
    
    # --- Off-diagonal elements (Excitatory and Inhibitory)
    for name in ['exc', 'inh']:
        if name == 'exc':
            true_m = (A_truth > 0) & (~diag_m)
            pred_m = (A_fit > 0) & (~diag_m)
        else:  # inh
            true_m = (A_truth < 0) & (~diag_m)
            pred_m = (A_fit < 0) & (~diag_m)

        tp_m = true_m & pred_m
        tval = A_truth[tp_m]
        fval = A_fit[tp_m]
        res = fval - tval
        res_mean = np.mean(res) if res.size > 0 else 0
        res_std = np.std(res) if res.size > 0 else 0
        
        TP = np.sum(tp_m)
        FP = np.sum((~true_m) & pred_m)
        FN = np.sum(true_m & (~pred_m))
        
        out[name] = {
            'tval': tval,
            'fval': fval,
            'res_mean': res_mean,
            'res_std': res_std,
            'TP': TP,
            'FP': FP,
            'FN': FN
        }

    # Keep printing for compatibility
    for name, stats in [('exc', out['exc']), ('inh', out['inh'])]:
        res_N = stats['tval'].shape[0]
        se_s = 0.
        if res_N > 1:
            se_s = stats['res_std'] / np.sqrt(2 * (res_N - 1))
        
        print('A type=%s  TP:%d  FP:%d FN:%d  TP: mean=%.2f   std=%.3f +/- %.3f'%(name, stats['TP'], stats['FP'], stats['FN'], stats['res_mean'], stats['res_std'], se_s))
    
    diag_stats = out['diag']
    diag_N = diag_stats['tval'].shape[0]
    diag_se_s = 0.
    if diag_N > 1:
        diag_se_s = diag_stats['res_std'] / np.sqrt(2 * (diag_N - 1))
    
    print('A diag: mean=%.2f    std=%.3f +/- %.3f'%(diag_stats['res_mean'], diag_stats['res_std'], diag_se_s))
    
    return out

#...!...!..................
def generate_edge_plots(args, qaD, outName):
    #import matplotlib as mpl
    #mpl.use('TkAgg')   
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.))

    colors = {'exc': 'red', 'inh': 'blue', 'diag': 'brown', 'bterm': 'green'}

    for i, name in enumerate([ 'inh','exc', 'diag', 'bterm']):
        ax = axes[i]
        stats = qaD[name]
        color = colors[name]
        
        if stats['tval'].size > 0:
            ax.scatter(stats['tval'], stats['fval'], alpha=0.5, s=8, c=color)
            # Add y=x line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
            ax.set_aspect('equal', 'box')
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            # Add center of gravity cross
            mean_t = np.mean(stats['tval'])
            mean_f = np.mean(stats['fval'])
            ax.plot(mean_t, mean_f, '+', c='black', markersize=20, markeredgewidth=3)

        ax.grid()
        ax.set_title(name.capitalize())
        ax.set_xlabel('True Value')
        if i == 0:
            ax.set_ylabel('Fitted Value')

        # Add mean and std text
        mean_val = stats['res_mean']
        std_val = stats['res_std']
        ax.text(0.95, 0.05, f'N={stats["tval"].shape[0]}  \nmean={mean_val:.3f}\nstd={std_val:.3f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

        if name in ['exc', 'inh']:
            tp_val = stats['TP']
            fp_val = stats['FP']
            fn_val = stats['FN']
            ax.text(0.05, 0.95, f'TP={tp_val}\nFP={fp_val}\nFN={fn_val}',
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

    fig.suptitle(f'Fitted  {outName}  %d samples'%(args.samples))
    fig.tight_layout()
    plotF = os.path.join(args.out, f'{outName}_corr.png')
    fig.savefig(plotF)
    print(f'Saved   display  {plotF}')
    plt.show()

#...!...!..................
def generate_eigen_plot(args, A_truth, A_fit,outName):
    import matplotlib.pyplot as plt
    eigT = np.linalg.eigvals(A_truth)
    eigF = np.linalg.eigvals(A_fit)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    reT = np.real(eigT)
    imT = np.imag(eigT)
    ax.scatter(reT, imT, color='blue', marker='o',label='True',s=10)

    reF = np.real(eigF)
    imF = np.imag(eigF)
    ax.scatter(reF, imF, color='red', marker='o', facecolors='none',label='fit',s=20)
    
    ax.set_ylim(-0.1,)
    #ax.set_xlim(right=1)
    ax.axhline(0, linestyle='--', color='k', linewidth=1)
    ax.axvline(0, linestyle='--', color='k', linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.legend()
    ax.set_title(f'Eigenvalues for {outName}')
    
    plotF = os.path.join(args.out, f'{outName}_eigen.png')
    fig.savefig(plotF)
    print(f'Saved   display  {plotF}')
    plt.show()

#...!...!..................
def main():
    parser = argparse.ArgumentParser(description='UoI-VAR Poisson ADMM test')
    parser.add_argument('--dataPath', default='/pscratch/sd/y/yxu2/data', help='path to data directory')
    parser.add_argument('--dataName', default='daleM20_746c4b', help='dataset name (e.g., daleM20_746c4b)')
    parser.add_argument('--out', default='result', help='output directory')
    parser.add_argument('--samples', type=int, default=100000, help='number of data samples to use')
    parser.add_argument('--outName', default=None, help='output file name core')
    parser.add_argument('--freqWeight', action='store_true', help='use frequency dependent weights, default is False')
    args = parser.parse_args()
    
    outName = args.outName
    if outName is None:
        hash_str = secrets.token_hex(3)
        outName = f'{args.dataName}_uoi{hash_str}'

    # most important hyperparameters!
    confUoI = {
        'fit_VAR': True,
        'fit_intercept': False,
        'standardize': False,
        'manual_l1_range': [6e-7, 3e-6],
        'n_boots_sel': 6,
        'n_boots_est': 6,
        'selection_frac': 0.9,
        'n_lambdas': 4,
        'max_iter': 1000,
        'random_state': 22,
        'rho_scaler': 1.0,
        'n_admm': 32,
        'imbalance_tolerance': 10,
        'l1_suppression': 0,
        'solver': 'admm',
        'estimation_solver': "lbfgs",
        'dt': 0.01
    }

    lag=1
    assert lag==1

    rank = 0
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    if comm is not None:
        rank = comm.rank

    if rank == 0: 
        print('Start dataName=%s  samples=%d'%(args.dataName,args.samples))
        spikeF=os.path.join(args.dataPath,f'{args.dataName}.spikes.npz')
        data = np.load(spikeF)['spikes'][:args.samples].astype(np.double)
        data_pois = None

        if args.freqWeight: # enable freq-weighings
            rates=np.mean(data,axis = 0)/confUoI['dt']
            rates = np.clip(rates, 0.1, 50)
            w=1/rates
            w/=np.sum(w)
            w*=data.shape[1]
            print('rates:',rates)
            print('w',w)
        else:
            w = np.ones(data.shape[1])
    else:
        w = None

    w = comm.bcast(w, root=0)
    if rank == 0: print('%dk samples loaded to all %d  ranks'%(args.samples//1000,world_size ),flush=True)

    #fitting with multiple processes
    boot_comm = build_bootstrap_comm(comm, confUoI['n_admm'])
    uoi_poisson = UoI_Poisson(**confUoI, comm=boot_comm, global_comm=comm, weights=w)
    assert uoi_poisson.solver == "admm"
    
    start = time()
    if boot_comm is not None:  #if the global_rank is part of the boostrap distribution(not admm distribution)                
        if boot_comm.rank == 0:
            uoi_poisson.fit(lag, data = data, data_pois = data_pois)
        else:
            uoi_poisson.fit(lag)
    else:                
        uoi_poisson.admm_queue()
    end = time()

    if rank > 0: return
    
    print("rank " +str(comm.rank)+": Fitting complete in %d  seconds."%(end - start), flush = True)

    A_fit = uoi_poisson.VAR_coef_[0]
    B_fit = uoi_poisson.VAR_bias_
    
    outF = os.path.join(args.out, f'{outName}.npz')
    np.savez(outF, A_uoi=A_fit, B_uoi=B_fit)
    print('saved output to:',outF)

    #....  evaluation of results
    truthF=spikeF.replace('.spikes','.simTruth')
    A_truth = np.load(truthF)["A_true"]
    B_truth = np.load(truthF)["B_true"]

    TP, FP, TN, FN = matrix_comparison(A_truth, A_fit, threshold=0)
    # these two adds up == real sparsity in B_truth
    M=A_truth.shape[0]; M2=M*M
    print('dataName=%s  M=%d  M^2=%d'%(args.dataName,M,M2))
    print("TP  p=%.3e  n=%d "%(TP,TP*M2))
    print("FN  p=%.3e  n=%d "%(FN,FN*M2))               
    print("FP  p=%.3e  n=%d "%(FP,FP*M2))
    print("TN: ", TN)
    
    # estimation error
    est_mask = A_truth != 0
    print(" solution sparsity: ", np.count_nonzero(A_fit)/M2)

    print('detailed QA  %s  M=%d  M^2=%d  samples=%d/k' %(args.dataName,M,M2,args.samples/1000))
    qaD=qa_Afit(A_truth, A_fit)
    qaD['bterm']=qa_Bfit(B_truth, B_fit)


    generate_edge_plots(args, qaD, outName)
    generate_eigen_plot(args, A_truth, A_fit,outName)
    
 
#...!...!..................
if __name__ == "__main__":
    main()
        














    
    


