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
   seed = 22 - random state seed
  fdr_rate=0.05  - False Discovery Rate (FDR) , p-value for selection of existing edges
'''

import  os,sys
import numpy as np
import scipy.sparse as sparse
from numpy.linalg import norm
#import importlib
import argparse
import secrets

from mpi4py import MPI
from time import time
from pprint import pprint
sys.path.append("/global/homes/b/balewski/prjs/2025_UoI-VAR/")
from examples.var_utils import *
from src.pyuoi.linear_model import *
sys.path.append("/global/homes/b/balewski/prjs/2025_UoI-VAR/src/pyuoi/linear_model")
from sparse_comm_util import build_bootstrap_comm
from var_utils import *
from Util_poissonFdr import qa_Bfit, qa_Afit, plot_edges_correl, plot_eigenvalues, plot_loss, extract_loss_traces

from Util_NumpyIO import read_data_npz, write_data_npz

#...!...!..................
def main():
    parser = argparse.ArgumentParser(description='UoI-VAR Poisson ADMM test')
    parser.add_argument('--dataPath', default='/pscratch/sd/y/yxu2/data', help='path to data directory')
    parser.add_argument('--dataName', default='daleM20_746c4b', help='dataset name (e.g., daleM20_746c4b)')
    parser.add_argument('--out', default='result', help='output directory')
    parser.add_argument('--samples', type=int, default=100000, help='number of data samples to use')
    parser.add_argument('--outName', default=None, help='output file name core')
    parser.add_argument('--freqWeight', action='store_true', help='use frequency dependent weights, default is False')
    parser.add_argument('--maxIter', type=int, default=1000, help='maximum number of iterations for ADMM')
    parser.add_argument('--fdrRate', type=float, default=0.01, help='False discovery rate level for support selection')
    parser.add_argument('--selectonFrac', type=float, default=0.9, help='fraction of data used per bootstrap selection')
    parser.add_argument('--verb', '-v', type=int, default=1, help='Verbosity level')
    args = parser.parse_args()
    
    hash_str = None
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
        'n_lambdas': 4,
        'n_boots_sel': 6,
        'n_boots_est': 6,
        'selection_frac': args.selectonFrac,
        'max_iter': args.maxIter,
        'random_state': 22,
        'rho_scaler': 1.0,
        'n_admm': 32,
        'imbalance_tolerance': 10,
        'solver': 'admm',
        'estimation_solver': "lbfgs",
        'dt': 0.01,  # must be hardcoded - or use broadcasting to all ranks
        'loss_stride': 10,
        'fdr_rate': args.fdrRate,
    }

    confMisc = {
        'model_lag': 1,
        'use_freq_weight': args.freqWeight,
        'num_samples': args.samples,
        'data_name': args.dataName,
        'uoi_hash': hash_str,
        'short_name': outName,
        'data_path': args.dataPath,
        'out_path': args.out,
    }

    loss_stride = confUoI.get('loss_stride', 10)

    lag = confMisc['model_lag']
    assert lag==1

    rank = 0
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    if comm is not None:
        rank = comm.rank

    if rank == 0: 
        for arg in vars(args):
            print( 'myArgs:',arg, getattr(args, arg))
        if args.verb > 1:
            print('confUoI:'); pprint(confUoI)
            print('confMisc:'); pprint(confMisc)
            
        print('Start dataName=%s  samples=%d'%(confMisc['data_name'],confMisc['num_samples']))
        spikeF=os.path.join(confMisc['data_path'],f"{confMisc['data_name']}.spikes.npz")
        #data = np.load(spikeF)['spikes'][:confMisc['num_samples']].astype(np.double)
        spikeD, spikeMD = read_data_npz(spikeF, verb=True)
        data=spikeD['spikes'][:confMisc['num_samples']].astype(np.double)
        data_pois = None    
        pprint(spikeMD)
        #confUoI['dt']=spikeMD['time_step_sec']  # can't do it now w/o broadcasting to all ranks
        
        if confMisc['use_freq_weight']: # enable freq-weighings
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
    if rank == 0: print('%dk samples loaded to all %d  ranks'%(confMisc['num_samples']//1000,world_size ),flush=True)

    #fitting with multiple processes
    boot_comm = build_bootstrap_comm(comm, confUoI['n_admm'])
    confUoI_fit = confUoI.copy()
    confUoI_fit.pop('loss_stride', None)

    uoi_poisson = UoI_Poisson(**confUoI_fit, comm=boot_comm, global_comm=comm, weights=w)
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
    
    sel_loss_iter, sel_loss_l1 = extract_loss_traces(getattr(uoi_poisson, '_selection_lm', None), loss_stride)
    est_loss_iter, est_loss_l1 = extract_loss_traces(getattr(uoi_poisson, '_estimation_lm', None), loss_stride)

    l1_loss_sel = np.vstack((sel_loss_iter, sel_loss_l1)) if sel_loss_iter.size > 0 else np.empty((2, 0))
    l1_loss_est = np.vstack((est_loss_iter, est_loss_l1)) if est_loss_iter.size > 0 else np.empty((2, 0))

    
    bigD={ 'A_fit':A_fit, 'B_fit':B_fit, 'l1_loss_sel':l1_loss_sel, 'l1_loss_est':l1_loss_est}
    outF = os.path.join(confMisc['out_path'], f"{confMisc['short_name']}.uoiFdr.npz")
    write_data_npz(bigD, outF, metaD=spikeMD)
    print('saved output to:',outF)

    #....  evaluation of results
    truthF=spikeF.replace('.spikes','.simTruth')
    A_truth = np.load(truthF)["A_true"]
    B_truth = np.load(truthF)["B_true"]

    TP, FP, TN, FN = matrix_comparison(A_truth, A_fit, threshold=0)
    # these two adds up == real sparsity in B_truth
    M=A_truth.shape[0]; M2=M*M
    print('dataName=%s  M=%d  M^2=%d'%(confMisc['data_name'],M,M2))
    print("TP  p=%.3e  n=%d "%(TP,TP*M2))
    print("FN  p=%.3e  n=%d "%(FN,FN*M2))               
    print("FP  p=%.3e  n=%d "%(FP,FP*M2))
    print("TN: ", TN)
    
    
    print('detailed QA  %s  M=%d  M^2=%d  samples=%d/k' %(confMisc['data_name'],M,M2,confMisc['num_samples']/1000))
    qaD=qa_Afit(A_truth, A_fit)
    qaD['bterm']=qa_Bfit(B_truth, B_fit)

    plot_loss(args, A_truth, A_fit, confMisc['short_name'], l1_loss_sel, loss_skip=50)
    plot_edges_correl(args, qaD, confMisc['short_name'])
    plot_eigenvalues(args, A_truth, A_fit,confMisc['short_name'])
     
 
#...!...!..................
if __name__ == "__main__":
    main()
        





    
    


