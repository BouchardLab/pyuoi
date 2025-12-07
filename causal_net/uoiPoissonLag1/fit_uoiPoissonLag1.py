#!/usr/bin/env python3

'''

IMG=nersc/causal-net:v5   # Nov 5 2025
export OMP_NUM_THREADS=2
salloc -q interactive -C cpu --image=$IMG -t 4:00:00 -A m2043 -N 4

time srun -n 128 --distribution=block:block shifter python  fit_uoiPoissonLag1.py  --samples 10_000

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
from Util_poissonFdr import  extract_loss_traces , qa_Bfit, qa_Afit

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
#...!...!..................
def main():
    parser = argparse.ArgumentParser(description='UoI-VAR Poisson ADMM test')
    parser.add_argument('--dataName', default='daleM20_746c4b', help='dataset name (e.g., daleM20_746c4b)')
    parser.add_argument("--dataPath", type=str, default="/pscratch/sd/b/balewski/2025_causalNet_tmp/")
    parser.add_argument("--outPath", type=str, default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="output directory for fit artifacts")

    parser.add_argument('--samples', type=int, default=100000, help='number of data samples to use')
    parser.add_argument('--fitName', default=None, help='output file name core')
    parser.add_argument('--dropFreqWeight', action='store_true', help='disable frequency dependent weights; default uses frequency weights')
    parser.add_argument('--maxIter', type=int, default=300, help='maximum number of iterations for ADMM')
    parser.add_argument('--fdrRate', type=float, default=0.02, help='False discovery rate level for support selection')
    parser.add_argument('--selectonFrac', type=float, default=0.8, help='fraction of data used per bootstrap selection')
    parser.add_argument('--numBoots', type=int, default=5, help='number of bootstrap samples for selection/estimation')
    parser.add_argument('--verb', '-v', type=int, default=1, help='Verbosity level')
    args = parser.parse_args()

    confUoI = {
        'fit_VAR': True,
        'fit_intercept': False,
        'standardize': False,
        #'manual_l1_range': [6e-7, 3e-6],   'n_lambdas': 4,
        'manual_l1_range': [6e-7, 7e-7],   'n_lambdas': 2,  # needs to be changed with M-neurons
        'n_boots_sel': args.numBoots,    'n_boots_est': args.numBoots,
        'selection_frac': args.selectonFrac,
        'max_iter': args.maxIter, 
        'rho_scaler': 1.0,      'imbalance_tolerance': 1, # do not change
        'n_admm': 32,     
        'solver': 'admm',
        'dt': 0.01,  # must be hardcoded - or use broadcasting to all ranks
        'fdr_rate': args.fdrRate,
        'estimation_solver': "lbfgs",
        'random_state': 22,
    }

    
    # end config print
    lag = 1 #confMisc['model_lag']
    assert lag==1

    rank = 0
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    if comm is not None:
        rank = comm.rank

    if rank == 0: 
        for arg in vars(args):
            print( 'myArgs:',arg, getattr(args, arg))

        hash_str = secrets.token_hex(3)
        outName = args.fitName
        if outName is None:       
            outName = f'{args.dataName}_uoi{hash_str}'
            
        if args.verb > 1:
            print('confUoI:'); pprint(confUoI)
            #Xprint('confMisc:'); pprint(confMisc)
            
        print('Start dataName=%s  samples=%d'%(args.dataName, args.samples))
        spikeF=os.path.join(args.dataPath, "{}.spikes.npz".format(args.dataName))
        spikeD, spikeMD = read_data_npz(spikeF, verb=True)
        data=spikeD['spikes'][:args.samples].astype(np.double)
        single_rates=spikeD['single_rates']
        data_pois = None    
      
        #1confUoI['dt']=spikeMD['time_step_sec']  # can't do it now w/o broadcasting to all ranks
        # Slurm context (assume run with srun)
        slurm_nodes = int(os.environ.get('SLURM_NNODES') or os.environ.get('SLURM_JOB_NUM_NODES') or 0)
        slumr_ranks = int(os.environ.get('SLURM_NTASKS') or world_size)
        uoiMD = { 'fit_output_name': outName, 'fit_input_name': args.dataName,  'fit_input_path':args.dataPath,'num_samples_used': args.samples, 'max_iter': args.maxIter,  'fdr_rate': args.fdrRate,   'use_freq_weight': not args.dropFreqWeight, 'num_neurons': data.shape[1], 'sampl_selection_frac': args.selectonFrac, 'slurm_nodes': slurm_nodes, 'slurm_ranks': slumr_ranks }
        outMD={'spike_data': spikeMD,'fit_uoi':uoiMD ,'fit_type':'uoiFdr','hash':hash_str,'conf_uoi':confUoI}
        #pprint(outMD)
        if args.dropFreqWeight: 
            w = np.ones(data.shape[1])
        else:
            rates=np.mean(data,axis = 0)/confUoI['dt']
            rates = np.clip(rates, 0.1, 50)
            w=1/rates
            w/=np.sum(w)
            w*=data.shape[1]
            print('rates:',rates)
            print('w',w)
           
    else:
        w = None

    w = comm.bcast(w, root=0)
    if rank == 0: print('%dk samples loaded to all %d  ranks'%(args.samples//1000,world_size ),flush=True)

    #fitting with multiple processes
    boot_comm = build_bootstrap_comm(comm, confUoI['n_admm'])
    confUoI_fit = confUoI.copy()
    confUoI_fit.pop('loss_stride', None)

    uoi_poisson = UoI_Poisson(**confUoI_fit, comm=boot_comm, global_comm=comm, weights=w)
    assert uoi_poisson.solver == "admm"
    
    Tstart = time()
    if boot_comm is not None:  #if the global_rank is part of the boostrap distribution(not admm distribution)                
        if boot_comm.rank == 0:
            uoi_poisson.fit(lag, data = data, data_pois = data_pois)
        else:
            uoi_poisson.fit(lag)
    else:                
        uoi_poisson.admm_queue()
   
    if rank > 0: return
    elaT= time()-Tstart 
    print("rank %d : Fitting complete in %d  seconds."%(comm.rank, elaT), flush = True)

    #... finalize meta-data
    confUoI['model_lag']=lag
    uoiMD['training_time_sec']=elaT
    outMD['short_name']=uoiMD['fit_output_name']
    A_fit = uoi_poisson.VAR_coef_[0]
    B_fit = uoi_poisson.VAR_bias_

    loss_stride =10 # hardcoded by Yao
    sel_loss_iter, sel_loss_l1 = extract_loss_traces(getattr(uoi_poisson, '_selection_lm', None), loss_stride)
    est_loss_iter, est_loss_l1 = extract_loss_traces(getattr(uoi_poisson, '_estimation_lm', None), loss_stride)

    l1_loss_sel = np.vstack((sel_loss_iter, sel_loss_l1)) if sel_loss_iter.size > 0 else np.empty((2, 0))
    l1_loss_est = np.vstack((est_loss_iter, est_loss_l1)) if est_loss_iter.size > 0 else np.empty((2, 0))
    
    bigD={ 'A_uoiFdr':A_fit, 'B_uoiFdr':B_fit, 'l1_loss_sel':l1_loss_sel, 'l1_loss_est':l1_loss_est, 'single_rates':single_rates}

    outF = os.path.join(args.outPath, "{}.uoiFdr.npz".format(outName))
      
    write_data_npz(bigD, outF, metaD=outMD)
    print('saved output to:',outF)
    #pprint(outMD)

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
    
    
    print('detailed QA  %s  M=%d  M^2=%d  samples=%d/k' %(args.dataName,M,M2,args.samples/1000))
    qaD=qa_Afit(A_truth, A_fit)
    qaD['bterm']=qa_Bfit(B_truth, B_fit)

    #if spikeMD['data_type']=='simDale':         flags=' -p  a  c  '
    #   else:
    flags=' -p a b  '
    print('\n  shifter   ./eval_fitUoI.py --dataPath $dataPath  --dataName %s  %s    -X ' % (outName,flags))
       
    print('    --dataPath '+args.dataPath)

 
#...!...!..................
if __name__ == "__main__":
    main()
        





    
    


