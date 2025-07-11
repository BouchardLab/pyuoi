#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 fit UoI-VAR with ADMM solver on neural activity data

Perlmutter
 IMG=nersc/causal-net:v5   # Mar 28
 export OMP_NUM_THREADS=2
 salloc -q interactive -C cpu --image=$IMG -t 4:00:00 -A m2043 -N 4
 
time  srun -n128 --distribution=block:block shifter python  fit_uoiVar.py  --basePath $basePath   --inpName   daleM80-9638705-c1b7554  --num_admm 8  --time_range 50 20_000 

  basePath=/global/cfs/cdirs/mpccc/balewski/bioDataVault2025/causalNet_tmp2/

Options:
  --inpName         Name of input HDF5 file containing neural activity features
  --fitName         Optional custom name for the fit output
  --time_range      Time range in seconds to use for fitting (default: [0., 5.0])
  --time_rebin      Number of time steps to average together (default: 1)
  --input_type      Data type: 'state' for raw data, 'rate' for exp(state) (default: state)
  --num_admm        Number of ADMM processes per bootstrap group (default: 16)
  --rho_scaler      Scaling factor for ADMM rho parameter updates (default: 2.0)
  --selection_frac  Fraction of data used for feature selection phase (default: 0.9)

''' 

import os,sys,hashlib
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from Util_CausalNet  import rebin_axis0_average

from time import time
from pprint import pprint
import numpy as np
from mpi4py import MPI

sys.path.append("/global/homes/b/balewski/prjs/2025_UoI-VAR/")
from examples.var_utils import * 
from src.pyuoi.linear_model import *
sys.path.append("/global/homes/b/balewski/prjs/2025_UoI-VAR/src/pyuoi/linear_model")
from sparse_comm_util import build_bootstrap_comm


# Record script start time
script_start_time = time()
omp_threads = os.environ.get("OMP_NUM_THREADS", "Not Set")
assert omp_threads=='2'
comm = MPI.COMM_WORLD

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2,3,4],  help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("--basePath",default='out',help="head dir for any results")
                        
    parser.add_argument("--inpName",  required=True,help='name of input data')
    parser.add_argument("--fitName",  default=None,help='fit name')

    #.... fit setup
    parser.add_argument('--time_range' , default=[50, 10_000],  nargs=2,   type=int, help='fit data time range')
    parser.add_argument("--time_rebin", type=int, default=1, help="num time steps to be averaged")
    parser.add_argument('--input_type' , default='state', choices=['state','rate'] , help=' rate=exp(state)')
    parser.add_argument("--num_admm", type=int, default=16, help="num of processes per node to solve the bootstrap variable selection problem in a distributed fashion")
    parser.add_argument("--rho_scaler", type=float, default=1.5, help="scaling factor for ADMM rho parameter")
    parser.add_argument("--selection_frac", type=float, default=0.9, help="fraction of data used for feature selection phase (default: 0.9)")
    
    args = parser.parse_args()
    args.rndSeed=42

    # UoI_Lasso parameters
    args.n_boots_sel = 12
    args.n_boots_est = 12
    args.n_lambdas = 48
    args.max_iter = 1000
    args.imbalance_tolerance = 10.0  # was 0.1
    args.eps = 1e-7
    args.solver = 'admm'
    args.estimation_solver = 'ls'

    args.dataPath=os.path.join(args.basePath,'input_fitter')    
    #args.dataPath=os.path.join(args.basePath,'input_uoi')    

    args.modelPath=os.path.join(args.basePath,'model_uoi')
   
    if  comm.rank == 0 :  
        print( 'myArg-program:',parser.prog)
        for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
        print('',flush=True)
    
    assert os.path.exists(args.dataPath)
    assert os.path.exists(args.modelPath)
    if args.time_range!=None: assert args.time_range[0] < args.time_range[1] 
    return args


#...!...!....................
def uoiVar_predict(bigD,md):
    fim=md['fit_uoi']
    data=bigD['fit_inp_data'][:,:500]
    print('test_data:',data.shape)
    lag=fim['lag_window']
    X,Y = vectorization(data, lag)
    print('pX:',X.shape)
    print('pY:',Y.shape)
    UoI_Lasso.predict(X)

   
#...!...!....................
def rank0_init_uoiVar(args):   
    inpF=args.inpName+'.spikes.h5'

    bigD,md=read4_data_hdf5(os.path.join(args.dataPath,inpF))
    pmd=md['dataset']
    sem=md['selector']
        
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        if args.verb>=3:
            print(expD)
        stop2
    featData=bigD['stateVec_data'].T
        
    #.... clip data
    tL,tR= args.time_range 
    print('FUV tbinLR:',tL,tR)
    assert tR < featData.shape[0]
    featData=featData[tL:tR].astype(np.float64)
    sem['time_range']=[args.time_range[0], args.time_range[1]]
    sem['time_rebin']=args.time_rebin
    sem['input_type']=args.input_type
       
    if args.time_rebin>1:  # averag data over time        
        sem['time_step']=args.time_rebin/fr
        featData= rebin_axis0_average(featData, args.time_rebin)
        
    if args.input_type=='rate':
         featData=np.exp( featData)
         
    sem['num_feature']=featData.shape[1]
    sem['num_time_bin']=featData.shape[0]
    bigD['fit_inp_data']=featData
    return bigD,md

#...!...!....................
def fit_uoiVar_M():   
    lag=1  # hardcoded, makes no sense to use larger
    num_samp,num_feat=mydata.shape  
    
    if rank == 0:
        bigD,md=expD,expMD 
        print('FUOI mydata:%s  lag:%d  numRank:%d '%(mydata.shape,lag,comm.Get_size()),flush=True)
        assert mydata.shape[0] > mydata.shape[1]  # UoI wants [timeBins,features]
        fim={};  md['fit_uoi']=fim
        fim['lag_depth']=lag
        fim['data_shape']=list(mydata.shape)
        fim['num_rank']=comm.Get_size()
        fim['num_admm']=args.num_admm
        
        # Compute 4-ratio
        ratio4 = (num_samp * args.selection_frac) / (num_feat * args.num_admm)
        fim['four_ratio'] = ratio4
        fim['s2f_ratio']=num_samp / num_feat
        
        # Add UoI_Lasso parameters
        fim['n_boots_sel'] = args.n_boots_sel
        fim['n_boots_est'] = args.n_boots_est
        fim['selection_frac'] = args.selection_frac
        fim['n_lambdas'] = args.n_lambdas
        fim['max_iter'] = args.max_iter
        fim['rho_scaler'] = args.rho_scaler
        fim['imbalance_tolerance'] = args.imbalance_tolerance
        fim['eps'] = args.eps
        fim['solver'] = args.solver
        fim['estimation_solver'] = args.estimation_solver

        fim['hash']=hashlib.md5(os.urandom(32)).hexdigest()[:6]
        if args.fitName==None:
            md['short_name']='fit-%s'%(fim['hash'])
        else:
            md['short_name']=args.fitName
        pprint(fim); print( flush=True)

    if num_ranks==1 and  comm.rank == 0: # dump input array
        dataF='%s-%s.npy'%(args.inpName,fim['hash'])
        # Save array to a file in fp16 format
        np.save(dataF, mydata.astype(np.float16))  # Saves in binary .npy format with fp16
        file_size_mb = os.path.getsize(dataF)/(1024*1024)  # Convert bytes to MB
        print('Saved:',dataF,'shape:',mydata.shape,'size: %.1f MB'%file_size_mb,'dtype: float16')
        exit(0)

    # All ranks: Initialize
    boot_comm = build_bootstrap_comm(comm, args.num_admm)
    
    uoi_lasso = UoI_Lasso(fit_VAR = True, fit_intercept=False, 
                         n_boots_sel=args.n_boots_sel, 
                         n_boots_est=args.n_boots_est, 
                         selection_frac=args.selection_frac, 
                         n_lambdas=args.n_lambdas, 
                         max_iter=args.max_iter, 
                         eps=args.eps, 
                         random_state=args.rndSeed, 
                         comm=boot_comm, 
                         global_comm=comm, 
                         n_admm=args.num_admm, 
                         rho_scaler=args.rho_scaler, 
                         imbalance_tolerance=args.imbalance_tolerance, 
                         solver=args.solver, 
                         estimation_solver=args.estimation_solver)

    # fit UoI_Lasso
    start_time = time()

    if boot_comm is not None:  #if the global_rank is part of the boostrap distribution(not admm distribution)
        if boot_comm.rank == 0:
            uoi_lasso.fit(lag, data = mydata)
        else:
            uoi_lasso.fit(lag)
    else:
        if uoi_lasso.solver == "admm":
            uoi_lasso.admm_queue()
    
    fit_time = time() - start_time

    # Rank 0 collects all fit times
    fit_times = comm.gather(fit_time, root=0)

    if rank != 0:  exit(0)
    
    avg_time = np.mean(fit_times)
    min_time = np.min(fit_times)
    max_time = np.max(fit_times)
    total_time = time() - script_start_time  # Total execution time from script start

    print("------------------------------------------------------------")
    #print("Fitting complete in %.1f sec | numRanks=%d" % (total_time, fim['num_rank']))
    print("Total Execution Time: %.3f sec" % total_time)
    print("Avg Fit Time: %.1f sec | Min: %.1f sec | Max: %.1f sec" % (avg_time, min_time, max_time))
    print("------------------------------------------------------------", flush=True)

    # Extract model coefficients
    B_model = uoi_lasso.coef_
    A_model = [B_model.reshape(num_feat, num_feat * lag).T[i * num_feat:(i + 1) * num_feat].T for i in range(lag)]
    A_model = np.array(A_model)

    print("B_model: (%d,) | A_model: (%d, %d, %d)" % (B_model.shape[0], A_model.shape[0], A_model.shape[1], A_model.shape[2]))
  
    fim['fit_time']=float(max_time)
    fim['total_run_time']=total_time
    bigD['fit_B_model']=B_model
    bigD['fit_A_model']=A_model

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=3)
    np.random.seed(args.rndSeed)
    
    num_ranks = comm.Get_size()
    if args.num_admm > num_ranks: args.num_admm = num_ranks

    assert  num_ranks % args.num_admm ==0

    rank = comm.Get_rank()

    if rank == 0:
        expD,expMD= rank0_init_uoiVar(args)
        mydata=expD['fit_inp_data']
    else:
        mydata = None
    
    # Broadcast data to all ranks
    mydata = comm.bcast(mydata, root=0)
      
    fit_uoiVar_M()
    # only rank0 will proceed further

    #...... WRITE   OUTPUT .........
    outF=os.path.join(args.modelPath,expMD['short_name']+'.fitUoI.h5')
    write4_data_hdf5(expD,outF,expMD)
    #pprint(expMD)
    fim=expMD['fit_uoi']
    

    print(' ./postproc_uoiVar.py --basePath $basePath -e %s  -p e a b -Y '%expMD['short_name'])
