#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 fit UoI-VAR

Perlmutter
 IMG=nersc/causal-net:v5   # Mar 28
 export OMP_NUM_THREADS=2
 salloc -q interactive -C cpu --image=$IMG -t 4:00:00 -A m2043 -N 1
 module load python


 dataPath=/global/cfs/cdirs/m2043/causal_inference/DIV13/features 
 basePath=/global/cfs/cdirs/mpccc/balewski/bioDataVault2025/causalNet_tmp/

Options:
  --matrixName  Path to the HDF5 file containing connectivity matrices.
  --sigma       Noise variance strength 
  --tau         (sec) Time constant for self-forgetting  (defulat 10 msec )
  --T           (sec) Total evolution time (default: 60  sec)
  --dt          (sec) time step (default:  1 msec )
  --outName     Output HDF5 file name for simulation results (default: simu_ac 
  --seed        Random seed for simulation (optional)


''' 

import os,sys,hashlib
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from toolbox.Util_Dale_LDS  import rebin_axis0_average

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
    parser.add_argument('--time_range' , default=[0.3, 1.],  nargs=2,   type=float, help='fit data time range')
    parser.add_argument("--time_rebin", type=int, default=1, help="num time steps to be averaged")
    parser.add_argument('--input_type' , default='state', choices=['state','rate'] , help=' rate=exp(state)')
    parser.add_argument("--num_admm", type=int, default=32, help="num of processes per node to solve the bootstrap variable selection problem in a distributed fashion")
    
    args = parser.parse_args()
    # make arguments  more flexible
    args.rndSeed=42
    args.admm_rho=None # ADMM penalty parameter: rho (need some heuristics)

    args.dataPath=os.path.join(args.basePath,'input_uoi')    
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
    data=bigD['fit_data'][:,:500]
    print('test_data:',data.shape)
    lag=fim['lag_window']
    X,Y = vectorization(data, lag)
    print('pX:',X.shape)
    print('pY:',Y.shape)
    UoI_Lasso.predict(X)

   
#...!...!....................
def rank0_init_uoiVar(args):   
    inpF=args.inpName+'.act.h5'
    bigD,md=read4_data_hdf5(os.path.join(args.dataPath,inpF))
    pmd=md['payload']
    sem=md['selector']
    dt=sem['sampling_freq']
    
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        if args.verb>=3:
            print(expD)
        stop2
    featData=bigD['features']
        
    #.... clip data
    tL,tR=[int(x*dt) for x in args.time_range ]
    print('FUV tbinLR:',tL,tR)
    assert tR < featData.shape[0]
    featData=featData[tL:tR]
    sem['time_range']=[args.time_range[0], args.time_range[1]]
    sem['time_rebin']=args.time_rebin
    sem['input_type']=args.input_type
       
    if args.time_rebin>1:  # averag data over time        
        sem['time_step']=args.time_rebin*dt
        featData= rebin_axis0_average(featData, args.time_rebin)
        
    if args.input_type=='rate':
         featData=np.exp( featData)
         
    sem['num_feature']=featData.shape[1]
    sem['num_time_bin']=featData.shape[0]
    bigD['fit_data']=featData
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
        fim['admm_rho']=args.admm_rho
        

        fim['hash']=hashlib.md5(os.urandom(32)).hexdigest()[:6]
        if args.fitName==None:
            md['short_name']='fit-%s'%(fim['hash'])
        else:
            md['short_name']=args.fitName

    if 0 and  comm.rank == 0: # dump input array
        dataF='%s-%s.npy'%(args.inpName,fim['hash'])
        # Save array to a file
        np.save(dataF, mydata)  # Saves in binary .npy format
        print('Saved:',dataF,'shape:',mydata.shape)
        exit(0)

    # All ranks: Initialize
    boot_comm = build_bootstrap_comm(comm, args.num_admm)
    #uoi_lasso = UoI_Lasso(n_real_features = num_feat, fit_VAR = True, fit_intercept=False, random_state=42, comm = boot_comm, global_comm = comm, n_admm = args.num_admm, admm_rho = args.admm_rho, solver='admm', estimation_solver = "admm")
    uoi_lasso = UoI_Lasso( fit_VAR = True, fit_intercept=False, random_state=42, comm = boot_comm, global_comm = comm, n_admm = args.num_admm, admm_rho = args.admm_rho, max_iter = 50, rho_scaler = 1.2, solver='admm', estimation_solver = "ls")
    
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
    print("Avg Fit Time: %.1f sec | Min: %.1f sec | Max: %.1f sec" % (avg_time, min_time, max_time))
    print("Total Execution Time: %.3f sec" % total_time)
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
    assert args.num_admm <= num_ranks
    assert  num_ranks % args.num_admm ==0

    rank = comm.Get_rank()

    if rank == 0:
        expD,expMD= rank0_init_uoiVar(args)
        mydata=expD['fit_data']
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
    
    print('SUM2 %s fit time %.1f sec'%(expMD['short_name'],expMD['fit_uoi']['fit_time']))
    print('SUM0,job_name,fit_time,num_feat,num_tbin,lag_depth,num_rank')
    print('SUM1,%s,%.1f,%d,%d,%d,%d\n'%(expMD['short_name'],fim['fit_time'],fim['data_shape'][1],fim['data_shape'][0],fim['lag_depth'],fim['num_rank']))

    print(' ./postproc_uoiVar.py --basePath $basePath -e %s  -p e a b -Y '%expMD['short_name'])
