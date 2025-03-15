#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
 fit UoI-VAR

Perlmutter

shifter ./fit_uoiVar.py --dataPath /global/cfs/cdirs/m2043/causal_inference/DIV13/features --inpName HET_80k_1_samp1kHz 
>>> 60 sec

 srun -n64 shifter ./fit_uoiVar.py --dataPath /global/cfs/cdirs/m2043/causal_inference/DIV13/features --inpName HET_80k_1_samp1kHz  --time_range 7 9 --lag_depth 2 --num_feature 20 
>>> fit data: (2000, 20)
>>> Total Execution Time: 15.825 sec

--time_range 5 9  --num_feature 40 
>>> fit data: (4000, 40)
free ram: 217
>>> Total Execution Time: 141.401 sec

 --time_range 5 9 --lag_depth 3 --num_feature 40 
free ram 94
>>> Total Execution Time: 232.385 sec

 srun -n32 shifter ./fit_uoiVar.py --dataPath /global/cfs/cdirs/m2043/causal_inference/DIV13/features --inpName HET_80k_1_samp1kHz  --time_range 4 9 --lag_depth 3 --num_feature 50 
>>>OOM

-n16


''' 

import os,sys,hashlib
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from time import time
from pprint import pprint
import numpy as np

from pyuoi.linear_model import *
sys.path.append(os.path.abspath("../../"))
from examples.var_utils import *
from mpi4py import MPI

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
    parser.add_argument("--dataPath",default=None,help="direct input path")
                        
    parser.add_argument("--inpName",  required=True,help='name of input data')
    parser.add_argument("--fitName",  default=None,help='fit name')

    #.... fit params
    parser.add_argument('--num_feature', default=8, type=int, help='num of from full dataset')
    parser.add_argument('--time_range' , default=[0., 1.0],  nargs=2,   type=float, help='fit data time range')
    parser.add_argument('--test_time_range' , default=None,  nargs=2,   type=float, help='test data range')
    parser.add_argument('--lag_depth', default=10, type=int, help='depth of auotorgeression')
    
    args = parser.parse_args()
    # make arguments  more flexible
    if args.dataPath==None:
        args.dataPath=os.path.join(args.basePath,'features')
    
    args.modelPath=os.path.join(args.basePath,'model')
   
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
    nfeat=min(pmd['num_feature'],args.num_feature)
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        if args.verb>=3:
            print(expD)
        stop2
    data=bigD['feature'][:nfeat]
    if args.time_range:  #.... clip data
        tL,tR=[int(x*pmd['sampling_freq']) for x in args.time_range ]
        print('FUV tbinLR:',tL,tR)
        assert tL < pmd['num_time_bin']
        data=data[:,tL:tR]
        
    data=data.T # to match UoI input format
    print('fit data:',data.shape,data.dtype)
    bigD['fit_data']=data
    return bigD,md

#...!...!....................
def fit_uoiVar_M():   
    lag=args.lag_depth
    num_samp,num_feat=mydata.shape  
    X,Y = vectorization(mydata, lag)
    if rank == 0:
        bigD,md=expD,expMD
        print(rank, 'mydata:',mydata.shape,'fX:',X.shape, 'fY:',Y.shape,'lag=%d muRank=%d'%(lag,comm.Get_size()),flush=True)
        fim={};  md['fit_uoi']=fim
        fim['lag_depth']=lag
        fim['X_shape']=list(X.shape)
        fim['data_shape']=list(mydata.shape)
        fim['num_rank']=comm.Get_size()

        fim['hash']=hashlib.md5(os.urandom(32)).hexdigest()[:6]
        if args.fitName==None:
            md['short_name']='fit-%s'%(fim['hash'])
        else:
            md['short_name']=args.fitName

    if 0 and  comm.rank == 0: # dump input array
        dataF='%s-%s.npy'%(args.inpName,fim['hash'])
        # Save array to a file
        np.save(dataF, data)  # Saves in binary .npy format
        print('Saved:',dataF,'shape:',data.shape)
        exit(0)

    # All ranks: Initialize and fit UoI_Lasso
    uoi_lasso = UoI_Lasso(n_real_features=num_feat, fit_VAR=True, random_state=42, comm=comm)

    start_time = time()
    uoi_lasso.fit(X, Y)
    fit_time = time() - start_time

    # Rank 0 collects all fit times
    fit_times = comm.gather(fit_time, root=0)

    if rank != 0:  exit(0)
    
    avg_time = np.mean(fit_times)
    min_time = np.min(fit_times)
    max_time = np.max(fit_times)
    total_time = time() - script_start_time  # Total execution time from script start

    print("------------------------------------------------------------")
    print("Fitting complete in %.1f sec | numRanks=%d" % (total_time, fim['num_rank']))
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
    model = uoi_lasso.coef_
    bigD['fit_model']=model

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()
    np.set_printoptions(precision=3)

    rank = comm.Get_rank()

    if rank == 0:
        expD,expMD= rank0_init_uoiVar(args)
        mydata=expD['fit_data']
    else:
        mydata = None

    # Broadcast data to all ranks
    mydata = comm.bcast(mydata, root=0)
      
    fit_uoiVar_M()  # only rank0 will proceed this function
     
    #...... WRITE   OUTPUT .........
    outF=os.path.join(args.modelPath,expMD['short_name']+'.fit.h5')
    write4_data_hdf5(expD,outF,expMD)
    pprint(expMD)
    fim=expMD['fit_uoi']
    
    print('SUM2 %s fit time %.1f sec'%(expMD['short_name'],expMD['fit_uoi']['fit_time']))
    print('SUM0,job_name,fit_time,num_feat,num_tbin,lag_depth,num_rank')
    print('SUM1,%s,%.1f,%d,%d,%d,%d\n'%(expMD['short_name'],fim['fit_time'],fim['data_shape'][1],fim['data_shape'][0],fim['lag_depth'],fim['num_rank']))

