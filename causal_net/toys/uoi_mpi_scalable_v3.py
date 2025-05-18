#!/usr/bin/env python3

''' use cases
IMG=nersc/casual-net:v1 
salloc -q interactive -C cpu --image=$IMG -t 2:00:00 -A m2043 -N 4
export OMP_NUM_THREADS=2

shifter ./uoi_mpi_scalable.py --num_feat  15 --num_samp  100 --lag 1
>>>Total Execution Time: 12.218 sec

shifter ./uoi_mpi_scalable.py --inpName HET_80k_1_samp1kHz-064870.npy
>>> Total Execution Time: 3.075 sec

srun -n 4 shifter  ./uoi_mpi_scalable.py --num_feat  15 --num_samp  300 --lag 2
>>> Total Execution Time: 16.078 sec

srun -n 4 shifter  ./uoi_mpi_scalable.py  --inpName HET_80k_1_samp1kHz-064870.npy

'''
#...!...!....................


import sys
import os
import argparse
import numpy as np
from mpi4py import MPI
from time import time, sleep

# Record script start time
script_start_time = time()
omp_threads = os.environ.get("OMP_NUM_THREADS", "Not Set")
assert omp_threads=='2'


sys.path.append("/global/homes/b/balewski/prjs/2025_UoI-VAR/")
from examples.var_utils import *
from src.pyuoi.linear_model import *
sys.path.append("/global/homes/b/balewski/prjs/2025_UoI-VAR/src/pyuoi/linear_model")
from sparse_comm_util import build_bootstrap_comm


#...!...!....................
def generate_dummy_data(num_feat, num_samp, lag):
    # Select spectral radius based on lag
    rad_dict = {1: 0.98, 2: 0.70, 3: 0.30, 4: 0.20, 5: 0.20}
    rad = rad_dict.get(lag, 0.20)

    # Generate data (Only rank 0)
    data, transition_matrices, cov = generate_sparse_stationary_var_process(
        num_feat, num_samp, lag=lag, sparsity=0.5, spectral_radius=rad,
        process_type='gaussian', random_state=42
    )
       
    return data


use_admm = True
n_admm = 16  # n_process should be multiple of n_admm, and at MOST n_admm*n_boot*n_reg_param
rho = None
#rho = 1e10


#...!...!....................
def main(num_feat, num_samp, lag, inpName):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    # Synchronize all ranks and measure startup time
    comm.barrier()  # Ensure all ranks reach this point
    startup_time = time() - script_start_time  # Time from script start to this point

    if rank == 0:
        print("------------------------------------------------------------")
        print("MPI Startup Complete | numRanks=%d | Startup Time: %.3f sec" % (num_ranks, startup_time))
        print("------------------------------------------------------------", flush=True)

        if inpName==None:
            # Generate dummy data (Only rank 0)
            print("Generating data... nFeat=%d, nSamp=%d, lag=%d" % (num_feat, num_samp, lag), flush=True)
            mydata = generate_dummy_data(num_feat, num_samp, lag)
        else:
            print(' Load array back from file:',inpName, flush=True)    
            mydata = np.load(inpName)
            num_samp,num_feat=mydata.shape  
    else:
        mydata = None

    # Broadcast data to all ranks
    # mydata = comm.bcast(mydata, root=0)
    

    # X, Y = vectorization(mydata, lag)
    if rank == 0:
        print('mydata:',mydata.shape)


    start_time = time()
    
    if use_admm:        
        boot_comm = build_bootstrap_comm(comm, n_admm)
        uoi_lasso = UoI_Lasso(n_real_features = num_feat, fit_VAR = True, fit_intercept=False, random_state=42, comm = boot_comm, global_comm = comm, n_admm = n_admm, admm_rho = rho, solver='admm', estimation_solver = "admm")
        
        if boot_comm is not None:  #if the global_rank is part of the boostrap distribution(not admm distribution)
            if boot_comm.rank == 0:
                uoi_lasso.fit(lag, data = mydata)
            else:
                uoi_lasso.fit(lag)
        else:
            if uoi_lasso.solver == "admm":
                uoi_lasso.admm_queue()
    
    else:  #original implemetation
        uoi_lasso = UoI_Lasso(n_real_features = num_feat, fit_VAR = True, fit_intercept=False, random_state=42, comm = comm)    
    
        if comm.rank == 0:
            uoi_lasso.fit(lag, data = mydata)
        else:
            uoi_lasso.fit(lag)
    
    fit_time = time() - start_time

    # Rank 0 collects all fit times
    fit_times = comm.gather(fit_time, root=0)

    if rank != 0:  return
    
    avg_time = np.mean(fit_times)
    min_time = np.min(fit_times)
    max_time = np.max(fit_times)
    total_time = time() - script_start_time  # Total execution time from script start

    print("------------------------------------------------------------")
    print("Fitting complete in %.1f sec | numRanks=%d" % (total_time, num_ranks))
    print("Avg Fit Time: %.1f sec | Min: %.1f sec | Max: %.1f sec" % (avg_time, min_time, max_time))
    print("Total Execution Time: %.3f sec" % total_time)
    print("------------------------------------------------------------", flush=True)

    # Extract model coefficients
    # B_model = uoi_lasso.coef_
    # A_model = [B_model.reshape(num_feat, num_feat * lag).T[i * num_feat:(i + 1) * num_feat].T for i in range(lag)]
    # A_model = np.array(A_model)

    A_model = uoi_lasso.VAR_coef_

    print("A_model: (%d, %d, %d)" % (A_model.shape[0], A_model.shape[1], A_model.shape[2]))
    # print("B_model: (%d,) | A_model: (%d, %d, %d)" % (B_model.shape[0], A_model.shape[0], A_model.shape[1], A_model.shape[2]))

#=================================
#  M A I N 
#=================================

if __name__ == "__main__":
    # Use argparse for command-line inputs
    parser = argparse.ArgumentParser(description="MPI-based Dummy Lasso fitting.")
    parser.add_argument("--num_feat", type=int, default=10, help="Number of features (default: 20)")
    parser.add_argument("--num_samp", type=int, default=50, help="Number of samples (default: 300)")
    parser.add_argument("--lag", type=int, default=1, help="Lag value (default: 2)")
    parser.add_argument("--inpName",  default=None,help='input name, will define num features and num samples')
    
    args = parser.parse_args()
    
    main(args.num_feat, args.num_samp, args.lag, args.inpName)
