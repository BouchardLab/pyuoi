#!/usr/bin/env python3

''' 
IMG=nersc/causal-net:v4   # May 13
 export OMP_NUM_THREADS=2
 salloc -q interactive -C cpu --image=$IMG -t 4:00:00 -A m2043 -N 1

uoi_lasso = UoI_Lasso(fit_VAR = True, fit_intercept=False, n_boots_sel=n_boots_sel, n_boots_est=n_boots_est, n_lambdas = n_lambdas, random_state=None, comm = boot_comm, global_comm = comm, n_admm = n_admm, admm_rho = rho, max_iter = 50, solver='admm', estimation_solver = "ls")

- max_iter = 50; enough convergence and for faster runtime
- estimation_solver = "ls": using standard Linear Regression library is faster for parameter estimation than ADMM
- REMOVED n_real_features variable, since it's obsolete now


NOTE: distribution=block:block forces the ADMM processes for each bootstrap to localize to a single compute node for efficient communication

Minimal example testing code consistency:  38 sec
 srun -n 1 --distribution=block:block shifter python uoi_mpi_admm_v5.py --num_feat 10 --num_samp 50 --num_admm 1
>>> Fitting complete in 52.2 sec | numRanks=1

Doing meaningfull fit : 42 sec on N=1 
srun -n 128 --distribution=block:block shifter python uoi_mpi_admm_v5.py --num_feat 20 --num_samp 5000 --num_admm 16

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
def print_dale_matrix(A,nfeat=None):
    if nfeat==None: nfeat=A.shape[0]
    # Function to format values
    def format_value(val):
        if abs(val) < 0.01:
            return "  .  "  # Represent zero as '-'
        return f"{val:+5.2f}"  # Format as +0.12 or -0.23
    
    col_indices = "feat " + "     ".join(f"{i:2d}" for i in range(nfeat))
    print(col_indices)
    # Print row index and formatted values
    for i in range(nfeat):
        row=A[i]
        formatted_row = "  ".join(format_value(row[j]) for j in range(nfeat) )
        print(f"{i:2d}  {formatted_row}")  # Row index + formatted values

 
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


#...!...!....................
def main(num_feat, num_samp, lag, inpName, n_admm):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    # Synchronize all ranks and measure startup time
    comm.barrier()  # Ensure all ranks reach this point
    startup_time = time() - script_start_time  # Time from script start to this point

    if rank == 0:
        T1=time()
        print("------------------------------------------------------------")
        print("MPI Startup Complete | numRanks=%d | Startup Time: %.3f sec" % (num_ranks, startup_time))
        print("------------------------------------------------------------", flush=True)

        if inpName==None:
            # Generate dummy data (Only rank 0)
            print("Generating data... nFeat=%d, nSamp=%d, lag=%d" % (num_feat, num_samp, lag), flush=True)
            mydata = generate_dummy_data(num_feat, num_samp, lag)
            print(' runk0 generated data, elaT=%.1f min'%( (time()-T1)/60.), flush=True)
        else:
            print(' Load array back from file:',inpName, flush=True)    
            mydata = np.load(inpName)
            num_samp,num_feat=mydata.shape  
    else:
        mydata = None

    if rank == 0:
        print('n_admm , num_ranks:', n_admm , num_ranks)
        print('mydata:',mydata.shape,type(mydata),'start fit ...', flush=True)

    start_time = time()
    
    if use_admm:   # very scalable        
        assert n_admm <= num_ranks
        assert  num_ranks % n_admm ==0
        boot_comm = build_bootstrap_comm(comm, n_admm)

        n_boots_sel = 12
        n_boots_est = 12
        selection_frac = 0.9
        n_lambdas = 48
        max_iter = 1000
        seed = 42
        rho_scaler = 1.5  # was 2.0
        imbalance_tolerance = 10.  #was 0.1
        eps=1e-7
        uoi_lasso = UoI_Lasso(fit_VAR = True, fit_intercept=False, n_boots_sel=n_boots_sel, n_boots_est=n_boots_est, selection_frac = selection_frac, n_lambdas = n_lambdas, max_iter = max_iter, eps = eps, random_state=seed, comm = boot_comm, global_comm = comm, n_admm = n_admm, rho_scaler = rho_scaler, imbalance_tolerance = imbalance_tolerance, solver='admm', estimation_solver = "ls")


        if boot_comm is not None:  #if the global_rank is part of the boostrap distribution(not admm distribution)
            if boot_comm.rank == 0:
                uoi_lasso.fit(lag, data = mydata)
            else:
                uoi_lasso.fit(lag)
        else:
            if uoi_lasso.solver == "admm":
                uoi_lasso.admm_queue()
    
    else:  #original implemetation, not scalable
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
    A=A_model[0]
    nfeat=min(20,A.shape[0])
    print_dale_matrix(A,nfeat)  
    
#=================================
#  M A I N 
#=================================

if __name__ == "__main__":
    # Use argparse for command-line inputs
    parser = argparse.ArgumentParser(description="MPI-based Dummy Lasso fitting.")
    parser.add_argument("--num_feat", type=int, default=20, help="Number of features ")
    parser.add_argument("--num_samp", type=int, default=50_000, help="Number of samples")
    parser.add_argument("--num_admm", type=int, default=32, help="num of processes per node to solve the bootstrap variable selection problem in a distributed fashion")
    parser.add_argument("--lag", type=int, default=1, help="Lag value ")
    parser.add_argument("--inpName",  default=None,help='input name, will define num features and num samples')
    
    args = parser.parse_args()
    use_admm = True   # very scalable 
    #n_admm = 64  # n_process should be multiple of n_admm, and at MOST n_admm*n_boot*n_reg_param
    rho = None
    #rho = 1e10
    
    main(args.num_feat, args.num_samp, args.lag, args.inpName, args.num_admm)
