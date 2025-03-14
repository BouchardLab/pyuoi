#!/usr/bin/env python3

''' use cases
 ./dummy_mpi_lasso.py --num_feat  15 --num_samp  300 --lag 2

 srun -n 8 ./dummy_mpi_lasso.py --num_feat  15 --num_samp  300 --lag 2
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

# Dummy Lasso class (Replacing UoI_Lasso)
class DummyLasso:
    def __init__(self, n_real_features, fit_VAR, random_state, comm):
        self.n_real_features = n_real_features
        self.fit_VAR = fit_VAR
        self.random_state = random_state
        self.comm = comm
        self.coef_ = None  # Placeholder for model coefficients

    def fit(self, X, Y):
        np.random.seed(self.random_state)
        # Simulate computational workload
        sleep(np.random.uniform(2, 5))  # Simulate varied fit time
        self.coef_ = np.random.randn(X.shape[1])  # Dummy coefficients

# Dummy data generation function
def generate_dummy_data(num_feat, num_samp, lag):
    np.random.seed(42)
    X = np.random.randn(num_samp, num_feat * lag)
    Y = np.random.randn(num_samp, num_feat)
    return X, Y

def main(num_feat, num_samp, lag):
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

        print("Generating data... nFeat=%d, nSamp=%d, lag=%d" % (num_feat, num_samp, lag), flush=True)
        
        # Generate dummy data (Only rank 0)
        X, Y = generate_dummy_data(num_feat, num_samp, lag)
    else:
        X = None
        Y = None

    # Broadcast X, Y to all ranks
    X = comm.bcast(X, root=0)
    Y = comm.bcast(Y, root=0)

    # All ranks: Initialize and fit DummyLasso
    lasso_model = DummyLasso(n_real_features=num_feat, fit_VAR=True, random_state=42, comm=comm)

    start_time = time()
    lasso_model.fit(X, Y)
    fit_time = time() - start_time

    # Rank 0 collects all fit times
    fit_times = comm.gather(fit_time, root=0)

    if rank == 0:
        avg_time = np.mean(fit_times)
        min_time = np.min(fit_times)
        max_time = np.max(fit_times)
        total_time = time() - script_start_time  # Total execution time from script start

        print("------------------------------------------------------------")
        print("Fitting complete in %.1f sec | numRanks=%d" % (total_time, num_ranks))
        print("Avg Fit Time: %.3f sec | Min: %.3f sec | Max: %.3f sec" % (avg_time, min_time, max_time))
        print("Total Execution Time: %.3f sec" % total_time)
        print("------------------------------------------------------------", flush=True)

        # Dummy output for model coefficients
        B_model = lasso_model.coef_
        print("B_model: (%d,)" % (B_model.shape[0],))

if __name__ == "__main__":
    # Use argparse for command-line inputs
    parser = argparse.ArgumentParser(description="MPI-based Dummy Lasso fitting.")
    parser.add_argument("--num_feat", type=int, default=20, help="Number of features (default: 20)")
    parser.add_argument("--num_samp", type=int, default=300, help="Number of samples (default: 300)")
    parser.add_argument("--lag", type=int, default=2, help="Lag value (default: 2)")
    
    args = parser.parse_args()
    
    main(args.num_feat, args.num_samp, args.lag)
