import numpy as np

from pyuoi.linear_model import *
from var_utils import *

from mpi4py import MPI
from time import time
from pyuoi.linear_model.sparse_comm_util import build_bootstrap_comm


n_features = 10
n_samples = 2000
lag = 1

#srun -n 1000 --ntasks-per-node=250 --distribution=block:block python uoi_var_mpi_check.py

n_boots_sel=12
n_boots_est=12
n_lambdas = 48

use_admm = True
n_admm = 16  # n_process should be multiple of n_admm, and at MOST n_admm*n_boot*n_reg_param
rho = None
#rho = 1e10


rank = 0
comm = MPI.COMM_WORLD
if comm is not None:
    rank = comm.rank
    
if rank == 0:
    data, transition_matrices, cov = generate_sparse_stationary_var_process(
        n_features,
        n_samples,
        lag=lag,
        sparsity=0.5,
        spectral_radius=0.98,  # Ensures stationarity
        process_type='gaussian',
        random_state = 42
    
    )

    # OBSOLETE: used for older version of VAR fit
    # dense_matrices = [M.toarray() for M in  transition_matrices]
    # vecortized the ground truth transition matrices
    # B_truth = np.vstack([m.T for m in dense_matrices]).T.flatten()     
    # X,Y = vectorization(data, lag)

        
#fitting with multiple processes
if use_admm:        
    boot_comm = build_bootstrap_comm(comm, n_admm)
    uoi_lasso = UoI_Lasso(n_real_features = n_features, fit_VAR = True, fit_intercept=False, n_boots_sel=n_boots_sel, n_boots_est=n_boots_est, n_lambdas = n_lambdas, random_state=42, comm = boot_comm, global_comm = comm, n_admm = n_admm, admm_rho = rho, solver='admm', estimation_solver = "admm")
    
    start = time()
    if boot_comm is not None:  #if the global_rank is part of the boostrap distribution(not admm distribution)
        if boot_comm.rank == 0:
            uoi_lasso.fit(lag, data = data)
        else:
            uoi_lasso.fit(lag)
    else:
        if uoi_lasso.solver == "admm":
            uoi_lasso.admm_queue()
    end = time()

else:
    uoi_lasso = UoI_Lasso(n_real_features = n_features, fit_VAR = True, fit_intercept=False, n_boots_sel=n_boots_sel, n_boots_est=n_boots_est, n_lambdas = n_lambdas, random_state=42, comm = comm)    

    start = time()

    if comm.rank == 0:
        uoi_lasso.fit(lag, data = data)
    else:
        uoi_lasso.fit(lag)

    end = time()


if rank == 0:
    print("rank " +str(comm.rank)+": Fitting complete in "+str(end - start)+" seconds.", flush = True)

    #print(np.count_nonzero(uoi_lasso.VAR_coef_))
    if lag == 1:
        B_model = uoi_lasso.VAR_coef_[0]
        B_truth = transition_matrices[0].toarray()
        
        print("Selection accuracy: ", selection_accuracy(B_truth, B_model))
        
        # estimation error
        est_mask = B_model != 0
        print("Estimation error: ", np.linalg.norm(B_truth*est_mask - B_model)**2)


# indicator for whether comparing fitting results from single process fit vs multi-process fit
compare_to_single = False

if compare_to_single:
    #fitting with single process
    if rank == 0:
        uoi_lasso = UoI_Lasso(n_real_features = n_features, fit_VAR = True, fit_intercept = False, random_state=42)
        uoi_lasso.fit(lag, data = data)
        B_model_single = uoi_lasso.VAR_coef_[0]


        print("Selection accuracy: ", selection_accuracy(B_truth, B_model_single))
        # estimation error
        est_mask = B_model != 0
        print("Estimation error: ", np.linalg.norm(B_truth*est_mask - B_model_single)**2)
        

        print(np.allclose(B_model_single, B_model))

        print( np.linalg.norm(B_model_single-B_model))



        

