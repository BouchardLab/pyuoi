import numpy as np

from pyuoi.linear_model import *
from var_utils import *
import pdb, h5py, os
from mpi4py import MPI
from time import time
from pyuoi.linear_model.sparse_comm_util import build_bootstrap_comm

# most important hyperparameters!
n_admm =16
eps = 1e-3 # alpha grid min scaler: default is 1e-3
n_lambdas = 48

# n_process should be multiple of n_admm, and at MOST n_admm*n_boot*n_reg_param
#srun -n 960 --ntasks-per-node=240 --distribution=block:block python uoi_var_mpi_check.py

#srun -n 512 --ntasks-per-node=128 --distribution=block:block python uoi_var_mpi_check.py

imbalance_tolerance = 10

n_boots_sel=12
n_boots_est=12

max_iter = 400

selection_frac = 0.9


use_admm = True


rank = 0
comm = MPI.COMM_WORLD
if comm is not None:
    rank = comm.rank

# suffix = "20k"

# if rank == 0:
#     data = np.load("test_jan_admm_"+suffix+".npy")
#     B_truth = np.load("jan_truth.npy")

# 0.9 is the default bootstrap fraction
#assert(data.shape[0]*0.9/n_admm/data.shape[1] > 8, "Reduce n_admm so the n_samp/n_feat per ADMM process is optimal!")
    
seed = 42
lag = 1
n_features = 200
n_samples = 50000
    
if rank == 0:
    # Select spectral radius based on lag
    rad_dict = {1: 0.98, 2: 0.70, 3: 0.30, 4: 0.20, 5: 0.20}
    rad = rad_dict.get(lag, 0.20)


    generate_new_data = True
    
    if generate_new_data:
        data, transition_matrices, bias_terms, covariance_matrix = generate_sparse_stationary_var_process(
            n_features, n_samples, lag=lag, sparsity=0.8, spectral_radius=rad,
            process_type='gaussian', random_state=seed
        )

            
        with h5py.File('data/test_var_admm.h5', 'w') as f:
            g = f.create_group('data')
            g.create_dataset(name='data', data=data, compression="gzip")
            g.create_dataset(name='transition_matrices', data=transition_matrices[0].toarray(), compression="gzip")

        B_truth = transition_matrices[0].toarray()
        np.save("result/var_truth.npy", B_truth)
        np.save("result/var_bias_truth.npy", bias_terms)
    else:
        with h5py.File('data/test_var_admm.h5', 'r') as f:
            data = np.copy(f['data/data'][()])
            B_truth  = np.copy(f['data/transition_matrices'][()])



    # OBSOLETE: used for older version of VAR fit
    # dense_matrices = [M.toarray() for M in  transition_matrices]
    # vecortized the ground truth transition matrices
    # B_truth = np.vstack([m.T for m in dense_matrices]).T.flatten()     
    # X,Y = vectorization(data, lag)

rho_list = 1.1+np.arange(4,5, dtype = int)/10

#fitting with multiple processes
for rho_scaler in rho_list:
    if use_admm:        
        boot_comm = build_bootstrap_comm(comm, n_admm)
        uoi_lasso = UoI_Lasso(fit_VAR = True, fit_intercept=True, n_boots_sel=n_boots_sel, n_boots_est=n_boots_est, selection_frac = selection_frac, n_lambdas = n_lambdas, max_iter = max_iter, eps = eps, random_state=seed, comm = boot_comm, global_comm = comm, n_admm = n_admm, rho_scaler = rho_scaler, imbalance_tolerance = imbalance_tolerance, solver='admm', estimation_solver = "ls")
        
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
        uoi_lasso = UoI_Lasso(fit_VAR = True, fit_intercept=False, n_boots_sel=n_boots_sel, n_boots_est=n_boots_est, selection_frac = selection_frac, n_lambdas = n_lambdas,  max_iter = max_iter, random_state=seed, comm = comm)    
    
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
            model_bias = uoi_lasso.VAR_bias_
            
            np.save("result/var_model_"+str(rho_scaler)+".npy", B_model)
            np.save("result/var_model_bias_"+str(rho_scaler)+".npy", model_bias)


            TP, FP, TN, FN = matrix_comparison(B_truth, B_model, threshold=0)
            # these two adds up == real sparsity in B_truth
            print("TP: ", TP)
            print("FN: ", FN)
            
            print("TN: ", TN)
            print("FP: ", FP)            
            
            print("Selection accuracy: ", selection_accuracy(B_truth, B_model))
            print("Bias sparsity: ", np.count_nonzero(model_bias)/model_bias.shape[0])
            
            # estimation error
            est_mask = B_model != 0
            print("Estimation error: ", np.linalg.norm(B_truth*est_mask - B_model)**2)

            print(str(rho_scaler)+" solution sparsity: ", np.count_nonzero(B_model)/B_model.shape[0]**2)
            print(uoi_lasso.intercept_)

    

# indicator for whether comparing fitting results from single process fit vs multi-process fit
compare_to_single = False

if compare_to_single:
    #fitting with single process
    if rank == 0:
        uoi_lasso = UoI_Lasso(fit_VAR = True, fit_intercept = False, random_state=seed)
        uoi_lasso.fit(lag, data = data)
        B_model_single = uoi_lasso.VAR_coef_[0]


        print("Selection accuracy: ", selection_accuracy(B_truth, B_model_single))
        # estimation error
        est_mask = B_model != 0
        print("Estimation error: ", np.linalg.norm(B_truth*est_mask - B_model_single)**2)
        

        print(np.allclose(B_model_single, B_model))

        print( np.linalg.norm(B_model_single-B_model))



        

