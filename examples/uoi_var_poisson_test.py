# Division is default behavior in Python 3, so this import is no longer needed
# from __future__ import division
import pdb, h5py, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from numpy.linalg import norm
import importlib
from sparse_randn import sprandn
from mpi4py import MPI
from time import time

from sklearn.linear_model import LinearRegression

#from pyuoi.linear_model.admm_mpi_poisson import *



from pyuoi.linear_model import *
from var_utils import *
from pyuoi.linear_model.sparse_comm_util import build_bootstrap_comm

# most important hyperparameters!
n_admm = 32
n_lambdas = 4  
dt = 0.01  # sample time bin size for Poisson process
manual_l1_range = [6e-7, 3e-6]  # Hardcoded L1 penalty range
fdr_rate = 0.05

eps = 1e-9 # Obsolete if using manual_l1_rage. alpha grid min scaler: default is 1e-3
stability_selection=0.75  #Obsolete, this was for the occurence based parameter support seleciton method


# n_process should be multiple of n_admm, and at MOST n_admm*n_boot*n_reg_param
# Example run command on NERSC interactive compute node session:
# srun -n 768 --ntasks-per-node=192 --distribution=block:block python uoi_var_poisson_test.py

imbalance_tolerance = 10

n_boots_sel=6  
n_boots_est=6  

max_iter = 1000
seed = 22
selection_frac = 0.9  #Fraction of total data used for each bootstrap: default is 0.9

use_admm = True
generate_new_data = False

# Example usage
# n_features = 20
# n_samples = 20000
lag = 1

rank = 0
comm = MPI.COMM_WORLD
if comm is not None:
    rank = comm.rank


# if rank == 0:


#     if generate_new_data:
    
    
#         base_intensity = np.random.randint(1, 10, n_features).astype(np.float64)  # Base rates for each feature
        
#         data, transition_matrices,  bias_terms, _ = generate_sparse_stationary_var_process(
#             n_features,
#             n_samples,
#             lag=lag,
#             sparsity=0.8,
#             quench_factor = 1, 
#             spectral_radius=0.9,  # Ensures stationarity
#             process_type='poisson',
#             base_intensity=base_intensity
        
#         )    
    
#         B_truth = transition_matrices[0].toarray()
#         # np.save("result/poisson_truth.npy", B_truth)
#         # np.save("result/poisson_bias_truth.npy", bias_terms)    
#         # np.save("data/test_poisson_admm.npy", data)
    
#     else:
#             B_truth = np.load("data/poisson_truth.npy", allow_pickle=True)
#             data = np.load("data/test_poisson_admm.npy")
       
#     data_pois = None
    
# if rank == 0:       
#     with h5py.File('data/daleM40-a03d69a-c1edd9d.spike.h5', 'r') as f:
#         #spike = np.copy(f['spikes_data'][()])
#         data  = np.copy(f['stateVec_data'][()]).T[:15000].astype("float32")
#         B_truth  = np.copy(f['true_network_matrix'][()]).astype("float32")
    
#     data_pois = np.load("data/data_pois.npy").T[:15000]





if rank == 0: 

    # data = np.load('data/daleM120_12dff3.spikes.npz')['spikes'][:1000000].astype(np.double)
    # B_truth = np.load('data/daleM120_12dff3.truth.npz')["A_true"]

    # M20_746c4b
    # M40_e33e89
    # M80_285c84
    # M150_448b86

    data = np.load('/pscratch/sd/y/yxu2/data/daleM80_285c84.spikes.npz')['spikes'][:10000].astype(np.double)
    B_truth = np.load('/pscratch/sd/y/yxu2/data/daleM80_285c84.simTruth.npz')["A_true"]
    data_pois = None

    # w = 1/np.maximum(np.mean(data,axis = 0), 0.1*np.ones(data.shape[1]))
    # w = w/np.linalg.norm(w) * data.shape[1]
    w = np.ones(data.shape[1])
else:
    w = None

w = comm.bcast(w, root=0) 



rho_list = 1.5+np.arange(0,1, dtype = int)/10

#np.arange(3.759,3.761,0.0005)

for l1_suppression in np.arange(1):#(14,15):
    #fitting with multiple processes
    for rho_scaler in rho_list:
        if use_admm:        
            boot_comm = build_bootstrap_comm(comm, n_admm)
            uoi_poisson = UoI_Poisson(fit_VAR = True, fit_intercept=False, standardize = False, manual_l1_range = manual_l1_range, n_boots_sel=n_boots_sel, n_boots_est=n_boots_est, selection_frac = selection_frac, stability_selection=stability_selection, n_lambdas = n_lambdas, max_iter = max_iter, eps = eps, random_state=seed, comm = boot_comm, global_comm = comm, n_admm = n_admm, rho_scaler = rho_scaler, imbalance_tolerance = imbalance_tolerance, l1_suppression= l1_suppression, solver='admm', estimation_solver = "lbfgs", weights = w, dt = dt, fdr_rate = fdr_rate)
            
            start = time()
            if boot_comm is not None:  #if the global_rank is part of the boostrap distribution(not admm distribution)
                if boot_comm.rank == 0:
                    
                    uoi_poisson.fit(lag, data = data, data_pois = data_pois)
                else:
                    uoi_poisson.fit(lag)
            else:
                if uoi_poisson.solver == "admm":
                    uoi_poisson.admm_queue()
            end = time()
        else:
            uoi_poisson = UoI_Poisson(fit_VAR = True, fit_intercept=False, standardize = False, manual_l1_range = manual_l1_range, n_boots_sel=n_boots_sel, n_boots_est=n_boots_est, selection_frac = selection_frac, stability_selection=stability_selection, n_lambdas = n_lambdas, max_iter = max_iter, eps = eps, random_state=seed, comm = comm, rho_scaler = rho_scaler, imbalance_tolerance = imbalance_tolerance, l1_suppression= l1_suppression, weights = w, dt = dt, fdr_rate = fdr_rate)
            
            start = time()
           
            if comm.rank == 0:
                uoi_poisson.fit(lag, data = data, data_pois = data_pois)
            else:
                uoi_poisson.fit(lag)
    
            end = time()
        
        
        if rank == 0:
            print("l1_suppression: ", l1_suppression, flush = True)
            print("rank " +str(comm.rank)+": Fitting complete in "+str(end - start)+" seconds.", flush = True)
        
            #print(np.count_nonzero(uoi_poisson.VAR_coef_))
            if lag == 1:
                B_model = uoi_poisson.VAR_coef_[0]
                model_bias = uoi_poisson.VAR_bias_

                # print("L1-loss: ", uoi_poisson.loss["l1"])
                
                np.save("result/poisson_M80_"+str(rho_scaler)+"_intersect.npy", B_model)
                np.save("result/poisson_M80_bias_"+str(rho_scaler)+"_intersect.npy", model_bias)
                
                TP, FP, TN, FN = matrix_comparison(B_truth, B_model, threshold=0)
                # these two adds up == real sparsity in B_truth
                print("TP: ", TP)
                print("FN: ", FN)
                
                print("TN: ", TN)
                print("FP: ", FP)
                
                # estimation error
                est_mask = B_model != 0
                print("Estimation error: ", np.linalg.norm(B_truth*est_mask - B_model)**2)
    
                print(str(rho_scaler)+" solution sparsity: ", np.count_nonzero(B_model)/B_model.shape[0]**2)

                print("Bias sparsity: ", np.count_nonzero(model_bias)/model_bias.shape[0])
        














    
    


