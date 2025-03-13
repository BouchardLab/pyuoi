#!/usr/bin/env python3
#srun -n256 shifter --image $IMG ./uoi_var_mpi_checkBig.py

import sys,os
#print('M: MPI loaded')
from pyuoi.linear_model import *
sys.path.append(os.path.abspath("../../"))
from examples.var_utils import *

import numpy as np

#from pyuoi.linear_model import *
#from var_utils import *

from mpi4py import MPI
from time import time
# boolean for whether comparing fitting result from single vs multiple process
compare = False

n_features = 25
n_samples = 1000
#lag = 1 ; rad=0.98
lag = 2 ; rad=0.70
lag = 3 ; rad=0.30
lag = 4 ; rad=0.20
#lag = 5 ; rad=0.20

data, transition_matrices, cov = generate_sparse_stationary_var_process(
    n_features,
    n_samples,
    lag=lag,
    sparsity=0.5,
    spectral_radius=rad,
    process_type='gaussian',
    random_state = 42

)

comm = MPI.COMM_WORLD
num_ranks = comm.Get_size()
if  comm.rank == 0 :
    print('data generated nFeat=%d, nSamp=%d  lag=%d numRanks=%d'%(n_features,n_samples,lag,num_ranks), flush = True)
dense_matrices = [M.toarray() for M in  transition_matrices]
# vecortized the ground truth transition matrices
B_truth = np.vstack([m.T for m in dense_matrices]).T.flatten()     
X,Y = vectorization(data, lag)

if compare:
    #fitting with single process
    uoi_lasso = UoI_Lasso(n_real_features = n_features, fit_VAR = True, random_state=42)
    uoi_lasso.fit(X, Y)
    B_model = uoi_lasso.coef_


#fitting with multiple processes

uoi_lasso = UoI_Lasso(n_real_features = n_features, fit_VAR = True, random_state=42,comm = comm)


start = time()

uoi_lasso.fit(X, Y)
end = time()
B_model_mpi = uoi_lasso.coef_

if  comm.rank == 0 :
    print("Fitting complete in %.1f sec  numRank=%d"%((end - start),num_ranks), flush = True)

if compare:
    print(np.allclose(B_model, B_model_mpi))


