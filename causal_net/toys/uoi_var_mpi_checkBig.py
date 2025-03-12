#!/usr/bin/env python3
import sys,os
print('M: MPI loaded')
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

n_features = 60
n_samples = 1000
lag = 1

# srun -n 64 python uoi_var_mpi_check.py

data, transition_matrices, cov = generate_sparse_stationary_var_process(
    n_features,
    n_samples,
    lag=lag,
    sparsity=0.5,
    spectral_radius=0.98,  # Ensures stationarity
    process_type='gaussian',
    random_state = 42

)
print('data generated nFeat=%d, nSamp=%d'%(n_features,n_samples))
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
comm = MPI.COMM_WORLD

uoi_lasso = UoI_Lasso(n_real_features = n_features, fit_VAR = True, random_state=42,comm = comm)


start = time()

uoi_lasso.fit(X, Y)
end = time()
B_model_mpi = uoi_lasso.coef_

if comm.rank == 0:
    print("Fitting complete in "+str(end - start)+" seconds.", flush = True)

if compare:
    print(np.allclose(B_model, B_model_mpi))


