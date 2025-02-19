#!/usr/bin/env python3
import sys,os
import numpy as np

from mpi4py import MPI
print('M: MPI loaded')
from pyuoi.linear_model import *
sys.path.append(os.path.abspath("../../"))
from examples.var_utils import *


n_features = 20
n_samples = 20
lag = 1


data, transition_matrices, cov = generate_sparse_stationary_var_process(
    n_features,
    n_samples,
    lag=lag,
    sparsity=0.5,
    spectral_radius=0.98,  # Ensures stationarity
    process_type='gaussian',
    random_state = 42

)

dense_matrices = [M.toarray() for M in  transition_matrices]
# vecortized the ground truth transition matrices
B_truth = np.vstack([m.T for m in dense_matrices]).T.flatten()     
X,Y = vectorization(data, lag)


#fitting with single process
uoi_lasso = UoI_Lasso(n_real_features = n_features, fit_VAR = True, random_state=42)
uoi_lasso.fit(X, Y)
B_model = uoi_lasso.coef_


#fitting with multiple processes
comm = MPI.COMM_WORLD

uoi_lasso = UoI_Lasso(n_real_features = n_features, fit_VAR = True, random_state=42,comm = comm)
uoi_lasso.fit(X, Y)
B_model_mpi = uoi_lasso.coef_

print(np.allclose(B_model, B_model_mpi))
