#!/usr/bin/env python3

"""
Based on https://github.com/BouchardLab/pyuoi/blob/uoi-var/examples/UoI_VAR.ipynb

.. _uoi_lasso:

UoI-Lasso for sparse, minimal bias, regression
=============================r[i================

This example with demonstrate the ability of UoI-Lasso to recover sparse
models with minimal bias.

"""

###############################################################################
# Load synthetic data
# -------------------
#
# The synthetic data will have 40 features, 10 of which are informative and
# 1 response variable.

import sys,os
import numpy as np

from sklearn.linear_model import LinearRegression, LassoCV

from pyuoi.linear_model import *
from pyuoi.datasets import make_linear_regression

sys.path.append(os.path.abspath("../../"))
from examples.var_utils import *

#...!...!....................
def make_gauss_data( n_features, n_samples, lag   ):

    data, transition_matrices, cov = generate_sparse_stationary_var_process(
        n_features,
        n_samples,
        lag=lag,
        sparsity=0.5,
        spectral_radius=0.98,  # Ensures stationarity
        process_type='gaussian'
        
    )
    print('process_type=gaussian')
    return data, transition_matrices

    
#=================================
#  M A I N
#=================================
if __name__ == "__main__":
    np.set_printoptions(precision=3)
    n_features = 10;    n_samples = 1000;    lag = 1;
    maxIter=1000
    fitTol=1e-4
    
    data, transition_matrices=make_gauss_data( n_features, n_samples, lag)
    print('data:',data.shape,data.dtype)
    
    dense_matrices = [M.toarray() for M in  transition_matrices]

    # vecortized the ground truth transition matrices
    B_truth = np.vstack([m.T for m in dense_matrices]).T.flatten()     
    X,Y = vectorization(data, lag)

    print('X:',X.shape)
    print('Y:',Y.shape)

    uoi_var = UoI_Lasso(n_real_features = n_features, fit_VAR = True, max_iter=maxIter, tol=fitTol)
    uoi_var.fit(X, Y)
    B_model = uoi_var.coef_
    print('B_model:',B_model)

    acc=selection_accuracy(B_truth, B_model)
    print('accuracy:',acc)

    # estimation error
    est_mask = B_model != 0
    norm=np.linalg.norm(B_truth*est_mask - B_model)**2
    print('norm:',norm)

    print()
