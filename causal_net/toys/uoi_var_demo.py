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

#...!...!....................
def make_poisson_data( n_features, n_samples, lag   ):
    base_intensity = np.random.randint(1, 10, n_features).astype(np.float64)  # Base rates for each feature

    data, transition_matrices, _ = generate_sparse_stationary_var_process(
        n_features,
        n_samples,
        lag=lag,
        sparsity=0.5,
        quench_factor = 1, 
        spectral_radius=0.9,  # Ensures stationarity
        process_type='poisson',
        base_intensity=base_intensity
    )
    print('process_type=poisson')
    return data, transition_matrices
    
#=================================
#  M A I N
#=================================
if __name__ == "__main__":
    np.set_printoptions(precision=3)
    n_features = 10;    n_samples = 20;    lag = 1;

    #data, transition_matrices=make_gauss_data( n_features, n_samples, lag)
    data, transition_matrices=make_poisson_data( n_features, n_samples, lag)    
    
    dense_matrices = [M.toarray() for M in  transition_matrices]

    # vecortized the ground truth transition matrices
    B_truth = np.vstack([m.T for m in dense_matrices]).T.flatten()     
    X,Y = vectorization(data, lag)

    print('X:',X.shape)
    print('Y:',Y.shape)

    uoi_var = UoI_Lasso(n_real_features = n_features, fit_VAR = True)
    uoi_var.fit(X, Y)
    B_model = uoi_var.coef_
    print('B_model:',B_model)

    acc=selection_accuracy(B_truth, B_model)
    print('accuracy:',acc)

    # estimation error
    est_mask = B_model != 0
    norm=np.linalg.norm(B_truth*est_mask - B_model)**2
    print('norm:',norm)

    # have to set fit_intercept = False since the curretn VAR implementation doesn consider constant model terms
    poisson = UoI_Poisson(n_real_features = n_features, fit_VAR =True, max_iter=2500, fit_intercept=False)

    poisson.fit(X, Y)

    
