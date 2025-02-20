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

import cProfile
import pstats
from pstats import SortKey

from sklearn.linear_model import LinearRegression, LassoCV

from pyuoi.linear_model import *
from pyuoi.datasets import make_linear_regression

sys.path.append(os.path.abspath("../../"))
from examples.var_utils import *

#...!...!....................
def make_gauss_data( n_features, n_samples, lag, radius   ):

    data, transition_matrices, cov = generate_sparse_stationary_var_process(
        n_features,
        n_samples,
        lag=lag,
        sparsity=0.5,
        spectral_radius=radius,  # Ensures stationarity
        process_type='gaussian'
        
    )
    print('process_type=gaussian')
    return data, transition_matrices

    
#=================================
#  M A I N
#=================================
if __name__ == "__main__":
    np.set_printoptions(precision=3)
    n_features = 10;    n_samples = 1000;
    lag = 1; radius=0.98
    lag = 2; radius=0.7
    lag = 5; radius=0.5
    

    data, transition_matrices=make_gauss_data( n_features, n_samples, lag,radius)
    
    
    dense_matrices = [M.toarray() for M in  transition_matrices]

    # vecortized the ground truth transition matrices
    B_truth = np.vstack([m.T for m in dense_matrices]).T.flatten()     
    X,Y = vectorization(data, lag)

    print('X:',X.shape)
    print('Y:',Y.shape)

    uoi_var = UoI_Lasso(n_real_features = n_features, fit_VAR = True)

    #... slow process starts here
    # Profile a block of code using context manager
    with cProfile.Profile() as pr:
        uoi_var.fit(X, Y)
    # After the context manager exits, analyze with pstats
    ps = pstats.Stats(pr)
    ps.strip_dirs()  # Remove directory paths for cleaner output

    # Different sorting methods for analysis
    #1ps.sort_stats(SortKey.CUMULATIVE).print_stats(10)  # Sort by cumulative time
    ps.sort_stats(SortKey.TIME).print_stats(10)        # Sort by internal time
    ps.sort_stats(SortKey.CALLS).print_stats(10)       # Sort by call count

    # Save results to file for later analysis
    pr.dump_stats('profile_output.prof')

    exit(0)
    B_model = uoi_var.coef_
    print('B_model:',B_model)

    acc=selection_accuracy(B_truth, B_model)
    print('accuracy:',acc)

    # estimation error
    est_mask = B_model != 0
    norm=np.linalg.norm(B_truth*est_mask - B_model)**2
    print('norm:',norm)

    print()
