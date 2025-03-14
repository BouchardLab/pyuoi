#!/usr/bin/env python3
#srun -n256 shifter  ./uoi_mpi_checkBig.py

import sys,os
#print('M: MPI loaded')
from pyuoi.linear_model import *
sys.path.append(os.path.abspath("../../"))
from examples.var_utils import *

from pprint import pprint
import numpy as np
from mpi4py import MPI
from time import time

def  analyze_sparsity(A):
    """
    Analyzes the sparsity and value distribution of a square NumPy array A.

    Returns:
        - Row-wise and column-wise statistics nested in a structured dictionary
        - Global statistics (median, percentiles, sum of absolute values, etc.)
    """
    assert A.shape[0] == A.shape[1], "Matrix A must be square."

    n = A.shape[0]  # Number of rows/columns
    
    # Sparsity per row (fraction of zero elements)
    row_sparsity = np.sum(A == 0, axis=1) / n

    # Sparsity per column (fraction of zero elements)
    col_sparsity = np.sum(A == 0, axis=0) / n

    # Extract nonzero values
    nonzero_values = A[A != 0]

    # Compute median and 95th percentile of nonzero values
    median_value = np.median(nonzero_values) if nonzero_values.size > 0 else 0
    percentile_95 = np.percentile(nonzero_values, 95) if nonzero_values.size > 0 else 0

    # Count positive and negative values per row
    pos_per_row = np.sum(A > 0, axis=1)
    neg_per_row = np.sum(A < 0, axis=1)

    # Count positive and negative values per column
    pos_per_col = np.sum(A > 0, axis=0)
    neg_per_col = np.sum(A < 0, axis=0)

    # Compute polarization per row: (n+ - n-)/(n+ + n-), handle division by zero
    polarization_row = np.where(
        (pos_per_row + neg_per_row) > 0,
        (pos_per_row - neg_per_row) / (pos_per_row + neg_per_row),
        0
    )

    # Compute polarization per column: (n+ - n-)/(n+ + n-), handle division by zero
    polarization_col = np.where(
        (pos_per_col + neg_per_col) > 0,
        (pos_per_col - neg_per_col) / (pos_per_col + neg_per_col),
        0
    )

    # Compute average polarization per row and per column
    avg_polarization_row = np.mean(polarization_row)
    avg_polarization_col = np.mean(polarization_col)

    # Compute sum of absolute values of all elements
    sum_abs_values = np.sum(np.abs(A))

    # Compute average absolute value over nonzero elements
    avg_abs_nonzero = np.mean(np.abs(nonzero_values)) if nonzero_values.size > 0 else 0

    return {
        "per_row": {
            "sparsity": row_sparsity,
            "polarization": polarization_row
        },
        "per_column": {
            "sparsity": col_sparsity,
            "polarization": polarization_col
        },
        "global": {
            "avg_polarization_row": avg_polarization_row,
            "avg_polarization_col": avg_polarization_col,
            "median_nonzero": median_value,
            "percentile_95_nonzero": percentile_95,
            "sum_abs_values": sum_abs_values,
            "avg_abs_nonzero": avg_abs_nonzero
        }
    }


n_features = 25
n_samples = 10000
#lag = 1 ; rad=0.98
lag = 2 ; rad=0.70
lag = 3 ; rad=0.30
#lag = 4 ; rad=0.20
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
num_rank = comm.Get_size()
if  comm.rank == 0 :
    print('data generated nFeat=%d, nSamp=%d  lag=%d numRanks=%d'%(n_features,n_samples,lag,num_rank), flush = True)
dense_matrices = [M.toarray() for M in  transition_matrices]
# vecortized the ground truth transition matrices
B_truth = np.vstack([m.T for m in dense_matrices]).T.flatten()     
X,Y = vectorization(data, lag)


#prep for fitting with multiple processes
uoi_lasso = UoI_Lasso(n_real_features = n_features, fit_VAR = True, random_state=42,comm = comm)


start = time()
uoi_lasso.fit(X, Y)
end = time()


if  comm.rank != 0 : exit(0)

print("Fitting complete in %.1f sec  numRank=%d"%((end - start),num_ranks), flush = True)
B_model = uoi_lasso.coef_
# construct the lagged connectivity matrices from the model coefficient
A_model = [B_model.reshape(n_features,n_features*lag).T[i*n_features:(i+1)*n_features].T for i in range(lag)]
A_model=np.array((A_model))
print('B_model:',B_model.shape, 'A_model:',A_model.shape)
np.set_printoptions(precision=2)  
for il in range(lag):
    result = analyze_sparsity(A_model[il])
    print('\nil=%d'%il);  pprint(result)
    break
    


