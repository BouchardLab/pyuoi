"""
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


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression, LassoCV

from pyuoi.linear_model import UoI_Lasso
from pyuoi.datasets import make_linear_regression
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import expit  # logistic function
from sklearn.metrics import precision_score, recall_score, f1_score

from scipy import sparse

def vectorization(data, lag):
    # vectorize the VAR data for use with LASSO algorithm
    # data: time series data in np array with shape n_samples X n_features
        
    # flipup so the last time sample in data is now first row
    data = np.flipud(data)
    n_samples = data.shape[0]
    n_features = data.shape[1]

    # shape of Y: (T - D) X (n_features)   *T - D: total number of samples - lag
    Y = data[: n_samples-lag]
    # ones = np.ones((n_samples-lag,1))  # for VAR porocess with intercept
    Y = Y.T.flatten()    
    
    # X.shape: (n_samples - lag) X (lag * n_features); 
    X_row = [np.hstack(data[i : lag+i]) for i in range(1,n_samples-lag+1)]
    X = np.vstack(X_row)
    # use kronecker product for vectorization of matrix multiplication
    X = np.kron(np.eye(n_features), X)    
    
    return X, Y

def vectorization_mbb(data, n_boots):
    # OBSOLETE for moving block boostraps: X_mbb.shape: n_boots X (lag * n_features); 
    
    data = np.flipud(data)
    Y = data[: n_samples-lag]

    data_mbb, boot_idx = mbbs_np(data, n_blocks=n_boots, block_length = lag)

    X_mbb_row = [np.hstack(data_mbb[lag*i : lag*(i+1)]) for i in range(n_boots)]
    X_mbb = np.vstack(X_mbb_row)
    # effectiver number of features is (lag * n_feat * n_feat), i.e. total number of entries in all transition matrices
    X_mbb = np.kron(np.eye(n_features), X_mbb)   

    # moving block bootstrap samples for Y:
    Y_mbb = Y[boot_idx]
    Y_mbb = Y_mbb.T.flatten()  #vectorization
    
    return X_mbb, Y_mbb


def mbbs_df(data, n_blocks=10, block_length=None):
    # number of datapoints
    n = len(data)
    
    # assign a default value for block_length
    if block_length is None:
        block_length = n // n_blocks
        
    # Create empty DataFrame with correct size and columns
    total_rows = n_blocks * block_length
    mbbs_sample = pd.DataFrame(columns=data.columns, index=range(total_rows))
    
    # generate starting indices for each block
    block_indices = np.random.randint(low=0, high=n - block_length, size=n_blocks)
    
    # Create the bootstrap sample
    for i in range(n_blocks):
        start_index = block_indices[i]
        end_index = start_index + block_length
        mbbs_sample.iloc[i * block_length : (i + 1) * block_length] = data.iloc[start_index:end_index].values
        
    return mbbs_sample

def mbbs_np(data, n_blocks=10, block_length=None):
    # data should be a numpy array
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    # number of datapoints
    n = len(data)
    
    # assign a default value for block_length
    if block_length is None:
        block_length = n // n_blocks
    
    # Create empty array with correct size
    total_rows = n_blocks * block_length
    mbbs_sample = np.empty((total_rows,) + data.shape[1:])
    
    # generate starting indices for each block
    block_indices = np.random.randint(low=0, high=n - block_length, size=n_blocks)
    
    # Create the bootstrap sample
    for i in range(n_blocks):
        start_index = block_indices[i]
        end_index = start_index + block_length
        mbbs_sample[i * block_length : (i + 1) * block_length] = data[start_index:end_index]
        
    return mbbs_sample, block_indices





def generate_sparse_stationary_var_process(n_features, n_samples, lag=1, sparsity=0.9, spectral_radius=0.8, quench_factor = 1, 
                                         process_type='gaussian', random_state=None, 
                                         base_intensity=None, base_probability=None):
    """
    Generate data from a stationary Vector Autoregressive (VAR) process with sparse transition matrices.
    
    Parameters:
    -----------
    n_features : int
        Number of features (dimensions) in the VAR process
    n_samples : int
        Number of time points to generate
    lag : int, default=1
        Order of the VAR process (number of past observations to use)
    sparsity : float, default=0.9
        Desired sparsity level (proportion of zero elements) in transition matrices
    spectral_radius : float, default=0.8
        Desired spectral radius of the VAR process (must be < 1 for stationarity)
    process_type : str, default='gaussian'
        Type of VAR process. Either 'gaussian', 'poisson', or 'bernoulli'
    random_state : int or None, default=None
        Random seed for reproducibility
    base_intensity : ndarray or None, default=None
        Base intensity for Poisson process. Required if process_type='poisson'
    base_probability : ndarray or None, default=None
        Base probability for Bernoulli process. Required if process_type='bernoulli'
    
    Returns:
    --------
    tuple:
        - data: ndarray of shape (n_samples, n_features)
        - transition_matrices: list of sparse matrices, each of shape (n_features, n_features)
        - covariance_matrix: ndarray of shape (n_features, n_features), only for Gaussian process
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if process_type not in ['gaussian', 'poisson', 'bernoulli']:
        raise ValueError("process_type must be either 'gaussian', 'poisson', or 'bernoulli'")
    
    if not 0 <= sparsity < 1:
        raise ValueError("sparsity must be between 0 and 1")
        
    if not 0 <= spectral_radius < 1:
        raise ValueError("spectral_radius must be between 0 and 1 for stationarity")
        
    if process_type == 'poisson' and base_intensity is None:
        raise ValueError("base_intensity must be provided for Poisson process")
        
    if process_type == 'bernoulli':
        if base_probability is None:
            raise ValueError("base_probability must be provided for Bernoulli process")
        if not np.all((base_probability >= 0) & (base_probability <= 1)):
            raise ValueError("base_probability values must be between 0 and 1")
    
    def get_companion_spectral_radius(matrices):
        """Calculate spectral radius of the companion matrix"""
        n = matrices[0].shape[0]
        p = len(matrices)
        
        # Convert sparse matrices to dense for companion matrix construction
        dense_matrices = [M.toarray() for M in matrices]
        companion = np.zeros((n * p, n * p))
        
        companion[:n] = np.hstack(dense_matrices)
        if p > 1:
            companion[n:, :-n] = np.eye(n * (p - 1))
            
        eigenvals = np.linalg.eigvals(companion)
        return np.max(np.abs(eigenvals))
    
    def generate_sparse_matrix(n, sparsity):
        """Generate a sparse random matrix with controlled density"""
        # Generate mask for non-zero elements
        mask = np.random.random((n, n)) > sparsity
        
        # Generate random values for non-zero elements
        values = np.random.randn(n, n)
        values = values * mask
        
        return sparse.csr_matrix(values)
    
    # Generate transition matrices with controlled spectral radius
    transition_matrices = []
    
    # First generate matrices without scaling
    for _ in range(lag):
        matrix = generate_sparse_matrix(n_features, sparsity)
        transition_matrices.append(matrix)
    
    # Calculate current spectral radius
    current_radius = get_companion_spectral_radius(transition_matrices)
    
    # Scale matrices to achieve desired spectral radius
    scaling_factor = spectral_radius / current_radius /quench_factor
    transition_matrices = [matrix * scaling_factor for matrix in transition_matrices]
    
    # Verify stationarity
    final_radius = get_companion_spectral_radius(transition_matrices)
    #print(final_radius)
    assert final_radius < 1, "Failed to achieve stationarity"
    
    # Process-specific scaling
    process_scale_factors = {
        'gaussian': 1.0,
        'poisson': 0.6,  # More conservative for count data
        'bernoulli': 0.4  # Even more conservative for binary data
    }
    scale_factor = process_scale_factors[process_type]
    transition_matrices = [matrix * scale_factor for matrix in transition_matrices]
    
    # Initialize process-specific parameters and initial data
    if process_type == 'gaussian':
        # Generate innovation covariance matrix (positive definite)
        A = np.random.randn(n_features, n_features)
        covariance_matrix = A @ A.T + np.eye(n_features)
        
        # Calculate stationary covariance using dense matrices
        A_total = sum(M.toarray() for M in transition_matrices)
        stationary_cov = solve_discrete_lyapunov(A_total, covariance_matrix)
        
        # Initialize data with stationary distribution
        data = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=stationary_cov,
            size=lag
        )
    elif process_type == 'poisson':
        covariance_matrix = None
        data = np.random.poisson(lam=base_intensity, size=(lag, n_features))
    else:  # Bernoulli
        covariance_matrix = None
        data = np.random.binomial(n=1, p=base_probability, size=(lag, n_features))
    
    # Generate the rest of the time series
    for t in range(lag, n_samples):
        if process_type == 'gaussian':
            new_point = np.zeros(n_features)
        elif process_type == 'poisson':
            new_point = base_intensity.copy()
        else:  # Bernoulli
            new_point = np.zeros(n_features)
        
        # Add contribution from each lag using sparse matrix multiplication
        for i in range(lag):
            new_point += transition_matrices[i].dot(data[t - i - 1])
            
        if process_type == 'gaussian':
            new_point += np.random.multivariate_normal(
                mean=np.zeros(n_features),
                cov=covariance_matrix
            )
        elif process_type == 'poisson':
            new_point = np.maximum(new_point, 0)
            new_point = np.random.poisson(lam=new_point)
        else:  # Bernoulli
            new_point += np.log(base_probability / (1 - base_probability))
            probabilities = expit(new_point)
            new_point = np.random.binomial(n=1, p=probabilities)
        
        data = np.vstack([data, new_point])
    
    return data, transition_matrices, covariance_matrix

def print_process_info(data, transition_matrices, covariance_matrix):
    """
    Print information about the generated VAR process.
    
    Parameters:
    -----------
    data : ndarray
        The generated time series data
    transition_matrices : list of ndarrays
        The transition matrices of the VAR process
    covariance_matrix : ndarray
        The innovation covariance matrix
    """
    print(f"Data shape: {data.shape}")
    print("\nTransition matrices:")
    for i, matrix in enumerate(transition_matrices):
        print(f"\nLag {i+1}:")
        print(matrix)
    print("\nInnovation covariance matrix:")
    print(covariance_matrix)
    
    # Print some basic statistics
    print("\nData statistics:")
    print(f"Mean:\n{np.mean(data, axis=0)}")
    print(f"\nEmpirical covariance:\n{np.cov(data.T)}")
    
    
    



def evaluate_sparse_var_estimation(true_matrices, estimated_matrices, threshold=1e-5):
    """
    Compute various metrics for evaluating sparse VAR estimation accuracy.
    
    Parameters:
    -----------
    true_matrices : list of sparse matrices or ndarrays
        True transition matrices
    estimated_matrices : list of sparse matrices or ndarrays
        Estimated transition matrices
    threshold : float, default=1e-5
        Threshold for considering an element as non-zero
        
    Returns:
    --------
    dict:
        Dictionary containing various evaluation metrics
    """
    metrics = {}
    
    # Convert to dense if needed for calculations
    true_dense = [M.toarray() if sparse.issparse(M) else M for M in true_matrices]
    est_dense = [M.toarray() if sparse.issparse(M) else M for M in estimated_matrices]
    
    # Combine all matrices for overall metrics
    true_combined = np.concatenate([M.flatten() for M in true_dense])
    est_combined = np.concatenate([M.flatten() for M in est_dense])
    
    # Create binary masks for sparsity pattern
    true_mask = np.abs(true_combined) > threshold
    est_mask = np.abs(est_combined) > threshold
    
    # 1. Sparsity Pattern Metrics
    metrics['precision'] = precision_score(true_mask, est_mask)
    metrics['recall'] = recall_score(true_mask, est_mask)   #The recall is intuitively the ability of the classifier to find all the positive samples.
    metrics['f1'] = f1_score(true_mask, est_mask) 
    
    # 2. Relative Frobenius Error
#     frob_errors = []
#     for true_M, est_M in zip(true_dense, est_dense):
#         frob_error = np.linalg.norm(true_M - est_M, 'fro') / np.linalg.norm(true_M, 'fro')
#         frob_errors.append(frob_error)
#     metrics['relative_frobenius'] = np.mean(frob_errors)
    
    # 3. Support Recovery Error
    support_errors = []
    for true_M, est_M in zip(true_dense, est_dense):
        true_supp = np.abs(true_M) > threshold
        est_supp = np.abs(est_M) > threshold
        support_error = np.sum(true_supp != est_supp) / true_M.size
        support_errors.append(support_error)
    metrics['support_error'] = np.mean(support_errors)
    
    # 4. Element-wise Metrics for Non-zero Elements
    true_nonzero = true_combined[true_mask]
    est_nonzero = est_combined[true_mask]
    
    if len(true_nonzero) > 0:
        metrics['mae_nonzero'] = np.mean(np.abs(true_nonzero - est_nonzero))
        metrics['rmse_nonzero'] = np.sqrt(np.mean((true_nonzero - est_nonzero)**2))
        metrics['mape_nonzero'] = np.mean(np.abs((true_nonzero - est_nonzero) / true_nonzero)) * 100
    
    # 5. Sparsity Level Comparison
    metrics['true_sparsity'] = 1 - np.mean(true_mask)
    metrics['estimated_sparsity'] = 1 - np.mean(est_mask)
    
    return metrics

def visualize_matrix_comparison(true_matrix, estimated_matrix, threshold=1e-5):
    """
    Create visualization comparing true and estimated matrices.
    Returns plotting data that can be used with your preferred visualization library.
    """
    # Convert to dense if sparse
    true_dense = true_matrix.toarray() if sparse.issparse(true_matrix) else true_matrix
    est_dense = estimated_matrix.toarray() if sparse.issparse(estimated_matrix) else estimated_matrix
    
    # Create masks for zero/nonzero elements
    true_mask = np.abs(true_dense) > threshold
    est_mask = np.abs(est_dense) > threshold
    
    # Categorize elements
    comparison = {
        'TP': np.logical_and(true_mask, est_mask),
        'FP': np.logical_and(~true_mask, est_mask),
        'FN': np.logical_and(true_mask, ~est_mask),
        'TN': np.logical_and(~true_mask, ~est_mask),
        'error_magnitude': np.abs(true_dense - est_dense)
    }
    
    return comparison

# Example usage:
def example_usage():
    # Generate example matrices
    n_features = 10
    true_matrices = [
        sparse.random(n_features, n_features, density=0.1).toarray() * 0.5
        for _ in range(2)  # VAR(2) process
    ]
    
    # Add some noise to create estimated matrices
    estimated_matrices = [
        M + np.random.normal(0, 0.1, M.shape) 
        for M in true_matrices
    ]
    
    # Compute metrics
    metrics = evaluate_sparse_var_estimation(true_matrices, estimated_matrices)
    
    # Print results
    print("\nEvaluation Metrics:")
    print("------------------")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics


def selection_accuracy(B_truth, B_model):
    comparison = visualize_matrix_comparison(B_truth, B_model, threshold=0)
    FN = np.count_nonzero(comparison["FN"])
    FP = np.count_nonzero(comparison["FP"])
    M = np.count_nonzero(B_truth)
    return 1-(FN+FP)/(M+FP) 



#stable transition matrices check:
def stability_check(dense_matrices):
    n_features = dense_matrices[0].shape[0]
    for _ in range(1000):
        mat = np.eye(n_features)
        for A in dense_matrices:
            z = np.random.rand(n_features)
            z = z/(np.linalg.norm(z)+10)
            mat-=A@z
        assert(np.linalg.det(mat)!=0)

