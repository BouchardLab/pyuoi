import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy import sparse
from scipy.linalg import solve_discrete_lyapunov

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

if __name__ == '__main__':
    # Example usage

    n_features = 10
    n_samples = 1000
    lag = 1
    
    
    data, transition_matrices, cov = generate_sparse_stationary_var_process(
        n_features,
        n_samples,
        lag=lag,
        sparsity=0.5,
        spectral_radius=0.6,  # Ensures stationarity
        process_type='gaussian'
    
    )

    print(transition_matrices[0].A)