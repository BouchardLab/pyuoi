import numpy as np
from statsmodels.stats.multitest import fdrcorrection

def stability_selection_to_threshold(stability_selection, n_boots):
    """Converts user inputted stability selection to an array of
    thresholds. These thresholds correspond to the number of bootstraps
    that a feature must appear in to guarantee placement in the selection
    profile.

    Parameters
    ----------
    stability_selection : int, float, or array-like
        If int, treated as the number of bootstraps that a feature must
        appear in to guarantee placement in selection profile. If float,
        must be between 0 and 1, and is instead the proportion of
        bootstraps. If array-like, must consist of either ints or floats
        between 0 and 1. In this case, each entry in the array-like object
        will act as a separate threshold for placement in the selection
        profile.

    n_boots: int
        The number of bootstraps that will be used for selection
    """

    # single float, indicating proportion of bootstraps
    if isinstance(stability_selection, float):
        selection_thresholds = np.array([int(
            stability_selection * n_boots
        )])

    # single int, indicating number of bootstraps
    elif isinstance(stability_selection, int):
        selection_thresholds = np.array([int(
            stability_selection
        )])

    # list, to be converted into numpy array
    elif isinstance(stability_selection, list):
        # list of floats
        if all(isinstance(idx, float) for idx in stability_selection):
            selection_thresholds = \
                n_boots * np.array(stability_selection)

        # list of ints
        elif all(isinstance(idx, int) for idx in stability_selection):
            selection_thresholds = np.array(stability_selection)

        else:
            raise ValueError("Stability selection list must consist of "
                             "floats or ints.")

    # numpy array
    elif isinstance(stability_selection, np.ndarray):
        # np array of floats
        if np.issubdtype(stability_selection.dtype.type, np.floating):
            selection_thresholds = n_boots * stability_selection

        # np array of ints
        elif np.issubdtype(stability_selection.dtype.type, np.integer):
            selection_thresholds = stability_selection

        else:
            raise ValueError("Stability selection array must consist of "
                             "floats or ints.")

    else:
        raise ValueError("Stability selection must be a valid float, int "
                         "or array.")

    # ensure that ensuing list of selection thresholds satisfies
    # the correct bounds
    selection_thresholds = selection_thresholds.astype('int')
    if not (
        np.all(selection_thresholds <= n_boots) and
        np.all(selection_thresholds >= 1)
    ):
        raise ValueError("Stability selection thresholds must be within "
                         "the correct bounds.")

    return selection_thresholds


def intersection(coefs, selection_thresholds=None,  magnitude_threshold=0.01):
    """Performs the intersection operation on selection coefficients
    using stability selection criteria.

    The coefficients must be provided in the shape
        bootstraps x lambdas x features.
    The intersection operation finds, for each lambda, the features that
    exist in all bootstraps (hard intersection) or in some subset of them
    (the exact subset is provided by selection_thresholds).

    This parameter selection_thresholds provides the number of bootstraps
    that a feature must exist in to pass the intersection. Importantly,
    this function can take intersections with multiple selection_thresholds
    (thus, selection_thresholds is array-like).

    This function then outputs an array of supports, each as a binary mask.
    Only unique supports are provided, so duplicates are tossed out.

    Parameters
    ----------
    coefs : np.ndarray, shape (# bootstraps, # lambdas, # features)
        The coefficients obtained from the selection sweep, corresponding to
        each bootstrap and choice of L1 regularization strength.

    selection_thresholds: array-like, int
        The selection thresholds to perform intersection across. By default,
        use *coefs.shape[0]*.

    Returns
    -------
    supports : np.ndarray, shape (# supports, # features), bool
        A list of supports (each as a binary mask with size n_features)
        obtained by performing the intersection across the coefficients. Each
        support is unique.
    """

    if selection_thresholds is None:
        selection_thresholds = np.array([coefs.shape[0]])

    n_selection_thresholds = len(selection_thresholds)
    n_reg_params = coefs.shape[1]
    n_features = coefs.shape[2]
    supports = np.zeros(
        (n_selection_thresholds, n_reg_params, n_features),
        dtype=bool
    )

    # Count how many bootstraps have |coefficient| >= magnitude_threshold
    # for each lambda and feature
    significant_selections = np.sum(
        np.abs(coefs) > magnitude_threshold, 
        axis=0
    )


    # significant_selections = np.count_nonzero(coefs, axis=0)
    

    # iterate over each stability selection threshold
    for thresh_idx, threshold in enumerate(selection_thresholds):
        # calculate the support given the specific selection threshold
        supports[thresh_idx, ...] = significant_selections >= threshold

    # unravel the dimension corresponding to selection thresholds

    supports = np.squeeze(np.reshape(
        supports,
        (n_selection_thresholds * n_reg_params, n_features)
    ))

    supports = np.unique(supports, axis=0)

    return supports

def intersection_dirty(coefs, selection_thresholds=None, n_sig=2.5, min_w=0.01):
    """Performs intersection operation using both minimum weight and SNR criteria.
    
    A feature is considered "selected" only if:
    1. Its upper confidence bound (mean + n_sig*std) exceeds min_w (not negligible)
    2. Its signal-to-noise ratio (mean/std) exceeds n_sig (consistent)
    
    Parameters
    ----------
    coefs : np.ndarray, shape (# bootstraps, # lambdas, # features)
        The coefficients obtained from the selection sweep.
    
    selection_thresholds: array-like, int
        The selection thresholds to perform intersection across.
    
    n_sig : float, default=2.5
        Signal-to-noise ratio threshold and confidence interval multiplier.
        Features must have |mean|/std > n_sig AND |mean| + n_sig*std > min_w.
    
    min_w : float, default=0.05
        Minimum weight threshold. Features with upper confidence bound
        below this are considered negligible regardless of SNR.
    
    Returns
    -------
    supports : np.ndarray, shape (# supports, # features), bool
        Unique supports obtained by performing the intersection.
    """
    
    if selection_thresholds is None:
        selection_thresholds = np.array([1])
    
    n_selection_thresholds = len(selection_thresholds)
    n_reg_params = coefs.shape[1]
    n_features = coefs.shape[2]
    supports = np.zeros(
        (n_selection_thresholds, n_reg_params, n_features),
        dtype=bool
    )
    
    # Calculate statistics across bootstraps
    coef_mean = np.mean(coefs, axis=0)  # Shape: (n_lambdas, n_features)
    coef_std = np.std(coefs, axis=0)    # Shape: (n_lambdas, n_features)
    coef_mean_abs = np.abs(coef_mean)
    
    # Apply dual criteria for each lambda and feature
    for lambda_idx in range(n_reg_params):
        for feature_idx in range(n_features):
            xmean = coef_mean_abs[lambda_idx, feature_idx]
            xstd = coef_std[lambda_idx, feature_idx]
            
            # Condition 1: Check if upper confidence bound exceeds minimum weight
            # This filters out consistently small coefficients
            if xmean + n_sig * xstd < min_w:
                continue
            
            # Condition 2: Check signal-to-noise ratio
            # This filters out inconsistent coefficients
            if xstd > 1e-10:  # Avoid division by zero
                if xmean / xstd < n_sig:
                    continue
            else:
                # If std is essentially zero, only check mean against min_w
                if xmean < min_w:
                    continue
            
            # Both conditions passed - mark as selected
            for thresh_idx in range(n_selection_thresholds):
                supports[thresh_idx, lambda_idx, feature_idx] = True
    
    # Unravel and get unique supports
    supports = np.squeeze(np.reshape(
        supports,
        (n_selection_thresholds * n_reg_params, n_features)
    ))
    
    supports = np.unique(supports, axis=0)
    
    return supports 

def ie_type(coef, lag, num_feat, scheme = 2):
    A_model = coef.reshape(coef.shape[:-1] + (num_feat,lag,num_feat)) # Reshape last axis from to the connectivity matrix
    # shape: n_boot * n_reg_param * lag * n_feat * n_feat
    A_model = np.transpose(A_model, (0,1,3,2,4))   #switch the axis on the lag and n_feat(because we column stacked)
    
    # shape:  lag * n_boot * n_reg_param * n_feat * n_feat (transpose is so useful!!!)
    A_model = np.transpose(A_model, (2,0,1,3,4)) 

    p_count = np.sum(A_model > 0, axis=-1)
    n_count = np.sum(A_model < 0, axis=-1)    

    if scheme == 1:
        # scheme #1: decision on aggregated counts
        for i in range(lag):
            p_count_sum =  np.sum(p_count[i], axis=tuple(range(p_count[i].ndim-1)))
            n_count_sum =  np.sum(n_count[i], axis=tuple(range(n_count[i].ndim-1)))
            
            node_type =  2 * (p_count_sum > n_count_sum) -1
            
            # for draws, take random pick
            # this accounts for all zeros for a node; p_count==n_count for a node; 
            # all zeros for a node for all bootstrap(no support anyways);  
            print("draw: ",np.where(p_count_sum == n_count_sum)[0])
            for idx in np.where(p_count_sum == n_count_sum)[0]:
                node_type[idx] = np.random.choice([-1,1])    
            print("Lag "+str(i+1)+":", node_type)
    elif scheme == 2:
        # scheme #2: aggregate of individual bootstrap decisions
        for i in range(lag):
            tie = p_count[i] == n_count[i]
            
            p_comparison = p_count[i] > n_count[i]
            n_comparison = p_count[i] < n_count[i]
            
            p_comparison_aggregate = np.sum(p_comparison, axis=tuple(range(p_comparison.ndim-1)))
            n_comparison_aggregate = np.sum(n_comparison, axis=tuple(range(n_comparison.ndim-1)))
            
            node_type =  2 * (p_comparison_aggregate > n_comparison_aggregate)-1
        
            # for draws, take random pick
            # this accounts for all zeros for a node; p_count==n_count for a node; 
            # all zeros for a node for all bootstrap(no support anyways); I_candidate count == E_candidate count
            print("draw: ",np.where(p_comparison_aggregate == n_comparison_aggregate)[0])
            for idx in np.where(p_comparison_aggregate == n_comparison_aggregate)[0]:
                node_type[idx] = np.random.choice([-1,1])
            print("Lag "+str(i+1)+":", node_type)

    return node_type


def _unpack_coef(coef, lag, n_features, has_bias=True):

    # coef shape: (n_boot, n_reg, n_features * (lag * n_features + has_bias))
    # so we need to flip the first two axis first to align with the FDR indexing scheme
    coef = np.transpose(coef, (1, 0, 2))
    # coef shape: (n_reg, n_boot, n_features * (lag * n_features + has_bias))
    n_reg, n_boot = coef.shape[:2]
    
    # reshape back to matrix form while preserving n_reg and n_boot dimensions
    # reshape to (n_reg, n_boot, n_features, lag * n_features + has_bias)
    coef = coef.reshape(n_reg, n_boot, n_features, lag * n_features + has_bias)
    
    # transpose last two dims: (n_reg, n_boot, lag * n_features + has_bias, n_features)
    coef = np.transpose(coef, (0, 1, 3, 2))
    
    if has_bias:
        # b shape: (n_reg, n_boot, n_features)
        b = coef[:, :, 0, :]
        # adjacency_flat shape: (n_reg, n_boot, lag*n_features, n_features)
        adjacency_flat = coef[:, :, 1:, :]
    else:
        b = None
        # adjacency_flat shape: (n_reg, n_boot, lag*n_features, n_features)
        adjacency_flat = coef
    
    # reshape into adjacency matrices per lag
    # shape: (n_reg, n_boot, lag, n_features, n_features)
    adjacency_matrices = adjacency_flat.reshape(n_reg, n_boot, lag, n_features, n_features)
    
    # transpose the last two dimensions for each lag
    # final shape: (n_reg, n_boot, lag, n_features, n_features)
    A = np.transpose(adjacency_matrices, (0, 1, 2, 4, 3))
    
    return A, b

def _pack_coef(A, b=None):
    """
    Convert VAR parameters (adjacency matrices and intercept) to vectorized form.
    
    Parameters:
    -----------
    A : np.ndarray
        Adjacency matrices of shape (lag, n_features, n_features)
    b : np.ndarray or None
        Intercept vector of shape (n_features,) if present
    
    Returns:
    --------
    coef : np.ndarray
        Vectorized coefficients
    """
    lag, n_features, _ = A.shape
    has_bias = b is not None
    
    # Transpose A back: (lag, n_features, n_features) -> (lag, n_features, n_features)
    adjacency_matrices = np.transpose(A, (0, 2, 1))
    
    # Flatten adjacency matrices: (lag, n_features, n_features) -> (lag*n_features, n_features)
    adjacency_flat = adjacency_matrices.reshape(lag * n_features, n_features)
    
    if has_bias:
        # Stack intercept with adjacency matrices
        coef = np.vstack([b.reshape(1, -1), adjacency_flat])  # shape: (lag*n_features + 1, n_features)
    else:
        coef = adjacency_flat  # shape: (lag*n_features, n_features)
    
    # Transpose and flatten to match original input format
    coef = coef.T.flatten()
    
    return coef

def edge_selector_fdr(A_edges_real_list, A_edges_shuf_list, alpha=0.05):
    """
    Apply row-wise FDR to select significant edges.

    Parameters
    ----------
    A_edges_real_list : list of np.ndarray
        List of Kreal real-data edge matrices (off-diagonals only).
    A_edges_shuf_list : list of np.ndarray
        List of Kshuf shuffled-data edge matrices (off-diagonals only).
    alpha : float
        FDR control level.

    Returns
    -------
    W_edges : np.ndarray
        Binary mask of significant edges (same shape as A_edges_real).
    W_pval : np.ndarray
        P-value matrix for each edge.
    summary : dict
        Summary statistics.
    """

    Kreal = len(A_edges_real_list)
    Kshuf = len(A_edges_shuf_list)

    # Stack into arrays: shape (K, N, N)
    A_real = np.stack(A_edges_real_list, axis=0)
    A_shuf = np.stack(A_edges_shuf_list, axis=0)


    N = A_real.shape[1]

    # Statistic: median magnitude across bootstraps for each edge
    stat_obs = np.median(np.abs(A_real), axis=0)  # shape (N, N)

    # Null distribution: pool shuffled bootstraps per row
    W_pval = np.ones((N, N))
    W_edge_mask = np.zeros((N, N), dtype=bool)
    #W_edge_mask is a binary mask of significant edges after row‑wise FDR.

    # Statistics tracking    
    edges_per_row = []

    for i in range(N):
        # Pool null values for row i from shuffled data
        null_vals_row = np.abs(A_shuf[:, i, :]).ravel()        
        # Compute p-values for all j in this row
        for j in range(N):
            if i == j:
                continue
            obs_val = stat_obs[i, j]
            # Empirical p-value
            pval = np.mean(null_vals_row >= obs_val)
            W_pval[i, j] = pval

        # Apply FDR for this row
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        #......  per‑row FDR selection, not global FDR
        reject, _ = fdrcorrection(W_pval[i, mask], alpha=alpha)
        W_edge_mask[i, mask] = reject
        
        # Count selected edges for this row
        edges_per_row.append(int(reject.sum()))

    # Additional statistics
    edges_per_row = np.array(edges_per_row)
    
    # Count zeros in real data matrices
    total_elements = A_real.size
    zero_elements = np.sum(A_real == 0.0)
    sparsity_A_real = zero_elements / total_elements
    
    # Compute sparsity of W_edge_mask (fraction of zeros in off-diagonal elements)
    off_diag_elements = N * (N - 1)  # Total off-diagonal elements
    selected_edges = int(W_edge_mask.sum()) - N  # Subtract diagonal (always True)
    sparsity_W_mask = 1.0 - (selected_edges / off_diag_elements)
    
    # Compute pooled values from A_shuf dimensions: A_shuf shape is (Kshuf, N, N)
    # For each row i, we pool A_shuf[:, i, :].ravel() which gives Kshuf*N values
    pooled_values_per_row = Kshuf * N
    pooled_values_total = N * pooled_values_per_row  # N rows total

    summary = {
        "alpha": alpha,
        "num_bootstraps_real": Kreal,
        "num_bootstraps_shuf": Kshuf,
        "matrix_size": N,
        "num_significant_edges": int(W_edge_mask.sum()) - N,  # Exclude diagonal
        "pooled_values_per_row": int(pooled_values_per_row),
        "pooled_values_total": int(pooled_values_total),
        "zero_elements_A_lasso": int(zero_elements),
        "total_elements_A_lasso": int(total_elements),
        "sparsity_A_lasso": float(sparsity_A_real),
        "sparsity_W_edge_mask": float(sparsity_W_mask),
        "edges_per_row_avg": float(edges_per_row.mean()),
        "edges_per_row_std": float(edges_per_row.std()),
        "edges_per_row_min": int(edges_per_row.min()),
        "edges_per_row_max": int(edges_per_row.max()),
    }

    return W_edge_mask, W_pval, summary

def intersection_FDR(coefs, coefs_shuf, lag, n_features, fdr_rate = 0.05):
    
    # _unpack_coef makes n_reg X n_boot X n_feat X n_feat...
    A_real, _ = _unpack_coef(coefs, lag, n_features, has_bias=True)
    A_shuf, _ = _unpack_coef(coefs_shuf, lag, n_features, has_bias=True)
    
    # in case if lag = 1, get rid of the redundant dimension
    A_real = np.squeeze(A_real)
    A_shuf = np.squeeze(A_shuf)
    
    # set diagonal to zero, since we are do doing variable selection on diagonals
    A_real[:, :, np.arange(n_features), np.arange(n_features)] = 0
    A_shuf[:, :, np.arange(n_features), np.arange(n_features)] = 0

    # also need to clear the digonals of the A_*
    supports = []
    #iterating through the L1-penalty list
    count_list = []
    for i_reg in range(A_real.shape[0]):
        W_edge_mask, W_pval, summary = edge_selector_fdr(A_real[i_reg], A_shuf[i_reg], alpha = fdr_rate)
        
        
        # W_edge_mask is n_feat X n_feat
        np.fill_diagonal(W_edge_mask, 1)  # add digonal back in
        W_edge_mask = W_edge_mask[np.newaxis]  # making the dimension for LAG

        # Count all nonzero - diagonal nonzero for each array
        total_nonzero = np.count_nonzero(W_edge_mask, axis=(1, 2))
        diag_nonzero = np.count_nonzero(W_edge_mask.diagonal(axis1=1, axis2=2), axis=1)
        offdiag_counts = total_nonzero - diag_nonzero

        count_list.append(offdiag_counts)  # Array of shape (lag,) with counts for each matrix

        # the biase terms are always assumed to be fully dense, so there’s no selection on it
        support_i = _pack_coef(W_edge_mask, b=np.ones(A_real.shape[-1]))
        supports.append(support_i)

    
    return np.array(supports), count_list   #  supports SHOULD have shape n_reg_params X n_coef!!!
    