import numpy as np


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


def intersection(coefs, selection_thresholds=None):
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

    # iterate over each stability selection threshold
    for thresh_idx, threshold in enumerate(selection_thresholds):
        # calculate the support given the specific selection threshold
        supports[thresh_idx, ...] = \
            np.count_nonzero(coefs, axis=0) >= threshold

    # unravel the dimension corresponding to selection thresholds

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



