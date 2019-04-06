import numpy as np


def column_select(V, c, leverage_sort=False):
    """Chooses column indices from a matrix given its SVD.

    Parameters
    ----------
    V : ndarray, shape (n_features, rank)
        The set of singular vectors.

    c : float
        The expected number of columns to select.

    leverage_sort : bool
        If True, resorts the column indices in increasing order of leverage
        score. If False, the column indices are sorted normally.

    Returns
    -------
    column_indices : ndarray of ints
        An array of indices denoting which columns were selected. If
        leverage_sort was true, this array is arranged by increasing leverage
        score.
    """
    # extract number of samples and rank
    n_features, k = V.shape

    # calculate normalized leverage score
    pi = np.sum(V**2, axis=1) / k

    # iterate through columns
    column_flags = np.zeros(n_features, dtype=bool)
    for column in range(n_features):
        # Mahoney (2009), eqn 3
        p = min(1, c * pi[column])
        # selected column randomly
        column_flags[column] = p > np.random.rand()

    column_indices = np.argwhere(column_flags).ravel()

    if leverage_sort:
        pi_subset = pi[column_indices]
        column_indices = column_indices[np.argsort(pi_subset)]

    return column_indices