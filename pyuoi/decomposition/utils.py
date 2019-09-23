import numpy as np


def column_select(V, c, leverage_sort=False, random_state=None):
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
    # random state
    rng = np.random.RandomState(seed=None)

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
        column_flags[column] = p > rng.rand()

    column_indices = np.argwhere(column_flags).ravel()

    # if desired, sort by increasing leverage score
    if leverage_sort:
        pi_subset = pi[column_indices]
        column_indices = column_indices[np.argsort(pi_subset)]

    return column_indices


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
        bootstraps.

    n_boots: int
        The number of bootstraps that will be used for selection
    """

    # float, indicating proportion of bootstraps
    if isinstance(stability_selection, float):
        selection_threshold = int(stability_selection * n_boots)

    # int, indicating number of bootstraps
    elif isinstance(stability_selection, int):
        selection_threshold = stability_selection

    else:
        raise ValueError("Stability selection must be a valid float or int.")

    # ensure that ensuing list of selection thresholds satisfies
    # the correct bounds
    if not (
        selection_threshold <= n_boots and selection_threshold >= 1
    ):
        raise ValueError("Stability selection thresholds must be within "
                         "the correct bounds.")

    return selection_threshold


def dissimilarity(H1, H2):
    """Calculates the dissimilarity between two sets of NMF bases.

    Parameters
    ----------
    H1 : ndarray, shape (n_components, n_features)
        First set of bases.

    H2 : ndarray, shape (n_components, n_features)
        Second set of bases.

    Returns
    -------
    diss : float
        Dissimilarity between the two sets of bases.
    """
    k = H1.shape[0]
    H1 = H1 / np.linalg.norm(H1, axis=1, keepdims=True)
    H2 = H2 / np.linalg.norm(H2, axis=1, keepdims=True)
    C = np.dot(H1, H2.T)
    diss = 1 - ((np.max(C, axis=0).sum() + np.max(C, axis=1).sum()) / (2. * k))
    return diss
