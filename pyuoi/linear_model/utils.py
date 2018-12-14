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
