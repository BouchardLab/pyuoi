import numpy as np
from sklearn.utils import check_random_state

from ..utils import softmax, sigmoid


def load_swimmer(flatten=True):
    from pkg_resources import resource_filename
    import h5py
    with h5py.File(resource_filename('pyuoi', 'data/Swimmer.h5'), 'r+') as f:
        swimmers = f['Y'][:].astype(float)
    if flatten:
        swimmers = swimmers.T.reshape(256, 1024)
    return swimmers


def make_linear_regression(n_samples=100, n_features=5, n_informative=2,
                           X_loc=3., X_scale=1., snr=5.,
                           beta=None, beta_low=1., beta_high=3.,
                           include_intercept=False, random_state=None):
    """Make a Linear regression dataset.

    Parameters
    ----------
    n_samples : int
        The number of samples to make.
    n_features : int
        The number of feature to use.
    n_informative : int
        The number of feature with non-zero weights.
    X_loc : float
        The mean of the features in the design matrix.
    X_scale : float
        The standard deviation of the features in the design matrix.
    snr : float
        The signal-to-noise ratio, which informs the variance of the noise
        term.
    beta : np.ndarray or None
        The beta values to use. If None, beta values will be drawn from a
        uniform distribution.
    beta_low : float
        The lower bound for the beta values.
    beta_high : float
        The upper bound for the beta values.
    include_intercept : bool
        If true, includes an intercept in the model, if False, the intercept is
        set to 0.
    random_state : int, np.random.RandomState instance, or None
        Random number seed or state.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        The design matrix.
    y : ndarray, shape (n_samples,)
        The response vector.
    beta : ndarray, shape (n_features,)
        The feature coefficients.
    intercept : float
        The intercept. If include_intercept is False, then intercept is zero.
    """
    rng = check_random_state(random_state)

    # create design matrix
    X = rng.normal(loc=X_loc,
                   scale=X_scale,
                   size=(n_samples, n_features))

    # create coefficients
    if beta is None:
        # draw beta values from gamma distribution
        beta = rng.uniform(low=beta_low,
                           high=beta_high,
                           size=n_features)

        # choose sparsity mask
        zero_idx = np.zeros(n_features)
        zero_idx[:n_informative] = 1
        rng.shuffle(zero_idx)
        # randomly assign beta elements to zero
        beta = beta * zero_idx

    # create intercept
    if include_intercept:
        intercept = rng.uniform(low=beta_low, high=beta_high)
    else:
        intercept = 0

    # draw response variable
    eta = intercept + np.dot(X, beta)
    signal_var = np.var(eta)
    noise_var = signal_var / snr
    noise = rng.normal(loc=0, scale=np.sqrt(noise_var), size=eta.shape)
    y = eta + noise

    return X, y, beta, intercept


def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_classes=2, shared_support=False, random_state=None,
                        w_scale=1., include_intercept=False):
    """Make a linear classification dataset.

    Parameters
    ----------
    n_samples : int
        The number of samples to make.
    n_features : int
        The number of feature to use.
    n_informative : int
        The number of feature with non-zero weights.
    n_classes : int
        The number of classes.
    shared_support : bool
        If True, all classes will share the same random support. If False, they
        will each have randomly chooses support.
    random_state : int or np.random.RandomState instance
        Random number seed or state.
    w_scale : float
        The model parameter matrix, w, will be drawn from a normal distribution
        with std=w_scale.
    include_intercept : bool
        If true, includes an intercept in the model, if False, the intercept is
        set to 0.
    """
    if isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    else:
        rng = random_state
    n_not_informative = n_features - n_informative

    X = rng.randn(n_samples, n_features)
    X -= X.mean(axis=-1, keepdims=True)
    X /= X.std(axis=-1, keepdims=True)

    if n_classes > 2:
        w = rng.randn(n_features, n_classes)
        if include_intercept:
            intercept = rng.randn(1, n_classes)
            intercept -= intercept.max()
        else:
            intercept = np.zeros((1, n_classes))
        if n_not_informative > 0:
            if shared_support:
                idxs = rng.permutation(n_features)[:n_not_informative]
                w[idxs] = 0.
            else:
                for ii in range(n_classes):
                    idxs = rng.permutation(n_features)[:n_not_informative]
                    w[idxs, ii * np.ones_like(idxs, dtype=int)] = 0.
    else:
        w = rng.randn(n_features, 1)
        if include_intercept:
            intercept = rng.randn(1, 1)
        else:
            intercept = np.zeros((1, 1))
        if n_not_informative > 0:
            idxs = rng.permutation(n_features)[:n_not_informative]
            w[idxs] = 0.
    w *= w_scale

    log_p = X.dot(w)
    if include_intercept:
        log_p += intercept
    if n_classes > 2:
        p = softmax(log_p)
        y = np.array([rng.multinomial(1, pi) for pi in p])
        y = y.argmax(axis=-1)
    else:
        p = sigmoid(np.squeeze(log_p))
        y = np.array([rng.binomial(1, pi) for pi in p])

    return X, y, w.T, intercept


def make_poisson_regression(n_samples=100, n_features=5, n_informative=2,
                            X_loc=0., X_scale=1. / 8,
                            beta=None, beta_shape=1., beta_scale=3.,
                            include_intercept=False, random_state=None):
    """Make a Poisson regression dataset.

    Parameters
    ----------
    n_samples : int
        The number of samples to make.
    n_features : int
        The number of feature to use.
    n_informative : int
        The number of feature with non-zero weights.
    X_loc : float
        The mean of the features in the design matrix.
    X_scale : float
        The standard deviation of the features in the design matrix.
    beta : np.ndarray or None
        The beta values to use. If None, beta values will be drawn from a gamma
        distribution.
    beta_shape : float
        The shape parameter for the beta values.
    beta_scale : float
        The scale parameter for the beta values.
    include_intercept : bool
        If true, includes an intercept in the model, if False, the intercept is
        set to 0.
    random_state : int, np.random.RandomState instance, or None
        Random number seed or state.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        The design matrix.
    y : ndarray, shape (n_samples,)
        The response vector.
    beta : ndarray, shape (n_features,)
        The feature coefficients.
    intercept : float
        The intercept. If include_intercept is False, then intercept is zero.
    """
    rng = check_random_state(random_state)

    # create design matrix
    X = rng.normal(loc=X_loc,
                   scale=X_scale,
                   size=(n_samples, n_features))

    # create coefficients
    if beta is None:
        # draw beta values from gamma distribution
        beta = rng.gamma(shape=beta_shape,
                         scale=beta_scale,
                         size=n_features)
        # choose sparsity mask
        zero_idx = np.zeros(n_features)
        zero_idx[:n_informative] = 1
        rng.shuffle(zero_idx)
        # randomly assign beta elements to zero
        beta = beta * zero_idx

    # create intercept
    if include_intercept:
        intercept = rng.gamma(shape=beta_shape, scale=beta_scale)
    else:
        intercept = 0

    # draw response variable
    eta = intercept + np.dot(X, beta)
    y = rng.poisson(np.exp(eta))

    return X, y, beta, intercept
