"""Utility functions for pyuoi package.
"""
import numpy as np


def softmax(y, axis=-1):
    """Calculates the softmax distribution.

    Parameters
    ----------
    y : ndarray
        Log-probabilities.
    """

    yp = y - y.max(axis=axis, keepdims=True)
    epy = np.exp(yp)
    return epy / np.sum(epy, axis=axis, keepdims=True)


def sigmoid(x):
    """Calculates the bernoulli distribution.

    Parameters
    ----------
    y : ndarray
        Log-probabilities.
    """
    return np.exp(-np.logaddexp(0, -x))


def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_classes=2, shared_support=False, random_state=None,
                        w_scale=1.):
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
        if n_not_informative > 0:
            idxs = rng.permutation(n_features)[:n_not_informative]
            w[idxs] = 0.
    w *= w_scale

    log_p = X.dot(w)
    if n_classes > 2:
        p = softmax(log_p)
        y = np.array([rng.multinomial(1, pi) for pi in p])
        y = y.argmax(axis=-1)
    else:
        p = sigmoid(np.squeeze(log_p))
        y = np.array([rng.binomial(1, pi) for pi in p])

    return X, y, w.T


def log_likelihood_glm(model, y_true, y_pred):
    """Calculates the log-likelihood of a generalized linear model given the
    true response variables and the "predicted" response variables. The
    "predicted" response variable varies by the specific generalized linear
    model under consideration.

    Parameters
    ----------
    model : string
        The generalized linear model to calculate the log-likelihood for.

    y_true : nd-array, shape (n_samples)
        Array of true response values.

    y_pred : nd-array, shape (n_samples)
        Array of predicted response values (conditional mean).

    Results
    -------
    ll : float
        The log-likelihood.
    """
    if model == 'normal':
        # this log-likelihood is calculated under the assumption that the
        # variance is the value that maximize the log-likelihood
        rss = (y_true - y_pred)**2
        n_samples = y_true.size
        ll = -n_samples / 2 * (1 + np.log(np.mean(rss)))
    elif model == 'poisson':
        ll = np.mean(y_true * np.log(y_pred) - y_pred)
    else:
        raise ValueError('Model is not available.')
    return ll


def BIC(ll, n_features, n_samples):
    """Calculates the Bayesian Information Criterion.

    Parameters
    ----------
    ll : float
        The log-likelihood of the model.

    n_features : int
        The number of features used in the model.

    n_samples : int
        The number of samples in the dataset being tested.

    Returns
    -------
    BIC : float
        Bayesian Information Criterion
    """
    BIC = n_features * np.log(n_samples) - 2 * ll
    return BIC


def AIC(ll, n_features):
    """Calculates the Akaike Information Criterion.

    Parameters
    ----------
    ll : float
        The log-likelihood of the model.

    n_features : int
        The number of features used in the model.

    n_samples : int
        The number of samples in the dataset being tested.

    Returns
    -------
    AIC : float
        Akaike Information Criterion
    """

    AIC = 2 * n_features - 2 * ll
    return AIC


def AICc(ll, n_features, n_samples):
    """Calculate the corrected Akaike Information Criterion. This criterion is
    useful in cases when the number of samples is small.

    If the number of features is equal to the number of samples plus one, then
    the AIC is returned (the AICc is undefined in this case).

    Parameters
    ----------
    ll : float
        The log-likelihood of the model.

    n_features : int
        The number of features used in the model.

    n_samples : int
        The number of samples in the dataset being tested.

    Returns
    -------
    AIC : float
        Akaike Information Criterion
    """
    AICc = AIC(ll, n_features)
    if n_samples > (n_features + 1):
        AICc += 2 * (n_features**2 + n_features) / (n_samples - n_features - 1)
    return AICc
