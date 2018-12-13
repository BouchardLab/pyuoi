"""Utility functions for pyuoi package.
"""
import numpy as np
import scipy.sparse as sparse
from numpy.linalg import norm, cholesky


def BIC(y_true, y_pred, n_features):
    """Calculate the Bayesian Information Criterion under the assumption of
    normally distributed disturbances.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true response values.

    y_pred : np.ndarray
        Array of predicted response values.

    n_features : int
        Number of features used in the model.

    Returns
    -------
    BIC : float
        Bayesian Information Criterion
    """

    n_samples = y_true.size
    # calculate residual sum of squares
    rss = np.sum((y_true - y_pred)**2)

    BIC = n_samples * np.log(rss / n_samples) + \
        n_features * np.log(n_samples)
    return BIC


def AIC(y_true, y_pred, n_features):
    """Calculate the Akaike Information Criterion under the assumption of
    normally distributed disturbances. Utilizes a softer penalty on the
    model parsimony than the BIC.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true response values.

    y_pred : np.ndarray
        Array of predicted response values.

    n_features : int
        Number of features used in the model.

    Returns
    -------
    AIC : float
        Akaike Information Criterion
    """

    n_samples = y_true.size
    # calculate residual sum of squares
    rss = np.sum((y_true - y_pred)**2)

    AIC = n_samples * np.log(rss / n_samples) + \
        n_features * 2
    return AIC


def AICc(y_true, y_pred, n_features):
    """Calculate the corrected Akaike Information Criterion under the
    assumption of normally distributed disturbances. Modifies the parsimony
    penalty. Useful in cases when the number of samples is small.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true response values.

    y_pred : np.ndarray
        Array of predicted response values.

    n_features : int
        Number of features used in the model.

    Returns
    -------
    AICc : float
        corrected Akaike Information Criterion
    """

    n_samples = y_true.size
    # calculate residual sum of squares
    rss = np.sum((y_true - y_pred)**2)
    AICc = n_samples * np.log(rss / n_samples) + \
        n_features * 2 + \
        2 * (n_features**2 + n_features) / (n_samples - n_features - 1)
    return AICc


def lasso_admm(
    X, y, lamb, rho=1., alpha=1.,
    max_iter=1000, abs_tol=1e-5, rel_tol=1e-3,
    verbose=False
):
    """Solve the Lasso optimization problem using Alternating Direction Method
    of Multipliers (ADMM) Convergence criteria are given in section 3.3.1 in
    the Boyd manuscript (equation 3.12).
    """
    n_samples, n_features = X.shape

    # initialize parameter estimates x/z and dual estimates u (equivalent to y)
    x = np.zeros((n_features, 1))
    z = np.zeros((n_features, 1))
    # dual; equivalent to y in most formulations
    u = np.zeros((n_features, 1))

    Xy = np.dot(X.T, y).reshape((n_features, 1))
    inv = np.linalg.inv(np.dot(X.T, X) + rho * np.identity(n_features))

    for iteration in range(max_iter):
        # update x estimates
        x = np.dot(inv, Xy + rho * (z - u))

        # handle the over-relaxation term
        z_old = np.copy(z)
        x_hat = alpha * x + (1 - alpha) * z_old

        # update z term with over-relaxation
        z = shrinkage(x=x_hat, threshold=lamb / rho)

        # update dual
        u += x_hat - z

        # check convergence using eqn 3.12
        r_norm = norm(x - z)
        s_norm = norm(rho * (z - z_old))

        eps_primal = np.sqrt(n_features) * abs_tol + \
            np.maximum(norm(x), norm(z)) * rel_tol
        eps_dual = np.sqrt(n_features) * abs_tol + norm(u) * rel_tol

        if (r_norm <= eps_primal) and (s_norm <= eps_dual):
            if verbose:
                print('Convergence: iteration %s' % iteration)
            break
    return z.ravel()


def shrinkage(x, threshold):
    return np.maximum(0., x - threshold) - np.maximum(0., -x - threshold)


def factor(X, rho):
    n_samples, n_features = X.shape
    if n_samples >= n_features:
        L = cholesky(np.dot(X.T, X) + rho * sparse.eye(n_features))
    else:
        L = cholesky(sparse.eye(n_samples) + 1. / rho * (np.dot(X, X.T)))
    L = sparse.csc_matrix(L)
    U = sparse.csc_matrix(L.T)
    return L, U
