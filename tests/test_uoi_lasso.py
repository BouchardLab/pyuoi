import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal_nulp,
                           assert_equal, assert_allclose)

from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

from pyuoi import UoI_Lasso


def test_variable_selection():
    """Test basic functionality of UoI_Lasso and that it finds right model"""

    X, y, w = make_regression(coef=True, random_state=1)
    lasso = UoI_Lasso()
    lasso.fit(X, y)
    true_coef = np.nonzero(w)[0]
    fit_coef = np.nonzero(lasso.coef_)[0]
    assert_array_equal(true_coef, fit_coef)
    assert_array_almost_equal_nulp(true_coef, fit_coef)


def test_estimation_score_usage():
    """Test the ability to change the estimation score in UoI Lasso."""

    methods = ('r2', 'AIC', 'AICc', 'BIC')
    X, y = make_regression(n_features=10, n_informative=3,
                           random_state=10)
    scores = []
    for method in methods:
        lasso = UoI_Lasso(estimation_score=method)
        assert_equal(lasso.estimation_score, method)
        lasso.fit(X, y)
        y_hat = lasso.predict(X)
        assert_equal(r2_score(y, y_hat), lasso.score(X, y))
        score = np.max(lasso.scores_)
        scores.append(score)
    assert_equal(len(np.unique(scores)), len(methods))


def test_set_random_state():
    """Tests whether random states are handled correctly."""
    X, y = make_regression(n_features=5, n_informative=3,
                           random_state=16, noise=.5)
    # same state
    l1log_0 = UoI_Lasso(random_state=13)
    l1log_1 = UoI_Lasso(random_state=13)
    l1log_0.fit(X, y)
    l1log_1.fit(X, y)
    assert_array_equal(l1log_0.coef_, l1log_1.coef_)

    # different state
    l1log_1 = UoI_Lasso(random_state=14)
    l1log_1.fit(X, y)
    assert not np.array_equal(l1log_0.coef_, l1log_1.coef_)

    # different state, not set
    l1log_0 = UoI_Lasso()
    l1log_1 = UoI_Lasso()
    l1log_0.fit(X, y)
    l1log_1.fit(X, y)
    assert not np.array_equal(l1log_0.coef_, l1log_1.coef_)


def test_uoi_lasso_toy():
    """Test UoI Lasso on a toy example."""

    X = np.array([
        [-1, 2],
        [4, 1],
        [1, 3],
        [4, 3],
        [8, 11]], dtype=float)
    beta = np.array([1, 4], dtype=float)
    y = np.dot(X, beta)

    # choose selection_frac to be slightly smaller to ensure that we get
    # good test sets
    lasso = UoI_Lasso(
        fit_intercept=False,
        selection_frac=0.75,
        estimation_frac=0.75,
        standardize=False
    )
    lasso.fit(X, y)

    assert_allclose(lasso.coef_, beta)


def test_get_reg_params():
    """Tests whether get_reg_params works correctly for UoI Lasso."""

    X = np.array([
        [-1, 2],
        [0, 1],
        [1, 3],
        [4, 3]])
    y = np.array([7, 4, 13, 16])

    # calculate regularization parameters manually
    alpha_max = np.max(np.dot(X.T, y) / 4)
    alphas = [{'alpha': alpha_max}, {'alpha': alpha_max / 10.}]

    # calculate regularization parameters with UoI_Lasso object
    lasso = UoI_Lasso(
        n_lambdas=2,
        fit_intercept=False,
        eps=0.1)
    reg_params = lasso.get_reg_params(X, y)

    # check each regularization parameter and key
    for estimate, true in zip(reg_params, alphas):
        assert estimate.keys() == true.keys()
        assert_allclose(list(estimate.values()), list(true.values()))


def test_intercept():
    """Test that UoI Lasso properly calculates the intercept when centering
    the response variable."""

    X = np.array([
        [-1, 2],
        [0, 1],
        [1, 3],
        [4, 3]], dtype=float)
    y = np.array([8, 5, 14, 17], dtype=float)

    lasso = UoI_Lasso(
        fit_intercept=True,
        standardize=False)
    lasso.fit(X, y)

    assert lasso.intercept_ == (np.mean(y) -
                                np.dot(X.mean(axis=0), lasso.coef_))


def test_lasso_selection_sweep():
    """Tests uoi_selection_sweep for UoI_Lasso."""

    # toy data
    X = np.array([
        [-1, 2, 3],
        [4, 1, -7],
        [1, 3, 1],
        [4, 3, 12],
        [8, 11, 2]], dtype=float)
    beta = np.array([1, 4, 2], dtype=float)
    y = np.dot(X, beta)

    # toy regularization
    reg_param_values = [{'alpha': 1.0}, {'alpha': 2.0}]
    lasso = UoI_Lasso(fit_intercept=True, warm_start=False)
    lasso1 = Lasso(alpha=1.0, fit_intercept=True, max_iter=lasso.max_iter)
    lasso2 = Lasso(alpha=2.0, fit_intercept=True, max_iter=lasso.max_iter)
    lasso.output_dim = 1

    coefs = lasso.uoi_selection_sweep(X, y, reg_param_values)
    lasso1.fit(X, y)
    lasso2.fit(X, y)

    assert np.allclose(coefs[0], lasso1.coef_)
    assert np.allclose(coefs[1], lasso2.coef_)
