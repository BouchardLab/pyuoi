import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal_nulp,
                           assert_equal, assert_allclose)

from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

from pyuoi import UoI_ElasticNet


def test_variable_selection():
    """Test basic functionality of UoI_ElasticNet and that it
    finds right model"""

    X, y, w = make_regression(coef=True, random_state=1)
    enet = UoI_ElasticNet(alphas=[1., .9])
    enet.fit(X, y)
    true_coef = np.nonzero(w)[0]
    fit_coef = np.nonzero(enet.coef_)[0]
    assert_array_equal(true_coef, fit_coef)
    assert_array_almost_equal_nulp(true_coef, fit_coef)


def test_estimation_score_usage():
    """Test the ability to change the estimation score in UoI ElasticNet."""

    methods = ('r2', 'AIC', 'AICc', 'BIC')
    X, y = make_regression(n_features=10, n_informative=3,
                           random_state=10)
    scores = []
    for method in methods:
        enet = UoI_ElasticNet(estimation_score=method)
        assert_equal(enet.estimation_score, method)
        enet.fit(X, y)
        y_hat = enet.predict(X)
        assert_equal(r2_score(y, y_hat), enet.score(X, y))
        score = np.max(enet.scores_)
        scores.append(score)
    assert_equal(len(np.unique(scores)), len(methods))


def test_set_random_state():
    """Tests whether random states are handled correctly."""
    X, y = make_regression(n_features=5, n_informative=3,
                           random_state=16, noise=.5)
    # same state
    l1log_0 = UoI_ElasticNet(random_state=13)
    l1log_1 = UoI_ElasticNet(random_state=13)
    l1log_0.fit(X, y)
    l1log_1.fit(X, y)
    assert_array_equal(l1log_0.coef_, l1log_1.coef_)

    # different state
    l1log_1 = UoI_ElasticNet(random_state=14)
    l1log_1.fit(X, y)
    assert not np.array_equal(l1log_0.coef_, l1log_1.coef_)

    # different state, not set
    l1log_0 = UoI_ElasticNet()
    l1log_1 = UoI_ElasticNet()
    l1log_0.fit(X, y)
    l1log_1.fit(X, y)
    assert not np.array_equal(l1log_0.coef_, l1log_1.coef_)


def test_uoi_enet_toy():
    """Test UoI ElasticNet on a toy example."""

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
    enet = UoI_ElasticNet(
        fit_intercept=False,
        selection_frac=0.75,
        estimation_frac=0.75
    )
    enet.fit(X, y)

    assert_allclose(enet.coef_, beta)


def test_get_reg_params():
    """Tests whether get_reg_params works correctly for UoI ElasticNet."""

    X = np.array([
        [-1, 2],
        [0, 1],
        [1, 3],
        [4, 3]])
    y = np.array([7, 4, 13, 16])

    # calculate regularization parameters manually
    l1_ratio = .5
    alpha_max = np.max(np.dot(X.T, y) / 4) / l1_ratio
    alphas = [{'alpha': alpha_max, 'l1_ratio': .5},
              {'alpha': alpha_max / 10., 'l1_ratio': .5}]

    # calculate regularization parameters with UoI_ElasticNet object
    enet = UoI_ElasticNet(
        n_lambdas=2,
        normalize=False,
        fit_intercept=False,
        eps=0.1)
    reg_params = enet.get_reg_params(X, y)

    # check each regularization parameter and key
    for estimate, true in zip(reg_params, alphas):
        assert estimate.keys() == true.keys()
        assert_allclose(list(estimate.values()), list(true.values()))


def test_intercept():
    """Test that UoI ElasticNet properly calculates the intercept when centering
    the response variable."""

    X = np.array([
        [-1, 2],
        [0, 1],
        [1, 3],
        [4, 3]])
    y = np.array([8, 5, 14, 17])

    enet = UoI_ElasticNet(
        normalize=False,
        fit_intercept=True)
    enet.fit(X, y)

    assert enet.intercept_ == np.mean(y) - np.dot(X.mean(axis=0), enet.coef_)


def test_enet_selection_sweep():
    """Tests uoi_selection_sweep for UoI_ElasticNet."""

    # toy data
    X = np.array([
        [-1, 2, 3],
        [4, 1, -7],
        [1, 3, 1],
        [4, 3, 12],
        [8, 11, 2]])
    beta = np.array([1, 4, 2])
    y = np.dot(X, beta)

    # toy regularization
    reg_param_values = [{'alpha': 1.0}, {'alpha': 2.0}]
    enet1 = ElasticNet(alpha=1.0, fit_intercept=True, normalize=True)
    enet2 = ElasticNet(alpha=2.0, fit_intercept=True, normalize=True)
    enet = UoI_ElasticNet(fit_intercept=True, normalize=True)

    coefs = enet.uoi_selection_sweep(X, y, reg_param_values)
    enet1.fit(X, y)
    enet2.fit(X, y)

    assert np.allclose(coefs[0], enet1.coef_)
    assert np.allclose(coefs[1], enet2.coef_)
