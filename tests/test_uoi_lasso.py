import pytest
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal_nulp,
                           assert_equal, assert_allclose)

from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.linear_model.coordinate_descent import _alpha_grid
try:
    import pycasso
except ImportError:
    pycasso = None

from pyuoi import UoI_Lasso
from pyuoi.linear_model.lasso import PycLasso
from pyuoi.datasets import make_linear_regression


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

    # Also test both choices of solver
    lasso = UoI_Lasso(
        fit_intercept=False,
        selection_frac=0.75,
        estimation_frac=0.75,
        solver='cd'
    )
    lasso.fit(X, y)
    lasso.fit(X, y, verbose=True)

    assert_allclose(lasso.coef_, beta)

    if pycasso is not None:
        lasso = UoI_Lasso(
            fit_intercept=False,
            selection_frac=0.75,
            estimation_frac=0.75,
            solver='pyc'
        )
        lasso.fit(X, y)

        assert_allclose(lasso.coef_, beta)


def test_uoi_lasso_estimation_shape_match():
    """Test UoI Lasso on a toy example."""

    X = np.array([
        [-1, 2],
        [4, 1],
        [1, 3],
        [4, 3],
        [8, 11]], dtype=float)
    beta = np.array([1, 4], dtype=float)
    y = np.dot(X, beta)

    lasso = UoI_Lasso()
    lasso.fit(X, y)
    with pytest.raises(ValueError, match='Targets and predictions are ' +
                       'not the same shape.'):
        support = np.arange(2)
        boot_idxs = [np.arange(4)] * 2
        lasso.coef_ = np.array([[1, 4], [1, 4]])
        lasso._score_predictions('r2', lasso, X, y,
                                 support, boot_idxs)

    with pytest.raises(ValueError, match='y should either have'):
        support = np.arange(2)
        boot_idxs = [np.arange(4)] * 2
        lasso._score_predictions('r2', lasso, X, y[:, np.newaxis, np.newaxis],
                                 support, boot_idxs)


def test_uoi_lasso_fit_shape_match():
    """Test UoI Lasso on a toy example."""

    X = np.array([
        [-1, 2],
        [4, 1],
        [1, 3],
        [4, 3],
        [8, 11]], dtype=float)
    beta = np.array([1, 4], dtype=float)
    y = np.dot(X, beta)

    lasso = UoI_Lasso()
    lasso.fit(X, y)

    # Check that second axis gets squeezed
    lasso.fit(X, y[:, np.newaxis])

    # Check that second axis gets squeezed
    message = 'y should either have shape'
    with pytest.raises(ValueError, match=message):
        lasso.fit(X, np.tile(y[:, np.newaxis], (1, 2)))
    with pytest.raises(ValueError, match=message):
        lasso.fit(X, y[:, np.newaxis, np.newaxis])


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


def test_intercept_and_coefs_no_selection():
    """Test that UoI Lasso properly calculates the intercept with and without
    standardization."""
    # create line model
    X, y, beta, intercept = make_linear_regression(
        n_samples=500,
        n_features=2,
        n_informative=2,
        snr=10.,
        include_intercept=True,
        random_state=2332)

    # without standardization
    lasso = UoI_Lasso(
        standardize=False,
        fit_intercept=True
    )
    lasso.fit(X, y)
    assert_allclose(lasso.intercept_, intercept, rtol=0.25)
    assert_allclose(lasso.coef_, beta, rtol=0.25)

    # with standardization
    lasso = UoI_Lasso(
        standardize=True,
        fit_intercept=True
    )
    lasso.fit(X, y)
    assert_allclose(lasso.intercept_, intercept, rtol=0.25)
    assert_allclose(lasso.coef_, beta, rtol=0.25)


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


def test_fit_intercept():
    """Tests whether `include_intercept` in passed through to the linear models.
    """
    lasso = UoI_Lasso(fit_intercept=True)
    assert lasso._selection_lm.fit_intercept
    assert lasso._estimation_lm.fit_intercept

    lasso = UoI_Lasso(fit_intercept=False)
    assert not lasso._selection_lm.fit_intercept
    assert not lasso._estimation_lm.fit_intercept


@pytest.mark.skipif(pycasso is None, reason='pycasso not installed')
def test_choice_of_solver():
    '''Tests whether one can correctly switch between solvers in UoI Lasso'''

    uoi1 = UoI_Lasso(solver='cd')
    assert(isinstance(uoi1._selection_lm, Lasso))

    uoi2 = UoI_Lasso(solver='pyc')
    assert(isinstance(uoi2._selection_lm, PycLasso))


@pytest.mark.skipif(pycasso is not None, reason='pycasso is installed')
def test_pycasso_error():
    """Tests whether an error is raised if pycasso is not installed.
    """

    with pytest.raises(ImportError):
        uoi2 = UoI_Lasso(solver='pyc')
        assert(isinstance(uoi2._selection_lm, PycLasso))


@pytest.mark.skipif(pycasso is None, reason='pycasso not installed')
def test_pyclasso():
    """Tests whether the PycLasso class is working"""

    pyclasso = PycLasso(fit_intercept=False, max_iter=1000)

    # Test that we can set params correctly
    pyclasso.set_params(fit_intercept=True)
    assert(pyclasso.fit_intercept)
    pyclasso.set_params(max_iter=500)
    assert(pyclasso.max_iter == 500)
    pyclasso.set_params(alphas=np.arange(100))
    assert(np.array_equal(pyclasso.alphas, np.arange(100)))

    # Test that spurious parameters are rejected
    try:
        pyclasso.set_params(blah=5)
        Exception('No exception thrown!')
    except ValueError:
        pass
    finally:
        Exception('Unexpected Exception raised')

    # Tests against a toy problem
    X = np.array([
        [-1, 2, 3],
        [4, 1, -7],
        [1, 3, 1],
        [4, 3, 12],
        [8, 11, 2]], dtype=float)
    beta = np.array([1, 4, 2], dtype=float)
    y = np.dot(X, beta)

    alphas = _alpha_grid(X, y)
    pyclasso.set_params(alphas=alphas, fit_intercept=False)
    pyclasso.fit(X, y)
    assert(np.array_equal(pyclasso.coef_.shape, (100, 3)))
    y_pred = pyclasso.predict(X)
    scores = np.array([r2_score(y, y_pred[:, j]) for j in range(100)])
    assert(np.allclose(1, max(scores)))


def test_lass_bad_est_score():
    """Test that UoI Lasso raises an error when given a bad
    estimation_score value.
    """
    X = np.random.randn(20, 5)
    y = np.random.randn(20)

    with pytest.raises(ValueError):
        UoI_Lasso(estimation_score='z',
                  n_boots_sel=10, n_boots_est=10).fit(X, y)
