import numpy as np

from numpy.testing import (assert_allclose,
                           assert_equal,
                           assert_raises)

from pyuoi.linear_model import (Poisson,
                                UoI_Poisson)
from pyuoi.linear_model.poisson import (_poisson_loss_and_grad,
                                        PoissonInterceptFitterNoFeatures)
from pyuoi.datasets import make_poisson_regression

from sklearn.exceptions import NotFittedError

# poisson GLM model by hand

# design matrix
X = np.array([
    [0.35, 0.84, 0.95, 0.77, 0.88],
    [0.43, 0.76, 0.47, 0.09, 0.34],
    [0.41, 0.40, 0.08, 0.82, 0.49],
    [0.73, 0.93, 0.39, 0.77, 0.72],
    [0.69, 0.88, 0.32, 0.54, 0.26],
    [0.34, 0.10, 0.55, 0.20, 0.20],
    [0.20, 0.15, 0.23, 0.16, 0.74],
    [0.94, 0.08, 0.97, 0.03, 0.48],
    [0.61, 0.55, 0.72, 0.21, 0.27],
    [0.54, 0.21, 0.98, 0.26, 0.01]])

# true parameters
beta = np.array([3., 1., 0., -2., 0.])

# response variable
y = np.array([2, 6, 0, 4, 6, 3, 0, 16, 8, 7])


def test_soft_threshold():
    """Tests the soft threshold function against equivalent output from a
    MATLAB implementation."""

    true_threshold = np.array([
        [0, 0.34, 0.45, 0.27, 0.38],
        [0, 0.26, 0, 0, 0],
        [0, 0, 0, 0.32, 0],
        [0.23, 0.43, 0, 0.27, 0.22],
        [0.19, 0.38, 0, 0.04, 0],
        [0, 0, 0.05, 0, 0],
        [0, 0, 0.0, 0, 0.24],
        [0.44, 0, 0.47, 0, 0],
        [0.11, 0.05, 0.22, 0, 0],
        [0.04, 0, 0.48, 0, 0]])

    assert_allclose(true_threshold,
                    Poisson.soft_threshold(X, 0.5))


def test_adjusted_response():
    """Tests the adjusted response function against equivalent output from a
    MATLAB implementation."""

    w, z = Poisson.adjusted_response(X, y, beta)

    w_true = np.array([
        1.419067548593257, 6.488296399286710, 0.990049833749168,
        4.854955811237434, 6.488296399286709, 2.054433210643888,
        1.537257523548281, 17.115765537145876, 7.099327065156633,
        3.706173712210199])

    z_true = np.array([
        0.759376179437427, 1.794741970890788, -1.01,
        1.403900392819534, 1.794741970890788, 1.180256767879915,
        -0.57, 2.774810655432013, 2.086867367368360,
        2.198740394692808])

    assert_allclose(w_true, w)
    assert_allclose(z_true, z)


def test_predict():
    """Test the predict function in the Poisson class"""
    # design matrix
    X = np.array([[np.log(2.5), -1, -3],
                  [np.log(3.5), -2, -4],
                  [np.log(4.5), -3, -5],
                  [np.log(5.5), -4, -6]])

    poisson = Poisson()

    # test for NotFittedError
    assert_raises(NotFittedError, poisson.predict, X)

    # create "fit"
    poisson.coef_ = np.array([1, 0, 0])
    poisson.intercept_ = 0
    y_pred = poisson.predict(X)
    y_mode = np.array([2, 3, 4, 5])

    # test for predict
    assert_allclose(y_pred, y_mode)


def test_predict_mean():
    """Test the predict function in the Poisson class"""
    # design matrix
    X = np.array([[np.log(2.5), -1, -3],
                  [np.log(3.5), -2, -4],
                  [np.log(4.5), -3, -5],
                  [np.log(5.5), -4, -6]])

    poisson = Poisson()

    # test for NotFittedError
    assert_raises(NotFittedError, poisson.predict_mean, X)

    # create "fit"
    poisson.coef_ = np.array([1, 0, 0])
    poisson.intercept_ = 0
    y_pred = poisson.predict_mean(X)
    y_mean = np.array([2.5, 3.5, 4.5, 5.5])

    # test for predict
    assert_allclose(y_pred, y_mean)


def test_score_predictions():
    """Test the score predictions function in UoI Poisson."""
    X = np.array([[np.log(2), -1, -3],
                  [np.log(3), -2, -4],
                  [np.log(4), -3, -5],
                  [np.log(5), -4, -6]])
    y = 1. / np.log([2., 3., 4., 5.])
    support = np.array([True, False, False])
    n_samples = y.size

    # create fitter by hand
    fitter = Poisson()
    fitter.coef_ = np.array([1])
    fitter.intercept_ = 0

    uoi_fitter = UoI_Poisson()

    # test log-likelihood
    ll = uoi_fitter._score_predictions(
        metric='log',
        fitter=fitter,
        X=X, y=y, support=support)
    assert_allclose(ll, -2.5)

    # test information criteria
    total_ll = ll * n_samples
    aic = uoi_fitter._score_predictions(
        metric='AIC',
        fitter=fitter,
        X=X, y=y, support=support)

    assert_allclose(aic, 2 * total_ll - 2)

    aicc = uoi_fitter._score_predictions(
        metric='AICc',
        fitter=fitter,
        X=X, y=y, support=support)
    assert_allclose(aicc, aic - 2)

    bic = uoi_fitter._score_predictions(
        metric='BIC',
        fitter=fitter,
        X=X, y=y, support=support)
    assert_allclose(bic, 2 * total_ll - np.log(y.size))

    # test invalid metric
    assert_raises(ValueError,
                  uoi_fitter._score_predictions,
                  'fake',
                  fitter,
                  X, y, support)


def test_poisson_intercept_fitter_no_features():
    """Tests the PoissonInterceptFitterNoFeatures class."""
    y = np.array([0, 1, 2])
    poisson = PoissonInterceptFitterNoFeatures(y)

    X = np.random.normal(size=(10, y.size))
    assert_equal(poisson.intercept_, 0.)
    assert_equal(poisson.predict(X), 1.)
    assert_equal(poisson.predict_mean(X), 1.)


def test_poisson_loss_and_grad():
    """Tests the poisson loss and gradient function."""
    n_features = 5
    n_samples = 100

    # test with empty coefficients
    X = np.random.normal(loc=0, scale=1., size=(n_samples, n_features))
    y = np.random.normal(loc=0, scale=1, size=(n_samples))

    loss, grad = _poisson_loss_and_grad(np.zeros(n_features), X, y, 0.)

    assert_equal(loss, 1.)
    assert_allclose(grad, -np.dot(X.T, y - 1) / n_samples)

    # test with non-empty coefficients
    X = np.ones((n_samples, n_features))
    y = np.zeros(n_samples)

    loss, grad = _poisson_loss_and_grad(np.ones(n_features), X, y, 0)
    assert_equal(loss, np.exp(n_features).mean())
    assert_allclose(grad, np.mean(X * np.exp(n_features), axis=0))


def test_poisson_reg_params():
    """Test whether the upper bound on the regularization parameters correctly
    zero out the coefficients."""
    n_features = 5
    n_samples = 1000

    X, y, beta, intercept = make_poisson_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        random_state=2332)

    alpha_sets = [np.array([0.5]), np.array([1.0])]

    for alpha_set in alpha_sets:
        uoi_poisson = UoI_Poisson(alphas=alpha_set)
        reg_params = uoi_poisson.get_reg_params(X, y)
        alpha = reg_params[0]['alpha']
        l1_ratio = reg_params[0]['l1_ratio']

        # check that coefficients get set to zero
        poisson = Poisson(alpha=1.01 * alpha,
                          l1_ratio=l1_ratio,
                          standardize=False,
                          fit_intercept=True)
        poisson.fit(X, y)
        assert_equal(poisson.coef_, 0.)

        # check that coefficients below the bound are not set to zero
        poisson = Poisson(alpha=0.99 * alpha,
                          l1_ratio=l1_ratio,
                          standardize=False,
                          fit_intercept=True)
        poisson.fit(X, y)
        assert np.count_nonzero(poisson.coef_) > 0


def test_poisson_no_intercept():
    """Tests the Poisson fitter with no intercept."""
    n_features = 3
    n_samples = 10000

    # create data
    X, y, beta, _ = make_poisson_regression(n_samples=n_samples,
                                            n_features=n_features,
                                            n_informative=n_features,
                                            beta=np.array([0.5, 1.0, 1.5]),
                                            random_state=2332)

    # lbfgs
    poisson = Poisson(alpha=0., l1_ratio=0., fit_intercept=False,
                      solver='lbfgs', max_iter=5000)
    poisson.fit(X, y)
    assert_allclose(poisson.coef_, beta, rtol=0.5)

    # coordinate descent
    poisson = Poisson(alpha=0., l1_ratio=0., fit_intercept=False, solver='cd',
                      max_iter=5000)
    poisson.fit(X, y)
    assert_allclose(poisson.coef_, beta, rtol=0.5)

    # broken solver
    poisson = Poisson(alpha=0., l1_ratio=0., fit_intercept=False, solver='ruff',
                      max_iter=5000)
    assert_raises(ValueError, poisson.fit, X, y)


def test_poisson_warm_start():
    """Tests if the warm start initializes coefficients correctly."""
    n_features = 3
    n_samples = 10000

    # create data
    X, y, beta, _ = make_poisson_regression(n_samples=n_samples,
                                            n_features=n_features,
                                            n_informative=n_features,
                                            beta=np.array([0.5, 1.0, 1.5]),
                                            random_state=2332)

    # lbfgs
    poisson = Poisson(alpha=0., l1_ratio=0., fit_intercept=False,
                      solver='lbfgs', max_iter=5000, warm_start=True)
    poisson.fit(X, y)
    first_coef = poisson.coef_

    poisson.coef_ = np.zeros(n_features)
    poisson.fit(X, y)
    second_coef = poisson.coef_

    assert_allclose(first_coef, second_coef, rtol=0.1)

    # cd
    poisson = Poisson(alpha=0., l1_ratio=0., fit_intercept=False,
                      solver='cd', max_iter=5000, warm_start=True)
    poisson.fit(X, y)
    first_coef = poisson.coef_

    poisson.coef_ = np.zeros(n_features)
    poisson.fit(X, y)
    second_coef = poisson.coef_

    assert_allclose(first_coef, second_coef, rtol=0.1)


def test_poisson_with_intercept():
    """Tests the Poisson fitter with no intercept."""
    n_features = 3
    n_samples = 10000

    # create data
    X, y, beta, intercept = make_poisson_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        beta=np.array([0.5, 1.0, 1.5]),
        include_intercept=True,
        random_state=2332)

    # lbfgs
    poisson = Poisson(alpha=0., l1_ratio=0., fit_intercept=True,
                      solver='lbfgs', max_iter=5000)
    poisson.fit(X, y)

    assert_allclose(poisson.coef_, beta, rtol=0.5)
    assert_allclose(poisson.intercept_, intercept, rtol=0.5)

    # coordinate descent
    poisson = Poisson(alpha=0., l1_ratio=0., fit_intercept=True, solver='cd',
                      max_iter=5000)
    poisson.fit(X, y)

    assert_allclose(poisson.coef_, beta, rtol=0.5)
    assert_allclose(poisson.intercept_, intercept, rtol=0.5)


def test_poisson_standardize():
    """Tests the Poisson fitter `standardize=True`."""
    n_features = 3
    n_samples = 200

    # create data
    X, y, beta, intercept = make_poisson_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        beta=np.array([0.5, 1.0, 1.5]),
        include_intercept=True,
        random_state=2332)

    # lbfgs
    poisson = Poisson(alpha=0., l1_ratio=0., fit_intercept=True,
                      solver='lbfgs', standardize=True)
    poisson.fit(X, y)


def test_poisson_with_sparsity():
    """Tests the Poisson fitter with no intercept."""
    n_features = 3
    n_samples = 10000

    # create data
    X, y, beta, _ = make_poisson_regression(n_samples=n_samples,
                                            n_features=n_features,
                                            n_informative=n_features,
                                            beta=np.array([0., 1.0, 1.5]),
                                            random_state=2332)

    # lbfgs
    poisson = Poisson(alpha=0.1, l1_ratio=1., fit_intercept=False,
                      solver='lbfgs', max_iter=5000)
    poisson.fit(X, y)

    assert_equal(np.abs(poisson.coef_[0]), 0)

    # coordinate descent
    poisson = Poisson(alpha=0.1, l1_ratio=1., fit_intercept=False, solver='cd',
                      max_iter=5000)
    poisson.fit(X, y)

    assert_equal(np.abs(poisson.coef_[0]), 0.)


def test_poisson_intercept_fitter_all_zeros():
    """Tests that the Poisson intercept fitter properly assigns an intercept
    without warning when the response variable is all zeros."""
    n_samples = 10

    y = np.zeros(n_samples)

    poisson = PoissonInterceptFitterNoFeatures(y)

    assert_equal(poisson.intercept_, -np.inf)


def test_Poisson_response_constant():
    """Test that UoI Poisson correctly fits the data when the response variable
    is constant."""
    n_features = 5
    n_samples = 100

    X = np.random.normal(size=(n_samples, n_features))
    y = np.zeros(n_samples)

    poisson = Poisson()
    poisson.fit(X, y)

    assert_equal(poisson.intercept_, -np.inf)
    assert_equal(poisson.coef_, np.zeros(n_features))

    y += 1.
    poisson.fit(X, y)

    assert_equal(poisson.intercept_, 0.)
    assert_equal(poisson.coef_, np.zeros(n_features))


def test_UoI_Poisson_all_zeros():
    """Test that UoI Poisson correctly fits the data when the response variable
    is all zeros."""
    n_features = 5
    n_samples = 100

    X = np.random.normal(size=(n_samples, n_features))
    y = np.zeros(n_samples)

    poisson = UoI_Poisson()
    poisson.fit(X, y)

    assert_equal(poisson.intercept_, -np.inf)
    assert_equal(poisson.coef_, np.zeros(n_features))


def test_UoI_Poisson():
    """Tests the UoI Poisson fitter with lbfgs solver."""
    n_features = 3
    n_samples = 5000

    # create data
    X, y, beta, _ = make_poisson_regression(n_samples=n_samples,
                                            n_features=n_features,
                                            n_informative=n_features,
                                            beta=np.array([0., 1.0, 1.5]),
                                            random_state=2332)

    # lbfgs
    poisson = UoI_Poisson(n_lambdas=48, n_boots_sel=30, n_boots_est=30,
                          alphas=np.array([1.0]), warm_start=False)
    poisson.fit(X, y)

    assert_allclose(poisson.coef_, beta, atol=0.5)


def test_fit_intercept():
    """Tests whether `include_intercept` in passed through to the linear models.
    """
    poi = UoI_Poisson(fit_intercept=True)
    assert poi._selection_lm.fit_intercept
    assert poi._estimation_lm.fit_intercept

    poi = UoI_Poisson(fit_intercept=False)
    assert not poi._selection_lm.fit_intercept
    assert not poi._estimation_lm.fit_intercept
