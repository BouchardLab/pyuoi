import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from pyuoi.utils import log_likelihood_glm
from pyuoi.utils import (AIC, BIC, AICc)


def test_ll():
    """Tests that the log-likelihood for generalized linear models is correctly
    calculated."""

    # identity
    y_true = np.array([1, 2, 3])
    y_pred = np.array([np.e + 1, np.e + 2, np.e + 3])
    ll = log_likelihood_glm('normal', y_true, y_pred)
    assert_almost_equal(ll, -4.5)

    # poisson
    y_true = np.array([1 / np.log(2.), 1 / np.log(3.), 1 / np.log(4.)])
    y_pred = np.array([2., 3., 4.])
    ll = log_likelihood_glm('poisson', y_true, y_pred)
    assert_almost_equal(ll, -2)


def test_ll_error():
    """Tests that the log-likelihood function correctly raises an error when an
    incorrect string is passed as a parameter."""

    y_true = np.array([1., 2., 3.])
    y_pred = np.array([3., 4., 5.])

    assert_raises(ValueError,
                  log_likelihood_glm,
                  'error',
                  y_true,
                  y_pred)


def test_information_criteria():
    """Tests the information criteria (AIC, AICc, BIC) functions."""
    ll = -1.
    n_features = 5
    n_samples = 1000

    aic = AIC(ll, n_features)
    assert_equal(aic, 12.)

    aicc = AICc(ll, n_features, n_samples)
    assert_equal(aicc, 12. + 30. / 497.)

    # additional test: AICc should equal AIC if the number of samples is one
    # greater than the number of features
    aicc = AICc(ll, n_features, n_features + 1)
    assert_equal(aicc, aic)

    bic = BIC(ll, n_features, n_samples)
    assert_equal(bic, 5 * np.log(1000) + 2)
