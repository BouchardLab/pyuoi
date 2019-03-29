import numpy as np

from numpy.testing import assert_array_equal, assert_allclose, assert_equal

from pyuoi import UoI_L1Logistic
from pyuoi.linear_model.logistic import (fit_intercept_fixed_coef,
                                         MaskedCoefLogisticRegression)
from pyuoi.utils import make_classification


def test_fit_intercept_fixed_coef():
    """Test that the intercept in fit correctly for fixed coefficients."""
    X = np.zeros((6, 5))
    coef = np.ones((1, 5))
    y = np.ones(6)
    y[:3] = 0.
    b = fit_intercept_fixed_coef(X, coef, y, 2)
    assert_allclose(b, 0.)


def test_l1logistic_intercept():
    """Test that binary L1 Logistic fits an intercept when run."""
    for fi in [True, False]:
        X, y, w, b = make_classification(n_samples=100,
                                         random_state=11,
                                         n_features=4,
                                         w_scale=4.,
                                         include_intercept=fi)
        l1log = UoI_L1Logistic(fit_intercept=fi,
                               n_boots_sel=3,
                               n_boots_est=3).fit(X, y)
        if not fi:
            assert_array_equal(l1log.intercept_, 0.)
        else:
            l1log.intercept_


def test_l1logistic_binary():
    """Test that binary L1 Logistic runs in the UoI framework."""
    n_inf = 4
    methods = ('acc', 'log')
    X, y, w, b = make_classification(n_samples=2000,
                                     random_state=6,
                                     n_informative=n_inf,
                                     n_features=20,
                                     w_scale=4.,
                                     include_intercept=True)

    for method in methods:
        l1log = UoI_L1Logistic(random_state=10,
                               estimation_score=method).fit(X, y)
        print()
        print(w)
        print(l1log.coef_)
        assert (np.sign(w) == np.sign(l1log.coef_)).mean() >= .8
        assert_allclose(w, l1log.coef_, rtol=.5, atol=.5)


def test_l1logistic_multiclass():
    """Test that multiclass L1 Logistic runs in the UoI framework when all
       classes share a support."""
    n_features = 20
    n_inf = 5
    X, y, w, b = make_classification(n_samples=1000,
                                     random_state=10,
                                     n_classes=5,
                                     n_informative=n_inf,
                                     n_features=n_features,
                                     shared_support=True,
                                     w_scale=4.)
    l1log = UoI_L1Logistic(estimation_score='log').fit(X, y)
    assert (np.sign(w.ravel()) == np.sign(l1log.coef_.ravel())).mean() >= .8
    assert_allclose(w, l1log.coef_, rtol=.5, atol=3.)


def test_l1logistic_multiclass_not_shared():
    """Test that multiclass L1 Logistic runs in the UoI framework when all
       classes share a support."""
    n_features = 5
    n_inf = 4
    X, y, w, b = make_classification(n_samples=100,
                                     random_state=10,
                                     n_classes=3,
                                     n_informative=n_inf,
                                     n_features=n_features,
                                     shared_support=False,
                                     w_scale=4.)
    l1log = UoI_L1Logistic(estimation_score='log',
                           shared_support=False).fit(X, y)
    print()
    print(w.ravel())
    print(l1log.coef_.ravel())
    assert (np.sign(w.ravel()) == np.sign(l1log.coef_.ravel())).mean() >= .8
    assert_allclose(w, l1log.coef_, rtol=.5, atol=3.)


def test_masked_logistic():
    """Test the masked logistic regression class."""
    for shared_support in [True, False]:
        for n_classes in [2, 3]:
            n_features = 40
            n_inf = 5
            X, y, w, b = make_classification(n_samples=20000,
                                             random_state=10,
                                             n_classes=n_classes,
                                             n_informative=n_inf,
                                             n_features=n_features,
                                             shared_support=shared_support,
                                             w_scale=4.)
            mask = np.squeeze(np.logical_not(np.equal(w, 0)))
            lr = MaskedCoefLogisticRegression().fit(X, y, coef_mask=mask)
            coef_idxs = np.flatnonzero(np.logical_not(np.equal(lr.coef_, 0.)))
            mask_idxs = np.flatnonzero(np.logical_not(np.equal(mask, 0)))
            assert set(coef_idxs.tolist()).issubset(set(mask_idxs.tolist()))
            assert_allclose(w, lr.coef_, rtol=.5, atol=3.)


def test_estimation_score_usage():
    """Test the ability to change the estimation score in UoI L1Logistic"""
    methods = ('acc', 'log')
    X, y, w, b = make_classification(n_samples=100,
                                     random_state=6,
                                     n_informative=2,
                                     n_features=6)
    scores = []
    for method in methods:
        l1log = UoI_L1Logistic(random_state=12, estimation_score=method)
        assert_equal(l1log.estimation_score, method)
        l1log.fit(X, y)
        score = np.max(l1log.scores_)
        scores.append(score)
    assert_equal(len(set(scores)), len(methods))


def test_set_random_state():
    """Tests whether random states are handled correctly."""
    X, y, w, b = make_classification(n_samples=100,
                                     random_state=60,
                                     n_informative=4,
                                     n_features=5,
                                     w_scale=4.)
    # same state
    l1log_0 = UoI_L1Logistic(random_state=13)
    l1log_1 = UoI_L1Logistic(random_state=13)
    l1log_0.fit(X, y)
    l1log_1.fit(X, y)
    assert_array_equal(l1log_0.coef_, l1log_1.coef_)

    # different state
    l1log_1 = UoI_L1Logistic(random_state=14)
    l1log_1.fit(X, y)
    assert not np.array_equal(l1log_0.coef_, l1log_1.coef_)

    # different state, not set
    l1log_0 = UoI_L1Logistic()
    l1log_1 = UoI_L1Logistic()
    l1log_0.fit(X, y)
    l1log_1.fit(X, y)
    assert not np.array_equal(l1log_0.coef_, l1log_1.coef_)
