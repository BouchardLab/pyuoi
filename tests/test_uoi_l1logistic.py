import pytest

import numpy as np

from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from sklearn.preprocessing import normalize

from pyuoi import UoI_L1Logistic
from pyuoi.linear_model.logistic import fit_intercept_fixed_coef
from pyuoi.utils import sigmoid, softmax


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
    intercept *= w_scale * 2.

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
        l1log = UoI_L1Logistic(fit_intercept=fi).fit(X, y)
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
                                     n_features=10,
                                     w_scale=4.,
                                     include_intercept=True)

    for method in methods:
        l1log = UoI_L1Logistic(random_state=10,
                               estimation_score=method).fit(X, y)
        assert (np.sign(w) == np.sign(l1log.coef_)).mean() >= .8
        assert_allclose(w, l1log.coef_, rtol=.5, atol=.5)


@pytest.mark.skip(reason="Logistic is not currently finished")
def test_l1logistic_multiclass():
    """Test that multiclass L1 Logistic runs in the UoI framework when all
       classes share a support."""
    n_features = 4
    n_inf = 3
    X, y, w = make_classification(n_samples=1000,
                                  random_state=6,
                                  n_classes=3,
                                  n_informative=n_inf,
                                  n_features=n_features,
                                  shared_support=True)
    X = normalize(X, axis=0)
    l1log = UoI_L1Logistic().fit(X, y)
    print()
    print(w)
    print(l1log.coef_)
    assert_array_equal(np.sign(w), np.sign(l1log.coef_))
    assert_allclose(w, l1log.coef_, atol=.5)


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
