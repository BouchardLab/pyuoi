import pytest

import numpy as np

from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from sklearn.preprocessing import normalize

from pyuoi import UoI_L1Logistic


def softmax(y, axis=-1):
    yp = y - y.max(axis=axis, keepdims=True)
    epy = np.exp(yp)
    return epy / np.sum(epy, axis=axis, keepdims=True)


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_classes=2, shared_support=False, random_state=None):
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

    log_p = X.dot(w)
    if n_classes > 2:
        p = softmax(log_p)
        y = np.array([rng.multinomial(1, pi) for pi in p])
        y = y.argmax(axis=-1)
    else:
        p = sigmoid(np.squeeze(log_p))
        y = np.array([rng.binomial(1, pi) for pi in p])

    return X, y, w.T


def test_l1logistic_binary():
    """Test that binary L1 Logistic runs in the UoI framework."""
    n_inf = 4
    X, y, w = make_classification(n_samples=1000,
                                  random_state=6,
                                  n_informative=n_inf,
                                  n_features=6)

    l1log = UoI_L1Logistic(random_state=10).fit(X, y)
    assert_array_equal(np.sign(w), np.sign(l1log.coef_))
    assert_allclose(w, l1log.coef_, atol=.5)


def test_estimation_score_usage():
    """Test the ability to change the estimation score in UoI L1Logistic"""
    methods = ('acc', 'log')
    X, y, w = make_classification(n_samples=100,
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
    X, y, w = make_classification(n_samples=100,
                                  random_state=60,
                                  n_informative=3,
                                  n_features=5)
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


@pytest.mark.skip(reason="Logistic is not currently finished")
def test_l1logistic_multiclass():
    """Test that multiclass L1 Logistic runs in the UoI framework"""
    n_inf = 3
    X, y, w = make_classification(n_samples=100,
                                  random_state=6,
                                  n_classes=3,
                                  n_informative=n_inf,
                                  n_features=4)
    X = normalize(X, axis=0)
    l1log = UoI_L1Logistic().fit(X, y)
    # ensure shape conforms to sklearn convention
    assert l1log.coef_.shape == (3, 4)
    # check that we have weights on at least one of the informative features
    assert np.abs(np.sum(l1log.coef_[:, :n_inf])) > 0.0
