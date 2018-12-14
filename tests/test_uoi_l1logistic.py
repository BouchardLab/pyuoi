import numpy as np

from numpy.testing import assert_array_equal, assert_allclose

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import normalize

from pyuoi import UoI_L1Logistic

import pytest


def softmax(y, axis=-1):
    yp = y - y.max(axis=axis, keepdims=True)
    epy = np.exp(yp)
    return epy / np.sum(epy, axis=axis, keepdims=True)


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_classes=2, random_state=None):
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    n_not_informative = n_features - n_informative

    X = random_state.randn(n_samples, n_features)
    X -= X.mean(axis=-1, keepdims=True)
    X /= X.std(axis=-1, keepdims=True)

    if n_classes > 2:
        w = random_state.randn(n_features, n_classes)
        if n_not_informative > 0:
            for ii in range(n_classes):
                idxs = random_state.permutation(n_features)[:n_not_informative]
                w[idxs, ii * np.ones_like(idxs, dtype=int)] = 0.
    else:
        w = random_state.randn(n_features, 1)
        if n_not_informative > 0:
            idxs = random_state.permutation(n_features)[:n_not_informative]
            w[idxs] = 0.

    log_p = X.dot(w)
    if n_classes > 2:
        p = softmax(log_p)
        y = np.array([random_state.multinomial(1, pi) for pi in p])
        y = y.argmax(axis=-1)
    else:
        p = sigmoid(np.squeeze(log_p))
        y = np.array([random_state.binomial(1, pi) for pi in p])

    return X, y, w.T

def test_l1logistic_binary():
    """Test that binary L1 Logistic runs in the UoI framework"""
    n_inf = 4
    X, y, w = make_classification(n_samples=1000,
                               random_state=6,
                               n_informative=n_inf,
                               n_features=6)
    #X = normalize(X, axis=0)
    l1log = UoI_L1Logistic(random_state=10).fit(X, y)
    # ensure shape conforms to sklearn convention
    assert_array_equal(np.sign(w), np.sign(l1log.coef_))
    assert_allclose(w, l1log.coef_, atol=.5)


@pytest.mark.skip(reason="Logistic is not currently finished")
def test_l1logistic_multiclass():
    """Test that multiclass L1 Logistic runs in the UoI framework"""
    n_inf = 3
    X, y, w = make_classification(n_samples=100,
                               random_state=6,
                               n_classes=3,
                               n_informative=n_inf,
                               n_features=4)
    #X = normalize(X, axis=0)
    l1log = UoI_L1Logistic().fit(X, y)
    print(w)
    print(l1log.coef_)
    # ensure shape conforms to sklearn convention
    assert l1log.coef_.shape == (3, 4)
    # check that we have weights on at least one of the informative features
    assert np.abs(np.sum(l1log.coef_[:, :n_inf])) > 0.0
