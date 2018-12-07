import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import normalize
from sklearn.svm import l1_min_c
from pyuoi import UoI_L1Logistic


def test_l1logistic_binary():
    """Test that binary L1 Logistic runs in the UoI framework"""
    n_inf = 2
    X, y = make_classification(n_samples=100,
                               random_state=6,
                               n_informative=n_inf,
                               n_features=4,
                               n_repeated=0,
                               n_redundant=0)
    X = normalize(X, axis=0)
    l1log = UoI_L1Logistic().fit(X, y)
    # ensure shape conforms to sklearn convention
    assert l1log.coef_.shape == (1, 4)
    # check that we have weights on at least one of the informative features
    assert np.abs(np.sum(l1log.coef_[:, :n_inf])) > 0.0

def test_l1logistic_multiclass():
    """Test that multiclass L1 Logistic runs in the UoI framework"""
    n_inf = 3
    X, y = make_classification(n_samples=100,
                               random_state=6,
                               n_classes=3,
                               n_informative=n_inf,
                               n_features=4,
                               n_repeated=0,
                               n_redundant=0)
    X = normalize(X, axis=0)
    l1log = UoI_L1Logistic().fit(X, y)
    # ensure shape conforms to sklearn convention
    assert l1log.coef_.shape == (3, 4)
    # check that we have weights on at least one of the informative features
    assert np.abs(np.sum(l1log.coef_[:, :n_inf])) > 0.0
