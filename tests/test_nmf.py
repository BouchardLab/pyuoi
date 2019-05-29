import numpy as np

from numpy.testing import assert_array_equal
from pyuoi.decomposition import UoI_NMF


def test_UoI_NMF_initialization():
    """Tests the initialization of UoI NMF."""
    n_boots = 30
    ranks = 10
    uoi = UoI_NMF(n_boots=n_boots, ranks=ranks)

    assert_array_equal(uoi.ranks, np.arange(2, ranks + 1))
    assert uoi.nmf.solver == 'mu'
    assert uoi.nmf.beta_loss == 'kullback-leibler'
    assert uoi.cluster.min_samples == n_boots / 2


def test_UoI_NMF_fitting():
    """Tests that the fitting procedure of UoI NMF runs without error."""
    W = np.random.randint(0, high=2, size=(500, 5))
    H = np.random.randint(0, high=2, size=(5, 2))
    X = np.dot(W, H)

    n_boots = 1
    ranks = 5
    uoi = UoI_NMF(n_boots=n_boots,
                  ranks=[ranks],
                  nmf_max_iter=1000,
                  random_state=2332)
    uoi.fit(X)