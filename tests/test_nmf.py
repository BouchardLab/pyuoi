import pytest
import numpy as np

from numpy.testing import assert_array_equal
from pyuoi.decomposition import UoI_NMF
from pyuoi.decomposition.utils import dissimilarity


def setup():
    W = np.random.randint(0, high=2, size=(500, 5))
    H = np.random.randint(0, high=2, size=(5, 2))
    X = np.dot(W, H)
    noise = np.random.normal(loc=0, scale=0.5, size=X.shape)**2
    X = X + noise

    n_boots = 2
    ranks = 5
    uoi = UoI_NMF(n_boots=n_boots,
                  ranks=[ranks],
                  nmf_max_iter=1000,
                  random_state=2332)
    return uoi, X


def test_dissimilarity():
    """Test the dissimilarity function."""
    k = 5
    n_features = 20

    # same bases should be a dissimilarity of zero
    H1 = np.random.randint(low=0, high=3, size=(k, n_features))
    H2 = np.copy(H1)
    assert np.allclose(dissimilarity(H1, H2), 0.)


@pytest.mark.fast
def test_UoI_NMF_initialization():
    """Tests the initialization of UoI NMF."""
    n_boots = 30
    ranks = 10
    uoi = UoI_NMF(n_boots=n_boots, ranks=ranks)
    assert_array_equal(uoi.ranks, np.arange(2, ranks + 1))
    assert uoi.nmf.solver == 'mu'
    assert uoi.nmf.beta_loss == 'kullback-leibler'
    assert uoi.cluster.min_samples == n_boots / 2


@pytest.mark.fast
def test_UoI_NMF_fit():
    """Tests that the fitting procedure of UoI NMF runs without error."""
    uoi, X = setup()
    uoi.fit(X)
    assert hasattr(uoi, 'components_')


@pytest.mark.fast
def test_UoI_NMF_fit_no_dissimilarity():
    """Tests that the fitting procedure of UoI NMF runs without error, when
    the algorithm does not use dissimilarity to choose a rank."""
    uoi, X = setup()
    uoi.set_params(use_dissimilarity=False)
    uoi.fit(X)
    assert hasattr(uoi, 'components_')


@pytest.mark.fast
def test_UoI_NMF_transform():
    """Tests that the transform procedure of UoI NMF runs without error."""
    uoi, X = setup()
    X_tfm = uoi.fit_transform(X)
    assert hasattr(uoi, 'components_')
    assert X_tfm is not None


@pytest.mark.fast
def test_UoI_NMF_reconstruction_error():
    """Tests that a reconstruction error is calculated when data is
    transformed."""
    uoi, X = setup()
    uoi.fit(X)
    X_tfm = uoi.transform(X, reconstruction_err=True)
    assert hasattr(uoi, 'components_')
    assert hasattr(uoi, 'reconstruction_err_')
    assert uoi.reconstruction_err_ is not None
    assert X_tfm is not None
