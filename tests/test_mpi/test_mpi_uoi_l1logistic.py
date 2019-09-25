import pytest
import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from pyuoi import UoI_L1Logistic
from pyuoi.datasets import make_classification


@pytest.mark.skipif(MPI is None, reason='MPI not installed.')
def test_l1logistic_binary():
    """Test that binary L1 Logistic runs in the UoI framework."""
    n_inf = 10
    X, y, w, b = make_classification(n_samples=200,
                                     random_state=6,
                                     n_informative=n_inf,
                                     n_features=20,
                                     w_scale=4.,
                                     include_intercept=True)

    l1log = UoI_L1Logistic(random_state=10, comm=MPI.COMM_WORLD).fit(X, y)
    assert (np.sign(abs(w)) == np.sign(abs(l1log.coef_))).mean() >= .7


@pytest.mark.skipif(MPI is None, reason='MPI not installed.')
def test_l1logistic_multiclass():
    """Test that multiclass L1 Logistic runs in the UoI framework when all
       classes share a support."""
    n_features = 20
    n_inf = 10
    X, y, w, b = make_classification(n_samples=200,
                                     random_state=10,
                                     n_classes=5,
                                     n_informative=n_inf,
                                     n_features=n_features,
                                     shared_support=True,
                                     w_scale=4.)
    l1log = UoI_L1Logistic(comm=MPI.COMM_WORLD).fit(X, y)
    assert (np.sign(abs(w)) == np.sign(abs(l1log.coef_))).mean() >= .8
