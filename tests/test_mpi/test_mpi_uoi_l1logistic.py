import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from mpi4py import MPI

from pyuoi import UoI_L1Logistic
from pyuoi.utils import make_classification


def test_l1logistic_binary():
    """Test that binary L1 Logistic runs in the UoI framework."""
    n_inf = 4
    X, y, w, b = make_classification(n_samples=1000,
                                     random_state=6,
                                     n_informative=n_inf,
                                     n_features=6)

    l1log = UoI_L1Logistic(random_state=10, comm=MPI.COMM_WORLD).fit(X, y)
    assert_array_equal(np.sign(w), np.sign(l1log.coef_))
    assert_allclose(w, l1log.coef_, atol=.5, rtol=.5)


@pytest.mark.skip(reason="Logistic is not currently finished")
def test_l1logistic_multiclass():
    """Test that multiclass L1 Logistic runs in the UoI framework when all
       classes share a support."""
    n_features = 4
    n_inf = 3
    X, y, w, b = make_classification(n_samples=1000,
                                     random_state=6,
                                     n_classes=3,
                                     n_informative=n_inf,
                                     n_features=n_features,
                                     shared_support=True)
    l1log = UoI_L1Logistic(comm=MPI.COMM_WORLD).fit(X, y)
    print()
    print(w)
    print(l1log.coef_)
    assert_array_equal(np.sign(w), np.sign(l1log.coef_))
    assert_allclose(w, l1log.coef_, atol=.5)
