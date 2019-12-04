import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal_nulp
from sklearn.datasets import make_regression
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from pyuoi.datasets import make_classification, make_poisson_regression
from pyuoi.linear_model import (UoI_Lasso,
                                UoI_L1Logistic,
                                UoI_ElasticNet,
                                UoI_Poisson)


@pytest.mark.skipif(MPI is None, reason='MPI not installed.')
def test_variable_selection_lasso():
    """Test basic functionality of UoI_Lasso and that it finds right model"""
    X, y, w = make_regression(coef=True, random_state=1)
    lasso = UoI_Lasso(comm=MPI.COMM_WORLD)
    lasso.fit(X, y)
    true_coef = np.nonzero(w)[0]
    fit_coef = np.nonzero(lasso.coef_)[0]
    assert_array_equal(true_coef, fit_coef)
    assert_array_almost_equal_nulp(true_coef, fit_coef)


@pytest.mark.skipif(MPI is None, reason='MPI not installed.')
def test_variable_selection_enet():
    """Test basic functionality of UoI_Lasso and that it finds right model"""
    X, y, w = make_regression(coef=True, random_state=1)
    enet = UoI_ElasticNet(comm=MPI.COMM_WORLD)
    enet.fit(X, y)
    true_coef = np.nonzero(w)[0]
    fit_coef = np.nonzero(enet.coef_)[0]
    assert_array_equal(true_coef, fit_coef)
    assert_array_almost_equal_nulp(true_coef, fit_coef)


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


@pytest.mark.skipif(MPI is None, reason='MPI not installed.')
def test_poisson():
    """Test basic functionality of UoI_Lasso and that it finds right model"""
    n_features = 20
    n_inf = 10
    X, y, w, b = make_poisson_regression(n_samples=200,
                                         n_features=n_features,
                                         n_informative=n_inf,
                                         random_state=10)
    poisson = UoI_Poisson(comm=MPI.COMM_WORLD)
    poisson.fit(X, y)
    assert (np.sign(abs(w)) == np.sign(abs(poisson.coef_))).mean() >= .6
