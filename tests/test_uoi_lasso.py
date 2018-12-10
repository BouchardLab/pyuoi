import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal_nulp

from pyuoi import UoI_Lasso


def test_variable_selection():
    """Test basic functionality of UoI_Lasso and that it finds the write model"""
    from sklearn.datasets import make_regression
    X, y, w = make_regression(coef=True, random_state=1)
    lasso = UoI_Lasso()
    lasso.fit(X, y)
    true_coef = np.nonzero(w)[0]
    fit_coef = np.nonzero(lasso.coef_)[0]
    assert_array_equal(true_coef, fit_coef)
    assert_array_almost_equal_nulp(true_coef, fit_coef)


def test_uoi_lasso_small():
    # TODO: test uoi lasso on a tiny, noiseless dataset.
    pass


def test_uoi_lasso_to_lassoCV():
    # TODO: UoI Lasso and LassoCV are equivalent under a certain set of
    # hyperparameters. Test for this condition.
    pass


def test_lambdas():
    # TODO: test that UoI Lasso obtains the correct set of lambda parameters
    # to sweep over.
    pass


def test_intercept():
    # TODO: test that UoI Lasso properly calculates the intercept when
    # fit_intercept = True.
    pass


def test_bootstraps():
    # TODO: test that UoI Lasso is properly taking samples of the data when
    # a RandomState is provided.
    pass


def test_lasso_selection_sweep():
    # TODO: test the lasso selection sweep.
    pass


def test_score_predictions():
    # TODO: test the score_predictions function.
    pass


def test_get_n_coef():
    # TODO: test the get_n_coef function.
    pass
