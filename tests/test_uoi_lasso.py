import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal_nulp

from pyuoi.linear_model.utils import stability_selection_to_threshold
from pyuoi import UoI_Lasso


def test_stability_selection_to_threshold_float():
    n_boots_sel = 48
    # stability selection is a single float
    test_float = 0.5
    selection_thresholds = stability_selection_to_threshold(test_float, n_boots_sel)
    assert_array_equal(selection_thresholds, [24])

def test_stability_selection_to_threshold_int():
    n_boots_sel = 48
    # stability selection is a single integer
    test_int = 36
    selection_thresholds = stability_selection_to_threshold(test_int, n_boots_sel)
    assert_array_equal(selection_thresholds, [36])

def test_stability_selection_to_threshold_floats():
    n_boots_sel = 48
    # stability selection is a list of floats
    test_floats = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    selection_thresholds = stability_selection_to_threshold(test_floats, n_boots_sel)
    assert_array_equal(selection_thresholds, [24, 28, 33, 38, 43, 48])

def test_intersection_happy_path():
    #TODO: fill this in
    pass

def test_intersection_edge_case1():
    #TODO: fill this in and rename test to be more explanatory
    pass

def test_intersection_edge_case2():
    #TODO: fill this in and rename test to be more explanatory
    pass

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
