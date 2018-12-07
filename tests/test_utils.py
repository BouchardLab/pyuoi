import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal_nulp

from pyuoi.linear_model.utils import stability_selection_to_threshold


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
