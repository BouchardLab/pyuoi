import numpy as np
from numpy.testing import assert_array_equal

from pyuoi import UoI_Lasso


def test_stability_selection_to_threshold_float():
    n_boots_sel = 48
    # stability selection is a single float
    test_float = 0.5
    selection_thresholds = UoI_Lasso._stability_selection_to_threshold(test_float, n_boots_sel)
    assert_array_equal(selection_thresholds, [24])

def test_stability_selection_to_threshold_int():
    n_boots_sel = 48
    # stability selection is a single integer
    test_int = 36
    selection_thresholds = UoI_Lasso._stability_selection_to_threshold(test_int, n_boots_sel)
    assert_array_equal(selection_thresholds, [36])

def test_stability_selection_to_threshold_floats():
    n_boots_sel = 48
    # stability selection is a list of floats
    test_floats = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    selection_thresholds = UoI_Lasso._stability_selection_to_threshold(test_floats, n_boots_sel)
    assert_array_equal(selection_thresholds, [24, 28, 33, 38, 43, 48])
