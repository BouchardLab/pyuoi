import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal_nulp
from numpy.testing import assert_almost_equal

from pyuoi.linear_model.utils import stability_selection_to_threshold
from pyuoi.utils import BIC, AIC, AICc

def test_stability_selection_to_threshold_int():
    n_boots_sel = 48
    # stability selection is a single integer
    test_int = 36
    selection_thresholds = stability_selection_to_threshold(
        test_int, n_boots_sel)
    assert_array_equal(selection_thresholds, [36])


def test_stability_selection_to_threshold_float():
    n_boots_sel = 48
    # stability selection is a single float
    test_float = 0.5
    selection_thresholds = stability_selection_to_threshold(
        test_float, n_boots_sel)
    assert_array_equal(selection_thresholds, [24])


def test_stability_selection_to_threshold_ints():
    # TODO: test stability selection function using a list of ints
    pass


def test_stability_selection_to_threshold_floats():
    n_boots_sel = 48
    # stability selection is a list of floats
    test_floats = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    selection_thresholds = stability_selection_to_threshold(
        test_floats, n_boots_sel)
    assert_array_equal(selection_thresholds, [24, 28, 33, 38, 43, 48])


def test_stability_selection_to_threshold_ints_np():
    # TODO: test stability selection using a numpy array of ints
    pass


def test_stability_selection_to_threshold_floats_np():
    # TODO: test stability selection using a numpy array of floats
    pass


def test_stability_selection_edge_case():
    # TODO: check that stability selection passes the edge case where
    # stability_selection = 1 and n_boots = 1
    pass


def test_intersection():
    # TODO: test the intersection function for linear models by hand
    pass


def test_intersection_with_stability_selection():
    # TODO: test the intersection function using stability selection
    pass


def test_intersection_duplicates():
    # TODO: test the intersection function specifically to ensure duplicate
    # supports are not outputted.
    pass


def test_bic():
    n = 20
    k = 10
    np.random.seed(15)
    y_true = np.random.normal(0, 1, n)
    y_pred = y_true + np.sqrt(np.e**(1/n))
    expected = 1 + k*np.log(n)
    assert_almost_equal(expected, BIC(y_true, y_pred, k))


def test_aic():
    n = 20
    k = 10
    np.random.seed(15)
    y_true = np.random.normal(0, 1, n)
    y_pred = y_true + np.sqrt(np.e**(1/n))
    expected = 1 + 2*k
    assert_almost_equal(expected, AIC(y_true, y_pred, k))


def test_aicc():
    n = 20
    k = 10
    np.random.seed(15)
    y_true = np.random.normal(0, 1, n)
    y_pred = y_true + np.sqrt(np.e**(1/n))
    expected = 1 + 2*k + (2*k**2 + 2*k)/(n - k - 1)
    assert_almost_equal(expected, AICc(y_true, y_pred, k))





