import numpy as np
from numpy.testing import assert_array_equal

from pyuoi.UoI_Lasso import UoI_Lasso
from pyuoi.selection import stability_selection_to_threshold


def test_stability_selection_to_threshold():
    """Test different input types for stability selection."""
    n_boots_sel = 48

    # stability selection is a single float
    test_float = 0.5
    selection_thresholds = stability_selection_to_threshold(test_float, n_boots_sel)
    assert_array_equal(
        selection_thresholds,
        np.array([int(test_float * n_boots_sel)])
    )

    # stability selection is a single integer
    test_int = 36
    selection_thresholds = stability_selection_to_threshold(test_int, n_boots_sel)
    assert_array_equal(
        selection_thresholds,
        np.array([test_int])
    )

    # stability selection is a list of floats
    test_floats = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    selection_thresholds = stability_selection_to_threshold(test_floats, n_boots_sel)
    print((n_boots_sel * np.array(test_floats)).astype('int'))
    assert_array_equal(
        selection_thresholds,
        (n_boots_sel * np.array(test_floats)).astype('int')
    )
