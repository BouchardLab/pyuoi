import numpy as np

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_raises
from pyuoi.decomposition import CUR, UoI_CUR
from pyuoi.decomposition.utils import (column_select,
                                       stability_selection_to_threshold)

X = np.array([
    [0, 0, 0, 4, 2],
    [0, 4, 2, 1, 0],
    [3, 0, 0, 2, 1],
    [2, 2, 0, 1, 0],
    [1, 2, 4, 1, 3],
    [1, 4, 0, 0, 4],
    [3, 3, 4, 0, 0],
    [3, 2, 3, 0, 4],
    [0, 1, 2, 1, 4],
    [1, 4, 0, 2, 4]])


def test_column_select_all():
    """Test that column select function selects all columns when provided the
    entire SVD and a suitable value of c."""
    _, n_features = X.shape
    _, _, V = np.linalg.svd(X)
    column_indices = column_select(V.T, c=5)

    assert_array_equal(column_indices, np.arange(n_features))


def test_column_select():
    """Test that the column select function selects the column with the highest
    leverage score most often."""
    n_samples, n_features = X.shape
    rank = 3
    n_reps = 1000

    _, _, V = np.linalg.svd(X)
    V_subset = V[:rank].T
    column_flags = np.zeros((n_reps, n_features))

    for rep in range(n_reps):
        column_indices = column_select(V_subset, c=1)
        column_flags[rep, column_indices] = 1

    counts = np.sum(column_flags, axis=0)

    assert_equal(np.argmax(counts), np.argmax(np.sum(V_subset**2, axis=1)))


def test_stability_selection_to_threshold_int():
    """Tests whether stability_selection_to_threshold correctly outputs the
    correct threshold when provided a single integer."""

    n_boots_sel = 48
    # stability selection is a single integer
    test_int = 36
    selection_thresholds = stability_selection_to_threshold(test_int,
                                                            n_boots_sel)

    assert_array_equal(selection_thresholds, test_int)


def test_stability_selection_to_threshold_float():
    """Tests whether stability_selection_to_threshold correctly outputs the
    correct threshold when provided a single float."""

    n_boots_sel = 48
    # stability selection is a single float
    test_float = 0.5
    selection_thresholds = stability_selection_to_threshold(test_float,
                                                            n_boots_sel)

    assert_array_equal(selection_thresholds, np.array([24]))


def test_stability_selection_to_threshold_exceeds_n_bootstraps():
    """Tests whether stability_selection_to_threshold correctly outputs an
    error when provided an input that results in bootstraps exceeding
    n_boots_sel."""

    n_boots_sel = 48
    # stability selection is a list of floats
    test_float = 1.1
    test_int = 50

    assert_raises(
        ValueError,
        stability_selection_to_threshold,
        test_int,
        n_boots_sel)

    assert_raises(
        ValueError,
        stability_selection_to_threshold,
        test_float,
        n_boots_sel)


def test_stability_selection_to_threshold_input_value_error():
    """Tests whether stability_selection_to_threshold properly raises an error
    when it receives objects without ints or floats."""
    n_boots_sel = 48
    stability_selection_list = [0, 1, 'a']
    stability_selection_np_array = np.array([0, 1, 'a'])
    stability_selection_dict = {0: 'a', 1: 'b'}

    assert_raises(
        ValueError,
        stability_selection_to_threshold,
        stability_selection_list,
        n_boots_sel)

    assert_raises(
        ValueError,
        stability_selection_to_threshold,
        stability_selection_np_array,
        n_boots_sel)

    assert_raises(
        ValueError,
        stability_selection_to_threshold,
        stability_selection_dict,
        n_boots_sel)


def test_CUR():
    """Tests that the CUR fitter extracts columns correctly."""
    _, n_features = X.shape
    max_k = 3

    cur = CUR(max_k=max_k)

    cur.fit(X, c=3)
    column_indices = cur.column_indices_
    columns = cur.components_

    assert np.setdiff1d(column_indices, np.arange(n_features)).size == 0
    assert_array_equal(X[:, column_indices], columns)


def test_CUR_fit():
    """Tests that the CUR fitter extracts the correct columns."""
    n_features = 5
    n_samples = 30
    max_k = 3

    # matrix has only one non-zero entry
    X = np.zeros((n_samples, n_features))
    X[0, 0] = 1
    true_columns = np.array([0, 2, 3])

    # fit CUR decomposition
    cur = CUR(max_k=max_k)
    X_new = cur.fit_transform(X)

    assert_array_equal(cur.column_indices_, true_columns)
    assert_array_equal(X_new, X[:, true_columns])


def test_UoI_CUR_check_ks_and_cs():
    """Tests the check_ks_and_cs function in UoI_CUR."""
    n_boots = 5
    max_k = 10
    boots_frac = 0.9

    uoi_cur = UoI_CUR(n_boots=n_boots,
                      max_k=max_k,
                      boots_frac=boots_frac)

    # check ks
    ks, cs = uoi_cur.check_ks_and_cs(ks=1)
    assert_array_equal(ks, np.array([1]))
    assert_array_equal(cs, ks + 20)

    ks, cs = uoi_cur.check_ks_and_cs(ks=[1, 2, 3])
    assert_array_equal(ks, np.array([1, 2, 3]))
    assert_array_equal(cs, ks + 20)

    ks, cs = uoi_cur.check_ks_and_cs(ks=None)
    assert_array_equal(ks, 1 + np.arange(max_k))
    assert_array_equal(cs, ks + 20)

    # check cs
    ks, cs = uoi_cur.check_ks_and_cs(ks=[1, 2], cs=[3, 4])
    assert_array_equal(cs, np.array([3, 4]))

    ks, cs = uoi_cur.check_ks_and_cs(ks=[1, 2, 3], cs=1)
    assert_array_equal(cs, np.array([1, 1, 1]))

    ks, cs = uoi_cur.check_ks_and_cs(ks=[1, 2], cs=2.4)
    assert_array_equal(cs, np.array([2.4, 2.4]))

    # value errors for ks
    assert_raises(ValueError, uoi_cur.check_ks_and_cs, -1)
    assert_raises(ValueError, uoi_cur.check_ks_and_cs, [11])
    assert_raises(ValueError, uoi_cur.check_ks_and_cs, [0.1, -1, 2, 12])
    assert_raises(ValueError, uoi_cur.check_ks_and_cs, 2.0)
    assert_raises(ValueError, uoi_cur.check_ks_and_cs, uoi_cur)

    # value errors for cs
    assert_raises(ValueError, uoi_cur.check_ks_and_cs, None, -1)
    assert_raises(ValueError, uoi_cur.check_ks_and_cs, None, [-11])
    assert_raises(ValueError, uoi_cur.check_ks_and_cs, None, np.array([-12]))
    assert_raises(ValueError, uoi_cur.check_ks_and_cs, 1, [2, 3])


def test_UoI_CUR_basic():
    """Test UoI CUR with no bootstrapping."""
    n_samples, n_features = X.shape
    max_k = 3
    n_boots = 1
    boots_frac = 1

    _, _, V = np.linalg.svd(X)
    V_subset = V[:max_k].T

    uoi_cur = UoI_CUR(n_boots=n_boots,
                      max_k=max_k,
                      boots_frac=boots_frac)
    uoi_cur.fit(X, cs=3)

    max_col = np.argmax(np.sum(V_subset**2, axis=1))

    assert (max_col in uoi_cur.column_indices_)


def test_UoI_CUR_fit():
    """Tests that the CUR fitter extracts the correct columns."""
    n_features = 5
    n_samples = 30
    max_k = 3
    n_boots = 10
    boots_frac = 0.95

    # matrix has only one non-zero entry
    X = np.zeros((n_samples, n_features))
    X[0, 0] = 1
    true_columns = np.array([0, 2, 3])

    # fit CUR decomposition
    uoi_cur = UoI_CUR(n_boots=n_boots,
                      max_k=max_k,
                      boots_frac=boots_frac,
                      random_state=2332)
    X_new = uoi_cur.fit_transform(X)

    assert_array_equal(uoi_cur.column_indices_, true_columns)
    assert_array_equal(uoi_cur.components_, X[:, true_columns])
    assert_array_equal(X_new, X[:, true_columns])


def test_UoI_CUR_vs_CUR():
    """Tests that the CUR fitter extracts columns correctly."""
    _, n_features = X.shape
    max_k = 3
    n_boots = 10
    boots_frac = 0.90

    cur = CUR(max_k=max_k,
              random_state=2332)
    cur.fit(X, c=3)

    uoi_cur = UoI_CUR(n_boots=n_boots,
                      max_k=max_k,
                      boots_frac=boots_frac,
                      random_state=2332)
    uoi_cur.fit(X, cs=3, ks=3)

    assert uoi_cur.column_indices_.size <= cur.column_indices_.size
