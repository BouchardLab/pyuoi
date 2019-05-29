import numpy as np

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_raises
from pyuoi.decomposition import CUR, UoI_CUR
from pyuoi.decomposition.utils import column_select

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


def test_UoI_CUR_check_ks():
    """Tests the check_ks function in UoI_CUR."""
    n_boots = 5
    max_k = 10
    boots_frac = 0.9

    uoi_cur = UoI_CUR(n_boots=n_boots,
                      max_k=max_k,
                      boots_frac=boots_frac)

    ks = uoi_cur.check_ks(1)
    assert_array_equal(ks, np.array([1]))

    ks = uoi_cur.check_ks([1, 2, 3])
    assert_array_equal(ks, np.array([1, 2, 3]))

    ks = uoi_cur.check_ks(None)
    assert_array_equal(ks, 1 + np.arange(max_k))

    assert_raises(ValueError, uoi_cur.check_ks, -1)
    assert_raises(ValueError, uoi_cur.check_ks, [11])
    assert_raises(ValueError, uoi_cur.check_ks, [0.1, -1, 2, 12])
    assert_raises(ValueError, uoi_cur.check_ks, 2.0)
    assert_raises(ValueError, uoi_cur.check_ks, uoi_cur)


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
    uoi_cur.fit(X, c=3)

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
    uoi_cur.fit(X, c=3, ks=3)

    assert uoi_cur.column_indices_.size <= cur.column_indices_.size
