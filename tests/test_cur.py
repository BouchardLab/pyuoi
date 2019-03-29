import h5py, pytest
import numpy as np

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from pyuoi.decomposition import CUR

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
    X = np.random.randint(low=1, high=5, size=(10, 5))
    _, _, V = np.linalg.svd(X)
    column_flags = CUR.column_select(V.T, c=5)

    assert_array_equal(column_flags, np.array([True, True, True, True, True]))


def test_column_select():
    """Test that the column select function selects the vector with the highest
    leverage score most often."""
    n_samples, n_features = X.shape
    rank = 3
    n_reps = 5000

    _, _, V = np.linalg.svd(X)
    V_subset = V[:rank].T
    column_flags = np.zeros((n_reps, n_features))

    for rep in range(n_reps):
        column_flags[rep] = CUR.column_select(V_subset, c=1)

    counts = np.sum(column_flags, axis=0)

    assert_equal(np.argmax(counts), np.argmax(np.sum(V_subset**2, axis=1)))


def test_UoI_CUR_basic():
    """Test UoI CUR with no bootstrapping."""
    n_samples, n_features = X.shape
    max_k = 3
    n_boots = 1
    boots_frac = 1

    _, _, V = np.linalg.svd(X)
    V_subset = V[:max_k].T

    uoi_cur = CUR(n_boots=n_boots,
                  max_k=max_k,
                  boots_frac=boots_frac)
    uoi_cur.fit(X, c=3)

    max_col = np.argmax(np.sum(V_subset**2, axis=1))

    assert (max_col in uoi_cur.columns_)


@pytest.mark.skip(reason='Waiting on dataset.')
def test_UoI_CUR_vs_CUR():
    with h5py.File('tests/cur.h5', 'r') as data:
        X = data['X'][:]

    max_k = 50

    uoi_cur = CUR(n_boots=4,
                  max_k=max_k,
                  boots_frac=0.9)
    uoi_cur.fit(X, c=None)
    n_uoi = uoi_cur.columns_.size
    _, _, V = np.linalg.svd(X)

    cur_flags = CUR.column_select(V[:max_k].T, max_k + 20)
    n_cur = np.count_nonzero(cur_flags)

    assert n_uoi < n_cur
