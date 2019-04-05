import h5py, pytest
import numpy as np

from numpy.testing import assert_array_equal
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from pyuoi.mpi_utils import (Bcast_from_root, Gatherv_rows,
                             load_data_MPI)


@pytest.mark.skipif(MPI is None, reason='MPI not installed.')
def test_load_data_MPI(tmpdir):
    """Tests loading data from an HDF5 file into all ranks.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    root = 0
    X = np.random.randn(5, 10)
    y = np.random.randint(5, size=5)

    fname = tmpdir.join('temp.h5')
    if rank == root:
        with h5py.File(str(fname), 'w') as f:
            f.create_dataset('X', data=X)
            f.create_dataset('Xp', data=X)
            f.create_dataset('y', data=y)
            f.create_dataset('yp', data=y)

    # Default keys
    X_load, y_load = load_data_MPI(fname)
    if rank == root:
        assert_array_equal(X, X_load)
        assert_array_equal(y, y_load)

    # Set keys
    X_load, y_load = load_data_MPI(fname,
                                   X_key='Xp',
                                   y_key='yp')
    if rank == root:
        assert_array_equal(X, X_load)
        assert_array_equal(y, y_load)


@pytest.mark.skipif(MPI is None, reason='MPI not installed.')
def test_Bcast_from_root():
    """Test the Bcast_from_root function for broadcasting
    an array from root to all ranks.
    """
    comm = MPI.COMM_WORLD
    root = 0

    dims = [2, 3, 5]

    for dtype in [int, float]:
        for ndim in range(1, 4):
            my_dim = dims[:ndim]
            X = None
            if comm.rank == root:
                X = np.arange(np.prod(my_dim), dtype=dtype)
                X = X.reshape(my_dim)
            X = Bcast_from_root(X, comm, root)
            Xp = np.arange(np.prod(my_dim), dtype=dtype)
            Xp = Xp.reshape(my_dim)
            assert_array_equal(X, Xp)
            assert X.dtype == dtype
            assert X.ndim == len(my_dim)


@pytest.mark.skipif(MPI is None, reason='MPI not installed.')
def test_Gatherv_rows():
    """Test the Gatherv_rows function for Gathering and
    concatenating ndarrys along their first axes to root.
    """
    comm = MPI.COMM_WORLD
    root = 0
    rank = comm.rank
    size = comm.size

    for dtype in [int, float]:
        # Multiple rows per rank
        X = np.arange(151 * 3, dtype=dtype).reshape(151, 3)
        my_rows = np.array_split(X, size)[rank]
        Xp = Gatherv_rows(my_rows, comm, root)
        if rank == root:
            assert_array_equal(X, Xp)
            assert Xp.dtype == dtype

        # Fewer rows than ranks
        X = np.arange(2 * 3, dtype=dtype).reshape(2, 3)
        my_rows = np.array_split(X, size)[rank]
        Xp = Gatherv_rows(my_rows, comm, root)
        if rank == root:
            assert_array_equal(X, Xp)
            assert Xp.dtype == dtype

        # Multiple rows per rank, 3d
        X = np.arange(151 * 2 * 3, dtype=dtype).reshape(151, 2, 3)
        my_rows = np.array_split(X, size)[rank]
        Xp = Gatherv_rows(my_rows, comm, root)
        if rank == root:
            assert_array_equal(X, Xp)
            assert Xp.dtype == dtype

        # Fewer rows than ranks, 3d
        X = np.arange(2 * 3 * 5, dtype=dtype).reshape(2, 3, 5)
        my_rows = np.array_split(X, size)[rank]
        Xp = Gatherv_rows(my_rows, comm, root)
        if rank == root:
            assert_array_equal(X, Xp)
            assert Xp.dtype == dtype


@pytest.mark.skipif(MPI is None, reason='MPI not installed.')
def test_Gatherv_random_rows():
    """Test Gatherv_rows for gathering ndarrays with random
    shapes along their first axis
    """

    comm = MPI.COMM_WORLD
    root = 0
    rank = comm.rank

    data = np.random.normal(size=(np.random.randint(1, 10), 1000))
    sizes = comm.gather(data.shape[0], root=root)
    data = Gatherv_rows(data, comm, root)

    if rank == root:
        assert(data.shape[0] == np.sum(sizes))
