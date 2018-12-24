"""
Helper functions for working with MPI.
"""
import math, h5py
import numpy as np

from mpi4py import MPI
_np2mpi = {np.dtype(np.float32): MPI.FLOAT,
           np.dtype(np.float64): MPI.DOUBLE,
           np.dtype(np.int): MPI.LONG,
           np.dtype(np.intc): MPI.INT}


def load_data_MPI(h5_name, X_key='X', y_key='y'):
    """Load data from an h5 file and broadcast it across MPI ranks.

    Parameters
    ----------
    h5_name : str
        Path to h5 file.
    X_key : str
        Key for the features dataset. (default: 'X')
    y_key : str
        Key for the targets dataset. (default: 'y')

    Returns
    -------
    X : ndarray
        Features on all MPI ranks.
    y : ndarray
        Targets on all MPI ranks.
    """

    comm = MPI.COMM_WORLD
    rank = comm.rank
    with h5py.File(h5_name, 'r') as f:
        if rank == 0:
            X = f[X_key].value
            y = f[y_key].value
        else:
            X = np.empty(f[X_key].shape, dtype=f[X_key].dtype)
            y = np.empty(f[y_key].shape, dtype=f[y_key].dtype)
    comm.Bcast([X, _np2mpi[np.dtype(X.dtype)]], root=0)
    comm.Bcast([y, _np2mpi[np.dtype(y.dtype)]], root=0)
    return X, y


def Bcast_from_root(send, comm, root=0):
    """Broadcast an array from root to all MPI ranks.

    Parameters
    ----------
    send : ndarray
        The arrays to concatenate. All dimensions must be equal except for the
        first.
    comm : MPI.COMM_WORLD
        MPI communicator.
    root : int, default 0
        This rank contains the array to send.
    """
    rank = comm.rank
    if rank == 0:
        dtype = send.dtype
        shape = send.shape
    else:
        dtype = None
        shape = None
    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)
    if rank != 0:
        send = np.empty(shape, dtype=dtype)
    comm.Bcast([send, _np2mpi[np.dtype(dtype)]], root=root)
    return send


def Gatherv_rows(send, comm, root=0):
    """Concatenate arrays along the first axis using Gatherv.

    Parameters
    ----------
    send : ndarray
        The arrays to concatenate. All dimensions must be equal except for the
        first.
    comm : MPI.COMM_WORLD
        MPI communicator.
    root : int, default 0
        This rank will contain the Gatherv'ed array.
    """

    rank = comm.rank
    size = comm.size
    dtype = send.dtype
    shape = send.shape
    tot = np.zeros(1, dtype=int)
    comm.Reduce(np.array(shape[0], dtype=int),
                [tot, _np2mpi[tot.dtype]], op=MPI.SUM, root=root)
    if rank == root:
        rec_shape = (tot[0],) + shape[1:]
        rec = np.empty(rec_shape, dtype=dtype)
        idxs = np.array_split(np.arange(rec_shape[0]), size)
        sizes = [idx.size * np.prod(rec_shape[1:]) for idx in idxs]
        disps = np.insert(np.cumsum(sizes), 0, 0)[:-1]
    else:
        rec = None
        idxs = None
        sizes = None
        disps = None

    comm.Gatherv(send, [rec, sizes, disps, _np2mpi[dtype]], root=0)
    return rec


def get_chunk_size(rank, size, n):
    """
    This functions computes two quantities that are useful when distributing
    work across an MPI group.

    If *n* % *size* == 0, the returned values should be identitical (regardless
    of rank) otherwise, the difference between *buf_len* and *chunk_size* should
    be used for padding buffers required for MPI Gather operations.

    Args:
        rank: the rank of the MPI process
        size: the size of the MPI group
        n:    the total number of items to process

    Returns:
        chunk_size: the size of the chunk for the rank to process
        buf_len:    the size of the buffer to allocate
    """
    chunk_size = int(math.ceil((n - rank) / size))
    buf_len = int(math.ceil(n / size))
    return chunk_size, buf_len


def get_buffer_mask(size, n):
    """
    This functions computes a mask for filtering out extra elements from padding
    prior to a MPI Gather operation.

    If *n* % *size* == 0 (no padding necessary for a Gather operation), the
    returned value will be a bool array of length *n* containing all `True`
    values, otherwise, the returned values should be for filtering out padded
    elements.

    Args:
        size: the size of the MPI group
        n:    the total number of items to process

    Returns:
        mask: the boolean mask identifying the elements to keep from a receive
        buffer
    """
    buf_len = int(math.ceil(n / size))
    buf_tot = size * buf_len
    ret = np.ones(buf_tot, dtype=bool)
    fpr = n % size                 # first padding rank
    if fpr > 0:
        st = (fpr + 1) * buf_len - 1
        mask = np.arange(st, buf_tot, buf_len)
        ret[mask] = False
    return ret
