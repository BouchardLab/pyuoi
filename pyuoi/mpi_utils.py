import math
import numpy as np

"""
This module contains some helper functions for working with MPI.
"""
def get_chunk_size(rank, size, n):
    """
    This functions computes two quantities that are useful when distributing work
    across an MPI group.

    If *n* % *size* == 0, the returned values should be identitical (regardless of rank)
    otherwise, the difference between *buf_len* and *chunk_size* should be used for padding
    buffers required for MPI Gather operations.

    Args:
        rank: the rank of the MPI process
        size: the size of the MPI group
        n:    the total number of items to process

    Returns:
        chunk_size: the size of the chunk for the rank to process
        buf_len:    the size of the buffer to allocate
    """
    chunk_size = int(math.ceil((n-rank)/size))
    buf_len = int(math.ceil(n/size))
    return chunk_size, buf_len

def get_buffer_mask(size, n):
    """
    This functions computes a mask for filtering out extra elements from padding
    prior to a MPI Gather operation.

    If *n* % *size* == 0 (no padding necessary for a Gather operation), the returned value will be
    a bool array of length *n* containing all `True` values, otherwise, the returned values
    should be for filtering out padded elements.

    Args:
        size: the size of the MPI group
        n:    the total number of items to process

    Returns:
        mask: the boolean mask identifying the elements to keep from a receive buffer
    """
    buf_len = int(math.ceil(n/size))
    buf_tot = size*buf_len
    ret = np.ones(buf_tot, dtype=bool)
    fpr = n % size                 # first padding rank
    if fpr > 0:
        st = (fpr+1) * buf_len - 1
        mask = np.arange(st, buf_tot, buf_len)
        ret[mask] = False
    return ret
