import numpy as np
from numpy.testing import assert_array_equal

from pyuoi.mpi_utils import get_buffer_mask, get_chunk_size


def test_get_chunk_size_unbalanced():
    chunk_size, buf_len = get_chunk_size(0, 4, 13)
    assert chunk_size == 4
    assert buf_len == 4
    chunk_size, buf_len = get_chunk_size(1, 4, 13)
    assert chunk_size == 3
    assert buf_len == 4
    chunk_size, buf_len = get_chunk_size(2, 4, 13)
    assert chunk_size == 3
    assert buf_len == 4
    chunk_size, buf_len = get_chunk_size(3, 4, 13)
    assert chunk_size == 3
    assert buf_len == 4

def test_get_chunk_size_balanced():
    chunk_size, buf_len = get_chunk_size(0, 4, 16)
    assert chunk_size == 4
    assert buf_len == 4
    chunk_size, buf_len = get_chunk_size(1, 4, 16)
    assert chunk_size == 4
    assert buf_len == 4
    chunk_size, buf_len = get_chunk_size(2, 4, 16)
    assert chunk_size == 4
    assert buf_len == 4
    chunk_size, buf_len = get_chunk_size(3, 4, 16)
    assert chunk_size == 4
    assert buf_len == 4

def test_get_mask_unbalanced():
    expected = [True, True, True, True,
     True, True, True, False,
     True, True, True, False,
     True, True, True, False]
    mask = get_buffer_mask(4, 13)
    assert_array_equal(expected, mask)

def test_get_mask_balanced():
    expected = [True, True, True, True,
     True, True, True, True,
     True, True, True, True,
     True, True, True, True]
    mask = get_buffer_mask(4, 16)
    assert_array_equal(expected, mask)
