import pytest

import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_raises

from pyuoi.linear_model.utils import stability_selection_to_threshold
from pyuoi.linear_model.utils import intersection

from pyuoi.utils import check_logger

import logging
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def test_stability_selection_to_threshold_int():
    """Tests whether stability_selection_to_threshold correctly outputs the
    correct threshold when provided a single integer."""

    n_boots_sel = 48
    # stability selection is a single integer
    test_int = 36
    selection_thresholds = stability_selection_to_threshold(
        test_int, n_boots_sel)

    assert_array_equal(selection_thresholds, np.array([36]))


def test_stability_selection_to_threshold_float():
    """Tests whether stability_selection_to_threshold correctly outputs the
    correct threshold when provided a single float."""

    n_boots_sel = 48
    # stability selection is a single float
    test_float = 0.5
    selection_thresholds = stability_selection_to_threshold(
        test_float, n_boots_sel)

    assert_array_equal(selection_thresholds, np.array([24]))


def test_stability_selection_to_threshold_ints():
    """Tests whether stability_selection_to_threshold correctly outputs the
    correct threshold when provided a list of ints."""

    n_boots_sel = 48
    # stability selection is a list of ints
    test_ints = [24, 28, 33, 38, 43, 48]
    selection_thresholds = stability_selection_to_threshold(
        test_ints, n_boots_sel)

    assert_array_equal(
        selection_thresholds,
        np.array([24, 28, 33, 38, 43, 48]))


def test_stability_selection_to_threshold_floats():
    """Tests whether stability_selection_to_threshold correctly outputs the
    correct threshold when provided a list of floats."""
    n_boots_sel = 48
    # stability selection is a list of floats
    test_floats = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    selection_thresholds = stability_selection_to_threshold(
        test_floats, n_boots_sel)

    assert_array_equal(
        selection_thresholds,
        np.array([24, 28, 33, 38, 43, 48]))


def test_stability_selection_to_threshold_ints_np():
    """Tests whether stability_selection_to_threshold correctly outputs the
    correct threshold when provided a numpy array of ints."""

    n_boots_sel = 48
    # stability selection is a list of ints
    test_ints_np = np.array([24, 28, 33, 38, 43, 48])
    selection_thresholds = stability_selection_to_threshold(
        test_ints_np, n_boots_sel)

    assert_array_equal(
        selection_thresholds,
        np.array([24, 28, 33, 38, 43, 48]))


def test_stability_selection_to_threshold_floats_np():
    """Tests whether stability_selection_to_threshold correctly outputs the
    correct threshold when provided a numpy array of ints."""

    n_boots_sel = 48
    # stability selection is a list of floats
    test_floats_np = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    selection_thresholds = stability_selection_to_threshold(
        test_floats_np, n_boots_sel)

    assert_array_equal(
        selection_thresholds,
        np.array([24, 28, 33, 38, 43, 48]))


def test_stability_selection_to_threshold_exceeds_n_bootstraps():
    """Tests whether stability_selection_to_threshold correctly outputs an
    error when provided an input that results in bootstraps exceeding
    n_boots_sel."""

    n_boots_sel = 48
    # stability selection is a list of floats
    test_floats = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    test_ints = np.array([24, 28, 33, 38, 43, 48, 52])

    assert_raises(
        ValueError,
        stability_selection_to_threshold,
        test_ints,
        n_boots_sel)

    assert_raises(
        ValueError,
        stability_selection_to_threshold,
        test_floats,
        n_boots_sel)


def test_stability_selection_to_threshold_one_bootstrap():
    """Tests whether stability_selection_to_threshold correctly handles the
    edge case where one bootstrap is requested."""

    n_boots_sel = 1
    # stability selection can only be one value
    threshold = 1

    selection_thresholds = stability_selection_to_threshold(
        n_boots_sel,
        threshold)

    assert_array_equal(
        selection_thresholds,
        np.array([1]))


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


def test_stability_selection_reject_negative_numbers():
    """Tests whether stability_selection_to_threshold correctly rejects
    negative thresholds."""

    n_boots_sel = 48

    # stability selection is a list of floats
    test_negative = -1 * np.array([24, 28, 33, 38, 43, 48, 52])

    assert_raises(
        ValueError,
        stability_selection_to_threshold,
        test_negative,
        n_boots_sel)


def test_intersection():
    """Tests whether intersection correctly performs a hard intersection."""

    coefs = np.array([
        [[2, 1, -1, 0, 4],
         [4, 0, 2, -1, 5],
         [1, 2, 3, 4, 5]],
        [[2, 0, 0, 0, 0],
         [3, 1, 1, 0, 3],
         [6, 7, 8, 9, 10]],
        [[2, 0, 0, 0, 0],
         [2, -1, 3, 0, 2],
         [2, 4, 6, 8, 9]]])

    true_intersection = np.array([
        [True, False, False, False, False],
        [True, False, True, False, True],
        [True, True, True, True, True]])

    selection_thresholds = np.array([3])
    estimated_intersection = intersection(
        coefs=coefs,
        selection_thresholds=selection_thresholds)

    # we sort the supports since they might not be in the same order
    assert_array_equal(
        np.sort(true_intersection, axis=0),
        np.sort(estimated_intersection, axis=0))


def test_intersection_with_stability_selection_one_threshold():
    """Tests whether intersection correctly performs a soft intersection."""

    coefs = np.array([
        [[2, 1, -1, 0, 4],
         [4, 0, 2, -1, 5],
         [1, 2, 3, 4, 5]],
        [[2, 0, 0, 0, 0],
         [3, 1, 1, 0, 3],
         [6, 7, 8, 9, 10]],
        [[2, 0, 0, 0, 0],
         [2, -1, 3, 0, 2],
         [2, 4, 6, 8, 9]]])

    true_intersection = np.array([
        [True, False, False, False, False],
        [True, True, True, False, True],
        [True, True, True, True, True]])

    selection_thresholds = np.array([2])
    estimated_intersection = intersection(
        coefs=coefs,
        selection_thresholds=selection_thresholds)

    # we sort the supports since they might not be in the same order
    assert_array_equal(
        np.sort(true_intersection, axis=0),
        np.sort(estimated_intersection, axis=0))


def test_intersection_with_stability_selection_multiple_thresholds():
    """Tests whether intersection correctly performs an intersection with
    multiple thresholds. This test also covers the case when there are
    duplicates."""

    coefs = np.array([
        [[2, 1, -1, 0, 4],
         [4, 0, 2, -1, 5],
         [1, 2, 3, 4, 5]],
        [[2, 0, 0, 0, 0],
         [3, 1, 1, 0, 3],
         [6, 7, 8, 9, 10]],
        [[2, 0, 0, 0, 0],
         [2, -1, 3, 0, 2],
         [2, 4, 6, 8, 9]]])

    true_intersection = np.array([
        [True, False, False, False, False],
        [True, True, True, False, True],
        [True, True, True, True, True],
        [True, False, True, False, True]])

    selection_thresholds = np.array([2, 3])
    estimated_intersection = intersection(
        coefs=coefs,
        selection_thresholds=selection_thresholds)

    # we sort the supports since they might not be in the same order
    assert_array_equal(
        np.sort(true_intersection, axis=0),
        np.sort(estimated_intersection, axis=0))


def test_intersection_no_thresholds():
    """Tests that the intersection method correctly calculates the intersection
    using the number of bootstraps as the default selection threshold."""

    coefs = np.array([
        [[2, 1, -1, 0, 4],
         [4, 0, 2, -1, 5],
         [1, 2, 3, 4, 5]],
        [[2, 0, 0, 0, 0],
         [3, 1, 1, 0, 3],
         [6, 7, 8, 9, 10]],
        [[2, 0, 0, 0, 0],
         [2, -1, 3, 0, 2],
         [2, 4, 6, 8, 9]]])

    true_intersection = np.array([
        [True, False, False, False, False],
        [True, True, True, True, True],
        [True, False, True, False, True]])

    estimated_intersection = intersection(
        coefs=coefs,
        selection_thresholds=None)

    # we sort the supports since they might not be in the same order
    assert_array_equal(
        np.sort(true_intersection, axis=0),
        np.sort(estimated_intersection, axis=0))


@pytest.mark.fast
def test_check_logger():
    """Test that check_logger builds logger correctly"""
    ret = check_logger(None, name="test_check_logger")
    assert ret is not None
    assert ret.name == 'test_check_logger'


@pytest.mark.fast
@pytest.mark.skipif(MPI is None, reason='MPI not installed.')
def test_check_logger_mpi():
    """Test that passing in a MPI communicatorj object works with
    check_logger"""
    comm = MPI.COMM_WORLD
    ret = check_logger(None, comm=comm)
    assert ret is not None


@pytest.mark.fast
def test_check_logger_exists():
    """Test that logger returns the argued logger when it gets passed in"""
    logger = logging.getLogger()
    ret = check_logger(logger)
    assert ret is logger
