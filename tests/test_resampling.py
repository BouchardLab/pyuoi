import numpy as np
from sklearn.model_selection import train_test_split

from pyuoi.resampling import resample


def test_sampling_without_replacement():
    """Test that our bootstrap function gives identical results to
    train_test_split"""

    random_state = 5
    n_samples = 200

    # No stratification
    train_idxs1, test_idxs1 = train_test_split(np.arange(n_samples),
                                               train_size=0.75,
                                               random_state=random_state)

    train_idxs2, test_idxs2 = resample('bootstrap', np.arange(n_samples),
                                       random_state=random_state,
                                       replace=False, sampling_frac=0.75)

    assert(np.array_equal(np.sort(train_idxs1), np.sort(train_idxs2)))
    assert(np.array_equal(np.sort(test_idxs1), np.sort(test_idxs2)))

    # Stratification
    # Divide into 3 categories
    stratification = np.zeros(n_samples)
    stratification[0:50] = 0
    stratification[50:100] = 1
    stratification[100::] = 2

    train_idxs1, test_idxs1 = train_test_split(np.arange(n_samples),
                                               train_size=0.75,
                                               random_state=random_state,
                                               stratify=stratification)

    train_idxs2, test_idxs2 = resample('bootstrap', np.arange(n_samples),
                                       random_state=random_state,
                                       replace=False, sampling_frac=0.75,
                                       stratify=stratification)

    assert(np.array_equal(train_idxs1, train_idxs2))
    assert(np.array_equal(test_idxs1, test_idxs2))


def test_sampling_with_replacement():
    """Some tests for using resampling with replacement (proper bootstrap)"""

    # Check that duplicates arise when sampling with replacement
    n_samples = 200
    random_state = 5

    train_idxs, test_idxs = resample('bootstrap', np.arange(n_samples),
                                     random_state=random_state,
                                     replace=True, sampling_frac=1.1)

    # Duplicates in train_idxs
    assert(len(np.unique(train_idxs)) < len(train_idxs))

    # Check that test_idxs have no overlap with train idxs
    train_idxs, test_idxs = resample('bootstrap', np.arange(n_samples),
                                     random_state=random_state,
                                     replace=True, sampling_frac=0.75)

    assert(np.intersect1d(train_idxs, test_idxs).size == 0)

    # Check that when using stratify, the class proportions are preserved
    stratification = np.zeros(n_samples)
    stratification[0:50] = 0
    stratification[50:100] = 1
    stratification[100::] = 2

    train_idxs, test_idxs = resample('bootstrap', np.arange(n_samples),
                                     random_state=random_state,
                                     replace=True, sampling_frac=0.75,
                                     stratify=stratification)

    train_stratification = stratification[train_idxs]
    assert(train_stratification[train_stratification == 0].size
           / (0.75 * n_samples) - 0.25 <= 0.01)
    assert(train_stratification[train_stratification == 1].size
           / (0.75 * n_samples) - 0.25 <= 0.01)
    assert(train_stratification[train_stratification == 2].size
           / (0.75 * n_samples) - 0.5 <= 0.01)


def test_random_state():
    """Test that random state assignment works properly"""
    n_samples = 200
    random_state = 5

    train_idxs1, test_idxs1 = resample('bootstrap', np.arange(n_samples),
                                       random_state=random_state,
                                       replace=True, sampling_frac=0.75)

    train_idxs2, test_idxs2 = resample('bootstrap', np.arange(n_samples),
                                       random_state=random_state,
                                       replace=True, sampling_frac=0.75)

    assert(np.array_equal(train_idxs1, train_idxs2))

    random_state = np.random.RandomState(5)

    train_idxs1, test_idxs1 = resample('bootstrap', np.arange(n_samples),
                                       random_state=random_state,
                                       replace=True, sampling_frac=0.75)

    random_state = np.random.RandomState(5)

    train_idxs2, test_idxs2 = resample('bootstrap', np.arange(n_samples),
                                       random_state=random_state,
                                       replace=True, sampling_frac=0.75)

    assert(np.array_equal(train_idxs1, train_idxs2))

    # No random state, should be different
    train_idxs1, test_idxs1 = resample('bootstrap', np.arange(n_samples),
                                       replace=True, sampling_frac=0.75)

    train_idxs2, test_idxs2 = resample('bootstrap', np.arange(n_samples),
                                       replace=True, sampling_frac=0.75)

    assert(not np.array_equal(train_idxs1, train_idxs2))


def test_block_boostrap():
    """Test the moving block bootstrap"""

    n_samples = 1000
    L = 50
    random_state = 5
    block_frac = 0.75

    # Check that the the length of train indices matches the expected length
    train_idxs, test_idxs = resample('block', np.arange(n_samples),
                                     random_state=random_state,
                                     L=L, block_frac=block_frac)

    """There should be n_samples - L + 1 blocks. Sampling block_frac
    should then give a train size of block_frac * L * (n_samples - L + 1)
    data points"""
    n_train_blocks = np.round(block_frac * (n_samples - L + 1))

    assert(len(train_idxs) == L * n_train_blocks)

    # Check that the returned indices are contiguous in blocks of length L
    blocks = np.split(train_idxs, np.where(np.diff(train_idxs) != 1)[0] + 1)
    block_sizes = [block.size for block in blocks]

    assert(not np.all(np.mod(block_sizes, L)))

    blocks = np.split(test_idxs, np.where(np.diff(test_idxs) != 1)[0] + 1)
    block_sizes = [block.size for block in blocks]
    assert(not np.all(np.mod(block_sizes, L)))

    # Check proper catching of incorrect block sizes
    try:
        _, _ = resample('block', np.arange(n_samples),
                        random_state=random_state,
                        L=L - 1, block_frac=block_frac)
    except ValueError:
        pass
    else:
        raise Exception('Unknown exception raised')
