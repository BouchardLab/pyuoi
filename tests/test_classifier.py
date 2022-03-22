import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.random import default_rng
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support


class TestMetrics:
    def test_jensen_shannon_symmetry(self):
        rng = default_rng(10)
        le = LabelEncoder()
        choices = np.array(['Walk', 'ProneStill', 'Dance'])
        le.fit(choices)

        freq_a = np.bincount(le.transform(rng.choice(choices, size=10)))
        freq_b = np.bincount(le.transform(rng.choice(choices, size=10)))

        assert_equal(jensenshannon(freq_a, freq_b),
                     jensenshannon(freq_b, freq_a))

    def test_precision_recall_fscore(self):
        rng = default_rng(10)
        le = LabelEncoder()
        choices = np.array(['Walk', 'ProneStill', 'Dance'])
        le.fit(choices)

        a = rng.choice(choices, size=50)
        b = rng.choice(choices, size=50)

        for true, predicted in zip(precision_recall_fscore_support(a, b), (np.array([0.54545455, 0.23809524, 0.27777778]),
                                                                           np.array(
                                                                               [0.27272727, 0.41666667, 0.3125]),
                                                                           np.array(
                                                                               [0.36363636, 0.3030303, 0.29411765]),
                                                                           np.array([22, 12, 16]))
                                   ):
            assert_array_almost_equal(true, predicted)
