from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
import numpy as np


class CUR(BaseEstimator):
    def __init__(self, n_bootstraps, ranks, fraction):

        self.__initialize(
            n_bootstraps=n_bootstraps,
            ranks=ranks,
            fraction=fraction
        )

    def fit(self, X):
        return

    def column_select(V, c):
        """Chooses column indices from a matrix given its SVD."""
        # extract number of samples and rank
        n_features, k = V.shape

        # calculate normalized leverage score
        pi = np.sum(V**2, axis=1) / k

        # iterate through columns
        column_flags = np.zeros(n_features, dtype=bool)

        for column in range(n_features):
            p = min(1, c * pi[column])

            column_flags[column] = p > np.random.rand()

        return column_flags
