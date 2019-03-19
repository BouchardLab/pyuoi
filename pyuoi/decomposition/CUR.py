from sklearn.base import BaseEstimator
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
