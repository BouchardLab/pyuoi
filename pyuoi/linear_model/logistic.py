from .base import AbstractUoILinearClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import l1_min_c

import numpy as np


class UoI_L1Logistic(AbstractUoILinearClassifier):

    def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
        n_C=48, stability_selection=1., warm_start=True,
        estimation_score='acc',
        copy_X=True, fit_intercept=True, normalize=True, random_state=None, max_iter=1000,
        logger=None,
        comm=None
    ):
        super(UoI_L1Logistic, self).__init__(
            n_boots_sel = n_boots_sel,
            n_boots_est = n_boots_est,
            selection_frac=selection_frac,
            stability_selection=stability_selection,
            copy_X=copy_X,
            fit_intercept=fit_intercept,
            normalize=normalize,
            random_state=random_state,
            logger=logger,
            comm=comm
        )
        self.n_C = n_C
        self.Cs = None
        self.__selection_lm = LogisticRegression(
            penalty='l1',
            solver='saga',
            max_iter=max_iter,
            warm_start=warm_start,
            random_state=self.random_state
        )
        # sklearn cannot do LogisticRegression without penalization, due to the ill-posed nature
        # of the problem. We may want to set C=np.inf for no penalization, but
        # we risk no convergence.
        self.__estimation_lm = LogisticRegression(solver='saga', penalty='l1', random_state=random_state, warm_start=warm_start)

    @property
    def estimation_lm(self):
        return self.__estimation_lm

    @property
    def selection_lm(self):
        return self.__selection_lm

    def get_reg_params(self, X, y):
        if self.Cs is None:
            self.Cs = l1_min_c(X, y, loss='log') * np.logspace(0, 7, self.n_C)
        ret = list()
        for c in reversed(self.Cs):
            ret.append(dict(C=c))
        return ret
