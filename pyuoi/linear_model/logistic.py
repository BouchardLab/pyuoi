from .base import AbstractUoILinearClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import l1_min_c
from sklearn.metrics import log_loss

from scipy.optimize import minimize

import numpy as np

from ..utils import sigmoid


class UoI_L1Logistic(AbstractUoILinearClassifier):

    def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
                 estimation_frac=0.9, n_C=48, stability_selection=1.,
                 warm_start=True, estimation_score='acc',
                 copy_X=True, fit_intercept=True, normalize=True,
                 random_state=None, max_iter=1000,
                 comm=None):
        super(UoI_L1Logistic, self).__init__(
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            estimation_frac=estimation_frac,
            stability_selection=stability_selection,
            estimation_score=estimation_score,
            random_state=random_state,
            copy_X=copy_X,
            fit_intercept=fit_intercept,
            normalize=normalize,
            comm=comm)
        self.n_C = n_C
        self.Cs = None
        self.__selection_lm = LogisticRegression(
            penalty='l1',
            solver='saga',
            max_iter=max_iter,
            warm_start=warm_start,
            random_state=random_state,
            multi_class='auto',
            fit_intercept=fit_intercept,
        )
        # sklearn cannot do LogisticRegression without penalization, due to the
        # ill-posed nature of the problem. We may want to set C=np.inf for no
        # penalization, but we risk no convergence.
        self.__estimation_lm = LogisticRegression(
            C=1e10,
            random_state=random_state,
            solver='lbfgs',
            multi_class='auto',
            fit_intercept=fit_intercept)

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
        for c in self.Cs:
            ret.append(dict(C=c))
        return ret

    def _fit_intercept_no_features(self, y):
        """"Fit a model with only an intercept.

        This is used in cases where the model has no support selected.
        """
        return LogisticInterceptFitterNoFeatures(y)

    def _fit_intercept(self, X, y):
        if self.fit_intercept:
            self.intercept_ = fit_intercept_fixed_coef(X, self.coef_,
                                                       y, self._n_classes)
        else:
            n = self._n_classes
            if self._n_classes == 2:
                n = 1
            self.intercept_ = np.zeros(n)


def fit_intercept_fixed_coef(X, coef_, y, n_classes):
    if n_classes == 2:
        n_classes = 1

    def f_df(bias):
        py = sigmoid(X.dot(coef_.T) + bias)
        dfdb = py.mean() - y.mean()
        return log_loss(y, py), np.atleast_1d(dfdb)

    opt = minimize(f_df, np.atleast_1d(np.zeros(n_classes)),
                   method='BFGS', jac=True)
    return opt.x


class LogisticInterceptFitterNoFeatures(object):
    def __init__(self, y):
        p = y.mean()
        self.intercept_ = np.log(p / (1. - p))

    def predict(self, X):
        n_samples = X.shape[0]
        return np.tile(int(self.intercept_ >= 0.), n_samples)

    def predict_proba(self, X):
        n_samples = X.shape[0]
        return np.tile(sigmoid(self.intercept_), n_samples)
