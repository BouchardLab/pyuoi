import numpy as np

from .base import AbstractUoILinearRegressor

from sklearn.linear_model import LinearRegression
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.linear_model import ElasticNet


class UoI_ElasticNet(AbstractUoILinearRegressor):

    def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
                 n_lambdas=48,
                 alphas=np.array([1., 0.99, 0.95, 0.9, 0.5, 0.3, 0.1]),
                 stability_selection=1., eps=1e-3, warm_start=True,
                 estimation_score='r2',
                 copy_X=True, fit_intercept=True, normalize=True,
                 random_state=None, max_iter=1000,
                 comm=None):
        super(UoI_ElasticNet, self).__init__(
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            stability_selection=stability_selection,
            copy_X=copy_X,
            fit_intercept=fit_intercept,
            normalize=normalize,
            comm=comm
        )
        self.n_lambdas = n_lambdas
        self.alphas = alphas
        self.n_alphas = len(alphas)
        self.eps = eps
        self.lambdas = None
        self.__selection_lm = ElasticNet(
            normalize=normalize,
            max_iter=max_iter,
            warm_start=warm_start,
            random_state=random_state
        )
        self.__estimation_lm = LinearRegression()

    @property
    def estimation_lm(self):
        return self.__estimation_lm

    @property
    def selection_lm(self):
        return self.__selection_lm

    def get_reg_params(self, X, y):
        if self.lambdas is None:
            self.lambdas = np.zeros((self.n_alphas, self.n_lambdas))
            for alpha_idx, alpha in enumerate(self.alphas):
                self.lambdas[alpha_idx, :] = _alpha_grid(
                    X=X, y=y,
                    l1_ratio=alpha,
                    fit_intercept=self.fit_intercept,
                    eps=self.eps,
                    n_alphas=self.n_lambdas,
                    normalize=self.normalize
                )
        ret = list()
        for alpha_idx, alpha in enumerate(self.alphas):
            for lamb_idx, lamb in enumerate(self.lambdas[alpha_idx]):
                # reset the regularization parameter
                ret.append(dict(alpha=lamb, l1_ratio=alpha))
        return ret
