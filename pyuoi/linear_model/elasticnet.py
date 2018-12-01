from .base import AbstractUoILinearRegressor

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.linear_model.coordinate_descent import _alpha_grid


class UoI_ElasticNet(AbstractUoILinearRegressor):

    def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
        n_lambdas=48, stability_selection=1., eps=1e-3, warm_start=True,
        estimation_score='r2',
        copy_X=True, fit_intercept=True, normalize=True, random_state=None, max_iter=1000
    ):
        super(UoI_Lasso, self).__init__(
            n_boots_sel = n_boos_sel,
            n_boots_est = n_boos_est,
            selection_frac=selection_frac,
            stability_selection=stability_selection,
            copy_X=copy_X,
            fit_intercept=fit_intercept,
            normalize=normalize
        )
        self.n_lambdas = n_lambdas
        self.eps = eps
        self.__selection_lm = Lasso(
            normalize=normalize,
            max_iter=max_iter,
            warm_start=warm_start,
            random_state=random_state
        )
        self.__estimation_lm = LinearRegression(random_state=random_state)

    @property
    def estimation_lm(self):
        return self.__estimation_lm

    @property
    def selection_lm(self):
        return self.__selection_lm

    def get_reg_params(self, X, y):
        return _alpha_grid(
            X=X, y=y,
            l1_ratio=1.0,
            fit_intercept=self.fit_intercept,
            eps=self.eps,
            n_alphas=self.n_lambdas,
            normalize=self.normalize
        )
