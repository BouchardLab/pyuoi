import numpy as np

from .base import AbstractUoILinearRegressor

from sklearn.linear_model import LinearRegression
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.linear_model import ElasticNet


class UoI_ElasticNet(AbstractUoILinearRegressor):

    def __init__(self, n_lambdas=48, alphas=np.array([0.5]),
                 n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
                 stability_selection=1., estimation_score='r2',
                 warm_start=True, eps=1e-3,
                 copy_X=True, fit_intercept=True, normalize=True,
                 random_state=None, max_iter=1000,
                 comm=None):
        super(UoI_ElasticNet, self).__init__(
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            stability_selection=stability_selection,
            estimation_score=estimation_score,
            copy_X=copy_X,
            fit_intercept=fit_intercept,
            normalize=normalize,
            comm=comm
        )
        self.n_lambdas = n_lambdas
        self.alphas = alphas
        self.n_alphas = len(alphas)
        self.warm_start = warm_start
        self.eps = eps
        self.lambdas = None
        self.__selection_lm = ElasticNet(
            fit_intercept=fit_intercept,
            normalize=normalize,
            max_iter=max_iter,
            copy_X=copy_X,
            warm_start=warm_start,
            random_state=random_state)
        self.__estimation_lm = LinearRegression()

    @property
    def estimation_lm(self):
        return self.__estimation_lm

    @property
    def selection_lm(self):
        return self.__selection_lm

    def get_reg_params(self, X, y):
        """Calculates the regularization parameters (alpha and lambda) to bbe
        used for the provided data.

        Note that the Elastic Net has two regularization parameters: alpha,
        which specifies the ratio between the L1 and L2 penalties, and lambda,
        which specifies the penalty on both regularization terms. We adhere to
        these variable names, but scikit-learn uses l1_ratio for alpha and
        alpha for lambda, which can become confusing.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The design matrix.

        y : array-like, shape (n_samples)
            The response vector.

        Returns
        -------
        reg_params : a list of dictionaries
            A list containing dictionaries with the value of each
            (lambda, alpha) describing the type of regularization to impose.
            The keys adhere to scikit-learn's terminology (lambda->alpha,
            alpha->l1_ratio). This allows easy passing into the ElasticNet
            object.
        """
        if self.lambdas is None:
            self.lambdas = np.zeros((self.n_alphas, self.n_lambdas))
            # a set of lambdas are generated for each alpha value (l1_ratio in
            # sci-kit learn parlance)
            for alpha_idx, alpha in enumerate(self.alphas):
                self.lambdas[alpha_idx, :] = _alpha_grid(
                    X=X, y=y,
                    l1_ratio=alpha,
                    fit_intercept=self.fit_intercept,
                    eps=self.eps,
                    n_alphas=self.n_lambdas,
                    normalize=self.normalize)

        # place the regularization parameters into a list of dictionaries
        reg_params = list()
        for alpha_idx, alpha in enumerate(self.alphas):
            for lamb_idx, lamb in enumerate(self.lambdas[alpha_idx]):
                # reset the regularization parameter
                reg_params.append(dict(alpha=lamb, l1_ratio=alpha))

        return reg_params
