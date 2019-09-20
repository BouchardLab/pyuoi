import numpy as np

from .base import AbstractUoILinearRegressor

from sklearn.linear_model import LinearRegression
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.linear_model import ElasticNet


class UoI_ElasticNet(AbstractUoILinearRegressor, LinearRegression):
    r"""UoI\ :sub:`ElasticNet` solver.

    Parameters
    ----------
    n_boots_sel : int
        The number of data bootstraps to use in the selection module.
        Increasing this number will make selection more strict.
    n_boots_est : int
        The number of data bootstraps to use in the estimation module.
        Increasing this number will relax selection and decrease variance.
    selection_frac : float
        The fraction of the dataset to use for training in each resampled
        bootstrap, during the selection module. Small values of this parameter
        imply larger "perturbations" to the dataset.
    estimation_frac : float
        The fraction of the dataset to use for training in each resampled
        bootstrap, during the estimation module. The remaining data is used
        to obtain validation scores. Small values of this parameters imply
        larger "perturbations" to the dataset. IGNORED - Leaving this here
        to double check later
    n_lambdas : int
        The number of regularization values to use for selection.
    alphas : list or ndarray
        The parameter that trades off L1 versus L2 regularization for a given
        lambda.
    stability_selection : int, float, or array-like
        If int, treated as the number of bootstraps that a feature must
        appear in to guarantee placement in selection profile. If float,
        must be between 0 and 1, and is instead the proportion of
        bootstraps. If array-like, must consist of either ints or floats
        between 0 and 1. In this case, each entry in the array-like object
        will act as a separate threshold for placement in the selection
        profile.
    estimation_score : string, "r2" | "AIC" | "AICc" | "BIC"
        Objective used to choose the best estimates per bootstrap.
    estimation_target : string, "train" | "test"
        Decide whether to assess the estimation_score on the train
        or test data across each bootstrap. By default, a sensible
        choice is made based on the chosen estimation_score.
    warm_start : bool
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution
    eps : float
        Length of the lasso path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.
    copy_X : bool
        If ``True``, X will be copied; else, it may be overwritten.
    fit_intercept : bool
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    standardize : bool
        If True, the regressors X will be standardized before regression by
        subtracting the mean and dividing by their standard deviations.
    max_iter : int
        Maximum number of iterations for iterative fitting methods.
    random_state : int, RandomState instance, or None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.
    comm : MPI communicator
        If passed, the selection and estimation steps are parallelized.
    logger : Logger
        The logger to use for messages when ``verbose=True`` in ``fit``.
        If *None* is passed, a logger that writes to ``sys.stdout`` will be
        used.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
    intercept_ : float
        Independent term in the linear model.
    supports_ : ndarray, shape (n_supports, n_features)
        Boolean array indicating whether a given regressor (column) is selected
        for estimation for a given regularization parameter value (row).
    """
    def __init__(self, n_boots_sel=24, n_boots_est=24, selection_frac=0.9,
                 estimation_frac=0.9, n_lambdas=48,
                 alphas=np.array([0.5]), stability_selection=1.,
                 estimation_score='r2', estimation_target=None,
                 warm_start=True, eps=1e-3, copy_X=True,
                 fit_intercept=True, standardize=True,
                 max_iter=1000, random_state=None, comm=None, logger=None):
        super(UoI_ElasticNet, self).__init__(
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            estimation_frac=estimation_frac,
            stability_selection=stability_selection,
            estimation_score=estimation_score,
            estimation_target=estimation_target,
            copy_X=copy_X,
            fit_intercept=fit_intercept,
            standardize=standardize,
            random_state=random_state,
            comm=comm,
            max_iter=max_iter,
            logger=logger)
        self.n_lambdas = n_lambdas
        self.alphas = alphas
        self.n_alphas = len(alphas)
        self.warm_start = warm_start
        self.eps = eps
        self.lambdas = None
        self._selection_lm = ElasticNet(
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            copy_X=copy_X,
            warm_start=warm_start,
            random_state=random_state)
        self._estimation_lm = LinearRegression(fit_intercept=fit_intercept)

    def get_reg_params(self, X, y):
        r"""Calculates the regularization parameters (alpha and lambda) to be
        used for the provided data.

        Note that the Elastic Net penalty is given by

        .. math::
           \frac{1}{2\ \text{n_samples}} ||y - Xb||^2_2
           + \lambda (\alpha |b|_1 + 0.5 (1 - \alpha) |b|^2_2)

        where lambda and alpha are regularization parameters.

        ``scikit-learn`` does not use these names. Instead, ``scitkit-learn``
        denotes alpha by 'l1_ratio' and lambda by 'alpha'.

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
                    n_alphas=self.n_lambdas)

        # place the regularization parameters into a list of dictionaries
        reg_params = list()
        for alpha_idx, alpha in enumerate(self.alphas):
            for lamb_idx, lamb in enumerate(self.lambdas[alpha_idx]):
                # reset the regularization parameter
                reg_params.append(dict(alpha=lamb, l1_ratio=alpha))

        return reg_params
