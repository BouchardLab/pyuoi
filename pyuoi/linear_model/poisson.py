import numpy as np

from .base import AbstractUoIGeneralizedLinearRegressor

from pyuoi import utils

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from ..lbfgs import fmin_lbfgs


class Poisson(BaseEstimator):
    r"""Generalized Linear Model with exponential link function (i.e. Poisson)
    trained with L1/L2 regularizer (i.e. Elastic net penalty).

    The log-likelihood of the Poisson GLM is optimized by performing coordinate
    descent on a linearized quadratic approximation. See Chapter 5 of Hastie,
    Tibshirani, and Wainwright (2016) for more details.

    Parameters
    ----------
    alpha : float
        Constant that multiplies the L1 term. Defaults to 1.0.
    l1_ratio : float
        Float between 0 and 1 acting as a scaling between l1 and l2 penalties.
        For ``l1_ratio = 0`` the penalty is an L2 penalty. For ``l1_ratio = 1``
        it is an L1 penalty. For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2 penalties.
    fit_intercept : bool
        Whether to fit an intercept or not.
    standardize : bool
        If ``True``, centers the design matrix across samples and rescales them
        to have standard deviation of 1.
    tol : float
        The tolerance for the optimization: if the updates are smaller than
        ``tol``, the optimization code checks the dual gap for optimality and
        continues until it is smaller than ``tol``.
    warm_start : bool
        When set to ``True``, reuse the solution of the previous call to
        fit as initialization, otherwise, just erase the previous solution.
    solver : string, 'lbfgs' | 'cd'
        The solver to use. Options are 'lbfgs' (orthant-wise LBFGS) and 'cd'
        (coordinate descent).

    Attributes
    ----------
    coef_ : ndarray, shape (n_features,)
        The fitted parameter vector.
    intercept_ : float
        The fitted intercept.
    """
    def __init__(self, alpha=1.0, l1_ratio=1., fit_intercept=True,
                 standardize=False, max_iter=1000, tol=1e-5, warm_start=False,
                 solver='lbfgs'):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.solver = solver

    def fit(self, X, y, sample_weight=None):
        """Fit the Poisson GLM.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The design matrix.
        y : ndarray, shape (n_samples,)
            Response vector. Will be cast to X's dtype if necessary. Currently,
            this implementation does not handle multiple response variables.
        sample_weight : array-like, shape (n_samples,)
            Array of weights assigned to the individual samples. If ``None``,
            then each sample is provided an equal weight.
        """
        self.n_samples, self.n_features = X.shape
        X, y = self._pre_fit(X, y)

        # check if the response variable is constant
        if np.unique(y).size == 1:
            self.coef_ = np.zeros(self.n_features)
            if np.any(y):
                self.intercept_ = np.log(y.mean())
            else:
                self.intercept_ = -np.inf
            return self

        if self.solver == 'lbfgs':
            coef = np.zeros(self.n_features)
            intercept = 0
            if self.warm_start:
                if hasattr(self, 'coef_'):
                    if self.standardize:
                        coef = self.coef_ * self._X_scaler.scale_
                    else:
                        coef = self.coef_
                if hasattr(self, 'intercept_') and self.fit_intercept:
                    intercept = self.intercept_

            if self.fit_intercept:
                coef = np.append(coef, intercept)

            # create lbfgs function
            def func(x, g, *args):
                loss, grad = _poisson_loss_and_grad(x, *args)
                g[:] = grad
                return loss

            l1_penalty = self.alpha * self.l1_ratio
            l2_penalty = self.alpha * (1 - self.l1_ratio)

            # orthant-wise lbfgs optimization
            coef = fmin_lbfgs(func, coef,
                              orthantwise_c=l1_penalty,
                              args=(X, y, l2_penalty, sample_weight),
                              max_iterations=self.max_iter,
                              epsilon=self.tol,
                              orthantwise_end=self.n_features)

            if self.fit_intercept:
                self.coef_ = coef[:self.n_features]
                self.intercept_ = coef[-1]
            else:
                self.coef_ = coef

        # coordinate descent
        elif self.solver == 'cd':
            self.coef_, intercept = self._cd(X=X, y=y,
                                             sample_weight=sample_weight)

            if self.fit_intercept:
                self.intercept_ = intercept

        else:
            raise ValueError('Solver not available.')

        self._post_fit(X, y)
        return self

    def predict(self, X):
        """Predicts the response variable given a design matrix. The output is
        the mode of the Poisson distribution.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Design matrix to predict on.

        Returns
        -------
        mode : array_like, shape (n_samples)
            The predicted response values, i.e. the modes.
        """
        check_is_fitted(self, ['coef_', 'intercept_'])
        mu = np.exp(self.intercept_ + np.dot(X, self.coef_))
        mode = np.floor(mu)
        return mode

    def predict_mean(self, X):
        """Calculates the mean response variable given a design matrix.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Design matrix to predict on.

        Returns
        -------
        mu : array_like, shape (n_samples)
            The predicted response values, i.e. the conditional means.
        """
        check_is_fitted(self, ['coef_', 'intercept_'])
        mu = np.exp(self.intercept_ + np.dot(X, self.coef_))
        return mu

    def _cd(self, X, y, sample_weight=None):
        """Performs coordinate descent on a dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The design matrix.
        y : ndarray, shape (n_samples,)
            The response vector.

        sample_weight : array-like, shape (n_samples,), default None
            Array of weights assigned to the individual samples. If None, then
            each sample is provided an equal weight.

        Returns
        -------
        coef : ndarray, shape (n_features,)
            The fitted coefficients.
        intercept : float
            The fitted intercept. If fit_intercept = False, this will be equal
            to zero.
        """
        n_samples, n_features = X.shape

        if sample_weight is None:
            sample_weight = np.ones(n_samples)

        # initialization of coef and intercepts
        coef = np.zeros(n_features)
        intercept = 0
        # warm start coefficients, if necessary
        if self.warm_start:
            if hasattr(self, 'coef_'):
                if self.standardize:
                    coef = self.coef_ * self._X_scaler.scale_
                else:
                    coef = self.coef_

            if self.fit_intercept:
                if hasattr(self, 'intercept_'):
                    intercept = self.intercept_

        # every coefficient is active unless there's a warm start
        if self.warm_start:
            active_idx = np.argwhere(coef != 0)
        else:
            active_idx = np.arange(self.n_features)

        prev_coef = np.copy(coef)
        # perform coordinate descent updates
        for iteration in range(self.max_iter):
            # linearize the log-likelihood
            w, z = self.adjusted_response(X, y, coef, intercept)
            # rescale weights by sample_weight
            w *= sample_weight

            # perform an update of coordinate descent
            coef, intercept = self._cd_sweep(
                coef=coef, intercept=intercept, X=X, w=w, z=z,
                active_idx=active_idx)

            # check convergence
            if np.max(np.abs(prev_coef - coef)) < self.tol:
                break

            prev_coef = np.copy(coef)

            # update the active features
            active_idx = np.argwhere(coef != 0).ravel()

        return coef, intercept

    def _cd_sweep(self, coef, X, w, z, active_idx, intercept=0):
        """Performs one sweep of coordinate descent updates over a set of
        'active' features.

        Parameters
        ----------
        coef : ndarray, shape (n_features,)
            The current estimates of the parameters.
        X : ndarray, shape (n_samples, n_features)
            The design matrix.
        w : ndarray, shape (n_samples,)
            The weights applied to each sample after linearization of the
            log-likelihood.
        z : ndarray, shape (n_samples,)
            The working response after linearization of the log-likelihood.
        active_idx : ndarray, shape (n_features,)
            An array of ints denoting the indices of features that should be
            updated.
        intercept : float
            The current estimate of the intercept.

        Returns
        -------
        coef : ndarray, shape (n_features,)
            The updated parameters after one coordinate descent sweep.
        intercept : float
            The update intercept after one coordinate descent sweep.
        """
        n_features = coef.size
        n_samples = X.shape[0]

        # intercept is not penalized
        if self.fit_intercept:
            z_hat = np.dot(X, coef)
            residuals = z - z_hat
            # update intercept
            intercept = np.dot(w, residuals) / np.sum(w)
        else:
            intercept = 0

        # iterate over the active features
        for idx in active_idx:
            # remove the current feature from the update rules
            mask = np.ones(n_features)
            mask[idx] = 0

            z_hat = intercept + np.dot(X, coef * mask)
            x = X[:, idx]
            # equation 5.44; Hastie, Tibshirani, Wainwright (2016)
            num_update = np.dot(w, x * (z - z_hat)) / n_samples
            den_update = np.dot(w, x**2) / n_samples
            # replace coefficients sequentially
            coef[idx] = \
                self.soft_threshold(num_update, self.l1_ratio * self.alpha) \
                / (den_update + self.alpha * (1 - self.l1_ratio))

        return coef, intercept

    def _pre_fit(self, X, y):
        """Perform standardization, if needed, before fitting."""
        if self.standardize:
            self._X_scaler = StandardScaler()
            X = self._X_scaler.fit_transform(X)
        return X, y

    def _post_fit(self, X, y):
        """Rescale coefficients, if needed, after fitting."""
        if self.standardize:
            sX = self._X_scaler
            self.coef_ /= sX.scale_

    @staticmethod
    def soft_threshold(X, threshold):
        """Performs the soft-thresholding necessary for coordinate descent
        lasso updates.

        Parameters
        ----------
        X : array-like, shape (n_features, n_samples)
            Matrix to be thresholded.
        threshold : float
            Soft threshold.

        Returns
        -------
        X_soft_threshold : array-like
            Soft thresholded X.
        """
        X_thresholded = (np.abs(X) > threshold).astype('int')
        X_soft_threshold = np.sign(X) * (np.abs(X) - threshold) * X_thresholded
        return X_soft_threshold

    @staticmethod
    def adjusted_response(X, y, coef, intercept=0):
        """Calculates the adjusted response when posing the fitting procedure
        as Iteratively Reweighted Least Squares (Newton update on log
        likelihood).

        Parameters
        ----------
        X : array-like, shape (n_features, n_samples)
            The design matrix.
        y : array-like, shape (n_samples)
            The response vector.
        coef : array-like, shape (n_features)
            Current estimate of the parameters.
        intercept : float
            The current estimate of the intercept.

        Returns
        -------
        w : array-like, shape (n_samples)
            Weights for samples. The log-likelihood for a GLM, when posed as
            a linear regression problem, requires reweighting the samples.
        z : array-like, shape (n_samples)
            Working response. The linearized response when rewriting coordinate
            descent for the Poisson likelihood as a iteratively reweighted
            least squares.
        """
        Xbeta = intercept + np.dot(X, coef)
        w = np.exp(Xbeta)
        z = Xbeta + (y / w) - 1.
        return w, z


class UoI_Poisson(AbstractUoIGeneralizedLinearRegressor, Poisson):
    r"""UoI\ :sub:`Poisson` solver.

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
        larger "perturbations" to the dataset.
    n_lambdas : int
        The number of regularization values to use for selection.
    alphas : list or ndarray of floats
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
    estimation_score : string, "log" | "AIC", | "AICc" | "BIC"
        Objective used to choose the best estimates per bootstrap.
    estimation_target : string, "train" | "test"
        Decide whether to assess the estimation_score on the train or test data
        across each bootstrap. By deafult, a sensible choice is made based on
        the chosen ``estimation_score``.
    solver : string, 'lbfgs' | 'cd'
        The solver to use. Options are 'lbfgs' (orthant-wise LBFGS) and 'cd'
        (coordinate descent).
    warm_start : bool
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution
    eps : float
        Length of the lasso path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.
    tol : float
        Stopping criteria for solver.
    copy_X : boolean
        If ``True``, X will be copied; else, it may be overwritten.
    fit_intercept : boolean
        Whether to calculate the intercept for this model. If set to ``False``,
        no intercept will be used in calculations (e.g. data is expected to be
        already centered).
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
        RandomState instance used by ``np.random``.
    comm : MPI communicator
        If passed, the selection and estimation steps are parallelized.

    Attributes
    ----------
    coef_ : ndarray, shape (n_features,) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
    intercept_ : float
        Independent term in the linear model.
    supports_ : ndarray, shape (n_supports, n_features)
        Boolean array indicating whether a given regressor (column) is selected
        for estimation for a given regularization parameter value (row).
    """
    def __init__(self, n_boots_sel=24, n_boots_est=24, n_lambdas=48,
                 alphas=np.array([1.]), selection_frac=0.8,
                 estimation_frac=0.8, stability_selection=1.,
                 estimation_score='log', estimation_target=None,
                 solver='lbfgs', warm_start=True,
                 eps=1e-3, tol=1e-5, copy_X=True, fit_intercept=True,
                 standardize=True, max_iter=1000,
                 random_state=None, comm=None, logger=None):
        super(UoI_Poisson, self).__init__(
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            estimation_frac=estimation_frac,
            stability_selection=stability_selection,
            estimation_score=estimation_score,
            estimation_target=estimation_target,
            copy_X=copy_X,
            fit_intercept=fit_intercept,
            random_state=random_state,
            comm=comm,
            logger=logger)
        self.n_lambdas = n_lambdas
        self.alphas = alphas
        self.n_alphas = len(alphas)
        self.warm_start = warm_start
        self.eps = eps
        self.lambdas = None
        self._selection_lm = Poisson(
            fit_intercept=fit_intercept,
            standardize=standardize,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            solver=solver)
        # estimation is a Poisson regression with no regularization
        self._estimation_lm = Poisson(
            alpha=0.,
            l1_ratio=1.,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            warm_start=False,
            solver=solver)

    def get_reg_params(self, X, y):
        r"""Calculates the regularization parameters (alpha and lambda) to be
        used for the provided data.

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
        n_samples = X.shape[0]
        if self.lambdas is None:
            self.lambdas = np.zeros((self.n_alphas, self.n_lambdas))
            # a set of lambdas are generated for each alpha value (l1_ratio in
            # sci-kit learn parlance)
            for alpha_idx, alpha in enumerate(self.alphas):
                # calculate upper bound for lambda sweep
                ybar = y.mean()
                lambda_max = np.max(np.abs(np.dot(X.T, y - ybar)))
                lambda_max /= n_samples * alpha
                self.lambdas[alpha_idx, :] = np.logspace(
                    start=np.log10(lambda_max),
                    stop=np.log10(self.eps * lambda_max),
                    num=self.n_lambdas)

        # place the regularization parameters into a list of dictionaries
        reg_params = list()
        for alpha_idx, alpha in enumerate(self.alphas):
            for lamb_idx, lamb in enumerate(self.lambdas[alpha_idx]):
                # reset the regularization parameter
                reg_params.append(dict(alpha=lamb, l1_ratio=alpha))

        return reg_params

    def _score_predictions(self, metric, fitter, X, y, support, boot_idxs=None):
        """Score, according to some metric, predictions provided by a model.

        The resulting score will be negated if an information criterion is
        specified.

        Parameters
        ----------
        metric : string
            The type of score to run on the prediction. Valid options include
            'r2' (explained variance), 'BIC' (Bayesian information criterion),
            'AIC' (Akaike information criterion), and 'AICc' (corrected AIC).
        fitter : Poisson object
            The Poisson object that has been fit to the data with the
            respective hyperparameters.
        X : ndarray, shape (n_samples, n_features)
            The design matrix.
        y : ndarray, shape (n_samples,)
            The response vector.
        support: ndarray
            The indices of the non-zero features.
        boot_idxs : 2-tuple of array-like objects
            Tuple of (train_idxs, test_idxs) generated from a bootstrap
            sample. If this is specified, then the appropriate set of
            data will be used for evaluating scores: test data for r^2,
            and training data for information criteria

        Returns
        -------
        score : float
            The score.
        """

        # Select the train data
        if boot_idxs is not None:
            X = X[boot_idxs[self._estimation_target]]
            y = y[boot_idxs[self._estimation_target]]

        # for Poisson, use predict_mean to calculate the "predicted" values
        y_pred = fitter.predict_mean(X[:, support])
        # calculate the log-likelihood
        ll = utils.log_likelihood_glm(model='poisson', y_true=y, y_pred=y_pred)
        if metric == 'log':
            score = ll
        # information criteria
        else:
            n_features = np.count_nonzero(support)
            if fitter.intercept_ != 0:
                n_features += 1
            n_samples = X.shape[0]
            if metric == 'BIC':
                score = utils.BIC(n_samples * ll, n_features, n_samples)
            elif metric == 'AIC':
                score = utils.AIC(n_samples * ll, n_features)
            elif metric == 'AICc':
                score = utils.AICc(n_samples * ll, n_features, n_samples)
            else:
                raise ValueError(metric + ' is not a valid metric.')
            # negate the score since lower information criterion is preferable
            score = -score

        return score

    def _pre_fit(self, X, y):
        """Perform standardization, if needed, before fitting."""
        if self.standardize:
            self._X_scaler = StandardScaler()
            X = self._X_scaler.fit_transform(X)
        if y.ndim == 2:
            self.output_dim = y.shape[1]
        else:
            self.output_dim = 1
        return X, y

    def _post_fit(self, X, y):
        """Rescale coefficients, if needed, after fitting."""
        if self.output_dim == 1:
            self.coef_ = np.squeeze(self.coef_)
        if self.standardize:
            sX = self._X_scaler
            if self.fit_intercept:
                self.intercept_ -= np.sum((sX.mean_ * self.coef_) / sX.scale_)
            self.coef_ /= sX.scale_

    def _fit_intercept(self, X, y):
        """"Fit a model with an intercept and fixed coefficients.

        This is used to re-fit the intercept after the coefficients are
        estimated.
        """
        if self.fit_intercept:
            mu = np.exp(np.dot(X, self.coef_.T))
            if np.any(y):
                self.intercept_ = np.log(np.mean(y) / np.mean(mu))
            else:
                self.intercept_ = -np.inf
        else:
            self.intercept_ = np.zeros(1)

    def _fit_intercept_no_features(self, y):
        """"Fit a model with only an intercept.

        This is used in cases where the model has no support selected.
        """
        return PoissonInterceptFitterNoFeatures(y)


class PoissonInterceptFitterNoFeatures(object):
    def __init__(self, y):
        if np.any(y):
            self.intercept_ = np.log(y.mean())
        else:
            self.intercept_ = -np.inf

    def predict(self, X):
        """Predicts the response variable given a design matrix. The output is
        the mode of the Poisson distribution.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Design matrix to predict on.

        Returns
        -------
        mode : array_like, shape (n_samples)
            The predicted response values, i.e. the modes.
        """
        mu = np.exp(self.intercept_)
        mode = np.floor(mu)
        return mode

    def predict_mean(self, X):
        """Calculates the mean response variable given a design matrix.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Design matrix to predict on.

        Returns
        -------
        mu : array_like, shape (n_samples)
            The predicted response values, i.e. the conditional means.
        """
        mu = np.exp(self.intercept_)
        return mu


def _poisson_loss_and_grad(coef, X, y, l2_penalty, sample_weight=None):
    """Computes the Poisson loss and gradient.

    Parameters
    ----------
    coef : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : array-like, shape (n_samples, n_features)
        Design matrix.
    y : ndarray, shape (n_samples,)
        Response vector.
    l2_penalty : float
        Regularization parameter on l2-norm.
    sample_weight : array-like, shape (n_samples,), default None
        Array of weights assigned to the individual samples. If None, then each
        sample is provided an equal weight.

    Returns
    -------
    out : float
        Negative log-likelihood of Poisson GLM.
    grad : ndarray, shape (n_features,) or (n_features + 1,)
        Poisson gradient.
    """
    n_samples, n_features = X.shape
    grad = np.empty_like(coef)

    if sample_weight is None:
        sample_weight = np.ones(n_samples)

    # extract intercept
    if grad.shape[0] > n_features:
        intercept = coef[-1]
        coef = coef[:n_features]
    else:
        intercept = 0

    # calculate likelihood
    eta = intercept + np.dot(X, coef)
    out = -np.sum(sample_weight * (y * eta - np.exp(eta))) / n_samples
    out += 0.5 * l2_penalty * np.dot(coef, coef)

    # gradient of parameters
    y_res = sample_weight * (y - np.exp(eta))
    grad[:n_features] = -np.dot(X.T, y_res) / n_samples + l2_penalty * coef

    # gradient of intercept
    if grad.shape[0] > n_features:
        grad[-1] = -np.mean(y_res)

    return out, grad
