import numpy as np

from .base import AbstractUoILinearRegressor

from pyuoi import utils

from sklearn.exceptions import NotFittedError
from sklearn.linear_model.base import LinearModel
from sklearn.linear_model.base import _preprocess_data
from sklearn.utils import check_X_y

class UoI_Poisson(AbstractUoILinearRegressor):

    _AbstractUoILinearRegressor__valid_estimation_metrics = \
        ('log', 'AIC', 'AICc', 'BIC')

    def __init__(self, n_lambdas=48, alphas=np.array([0.5]),
                 n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
                 estimation_frac=0.9, stability_selection=1.,
                 estimation_score='log', warm_start=True, eps=1e-3,
                 tol=1e-5, copy_X=True, fit_intercept=True,
                 standardize=True, random_state=None, max_iter=1000,
                 comm=None):
        super(UoI_Poisson, self).__init__(
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            estimation_frac=estimation_frac,
            stability_selection=stability_selection,
            estimation_score=estimation_score,
            copy_X=copy_X,
            fit_intercept=fit_intercept,
            standardize=standardize,
            random_state=random_state,
            comm=comm)
        self.n_lambdas = n_lambdas
        self.alphas = alphas
        self.n_alphas = len(alphas)
        self.warm_start = warm_start
        self.eps = eps
        self.lambdas = None
        self.__selection_lm = Poisson(
            fit_intercept=fit_intercept,
            standardize=standardize,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start)
        # estimation is a Poisson regression with no regularization
        self.__estimation_lm = Poisson(
            alpha=0,
            l1_ratio=0,
            fit_intercept=fit_intercept,
            standardize=standardize,
            max_iter=max_iter,
            tol=tol,
            warm_start=False)

    @property
    def estimation_lm(self):
        return self.__estimation_lm

    @property
    def selection_lm(self):
        return self.__selection_lm

    def fit(self, X, y, stratify=None, verbose=False):
        """Fit data according to the UoI algorithm.

        Additionaly, perform X-y checks, data preprocessing, and setting
        intercept.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            The design matrix.

        y : ndarray, shape (n_samples,)
            Response vector. Will be cast to X's dtype if necessary.
            Currently, this implementation does not handle multiple response
            variables.

        stratify : array-like or None, default None
            Ensures groups of samples are alloted to training/test sets
            proportionally. Labels for each group must be an int greater
            than zero. Must be of size equal to the number of samples, with
            further restrictions on the number of groups.
        """
        # perform checks
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         y_numeric=True, multi_output=True)
        # preprocess data
        X, y, X_offset, y_offset, X_scale = self.preprocess_data(X, y)
        super(AbstractUoILinearRegressor, self).fit(X, y, stratify=stratify,
                                                    verbose=verbose)
        self.coef_ = self.coef_.squeeze()
        self._fit_intercept(X, y)
        return self

    def get_reg_params(self, X, y):
        """Calculates the regularization parameters (alpha and lambda) to be
        used for the provided data.

        Note that the Elastic Net penalty is given by

                1 / (2 * n_samples) * ||y - Xb||^2_2
            + lambda * (alpha * |b|_1 + 0.5 * (1 - alpha) * |b|^2_2)

        where lambda and alpha are regularization parameters.

        Scikit-learn does not use these names. Instead, scitkit-learn
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
                self.lambdas[alpha_idx, :] = np.logspace(
                    start=np.log10(3),
                    stop=-3,
                    num=self.n_lambdas)

        # place the regularization parameters into a list of dictionaries
        reg_params = list()
        for alpha_idx, alpha in enumerate(self.alphas):
            for lamb_idx, lamb in enumerate(self.lambdas[alpha_idx]):
                # reset the regularization parameter
                reg_params.append(dict(alpha=lamb, l1_ratio=alpha))

        return reg_params

    def score_predictions(self, metric, fitter, X, y, support):
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

        X : nd-array
            The design matrix.

        y : nd-array
            The response vector.

        support: array-like
            The indices of the non-zero features.

        Returns
        -------
        score : float
            The score.
        """
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
            n_samples = y.size
            if metric == 'BIC':
                score = utils.BIC(ll, n_features, n_samples)
            elif metric == 'AIC':
                score = utils.AIC(ll, n_features)
            elif metric == 'AICc':
                score = utils.AICc(ll, n_features, n_samples)
            else:
                raise ValueError(metric + ' is not a valid metric.')
            # negate the score since lower information criterion is
            # preferable
            score = -score

        return score

    def _fit_intercept(self, X, y):
        """"Fit a model with an intercept and fixed coefficients.

        This is used to re-fit the intercept after the coefficients are
        estimated.
        """
        if self.fit_intercept:
            mu = np.exp(np.dot(X, self.coef_))
            self.intercept_ = np.log(np.mean(y)/np.mean(mu))
        else:
            self.intercept_ = np.zeros(1)


class Poisson(LinearModel):
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 standardize=False, max_iter=1000, tol=1e-5, warm_start=True):
        """Generalized Linear Model with exponential link function
        (i.e. Poisson) trained with L1/L2 regularizer (i.e. Elastic net
        penalty).

        The log-likelihood of the Poisson GLM is optimized by performing
        coordinate descent on a linearized quadratic approximation. See
        Chapter 5 of Hastie, Tibshirani, and Wainwright (2016) for more
        details.

        Parameters
        ----------
        alpha : float, optional
            Constant that multiplies the L1 term. Defaults to 1.0.

        l1_ratio : float, optional
            float between 0 and 1 acting as a scaling between
            l1 and l2 penalties). For ``l1_ratio = 0`` the penalty is an
            L2 penalty. For ``l1_ratio = 1`` it is an L1 penalty. For ``0
            < l1_ratio < 1``, the penalty is a combination of L1 and L2

        fit_intercept : boolean, default True
            Whether to fit an intercept or not.

        standardize : boolean, optional, default False
            If True, the regressors X will be standardized before regression by
            subtracting the mean and dividing by the l2-norm.

        tol : float, optional
            The tolerance for the optimization: if the updates are
            smaller than ``tol``, the optimization code checks the
            dual gap for optimality and continues until it is smaller
            than ``tol``.

        warm_start : bool, optional
            When set to ``True``, reuse the solution of the previous call to
            fit as initialization, otherwise, just erase the previous solution.

        Attributes
        ----------
        coef_ : array, shape (n_features,)
            The fitted parameter vector.

        intercept_ : float
            The fitted intercept.
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start

    def fit(self, X, y, init=None):
        """Fit the Poisson GLM with coordinate descent.

        Parameters
        ----------
        X : nd-array, shape (n_samples, n_features)
            The design matrix.

        y : nd-array, shape (n_samples,)
            Response vector. Will be cast to X's dtype if necessary.
            Currently, this implementation does not handle multiple response
            variables.

        init : nd-array, shape (n_features)
            Initialization for parameters.
        """
        self.n_samples, self.n_features = X.shape

        # initialization
        if self.warm_start and hasattr(self, 'coef_'):
            coef = self.coef_
        else:
            coef = np.zeros(shape=(self.n_features))

        if init is not None:
            coef = init

        intercept = 0

        # we will handle the intercept by hand: only preprocess the design
        # matrix
        X, _, X_offset, _, X_scale = _preprocess_data(
            X, y, fit_intercept=False, standardize=self.standardize)

        # all features are initially active
        active_idx = np.arange(self.n_features)

        coef_update = np.zeros(coef.shape)
        # perform coordinate descent updates
        for iteration in range(self.max_iter):

            # linearize the log-likelihood
            w, z = self.adjusted_response(X, y, coef, intercept)

            # perform an update of coordinate descent
            coef_update, intercept = self.cd_sweep(
                coef=coef, intercept=intercept, X=X, w=w, z=z,
                active_idx=active_idx)

            # check convergence
            if np.max(np.abs(coef_update - coef)) < self.tol:
                break

            coef = coef_update

            # update the active features
            active_idx = np.argwhere(coef != 0).ravel()

        self.intercept_ = intercept
        self.coef_ = coef_update / X_scale

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
        if hasattr(self, 'coef_') and hasattr(self, 'intercept_'):
            mu = np.exp(self.intercept_ + np.dot(X, self.coef_))
            mode = np.floor(mu)
            return mode
        else:
            raise NotFittedError('Poisson model is not fit.')

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
        if hasattr(self, 'coef_') and hasattr(self, 'intercept_'):
            mu = np.exp(self.intercept_ + np.dot(X, self.coef_))
            return mu
        else:
            raise NotFittedError('Poisson model is not fit.')

    def cd_sweep(self, coef, X, w, z, active_idx, intercept=0):
        """Performs one sweep of coordinate descent updates over a set of
        'active' features.

        Parameters
        ----------
        coef : nd-array, shape (n_features,)
            The current estimates of the parameters.

        X : nd-array, shape (n_samples, n_features)
            The design matrix.

        w : nd-array, shape (n_samples,)
            The weights applied to each sample after linearization of the
            log-likelihood.

        z : nd-array, shape (n_samples,)
            The working response after linearization of the log-likelihood.

        active_idx : nd-array, shape (n_features,)
            An array of ints denoting the indices of features that should be
            updated.

        intercept : float
            The current estimate of the intercept.

        Returns
        -------
        coef_update : nd-array, shape (n_features,)
            The updated parameters after one coordinate descent sweep.
        """
        n_active = active_idx.size
        n_features = coef.size
        coef_update = np.zeros(coef.size)

        # numerator and denominator update terms
        num_update = np.zeros(n_active)
        den_update = np.zeros(n_active)

        # iterate over the active features
        for idx, coordinate in enumerate(active_idx):
            # remove the current feature from the update rules
            mask = np.ones(n_features)
            mask[coordinate] = 0

            z_hat = intercept + np.dot(X, coef * mask)
            num_update[idx] = np.dot(w, X[:, coordinate] * (z - z_hat))
            den_update[idx] = np.dot(w, X[:, coordinate]**2)

        # intercept is not penalized and requires a separate update rule
        if self.fit_intercept:
            z_hat = np.dot(X, coef)
            residuals = z - z_hat
            # update intercept
            intercept_update = np.dot(w, residuals)/np.sum(w)
        else:
            intercept_update = 0

        # see equation 5.44 in hastie, tibshirani, wainwright (2016)
        updates = self.soft_threshold(num_update, self.l1_ratio * self.alpha) \
            / (den_update + self.alpha * (1 - self.l1_ratio))

        coef_update[active_idx] = updates

        return coef_update, intercept_update

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
        z = Xbeta + (y / w) - 1
        return w, z

    def _fit_intercept_no_features(self, y):
        """"Fit a model with only an intercept.

        This is used in cases where the model has no support selected.
        """
        return PoissonInterceptFitterNoFeatures(y)

class PoissonInterceptFitterNoFeatures(object):
    def __init__(self, y):
        self.intercept_ = np.log(y.mean())
        raise NotImplementedError

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
