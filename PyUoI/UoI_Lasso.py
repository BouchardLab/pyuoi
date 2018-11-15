import numpy as np

from tqdm import trange

import sklearn.linear_model as lm
from sklearn.linear_model.base import (_preprocess_data, _rescale_data,
                                       SparseCoefMixin)
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.metrics import r2_score
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_random_state

from PyUoI import utils


class UoI_Lasso(lm.base.LinearModel, SparseCoefMixin):
    """The UoI-Lasso algorithm.


    See Bouchard et al., NIPS, 2017, for more details on UoI-Lasso and the
    Union of Intersections framework.

    Parameters
    ----------
    n_lambdas : int, default 48
        The number of L1 penalties to sweep across. For each lambda value,
        UoI-Lasso will fit that model over many bootstraps of the data. A
        larger set of L1 penalties will consider a more diverse set of supports
        while increasing compute time.

    n_boots_sel : int, default 48
        The number of data bootstraps to use in the selection module.
        Increasing this number will make selection more strict.

    n_boots_est : int, default 48
        The number of data bootstraps to use in the estimation module.
        Increasing this number will relax selection and decrease variance.

    selection_frac : float, default 0.9
        The fraction of the dataset to use for training in each resampled
        bootstrap, during the selection module. Small values of this parameter
        imply larger "perturbations" to the dataset.

    estimation_frac : float, default 0.9
        The fraction of the dataset to use for training in each resampled
        bootstrap, during the estimation module. The remaining data is used
        to obtain validation scores. Small values of this parameters imply
        larger "perturbations" to the dataset.

    stability_selection : float, or int, or 1-D array-like, default 1
        Enfore

    copy_X : boolean, default True
        If True, X will be copied; else, it may be overwritten.

    fit_intercept : boolean, default True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    random_state : int, RandomState instance or None, default None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    Attributes
    ----------
    coef_ : array, shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.

    intercept_ : float
        Independent term in the linear model.

    supports_ : array, shape (n_fea)
        boolean array indicating whether a given regressor (column) is selected
        for estimation for a given lambda (row).
    """

    def __init__(
        self,
        n_boots_sel=48, n_boots_est=48,
        selection_frac=0.9, estimation_frac=0.9,
        n_lambdas=48, stability_selection=1.,
        estimation_score='BIC',
        copy_X=True, fit_intercept=True, normalize=False, random_state=None
    ):
        # data split fractions
        self.selection_frac = selection_frac
        self.estimation_frac = estimation_frac
        # number of bootstraps
        self.n_boots_sel = n_boots_sel
        self.n_boots_est = n_boots_est
        # other hyperparameters
        self.n_lambdas = n_lambdas
        self.stability_selection = stability_selection
        self.estimation_score = estimation_score
        # preprocessing
        self.copy_X = copy_X
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.random_state = random_state

    def fit(
        self, X, y, groups=None, verbose=False, sample_weight=None,
        option=True
    ):
        """Fit data according to the UoI-Lasso algorithm.
        Relevant information (fits, residuals, model performance) is stored
        within object. Thus, nothing is returned by this function.

        Parameters
        ----------
        X : np array (2d)
            the design matrix, containing the predictors.
            its shape is assumed to be (number of samples, number of features).

        y : np array (1d)
            the vector of dependent variables.
            its length is assumed to be (number of samples,).

        seed : int
            a seed for the random number generator. this number is relevant
            for the choosing bootstraps and dividing the data into training and
            test sets.

        verbose : boolean
            a boolean switch indicating whether the fitting should print out
            its progress.
        """

        # perform checks
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         y_numeric=True, multi_output=True)
        rng = check_random_state(self.random_state)

        # preprocess data
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X
        )

        if sample_weight is not None and np.atleast_1d(sample_weight).ndim > 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        # extract model dimensions
        self.n_samples, self.n_features = X.shape

        # group leveling
        if groups is None:
            self.groups_ = np.ones(self.n_samples_)
        else:
            self.groups_ = np.array(groups)

        ####################
        # Selection Module #
        ####################
        if verbose:
            print('(1) Loaded data.\n %s samples with %s features.'
                  % (self.n_samples, self.n_features))

        self.lambdas = _alpha_grid(
            X=X, y=y,
            l1_ratio=1.0,
            fit_intercept=self.fit_intercept,
            eps=1e-3,
            n_alphas=self.n_lambdas,
            normalize=self.normalize
        )

        # sweep over the grid of regularization strengths
        estimates_selection, _ = \
            self.lasso_sweep(
                X, y, self.lambdas, self.train_frac_sel, self.n_boots_sel,
                desc='fine lasso sweep', verbose=verbose
            )

        # perform the intersection step
        self.intersection(estimates_selection)

        #####################
        # Estimation Module #
        #####################
        # we'll use the supports obtained in the selection module to calculate
        # bagged OLS estimates over bootstraps

        if verbose:
            print('(3) Beginning model estimation, with %s bootstraps.'
                  % self.n_boots_est)

        # set up data arrays
        estimates = np.zeros(
            (self.n_boots_est, self.n_lambdas, self.n_features_),
            dtype=np.float32
        )
        scores = np.zeros((self.n_boots_est, self.n_lambdas), dtype=np.float32)

        # iterate over bootstrap samples
        for bootstrap in trange(
            self.n_boots_est, desc='Model Estimation', disable=not verbose
        ):

            # extract the bootstrap indices, keeping a fraction of the data
            # available for testing
            train_idx, test_idx = utils.leveled_randomized_ids(
                self.groups_, self.train_frac_est
            )

            # iterate over the regularization parameters
            for lamb_idx, lamb in enumerate(self.lambdas):
                # extract current support set
                support = self.supports_[lamb_idx]

                # extract response vectors
                y_train = y[train_idx]
                y_test = y[test_idx]

                # if nothing was selected, we won't bother running OLS
                if np.any(support):
                    # get design matrices
                    X_train = X[train_idx][:, support]
                    X_test = X[test_idx][:, support]

                    # compute ols estimate
                    ols = lm.LinearRegression()
                    ols.fit(X_train, y_train)

                    # store the fitted coefficients
                    estimates[bootstrap, lamb_idx, support] = ols.coef_

                    # obtain predictions for scoring
                    y_pred = ols.predict(X_test)
                else:
                    # no prediction since nothing was selected
                    y_pred = np.zeros(y_test.size)

                # calculate estimation score
                if self.estimation_score == 'r2':
                    scores[bootstrap, lamb_idx] = r2_score(
                        y_true=y_test,
                        y_pred=y_pred
                    )
                elif self.estimation_score == 'BIC':
                    y_pred = ols.predict(X_test)
                    n_features = np.count_nonzero(support)
                    scores[bootstrap, lamb_idx] = -utils.BIC(
                        y_true=y_test,
                        y_pred=y_pred,
                        n_features=n_features
                    )
                elif self.estimation_score == 'AIC':
                    y_pred = ols.predict(X_test)
                    n_features = np.count_nonzero(support)
                    scores[bootstrap, lamb_idx] = -utils.AIC(
                        y_true=y_test,
                        y_pred=y_pred,
                        n_features=n_features
                    )
                elif self.estimation_score == 'AICc':
                    y_pred = ols.predict(X_test)
                    n_features = np.count_nonzero(support)
                    scores[bootstrap, lamb_idx] = -utils.AICc(
                        y_true=y_test,
                        y_pred=y_pred,
                        n_features=n_features
                    )
                else:
                    raise ValueError(
                        str(self.estimation_score) + ' is not a valid option.'
                    )

        self.lambda_max_idx = np.argmax(scores, axis=1)
        # extract the estimates over bootstraps from model with best lambda
        best_estimates = estimates[
            np.arange(self.n_boots_est), self.lambda_max_idx, :
        ]
        # take the median across estimates for the final, bagged estimate
        self.coef_ = np.median(best_estimates, axis=0)

        if verbose:
            print("UoI Lasso complete.")

        self._set_intercept(X_offset, y_offset, X_scale)

        return self

    def intersection(self, estimates):
        # create support matrix
        self.supports_ = np.zeros(
            (self.n_selection_thres, self.n_lambdas, self.n_features_),
            dtype=bool
        )

        # choose selection fraction threshold values to use
        selection_frac_thresholds = np.linspace(
            self.selection_thres_min,
            self.selection_thres_max,
            self.n_selection_thres
        )
        # calculate the actual number of thresholds, but delete any repetitions
        selection_thresholds = np.sort(np.unique(
            (self.n_boots_sel * selection_frac_thresholds).astype('int'))
        )

        # iterate over each stability selection threshold
        for thres_idx, threshold in enumerate(selection_thresholds):
            # calculate the support given the specific selection threshold
            self.supports_[thres_idx, :] = \
                np.count_nonzero(estimates, axis=0) >= threshold

        self.supports_ = np.reshape(
            self.supports_,
            (self.n_selection_thres * self.n_lambdas, self.n_features_)
        )

        return self.supports_

    def score(self, X, y, metric='r2'):
        # make predictions
        if self.fit_intercept:
            y_hat = np.dot(X, self.coef_) + self.intercept_
        else:
            y_hat = np.dot(X, self.coef_)

        if metric == 'r2':
            return r2_score(y, y_hat)
        elif metric == 'BIC':
            if self.fit_intercept:
                n_features = np.count_nonzero(self.coef_) + 1
            else:
                n_features = np.count_nonzero(self.coef_)

            rss = np.sum((y - y_hat)**2)
            return utils.BIC(
                n_features=n_features,
                n_samples=y.shape[0],
                rss=rss
            )
        else:
            raise ValueError('Incorrect metric specified.')

    def lasso_sweep(
        self, X, y, lambdas, train_frac, n_bootstraps, desc='', verbose=False
    ):
        """Perform Lasso regression across bootstraps of a dataset for a sweep
        of L1 penalty values.

        Parameters
        ----------
        X : np.array
            data array containing regressors; assumed to be 2-d array with
            shape n_samples x n_features

        y : np.array
            data array containing dependent variable; assumed to be a 1-d array
            with length n_samples

        lambdas : np.array
            the set of regularization parameters to run boostraps over

        train_frac : float
            float between 0 and 1; the fraction of data to use for training

        n_bootstraps : int
            the number of bootstraps to obtain from the dataset; each bootstrap
            will undergo a Lasso regression

        Returns
        -------
        estimates : np.array
            predicted regressors for each bootstrap and lambda value; shape is
            (n_bootstraps, n_lambdas, n_features)

        scores : np.array
            scores by the model for each bootstrap and lambda
            value; shape is (n_bootstraps, n_lambdas)
        """

        n_lambdas = len(lambdas)
        n_samples, n_features = X.shape
        # create arrays to collect results
        estimates = np.zeros(
            (n_bootstraps, n_lambdas, n_features),
            dtype=np.float32
        )
        scores = np.zeros((n_bootstraps, n_lambdas), dtype=np.float32)
        # apply the Lasso to bootstrapped datasets
        for bootstrap in trange(n_bootstraps, desc=desc, disable=not verbose):
            # for each bootstrap, we'll split the data into a randomly assigned
            # training and test set
            train, test = utils.leveled_randomized_ids(
                self.groups_,
                train_frac
            )
            # iterate over the provided L1 penalty values
            for lamb_idx, lamb in enumerate(lambdas):
                # run the Lasso on the training set
                lasso = lm.Lasso(
                    alpha=lamb,
                    max_iter=10000,
                    fit_intercept=self.fit_intercept
                )
                lasso.fit(
                    X[train], y[train] - y[train].mean()
                )
                estimates[bootstrap, lamb_idx, :] = lasso.coef_

                # run trained Lasso on the test set and obtain predictions
                y_hat = X[test].dot(estimates[bootstrap, lamb_idx, :])
                y_true = y[test] - y[test].mean()
                # calculate the explained variance using the predicted values
                scores[bootstrap, lamb_idx] = r2_score(y_true, y_hat)

        return estimates, scores
