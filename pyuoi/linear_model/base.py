import abc as _abc
import six as _six
import numpy as np

from tqdm import trange

from sklearn.linear_model.base import (
    LinearModel, _preprocess_data, SparseCoefMixin)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y

from pyuoi import utils

from .utils import stability_selection_to_threshold, intersection


class AbstractUoILinearModel(_six.with_metaclass(_abc.ABCMeta, LinearModel, SparseCoefMixin)):
    """An abstract base class for UoI linear model classes

    See Bouchard et al., NIPS, 2017, for more details on the Union of Intersections framework.

    Parameters
    ----------
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
        larger "perturbations" to the dataset. IGNORED - Leaving this here
        to double check later

    stability_selection : int, float, or array-like, default 1
        If int, treated as the number of bootstraps that a feature must
        appear in to guarantee placement in selection profile. If float,
        must be between 0 and 1, and is instead the proportion of
        bootstraps. If array-like, must consist of either ints or floats
        between 0 and 1. In this case, each entry in the array-like object
        will act as a separate threshold for placement in the selection
        profile.

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
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.

    intercept_ : float
        Independent term in the linear model.

    supports_ : array, shape
        boolean array indicating whether a given regressor (column) is selected
        for estimation for a given lambda (row).
    """

    def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9, stability_selection=1., random_state=None):
        # data split fractions
        self.selection_frac = selection_frac
        # number of bootstraps
        self.n_boots_sel = n_boots_sel
        self.n_boots_est = n_boots_est
        # other hyperparameters
        self.stability_selection = stability_selection
        # preprocessing
        self.random_state = random_state

        # extract selection thresholds from user provided stability selection
        self.selection_thresholds = stability_selection_to_threshold(self.stability_selection, self.n_boots_sel)

        self.n_selection_thresholds = self.selection_thresholds.size

    @_abc.abstractproperty
    def selection_lm(self):
        pass

    @_abc.abstractproperty
    def estimation_lm(self):
        pass

    @_abc.abstractproperty
    def estimation_score(self):
        pass

    @_abc.abstractmethod
    def get_reg_params(self):
        pass

    @_abc.abstractstaticmethod
    def score_predictions(metric, y_true, y_pred, supports):
        pass

    @_abc.abstractmethod
    def preprocess_data(self, X, y):
        """

        """
        pass

    def fit(self, X, y, stratify=None, verbose=False):
        """Fit data according to the UoI algorithm.

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

        verbose : boolean
            A switch indicating whether the fitting should print out messages
            displaying progress. Utilizes tqdm to indicate progress on
            bootstraps.
        """

        # extract model dimensions
        self.n_samples, self.n_features = X.shape

        ####################
        # Selection Module #
        ####################
        # choose the regularization parameters for selection sweep
        self.reg_params_ = self.get_reg_params(X, y)
        self.n_reg_params_ = len(self.reg_params_)

        # initialize selection
        selection_coefs = np.zeros(
            (self.n_boots_sel, self.n_reg_params_, self.n_features),
            dtype=np.float32
        )

        # iterate over bootstraps
        for bootstrap in trange(
            self.n_boots_sel, desc='Model Selection', disable=not verbose
        ):
            # draw a resampled bootstrap
            X_rep, X_test, y_rep, y_test = train_test_split(
                X, y,
                train_size=self.selection_frac,
                stratify=stratify,
                random_state=self.random_state
            )

            # TODO: consider using this way intsead, since
            # this is the real way to do a bootstrap
            # X_rep, y_rep = resample(X, y)

            ## This should be the same as the above call to train_test_split
            # X_rep, y_rep = resample(X, y, replace=False, n_samples=int(self.selection_frac*self.n_samples))

            # perform a sweep over the regularization strengths
            selection_coefs[bootstrap, :, :] = self.uoi_selection_sweep(
                X=X_rep, y=y_rep,
                reg_param_values=self.reg_params_
            )

        # perform the intersection step
        self.supports_ = intersection(selection_coefs, self.selection_thresholds)

        #####################
        # Estimation Module #
        #####################
        # set up data arrays
        estimates = np.zeros(               # coef_ for each bootstrap for each support
            (
                self.n_boots_est,
                self.n_selection_thresholds * self.n_reg_params_,
                self.n_features
            ),
            dtype=np.float32
        )
        self.scores_ = np.zeros(            # score (r2/AIC/AICc/BIC) for each bootstrap for each support
            (
                self.n_boots_est,
                self.n_selection_thresholds * self.n_reg_params_
            ),
            dtype=np.float32
        )

        # iterate over bootstrap samples
        for bootstrap in trange(
            self.n_boots_est, desc='Model Estimation', disable=not verbose
        ):

            # draw a resampled bootstrap
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                train_size=self.selection_frac,
                stratify=stratify,
                random_state=self.random_state
            )

            # iterate over the regularization parameters
            for rp_idx, reg_param in enumerate(self.reg_params_):
                # extract current support set
                support = self.supports_[rp_idx]
                # if nothing was selected, we won't bother running OLS
                if np.any(support):
                    # compute ols estimate
                    self.estimation_lm.fit(
                        X_train[:, support],
                        y_train
                    )

                    # store the fitted coefficients
                    estimates[bootstrap, rp_idx, support] = self.estimation_lm.coef_.ravel()

                    # obtain predictions for scoring
                    y_pred = self.estimation_lm.predict(X_test[:, support])
                else:
                    # no prediction since nothing was selected
                    y_pred = np.zeros(y_test.size)


                # calculate estimation score
                self.scores_[bootstrap, rp_idx] = self.score_predictions(self.estimation_score, y_test, y_pred, support)

        self.rp_max_idx_ = np.argmax(self.scores_, axis=1)
        # extract the estimates over bootstraps from model with best lambda
        best_estimates = estimates[
            np.arange(self.n_boots_est), self.rp_max_idx_, :
        ]
        # take the median across estimates for the final, bagged estimate
        self.coef_ = np.median(best_estimates, axis=0)

        return self

    def uoi_selection_sweep(self, X, y, reg_param_values):
        """Perform model training on a dataset over a sweep of regularization
        parameter values.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            The design matrix.

        y : ndarray, shape (n_samples,)
            Response vector.

        reg_param_values: list of dicts
            A list of dictionaries containing the regularization parameter values
            to iterate over.

        Returns
        -------
        coefs : nd.array, shape (n_param_values, n_features)
            Predicted parameter values for each regularization strength.
        """

        n_param_values = len(reg_param_values)
        n_features = X.shape[1]

        coefs = np.zeros(
            (n_param_values, n_features),
            dtype=np.float32
        )

        # initialize Linear model fit object
        params = dict()

        # apply the linear model to bootstrapped datasets
        for reg_param_idx, reg_params in enumerate(reg_param_values):
            # reset the regularization parameter
            self.selection_lm.set_params(**reg_params)
            # rerun fit
            self.selection_lm.fit(X, y)
            # store coefficients
            coefs[reg_param_idx, :] = self.selection_lm.coef_

        return coefs


class AbstractUoILinearRegressor(_six.with_metaclass(_abc.ABCMeta, AbstractUoILinearModel)):

    __valid_estimation_metrics = ('r2', 'AIC', 'AICc', 'BIC')

    def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
        stability_selection=1., warm_start=True,
        estimation_score='r2',
        copy_X=True, fit_intercept=True, normalize=True, random_state=None, max_iter=1000
    ):
        super(AbstractUoILinearRegressor, self).__init__(
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            stability_selection=stability_selection,
        )
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X

        if estimation_score not in self.__valid_estimation_metrics:
            raise ValueError("invalid estimation metric: '%s'" % estimation_score)

        self.__estimation_score = estimation_score

    def preprocess_data(self, X, y):
        return _preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X
        )

    @property
    def estimation_score(self):
        return self.__estimation_score

    @staticmethod
    def score_predictions(metric, y_true, y_pred, supports):
        """Score, according to some metric, predictions provided by a model.

        the resulting score will be negated if an information criterion is
        specified

        Parameters
        ----------
        metric : string
            The type of score to run on the prediction. Valid options include
            'r2' (explained variance), 'BIC' (Bayesian information criterion),
            'AIC' (Akaike information criterion), and 'AICc' (corrected AIC).

        y_true : array-like
            The true response variables.

        y_pred : array-like
            The predicted response variables.

        supports: array-like
            The value of the supports for the model that was used to generate *y_pred*

        Returns
        -------
        score : float
            The score.
        """
        if metric == 'r2':
            score = r2_score(y_true, y_pred)
        else:
            n_features=np.count_nonzero(supports)
            if metric == 'BIC':
                score = utils.BIC(y_true, y_pred, n_features)
            elif metric == 'AIC':
                score = utils.AIC(y_true, y_pred, n_features)
            elif metric == 'AICc':
                score = utils.AICc(y_true, y_pred, n_features)
            else:
                raise ValueError(metric + ' is not a valid option.')
            score = -score
        return score

    def fit(self, X, y, stratify=None, verbose=False):
        """Fit data according to the UoI algorithm.

        Additionaly, perform X-y checks, data preprocessing, and setting interecept

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

        verbose : boolean
            A switch indicating whether the fitting should print out messages
            displaying progress. Utilizes tqdm to indicate progress on
            bootstraps.
        """
        # perform checks
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], y_numeric=True, multi_output=True)
        # preprocess data
        X, y, X_offset, y_offset, X_scale = self.preprocess_data(X, y)
        super(AbstractUoILinearRegressor, self).fit(X, y, stratify=stratify, verbose=verbose)
        self._set_intercept(X_offset, y_offset, X_scale)
        return self


class AbstractUoILinearClassifier(_six.with_metaclass(_abc.ABCMeta, AbstractUoILinearModel)):

    __valid_estimation_metrics = ('acc',)

    def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
        stability_selection=1., warm_start=True,
        estimation_score='acc',
        multi_class='ovr',
        copy_X=True, fit_intercept=True, normalize=True, random_state=None, max_iter=1000
    ):
        super(AbstractUoILinearClassifier, self).__init__(
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            stability_selection=stability_selection,
        )
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X

        if estimation_score not in self.__valid_estimation_metrics:
            raise ValueError("invalid estimation metric: '%s'" % estimation_score)
        self.__estimation_score = estimation_score

    @staticmethod
    def preprocess_data(self, X, y):
        return _preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X
        )

    @property
    def estimation_score(self):
        return self.__estimation_score

