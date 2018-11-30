import abc as _abc
import six as _six
import numpy as np

from tqdm import trange

from sklearn.linear_model.base import (
    LinearModel, _preprocess_data, SparseCoefMixin)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y

from PyUoI import utils


class AbstractUoILinearModel(_six.with_metaclass(_abc.ABCMeta, LinearModel, SparseCoefMixin))
    """An abstract base class for UoI linear model classes

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

    def __init__(
        self,
        n_boots_sel=48, n_boots_est=48,
        selection_frac=0.9,
        stability_selection=1.,
        estimation_score='r2',
        copy_X=True, fit_intercept=True, normalize=True, random_state=None
    ):
        # data split fractions
        self.selection_frac = selection_frac
        # number of bootstraps
        self.n_boots_sel = n_boots_sel
        self.n_boots_est = n_boots_est
        # other hyperparameters
        self.stability_selection = stability_selection
        self.estimation_score = estimation_score
        # preprocessing
        self.copy_X = copy_X
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.random_state = random_state

        # extract selection thresholds from user provided stability selection
        self.selection_thresholds = self._stability_selection_to_threshold(self.stability_selection, self.n_boots_sel)

        self.n_selection_thresholds = self.selection_thresholds.size

    @_abc.abstractproperty
    def selection_lm(self):
        pass

    @_abc.abstractproperty
    def estimation_lm(self):
        pass

    @_abc.abstractmethod
    def get_reg_params(self):
        pass

    def fit(
        self, X, y, stratify=None, verbose=False
    ):
        """Fit data according to the UoI-Lasso algorithm.

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
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         y_numeric=True, multi_output=True)

        # preprocess data
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X
        )

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
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                train_size=self.selection_frac,
                stratify=stratify,
                random_state=self.random_state
            )

            ## This should be the same as the above code
            # X_rep, y_rep = resample(X, y, replace=False, n_samples=int(self.selection_frac*self.n_samples))
            # X_rep, y_rep = resample(X, y)

            # perform a sweep over the regularization strengths
            selection_coefs[bootstrap, :, :] = self.uoi_selection_sweep(
                X=X_rep, y=y_rep,
                reg_param_values=self.reg_params_
            )

        # perform the intersection step
        self.intersection(selection_coefs)

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
            X_rep, y_rep = resample(X, y)

            # iterate over the regularization parameters
            for rp_idx, lamb in enumerate(self.reg_params_):
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
                self.scores_[bootstrap, lamb_idx] = self.score_predictions(
                    score=self.estimation_score,
                    y_true=y_test,
                    y_pred=y_pred,
                    n_features=np.count_nonzero(support),
                    metric=self.estimation_score,
                )

        self.lambda_max_idx = np.argmax(self.scores_, axis=1)
        # extract the estimates over bootstraps from model with best lambda
        best_estimates = estimates[
            np.arange(self.n_boots_est), self.lambda_max_idx, :
        ]
        # take the median across estimates for the final, bagged estimate
        self.coef_ = np.median(best_estimates, axis=0)

        self._set_intercept(X_offset, y_offset, X_scale)

        return self

    @staticmethod
    def _stability_selection_to_threshold(stability_selection, n_boots):
        """Converts user inputted stability selection to an array of
        thresholds. These thresholds correspond to the number of bootstraps
        that a feature must appear in to guarantee placement in the selection
        profile.

        Parameters
        ----------
        stability_selection : int, float, or array-like
            If int, treated as the number of bootstraps that a feature must
            appear in to guarantee placement in selection profile. If float,
            must be between 0 and 1, and is instead the proportion of
            bootstraps. If array-like, must consist of either ints or floats
            between 0 and 1. In this case, each entry in the array-like object
            will act as a separate threshold for placement in the selection
            profile.
        """

        # single float, indicating proportion of bootstraps
        if isinstance(stability_selection, float):
            selection_thresholds = np.array([int(
                stability_selection * n_boots
            )])

        # single int, indicating number of bootstraps
        elif isinstance(stability_selection, int):
            selection_thresholds = np.array([int(
                stability_selection
            )])

        # list, to be converted into numpy array
        elif isinstance(stability_selection, list):
            # list of floats
            if all(isinstance(idx, float) for idx in stability_selection):
                selection_thresholds = \
                    n_boots * np.array(stability_selection)

            # list of ints
            elif all(isinstance(idx, int) for idx in stability_selection):
                selection_thresholds = np.array(stability_selection)

            else:
                raise ValueError("Stability selection list must consist of "
                                 "floats or ints.")

        # numpy array
        elif isinstance(stability_selection, np.ndarray):
            # np array of floats
            if np.issubdtype(stability_selection.dtype.type, np.floating):
                selection_thresholds = n_boots * stability_selection

            # np array of ints
            elif np.issubdtype(stability_selection.dtype.type, np.integer):
                selection_thresholds = stability_selection

            else:
                raise ValueError("Stability selection array must consist of "
                                 "floats or ints.")

        else:
            raise ValueError("Stability selection must be a valid float, int "
                             "or array.")

        # ensure that ensuing list of selection thresholds satisfies
        # the correct bounds
        selection_thresholds = selection_thresholds.astype('int')
        if not (
            np.all(selection_thresholds <= n_boots) and
            np.all(selection_thresholds > 1)
        ):
            raise ValueError("Stability selection thresholds must be within "
                             "the correct bounds.")

        return selection_thresholds

    @staticmethod
    def intersection(coefs, selection_thresholds):
        """Performs the intersection operation on selection coefficients
        using stability selection criteria.

        Parameters
        ----------
        coefs : np.ndarray, shape (# bootstraps, # lambdas, # features)
            The coefficients obtain from the selection sweep, corresponding to
            each bootstrap and choice of L1 regularization strength.
        """

        n_selection_thresholds = selection_thresholds
        n_reg_params = coefs.shape[1]
        n_features = coefs.shape[2]
        supports = np.zeros(
            (n_selection_thresholds, n_reg_params, n_features),
            dtype=bool
        )

        # iterate over each stability selection threshold
        for thresh_idx, threshold in enumerate(selection_thresholds):
            # calculate the support given the specific selection threshold
            supports[thresh_idx, ...] = \
                np.count_nonzero(coefs, axis=0) >= threshold

        # unravel the dimension corresponding to selection thresholds
        supports = np.squeeze(np.reshape(
            supports,
            (n_selection_thresholds * n_reg_params, n_features)
        ))

        # # TODO: collapse duplicate supports
        return supports

    @staticmethod
    def score_predictions(metric, y_true, y_pred, metric='r2', negate=False, **kwargs):
        """Score, according to some metric, predictions provided by a model.

        the resulting score will be negated if an information criterion is
        specified

        Parameters
        ----------
        metric : str
            The scoring metric to use. Acceptible options are 'AIC', 'AICc', 'BIC', and 'r2'

        y_true : array-like
            The true response variables.

        y_pred : array-like
            The predicted response variables.

        metric : string
            The type of score to run on the prediction. Valid options include
            'r2' (explained variance), 'BIC' (Bayesian information criterion),
            'AIC' (Akaike information criterion), and 'AICc' (corrected AIC).

        negate : bool
            Whether to negate the score. Useful in cases like AIC and BIC,
            where minimum score is preferable.

        Returns
        -------
        score : float
            The score.
        """

        if metric == 'r2':
            score = r2_score(
                y_true=y_true,
                y_pred=y_pred
            )
        elif metric == 'BIC':
            score = utils.BIC(
                y_true=y_true,
                y_pred=y_pred,
                n_features=kwargs.get('n_features')
            )
        elif metric == 'AIC':
            score = utils.AIC(
                y_true=y_true,
                y_pred=y_pred,
                n_features=kwargs.get('n_features')
            )
        elif metric == 'AICc':
            score = utils.AICc(
                y_true=y_true,
                y_pred=y_pred,
                n_features=kwargs.get('n_features')
            )
        else:
            raise ValueError(
                metric + ' is not a valid option.'
            )

        if metric in ('BIC', 'AIC', 'AICc'):
            score = -score

        return score

    def uoi_selection_sweep(self, X, y, reg_param_values):
        """Perform Lasso regression on a dataset over a sweep
        of L1 penalty values.

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

        # apply the Lasso to bootstrapped datasets
        for reg_param_idx, reg_params in enumerate(reg_param_values):
            # reset the regularization parameter
            self.selection_lm.set_params(**reg_params)
            # rerun fit
            self.selection_lm.fit(X, y)
            # store coefficients
            coefs[reg_param_idx, :] = lm.coef_

        return coefs
