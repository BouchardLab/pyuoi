import abc as _abc
import six as _six
import numpy as np
import math

from tqdm import trange

from sklearn.linear_model.base import (
    LinearModel, _preprocess_data, SparseCoefMixin)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y

from pyuoi import utils
from pyuoi.mpi_utils import get_chunk_size, get_buffer_mask

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
        for estimation for a given regularization parameter value (row).
    """

    def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9, stability_selection=1., random_state=None, comm=None):
        # data split fractions
        self.selection_frac = selection_frac
        # number of bootstraps
        self.n_boots_sel = n_boots_sel
        self.n_boots_est = n_boots_est
        # other hyperparameters
        self.stability_selection = stability_selection
        # preprocessing
        self.comm = comm
        if isinstance(random_state, int):
            # make sure ranks use different seed
            if self.comm is not None:
                random_state += self.comm.rank
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

        # extract selection thresholds from user provided stability selection
        self.selection_thresholds_ = stability_selection_to_threshold(self.stability_selection, self.n_boots_sel)

        self.n_supports_ = None

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
    def intersect(self, coef, thresholds):
        """Intersect coefficients accross all thresholds"""
        pass

    @_abc.abstractmethod
    def preprocess_data(self, X, y):
        """

        """
        pass

    @_abc.abstractmethod
    def get_n_coef(self, X, y):
        """"Return the number of coefficients that will be estimated

        This should return the total number of coefficients estimated,
        accounting for all coefficients for multi-target regression or
        multi-class classification.
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
        n_samples, n_coef = self.get_n_coef(X, y)
        n_features = X.shape[1]

        ####################
        # Selection Module #
        ####################
        # choose the regularization parameters for selection sweep
        self.reg_params_ = self.get_reg_params(X, y)
        self.n_reg_params_ = len(self.reg_params_)

        rank = 0
        size = 1
        if self.comm is not None:
            rank = self.comm.rank
            size = self.comm.size
        chunk_size, buf_len = get_chunk_size(rank, size, self.n_boots_sel)

        # initialize selection
        selection_coefs = np.zeros((buf_len, self.n_reg_params_, n_coef), dtype=np.float32)

        # iterate over bootstraps
        for bootstrap in range(chunk_size):
            # draw a resampled bootstrap
            X_rep, X_test, y_rep, y_test = train_test_split(
                X, y,
                train_size=self.selection_frac,
                stratify=stratify,
                random_state=self.random_state
            )

            for reg_param_idx, reg_params in enumerate(self.reg_params_):
                # reset the regularization parameter
                self.selection_lm.set_params(**reg_params)
                # rerun fit
                self.selection_lm.fit(X_rep, y_rep)
                # store coefficients
                selection_coefs[bootstrap, reg_param_idx, :] = self.selection_lm.coef_.ravel()


        ## if distributed, gather selection coefficients to 0,
        ## perform intersection, and broadcast results
        if self.comm is not None:
            self.comm.Barrier()
            recv = None
            if rank == 0:
                recv = np.zeros((buf_len*size, self.n_reg_params_, n_coef), dtype=np.float32)
            self.comm.Gather(selection_coefs, recv, root=0)
            supports = None
            shape = None
            if rank == 0:
                mask = get_buffer_mask(size, self.n_boots_sel)
                recv = recv[mask]
                supports = self.intersect(recv, self.selection_thresholds_)
                shape = supports.shape
            shape = self.comm.bcast(shape, root=0)
            if rank != 0:
                supports = np.zeros(shape, dtype=np.float32)
            supports = self.comm.bcast(supports, root = 0)
            self.supports_ = supports
        else:
            self.supports_ = self.intersect(selection_coefs, self.selection_thresholds_)

        self.n_supports_ = self.supports_.shape[0]

        #####################
        # Estimation Module #
        #####################
        # set up data arrays

        chunk_size, buf_len = get_chunk_size(rank, size, self.n_boots_est)

        # coef_ for each bootstrap for each support
        self.estimates_ = np.zeros((buf_len, self.n_supports_, n_coef), dtype=np.float32)

        # score (r2/AIC/AICc/BIC) for each bootstrap for each support
        self.scores_ = np.zeros((buf_len, self.n_supports_),dtype=np.float32)

        n_tile = n_coef//n_features
        # iterate over bootstrap samples
        for bootstrap in range(chunk_size):

            # draw a resampled bootstrap
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                train_size=self.selection_frac,
                stratify=stratify,
                random_state=self.random_state
            )

            # iterate over the regularization parameters
            for supp_idx, support in enumerate(self.supports_):
                # extract current support set
                # if nothing was selected, we won't bother running OLS
                if np.any(support):
                    # compute ols estimate
                    self.estimation_lm.fit(X_train[:, support], y_train)

                    # store the fitted coefficients
                    self.estimates_[bootstrap, supp_idx, np.tile(support, n_tile)] = self.estimation_lm.coef_.ravel()

                    # obtain predictions for scoring
                    y_pred = self.estimation_lm.predict(X_test[:, support])
                else:
                    # no prediction since nothing was selected
                    y_pred = np.zeros(y_test.size)


                # calculate estimation score
                self.scores_[bootstrap, supp_idx] = self.score_predictions(self.estimation_score, y_test, y_pred, support)

        if self.comm is not None:
            self.comm.Barrier()
            est_recv = None
            scores_recv = None
            self.rp_max_idx_ = None
            best_estimates = None
            if rank == 0:
                est_recv = np.zeros((buf_len*size, self.n_supports_, n_coef), dtype=np.float32)
                scores_recv = np.zeros((buf_len*size, self.n_supports_),dtype=np.float32)
            self.comm.Gather(self.scores_, scores_recv, root=0)
            self.comm.Gather(self.estimates_, est_recv, root=0)
            if rank == 0:
                mask = get_buffer_mask(size, self.n_boots_est)
                self.estimates_ = est_recv[mask]
                self.scores_ = scores_recv[mask]
                self.rp_max_idx_ = np.argmax(self.scores_, axis=1)
                best_estimates = self.estimates_[np.arange(self.n_boots_est), self.rp_max_idx_, :]
            self.rp_max_idx_ = self.comm.bcast(self.rp_max_idx_, root=0)
            best_estimates = self.comm.bcast(best_estimates, root=0)
        else:
            self.rp_max_idx_ = np.argmax(self.scores_, axis=1)
            # extract the estimates over bootstraps from model with best regularization parameter value
            best_estimates = self.estimates_[np.arange(self.n_boots_est), self.rp_max_idx_, :]

        # take the median across estimates for the final, bagged estimate
        self.coef_ = np.median(best_estimates, axis=0).reshape(n_tile, n_features)

        return self

    def uoi_selection_sweep(self, X, y, reg_param_values):
        """Perform selection regression on a dataset over a sweep of regularization
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

        # apply the selection regression to bootstrapped datasets
        for reg_param_idx, reg_params in enumerate(reg_param_values):
            # reset the regularization parameter
            self.selection_lm.set_params(**reg_params)
            # rerun fit
            self.selection_lm.fit(X, y)
            # store coefficients
            coefs[reg_param_idx, :] = self.selection_lm.coef_.ravel()

        return coefs


class AbstractUoILinearRegressor(_six.with_metaclass(_abc.ABCMeta, AbstractUoILinearModel)):

    __valid_estimation_metrics = ('r2', 'AIC', 'AICc', 'BIC')

    def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
        stability_selection=1., warm_start=True,
        estimation_score='r2',
        copy_X=True, fit_intercept=True, normalize=True, random_state=None, max_iter=1000,
        comm=None
    ):
        super(AbstractUoILinearRegressor, self).__init__(
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            stability_selection=stability_selection,
            random_state=random_state,
            comm=comm,
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

    def get_n_coef(self, X, y):
        """"Return the number of coefficients that will be estimated

        This should return the shape of X.
        """
        return X.shape

    def intersect(self, coef, thresholds):
        """Intersect coefficients accross all thresholds"""
        return intersection(coef, thresholds)

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
        self.coef_ = self.coef_.squeeze()
        return self


class AbstractUoILinearClassifier(_six.with_metaclass(_abc.ABCMeta, AbstractUoILinearModel)):

    __valid_estimation_metrics = ('acc',)

    def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
        stability_selection=1., warm_start=True,
        estimation_score='acc',
        multi_class='ovr',
        copy_X=True, fit_intercept=True, normalize=True, random_state=None, max_iter=1000,
        comm=None
    ):
        super(AbstractUoILinearClassifier, self).__init__(
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            stability_selection=stability_selection,
            random_state=random_state,
            comm=comm,
        )
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X

        if estimation_score not in self.__valid_estimation_metrics:
            raise ValueError("invalid estimation metric: '%s'" % estimation_score)
        self.__estimation_score = estimation_score

    def get_n_coef(self, X, y):
        """"Return the number of coefficients that will be estimated

        This should return the shape of X if doing binary classification,
        else return (X.shape[0], X.shape[1]*n_classes).
        """
        n_samples, n_coef = X.shape
        self._n_classes = len(np.unique(y))
        if self._n_classes > 2:
            n_coef = n_coef * self._n_classes
        return n_samples, n_coef

    def intersect(self, coef, thresholds):
        """Intersect coefficients accross all thresholds

        This implementation will account for multi-class classification.
        """
        supports = intersection(coef, thresholds)
        ret = supports
        if self._n_classes > 2:
            # for each support, figure out which variables
            # are used
            ret = list()
            n_coef = supports[0].shape[0]//self._n_classes
            shape = (self._n_classes, n_coef)
            for supp in supports:
                ret.append(np.logical_or(*supp.reshape(shape)))
            uniq = set()
            for sup in ret:
                uniq.add(tuple(sup))
            ret = np.array(list(uniq))
        return ret

    @staticmethod
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
        if metric == 'acc':
            score = r2_score(y_true, y_pred)
        else:
            raise ValueError(metric + ' is not a valid option.')
        return score

