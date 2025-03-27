import abc as _abc
import numpy as np
import logging
from sklearn.linear_model._base import SparseCoefMixin
from sklearn.metrics import r2_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y
from sklearn.preprocessing import StandardScaler

from scipy.sparse import issparse, csr_matrix

from pyuoi import utils
from pyuoi.mpi_utils import (Gatherv_rows, Bcast_from_root)

from .utils import stability_selection_to_threshold, intersection
from ..utils import check_logger
import gc

def vectorization_bootstrap(raw_data, sample_idx, lag):
    # vectorize the VAR bootstrap data for use with LASSO algorithm
    # sample_idx: idx for sampling the response vectors of the VAR model
    # data: raw time series data in np array with shape n_samples X n_features
    # Construction strategy: first form X_boot, Y_boot, then vectorize both
        
    # flipup so the last time sample in data is now first row(no need anymore)
    data = deepcopy(raw_data)
    data = np.flipud(data)
    n_samples = data.shape[0]
    n_features = data.shape[1]

    # shape of Y: (n_boot_sample) X (n_features) 
    Y = data[sample_idx]   # sample_idx are in range: (0, n_samples-lag)
    Y = Y.T.flatten()    
    
    # X.shape: (n_boot_sample) X (lag * n_features); 
    X_row = [np.hstack(data[i+1 : lag+(i+1)]) for i in sample_idx]
    X = np.vstack(X_row)
    # use kronecker product for vectorization of matrix multiplication
    X = np.kron(np.eye(n_features), X)    

    X, Y = check_X_y(X, Y, accept_sparse=['csr', 'csc', 'coo'],
                 y_numeric=True, multi_output=True)
    return X, Y


def vectorization(raw_data, lag):
    # vectorize the ful raw VAR data for use with LASSO algorithm
    # data: time series data in np array with shape n_samples X n_features
        
    # flipup so the last time sample in data is now first row
    data = deepcopy(raw_data)
    data = np.flipud(data)
    n_samples = data.shape[0]
    n_features = data.shape[1]

    # shape of Y: (T - D) X (n_features)   *T - D: total number of samples - lag
    Y = data[: n_samples-lag]
    # ones = np.ones((n_samples-lag,1))  # for VAR porocess with intercept
    Y = Y.T.flatten()    
    
    # X.shape: (n_samples - lag) X (lag * n_features); 
    X_row = [np.hstack(data[i : lag+i]) for i in range(1,n_samples-lag+1)]
    X = np.vstack(X_row)
    # use kronecker product for vectorization of matrix multiplication
    X = np.kron(np.eye(n_features), X)    
    
    X, Y = check_X_y(X, Y, accept_sparse=['csr', 'csc', 'coo'],
                 y_numeric=True, multi_output=True)    
    return X, Y

class AbstractUoILinearModel(SparseCoefMixin, metaclass=_abc.ABCMeta):
    r"""An abstract base class for UoI ``linear_model`` classes.

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
    stability_selection : int, float, or array-like
        If int, treated as the number of bootstraps that a feature must appear
        in to guarantee placement in selection profile. If float, must be
        between 0 and 1, and is instead the proportion of bootstraps. If
        array-like, must consist of either ints or floats between 0 and 1.
        In this case, each entry in the array-like object will act as a
        separate threshold for placement in the selection profile.
    fit_intercept : bool
        Whether to calculate the intercept for this model. If set to False,
        no intercept will be used in calculations (e.g. data is expected to be
        already centered).
    standardize : bool
        If True, the regressors X will be standardized before regression by
        subtracting the mean and dividing by their standard deviations.
    shared_support : bool
        For models with more than one output (multinomial logistic regression)
        this determines whether all outputs share the same support or can
        have independent supports.
    max_iter : int
        Maximum number of iterations for iterative fitting methods.
    tol : float
        Stopping criteria for solver.
    random_state : int, RandomState instance, or None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by ``np.random``.
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
    supports_ : array, shape
        Boolean array indicating whether a given regressor (column) is selected
        for estimation for a given regularization parameter value (row).
    """

    def __init__(self, n_real_features = 1, fit_VAR = False, n_boots_sel=12, n_boots_est=12, selection_frac=0.9,
                 estimation_frac=0.9, stability_selection=0.75,
                 fit_intercept=True, standardize=True,
                 shared_support=True, max_iter=None, tol=None,
                 random_state=None, comm=None, logger=None):
        # data split fractions
        self.n_real_features = n_real_features  #n_real_features = 1 ==> not fitting a VAR model
        self.fit_VAR = fit_VAR        
        self.selection_frac = selection_frac
        self.estimation_frac = estimation_frac
        # number of bootstraps
        self.n_boots_sel = n_boots_sel
        self.n_boots_est = n_boots_est
        # other hyperparameters
        self.stability_selection = stability_selection
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.shared_support = shared_support
        self.max_iter = max_iter
        self.tol = tol
        self.comm = comm
        self.output_dim = 1  # by vectorization construction
        # preprocessing
        if isinstance(random_state, int):
            # make sure ranks use different seed
            if self.comm is not None:
                random_state += self.comm.rank
            self.random_state = np.random.RandomState(random_state)
        else:
            if random_state is None:
                self.random_state = np.random.RandomState()
            else:
                self.random_state = random_state

        # extract selection thresholds from user provided stability selection
        self.selection_thresholds_ = stability_selection_to_threshold(
            self.stability_selection, self.n_boots_sel)

        self.n_supports_ = None

        self._logger = check_logger(logger, 'uoi_linear_model', self.comm)

    @_abc.abstractproperty
    def estimation_score(self):
        pass

    @_abc.abstractmethod
    def get_reg_params(self):
        pass

    @_abc.abstractstaticmethod
    def _score_predictions(self, metric, fitter, X, y, supports, boot_idxs):
        pass

    @_abc.abstractmethod
    def intersect(self, coef, thresholds):
        """Intersect coefficients across all thresholds."""
        pass

    def _pre_fit_VAR(self, X):
        """Perform z-score scaling for VAR raw data for fit()."""
        if self.standardize:
            if self.fit_intercept and issparse(X):
                msg = ("Cannot center sparse matrices: "
                       "pass `fit_intercept=False`")
                raise ValueError(msg)
            self._X_scaler = StandardScaler(with_mean=self.fit_intercept)
            X = self._X_scaler.fit_transform(X)
       
        return X
        
    def _post_fit_VAR(self, lag, n_features):
        """Perform VAR model coefficiant rescaling if the raw data is z-score scaled."""
        if self.standardize:
            # construct the lagged connectivity matrices from the vectorized model coefficient
            A_model = np.array([self.coef_.reshape(n_features,n_features*lag).T[i*n_features:(i+1)*n_features].T for i in range(lag)])
            sX = self._X_scaler
            Sigma = np.outer(sX.scale_, 1/sX.scale_)
            A_model *= Sigma

            self.VAR_coef_ = A_model

            # adjustment for mean shifts from the data z-score scaling
            if self.fit_intercept:
                adjustment = (np.eye(n_features)-np.sum(A_model, axis = 0)) @ sX.mean_

                # check whether VAR model intercept is scalar or vector
                if len(self.intercept_) == n_features:  
                    self.intercept_ += adjustment
                elif len(self.intercept_) == 1:
                    self.intercept_ += np.mean(adjustment)
                        

    def _pre_fit(self, X, y):
        """Perform class-specific setup for fit()."""
        if self.standardize:
            if self.fit_intercept and issparse(X):
                msg = ("Cannot center sparse matrices: "
                       "pass `fit_intercept=False`")
                raise ValueError(msg)
            self._X_scaler = StandardScaler(with_mean=self.fit_intercept)
            X = self._X_scaler.fit_transform(X)
        if y.ndim == 2:
            self.output_dim = y.shape[1]
        else:
            self.output_dim = 1
        return X, y

    def _post_fit(self, X, y):
        """Perform class-specific cleanup for fit()."""
        if self.standardize:
            sX = self._X_scaler
            self.coef_ /= sX.scale_[np.newaxis]
            

    @_abc.abstractmethod
    def _fit_intercept(self, X, y):
        """Fit a model with an intercept and fixed coefficients.

        This is used to re-fit the intercept after the coefficients are
        estimated.
        """
        pass

    @_abc.abstractmethod
    def _fit_intercept_no_features(self, y):
        """Fit a model with only an intercept.

        This is used in cases where the model has no support selected.
        """
        pass

    def fit(self, lag, data = None, stratify=None, verbose=False):
        """Fit data according to the UoI algorithm.

        Parameters
        ----------
        data : ndarray or scipy.sparse matrix, (n_boot_samples * (lag+1), n_features)
            The boostrap design matrix.
        lag : integer
            Lag of the VAR model.
        stratify : array-like or None
            Ensures groups of samples are alloted to training/test sets
            proportionally. Labels for each group must be an int greater
            than zero. Must be of size equal to the number of samples, with
            further restrictions on the number of groups.
        verbose : bool
            A switch indicating whether the fitting should print out messages
            displaying progress.
        """
        if verbose:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.WARNING)
            
        rank = 0
        size = 1
        if self.comm is not None:
            rank = self.comm.rank
            size = self.comm.size



        # scaling the entrie vectorized data and picking the regularization parameters
        
        
        if rank == 0:
            data = self._pre_fit_VAR(data)

            # only used for getting regularization parameters
            X_all, y_all = vectorization(data, lag)      
            # choose the regularization parameters for selection sweep                
            reg_params_ = self.get_reg_params(X_all, y_all)

            # clear memory
            del X_all
            del y_all 
            gc.collect()
            
        else:
            reg_params_ = None

        if size > 1:
            # Broadcast pre-fitted raw data to all ranks
            data = self.comm.bcast(data, root=0)            
            reg_params_ = self.comm.bcast(reg_params_, root=0) 

        self.reg_params_ = reg_params_
        self.n_reg_params_ = len(self.reg_params_)


        # extract model dimensions(this specifically is for the vectorized VAR model)
        n_coef = lag * data.shape[1]**2
        n_features = n_coef
    
        # check if the response variable is constant
        if np.unique(data).size == 1:
            # self.coef_ = np.zeros((self.output_dim, n_features))
            # self._fit_intercept(X_all, y_all)
            # self._post_fit(X_all, y_all)
            return self


        ####################
        # Selection Module #
        ####################

        # initialize selection
        
        # assigning tasks to each process(from total of n_boot_sel*n_reg_param tasks)

        if size > self.n_boots_sel: 
            # more process than n_boots_sel, so divvy the n_reg_param task up also
            # each task is for a single boot and a single reg_parameter, so the num_tasks here will be Xn_reg_params more than the "else" clause
            tasks = np.array_split(np.arange(self.n_boots_sel *
                                             self.n_reg_params_), size)[rank]
            selection_coefs = np.empty((tasks.size, n_coef))
            # but my_boots is still indexed by the boots, since the reg_param is picked separately when fitting is about to begin
            my_boots = dict((task_idx // self.n_reg_params_, None)
                            for task_idx in tasks)
        else: 
            # less process than n_boots_sel, so some process will have multiple boots with all there reg_param runs as well
            # we can see the selection_coeff will have space for all the reg_params
            
            # split up bootstraps into processes
            tasks = np.array_split(np.arange(self.n_boots_sel),
                                   size)[rank]
            selection_coefs = np.empty((tasks.size, self.n_reg_params_,
                                        n_coef))
            my_boots = dict((task_idx, None) for task_idx in tasks)

        # this is where it assign the sample index to each bootstrap
        
        # with the rvals, rank 0 pick out the samples correponding rvals for each process, then sends them
        # it will need to know what process has what "boot" tho, which can be obtained from "tasks" before taking the "[rank]", it's split into num_processes
        # if size > self.n_boots_sel, then process "rank" will have boot=tasks[rank]//self.n_reg_params_
        for boot in range(self.n_boots_sel):
            if size > 1: # num_MPI_process
                if rank == 0:
                    # selecting the samples for this particular bootstrap
                    # rvals[0] is the training_set_idx, rvals[1] is the test_set_idx
                    rvals = train_test_split(np.arange(data.shape[0]-lag),    #select from the (num of raw samples - lag)
                                             test_size=1 - self.selection_frac,
                                             stratify=stratify,
                                             random_state=self.random_state)                      
                else:
                    rvals = [None] * 2

                # brocasting raw rvals to all processes and let them decide for themselves if rvals are their's to keep
                rvals = [Bcast_from_root(rval, self.comm, root=0)
                         for rval in rvals]
                
                if boot in my_boots.keys():
                    # the raw rvals are picking entries of the Y vector
                    my_boots[boot] = rvals
                    
            else: #non-MPI
                my_boots[boot] = train_test_split(
                    np.arange(X.shape[0]//self.n_real_features),
                    test_size=1 - self.selection_frac,
                    stratify=stratify,
                    random_state=self.random_state)               
     
        # iterate over bootstraps(and reg_params)
        # this is the actually doing the fit, 
        curr_boot_idx = None
        for ii, task_idx in enumerate(tasks):
            if size > self.n_boots_sel:
                boot_idx = task_idx // self.n_reg_params_
                reg_idx = task_idx % self.n_reg_params_
                my_reg_params = [self.reg_params_[reg_idx]]
            else:
                boot_idx = task_idx
                my_reg_params = self.reg_params_
            # Never warm start across bootstraps
            if (curr_boot_idx != boot_idx):
                if hasattr(self._selection_lm, 'coef_'):
                    self._selection_lm.coef_ *= 0.
                if hasattr(self._selection_lm, 'intercept_'):
                    self._selection_lm.intercept_ *= 0.

            # draw a resampled bootstrap, and vectorize it
            # minimize repeated vecotrization if using the same bootstrap
            if curr_boot_idx != boot_idx:
                idxs_train, idxs_test = my_boots[boot_idx]
                X, y = vectorization_bootstrap(data, idxs_train, lag)
            curr_boot_idx = boot_idx
 
            X_rep = X
            y_rep = y  

            # fit the coefficients
            if size > self.n_boots_sel:
                msg = ("selection bootstrap %d, "
                       "regularization parameter set %d"
                       % (boot_idx, reg_idx))
                self._logger.info(msg)

            else:
                self._logger.info("selection bootstrap %d" % (boot_idx))
            selection_coefs[ii] = np.squeeze(
                self.uoi_selection_sweep(X_rep, y_rep, my_reg_params))

        # if distributed, gather selection coefficients to 0,
        # perform intersection, and broadcast results
        if size > 1:
            selection_coefs = Gatherv_rows(selection_coefs, self.comm, root=0)
            if rank == 0:
                if size > self.n_boots_sel:
                    selection_coefs = selection_coefs.reshape(
                        self.n_boots_sel,
                        self.n_reg_params_,
                        n_coef)
                supports = self.intersect(
                    selection_coefs,
                    self.selection_thresholds_).astype(int)
            else:
                supports = None
            supports = Bcast_from_root(supports, self.comm, root=0)
            self.supports_ = supports.astype(bool)
        else:
            self.supports_ = self.intersect(selection_coefs,
                                            self.selection_thresholds_)

        self.n_supports_ = self.supports_.shape[0]

        if rank == 0:
            self._logger.info("Found %d supports" % self.n_supports_)

        #####################
        # Estimation Module #
        #####################
        # set up data arrays
        
        tasks = np.array_split(np.arange(self.n_boots_est *
                                         self.n_supports_), size)[rank]
        my_boots = dict((task_idx // self.n_supports_, None)
                        for task_idx in tasks)
        estimates = np.zeros((tasks.size, n_coef))
        

      
        for boot in range(self.n_boots_est):
            if size > 1:
                if rank == 0:
                    rvals = train_test_split(np.arange(X.shape[0]//self.n_real_features),
                                             test_size=1 - self.estimation_frac,
                                             stratify=stratify,
                                             random_state=self.random_state)
                else:
                    rvals = [None] * 2
                rvals = [Bcast_from_root(rval, self.comm, root=0)
                         for rval in rvals]
                if boot in my_boots.keys():
                    my_boots[boot] = rvals
            else:
                my_boots[boot] = train_test_split(
                    np.arange(X.shape[0]//self.n_real_features),
                    test_size=1 - self.estimation_frac,
                    stratify=stratify,
                    random_state=self.random_state)

        # score (r2/AIC/AICc/BIC) for each bootstrap for each support
        scores = np.zeros(tasks.size)
                
        # iterate over bootstrap samples and supports

        curr_boot_idx = None
        for ii, task_idx in enumerate(tasks):
            
            boot_idx = task_idx // self.n_supports_
            support_idx = task_idx % self.n_supports_
            support = self.supports_[support_idx]
            # draw a resampled bootstrap
            if curr_boot_idx != boot_idx:
                idxs_train, idxs_test = my_boots[boot_idx]
                X, y = vectorization_bootstrap(data, idxs_train, lag)
            curr_boot_idx = boot_idx
 
            X_rep = X
            y_rep = y  


            self._logger.info("estimation bootstrap %d, support %d"
                              % (boot_idx, support_idx))
            
            
            if np.any(support):
                

                # compute the estimate and store the fitted coefficients
                if self.shared_support:
                    self._estimation_lm.fit(X_rep[:, support], y_rep)
       
                    
                    estimates[ii, np.tile(support, self.output_dim)] = \
                        self._estimation_lm.coef_.ravel()
                else:
                    self._estimation_lm.fit(X_rep, y_rep, coef_mask=support)
                    estimates[ii] = self._estimation_lm.coef_.ravel()

                if self.estimation_score in ["r2", "acc", "log"]:  # using test set
                    X_score, y_score = vectorization_bootstrap(data, idxs_test, lag)

                else:   #using training set
                    X_score, y_score = X_rep, y_rep
                    
                scores[ii] = self._score_predictions(
                    metric=self.estimation_score,
                    fitter=self._estimation_lm,
                    X=X_score, y=y_score,
                    support=support)
            else:
                fitter = self._fit_intercept_no_features(y_rep)

                if self.estimation_score in ["r2", "acc", "log"]:  # using test set
                    X_score, y_score = vectorization_bootstrap(data, idxs_test, lag)
                else:  #using training set
                    X_score, y_score = X_rep, y_rep
                    
                if issparse(X_score):
                    X_score = csr_matrix(X_score.shape, dtype=X_score.dtype)    
                else:
                    X_score = np.zeros_like(X_score)
                    
                scores[ii] = self._score_predictions(
                    metric=self.estimation_score,
                    fitter=fitter,
                    X=X_score, y=y_score,
                    support=np.zeros(X.shape[1], dtype=bool))

        if size > 1:
            estimates = Gatherv_rows(send=estimates, comm=self.comm,
                                     root=0)
            scores = Gatherv_rows(send=scores, comm=self.comm,
                                  root=0)
            self.rp_max_idx_ = None
            best_estimates = None
            coef = None
            self.intercept_ = None
            if rank == 0:
                estimates = estimates.reshape(self.n_boots_est,
                                              self.n_supports_, n_coef)
                scores = scores.reshape(self.n_boots_est, self.n_supports_)
                self.rp_max_idx_ = np.argmax(scores, axis=1)
                best_estimates = estimates[np.arange(self.n_boots_est),
                                           self.rp_max_idx_]
                # take the median across estimates for the final estimate
                coef = np.median(best_estimates,
                                 axis=0).reshape(self.output_dim, n_features)
                self.coef_ = coef
                if X_all is None:
                    X_all, y_all = vectorization(data, lag)    
                self._fit_intercept(X_all, y_all)
            self.estimates_ = Bcast_from_root(estimates, self.comm, root=0)
            self.scores_ = Bcast_from_root(scores, self.comm, root=0)
            self.coef_ = Bcast_from_root(coef, self.comm, root=0)  #it's done again down below for recaled VAR coef
            self.intercept_ = Bcast_from_root(self.intercept_,
                                              self.comm, root=0)
            self.rp_max_idx_ = self.comm.bcast(self.rp_max_idx_, root=0)
            
        else:
            self.estimates_ = estimates.reshape(self.n_boots_est,
                                                self.n_supports_, n_coef)
            self.scores_ = scores.reshape(self.n_boots_est, self.n_supports_)
            self.rp_max_idx_ = np.argmax(self.scores_, axis=1)
            # extract the estimates over bootstraps from model with best
            # regularization parameter value
            best_estimates = self.estimates_[np.arange(self.n_boots_est),
                                             self.rp_max_idx_, :]
            # take the median across estimates for the final, bagged estimate
            self.coef_ = np.median(best_estimates,
                                   axis=0).reshape(self.output_dim, n_features)
            if X_all is None:
                X_all, y_all = vectorization(data, lag)               
            self._fit_intercept(X_all, y_all)
            
        if rank == 0:
            self._post_fit_VAR(lag, data.shape[1])
        
        if size > 1:
            self.VAR_coef_ = Bcast_from_root(self.VAR_coef, self.comm, root=0)

        return self

    def uoi_selection_sweep(self, X, y, reg_param_values):
        """Perform selection regression on a dataset over a sweep of
        regularization parameter values.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, shape (n_samples, n_features)
            The design matrix.
        y : ndarray, shape (n_samples,)
            Response vector.
        reg_param_values: list of dicts
            A list of dictionaries containing the regularization parameter
            values to iterate over.

        Returns
        -------
        coefs : ndarray, shape (n_param_values, n_features)
            Predicted parameter values for each regularization strength.
        """

        n_param_values = len(reg_param_values)
        n_coef = self.get_n_coef(X, y)

        coefs = np.zeros((n_param_values, n_coef))

        # apply the selection regression to bootstrapped datasets
        for reg_param_idx, reg_params in enumerate(reg_param_values):
            # reset the regularization parameter
            self._selection_lm.set_params(**reg_params)
            # rerun fit
            self._selection_lm.fit(X, y)
            # store coefficients
            
            coefs[reg_param_idx] = self._selection_lm.coef_.ravel()

        return coefs

    def get_n_coef(self, X, y):
        """Return the number of coefficients that will be estimated

        This should return the shape of X.
        """
        return X.shape[1] * self.output_dim


class AbstractUoILinearRegressor(AbstractUoILinearModel,
                                 metaclass=_abc.ABCMeta):
    """An abstract base class for UoI linear regression classes."""

    _valid_estimation_metrics = ('r2', 'AIC', 'AICc', 'BIC')

    _train_test_map = {'train': 0, 'test': 1}

    _default_est_targets = {'r2': 1, 'AIC': 0, 'AICc': 0, 'BIC': 0}

    def __init__(self, n_real_features = 1, fit_VAR = False, n_boots_sel=12, n_boots_est=12, selection_frac=0.9,
                 estimation_frac=0.9, stability_selection=0.75,
                 estimation_score='r2', estimation_target=None,
                 copy_X=True, fit_intercept=True,
                 standardize=True, random_state=None, max_iter=None, tol=None,
                 comm=None, logger=None):
        super(AbstractUoILinearRegressor, self).__init__(
            n_real_features = n_real_features,
            fit_VAR = fit_VAR,   
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            estimation_frac=estimation_frac,
            stability_selection=stability_selection,
            fit_intercept=fit_intercept,
            standardize=standardize,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            comm=comm,
            logger=logger)

        if estimation_score not in self._valid_estimation_metrics:
            raise ValueError(
                "invalid estimation metric: '%s'" % estimation_score)

        self.__estimation_score = estimation_score

        if estimation_target is not None:
            if estimation_target not in ['train', 'test']:
                raise ValueError(
                    "invalid estimation target: %s" % estimation_target)
            else:
                estimation_target = self._train_test_map[estimation_target]
        else:
            
            estimation_target = self._default_est_targets[estimation_score]
            
        self._estimation_target = estimation_target

    
    def _pre_fit(self, X, y):
        X, y = super()._pre_fit(X, y)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        elif y.ndim == 2:
            if y.shape[1] > 1:
                raise ValueError('y should either have shape ' +
                                 '(n_samples, ) or (n_samples, 1).')
        else:
            raise ValueError('y should either have shape ' +
                             '(n_samples, ) or (n_samples, 1).')
        if self.standardize:
            self._y_scaler = StandardScaler(with_mean=self.fit_intercept)
            y = self._y_scaler.fit_transform(y)
        y = np.squeeze(y)
        self.output_dim = 1
        return X, y

    def _post_fit(self, X, y):
        super()._post_fit(X, y)
        if self.standardize:
            sX = self._X_scaler
            sy = self._y_scaler
            self.coef_ *= sy.scale_[:, np.newaxis]
            if self.fit_intercept:
                self.intercept_ *= sy.scale_
                self.intercept_ += sy.mean_ - np.dot(sX.mean_,
                                                     self.coef_.T)
        self.coef_ = np.squeeze(self.coef_)

    def intersect(self, coef, thresholds):
        """Intersect coefficients accross all thresholds."""
        return intersection(coef, thresholds)

    @property
    def estimation_score(self):
        return self.__estimation_score

    def _score_predictions(self, metric, fitter, X, y, support):
        """Score, according to some metric, predictions provided by a model.

        The resulting score will be negated if an information criterion is
        specified.

        Parameters
        ----------
        metric : string
            The type of score to run on the prediction. Valid options include
            'r2' (explained variance), 'BIC' (Bayesian information criterion),
            'AIC' (Akaike information criterion), and 'AICc' (corrected AIC).
        fitter : object
            Must contain .predict and .predict_proba methods.
        X : array-like
            The selected design matrix for score prediction.
        y : array-like
            The selected Response vector for score prediction.
        supports : array-like
            The value of the supports for the model
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


        if y.ndim == 2:
            if y.shape[1] > 1:
                raise ValueError('y should either have shape ' +
                                 '(n_samples, ) or (n_samples, 1).')
            y = np.squeeze(y)
        elif y.ndim > 2:
            raise ValueError('y should either have shape ' +
                             '(n_samples, ) or (n_samples, 1).')

        y_pred = fitter.predict(X[:, support])
        if y.shape != y_pred.shape:
            raise ValueError('Targets and predictions are not the same shape.')

        if metric == 'r2':
            score = r2_score(y, y_pred)
        else:
            ll = utils.log_likelihood_glm(model='normal',
                                          y_true=y,
                                          y_pred=y_pred)
            n_features = np.count_nonzero(support)
            n_samples = X.shape[0]
            if metric == 'BIC':
                score = utils.BIC(ll, n_features, n_samples)
            elif metric == 'AIC':
                score = utils.AIC(ll, n_features)
            elif metric == 'AICc':
                score = utils.AICc(ll, n_features, n_samples)
            else:
                raise ValueError(metric + ' is not a valid option.')
            # negate the score since lower information criterion is preferable
            score = -score
        return score

    def _fit_intercept_no_features(self, y):
        """Fit a model with only an intercept.

        This is used in cases where the model has no support selected.
        """
        return LinearInterceptFitterNoFeatures(y)

    def _fit_intercept(self, X, y):
        """Fit the intercept."""
        if self.fit_intercept:
            self.intercept_ = (y.mean(axis=0) -
                               np.dot(X.mean(axis=0), self.coef_.T))
        else:
            self.intercept_ = np.zeros(1)


class LinearInterceptFitterNoFeatures(object):
    def __init__(self, y):
        self.intercept_ = y.mean()

    def predict(self, X):
        n_samples = X.shape[0]
        return np.tile(self.intercept_, n_samples)


class AbstractUoIGeneralizedLinearRegressor(AbstractUoILinearModel,
                                            metaclass=_abc.ABCMeta):
    """An abstract base class for UoI linear classifier classes."""

    _valid_estimation_metrics = ('log', 'BIC', 'AIC', 'AICc', 'acc')

    _train_test_map = {'train': 0, 'test': 1}

    _default_est_targets = {'log': 1, 'AIC': 0, 'AICc': 0,
                            'BIC': 0, 'acc': 1}

    def __init__(self, n_real_features = 1, fit_VAR = False, n_boots_sel=12, n_boots_est=12, selection_frac=0.9,
                 estimation_frac=0.9, stability_selection=0.75,
                 estimation_score='acc', estimation_target=None,
                 copy_X=True, fit_intercept=True, standardize=True,
                 random_state=None, max_iter=None, tol=None,
                 shared_support=True, comm=None, logger=None):
        super(AbstractUoIGeneralizedLinearRegressor, self).__init__(
            n_real_features = n_real_features,
            fit_VAR = fit_VAR,  
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            estimation_frac=estimation_frac,
            stability_selection=stability_selection,
            random_state=random_state,
            fit_intercept=fit_intercept,
            standardize=standardize,
            shared_support=shared_support,
            max_iter=max_iter,
            tol=tol,
            comm=comm,
            logger=logger)

        if estimation_score not in self._valid_estimation_metrics:
            raise ValueError(
                "invalid estimation metric: '%s'" % estimation_score)
        self.__estimation_score = estimation_score

        if estimation_target is not None:
            if estimation_target not in ['train', 'test']:
                raise ValueError(
                    "invalid estimation target: %s" % estimation_target)
            else:
                estimation_target = self._train_test_map[estimation_target]
        else:
            
            estimation_target = self._default_est_targets[estimation_score]

        self._estimation_target = estimation_target

    def _post_fit(self, X, y):
        super()._post_fit(X, y)
        if self.standardize and self.fit_intercept:
            sX = self._X_scaler
            self.intercept_ += np.dot(sX.mean_ * sX.scale_,
                                      self.coef_.T)

    def intersect(self, coef, thresholds):
        """Intersect coefficients accross all thresholds.

        This implementation will account for multi-class classification.
        """
        supports = intersection(coef, thresholds)
        if self.output_dim > 1 and self.shared_support:
            n_features = supports.shape[-1] // self.output_dim
            supports = supports.reshape((-1, self.output_dim, n_features))
            supports = np.sum(supports, axis=-2).astype(bool)
            supports = np.unique(supports, axis=0)
        return supports

    @property
    def estimation_score(self):
        return self.__estimation_score

    def _score_predictions(self, metric, fitter, X, y, support):
        """Score, according to some metric, predictions provided by a model.

        The resulting score will be negated if an information criterion is
        specified.

        Parameters
        ----------
        metric : string
            The type of score to run on the prediction. Valid options include
            'r2' (explained variance), 'BIC' (Bayesian information criterion),
            'AIC' (Akaike information criterion), and 'AICc' (corrected AIC).
        fitter : object
            Must contain .predict and .predict_proba methods.
        X : array-like
            The selected design matrix for score prediction.
        y : array-like
            The selected Response vector for score prediction.
        supports : array-like
            The value of the supports for the model
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

        if metric == 'acc':
            if self.shared_support:
                y_pred = fitter.predict(X[:, support])
            else:
                y_pred = fitter.predict(X)
            score = accuracy_score(y, y_pred)
        else:

            if self.shared_support:
                y_pred = fitter.predict_proba(X[:, support])
            else:
                y_pred = fitter.predict_proba(X)
            ll = -log_loss(y, y_pred, labels=self.classes_)
            if metric == 'log':
                score = ll
            else:
                n_features = np.count_nonzero(support)
                n_samples = X.shape[0]
                if metric == 'BIC':
                    score = utils.BIC(n_samples * ll, n_features, n_samples)
                elif metric == 'AIC':
                    score = utils.AIC(n_samples * ll, n_features)
                elif metric == 'AICc':
                    score = utils.AICc(n_samples * ll, n_features, n_samples)
                else:
                    raise ValueError(metric + ' is not a valid metric.')
                # negate the score since lower information criterion is
                # preferable
                score = -score

        return score
