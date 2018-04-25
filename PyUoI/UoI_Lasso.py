import numpy as np

from tqdm import trange

import sklearn.linear_model as lm
from sklearn.linear_model.base import (_preprocess_data, _rescale_data,
									   SparseCoefMixin)
from sklearn.metrics import r2_score
from sklearn.utils import check_X_y

from PyUoI import utils

# get rid of annoying LAPACK warning
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

class UoI_Lasso(lm.base.LinearModel, SparseCoefMixin):
	"""Class modeled after scikit-learn's linear_model suite of regression solvers.
	This class performs UoI-Lasso, developed by Bouchard et al. (2017).

	Attributes ending in an underscore should be read-only, not changed by the user.

	Parameters
	----------
	n_lambdas_ : int
		number of L1 penalty values to compare across (effectively sets the
		hyperparameter sweep)

	selection_thres_frac : float
		used for soft thresholding in the selection step. normally, UoI-Lasso
		requires regressors to be selected in _all_ bootstraps to be selected
		for use in the estimation module. this requirement can be softened with
		this variable, by requiring that a regressor appear in
		selection_thres_frac of the bootstraps.

	train_frac_sel : float
		fraction of dataset to be used for training in the selection module.

	train_frac_est : float
		fraction of dataset to be used for training in each bootstrap in the
		estimation module.

	train_frac_overall : float
		fraction of dataset to be used for training in the overall estimation module.

	n_boots_coarse : int
		number of bootstraps to use in the coarse lasso sweep.

	n_boots_sel : int
		number of bootstraps to use in the selection module (dense lasso sweep).

	n_boots_est : int
		number of bootstraps to use in the estimation module.

	bagging_options : int
		equal to 1: for each bootstrap sample, find the regularization
			parameter that gave the best results
		equal to 2: average estimates across bootstraps, and then find the
			regularization parameter that gives the best results

	n_minibatch : int
		number of minibatches to use in case SGD is used for Lasso (selection)
		or OLS (estimation).

	use_admm : boolean
		flag indicating whether to use the ADMM algorithm.

	n_samples_ : int
		number of samples in the dataset

	n_features_ : int
		number of features in the dataset

	fit_intercept : boolean, optional, default True
		whether to calculate the intercept for this model. If set
		to False, no intercept will be used in calculations
		(e.g. data is expected to be already centered).

	normalize : boolean, optional, default False
		This parameter is ignored when ``fit_intercept`` is set to False.
		If True, the regressors X will be normalized before regression by
		subtracting the mean and dividing by the l2-norm.
		If you wish to standardize, please use
		:class:`sklearn.preprocessing.StandardScaler` before calling ``fit`` on
		an estimator with ``normalize=False``.

	copy_X : boolean, optional, default True
		If True, X will be copied; else, it may be overwritten.


	Attributes
	----------
	coef_ : array, shape (n_features, ) or (n_targets, n_features)
		Estimated coefficients for the linear regression problem.
		If multiple targets are passed during the fit (y 2D), this
		is a 2D array of shape (n_targets, n_features), while if only
		one target is passed, this is a 1D array of length n_features.

	intercept_ : array
		Independent term in the linear model.

	scores_ : float
		contains the performance of each UoI-Lasso fit on a held out test set

	supports_ : np array of booleans, shape = (number of lambdas) x (number of features)
		boolean array indicating whether a given regressor (column) is selected for estimation
		for a given lambda (row).

	TODO:
		- SGDRegressor has not been tested in either the selection or estimation modules
		- usage with ADMM is built-in, but not tested
	"""

	def __init__(self, n_lambdas=48, selection_thres_frac=1.0,
				 train_frac_sel=0.8, train_frac_est=0.8, train_frac_overall=0.9, 
				 n_boots_coarse=10, n_boots_sel=48, n_boots_est=48, 
				 bagging_options=1, n_minibatch=10, use_admm=False, 
				 copy_X=True, fit_intercept=True, normalize=False):
		# hyperparameters
		self.n_lambdas = n_lambdas
		self.selection_thres_frac = selection_thres_frac
		# data split fractions
		self.train_frac_sel = train_frac_sel
		self.train_frac_est = train_frac_est
		self.train_frac_overall = train_frac_overall
		# number of bootstraps
		self.n_boots_coarse = n_boots_coarse
		self.n_boots_sel = n_boots_sel
		self.n_boots_est = n_boots_est
		# backup options and switches
		self.bagging_options = bagging_options
		self.n_minibatch = n_minibatch
		self.use_admm = use_admm
		# mimics sklearn.LinearModel.base.LinearRegression
		self.copy_X = copy_X
		self.fit_intercept = fit_intercept
		self.normalize = normalize

	def fit(self, X, y, seed=None, verbose=False, sample_weight=None):
		"""Fit data according to the UoI-Lasso algorithm.
		Relevant information (fits, residuals, model performance) is stored within object.
		Thus, nothing is returned by this function.

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
			for the choosing bootstraps and dividing the data into training and test sets.

		verbose : boolean
			a boolean switch indicating whether the fitting should print out its progress.
		"""
		# start taken from sklearn.LinearModels.base.LinearRegression
		X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
						 y_numeric=True, multi_output=True)

		# start the seed, if it's provided
		if seed is not None:
			np.random.seed(seed)

		# sklearn data preprocessing and checks
		X, y, X_offset, y_offset, X_scale = _preprocess_data(
			X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
			copy=self.copy_X)

		if sample_weight is not None and np.atleast_1d(sample_weight).ndim > 1:
			raise ValueError("Sample weights must be 1D array or scalar")

		if sample_weight is not None:
			# Sample weight can be implemented via a simple rescaling.
			X, y = _rescale_data(X, y, sample_weight)
		
		# extract model dimensions from design matrix
		self.n_samples_, self.n_features_ = X.shape

		if verbose:
			print('(1) Loaded data.\n %s samples with %s features.' % (self.n_samples_, self.n_features_))

		# perform an initial coarse sweep over the lambda parameters
		# this is to zero-in on the relevant regularization region.
		if self.n_lambdas == 1:
			lambda_coarse = np.array([1.0])
		else:
			lambda_coarse = np.logspace(-3., 3., self.n_lambdas, dtype=np.float64)
		# run the coarse lasso sweep
		estimates_coarse, scores_coarse = \
			self.lasso_sweep(X, y, lambda_coarse, self.train_frac_sel,
							 self.n_boots_coarse, self.n_minibatch,
							 self.use_admm, desc='coarse lasso sweep', verbose=verbose)
		# deduce the index which maximizes the explained variance over bootstraps
		lambda_max_idx = np.argmax(np.mean(scores_coarse, axis=0))
		# obtain the lambda which maximizes the explained variance over bootstraps
		lambda_max = lambda_coarse[lambda_max_idx]
		# in our dense sweep, we'll explore lambda values which encompass a
		# range that's one order of magnitude less than lambda_max itself
		d_lambda = 10 ** (np.floor(np.log10(lambda_max)) - 1)

		# now that we've narrowed down the regularization parameters, 
		# we'll run a dense sweep which begins the model selection module of UoI

		#######################
		### Model Selection ###
		#######################
		if verbose:
			print('(2) Beginning model selection. Exploring penalty region centered at %d.' % lambda_max)

		# create the final lambda set based on the coarse sweep
		if self.n_lambdas == 1:
			lambdas = np.array([lambda_max])
		else:
			lambdas = np.linspace(lambda_max - 5 * d_lambda,
								lambda_max + 5 * d_lambda, self.n_lambdas,
								dtype=np.float64)
		# run the lasso sweep with new lambda set
		estimates_dense, scores_dense = \
			self.lasso_sweep(X, y, lambdas, self.train_frac_sel,
							self.n_boots_sel, self.n_minibatch, self.use_admm,
							desc='fine lasso sweep', verbose=verbose)
		# intersect supports across bootstraps for each lambda value
		# we impose a (potentially) soft intersection
		threshold = int(self.selection_thres_frac * self.n_boots_sel)
		self.supports_ = np.count_nonzero(estimates_dense, axis=0) >= threshold

		########################
		### Model Estimation ###
		########################
		# we'll use the supports obtained in the selection module to calculate
		# bagged OLS estimates over bootstraps

		if verbose:
			print('(3) Model selection complete. Beginning model estimation, with %s bootstraps.' % self.n_boots_est)

		# create or overwrite arrays to collect final results
		self.coef_ = np.zeros(self.n_features_, dtype=np.float32)
		self.scores_ = np.zeros(1, dtype=np.float32)
		# determine how many samples will be used for overall training
		train_split = int(round(self.train_frac_overall * self.n_samples_))
		# determine how many samples will be used for training within a bootstrap
		boot_train_split = int(round(self.train_frac_est * train_split))

		# set up data arrays
		estimates = np.zeros((self.n_boots_est, self.n_lambdas, self.n_features_), dtype=np.float32)
		scores = np.zeros((self.n_boots_est, self.n_lambdas), dtype=np.float32)
		# either we plan on using a test set, or we'll use the entire dataset for training
		if self.train_frac_overall < 1:
			# generate indices for the global training and testing blocks
			indices = np.random.permutation(self.n_samples_)
			train, test = np.split(indices, [train_split])
			# compile the training and test sets
			X_train = X[train]
			y_train = y[train]
			X_test = X[test]
			y_test = y[test]
		else:
			X_train = X
			y_train = y

		# iterate over bootstrap samples
		for bootstrap in trange(self.n_boots_est, desc='Model Estimation', disable=not verbose):
			# extract the bootstrap indices, keeping a fraction of the data
			# available for testing
			bootstrap_indices = np.random.permutation(train_split)
			train_boot, test_boot = np.split(bootstrap_indices, [boot_train_split])
			# iterate over the regularization parameters
			for lamb_idx, lamb in enumerate(lambdas):
				if np.any(self.supports_[lamb_idx]):
					# fit OLS using the supports from selection module
					try:
						X_boot = X_train[train_boot]
						y_boot = y_train[train_boot]
						ols = lm.LinearRegression()
						ols.fit(
							X_boot[:, self.supports_[lamb_idx]],
							y_boot - y_boot.mean()
						)
					except:
						ols = lm.SGDRegressor(penalty='none')
						for batch_idx in range(self.n_minibatch):
							minibatch = X_boot[batch_idx::self.n_minibatch]
							ols.partial_fit(
								X_train[minibatch][:, self.supports_[lamb_idx]],
								y_train[minibatch] - y_train[minibatch].mean()
							)
					# store the fitted coefficients
					estimates[bootstrap, lamb_idx, self.supports_[lamb_idx]] = ols.coef_
					# calculate and store the performance on the test set
					y_hat_boot = np.dot(X_train[test_boot], estimates[bootstrap, lamb_idx, :])
					y_true_boot = y_train[test_boot] - y_train[test_boot].mean()
					# calculate sum of squared residuals
					rss = np.sum((y_hat_boot - y_true_boot)**2)
					# calculate BIC as our scoring function
					scores[bootstrap, lamb_idx] = utils.BIC(self.n_features_, boot_train_split, rss)
				else:
					# if no variables were selected, throw a message
					print('No variables selected in the support for lambda = %d.' % lamb)

		if verbose:
			print('(4) Bagging estimates, using bagging option %s.' % self.bagging_options)

		if self.bagging_options == 1:
			# bagging option 1: for each bootstrap sample, find the regularization parameter that gave the best results
			lambda_max_idx = np.argmax(scores, axis=1)
			# extract the estimates over bootstraps from the model with best lambda
			best_estimates = estimates[np.arange(self.n_boots_est), lambda_max_idx, :]
			# take the median across estimates for the final, bagged estimate
			self.coef_ = np.median(best_estimates, axis=0)
		elif self.bagging_options == 2:
			# bagging option 2: average estimates across bootstraps, and then find the regularization parameter that gives the best results
			mean_scores = np.mean(scores, axis=0)
			lambda_max_idx = np.argmax(mean_scores)
			self.coef_ = np.median(estimates[:, lambda_max_idx, :], 0)
		else:
			raise ValueError(
				'Bagging option %d is not available.' % self.bagging_options
			)
		# if we extracted a test set, evaluate the model
		if self.train_frac_overall < 1:
			# finally, see how the bagged estimates perform on the test set
			y_hat = np.dot(X_test, self.coef_)
			y_true = y_test - y_test.mean()
			# calculate and store performance of the final UoI_Lasso estimator over test set
			self.scores_ = r2_score(y_true, y_hat)
		else:
			self.scores_ = None

		if verbose:
			print("---> UoI Lasso complete.")

		if y.ndim == 1:
			self.coef_ = np.ravel(self.coef_)
		self._set_intercept(X_offset, y_offset, X_scale)

		return self

	@staticmethod
	def lasso_sweep(X, y, lambdas, train_frac, n_bootstraps, n_minibatch,
					use_admm=False, seed=None, desc='', verbose=False):
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

		n_minibatch : int
			number of minibatches to use in case SGD is used for the regression

		use_admm: bool
			switch to use the alternating direction method of multipliers (
			ADMM) algorithm

		Returns
		-------
		estimates : np.array
			predicted regressors for each bootstrap and lambda value; shape is
			(n_bootstraps, n_lambdas, n_features)

		scores : np.array
			scores by the model for each bootstrap and lambda
			value; shape is (n_bootstraps, n_lambdas)
		"""

		# if a seed is provided, seed the random number generator
		if seed is not None:
			np.random.seed(seed)
		#
		n_lambdas = len(lambdas)
		n_samples, n_features = X.shape
		n_train_samples = int(round(train_frac * n_samples))
		# create arrays to collect results
		estimates = np.zeros((n_bootstraps, n_lambdas, n_features), dtype=np.float32)
		scores = np.zeros((n_bootstraps, n_lambdas), dtype=np.float32)
		# apply the Lasso to bootstrapped datasets
		for bootstrap in trange(n_bootstraps, desc=desc, disable=not verbose):
			# for each bootstrap, we'll split the data into a randomly assigned training and test set
			indices = np.random.permutation(n_samples)
			train, test = np.split(indices, [n_train_samples])
			# iterate over the provided L1 penalty values
			for lamb_idx, lamb in enumerate(lambdas):
				# run the Lasso on the training set
				if not use_admm:
					# either use the sklearn Lasso class, or apply SGD if we run into problems
					try:
						lasso = lm.Lasso(alpha=lamb)
						lasso.fit(X[train], y[train] - y[train].mean())
					except:
						lasso = lm.SGDRegressor(penalty='l1', alpha=lamb)
						# run SGD over minibatches from the dataset
						for batch_idx in range(n_minibatch):
							minibatch = range(batch_idx, n_samples,
											  n_minibatch)
							lasso.partial_fit(X[minibatch],
											  y[minibatch] - y[
												  minibatch].mean())
					estimates[bootstrap, lamb_idx, :] = lasso.coef_
				else:
					estimates[bootstrap, lamb_idx, :] = utils.lasso_admm(X[train], (y[train] - y[train].mean()), alpha=lamb)
				# run trained Lasso on the test set and obtain predictions
				y_hat = X[test].dot(estimates[bootstrap, lamb_idx, :])
				y_true = y[test] - y[test].mean()
				# calculate the explained variance using the predicted values
				scores[bootstrap, lamb_idx] = r2_score(y_true, y_hat)

		return estimates, scores