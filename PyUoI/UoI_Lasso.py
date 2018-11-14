import numpy as np

from tqdm import trange

import sklearn.linear_model as lm
from sklearn.linear_model.base import (_preprocess_data, _rescale_data,
									   SparseCoefMixin)
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.metrics import r2_score
from sklearn.utils import check_X_y

from PyUoI import utils

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
	"""

	def __init__(
		self, n_lambdas=48, 
		selection_thres_max=1., selection_thres_min=1., n_selection_thres=1,
		train_frac_sel=0.8, train_frac_est=0.8, 
		n_boots_coarse=10, n_boots_sel=48, n_boots_est=48, 
		bagging_options=1, use_admm=False, estimation_score='BIC', 
		copy_X=True, fit_intercept=True, normalize=False,
		random_state=None
	):
		# hyperparameters
		self.n_lambdas = n_lambdas
		# data split fractions
		self.train_frac_sel = train_frac_sel
		self.train_frac_est = train_frac_est
		# number of bootstraps
		self.n_boots_coarse = n_boots_coarse
		self.n_boots_sel = n_boots_sel
		self.n_boots_est = n_boots_est
		# backup options and switches
		self.bagging_options = bagging_options
		self.use_admm = use_admm
		# selection thresholds
		self.selection_thres_max = selection_thres_max
		self.selection_thres_min = selection_thres_min
		self.n_selection_thres = n_selection_thres
		# scoring 
		self.estimation_score = estimation_score
		# mimics sklearn.LinearModel.base.LinearRegression
		self.copy_X = copy_X
		self.fit_intercept = fit_intercept
		self.normalize = normalize

	def fit(self, X, y, groups=None, seed=None, verbose=False, sample_weight=None, option=True):
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
		# initialize the seed, if it's provided
		if seed is not None:
			np.random.seed(seed)

		X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
						 y_numeric=True, multi_output=True)

		# preprocess data through centering and normalization
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
		# create or overwrite arrays to collect final results
		self.coef_ = np.zeros(self.n_features_, dtype=np.float32)
			
		# group leveling 
		if groups is None:
			self.groups_ = np.ones(self.n_samples_)
		else:
			self.groups_ = np.array(groups)

		if verbose:
			print('(1) Loaded data.\n %s samples with %s features.' % (self.n_samples_, self.n_features_))

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
				self.use_admm, desc='fine lasso sweep', verbose=verbose
			)

		# perform the intersection step
		self.intersection(estimates_selection)

		########################
		### Model Estimation ###
		########################
		# we'll use the supports obtained in the selection module to calculate
		# bagged OLS estimates over bootstraps

		if verbose:
			print('(3) Beginning model estimation, with %s bootstraps.' % self.n_boots_est)

		# compute number of samples per bootstrap
		n_samples_bootstrap = int(round(self.train_frac_est * self.n_samples_))

		# set up data arrays
		estimates = np.zeros((self.n_boots_est, self.n_lambdas, self.n_features_), dtype=np.float32)
		scores = np.zeros((self.n_boots_est, self.n_lambdas), dtype=np.float32)

		# iterate over bootstrap samples
		for bootstrap in trange(self.n_boots_est, desc='Model Estimation', disable=not verbose):

			# extract the bootstrap indices, keeping a fraction of the data available for testing
			train_idx, test_idx = utils.leveled_randomized_ids(self.groups_, self.train_frac_est)

			# iterate over the regularization parameters
			for lamb_idx, lamb in enumerate(self.lambdas):

				support = self.supports_[lamb_idx]

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

					# calculate estimation score
					if self.estimation_score == 'r2':
						scores[bootstrap, lamb_idx] = ols.score(X_test, y_test)
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
						raise ValueError(str(self.estimation_score) + ' is not a valid option.')
				else:
					if self.estimation_score == 'r2':
						scores[bootstrap, lamb_idx] = r2_score(
							y_true=y_test, 
							y_pred=np.zeros(y_test.size)
						)
					elif self.estimation_score == 'BIC':
						n_features = 0
						scores[bootstrap, lamb_idx] = -utils.BIC(
							y_true=y_test,
							y_pred=np.zeros(y_test.size),
							n_features=n_features
						)
					elif self.estimation_score == 'AIC':
						n_features = 0
						scores[bootstrap, lamb_idx] = -utils.AIC(
							y_true=y_test,
							y_pred=np.zeros(y_test.size),
							n_features=n_features
						)
					elif self.estimation_score == 'AICc':
						n_features = 0
						scores[bootstrap, lamb_idx] = -utils.AICc(
							y_true=y_test,
							y_pred=np.zeros(y_test.size),
							n_features=n_features
						)
					else:
						raise ValueError(str(self.estimation_score) + ' is not a valid option.')

		if verbose:
			print('(4) Bagging estimates, using bagging option %s.' % self.bagging_options)

		# bagging option 1: 
		#	for each bootstrap sample, find the regularization parameter that gave the best results
		if self.bagging_options == 1:
			self.lambda_max_idx = np.argmax(scores, axis=1)
			# extract the estimates over bootstraps from the model with best lambda
			best_estimates = estimates[np.arange(self.n_boots_est), self.lambda_max_idx, :]
			# take the median across estimates for the final, bagged estimate
			self.coef_ = np.median(best_estimates, axis=0)

		# bagging option 2: 
		#	average estimates across bootstraps, and then find the regularization parameter that gives the best results
		elif self.bagging_options == 2:
			mean_scores = np.mean(scores, axis=0)
			self.lambda_max_idx = np.argmax(mean_scores)
			self.coef_ = np.median(estimates[:, self.lambda_max_idx, :], 0)

		else:
			raise ValueError(
				'Bagging option %d is not available.' %self.bagging_options
			)

		if verbose:
			print("---> UoI Lasso complete.")

		self._set_intercept(X_offset, y_offset, X_scale)

		return self

	def intersection(self, estimates):
		# create support matrix
		self.supports_ = np.zeros((self.n_selection_thres, self.n_lambdas, self.n_features_), dtype=bool)

		# choose selection fraction threshold values to use
		selection_frac_thresholds = np.linspace(self.selection_thres_min, self.selection_thres_max, self.n_selection_thres)
		# calculate the actual number of thresholds, but delete any repetitions
		selection_thresholds = np.sort(np.unique((self.n_boots_sel * selection_frac_thresholds).astype('int')))

		# iterate over each stability selection threshold
		for thres_idx, threshold in enumerate(selection_thresholds):
			# calculate the support given the specific selection threshold
			self.supports_[thres_idx, :] = np.count_nonzero(estimates, axis=0) >= threshold

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
			return utils.BIC(n_features=n_features, n_samples=y.shape[0], rss=rss)
		else:
			raise ValueError('Incorrect metric specified.')

	def lasso_sweep(self, X, y, lambdas, train_frac, n_bootstraps,
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

		n_lambdas = len(lambdas)
		n_samples, n_features = X.shape
		n_train_samples = int(round(train_frac * n_samples))
		# create arrays to collect results
		estimates = np.zeros((n_bootstraps, n_lambdas, n_features), dtype=np.float32)
		scores = np.zeros((n_bootstraps, n_lambdas), dtype=np.float32)
		# apply the Lasso to bootstrapped datasets
		for bootstrap in trange(n_bootstraps, desc=desc, disable=not verbose):
			# for each bootstrap, we'll split the data into a randomly assigned training and test set
			train, test = utils.leveled_randomized_ids(self.groups_, train_frac)
			# iterate over the provided L1 penalty values
			for lamb_idx, lamb in enumerate(lambdas):
				# run the Lasso on the training set
				if not use_admm:
					# apply coordinate descent through the sklearn Lasso class
					lasso = lm.Lasso(
						alpha=lamb, 
						max_iter=10000, 
						fit_intercept=self.fit_intercept
					)
					lasso.fit(
						X[train], y[train] - y[train].mean()
					)
					estimates[bootstrap, lamb_idx, :] = lasso.coef_
				else:
					# apply ADMM using our utility function
					estimates[bootstrap, lamb_idx, :] = utils.lasso_admm(X[train], (y[train] - y[train].mean()), lamb=lamb)
				# run trained Lasso on the test set and obtain predictions
				y_hat = X[test].dot(estimates[bootstrap, lamb_idx, :])
				y_true = y[test] - y[test].mean()
				# calculate the explained variance using the predicted values
				scores[bootstrap, lamb_idx] = r2_score(y_true, y_hat)

		return estimates, scores
