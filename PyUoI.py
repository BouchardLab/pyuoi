import numpy as np
import sklearn.linear_model as lm

from sklearn.metrics import explained_variance_score


class UoI_Lasso():
	def __init__(self, X, y, train_frac_sel=0.8,
				n_bootstraps_coarse = 10, n_minibatch = 10,
				n_lambdas=48, n_boots_sel=48, n_boots_est=48,
				use_admm = False):
		self.X = X
		self.y = y
		self.n_samples, self.n_features = X.shape
		self.train_frac_sel = train_frac_sel
		self.n_bootstraps_coarse = n_bootstraps_coarse
		self.n_minibatch = n_minibatch
		self.n_lambdas = n_lambdas
		self.n_boots_sel = n_boots_sel
		self.n_boots_est = n_boots_est
		self.use_admm = use_admm

	def fit(self):
		# perform an initial coarse sweep over the lambda parameters
		if self.n_lambdas == 1:
			lambda_coarse = np.array([1.0])
		else:
			lambda_coarse = np.logspace(-3., 3., self.n_lambdas, dtype=np.float64)
		# run the coarse lasso sweep
		estimates_coarse, explained_variance_coarse = self.lasso_sweep(self.X, self.y, lambda_coarse, self.train_frac_sel, self.n_bootstraps_coarse, self.n_minibatch, self.use_admm)
		# deduce the index which maximizes the explained variance over bootstraps
		lambda_max_idx = np.argmax(np.mean(explained_variance_coarse, axis=0))
		# obtain the lambda which maximizes the explained variance over bootstraps
		lambda_max = lambda_coarse[lambda_max_idx]
		# in our dense sweep, we'll explore lambda values which encompass a range that's one order of magnitude 
		# less than lambda itself
		d_lambda = 10**(np.floor(np.log10(lambda_max)) - 1)

		#######################
		### Model Selection ###
		#######################
		# now that we've narrowed down the regularization parameters, we'll run a dense sweep
		# which begins the model selection module of UoI

		# create new lambda set based on the coarse sweep
		if self.n_lambdas == 1:
			lambda_dense = np.array([lambda_max])
		else:
			lambda_dense = np.logspace(lambda_max - d_lambda, lambda_max + d_lambda, self.n_lambdas, dtype=np.float64)
		# run the lasso sweep with new lambda set
		estimates_dense, explained_variance_dense = self.lasso_sweep(self.X, self.y, lambda_dense, self.train_frac_sel, self.n_boots_sel, self.n_minibatch, self.use_admm)
		# intersect supports across bootstraps for each lambda value
		supports = np.all(estimates_dense, axis=0)

		########################
		### Model Estimation ###
		########################


		return 

	@staticmethod
	def lasso_sweep(X, y, lambdas, train_frac, n_bootstraps, n_minibatch, use_admm=False, seed=None):
		"""
		Parameters
		----------
		X : np.array
			data array containing regressors; assumed to be 2-d array with shape n_samples x n_features

		y : np.array
			data array containing dependent variable; assumed to be a 1-d array with length n_samples
		
		lambdas : np.array
			the set of regularization parameters to run boostraps over

		train_frac : float
			float between 0 and 1; the fraction of data to use for training

		n_bootstraps : int
			the number of bootstraps to obtain from the dataset; each bootstrap will undergo a Lasso regression
		
		n_minibatch : int
			number of minibatches to use in case SGD is used for the regression

		use_admm: bool
			switch to use the alternating direction method of multipliers (ADMM) algorithm

		Returns
		-------
		estimates : np.array
			predicted regressors for each bootstrap and lambda value; shape is (n_bootstraps, n_lambdas, n_features)
		
		explained_variance : np.array
			explained variance by the model for each bootstrap and lambda value; shape is (n_bootstraps, n_lambdas)
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
		explained_variance = np.zeros((n_bootstraps, n_lambdas), dtype=np.float32)
		# apply the Lasso to bootstrapped datasets
		for bootstrap in range(n_bootstraps):
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
							minibatch = range(batch_idx, n_samples, n_minibatch)
							lasso.partial_fit(X[minibatch],
											   y[minibatch]-y[minibatch].mean())
					estimates[bootstrap, lamb_idx, :] = lasso.coef_
				else:
					estimates[bootstrap, lamb_idx, :] = admm.lasso_admm(X[train],
									(y[train]-y[train].mean())[..., np.newaxis],
									alpha=lamb)[0]
				# run trained Lasso on the test set and obtain predictions
				y_hat = X[test].dot(estimates[bootstrap, lamb_idx, :])
				y_true = y[test] - y[test].mean()
				# calculate the explained variance using the predicted values
				explained_variance[bootstrap, lamb_idx] = explained_variance_score(y_true, y_hat)

		return estimates, explained_variance