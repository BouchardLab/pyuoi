import numpy as np
import sklearn.linear_model as lm

from sklearn.metrics import explained_variance_score


class UoI_Lasso():

	def __init__(X, y, train_frac_coarse=0.8, train_frac_dense=0.8,
				n_bootstraps_coarse = 10, n_minibatch = 10,
				n_lambdas=48, n_boots_sel=48, n_boots_est=48,
				use_admm = False):
		self.X = X
		self.y = y
		self.n_samples, self.n_features = X.shape
		self.train_frac_coarse = train_frac_coarse
		self.n_bootstraps_coarse = n_bootstraps_coarse
		self.n_minibatch = n_minibatch
		self.n_lambdas = n_lambdas
		self.n_boots_sel = n_boots_sel
		self.n_boots_est = n_boots_est
		self.use_admm = use_admm

	def fit():
		# perform an initial coarse sweep over the lambda parameters
		if n_lambas == 1:
			lambda_coarse = np.array([1.0])
		else:
			lambda_coarse = np.logspace(-3., 3., self.n_lambdas, dtype=np.float64)

		estimates_coarse, r_sq_coarse = self.lasso_sweep(self.X, self.y, lambda_coarse, self.train_frac_coarse, self.n_bootstraps_coarse, self.n_minibatch, self.use_admm)
		Mt = np.fix(1e4 * np.mean(r_sq_coarse, axis = 0))
		max_lambdas = np.argmax(np.ma.masked_invalid(Mt))
	Lids = np.where(Mt == np.max(np.ma.masked_invalid(Mt)))[0]
	v = lamb0[Lids[len(Lids)//2]]
	dv = 10**(np.floor(np.log10(v))-1)
		#######################
		### Model Selection ###
		#######################


		########################
		### Model Estimation ###
		########################


		return 

	@staticmethod
	def lasso_sweep(X, y, lambdas, train_frac, n_bootstraps, n_minibatch, use_admm):
	"""
	Parameters
	----------
	X: np.array
			
	y: np.array
	lambdas: np.array
	nboots: int
	m_frac: int
	with_admm: bool
	n_minibatch: int

	Returns
	-------
	B: np.array
		model coefficients (nboots, nfeatures, nlambdas)
	R2m: np.array
		boots, nlambdas
	"""
	# 
	n_lambdas = len(lambdas)
	n_samples, n_features = X.shape
	n_train_samples = int(round(train_frac * n_features))

	# create arrays to collect results
	estimates = np.zeros((n_bootstraps, n_lambdas, n_features), dtype=np.float32)
	r_sq = np.zeros((n_bootstraps, n_lambdas), dtype=np.float32)

	for bootstrap in range(n_bootstraps):
		# for each bootstrap, we'll split the data into a randomly assigned training and test set
		indices = np.random.permutation(n_samples)
		train, test = np.split(indices, [n_train_samples])
		# iterate over L1 penalty values
		for lamb_idx, lamb in enumerate(lambdas):
			# run the Lasso on the training set
			if not use_admm:
				try:
					lasso = lm.Lasso(alpha=lamb)
					lasso.fit(X[train], y[train] - y[train].mean())
				except:
					lasso = lm.SGDRegressor(penalty='l1', alpha=lamb)
					for batch_idx in range(n_minibatch):
						minibatch = range(batch_idx, n_samples, n_minibatch)
						lasso.partial_fit(X[minibatch],
										   y[minibatch]-y[minibatch].mean())
				estimates[bootstrap, lamb_idx, :] = lasso.coef_
			else:
				estimates[bootstrap, lamb_idx, :] = admm.lasso_admm(X[train],
								(y[train]-y[train].mean())[..., np.newaxis],
								alpha=lamb)[0]
			# run trained Lasso on the test set, and store the explained variance
			y_hat = X[test].dot(estimates[bootstrap, lamb_idx, :])
			y_true = y[test] - y[test].mean()
			r_sq[bootstrap, i] = explained_variance_score(y_true, y_hat)

	return estimates, r_sq