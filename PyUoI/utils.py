import pdb, time
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm, cholesky

"""
Author  : Alex Bujan (adapted from http://www.stanford.edu/~boyd)
Date    : 12/06/2015
"""

def BIC(n_features, n_samples, rss):
	"""Calculate the Bayesian Information Criterion under the assumption of 
	normally distributed disturbances (which allows the BIC to take on the
	simple form below).
	
	Parameters
	----------
	n_features : int
		number of model features

	n_samples : int
		number of samples in the dataset

	rss : float
		the residual sum of squares

	Returns
	-------
	BIC : float
		Bayesian Information Criterion
	"""
	BIC = -n_samples * np.log(rss/n_samples) - n_features * np.log(n_samples)
	return BIC

def lasso_admm(X, y, alpha, rho=1., rel_par=1., MAX_ITER=50, ABSTOL=1e-3, RELTOL=1e-2):
	"""
	 Solve lasso problem via ADMM
	
	 [z, history] = lasso_admm(X,y,alpha,rho,rel_par)
	
	 Solves the following problem via ADMM:
	
		 minimize 1/2*|| Ax - y ||_2^2 + alpha || x ||_1
	
	 The solution is returned in the vector z.
	
	 history is a dictionary containing the objective value, the primal and
	 dual residual norms, and the tolerances for the primal and dual residual
	 norms at each iteration.
	
	 rho is the augmented Lagrangian parameter.
	
	 rel_par is the over-relaxation parameter (typical values for rel_par are
	 between 1.0 and 1.8).
	
	 More information can be found in the paper linked at:
	 http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
	"""
	# Data preprocessing
	n_samples, n_features = X.shape
	# save a matrix-vector multiply
	Xy = np.dot(X.T, y).reshape((n_features, 1))

	# ADMM solver
	x = np.zeros((n_features, 1))
	z = np.zeros((n_features, 1))
	u = np.zeros((n_features, 1))

	# cache the (Cholesky) factorization
	L, U = factor(X, rho)

	for k in range(MAX_ITER):
		# x-update 
		q = Xy + rho * (z - u)  # (temporary value)
		if n_samples >= n_features:
			x = spsolve(U, spsolve(L, q)).reshape((n_features, 1))
		else:
			ULXq = spsolve(U, spsolve(L, X.dot(q)))
			x = (q * 1. / rho) - ((np.dot(X.T, ULXq)) * 1. / (rho ** 2))
		# z-update with relaxation
		zold = np.copy(z)
		x_hat = rel_par * x + (1. - rel_par) * zold
		z = shrinkage(x_hat + u, alpha * 1. / rho)

		# u-update
		u += (x_hat - z)

		# diagnostics, reporting, termination checks
		objval = objective(X, y, alpha, x, z)
		r_norm = norm(x - z)
		s_norm = norm(-rho * (z - zold))
		eps_pri = np.sqrt(n_features) * ABSTOL + RELTOL * np.maximum(norm(x), norm(-z))
		eps_dual = np.sqrt(n_features) * ABSTOL + RELTOL * norm(rho * u)

		if (r_norm < eps_pri) and (s_norm < eps_dual):
			break

	return z.ravel()


def objective(X, y, alpha, x, z):
	return .5 * np.sum(np.square(np.dot(X, x) - y)) + alpha * norm(z, 1)


def shrinkage(x, kappa):
	return np.maximum(0., x - kappa) - np.maximum(0., -x - kappa)


def factor(X, rho):
	n_samples, n_features = X.shape
	if n_samples >= n_features:
			L = cholesky(np.dot(X.T, X) + rho * sparse.eye(n_features))
	else:
			L = cholesky(sparse.eye(n_samples) + 1. / rho * (np.dot(X, X.T)))
	L = sparse.csc_matrix(L)
	U = sparse.csc_matrix(L.T)
	return L, U
