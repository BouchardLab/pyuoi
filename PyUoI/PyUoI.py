import numpy as np

from PyUoI import UoI_Lasso

'''Wrapper class for all UoI methods.'''
class PyUoI():
	def __init__(self, method, use_MPI=False,
		# UoI Lasso
		n_lambdas=48, selection_thres_min=1., selection_thres_max=1., n_selection_thres=1.,
		train_frac_sel=0.8, train_frac_est=0.8, train_frac_overall=0.9, 
		n_boots_coarse=10, n_boots_sel=48, n_boots_est=48, 
		bagging_options=1, use_admm=False, estimation_score='BIC', 
		copy_X=True, fit_intercept=True, normalize=False
		# UoI VAR

		# UoI Python


	):
		if method == 'UoI_Lasso':
			if use_MPI:
				# store hyperparameters? and then run package when 
			else:
				self.uoi = UoI_Lasso(
					n_lambdas=n_lambdas, selection_thres_min=selection_thres_min, 
					selection_thres_max=selection_thres_max,
				)
		elif method == 'UoI_VAR':
			if use_MPI:
		else:
			raise ValueError('Specified UoI method %s does not exist.' %uoi_method)
