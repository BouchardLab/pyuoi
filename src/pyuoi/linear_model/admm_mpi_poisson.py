from __future__ import division
import pdb, time, h5py, os
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve, splu
from numpy.linalg import norm, cholesky
from mpi4py import MPI
from optparse import OptionParser
import gc
from copy import deepcopy
from .sparse_comm_util import *
from time import time


def adaptive_rho_update_boyd(r_res, s_res, rho, u, v, rho_scaler, imbalance_tolerance, n = None, p=None):
    
    if r_res > imbalance_tolerance * s_res:
        rho *= rho_scaler
        u /= rho_scaler
        v /= rho_scaler
    elif s_res > imbalance_tolerance * r_res:
        rho /= rho_scaler
        u *= rho_scaler
        v *= rho_scaler

    return rho, u, v 

def adaptive_rho_update(r_res, s_res, primal_eps, dual_eps, rho, u, v, rho_scaler, imbalance_tolerance, n = None, p=None):

    if r_res/primal_eps > imbalance_tolerance * s_res/dual_eps:
        rho *= rho_scaler
        u /= rho_scaler
        v /= rho_scaler
    elif s_res/dual_eps > imbalance_tolerance * r_res/primal_eps:
        rho /= rho_scaler
        u *= rho_scaler
        v *= rho_scaler

    return rho, u, v 
    

def objective(X, y, lamb, alpha, x, z, w):
    if alpha == 0:
        return 0
        # raise NotImplementedError("NO objectivefunction yet")
        #return .5 * np.square(X.dot(x) - y).sum()
    else:
        return 0
        #return .5 * np.square(X.dot(x) - y).sum() + alpha * norm(z, 1)



def sparse_factor(X, rho, lambda_2):
    m, n = X.shape
    
    if m >= n:
        A =  lambda_2*sparse.eye(n) + rho * (X.T.dot(X) + sparse.eye(n))
    else:
        raise NotImplementedError("Sparse factor not written for n_feature > n_sample")
        #A = sparse.eye(m) + 1. / rho * (X.dot(X.T))
    
    # Use sparse LU(more general than Cholesky) factorization
    # Note: scipy returns a factorized object, not the L matrix directly
    # try:
    factor = splu(A.tocsc())  # Convert to CSC for factorization
    # except Exception:
    #     print(A.shape, flush = True)
    
    
    return factor


def soft_threshold(v, k, mask):
    if mask is not None:
        mask = mask.reshape(-1, 1)
        k_masked = mask * k
    else: 
        k_masked = k

    return np.sign(v) * np.maximum(np.abs(v) - k_masked, 0.0)
    

def old_update_z(y, a, rho, tol=1e-3, max_iter=50):
    # Newton's method for convex scalar optimization(has overflow issue)
    z = a.copy()
    for i in range(len(y)):
        for _ in range(max_iter):
            exp_z = np.exp(z[i])
            grad = exp_z - y[i] + rho * (z[i] - a[i])
            hess = exp_z + rho
            step = grad / hess
            z[i] -= step
            if abs(step) < tol:
                break
    return z

def safe_exp(x):
    # Avoid overflow by capping
    return np.exp(np.clip(x, -100, 50))  # You can tune these bounds

def update_z(y, a, rho, w = 1, dt = 1, tol=1e-3, max_iter=10, eps = 1e-10):
    
    """
    Vectorized Newton solver for the z-update in Poisson regression ADMM.

    Args:
        y:  (n,) count data
        a:  (n,) target linear predictor (Xβ - u)
        w:  (n,) weights for each variable
        rho: scalar, penalty parameter
        tol: stopping tolerance
        max_iter: maximum Newton iterations

    Returns:
        z: updated vector (n,)
    """
    z = a.copy()  # Initial guess
    

    for i in range(max_iter):
        ez = safe_exp(z)
        stabilizer = (1+z)/(1+z+eps)
        
        grad = w * ez * dt - w * y * stabilizer + rho * (z - a)
        hess = w * ez * dt + rho
        step = grad / hess

        z_new = z - step

        
        if np.max(np.abs(step/z)) < tol:
            break

        z = z_new

    # print(i, flush = True)

    return z

def poisson_loss(x, beta, y, lamb):
    """
        Parameters
    ----------
    x : array, n_sample X n_features
        Design matrix/Predictor variable.
        
    beta : bool, default=True
        Model parameter estimates
        
    y : 1D array
        Predicted variable.

    """
    tse = np.sum(np.exp(x@beta)-y*(x@beta))
    l1_term = lamb * np.sum(np.abs(beta))
    
    return tse, l1_term

class ADMM_Poisson:
    """
    MPI ADMM implementation of Generalized Linear Model trained with Elastic-net regularizer 
    
    The optimization objective for Lasso is:
    
    E(w) = ***poisson stuff***0.5 * ||y - Xw||^2_2 + alpha (L1_regularizer * ||w||_1 +  (1-L1_regularizer)* ||w||_2)
    
    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1 term. alpha = 0 is equivalent to
        ordinary least squares. For numerical reasons, using alpha = 0 is
        not advised; instead, you should use Ridge or LinearRegression.
        
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to False,
        no intercept will be used in calculations.
        
    max_iter : int, default=1000
        The maximum number of iterations
        
    tol : float, default=1e-4
        The tolerance for the optimization.
        
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        
    processes: list
        the integer indices of MPI processes that are alotted for the MPI-ADMM fit
        
    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update.
        
    Attributes
    ----------
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        Parameter vector (w in the cost function formula).
        
    intercept_ : float | array, shape (n_targets,)
        Independent term in decision function.
        
    n_iter_ : int | array-like, shape (n_targets,)
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance.
    """
    
    def __init__(self, comm, rho = None, alpha=None, fit_intercept=False, max_iter=50,
                 abs_tol=1e-6, rel_tol = 1e-5,rho_scaler = 2, imbalance_tolerance = 10, warm_start=True, random_state=None, feature_weights = 1, dt = 1):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.comm = comm
        self.coef_ = 0
        self.rho_scaler = rho_scaler
        self.imbalance_tolerance = imbalance_tolerance
        self.feature_weights = feature_weights
        self.dt = dt
    
    def fit(self, X= None, y = None, w = None, sparse_input = True, param_mask = None):
        """
        Fit model with coordinate descent.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data. Only fed into the root rank.
            
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Only fed into the root rank.
        
            
        Returns
        -------
        self : object
            Returns self.
        """
        max_iter = self.max_iter
        abs_tol = self.abs_tol
        rel_tol = self.rel_tol

        comm = self.comm
        size = comm.Get_size()
        rank = comm.Get_rank()
        
    
        N = size

        # X_full = deepcopy(X)
        # y_full = deepcopy(y)
   
        '''
        Data
        '''
        if rank == 0:
            
            m, n = X.shape
            
           
            # scaled global alpha values, 
            # *m(number of samples in this bootstrap) accounts for the ADMM objective function being the total SSE
            # /N(n_admm) accounts for the change in L1-penalty scale when the SSE term optimization is distributed
            lamb = self.lamb * m / N 
            alpha = self.alpha
                
            weights = np.tile(self.feature_weights, (int(len(y)/len(self.feature_weights)), 1))
            weights = weights.T.flatten()

            loss = {}
            loss["tse"] = []
            loss["l1"] = []


        else:
            n = np.zeros(1).astype('int')
            m = np.zeros(1).astype('int')
            lamb = np.zeros(1).astype('double')
            alpha = np.zeros(1).astype('double')
            
    
        self.terminate_selection = False
        self.terminate_estimation = False
    
        n = comm.bcast(n, root=0)

        # using non-sensibel features number as termination signals
        if n == -1: 
            self.terminate_selection = True
            return self
        elif n == 0: 
            self.terminate_estimation = True
            return self            

        
        m = comm.bcast(m, root=0)


        alpha = comm.bcast(alpha, root=0)
        lamb = comm.bcast(lamb, root=0)


    
        if rank != 0:
            # X = np.empty((m, n), dtype=np.float64)
            y = np.empty(m, dtype=np.float64)
            weights = np.empty(m, dtype=np.float64)
            
    
        '''
        Broadcast data
        '''
        comm.Barrier()
        if rank == 0:
            for dest_rank in range(1, size):
    
                X_segment = X[dest_rank::N, :]
                 # since it's sparse csr matrix, don't have to Send the segment shape first, send the actual segment
                if sparse_input:
                    send_sparse_matrix(comm, X_segment, dest=dest_rank, tag = 11)
                else: 
                     # Send the segment shape first
                    segment_shape = np.array(X_segment.shape, dtype=np.int32)
                    comm.Send([segment_shape, MPI.INT], dest=dest_rank, tag=10)
                    
                    # Send the actual segment
                    comm.Send([np.ascontiguousarray(X_segment), MPI.DOUBLE], dest=dest_rank, tag=11)
            if sparse_input:
                X = X[0::N, :]
            else:
                X = np.ascontiguousarray(X[0::N, :])
    
            gc.collect()
        else: 
            if sparse_input:
                X = recv_sparse_matrix(comm, source=0, tag=11)
            else: 
                # First receive the shape of the incoming data
                segment_shape = np.empty(2, dtype=np.int32)
                comm.Recv([segment_shape, MPI.INT], source=0, tag=10)
                
                # Create a buffer to receive the segment
                X = np.empty(segment_shape, dtype=np.float64)
                
                # Receive the segment
                comm.Recv([X, MPI.DOUBLE], source=0, tag=11)
                
    
        '''
        Select sample block
        '''
        
        # X = X[rank::N, :]
        comm.Barrier()
        
        m, n = X.shape
        comm.Bcast([y, MPI.DOUBLE])
        comm.Bcast([weights, MPI.DOUBLE])
        param_mask = comm.bcast(param_mask, root=0) 

        # m is n_samp per ADMM process!
         # this is for accomdating the definition of MSE term in ADMM-LASSO convention
        #alpha *= m
        # good heuristric is to start with rho = l1-penalty

        # rho = lamb * alpha  / m 
        rho = lamb * alpha # admm parameter, modulating the constraint that aux variable equals the model variable
    
        # do the send-receive again for y?? or integrate back into the last send-receive operation?? or just Bcast it like right now
        y = np.ascontiguousarray(y.ravel()[rank::N].reshape((m, 1)))

        weights = np.ascontiguousarray(weights.ravel()[rank::N].reshape((m, 1)))
    

    
        # initialize ADMM solver
        if w is None: # w = None will be the default case!!!
            if self.warm_start and not np.all(self.coef_ == 0):
                w = deepcopy(self.coef_)
            else:
                np.random.seed(self.random_state)
                w = np.random.normal(scale=1, size=(n, 1))
        x = deepcopy(w)
        z = X.dot(x)
        
        u = np.zeros_like(z)
        v = np.zeros((n, 1))

    
        send = np.zeros(7)
        recv = np.zeros(7)

        

        
        # cache the (Cholesky) factorization
        if sparse_input:
            factor_obj = sparse_factor(X, rho, lamb * (1-alpha))
        else:
            L, U = factor(X, rho, lamb * (1-alpha))
    
        
        objval = []
        r_norm = []
        s_norm = []
        eps_pri = []
        eps_dual = []
    
        '''
        ADMM solver loop
        '''

        rho_history = []
        for k in range(max_iter):  # xrange -> range for Python 3

            if rank == 0 and k%10 == 0:
                tse, l1_term = poisson_loss(X, w, y, lamb)
                loss["tse"].append(tse)
                loss["l1"].append(l1_term)
    
            # x-update 
            q = rho * (X.T.dot(z + u) + (w + v))  # (temporary value)
    
            if sparse_input:
                if m >= n:
                    x = factor_obj.solve(q)
                else:
                    raise NotImplementedError()
                    # ULXq = factor_obj.solve(X.dot(q))
                    # x = (q * 1. / rho) - ((X.T.dot(ULXq)) * 1. / (rho**2))                    
            else:
                raise NotImplementedError()
                # if m >= n:  
                #     x = spsolve(U, spsolve(L, q))[..., np.newaxis] 
                # else:
                #     ULXq = spsolve(U, spsolve(L, X.dot(q)))[..., np.newaxis]
                #     x = (q * 1. / rho) - ((X.T.dot(ULXq)) * 1. / (rho**2))   

            
            # z-update (newton's method)
            a = X.dot(x) - u
            zprev = np.copy(z)
            z = update_z(y, a, rho, dt = self.dt, w = weights)
            
            w_input = x - v

            
    
            comm.Barrier()
            comm.Allreduce([w_input, MPI.DOUBLE], [w, MPI.DOUBLE]) # the resulting w is sum of N variants, so it has to be divided by N before all usage
            
            wprev = np.copy(w)
            
            # w-update(all proesses does the same computation)
            if lamb == 0:  #Linear regression case
                w  = w * 1. / N
            else:   
                w = soft_threshold(w * 1. / N, lamb * alpha * 1. / (N * rho), mask = param_mask)

            u += (z - X.dot(x)) # u is dual of z, so it's update is local
            v += (w - x)  # v is dual of w, so it's update should be global(but keep it local for now, since x is a vector that need to be Allreduced??!)                

            r1 = X.dot(x) - z
            r2 = x - w
            
            XTzz = X.T.dot(z-zprev)
            XTu = X.T.dot(u)
            Xx = X.dot(x)

            send[0] = (Xx).T.dot(Xx)[0][0]
            send[1] = z.T.dot(z)[0][0]
            send[2] = x.T.dot(x)[0][0]
            send[3] = XTzz.T.dot(XTzz)[0][0] 
            send[4] = XTu.T.dot(XTu)[0][0] 
            send[5] = v.T.dot(v)[0][0] 
            send[6] = r1.T.dot(r1)[0][0] + r2.T.dot(r2)[0][0]
            
            
            comm.Barrier()
            comm.Allreduce([send, MPI.DOUBLE], [recv, MPI.DOUBLE])
            
            r_res = np.sqrt(recv[6])
            s_res = rho * np.sqrt(recv[3] + N *  norm(w - wprev)**2)

            primal_eps = np.sqrt(n * N) * abs_tol + rel_tol * np.max(np.array([np.sqrt(recv[0]), np.sqrt(recv[1]), np.sqrt(recv[2]), np.sqrt(N) * norm(w)]))  
            dual_eps = np.sqrt(n * N) * abs_tol + rel_tol * rho * np.maximum(np.sqrt(recv[4]), np.sqrt(recv[5]))
            
    
            # diagnostics, reporting, termination checks
            objval.append(objective(X, y, lamb, alpha, x, z, w))
            r_norm.append(r_res)
            s_norm.append(s_res)
            eps_pri.append(primal_eps)
            eps_dual.append(dual_eps)
    
    
            if r_res < primal_eps and s_res < dual_eps and k > 0:
                break

            
            # adaptive rho selection based on residual
            #rho, u = adaptive_rho_update_boyd(r_res, s_res, rho, u, self.rho_scaler, self.imbalance_tolerance)
            if k%20 == 0:
                rho, u, v = adaptive_rho_update(r_res, s_res, primal_eps, dual_eps, rho, u, v, self.rho_scaler, self.imbalance_tolerance)

            

            if rank == 0:
                rho_history.append(rho)
        # if rank == 0:
        #     np.save("/global/homes/y/yxu2/packages/pyuoi/examples/rho_plot/rho_"+str(self.rho_scaler)+".npy", rho_history)
        
        # Set attributes after fitting
        self.coef_ = w   # we want w because it's a global variable, and converges to x in the limit
        self.intercept_ = 0
        self.n_iter_ = None
        if rank == 0:
            self.loss = loss

            
        return self

    def set_params(self, alpha, l1_ratio):
        #
        self.lamb = alpha
        self.alpha = l1_ratio
    
    def predict(self, X):
        """Predicts the response variable given a design matrix. The output is
        the mode of the Poisson distribution.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Design matrix to predict on.

        Returns
        -------
        mode : array_like, shape (n_samples)
            The predicted response values, i.e. the modes.
        """
        mu = self.predict_mean(X)

        mode = np.floor(mu)
        return mode

    def predict_mean(self, X):
        """Calculates the mean response variable given a design matrix.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Design matrix to predict on.

        Returns
        -------
        mu : array_like, shape (n_samples)
            The predicted response values, i.e. the conditional means.
        """
        mu = np.exp(np.clip(self.intercept_ + np.dot(X, self.coef_), -5, 5)) * self.dt
        
        return mu
        
    
    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.
            
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
            
        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        # Implementation would go here
        
        return None
    
    def _set_intercept(self, X_mean, y_mean, X_std):
        """
        Set the intercept based on the fit.
        """
        # Implementation would go here
        pass