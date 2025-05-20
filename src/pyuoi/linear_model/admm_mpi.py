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



def objective(X, y, alpha, x, z):
    if alpha == 0:
        return .5 * np.square(X.dot(x) - y).sum()
    else:
        return .5 * np.square(X.dot(x) - y).sum() + alpha * norm(z, 1)

def factor(X, rho):
    m, n = X.shape
    if m >= n:
       L = cholesky(X.T.dot(X) + rho * sparse.eye(n))
    else:
       L = cholesky(sparse.eye(m) + 1. / rho * (X.dot(X.T)))
    L = sparse.csc_matrix(L)
    U = sparse.csc_matrix(L.T)
    return L, U

def sparse_factor(X, rho):
    m, n = X.shape
    
    if m >= n:
        A = X.T.dot(X) + rho * sparse.eye(n)
    else:
        A = sparse.eye(m) + 1. / rho * (X.dot(X.T))
    
    # Use sparse LU(more general than Cholesky) factorization
    # Note: scipy returns a factorized object, not the L matrix directly
    factor = splu(A.tocsc())  # Convert to CSC for factorization
    
    return factor

def soft_threshold(v, k):
    v[np.where(v > k)] -= k
    v[np.where(v < -k)] += k
    v[np.intersect1d(np.where(v > -k), np.where(v < k))] = 0
    return v

class ADMM_Lasso:
    """
    MPI ADMM implementation of Linear Model trained with L1 prior as regularizer (aka the Lasso)
    
    The optimization objective for Lasso is:
    
    E(w) = 0.5 * ||y - Xw||^2_2 + alpha * ||w||_1
    
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
                 abs_tol=1e-3, rel_tol = 1e-2, warm_start=True, random_state=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.comm = comm
        self.coef_ = 0
    
    def fit(self, X= None, y = None, z = None, rho = None, sparse_input = True):
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
    
       
        '''
        Data
        '''
        if rank == 0:
            if rho is None:
                # heuristic rho selection
                if X.shape[1]<1e3:
                    rho = X.shape[0]/X.shape[1]
                else:
                    rho = 100/X.shape[1]
            #print(X.shape, rho, flush = True)
                
            m, n = X.shape
            
            # this is for accomdating the definition of regularization term in ADMM-LASSO convention
            alpha = self.alpha * m
        else:
            n = np.zeros(1).astype('int')
            m = np.zeros(1).astype('int')
            rho = np.zeros(1).astype('double')
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

        rho = comm.bcast(rho, root=0)
        alpha = comm.bcast(alpha, root=0)


    
        if rank != 0:
            # X = np.empty((m, n), dtype=np.float64)
            y = np.empty(m, dtype=np.float64)
    
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
    
        # do the send-receisve again for y?? or integrate back into the last send-receive operation?? or just Bcast it like right now
        y = np.ascontiguousarray(y.ravel()[rank::N].reshape((m, 1)))
    
    
    
        # save a matrix-vector multiply
        Xty = X.T.dot(y)

    
        # initialize ADMM solver

        if z is None:
        
            if self.warm_start and not np.all(self.coef_ == 0):
                z = deepcopy(self.coef_)
            else:
                z = np.random.normal(scale=0.1, size=(n, 1))
                #z = np.zeros((n, 1))
            
        x = deepcopy(z)


        u = np.zeros((n, 1))
        r = np.zeros((n, 1))
    
        send = np.zeros(3)
        recv = np.zeros(3)
    
        # cache the (Cholesky) factorization
        if sparse_input:
            factor_obj = sparse_factor(X, rho)
        else:
            L, U = factor(X, rho)
    

        objval = []
        r_norm = []
        s_norm = []
        eps_pri = []
        eps_dual = []
    
        '''
        ADMM solver loop
        '''
        for k in range(max_iter):  # xrange -> range for Python 3
    
            # u-update
            if k != 0:
                u += (x - z)
    
            # x-update 
            q = Xty + rho * (z - u)  # (temporary value)
    
            if sparse_input:
                if m >= n:
                    x = factor_obj.solve(q)
                else:
                    ULXq = factor_obj.solve(X.dot(q))
                    x = (q * 1. / rho) - ((X.T.dot(ULXq)) * 1. / (rho**2))                    
            else:
                if m >= n:  
                    x = spsolve(U, spsolve(L, q))[..., np.newaxis] 
                else:
                    ULXq = spsolve(U, spsolve(L, X.dot(q)))[..., np.newaxis]
                    x = (q * 1. / rho) - ((X.T.dot(ULXq)) * 1. / (rho**2))        
               
    
            w = x + u
    
            send[0] = r.T.dot(r)[0][0]
            send[1] = x.T.dot(x)[0][0]
            send[2] = u.T.dot(u)[0][0] / (rho**2)
    
            zprev = np.copy(z)
    
            comm.Barrier()
            comm.Allreduce([w, MPI.DOUBLE], [z, MPI.DOUBLE])
            comm.Allreduce([send, MPI.DOUBLE], [recv, MPI.DOUBLE])
    
            # z-update
            
            if alpha == 0:  #Linear regression case
                z = z * 1. / N
            else:
                z = soft_threshold(z * 1. / N, alpha * 1. / (N * rho))
    
            # diagnostics, reporting, termination checks
            objval.append(objective(X, y, alpha, x, z))
            # prires -> norm(x-z)
            r_norm.append(np.sqrt(recv[0]))
            # dualres -> norm(-rho*(z-zold))
            s_norm.append(np.sqrt(N) * rho * norm(z - zprev))
            eps_pri.append(np.sqrt(n * N) * abs_tol +
                           rel_tol * np.maximum(np.sqrt(recv[1]), np.sqrt(N) * norm(z)))
            eps_dual.append(np.sqrt(n * N) * abs_tol + rel_tol * np.sqrt(recv[2]))
    
    
            if r_norm[k] < eps_pri[k] and s_norm[k] < eps_dual[k] and k > 0:
                break
    
            # Compute residual
            r = x - z

        
        # Set attributes after fitting
        self.coef_ = z
        self.intercept_ = 0
        self.n_iter_ = None
        
        
        return self

    def set_params(self, alpha):
        self.alpha = alpha
    
    def predict(self, X):
        """
        Predict using the linear model.
        
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        y_pred = X @ self.coef_.ravel() + self.intercept_
        
        return y_pred
    
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