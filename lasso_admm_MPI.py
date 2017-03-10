from __future__ import division
import pdb,time,h5py,os
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm,cholesky
from mpi4py import MPI
from optparse import OptionParser

"""
Author  : Alex Bujan (adapted from http://www.stanford.edu/~boyd)
Date    : 12/06/2015
"""

def main():
    usage = '%prog [options]'
    parser = OptionParser(usage)
    parser.add_option("-o",type="string",default=os.getcwd(),\
        help="hdf5 file to store the results")
    parser.add_option("-i",type="string",default='',\
        help="hdf5 file containing input data (X,y pairs)")
    parser.add_option("--alpha",type="float",default=.5,\
        help="scalar regularization parameter for Lasso")
    parser.add_option("--rho",type="float",default=1.,\
        help="augmented Lagrangian parameter")
    parser.add_option("--max_iter",type="int",default=50,\
        help="max number of ADMM iterations")
    parser.add_option("--abs_tol",type="float",default=1e-3,\
        help="absolute tolerance for early stopping")
    parser.add_option("--rel_tol",type="float",default=1e-2,\
        help="relative tolerance for early stopping")
    parser.add_option("--verbose",action='store_true',\
        dest='verbose',help="print information in the terminal")
    parser.add_option("--debug",action='store_true',\
        dest='debug',help="print information in the terminal")

    (options, args) = parser.parse_args()

    if options.verbose:
        verbose=True
    else:
        verbose=False

    if options.debug:
        debug=True
    else:
        debug=False

    lasso_admm(inputFile=options.i,outputFile=options.o,\
               alpha=options.alpha,rho=options.rho,verbose=verbose,\
               max_iter=options.max_iter,abs_tol=options.abs_tol,\
               rel_tol=options.rel_tol,debug=debug)

def lasso_admm(X,y,alpha=.5,rho=1.,verbose=True,\
                max_iter=50,abs_tol=1e-3,rel_tol=1e-2,debug=False,\
                comm=None):
    """Solve Lasso regression via ADMM


    Parameters
    ----------
    
    InputFile: string
        HDF5 file with following contents:
            - data/X : array,
                        design matrix (dimensions: n_samples,n_features)
            - data/y : target variable (samples)
    Returns
    -------
    
        - hdf5 output file with following values:
            - x      : solution of the Lasso problem (weights)
            - objval : objective value
            - r_norm : primal residual norm
            - s_norm : dual residual norm
            - eps_pri: tolerance for primal residual norm
            - eps_pri: tolerance for dual residual norm

    Notes
    -----
    
     Lasso problem:

       minimize 1/2*|| Ax - y ||_2^2 + alpha || x ||_1

    """

    '''
    MPI
    '''
    if comm==None:
        comm = MPI.COMM_WORLD
    
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    N = size

    if rank==0:
        print "\n Rank %i/%i"%(rank,size)

    '''
    Select sample block
    '''
#    X   = X[rank::N,:]
#    y   = y.ravel()[rank::N].reshape((m,1))

    m,n = X.shape
    y   = y.ravel().reshape((m,1))

    #save a matrix-vector multiply
    Xty = X.T.dot(y)

    #initialize ADMM solver
    x = np.zeros((n,1))
    z = np.zeros((n,1))
    u = np.zeros((n,1))
    r = np.zeros((n,1))

    send = np.zeros(3)
    recv = np.zeros(3)

    # cache the (Cholesky) factorization
    L,U = factor(X,rho)

    '''
    ADMM solver loop
    '''
    for k in xrange(max_iter):

        # u-update
        if k!=0:
            u+=(x-z)

        # x-update 
        q = Xty+rho*(z-u) #(temporary value)

        if m>=n:
            x = spsolve(U,spsolve(L,q))[...,np.newaxis]
        else:
            ULXq = spsolve(U,spsolve(L,X.dot(q)))[...,np.newaxis]
            x = (q*1./rho)-((X.T.dot(ULXq))*1./(rho**2))

        w = x + u

        send[0] = r.T.dot(r)[0][0]
        send[1] = x.T.dot(x)[0][0]
        send[2] = u.T.dot(u)[0][0]/(rho**2)

        zprev = np.copy(z)

        comm.Barrier()
        comm.Allreduce([w,MPI.DOUBLE],[z,MPI.DOUBLE])
        comm.Allreduce([send,MPI.DOUBLE],[recv,MPI.DOUBLE])

        # z-update
        z = soft_threshold(z*1./N,alpha*1./(N*rho))

        # diagnostics, reporting, termination checks
        objval      = objective(X,y,alpha,x,z)
        r_norm      = np.sqrt(recv[0])
        s_norm      = np.sqrt(N)*rho*norm(z-zprev)
        eps_pri     = np.sqrt(n*N)*abs_tol+\
                       rel_tol*np.maximum(np.sqrt(recv[1]),np.sqrt(N)*norm(z))
        eps_dual    = np.sqrt(n*N)*abs_tol+rel_tol*np.sqrt(recv[2])


        if r_norm<eps_pri and s_norm<eps_dual and k>0:
            break

        #Compute residual
        r = x-z

    return z

def objective(X,y,alpha,x,z):
    return .5*np.square(X.dot(x)-y).sum()+alpha*norm(z,1)

def factor(X,rho):
    m,n = X.shape
    if m>=n:
       L = cholesky(X.T.dot(X)+rho*sparse.eye(n))
    else:
       L = cholesky(sparse.eye(m)+1./rho*(X.dot(X.T)))
    L = sparse.csc_matrix(L)
    U = sparse.csc_matrix(L.T)
    return L,U

def soft_threshold(v,k):
    v[np.where(v>k)]-=k
    v[np.where(v<-k)]+=k
    v[np.intersect1d(np.where(v>-k),np.where(v<k))] = 0
    return v

if __name__=='__main__':
    main()
