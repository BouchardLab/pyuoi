#!/usr/bin/env python3

# uses non-liner discrete time evolution

import itertools
import numpy as np
import time
import pickle
import scipy
import os
import sys
import scipy.stats
from tqdm import tqdm

###### Matrix generation and simulation parameters ######

# How many repetitions of a given Dale LDS matrix to generate
reps = 1   # was 20
# Number of excitatory neurons (total neurons is twice this)
M = 10  # was 100
Nn=2*M
# Synaptic connection probability
p = 0.25
# Diagonal time constants
g = 2
# Initial spectral absicca (larger initial values lead to more non-normal matrices)
R = np.linspace(0.75, 10, 25)[0:20]


# Number of time steps
T = 4000
# Integration time resolution, in sec
dt = 0.07
# Idle rate range in Hz
Ri=np.array([2,5.1])

#################### Matrix generation ##################
# Generate an initial network connectivity matrix
def gen_init_W(M, p, gamma, R, diag=0, rand=None):
    if rand is None:
        rand = np.random.default_rng()

    Ainit = np.zeros((2 * M, 2 * M))

    w = R/np.sqrt(p * (1 - p) * (1 + gamma**2)/2)

    # Excitatory
    for j in range(M):
        for k in range(2 * M):
            if rand.binomial(1, p):
                Ainit[j, k] = w/np.sqrt(2 * M)


    # Inhibitory
    for j in range(M):
        for k in range(2 * M):
            if rand.binomial(1,p):
                Ainit[j + M, k] = -gamma * w/np.sqrt(2 * M)

    # Setting diagonals to 0 initially
    np.fill_diagonal(Ainit, diag)
    return Ainit

# Optimize the inhibitory weights of a matrix A to render it stable (i.e. max re lambda < 0)
# Implements the algorithm described here: https://epubs.siam.org/doi/abs/10.1137/070704034?journalCode=sjope8
def stabilize(A, max_iter=1000, eta=10):

    # Regularization of the spectral absicca, described on pg. 8 of the supplement here:
    # https://www.sciencedirect.com/science/article/pii/S0896627314003602?via%3Dihub#app2
    C = 1.5
    B = 0.2

    alpha = np.max(np.real(np.linalg.eigvals(A)))
    if alpha < 0:
        return A

    iter_ = 0

    while alpha > 0 and iter_ < max_iter:

        alpha_e = max(C * alpha, C * alpha + B)
        Q = scipy.linalg.solve_continuous_lyapunov((A - alpha_e * np.eye(A.shape[0])).T, -2 * np.eye(A.shape[0]))   
        P = scipy.linalg.solve_continuous_lyapunov(A - alpha_e * np.eye(A.shape[0]), -2 * np.eye(A.shape[0]))

        grad = Q @ P/np.trace(Q @ P)

        # Adjust inhibitory weights
        inh_idx = np.argwhere(A < 0)
        for idx in inh_idx:
            A[idx[0], idx[1]] -= eta * grad[idx[0], idx[1]]
            # Make sure no inhibitory weights got turned into excitatory weights
            if A[idx[0], idx[1]] > 0:
                A[idx[0], idx[1]] = 0
        
        alpha = np.max(np.real(np.linalg.eigvals(A)))
        iter_ += 1
    
    return A

# Generate the full set of matrices for use in subsequent synthetic experiments
def gen_dale_matrics():    
    Alist = []
    for i in tqdm(range(reps)):
        Alist.append([])
        for j, r in enumerate(R):
            A = gen_init_W(M, p, g, r, -1)
            eig = np.linalg.eigvals(A)
            if np.max(np.real(eig)) >= 0:
                A = stabilize(A)
                eig = np.linalg.eigvals(A)
            assert(np.max(np.real(eig)) < 0)
            Alist[i].append(A)
    return Alist[0][0]

#################### Simulation ##################

def generate_poisson_var1(T, dt, seed=None, A=None, verb=0):
    """
    Generates a multivariate Poisson VAR(1) process:
        Y_t ~ Poisson(exp(A @ Y_{t-1} + b))

    Args:
        T (int): Number of time steps
        dt (float):  time step in seconds
        d (int): Number of dimensions (variables)
        seed (int): Random seed
        A (np.ndarray): d x d autoregressive coefficient matrix
        B (np.ndarray): d-dimensional intercept vector

    Returns:
        Y (np.ndarray): T x d time series of count data
    """
    #np.random.seed(seed)
    d=A.shape[0]
    
    # convert idle rate to B
    Bi=np.log(Ri)
    print('Idle Ri:%s   Bi:%s'%(Ri,Bi))
    B = np.random.uniform(Bi[0],Bi[1], size=(d,))
    
    Y = np.zeros((T, d), dtype=int)
    Y[0] = np.random.poisson(np.exp(B)*dt)  # initial state
    if verb>0:
        print('B avr=%.1f  vec:%s'%(np.mean(B),B))
        print('exp(B) avr=%.1f  vec:%s'%(np.mean(np.exp(B)),np.exp(B)))
        print('t=%d  Y[t] sum=%d  vec:%s'%(0,np.sum(Y[0]),Y[0]))

    #lambda_t=np.array([3]*d) # Hz
    for t in range(1, T):
        eta = A @ Y[t-1] + B
        #eta = B  # use it to see idle rate only
        lambda_t = np.exp(np.clip(eta, -5, 5))  # avoid overflow        
        Y[t] = np.random.poisson(lambda_t*dt)
        if verb>0 and t<5: print('t=%d  Y[t] sum=%d  vec:%s'%(t,np.sum(Y[t]),Y[t]))

    return Y,A,B

def  eval_spikes_stats(Y):
    time_evol=T*dt
    fac=time_evol*Nn
    print('steps T=%d, time_evol=%.1f sec,  dt=%.4f sec,  nStep=%d fac=%.1f'%(T,time_evol,dt,Y.shape[0],fac))
    
    Ys=np.sum(Y,axis=0)
    print('eval spikes Ysum[neur]:',Ys,Y.shape)
    myVar=np.var(Y, ddof=1)/dt
    #myVar=np.var(Ys, ddof=1)/dt/Nn 
    myRate=np.sum(Ys)/fac
    print(' counts/sec/neuron  EV=%.1f  var=%.1f'%(myRate, myVar))
if __name__ == '__main__':
    # Example usage

    A=None
    if 1:    # generate matrices
        A=gen_dale_matrics()
        print('M:generated  A shape:',A.shape)
        Nn=2*M
        B = np.random.uniform(-1.0, 1.0, size=(Nn,))/100

    #    Y, A, b = generate_poisson_var1(T=200, d=40)
    Y, A, b = generate_poisson_var1(T=T, dt=dt, A=A,verb=1)
    #print("A matrix:\n", A)

    eval_spikes_stats(Y)
   
    
    np.set_printoptions(precision=3, suppress=True)
    iTrial=3
    nTime=100
    t0=0
    for i in range(5):
        iNeur=i*3
        print('\n iNeur=%d '%(iNeur))
        print('Yt:',Y[t0:t0+nTime,iNeur])
        
