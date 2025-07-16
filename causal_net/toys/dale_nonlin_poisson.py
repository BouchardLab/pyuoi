#!/usr/bin/env python3

# uses non-liner discrete time evolution

import itertools
import numpy as np
import sdeint
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
# Synaptic connection probability
p = 0.25
# Diagonal time constants
g = 2
# Initial spectral absicca (larger initial values lead to more non-normal matrices)
R = np.linspace(0.75, 10, 25)[0:20]

tau = 3
# Noise variance strength
sigma = 1
# Number of time steps
T = 1000
# Integration time resolution
h = 1e-1

# Box-Cox transformation parameter 
# (Gaussianizes spike counts, set to None to get raw Poissonian counts)
boxcox = 0.5
# For each simulated firing rate trajectory, how many trials of spiking
# activity to sample?
num_trials = 10

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
def gen_matrices():    
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

    with open('Amats.pkl', 'wb') as f:
        f.write(pickle.dumps(Alist))
        f.write(pickle.dumps(R))

#################### Simulation ##################

def generate_poisson_var1(T=100, d=2, seed=42, A=None, b=None):
    """
    Generates a multivariate Poisson VAR(1) process:
        Y_t ~ Poisson(exp(A @ Y_{t-1} + b))

    Args:
        T (int): Number of time steps
        d (int): Number of dimensions (variables)
        seed (int): Random seed
        A (np.ndarray): d x d autoregressive coefficient matrix
        b (np.ndarray): d-dimensional intercept vector

    Returns:
        Y (np.ndarray): T x d time series of count data
    """
    np.random.seed(seed)
   
    # Initialize A and b if not given
    if A is None:
        A = np.random.uniform(-0.05/d, 0.1/d, size=(d, d))  # Keep A small for stability
    if b is None:
        b = np.random.uniform(-1.0/d, 1.0/d, size=(d,))
   
    Y = np.zeros((T, d), dtype=int)
    Y[0] = np.random.poisson(np.exp(b))  # initial state
   
    for t in range(1, T):
        eta = A @ Y[t-1] + b
        lambda_t = np.exp(np.clip(eta, -5, 5))  # avoid overflow
        Y[t] = np.random.poisson(lambda_t)

    return Y, A, b

if __name__ == '__main__':
    # Example usage

    # generate matrices
    gen_matrices()

    # load matrices
    with open('Amats.pkl', 'rb') as f:
        Alist = pickle.load(f)
        R = pickle.load(f)

    # A is a nested list with the first index being the repetition, 
    # the second being the initial spectral absicca
    A = Alist[0][0]
    print('M: A shape:',A.shape)

    #    Y, A, b = generate_poisson_var1(T=200, d=40)
    Y, A, b = generate_poisson_var1(T=T, d=2*M, A=A)
    print("A matrix:\n", A)
    print("Intercept vector b:\n", b)
    print("Sample data (first 5 rows):\n", Y)


    #np.set_printoptions(precision=3, suppress=True)
    iTrial=3
    nTime=100
    t0=20
    for i in range(5):
        iNeur=i*3
        print('\n iNeur=%d '%(iNeur))
        print('Yt:',Y[t0:t0+nTime,iNeur])
        
