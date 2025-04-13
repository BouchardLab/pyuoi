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
M = 30  # was 100
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

def gen_activity(W, seed=None):
    if seed is not None:
        generator = np.random.default_rng(seed)
    else:
        generator = np.random.default_rng()
    # f
    def f_(x, t):
        return 1/tau * (-1 * np.eye(W.shape[0]) @ x + W @ x)

    # G: linear i.i.d noise with sigma
    def g_(x, t):
        return sigma * np.eye(W.shape[0])

    # Generate random initial condition and then integrate over the desired time period
    tspace = np.linspace(0, T, int(T/h))
    
    x0 = generator.normal(size=(W.shape[0],))
    print('Integrating LDS')
    xt = sdeint.itoSRI2(f_, g_, x0, tspace, generator=generator)    
    print('Sampling spike counts')
    spike_rates_trials = []
    for _ in tqdm(range(num_trials)):
        spike_counts = np.random.poisson(np.exp(xt))
        if boxcox is not None:
            spike_rates = np.array([scipy.stats.boxcox(spike_count, boxcox) for spike_count in spike_counts])
        else:
            spike_rates = spike_counts
        spike_rates_trials.append(spike_rates)
    spike_rates_trials = np.array(spike_rates_trials)
    return xt, spike_rates_trials

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
    xt, spike_rates_trials = gen_activity(A)
    print('xt:',xt.shape,xt.dtype)
    print('rate:',spike_rates_trials.shape,spike_rates_trials.dtype)

    np.set_printoptions(precision=3, suppress=True)
    iTrial=3
    nTime=50
    t0=5000
    for i in range(5):
        iNeur=i*10
        print('\n iNeur=%d '%(iNeur))
        print('xt:',xt[t0:t0+nTime,iNeur])
        print('rate:',spike_rates_trials[iTrial,t0:t0+nTime,iNeur])
