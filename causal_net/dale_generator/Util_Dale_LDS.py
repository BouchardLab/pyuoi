#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Util_Dale_LDS.py

This module contains functions for generating Dale-type connectivity matrices
and simulating neural activity from a linear dynamical system (LDS). These functions
have been refactored to require all parameters be passed from the calling program.


Output



Functions:
  gen_init_W(M, p, gamma, R, diag=0, rand=None)
    - Generates an initial network connectivity matrix.
    
  stabilize(A, max_iter=1000, eta=10)
    - Adjusts inhibitory weights to make the matrix stable (maximum real eigenvalue < 0).

  gen_matrices(reps, M, p, g, R, diag=-1, pickle_file=None)
    - Generates a nested list of matrices for a set of R values and a given number of repetitions.
      Optionally writes the result to a pickle file.
      
  gen_activity(W, tau, sigma, T, h, boxcox, num_trials, seed=None)
    - Simulates neural activity from an LDS defined by connectivity matrix W.
      Returns the integrated state trajectory and spike rate trials.
"""

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


#...!...!....................
def gen_matrices(reps, M, p, g, R, diag=-1):
    """
    Generate a nested list of matrices.
    
    Parameters:
      reps       : Number of repetitions to generate.
      M          : Number of excitatory neurons.
      p          : Synaptic connection probability.
      g          : Inhibitory scaling factor.
      R          : R values (spectral scaling value).
      diag       : Diagonal value to set initially (default -1).
      pickle_file: If provided, write the output to a pickle file.

      diag: setting these values (often to a negative number such as −1, you introduce
          a baseline decay or self-inhibition in each neuron. This helps ensure that, 
          in isolation, each neuron would naturally decay toward zero rather than explode.

    Returns:
      Alist: A nested list of matrices
                Alist has shape [reps] each element is a (2*M)x(2*M) matrix.
    """
    Alist = []
    for i in tqdm(range(reps), desc="Generating matrix repetitions"):
        A = gen_init_W(M, p, g, R, diag)
        eig = np.linalg.eigvals(A)
        if np.max(np.real(eig)) >= 0:
            A = stabilize(A)
            eig = np.linalg.eigvals(A)
        assert np.max(np.real(eig)) < 0, "Matrix is not stable after stabilization."
        Alist.append(A)
  
    return Alist

#################### Matrix generation ##################
#...!...!....................
def gen_init_W(M, p, gamma, R, diag=0, rand=None):
    """
    Generate an initial connectivity matrix for a Dale-type network.
    
    Parameters:
      M    : Number of excitatory neurons (the total number of neurons will be 2*M)
      p    : Synaptic connection probability
      gamma: Inhibitory scaling factor
      R    : A scaling constant for the weights
      diag : Value for the diagonal elements (typically negative)
      rand : A NumPy random number generator instance (if None, a default generator is used)
    
    Returns:
      Ainit: The generated (2*M) x (2*M) connectivity matrix.
    """
    if rand is None:
        rand = np.random.default_rng()

    Ainit = np.zeros((2 * M, 2 * M))
    w = R / np.sqrt(p * (1 - p) * (1 + gamma**2) / 2)

    # Excitatory connections
    for j in range(M):
        for k in range(2 * M):
            if rand.binomial(1, p):
                Ainit[j, k] = w / np.sqrt(2 * M)

    # Inhibitory connections
    for j in range(M):
        for k in range(2 * M):
            if rand.binomial(1, p):
                Ainit[j + M, k] = -gamma * w / np.sqrt(2 * M)

    # Set diagonal elements to diag (typically -1)
    np.fill_diagonal(Ainit, diag)
    return Ainit

#...!...!....................
def stabilize(A, max_iter=1000, eta=10):
    """
    Adjust the inhibitory weights of matrix A until its maximum real eigenvalue is negative.
    
    Optimize the inhibitory weights of a matrix A to render it stable (i.e. max re lambda < 0)
    Implements the algorithm described here: 
    https://epubs.siam.org/doi/abs/10.1137/070704034?journalCode=sjope8

    Uses an iterative algorithm described in the literature.
    https://www.sciencedirect.com/science/article/pii/S0896627314003602?via%3Dihub#app2

    Parameters:
      A       : The connectivity matrix to stabilize.
      max_iter: Maximum number of iterations.
      eta     : Learning rate for weight adjustment.
    
    Returns:
      A       : The stabilized connectivity matrix.
    """
    # Regularization constants from referenced publications
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
        grad = Q @ P / np.trace(Q @ P)
        
        # Adjust inhibitory weights
        inh_idx = np.argwhere(A < 0)
        for idx in inh_idx:
            A[idx[0], idx[1]] -= eta * grad[idx[0], idx[1]]
            if A[idx[0], idx[1]] > 0:
                A[idx[0], idx[1]] = 0  # Ensure inhibitory weights remain inhibitory
        
        alpha = np.max(np.real(np.linalg.eigvals(A)))
        iter_ += 1
    return A

#################### Simulation ##################
#...!...!....................
def gen_activity(W, tau, sigma, T, h, boxcox, num_trials, seed=None):
    """
    Generate neural activity from a linear dynamical system defined by connectivity matrix W.
    
    Parameters:
      W         : Connectivity matrix.
      tau       : Time constant for simulation.
      sigma     : Noise variance strength.
      T         : Total simulation time.
      h         : Integration time resolution.
      boxcox    : Box-Cox transformation parameter (set to None to return raw counts).
      num_trials: Number of spiking trials to simulate.
      seed      : Optional random seed.
    
    Returns:
      xt                : Integrated state trajectory over time.
      spike_rates_trials: Array of simulated spike rates for each trial.
    """
    if seed is not None:
        generator = np.random.default_rng(seed)
    else:
        generator = np.random.default_rng()
    
    # Define the system dynamics
    def f_(x, t):
        return 1/tau * (-np.eye(W.shape[0]) @ x + W @ x)
    
    # Define noise (diffusion term)
    def g_(x, t):
        return sigma * np.eye(W.shape[0])
    
    tspace = np.linspace(0, T, int(T/h))
    x0 = generator.normal(size=(W.shape[0],))
    print("Integrating LDS ...")
    xt = sdeint.itoSRI2(f_, g_, x0, tspace, generator=generator)
    
    print("Sampling spike counts ...")
    spike_rates_trials = []
    for _ in tqdm(range(num_trials), desc="Simulating trials"):
        spike_counts = np.random.poisson(np.exp(xt))
        if boxcox is not None:
            spike_rates = np.array([scipy.stats.boxcox(spike_count, boxcox) for spike_count in spike_counts])
        else:
            spike_rates = spike_counts
        spike_rates_trials.append(spike_rates)
    spike_rates_trials = np.array(spike_rates_trials)
    return tspace,xt, spike_rates_trials
