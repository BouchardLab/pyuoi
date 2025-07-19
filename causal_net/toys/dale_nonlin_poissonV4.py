#!/usr/bin/env python3

# uses non-linear discrete time evolution

import numpy as np
import time
import scipy
import os
import sys
import scipy.stats
import argparse
from pprint import pprint

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
def stabilize(A, max_iter=1000, eta=10, C=1.5, B=0.2):

    # Regularization of the spectral absicca, described on pg. 8 of the supplement here:
    # https://www.sciencedirect.com/science/article/pii/S0896627314003602?via%3Dihub#app2
    
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
def gen_dale_matrics(conf):
    """Generates one stable Dale matrix based on configuration."""
    M, p, g, r = conf['M'], conf['p'], conf['g'], conf['R']
    A = gen_init_W(M, p, g, r, -1)
    eig = np.linalg.eigvals(A)
    if np.max(np.real(eig)) >= 0:
        A = stabilize(A, eta=conf['eta'], C=conf['C'], B=conf['B'])
        eig = np.linalg.eigvals(A)
    assert(np.max(np.real(eig)) < 0)
    return A

#################### Simulation ##################

def generate_poisson_var1(num_steps, dt, A, B_intercept, seed=None, verb=0):
    """
    Generates a multivariate Poisson VAR(1) process:
        Y_t ~ Poisson(exp(A @ Y_{t-1} + B_intercept))

    Args:
        num_steps (int): Number of time steps for the simulation.
        dt (float):  Integration time step in seconds.
        A (np.ndarray): N x N autoregressive coefficient matrix (the Dale matrix).
        B_intercept (np.ndarray): N-dimensional intercept vector (bias).
        seed (int): Random seed (currently not used).
        verb (int): Verbosity level for printing progress.

    Returns:
        (Y, A, B_intercept): Tuple containing:
            Y (np.ndarray): num_steps x N time series of spike counts.
            A (np.ndarray): The input connectivity matrix.
            B_intercept (np.ndarray): The input intercept vector.
    """
    if A is None:
        raise ValueError("Connectivity matrix A cannot be None.")
    d=A.shape[0]
    
    Y = np.zeros((num_steps, d), dtype=int)
    Y[0] = np.random.poisson(np.exp(B_intercept)*dt)  # initial state
    if verb>0:
        print('t=%d  Y[t] sum=%d  vec:%s'%(0,np.sum(Y[0]),Y[0]))

    # Main simulation loop
    for t in range(1, num_steps):
        eta = A @ Y[t-1] + B_intercept
        #eta = B_intercept  # use it to see idle rate only
        lambda_t = np.exp(np.clip(eta, -5, 5))  # avoid overflow        
        Y[t] = np.random.poisson(lambda_t*dt)
        if verb>0 and t<5: print('t=%d  Y[t] sum=%d  vec:%s'%(t,np.sum(Y[t]),Y[t]))

    return Y,A,B_intercept

def eval_spikes_stats(Y, dt, mxNn=10):
    """Evaluates and prints statistics of the generated spike data."""
    num_steps_sim, Nn_sim = Y.shape
    time_evol = num_steps_sim * dt
    print('steps num_steps=%d, time_evol=%.1f sec, dt=%.4f sec, Nn=%d' % (num_steps_sim, time_evol, dt, Nn_sim))

    spike_counts = np.sum(Y, axis=0)
    spike_rates = spike_counts / time_evol
    mean_counts_per_bin = np.mean(Y, axis=0)
    spike_variance = np.var(Y, axis=0)
    # Fano Factor can be undefined if mean is zero
    fano_factor = np.divide(spike_variance, mean_counts_per_bin, out=np.zeros_like(spike_variance), where=mean_counts_per_bin!=0)

    
    print('\n--- Spike Stats per Neuron (showing first %d)---' % mxNn)
    np.set_printoptions(precision=2)
    print('Total Spike Counts: %s' % spike_counts[:mxNn])
    print('Mean Firing Rate (Hz):                %s' % spike_rates[:mxNn])
    print('Mean Spike Count per bin (dt=%.3fs): %s' % (dt, mean_counts_per_bin[:mxNn]))
    print('Spike Count Variance per bin:         %s' % spike_variance[:mxNn])
    print('Fano Factor (Var/Mean):               %s' % fano_factor[:mxNn])


    total_avg_rate = np.mean(spike_rates)
    print('\nAverage firing rate across all neurons: %.2f Hz' % total_avg_rate)

def main():
    parser = argparse.ArgumentParser(description="Simulate a recurrent neural network with Dale's principle.")
    parser.add_argument("--num_excit_neurons", type=int, default=20, help="Number of excitatory neurons (M). Total neurons will be 2*M.")
    parser.add_argument("--num_steps", type=int, default=4000, help="Number of time steps for simulation.")
    parser.add_argument("--step_size", type=float, default=0.01, help="Integration time step size (dt) in seconds.")
    parser.add_argument("--idleRate", type=float, nargs=2, default=[5.0, 10.1], help="Range of idle firing rates [min, max] in Hz.")
    parser.add_argument("--verb", type=int, default=1, help="Verbosity level (0=quiet, 1=normal).")
    args = parser.parse_args()

    dale_conf = {
        'M': args.num_excit_neurons,
        'p': 0.25,  # Synaptic connection probability
        'g': 2,     # Inhibitory-to-excitatory synaptic strength ratio
        'R': 2.5,  # Initial spectral radius
        'eta': 10,  # Learning rate for stabilization algorithm
        'C': 1.5,   # Parameter for stabilization algorithm
        'B': 0.2    # Parameter for stabilization algorithm
    }

    print("\nStarting simulation with configuration:")
    print(vars(args))
    print("\n dale_conf")
    pprint(dale_conf)
    np.set_printoptions(precision=3, suppress=True)
    
    
    # Nn is total number of neurons (excitatory + inhibitory)
    Nn = 2 * dale_conf['M']

    # Generate the stable Dale connectivity matrix A
    print("Generating stable Dale matrix for Nn=%d..."%Nn)
    A=gen_dale_matrics(dale_conf)
    print('Generated A shape:',A.shape)

    # Initialize bias vector B based on idle firing rate
    Ri_arg = np.array(args.idleRate)
    Bi = np.log(Ri_arg)
    print('Idle Ri:%s   Bi:%s'%(Ri_arg,Bi))
    B_intercept = np.random.uniform(Bi[0],Bi[1], size=(Nn,))
    if args.verb > 0:
        print('B_intercept avr=%.1f  vec:%s ...'%(np.mean(B_intercept),B_intercept[:4]))
        print('exp(B_intercept) avr=%.1f  vec:%s ...'%(np.mean(np.exp(B_intercept)),np.exp(B_intercept[:4])))

    # Generate spike data using the Poisson VAR(1) process
    Y, A, b = generate_poisson_var1(num_steps=args.num_steps, dt=args.step_size, A=A, B_intercept=B_intercept, verb=args.verb)

    # Evaluate and print statistics of the simulated spikes
    eval_spikes_stats(Y, dt=args.step_size, mxNn=10)
   
    np.set_printoptions(precision=3, suppress=True)
    nTime=100
    t0=0
    # Print a sample of the spike trains for a few neurons
    print('\n--- Example Spike Trains ---')
    for i in range(3):
        iNeur=i*3
        if iNeur >= Nn: break
        print('\nNeuron %d:'%(iNeur))
        print('Spike counts (first %d steps): %s' % (nTime, Y[t0:t0+nTime,iNeur]))
        

if __name__ == '__main__':
    main() 
