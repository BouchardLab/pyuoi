#!/usr/bin/env python3
"""
This script simulates the activity of a recurrent neural network with biologically
inspired constraints. The key features of the simulation are:

1.  **Dale's Principle**: The network is composed of two populations of neurons:
    one purely excitatory and one purely inhibitory. This is enforced by the
    structure of the connectivity matrix `A`.

2.  **Network Stability**: The script ensures that the simulated network activity
    is stable (i.e., does not lead to runaway excitation) by numerically
    optimizing the inhibitory weights of the connectivity matrix `A` such that
    its eigenvalues have real parts less than zero.

3.  **Neural Dynamics**: The simulation uses a non-linear, discrete-time Poisson
    process to model spike generation. This is a form of a doubly stochastic
    Poisson process (or Cox process), where the firing rate of each neuron is
    dynamically influenced by the activity of the entire network at the previous
    time step. The core equation is:
    
        Y_t ~ Poisson(exp(A @ Y_{t-1} + B) * dt)

    where Y_t is the vector of spike counts at time t, A is the connectivity
    matrix, B is a bias/intercept term, and dt is the time step.

4.  **Configurability**: The script is highly configurable via command-line
    arguments, allowing for easy adjustment of network size, simulation duration,
    firing rates, and properties of the connectivity matrix like its initial
    spectral radius.

5.  **Analysis**: After the simulation, the script calculates and reports
    summary statistics for the generated spike trains, including mean firing
    rates and Fano factors, to characterize the network's activity.
"""
# uses non-linear discrete time evolution

import numpy as np
import time
import scipy
import os
import sys
import scipy.stats
import argparse

###### Matrix generation ##################
# Generate an initial network connectivity matrix
def gen_init_W(num_neurons, num_excite, p, gamma, R, diag=0, rand=None):
    if rand is None:
        rand = np.random.default_rng()

    num_inhib = num_neurons - num_excite
    Ainit = np.zeros((num_neurons, num_neurons))

    w = R/np.sqrt(p * (1 - p) * (1 + gamma**2)/2)

    # Excitatory neurons (rows 0 to num_excite-1)
    for j in range(num_excite):
        for k in range(num_neurons):
            if rand.binomial(1, p):
                Ainit[j, k] = w/np.sqrt(num_neurons)


    # Inhibitory neurons (rows num_excite to num_neurons-1)
    for j in range(num_inhib):
        for k in range(num_neurons):
            if rand.binomial(1,p):
                Ainit[j + num_excite, k] = -gamma * w/np.sqrt(num_neurons)

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
    num_neurons, num_excite, p, g, r = conf['num_neurons'], conf['num_excite'], conf['p'], conf['g'], conf['R']
    A = gen_init_W(num_neurons, num_excite, p, g, r, -1)
    eig = np.linalg.eigvals(A)
    if np.max(np.real(eig)) >= 0:
        A = stabilize(A, eta=conf['eta'], C=conf['C'], B=conf['B'])
        eig = np.linalg.eigvals(A)
    assert(np.max(np.real(eig)) < 0)
    return A

#################### Simulation ##################

def generate_poisson_var1(num_steps, dt, A, B_intercept, num_excite, seed=None, verb=0):
    """
    Generates a multivariate Poisson VAR(1) process:
        Y_t ~ Poisson(exp(A @ Y_{t-1} + B_intercept))

    Args:
        num_steps (int): Number of time steps for the simulation.
        dt (float):  Integration time step in seconds.
        A (np.ndarray): N x N autoregressive coefficient matrix (the Dale matrix).
        B_intercept (np.ndarray): N-dimensional intercept vector (bias).
        num_excite (int): Number of excitatory neurons.
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
    if verb>1:
        print('t=0  Y[t] sum=%d, Excit(first 3):%s, Inhib(first 3):%s'%(np.sum(Y[0]), Y[0][:3], Y[0][num_excite:num_excite+3]))

    kk=7
    # Main simulation loop
    for t in range(1, num_steps):
        eta = A @ Y[t-1] + B_intercept
        #eta = B_intercept  # use it to see idle rate only
        lambda_t = np.exp(np.clip(eta, -5, 5))  # avoid overflow        
        Y[t] = np.random.poisson(lambda_t*dt)
        if verb>0 and t<5:
            print('t=%d Y[t] sum=%d, Excit:%s, Inhib:%s'%(t, np.sum(Y[t]), Y[t][:kk], Y[t][num_excite:num_excite+kk]))

    return Y,A,B_intercept

def eval_spikes_stats(Y, dt, num_excite, mxNn=5):
    """Evaluates and prints statistics of the generated spike data."""
    num_steps_sim, Nn_sim = Y.shape
    num_inhib = Nn_sim - num_excite
    time_evol = num_steps_sim * dt
    print('steps num_steps=%d, time_evol=%.1f sec, Nn=%d (%d Excit, %d Inhib)' % (num_steps_sim, time_evol, Nn_sim, num_excite, num_inhib))

    spike_counts = np.sum(Y, axis=0)
    spike_rates = spike_counts / time_evol
    mean_counts_per_bin = np.mean(Y, axis=0)
    spike_variance = np.var(Y, axis=0)
    # Fano Factor can be undefined if mean is zero
    fano_factor = np.divide(spike_variance, mean_counts_per_bin, out=np.zeros_like(spike_variance), where=mean_counts_per_bin!=0)

    mxE = min(mxNn, num_excite)
    mxI = min(mxNn, num_inhib)

    print('\n--- Stats for first %d Excitatory Neurons ---' % mxE)
    np.set_printoptions(precision=2)
    print('Total Spike Counts:                   %s' % spike_counts[:mxE])
    print('Mean Firing Rate (Hz):                %s' % spike_rates[:mxE])
    print('Mean Spike Count per bin (dt=%.3fs): %s' % (dt, mean_counts_per_bin[:mxE]))
    print('Spike Count Variance per bin:         %s' % spike_variance[:mxE])
    print('Fano Factor (Var/Mean):               %s' % fano_factor[:mxE])

    if num_inhib > 0:
        print('\n--- Stats for first %d Inhibitory Neurons ---' % mxI)
        np.set_printoptions(precision=2)
        inhib_slice = slice(num_excite, num_excite + mxI)
        print('Total Spike Counts:                   %s' % spike_counts[inhib_slice])
        print('Mean Firing Rate (Hz):                %s' % spike_rates[inhib_slice])
        print('Mean Spike Count per bin (dt=%.3fs): %s' % (dt, mean_counts_per_bin[inhib_slice]))
        print('Spike Count Variance per bin:         %s' % spike_variance[inhib_slice])
        print('Fano Factor (Var/Mean):               %s' % fano_factor[inhib_slice])

    # --- Summary Stats ---
    print('\n--- Population Summary Statistics ---')
    # All neurons
    avg_rate_all = np.mean(spike_rates)
    std_rate_all = np.std(spike_rates)
    avg_fano_all = np.mean(fano_factor)
    std_fano_all = np.std(fano_factor)
    print('All    (%d neurons): Avg Rate=%.2f±%.2f Hz, Avg Fano=%.2f±%.2f' % (Nn_sim, avg_rate_all, std_rate_all, avg_fano_all, std_fano_all))

    # Excitatory neurons
    avg_rate_e = np.mean(spike_rates[:num_excite])
    std_rate_e = np.std(spike_rates[:num_excite])
    avg_fano_e = np.mean(fano_factor[:num_excite])
    std_fano_e = np.std(fano_factor[:num_excite])
    print('Excit (%d neurons): Avg Rate=%.2f±%.2f Hz, Avg Fano=%.2f±%.2f' % (num_excite, avg_rate_e, std_rate_e, avg_fano_e, std_fano_e))

    # Inhibitory neurons
    if num_inhib > 0:
        avg_rate_i = np.mean(spike_rates[num_excite:])
        std_rate_i = np.std(spike_rates[num_excite:])
        avg_fano_i = np.mean(fano_factor[num_excite:])
        std_fano_i = np.std(fano_factor[num_excite:])
        print('Inhib (%d neurons): Avg Rate=%.2f±%.2f Hz, Avg Fano=%.2f±%.2f' % (num_inhib, avg_rate_i, std_rate_i, avg_fano_i, std_fano_i))


def main():
    parser = argparse.ArgumentParser(description="Simulate a recurrent neural network with Dale's principle.")
    parser.add_argument("--num_neurons", type=int, default=30, help="Total number of neurons in the network.")
    parser.add_argument("--num_excite", type=int, default=20, help="Number of excitatory neurons.")
    parser.add_argument("--num_steps", type=int, default=8000, help="Number of time steps for simulation.")
    parser.add_argument("--step_size", type=float, default=0.01, help="Integration time step size (dt) in seconds.")
    parser.add_argument("--idleRate", type=float, nargs=2, default=[2.0, 15.1], help="Range of idle firing rates [min, max] in Hz.")
    parser.add_argument("--spectralR", type=float, default=2.5, help="Initial spectral radius (R).")
    parser.add_argument("--verb", type=int, default=1, help="Verbosity level (0=quiet, 1=normal).")
    args = parser.parse_args()

    if args.num_excite >= args.num_neurons:
        raise ValueError("Number of excitatory neurons must be less than the total number of neurons.")

    print("\nStarting simulation with configuration:")
    print(vars(args))
    print("")

    dale_conf = {
        'num_neurons': args.num_neurons,
        'num_excite': args.num_excite,
        'p': 0.25,  # Synaptic connection probability
        'g': 2,     # Inhibitory-to-excitatory synaptic strength ratio
        'R': args.spectralR,  # Initial spectral radius
        'eta': 10,  # Learning rate for stabilization algorithm
        'C': 1.5,   # Parameter for stabilization algorithm
        'B': 0.2    # Parameter for stabilization algorithm
    }
    
    # Nn is total number of neurons
    Nn = args.num_neurons

    # Generate the stable Dale connectivity matrix A
    print("Generating stable Dale matrix for Nn=%d (%d Excit, %d Inhib)..." % (Nn, args.num_excite, Nn - args.num_excite))
    A=gen_dale_matrics(dale_conf)
    print('Generated A shape:',A.shape)

    # Initialize bias vector B based on idle firing rate
    Ri_arg = np.array(args.idleRate)
    Bi = np.log(Ri_arg)
    print('Idle Ri:%s   Bi:%s'%(Ri_arg,Bi))
    B_intercept = np.random.uniform(Bi[0],Bi[1], size=(Nn,))
    if args.verb > 0:
        print('B_intercept avr=%.1f  vec:%s'%(np.mean(B_intercept),B_intercept))
        print('exp(B_intercept) avr=%.1f  vec:%s'%(np.mean(np.exp(B_intercept)),np.exp(B_intercept)))

    # Generate spike data using the Poisson VAR(1) process
    Y, A, b = generate_poisson_var1(num_steps=args.num_steps, dt=args.step_size, A=A, B_intercept=B_intercept, num_excite=args.num_excite, verb=args.verb)

    # Evaluate and print statistics of the simulated spikes
    eval_spikes_stats(Y, dt=args.step_size, num_excite=args.num_excite, mxNn=5)
   
    np.set_printoptions(precision=3, suppress=True)
    nTime=100
    t0=0
    # Print a sample of the spike trains for a few neurons
    print('\n--- Example Spike Trains ---')
    for i in range(5):
        iNeur=i*3
        if iNeur >= Nn: break
        print('\nNeuron %d:'%(iNeur))
        print('Spike counts (first %d steps): %s' % (nTime, Y[t0:t0+nTime,iNeur]))
        

if __name__ == '__main__':
    main() 
