#!/usr/bin/env python3
"""
python gen_dalePoisson.py --num_neurons 10 --num_excite 6 --num_steps 1000 --dataName test_dale


This script simulates the activity of a recurrent neural network with biologically
inspired constraints. The key features of the simulation are:

***Dale Poisson Simulator - Program Summary***
This program simulates the activity of a recurrent neuronal network using a discrete-time
Poisson generalized linear model (GLM). The network obeys Dale's principle, meaning that
each neuron is either excitatory (producing only positive outgoing weights) or inhibitory
(producing only negative outgoing weights).

Key Features:

Connectivity Matrix Generation:

The recurrent connectivity matrix A is generated with random weights that respect Dale's principle.
The parameter R scales the baseline synaptic strength and influences the initial spectral radius of A.
Regularization of A:

A stability check is performed by computing the maximum real part of the eigenvalues.
Inhibitory weights (A_ij < 0) are iteratively adjusted using a gradient derived from Lyapunov equations.
Updates continue until the maximum real eigenvalue falls below a threshold (-delta), ensuring stable dynamics.
Poisson Process Simulation:

At each time step t, the firing rate for each neuron is computed as: lambda(t, i) = exp( sum_j A_ij * Y(t-1, j) + B_i )
Spike counts Y(t, i) are drawn from a Poisson distribution with rate lambda(t, i) * dt.
Additional customization is provided via command-line arguments, allowing you to set the
number of neurons, number of excitatory neurons, spectral radius (R), time step (dt), and
simulation duration. The program saves the generated spike data, firing rates, connectivity
matrix, and other simulation details for further analysis.

"""

import numpy as np
import time,hashlib
import scipy
import os
import sys

import argparse
from pprint import pprint
from PlotterSimPoisson import Plotter
from UtilDalePoisson import estimate_rates, geom_edges_mask, true_edges_mask
from toolbox.Util_NumpyIO import write_data_npz

###### Matrix generation ##################
# Generate an initial network connectivity matrix
def gen_init_W(num_neurons, num_excite, rho_target, gamma, R, varyW, minW,diag=0):
    print(f"  Generating initial connectivity matrix: N={num_neurons}, excit={num_excite}, gamma={gamma:.1f}, R={R:.1f}")
    rand = np.random.default_rng()

    num_inhib = num_neurons - num_excite
    Ainit = np.zeros((num_neurons, num_neurons))
    
    # this scaling is a guess, may need adjustment
    p_eff = np.mean(rho_target) / num_neurons
    wC = R/np.sqrt(p_eff * (1 - p_eff) * (1 + gamma**2)/2)  # central value
    
    # decide how much variation in  weights
    assert varyW<1
    wL=wC*varyW  / np.sqrt(num_neurons)
    wR=wC/varyW  / np.sqrt(num_neurons)
    if wL< minW:  # shift both by the difference
        delW=minW-wL
        wL+=delW
        wR+=delW
    print(f"    Weight parameters: wL={wL:.3f}, wR={wR:.3f}, p_eff={p_eff:.3f}")
    
    for j in range(num_neurons):
        # Determine the number of connections for this neuron
        num_connections = rho_target[j]
        if num_connections == 0:
            continue
        
        # Choose target neurons to connect to
        targets = rand.choice(num_neurons, num_connections, replace=False)
        
        # Generate random weights for these connections
        weights = np.random.uniform(wL, wR, size=num_connections)
     
        if j < num_excite: # Excitatory neuron
            Ainit[j, targets] = weights
        else: # Inhibitory neuron
            Ainit[j, targets] = -gamma * weights

    # Setting diagonals to 0 initially
    np.fill_diagonal(Ainit, diag)
    excit_connections = np.sum(Ainit[:num_excite,:] > 1e-9)
    inhib_connections = np.sum(Ainit[num_excite:,:] < -1e-9)

    print(f"    Connections created: excit={excit_connections}, inhib={inhib_connections}")
    print(f"    Matrix stats: min={np.min(Ainit):.3f}, max={np.max(Ainit):.3f}, mean={np.mean(Ainit):.3f}")
    return Ainit

# Optimize the inhibitory weights of a matrix A to render it stable (i.e. max re lambda < 0)
# Implements the algorithm described here: https://epubs.siam.org/doi/abs/10.1137/070704034?journalCode=sjope8
def stabilize(A, max_iter=3000, eta=50, C=3.0, B=1.0,delta=0.2,minW=0.1):
    print(f"  Stabilizing matrix: max_iter={max_iter}, eta={eta}, C={C}, B={B}, delta={delta}")
    
    # delta: limits real part of eigen values
    # Regularization of the spectral absicca, described on pg. 8 of the supplement here:
    # https://www.sciencedirect.com/science/article/pii/S0896627314003602?via%3Dihub#app2
    # delat is my margin from 0 toward negative
    
    alpha = np.max(np.real(np.linalg.eigvals(A)))
    print(f"    Initial spectral radius: {alpha:.3f}")
    if alpha < -delta:
        print(f"    Matrix already stable (alpha={alpha:.3f} < -delta={-delta:.3f})")
        return A

    iter_ = 0

    while alpha > -delta and iter_ < max_iter:
        if iter_ % 100 == 0:
            print(f"    Iteration {iter_}: alpha={alpha:.3f}")

        alpha_e = max(C * alpha, C * alpha + B)
        Q = scipy.linalg.solve_continuous_lyapunov((A - alpha_e * np.eye(A.shape[0])).T, -2 * np.eye(A.shape[0]))   
        P = scipy.linalg.solve_continuous_lyapunov(A - alpha_e * np.eye(A.shape[0]), -2 * np.eye(A.shape[0]))

        grad = Q @ P/np.trace(Q @ P)

        # Adjust inhibitory weights
        inh_idx = np.argwhere(A < 0)
        weight_changes = 0
        for idx in inh_idx:
            old_weight = A[idx[0], idx[1]]
            A[idx[0], idx[1]] -= eta * grad[idx[0], idx[1]]
            # Make sure no inhibitory weights got turned into excitatory weights
            #if A[idx[0], idx[1]] > 0:
            if A[idx[0], idx[1]] > -minW:  # prevents too small weight for inhibitory cells
                A[idx[0], idx[1]] = 0
            if abs(A[idx[0], idx[1]] - old_weight) > 1e-6:
                weight_changes += 1
        
        alpha = np.max(np.real(np.linalg.eigvals(A)))
        iter_ += 1
        
        if iter_ % 500 == 0:
            print(f"      Weight changes: {weight_changes}/{len(inh_idx)}")
    
    print(f"    Stabilization complete after {iter_} iterations: final alpha={alpha:.3f}")
    return A

# Generate the full set of matrices for use in subsequent synthetic experiments
def gen_dale_matrics(conf, rho_target):
    """Generates one stable Dale matrix based on configuration."""
    print(f"\n=== Generating Dale Matrix ===")
    num_neurons, num_excite, g, r = conf['num_neurons'], conf['num_excite'], conf['g'], conf['R']
    print(f"Parameters: N={num_neurons}, excit={num_excite}, g={g:.1f}, r={r:.1f}")

    # additional configuration
    varyW=0.6  #  controls variation of excitatory weights
    minW=0.1  # sets minimal value of any weights, also after stabilization
    eigenGap=0.1 # sets threshold on Re(eigen value) after stabilization
    
    A = gen_init_W(num_neurons, num_excite, rho_target, g, r, varyW=varyW, minW=minW, diag=-1)
    eig = np.linalg.eigvals(A)
    print(f"Initial eigenvalues: real range [{np.min(np.real(eig)):.3f}, {np.max(np.real(eig)):.3f}]")
    
    if np.max(np.real(eig)) >= 0:
        A = stabilize(A, eta=conf['eta'], C=conf['C'], B=conf['B'],delta=eigenGap,minW=minW)
        eig = np.linalg.eigvals(A)
        print(f"After stabilization: real range [{np.min(np.real(eig)):.3f}, {np.max(np.real(eig)):.3f}]")
    else:
        print("Matrix already stable")
    
    assert(np.max(np.real(eig)) < 0)
    
    # Count non-zero excitatory and inhibitory weights (off-diagonal only)
    num_inhib = num_neurons - num_excite
    
    # Create diagonal mask
    diag_mask = np.eye(num_neurons, dtype=bool)
    off_diag_mask = ~diag_mask
    
    # Excitatory weights (rows 0 to num_excite-1, off-diagonal only)
    excit_weights = A[:num_excite, :]
    excit_off_diag = excit_weights[off_diag_mask[:num_excite, :]]
    excit_nonzero = np.sum(np.abs(excit_off_diag) > 1e-10)
    excit_total = excit_off_diag.size
    
    # Inhibitory weights (rows num_excite to num_neurons-1, off-diagonal only)
    inhib_weights = A[num_excite:, :]
    inhib_off_diag = inhib_weights[off_diag_mask[num_excite:, :]]
    inhib_nonzero = np.sum(np.abs(inhib_off_diag) > 1e-10)
    inhib_total = inhib_off_diag.size
    
    print(f'Dale matrix weights (off-diagonal): Excitatory {excit_nonzero}/{excit_total} ({excit_nonzero/excit_total*100:.1f}%), Inhibitory {inhib_nonzero}/{inhib_total} ({inhib_nonzero/inhib_total*100:.1f}%)')
    
    return A

#################### Simulation ##################

def generate_poissonV5(num_steps, dt, A, B_intercept, num_excite, verb=0):
    """
    Generates a multivariate Poisson VAR(1) process:
        Y_t ~ Poisson(exp(A @ Y_{t-1} + B_intercept))

    Args:
        num_steps (int): Number of time steps for the simulation.
        dt (float):  Integration time step in seconds.
        A (np.ndarray): N x N autoregressive coefficient matrix (the Dale matrix).
        B_intercept (np.ndarray): N-dimensional intercept vector (bias).
        num_excite (int): Number of excitatory neurons.
        verb (int): Verbosity level for printing progress.

    Returns:
        (Y, A, B_intercept): Tuple containing:
            Y (np.ndarray): num_steps x N time series of spike counts.
            A (np.ndarray): The input connectivity matrix.
            B_intercept (np.ndarray): The input intercept vector.
    """
    print(f"\n=== Generating Poisson  Process ===")
    print(f"Simulation parameters: steps={num_steps}, dt={dt:.3f}, neurons={A.shape[0]}, excit={num_excite}")
    print(f"Matrix A stats: min={np.min(A):.3f}, max={np.max(A):.3f}, mean={np.mean(A):.3f}")
    print(f"Bias B stats: min={np.min(B_intercept):.3f}, max={np.max(B_intercept):.3f}, mean={np.mean(B_intercept):.3f}")
    
    if A is None:
        raise ValueError("Connectivity matrix A cannot be None.")
    d=A.shape[0]
    
    Y = np.zeros((num_steps, d), dtype=int)
    Y[0] = np.random.poisson(np.exp(B_intercept)*dt)  # initial state
    print(f"Initial state: total spikes={np.sum(Y[0])}, excit spikes={np.sum(Y[0][:num_excite])}, inhib spikes={np.sum(Y[0][num_excite:])}")
    
    if verb>0:
        print('t=0 Y[t] sum=%d, Excit(first 3):%s, Inhib(first 3):%s'%(np.sum(Y[0]), Y[0][:3], Y[0][num_excite:num_excite+3]))

    kk=15
    # Main simulation loop
    print("Starting main simulation loop...")
    for t in range(1, num_steps):
        eta = A @ Y[t-1] + B_intercept
        #eta = B_intercept  # use it to see idle rate only
        lambda_t = np.exp(np.clip(eta, -5, 5))  # avoid overflow        
        Y[t] = np.random.poisson(lambda_t*dt)
        
        if verb>0 and t<5:
            print('t=%d Y[t] sum=%d, Excit:%s, Inhib:%s'%(t, np.sum(Y[t]), Y[t][:kk], Y[t][num_excite:num_excite+kk]))
        
        if t % (num_steps // 10) == 0:
            print(f"  Progress: {t}/{num_steps} steps ({t/num_steps*100:.1f}%) -  total spikes in this step: {np.sum(Y[t])}")

    print(f"Simulation complete. Final state: total spikes={np.sum(Y[-1])}")
    print(f"Spike data stats: min={np.min(Y)}, max={np.max(Y)}, mean={np.mean(Y):.2f}, total spikes={np.sum(Y)}")
    
    return Y

#########################
#  MAIN
#########################

def main():
    print("=" * 60)
    print("DALE POISSON SIMULATION V5")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="Simulate a recurrent neural network with Dale's principle.")
    parser.add_argument("--num_neurons", type=int, default=50, help="Total number of neurons in the network.")
    parser.add_argument("--num_excite", type=int, default=30, help="Number of excitatory neurons.")
    parser.add_argument("--edge_prob", type=float, nargs=2, default=[0.05, 0.2], help="Range of edge probability [min, max] for rho_target generation.")
    parser.add_argument("--num_steps", type=int, default=10_000, help="Number of time steps for simulation.")
    parser.add_argument("--step_size", type=float, default=0.01, help="Integration time step size (dt) in seconds.")
    parser.add_argument("--idleRate", type=float, nargs=2, default=[2, 10.], help="Range of idle firing rates [min, max] in Hz.")
    parser.add_argument("--expRate", action="store_true", help="Switch to exponentially decausing rate")
    parser.add_argument("--spectralR", type=float, default=2.0, help="Initial spectral radius (R).")
    parser.add_argument("--verb", type=int, default=1, help="Verbosity level (0=quiet, 1=normal).")
    parser.add_argument("--dataName", type=str, default=None, help="Base name for output files (default: dale_spikes_xx).")
    parser.add_argument("--outPath", type=str, default='/pscratch/sd/b/balewski/2025_causalNet_tmp/', help="Output directory for all files.")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting") 
    parser.add_argument("-p", "--showPlots",  default='a b', nargs='+',help="abc-string listing shown plots: a=Dale_matrix_and_eigen, b=histo_true_weights, c=rate_analysis")

    np.set_printoptions(precision=3, suppress=True)

    args = parser.parse_args()
    args.showPlots=''.join(args.showPlots)
    # Determine output file prefix
    if args.dataName is None:
        args.dataName='daleM%d_'%args.num_neurons+hashlib.md5(os.urandom(32)).hexdigest()[:6]
        
    
    print("\nStarting simulation with configuration:")
    print(vars(args))

    # Validation checks
    print("Performing parameter validation...")
    if args.num_excite >= args.num_neurons:
        raise ValueError("Number of excitatory neurons must be less than the total number of neurons.")

    assert os.path.exists(args.outPath)
    print("Output directory exists: %s" % args.outPath)

    Nn = args.num_neurons
    # Generate rho_target Vector
    print("\n=== Generating rho_target Vector ===")
    probLo, probHi = args.edge_prob
    min_rho = args.num_neurons * probLo
    max_rho = args.num_neurons * probHi
    rho_target = np.random.uniform(min_rho, max_rho, size=args.num_neurons)
    rho_target = np.maximum(5.0, rho_target).astype(int)  # 
    #1rho_target=np.linspace(1,args.num_neurons,args.num_neurons).astype(int)  # testing only, linear growth

    print(f"Generated rho_target from range [%.2f, %.2f] with min value 5.0" % (min_rho, max_rho))
    print("edge_count_target stats: min=%d, max=%d, mean=%.2f" % (np.min(rho_target), np.max(rho_target), np.mean(rho_target)))

    dale_conf = {
        'num_neurons': args.num_neurons,
        'num_excite': args.num_excite,
        'g': 2,     # Inhibitory-to-excitatory synaptic strength ratio
        'R': args.spectralR,  # Initial spectral radius
        'eta': 10,  # Learning rate for stabilization algorithm
        'C': 1.5,   # Parameter for stabilization algorithm
        'B': 0.2,    # Parameter for stabilization algorithm
        'edge_prob': args.edge_prob
    }

    if args.verb>1:        
        print("Dale configuration:"); pprint(dale_conf)
        
    # sanity checks
    assert Nn>=10
    assert args.num_excite>=5
    assert args.num_excite<Nn
    assert args.num_steps>=1000
    assert args.step_size>0.001
    assert args.spectralR>0.01
    if not args.expRate:
        assert args.idleRate[0]>0.5
        assert args.idleRate[1]>args.idleRate[0]
    
    # Generate the stable Dale connectivity matrix A
    print("Generating stable Dale matrix for Nn=%d (%d Excit, %d Inhib)..." % (Nn, args.num_excite, Nn - args.num_excite))
    A_dale=gen_dale_matrics(dale_conf, rho_target)
        
    # Calculate and print sparsity of the generated matrix
    total_connections = A_dale.size
    zero_connections = np.sum(np.abs(A_dale) < 1e-10)  # Count near-zero as zero
    non_zero_connections = total_connections - zero_connections
    sparsity = zero_connections / total_connections
    
    print(f'Matrix sparsity: {sparsity*100:.1f}% ({zero_connections}/{total_connections} connections are zero)')
    print(f'Non-zero connections: {non_zero_connections} ({(1-sparsity)*100:.1f}%)')

    evol_conf={
        'num_steps': args.num_steps,
        'step_size': args.step_size,
        'evol_time': args.num_steps*args.step_size,
        'expRate': args.expRate
    }

    if not args.expRate:
        # Initialize bias vector B based on idle firing rate
        evol_conf[ 'idleRate']= args.idleRate
        Ri_arg = np.array(args.idleRate)
        Bi = np.log(Ri_arg)
        B_idle = np.random.uniform(Bi[0],Bi[1], size=(Nn,))
        
    else:
        from UtilFreqGen import gen_exponential_freq
        rateGen_conf = {
            'scale1': 3.0,
            'scale2': 0.4,
            'mix_ratio': 0.7,
            'min_freq': 1.0,
            'max_freq': 15.0
        }
        evol_conf['rate_gen_conf']=rateGen_conf
        B_idle = np.log(gen_exponential_freq( num_samples=Nn,**rateGen_conf)    )
        
    # Generate spike data using the Poisson  process
    probLo, probHi = args.edge_prob
    min_rho = args.num_neurons * probLo
    max_rho = args.num_neurons * probHi
    rho_target = np.random.uniform(min_rho, max_rho, size=args.num_neurons)
    rho_target = np.maximum(5.0, rho_target).astype(int)
    #print(f"Generated rho_target from range [{min_rho:.2f}, {max_rho:.2f}] with min value 5.0")
    print("rho_target stats: min=%d, max=%d, mean=%.2f" % (np.min(rho_target), np.max(rho_target), np.mean(rho_target)))
    
    start_time = time.time()
    Y = generate_poissonV5(num_steps=args.num_steps, dt=args.step_size, A=A_dale, B_intercept=B_idle, num_excite=args.num_excite, verb=args.verb)
    sim_time = time.time() - start_time
    print("Spike generation completed in %.1f seconds" % sim_time)

    # Evaluate spike stats and compute firing rates
    stats_dict, rates_dict , neur_freqIdx= estimate_rates(Y, dt=args.step_size, num_excite=args.num_excite, max_samples=100000, mxNn=5)
   
    # Prepare data for saving - reorder by frequency (highest to lowest firing rate)
    Y_uchar = np.clip(Y, 0, 255).astype(np.uint8)
    Y_freq_sorted = Y_uchar[:, neur_freqIdx]  # Reorder neurons by frequency
    rates_freq_sorted = rates_dict['single_rates'][neur_freqIdx]  # Reorder rates by frequency
    
    # Create reverse mapping: natural_index -> frequency_sorted_position
    neur_natIdx = np.empty(len(neur_freqIdx), dtype=int)
    neur_natIdx[neur_freqIdx] = np.arange(len(neur_freqIdx))
        
    spikeD = {
        'spikes': Y_freq_sorted,
        'single_rates': rates_freq_sorted
    }
    spikeMD={ 'short_name':args.dataName,'time_step_sec':args.step_size,'data_type':'simDale' }
    
    # Reorder connectivity matrix to match frequency-sorted neuron order
    A_freq_sorted = A_dale[np.ix_(neur_freqIdx, neur_freqIdx)]  # Reorder both rows and columns
    B_freq_sorted = B_idle[neur_freqIdx]  # Reorder bias vector
    
    # Create metadata first (needed for geom_edges_mask)
    trueMD = {'dale_conf': dale_conf, 'evol_conf': evol_conf,'short_name':args.dataName,'dale_simu_stats':stats_dict}
    
    # Create masks in natural order first, then reorder to match frequency-sorted data
    maskD=geom_edges_mask(trueMD)
    
    # Reorder geometric masks to match frequency-sorted neuron order
    maskG = maskD['geom']
    
    # Reorder 2D masks (connectivity masks)
    maskG['diagA'] = maskG['diagA'][np.ix_(neur_freqIdx, neur_freqIdx)]
    maskG['excA'] = maskG['excA'][np.ix_(neur_freqIdx, neur_freqIdx)]
    maskG['inhA'] = maskG['inhA'][np.ix_(neur_freqIdx, neur_freqIdx)]
    
    # Reorder 1D masks (neuron type masks)
    maskG['exc_idx'] = maskG['exc_idx'][neur_freqIdx]
    maskG['inh_idx'] = maskG['inh_idx'][neur_freqIdx]
    
    # Save simulation data using utility function
    trueD = { 'A_true': A_freq_sorted, 'B_true': B_freq_sorted, 'neur_natIdx': neur_natIdx}
    
    true_edges_mask(maskD,trueD)

    for xx in maskD:
        recD=maskD[xx]
        for yy in recD:
            name='mask.%s.%s'%(xx,yy)
            trueD[name]=recD[yy]

    outFt = os.path.join(args.outPath, args.dataName + '.simTruth.npz')
    write_data_npz(trueD, outFt, metaD=trueMD)

   
    outFs = os.path.join(args.outPath, args.dataName + '.spikes.npz')
    write_data_npz(spikeD, outFs, metaD=spikeMD)

    print("\nNext step command:")
    print("  ./fit_lassoPoisson.py  --dataName %s  --n_epochs  50 " % args.dataName)
    
    #--------------------------------
    # ....  plotting ........
    args.prjName=args.dataName+'_sim'
    plot=Plotter(args)
    
    if 'a' in args.showPlots:
        plot.Dale_matrix_and_eigen(A_dale,trueMD,figId=1)
    if 'b' in args.showPlots:
        plot.histo_weights_rates(trueD,spikeD,trueMD,figId=2)
 
    if 'c' in args.showPlots:
        plot.histo_weights_rates(trueD,spikeD,trueMD,byFreq=True,figId=2)
  
    plot.display_all()
    print('Simulation completed successfully!')

if __name__ == '__main__':
    main() 
