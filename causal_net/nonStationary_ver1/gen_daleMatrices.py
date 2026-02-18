#!/usr/bin/env python3
"""
 ./gen_daleMatrices.py --num_neurons 100 --num_excite 70 --num_steps 10000 --dataName test1
 ./gen_daleMatrices.py --num_neurons 100 --num_excite 70 --spect_radius 0.2 0.4 0.8 --dataName test2

Dale Poisson Simulator — generates synthetic spike data from a recurrent
neuronal network obeying Dale's principle using a discrete-time Poisson
GLM (generalized linear model).

Dale's principle: each neuron is either excitatory (positive outgoing
weights) or inhibitory (negative outgoing weights), never both.

Pipeline:
1. Sparse connectivity mask E_true (shared across all spectral radii).
   Each neuron draws its own connection probability uniformly from
   --edge_prob [lo, hi], producing a binary N x N mask.

2. For each spectral radius R in --spect_radius:
   a) Weight matrix A_true is initialized with E-I balanced random
      weights, masked by E_true, then rescaled so that the spectral
      radius equals R.
   b) Bias vector B_true is sampled from log-uniform in
      [idleRate_lo * R, idleRate_hi * R], coupling baseline firing
      rates to the spectral radius.
   c) Spike time series Y is generated via a stationary VAR(1) Poisson
      process:  Y_t ~ Poisson(exp(A @ Y_{t-1} + B) * dt).
   d) Firing-rate statistics (rates, Fano factor, coincidence
      rate) are computed by estimate_rates().

3. All results are stacked along axis 0 (spectral-radius dimension)
   and saved into two .npz files:
     <dataName>.simTruth.npz  — A_true, B_true, E_true, metadata
     <dataName>.spikes.npz    — spikes, single_rates, variance, Fano

Output shapes (nR = len(spect_radius), N = num_neurons, T = num_steps):
  E_true        (N, N)       int    — shared connectivity mask
  A_true        (nR, N, N)   float  — weight matrices
  B_true        (nR, N)      float  — bias vectors
  spikes        (nR, T, N)   uint8  — spike counts
  single_rates  (nR, N)      float  — mean firing rates (Hz)
"""

import numpy as np
import time,hashlib
import scipy
import os
import sys

import argparse
from pprint import pprint

from toolbox.Util_NumpyIO import write_data_npz
from UtilDalePoisson import estimate_rates

###### Matrix generation ##################

def generate_sparse_mask(n_units, edge_prob):
    """
    Creates a binary mask representing the network topology.
    - n_units: Number of neurons
    - edge_prob: [lo, hi] range; each neuron gets a random connectivity drawn uniformly from this range
    """
    prob_lo, prob_hi = edge_prob
    conn_per_neuron = np.random.uniform(prob_lo, prob_hi, size=n_units)
    E_true = (np.random.rand(n_units, n_units) < conn_per_neuron[:, None]).astype(int)
    return E_true

# Generate an initial network connectivity matrix
def init_W(n_units, n_excite, E_true, R, varyW, verb=1):
    """
    Generalized weight initialization with E-I row-based balancing,
    zero-diagonal, spectral radius conserved.
    """
    # 1. Assertions to ensure valid population counts
    n_inhib = n_units - n_excite
    assert n_excite > 0, "n_excite must be greater than 0"
    assert n_inhib > 0, "n_inhib must be greater than 0 (n_units > n_excite)"
    
    # 2. Balance ratio: ensures the expected sum of the matrix is zero
    ie_ratio = n_excite / n_inhib
    
    # 3. Initialize the weight container
    W = np.zeros((n_units, n_units))
    
    # 4. Assign Excitatory Weights (Rows 0 to n_excite-1)
    W[:n_excite, :] = np.random.uniform(1.0 - varyW, 
                                        1.0 + varyW, 
                                        (n_excite, n_units))
    
    # 5. Assign Inhibitory Weights (Rows n_excite to n_units-1)
    i_center = -ie_ratio
    i_vary = varyW * ie_ratio
    W[n_excite:, :] = np.random.uniform(i_center - i_vary, 
                                        i_center + i_vary, 
                                        (n_inhib, n_units))
    
    # 6. Apply the connectivity mask (Topology)
    W = W * E_true
    
    # 7. Force zero diagonal (no self-loops)
    np.fill_diagonal(W, 0)
    
    # 8. Spectral Normalization to ensure Stability
    # This scales the entire cloud of eigenvalues to fit within radius R
    current_rho = np.max(np.abs(np.linalg.eigvals(W)))
    if verb > 0:
        print('initW current_rho :',current_rho )
    if current_rho > 0:
        W = W * (R / current_rho)
        
    return W

# Generate the full set of matrices for use in subsequent synthetic experiments
def gen_dale_matrics(conf, E_true, verb=1):
    """Generates one stable Dale matrix based on configuration."""
    num_neurons, num_excite,  Rmax= conf['num_neurons'], conf['num_excite'],conf[ 'spect_radius']
    if verb > 0:
        print(f"\n=== Generating Dale Matrix ===")
        print(f"Parameters: N={num_neurons}, excit={num_excite}, Rmax={Rmax:.1f}")

    # additional configuration
    varyW=0.2  #  controls variation of excitatory weights

    A=init_W(num_neurons, num_excite, E_true, Rmax, varyW, verb=verb)
    return A

#################### Simulation ##################

def set_flat_selfSpiking(Nn, idleRate, spect_radii, num_excite):
    """Generate B_idle per spectral radius; excitatory idle-rate range is scaled by sqrt(50/Nn)."""
    assert 0 < num_excite <= Nn
    #excit_rate_scale = np.sqrt(10 / float(Nn))
    excit_rate_scale = 10/Nn
    log_excit_shift = np.log(excit_rate_scale)
    num_radii = len(spect_radii)
    B_all = np.zeros((num_radii, Nn))
    for ir, R in enumerate(spect_radii):
        Ri_scaled = np.array(idleRate, dtype=float) * R
        Bi = np.log(Ri_scaled)
        B_all[ir] = np.random.uniform(Bi[0], Bi[1], size=(Nn,))
        #B_all[ir, :num_excite] -= log_excit_shift  # reduce rate of excitatpory neurons for larger Nn
        B_all[ir, :num_excite] -= 1.  # reduce inhibitory rate
        B_all[ir, num_excite:] += 1.  # increase excitatory rate
    return B_all

def gen_stationary_lag1_poisson(num_steps, dt, A, B_intercept, num_excite, verb=0):
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
    if verb > 0:
        print(f"\n=== Generating Poisson  Process ===")
        print(f"Simulation parameters: steps={num_steps}, dt={dt:.3f}, neurons={A.shape[0]}, excit={num_excite}")
        print(f"Matrix A stats: min={np.min(A):.3f}, max={np.max(A):.3f}, mean={np.mean(A):.3f}")
        print(f"Bias B stats: min={np.min(B_intercept):.3f}, max={np.max(B_intercept):.3f}, mean={np.mean(B_intercept):.3f}")
    
    if A is None:
        raise ValueError("Connectivity matrix A cannot be None.")
    d=A.shape[0]
    
    Y = np.zeros((num_steps, d), dtype=int)
    Y[0] = np.random.poisson(np.exp(B_intercept)*dt)  # initial state
    if verb > 0:
        print(f"Initial state: total spikes={np.sum(Y[0])}, excit spikes={np.sum(Y[0][:num_excite])}, inhib spikes={np.sum(Y[0][num_excite:])}")
    
    if verb>0:
        print('t=0 Y[t] sum=%d, Excit(first 3):%s, Inhib(first 3):%s'%(np.sum(Y[0]), Y[0][:3], Y[0][num_excite:num_excite+3]))

    kk=5
    # Main simulation loop
    if verb > 0:
        print("Starting main simulation loop...")
    for t in range(1, num_steps):
        eta = A @ Y[t-1] + B_intercept
        #eta = B_intercept  # use it to see idle rate only
        lambda_t = np.exp(np.clip(eta, -5, 5))  # avoid overflow        
        Y[t] = np.random.poisson(lambda_t*dt)
        
        if verb>0 and t<5:
            print('t=%d Y[t] sum=%d, Excit:%s, Inhib:%s'%(t, np.sum(Y[t]), Y[t][:kk], Y[t][num_excite:num_excite+kk]))
        
        if verb > 0 and t % (num_steps // 4) == 0:
            print(f"  Progress: {t}/{num_steps} steps ({t/num_steps*100:.1f}%) -  total spikes in this step: {np.sum(Y[t])}")

    if verb > 0:
        print(f"Simulation complete. Final state: total spikes={np.sum(Y[-1])}")
        print(f"Spike data stats: min={np.min(Y)}, max={np.max(Y)}, mean={np.mean(Y):.2f}, total spikes={np.sum(Y)}")
    
    return Y

#########################
#  MAIN
#########################

def main():
    print("=" * 60)
    print("DALE POISSON SIMULATION Lag=1")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="Simulate a recurrent neural network with Dale's principle.")
    parser.add_argument("--num_neurons", type=int, default=50, help="Total number of neurons in the network.")
    parser.add_argument("--num_excite", type=int, default=30, help="Number of excitatory neurons.")
    parser.add_argument("--edge_prob", type=float, nargs=2, default=[0.05, 0.2], help="Range of edge probability [min, max]; mean is used as mask connectivity.")
    parser.add_argument("--num_steps", type=int, default=10_000, help="Number of time steps for simulation.")
    parser.add_argument("--step_size", type=float, default=0.01, help="Integration time step size (dt) in seconds.")
    parser.add_argument("--spect_radius", type=float, nargs='+', default=[0.3, 0.95], help="Target spectral radius value(s) for the connectivity matrix.")
    parser.add_argument("--idleRate", type=float, nargs=2, default=[15, 30.], help="Range of idle firing rates [min, max] in Hz.")
    parser.add_argument('-v',"--verb", type=int, default=1, help="Verbosity level (0=quiet, 1=normal).")
    parser.add_argument("--dataName", type=str, default=None, help="Base name for output files (default: dale_spikes_xx).")
    parser.add_argument("--dataPath", type=str, default='/pscratch/sd/b/balewski/2025_causalNet_tmp/', help="Output directory for all files.")

    np.set_printoptions(precision=3, suppress=True)

    args = parser.parse_args()
    if args.dataName is None:
        args.dataName='daleM%d_'%args.num_neurons+hashlib.md5(os.urandom(32)).hexdigest()[:6]
        
    print("\nStarting simulation with configuration:")
    print(vars(args))

    # Validation checks
    Nn = args.num_neurons
    assert Nn >= 10
    assert args.num_excite >= 5
    assert args.num_excite < Nn
    assert os.path.exists(args.dataPath)
    assert args.num_steps>=1000
    assert args.step_size>0.001
    assert args.idleRate[0]>=0.5
    assert args.idleRate[1]>args.idleRate[0]
    # Generate sparse connectivity mask (per-neuron random connectivity in edge_prob range)
    E_true = generate_sparse_mask(Nn, args.edge_prob)
    print(f"\n=== Generated sparse mask: edge_prob={args.edge_prob}, actual={np.mean(E_true):.3f}, non-zero={np.sum(E_true)} ===")

    dale_conf = {
        'num_neurons': args.num_neurons,
        'num_excite': args.num_excite,
        'spect_radius': args.spect_radius,  
        'edge_prob': args.edge_prob,
    }

    print("Dale configuration:"); pprint(dale_conf)
        
    evol_conf={
        'num_steps': args.num_steps,
        'step_size': args.step_size,
        'evol_time': args.num_steps*args.step_size,
        'idleRate': args.idleRate
    }

    B_all = set_flat_selfSpiking(Nn, args.idleRate, args.spect_radius, args.num_excite)
    varTwindow=5 #(sec)

    num_radii = len(args.spect_radius)
    A_list, Y_list = [], []
    rates_list, rates_var_list, fano_list = [], [], []
    stats_list = []

    for ir, R in enumerate(args.spect_radius):
        verb_r = args.verb if ir == 0 else 0
        print(f"\n{'='*60}")
        print(f"  Spectral radius [{ir+1}/{num_radii}]: R={R:.3f}")
        print(f"{'='*60}")

        dale_conf_r = dale_conf.copy()
        dale_conf_r['spect_radius'] = R

        A_dale = gen_dale_matrics(dale_conf_r, E_true, verb=verb_r)

        if verb_r > 0:
            total_connections = A_dale.size
            zero_connections = np.sum(np.abs(A_dale) < 1e-10)
            non_zero_connections = total_connections - zero_connections
            sparsity = zero_connections / total_connections
            print(f'Matrix sparsity: {sparsity*100:.1f}% ({zero_connections}/{total_connections} connections are zero)')
            print(f'Non-zero connections: {non_zero_connections} ({(1-sparsity)*100:.1f}%)')

        B_idle = B_all[ir]
        start_time = time.time()
        Y = gen_stationary_lag1_poisson(num_steps=args.num_steps, dt=args.step_size, A=A_dale, B_intercept=B_idle, num_excite=args.num_excite, verb=verb_r)
        sim_time = time.time() - start_time
        print("Spike generation completed in %.1f seconds" % sim_time)

        stats_dict, rates_dict, _ = estimate_rates(Y, dt=args.step_size, num_excite=args.num_excite, max_samples=100000, varTwindow=varTwindow, mxNn=5, verb=verb_r, spect_radius=R)

        A_list.append(A_dale)
        Y_list.append(np.clip(Y, 0, 255).astype(np.uint8))
        rates_list.append(rates_dict['single_rates'])
        rates_var_list.append(rates_dict['sigle_rates_var'])
        fano_list.append(rates_dict['single_fano_fact'])
        stats_list.append(stats_dict)

    # Stack results along axis=0 (spectral radius dimension)
    trueD = {
        'A_true': np.stack(A_list, axis=0),
        'B_true': B_all,
        'E_true': E_true
    }
    
    trueMD = {'dale_conf': dale_conf, 'evol_conf': evol_conf, 'short_name': args.dataName, 'dale_simu_stats': stats_list}  
        
    spikeD = {
        'spikes': np.stack(Y_list, axis=0),
        'single_rates': np.stack(rates_list, axis=0),
        'sigle_rates_var': np.stack(rates_var_list, axis=0),
        'single_fano_fact': np.stack(fano_list, axis=0)
    }
    spikeMD={ 'short_name':args.dataName,'time_step_sec':args.step_size,'data_type':'simDale', 'var_time_window_sec':varTwindow }

    outFt = os.path.join(args.dataPath, args.dataName + '.simTruth.npz')
    write_data_npz(trueD, outFt, metaD=trueMD)
    if args.verb>1:  pprint(trueMD)
    outFs = os.path.join(args.dataPath, args.dataName + '.spikes.npz')
    write_data_npz(spikeD, outFs, metaD=spikeMD)
    if args.verb>1:  pprint(spikeMD)
        
    print("\nSimulation completed successfully!") 
    print("\nNext step commands:")
    print("     dataPath="+args.dataPath)
    print("  ./view_daleMatrix.py  --dataPath $dataPath   --dataName %s  -p b -i 0   -X  -p a c d  " % args.dataName)
    print("  ./view_spikesTrain.py  --dataPath $dataPath   --dataName %s  -p b -i 0   -X " % args.dataName)
   

if __name__ == '__main__':
    main()  
 
