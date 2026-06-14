#!/usr/bin/env python3
"""
 ./gen_daleMatrices3.py --num_neurons 100 --num_excite 70 --num_steps 10000 --dataName test1
 ./gen_daleMatrices3.py --num_neurons 100 --num_excite 70 --spectral_radius 0.5 --Boffsets 0 10 20 --dataName test2

Primary purpose: generate the ground-truth dictionary (A_true, B_true) for use
by gen_nonStationarySpikes3.py.  The stationary spike generation performed here
is for evaluation only (firing-rate sanity check per B-vector).

Dale's principle: each neuron is either excitatory (positive outgoing
weights) or inhibitory (negative outgoing weights), never both.

Pipeline:
1. Sparse connectivity mask E_true (N, N).
   Each neuron draws its own connection probability uniformly from
   --edge_prob [lo, hi].  Self-loops (diagonal) are always included.

2. Generate ONE weight matrix A_true for --spectral_radius R:
   - Excitatory rows (0..N_E-1): weights ~ Uniform(1-v, 1+v), v=0.2.
   - Inhibitory rows (N_E..N-1): weights ~ Uniform(-r-rv, -r+rv),
     r = N_E/N_I (balance ratio).
   - Mask with E_true, then rescale so rho(A_true) = R exactly.

3. For each offset in --Boffsets generate one bias vector B_m (N,):
   - idle rate range shifted by offset, converted to log scale,
     with separate corrections for excitatory and inhibitory neurons.
   - Stationary spikes Y are simulated (for evaluation) via
     Y_t ~ Poisson(exp(clip(A @ Y_{t-1} + B_m, max=eta_clip)) * dt).
   - Firing-rate statistics (rate, variance, Fano factor) computed.

Output files saved to <basePath>/truthDale/:
  <dataName>.simTruth.npz   — A_true, B_true, E_true + metadata
  <dataName>.spikes.npz     — stationary spikes + rate statistics

Output shapes (M = len(Boffsets), N = num_neurons, T = num_steps):
  E_true        (N, N)    int    — shared binary connectivity mask
  A_true        (N, N)    float  — single shared weight matrix
  B_true        (M, N)    float  — one bias vector per state/offset
  spikes        (M, T, N) uint8  — stationary spikes per bias vector
  single_rates  (M, N)    float  — mean firing rates (Hz) per bias vector
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
    - allows for  self-loops  (aka non-zero diagonal elements)  
    """
    prob_lo, prob_hi = edge_prob
    conn_per_neuron = np.random.uniform(prob_lo, prob_hi, size=n_units)
    E_true = (np.random.rand(n_units, n_units) < conn_per_neuron[:, None])
    np.fill_diagonal(E_true, True)  # allow for self-loops
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

    # 7. Enforce negative diagonal: flip sign of any positive self-loop.
    d_idx = np.diag_indices(n_units)
    W[d_idx] = np.where(W[d_idx] > 0.0, -W[d_idx], W[d_idx])
        
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

def spectral_radius_scaling(A, factors):
    n = A.shape[0]
    off_diag_mask = ~np.eye(n, dtype=bool)
    
    print(f"{'factor':>8s}  {'ρ(all)':>10s}  {'ρ(off-diag)':>12s}")
    print("-" * 34)
    
    for c in factors:
        # a) scale all elements
        rho_all = np.max(np.abs(np.linalg.eigvals(c * A)))
        
        # b) scale only off-diagonal
        A_off = A.copy()
        A_off[off_diag_mask] *= c
        rho_off = np.max(np.abs(np.linalg.eigvals(A_off)))
        
        print(f"{c:8.3f}  {rho_all:10.4f}  {rho_off:12.4f}")


#################### Simulation ##################

def set_flat_selfSpiking(Nn, idleRate, spect_radius, num_excite, boffsets):
    """Generate B_idle per B-offset; excitatory idle-rate range is scaled by sqrt(50/Nn)."""
    assert 0 < num_excite <= Nn
    sizeScale = np.sqrt(float(Nn)/100)
    B_all = np.zeros((len(boffsets), Nn))
    R = float(spect_radius)
    for ib, offset in enumerate(boffsets):
        idle_eff = np.array(idleRate, dtype=float) 
        Ri_scaled = idle_eff * R
        Bi = np.log(Ri_scaled)
        B_all[ib] = np.random.uniform(Bi[0], Bi[1], size=(Nn,))
        #B_all+= + float(offset)
        if 1:  # for ver Mar 11
            B_all[ib, :num_excite] +=1+offset  - R*1.5 - sizeScale # reduce excitatory rate
            B_all[ib, num_excite:] += offset  # reduce  inhibitory rate  
    return B_all

def gen_stationary_lag1_poisson(num_steps, dt, A, B_intercept, num_excite, eta_clip,verb=0):
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
        lambda_t = np.exp(np.clip(eta, max=eta_clip))  # avoid overflow        
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
    parser.add_argument("--num_excite", type=int, default=None, help="Number of excitatory neurons.")
    parser.add_argument("--edge_prob", type=float, nargs=2, default=[0.05, 0.2], help="Range of edge probability [min, max]; mean is used as mask connectivity.")
    parser.add_argument("--num_steps", type=int, default=10_001, help="Number of time steps for simulation.")
    parser.add_argument("--step_size", type=float, default=0.01, help="Integration time step size (dt) in seconds.")
    parser.add_argument("--spectral_radius", type=float, default=0.3, help="Target spectral radius value for the connectivity matrix.")
    parser.add_argument("--idleRate", type=float, nargs=2, default=[15, 30.], help="Range of idle firing rates [min, max] in Hz.")
    parser.add_argument("--Boffsets", type=float, nargs='+', default=[0.0], help="Per-state offsets added to idleRate range.")
    parser.add_argument('-v',"--verb", type=int, default=1, help="Verbosity level (0=quiet, 1=normal).")
    parser.add_argument("--dataName", type=str, default=None, help="Base name for output files (default: daleN<num_neurons>_<hash>).")
    parser.add_argument("--basePath", type=str, default='/pscratch/sd/b/balewski/2025_causalNet_tmp/', help="Output directory for all files.")

    np.set_printoptions(precision=3, suppress=True)

    args = parser.parse_args()
    args.varTwindow=5 #(sec)
    args.poisson_eta_clip=5  #~ [1e-3Hz , 1e3Hz]
    if args.dataName is None:
        args.dataName='daleN%d_'%args.num_neurons+hashlib.md5(os.urandom(32)).hexdigest()[:6]

    outPath=os.path.join(args.basePath, 'truthDale')
    if args.num_excite==None :   # per Roy & Kris wisdom
        args.num_excite=int(0.8*args.num_neurons)
    print('gen dale matrices args:', vars(args), '\n')

    # Validation checks
    Nn = args.num_neurons
    assert Nn >= 10
    assert args.num_excite >= 5 
    assert args.num_excite < Nn
    assert os.path.exists(args.basePath)
    assert os.path.exists(outPath)
    assert args.num_steps>=1000
    assert args.step_size>0.001
    assert args.idleRate[0]>=0.5
    assert args.idleRate[1]>args.idleRate[0]
    assert len(args.Boffsets) >= 1
    # Generate sparse connectivity mask (per-neuron random connectivity in edge_prob range)
    E_true = generate_sparse_mask(Nn, args.edge_prob)
    print(f"\n=== Generated sparse mask: edge_prob={args.edge_prob}, actual={np.mean(E_true):.3f}, non-zero={np.sum(E_true)} ===")

    dale_conf = {
        'num_neurons': args.num_neurons,
        'num_excite': args.num_excite,
        'spectral_radius': args.spectral_radius,
        'edge_prob': args.edge_prob,
        'idleRate': args.idleRate,
        'Boffsets': args.Boffsets,
    }

    print("Dale configuration:"); pprint(dale_conf)
        
    evol_conf={
        'num_steps': args.num_steps,
        'step_size': args.step_size,
        'evol_time': args.num_steps*args.step_size,
        'poisson_eta_clip': args.poisson_eta_clip
    }

    B_all = set_flat_selfSpiking(Nn, args.idleRate, args.spectral_radius, args.num_excite, boffsets=args.Boffsets)
    
    max_samples = 100_000

    Y_list = []
    rates_list, rates_var_list, fano_list = [], [], []
    stats_list = []

    verb_r = args.verb
    print(f"\n{'='*60}")
    print(f"  Spectral radius: R={args.spectral_radius:.3f}")
    print(f"{'='*60}")

    dale_conf_r = dale_conf.copy()
    dale_conf_r['spect_radius'] = args.spectral_radius

    A_dale = gen_dale_matrics(dale_conf_r, E_true, verb=verb_r)

    if 0:  # test scaling behavior of spectral radius when scaling A by different factors
        factors = [0.2, 0.4, 0.6, 0.8, 0.9, ]
        spectral_radius_scaling(A_dale, factors)
        exit(1)

    if verb_r > 0:
        total_connections = A_dale.size
        zero_connections = np.sum(np.abs(A_dale) < 1e-10)
        non_zero_connections = total_connections - zero_connections
        sparsity = zero_connections / total_connections
        print(f'Matrix sparsity: {sparsity*100:.1f}% ({zero_connections}/{total_connections} connections are zero)')
        print(f'Non-zero connections: {non_zero_connections} ({(1-sparsity)*100:.1f}%)')

    for ib, offset in enumerate(args.Boffsets):
        print(f"\n{'='*60}")
        print(f"  B offset [{ib+1}/{len(args.Boffsets)}]: offset={offset:.3f}")
        print(f"{'='*60}")

        B_idle = B_all[ib]
        start_time = time.time()
        Y = gen_stationary_lag1_poisson(num_steps=args.num_steps, dt=args.step_size, A=A_dale, B_intercept=B_idle, num_excite=args.num_excite,eta_clip=args.poisson_eta_clip, verb=verb_r if ib == 0 else 0)
        sim_time = time.time() - start_time
        print("Spike generation completed in %.1f seconds" % sim_time)

        stats_dict, rates_dict, _ = estimate_rates(Y, dt=args.step_size, num_excite=args.num_excite, max_samples=max_samples, varTwindow=args.varTwindow, mxNn=5, verb=0, spect_radius=args.spectral_radius)
        stats_dict['var_time_window_sec'] = float(args.varTwindow)
        stats_dict['max_samples'] = int(max_samples)

        Y_list.append(np.clip(Y, 0, 255).astype(np.uint8))
        rates_list.append(rates_dict['single_rates'])
        rates_var_list.append(rates_dict['sigle_rates_var'])
        fano_list.append(rates_dict['single_fano_fact'])
        stats_list.append(stats_dict)

    # Stack results along axis=0 (B-offset dimension)
    trueD = {
        'A_true': A_dale,
        'B_true': B_all,
        'E_true': E_true
    }
    
    trueMD = {'dale_conf': dale_conf, 'evol_conf': evol_conf, 'short_name': args.dataName,
              'provenance':{'state_model_file':args.dataName}}
    # delete , 'dale_simu_stats': stats_list}  
        
    spikeD = {
        'spikes': np.stack(Y_list, axis=0),
        'single_rates': np.stack(rates_list, axis=0),
        'sigle_rates_var': np.stack(rates_var_list, axis=0),
        'single_fano_fact': np.stack(fano_list, axis=0)
    }
    spikeMD={ 'short_name':args.dataName,'time_step_sec':args.step_size,'data_type':'simDaleStates', 'poisson_eta_clip': args.poisson_eta_clip }

    outFt = os.path.join(outPath,args.dataName + '.simTruth.npz')
    write_data_npz(trueD, outFt, metaD=trueMD)
    if args.verb>1:  pprint(trueMD)
    outFs = os.path.join(outPath, args.dataName + '.spikes.npz')
    write_data_npz(spikeD, outFs, metaD=spikeMD)
    if args.verb>1:  pprint(spikeMD)
        
    print("\nSimulation completed successfully!")
    print(f"\nRate Summary {args.dataName}  N={args.num_neurons}  exc={args.num_excite}, R={args.spectral_radius:.3f}  ")
    header = f"{'state':>5} {'B offset':>9} {'all rate (Hz)':>14} {'exc rate (Hz)':>14} {'inh rate (Hz)':>14}"
    print(header)
    print("-" * len(header))
    for ib, offset in enumerate(args.Boffsets):
        s = stats_list[ib]
        print(f"{ib:5d} {float(offset):9.1f} {s['avg_spike_rate_all']:14.1f} {s['avg_spike_rate_excit']:14.1f} {s['avg_spike_rate_inhib']:14.1f}")
    print("\nNext step commands:")
    print("     basePath="+args.basePath)
    print("  ./view_daleMatrix3.py  --basePath $basePath   --dataName %s  -p b -m 0   -X  -p a c d  " % args.dataName)
    print("  ./view_spikesTrain3.py  --basePath $basePath   --dataName %s  --time_range_sec 0 20 -p b -m 0   -X " % args.dataName)
    print("  ./gen_nonStationarySpikes3c.py  --basePath $basePath   --inputStates %s     --true_dwell_sec 1.0 " % args.dataName)
    print("  ./fit_lassoPoisson.py  --basePath $basePath  --inpPath ${basePath}/truthDale --dataName   %s   --num_epochs  50  " % args.dataName)
   

if __name__ == '__main__':
    main()  
 
