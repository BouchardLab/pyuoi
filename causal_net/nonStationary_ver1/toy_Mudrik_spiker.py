#!/usr/bin/env python3

import numpy as np
import argparse
import os

# the code you have refined is a direct implementation of a Switching Poisson Vector Autoregressive (S-PVAR) model, though in the specific context of the Mudrik paper, it is referred to as a Decomposed Linear Dynamical System (dLDS).

def simulate_dlds_evolution(n_steps, A_true, B_true, target_states, max_delta_c=0.02, dt=0.01, verb=1):
    """
    Simulates dLDS dynamics with smoothly varying coefficients and Poisson spiking.
    """
    M, n_units = B_true.shape
    y_history = np.zeros((n_steps, n_units), dtype=np.uint8)
    c_history = np.zeros((n_steps, M))
    
    # Initialize coefficients at the first target state
    c_curr = np.zeros(M)
    c_curr[target_states[0]] = 1.0
    
    if verb > 0:
        print(f"{'Step':<6} | {'Target':<6} | {'Coefficients (c_mt)':<30}")
        print("-" * 50)

    for t in range(n_steps):
        # 1. Smoothly update coefficients toward the target state
        target_vec = np.zeros(M)
        target_vec[target_states[t]] = 1.0
        
        # Slew-rate constraint: move c_curr toward target_vec
        diff = target_vec - c_curr
        step = np.clip(diff, -max_delta_c, max_delta_c)
        c_curr += step
        
        # Re-normalize to ensure sum is 1.0
        c_curr /= np.sum(c_curr)
        c_history[t] = c_curr

        if verb > 0 and t % 1 == 0:
            c_str = ", ".join([f"{val:.2f}" for val in c_curr])
            print(f"{t:<6} | {target_states[t]:<6} | [{c_str}]")

        # 2. Decompose Dynamics: F_t = sum(f_m * c_mt)
        A_eff = np.einsum('m,mkl->kl', c_curr, A_true)
        B_eff = c_curr @ B_true
        
        # 3. Compute potential and sample Poisson spikes
        prev_y = y_history[t-1].astype(float) if t > 0 else np.zeros(n_units)
        eta_t = A_eff @ prev_y + B_eff
        
        # Intensity (spikes/sec) with safety clipping
        lambda_t = np.exp(np.clip(eta_t, -5, 5)) 
        
        # Generate spike counts for this dt (Poisson process)
        # Poisson(rate * time)
        y_history[t] = np.random.poisson(lambda_t * dt).astype(np.uint8)
        
    return y_history, c_history

def main():
    parser = argparse.ArgumentParser(description="dLDS Stochastic Poisson Spike Generator")
    parser.add_argument('-t',"--num_steps", type=int, default=100, help="Number of time steps.")
    parser.add_argument("--step_size", type=float, default=0.01, help="dt in seconds.")
    parser.add_argument("--max_delta_c", type=float, default=0.05, help="Max change in c per step.")
    parser.add_argument("--dwell_steps", type=int, default=6, help="Mean number of steps to stay in a state.")
    parser.add_argument("--simTruth_path", type=str, default='/dataVault2026/neurodata_tmp/daleN100_4822fd.simTruth.npz', help="Path to input simTruth npz file.")
    parser.add_argument("--out_npz", type=str, default=None, help="Output npz path (default: <simTruth_base>.toySpikes.npz).")
    parser.add_argument("-v", "--verb", type=int, default=1, help="Verbosity level.")
    args = parser.parse_args()

    print('myArg-program:', parser.prog)
    for arg in vars(args): print('myArg:', arg, getattr(args, arg))

    # Load ground truth data
    data_path = args.simTruth_path
    data = np.load(data_path)
    A_true = data['A_true']
    B_true = data['B_true']
    M = B_true.shape[0]

    # 1. Generate Stochastic Target State Sequence (Markov Chain)
    # Probability of leaving = 1 / dwell_steps
    p_exit = 1.0 / args.dwell_steps
    p_stay = 1.0 - p_exit
    
    # Distribute p_exit uniformly among other M-1 states
    transition_matrix = np.full((M, M), p_exit / (M - 1))
    np.fill_diagonal(transition_matrix, p_stay)  
    
    target_states = np.zeros(args.num_steps, dtype=int)
    current_state = 0
    for t in range(args.num_steps):
        current_state = np.random.choice(M, p=transition_matrix[current_state])
        target_states[t] = current_state

    # 2. Run Simulation
    spikes, coeffs = simulate_dlds_evolution(
        n_steps=args.num_steps,
        A_true=A_true,
        B_true=B_true,
        target_states=target_states,
        max_delta_c=args.max_delta_c,
        dt=args.step_size,
        verb=args.verb
    )

    # Verification
    changes = np.diff(target_states) != 0
    durations = np.diff(np.where(np.concatenate(([True], changes, [True])))[0])
    print(f"\nSimulation complete.")
    print(f"Spikes matrix shape: {spikes.shape}, Dtype: {spikes.dtype}")
    print(f"Actual Mean Dwell Time: {np.mean(durations):.2f} steps (Target: {args.dwell_steps})")
    print('transition_matrix:\n',transition_matrix)

    # Save output in requested format:
    # spikes: (nT, Nn), S_true: (nT,)
    out_npz = args.out_npz
    if out_npz is None:
        out_npz = data_path.replace('.simTruth.npz', '.toySpikes.npz')
        if out_npz == data_path:
            root, _ = os.path.splitext(data_path)
            out_npz = root + '.toySpikes.npz'

    S_true = target_states.astype(np.int32)
    np.savez_compressed(out_npz, spikes=spikes, S_true=S_true)
    print('saved npz:', out_npz)
    print('  spikes shape:', spikes.shape, spikes.dtype)
    print('  S_true shape:', S_true.shape, S_true.dtype)

    for arg in vars(args): print('myArg:', arg, getattr(args, arg))
    
if __name__ == "__main__":
    main()
