#!/usr/bin/env python3
"""
Dale's Principle Neural Data Generator

This script generates synthetic neural time-series data from ground-truth connectivity
matrices that follow Dale's Principle. It creates both the "world model" (connectivity
matrix W) and simulates realistic neural dynamics to produce training datasets for
connectivity inference algorithms.

Dale's Principle Implementation:
- Excitatory neurons (first num_excite): all outgoing connections are positive
- Inhibitory neurons (remaining): all outgoing connections are negative  
- Diagonal elements: strongly negative (self-inhibition)
- Sparse connectivity: random connections based on sparsity parameter

Key Operations:
1. generate_dale_matrix(num_neuron, num_excite, sparsity):
   - Creates sparse weight matrix W following Dale's Principle
   - Generates binary mask E indicating connection existence/sign
   - Ensures system stability by eigenvalue rescaling (max Re(λ) ≤ 0.8)
   - Returns stable W, ground-truth E, and rescaling factor

2. simulate_evolution(W, T, tau, sigma):
   - Simulates neural dynamics: dx/dt = (-x + W @ x) / τ + noise
   - Discrete time integration: x[t+1] = x[t] + dx/dt + gaussian_noise
   - Generates realistic neural trajectories over T time steps

3. Data Output:
   - .npz file: W (connectivity), E (ground truth), trajectory, metadata
   - .png file: visualization with matrix plot, trajectories, weight histograms
   - Suggested commands for training inference models

Stability Assurance:
- Monitors eigenvalues of connectivity matrix W
- Automatically rescales W to ensure max Re(eigenvalue) ≤ 0.8
- Provides stability margin for reliable neural dynamics

Visualization Features:
- True connectivity matrix with Dale's structure highlighted
- Sample neural trajectories showing realistic dynamics
- Weight distribution histograms (diagonal, excitatory, inhibitory)
- Color-coded excitatory/inhibitory regions

Output Files:
- dataM{neurons}E{excitatory}_{hash}.npz: Complete dataset
- dataM{neurons}E{excitatory}_{hash}.png: Visualization
- Command suggestions for fit_dale_model.py and bayes_sparse_regression.py

Usage: ./gen_dale_data.py --numNeuron 30 --numExcite 20 --sparse 0.3 --T 100000
"""

import numpy as np
import matplotlib as mpl
import hashlib
import os
import argparse
import torch

def generate_dale_matrix(num_neuron, num_excite, sparsity):
    """Generates a sparse, stable W matrix according to Dale's principle."""
    # Start with a zero matrix
    W = np.zeros((num_neuron, num_neuron))
    
    # Create a sparse mask for non-zero connections
    is_connected = np.random.rand(num_neuron, num_neuron) < sparsity

    # Excitatory rows (all connections from these neurons are positive)
    excite_mask = is_connected[:num_excite, :]
    num_excite_conns = np.sum(excite_mask)
    W[:num_excite, :][excite_mask] = np.random.uniform(0.25, 0.75, size=num_excite_conns)

    # Inhibitory rows (all connections from these neurons are negative)
    inhibit_mask = is_connected[num_excite:, :]
    num_inhibit_conns = np.sum(inhibit_mask)
    W[num_excite:, :][inhibit_mask] = np.random.uniform(-0.75, -0.25, size=num_inhibit_conns)

    # Diagonal elements are set independently
    diag_weights = np.random.uniform(-2, -1, size=num_neuron)
    np.fill_diagonal(W, diag_weights)

    # Create the ternary ground truth matrix E
    E = np.sign(W)
    np.fill_diagonal(E, -1)

    # Ensure stability with margin: rescale W so max eigenvalue is 0.8
    target_max_eig = 0.8
    eigvals = np.linalg.eigvals(W)
    max_re_eig = np.max(np.real(eigvals))
    
    if max_re_eig > target_max_eig:
        print("Rescaling W matrix: max Re(eig)=%.3f -> target=%.1f" % (max_re_eig, target_max_eig))
        w_rescale_factor = max_re_eig / target_max_eig
        W = W / w_rescale_factor
        # Recompute for verification
        eigvals_new = np.linalg.eigvals(W)
        max_re_eig_new = np.max(np.real(eigvals_new))
        print("Info: rescaled W matrix, max Re(eig)=%.3f" % max_re_eig_new)
    else:
        w_rescale_factor = 1.0
        print("Info: W matrix already stable, max Re(eig)=%.3f" % max_re_eig)
    return W, E, w_rescale_factor

def simulate_evolution(W, T, tau, sigma):
    """Simulates the time evolution of the system."""
    M = W.shape[0]
    X = np.zeros((M, T))
    for t in range(T - 1):
        delta_X = (-X[:, t] + W @ X[:, t]) / tau
        g_t = np.random.normal(0, sigma, size=M)
        X[:, t + 1] = X[:, t] + delta_X + g_t
    return X

def generate_data_and_plot(args):
    num_neuron, num_excite, T, K, tau, sigma, sparsity = args.numNeuron, args.numExcite, args.T, args.K, args.tau, args.sigma, args.sparse
    print("generate_data START, args:", args)
    assert num_excite < num_neuron, "numExcite must be less than numNeuron"

    W, E, w_rescale_factor = generate_dale_matrix(num_neuron, num_excite, sparsity)

    num_inhibit = num_neuron - num_excite
    w_dims = np.array([num_neuron, num_excite, num_inhibit])

    if args.verb > 0:
        print("W-matrix (M=%d):" % num_neuron)
        with np.printoptions(precision=3, suppress=True, linewidth=400):
            print(W)
    
    if args.verb > 1:
        print("E-matrix (M=%d):" % num_neuron)
        with np.printoptions(linewidth=400):
            print(E.astype(int))

    X = simulate_evolution(W, T, tau, sigma)
    
    # Generate hash for filename
    hash_object = hashlib.md5(str(W).encode())
    hash_value = hash_object.hexdigest()[:6]
    
    base_name = "dataM%dE%d_%s" % (num_neuron, num_excite, hash_value)
    if args.simName is not None:
        base_name = args.simName
        
    filename = base_name + ".npz"
    filepath = os.path.join("data", filename)
    os.makedirs("data", exist_ok=True)

    np.savez_compressed(filepath, W=W, E=E, tau=tau, trajectory=X, w_rescale_factor=w_rescale_factor, w_dims=w_dims, sparsity_frac=sparsity)

    # Print filenames and command before plotting
    png_filename = base_name + ".png"
    png_filepath = os.path.join("data", png_filename)

    print("output .npz file: %s" % filepath)
    print("output .png file: %s" % png_filepath)
    print("./fit_dale_model.py --input %s --epochs 30 --batch 256 --lr 0.01 " % base_name)
    print("./bayes_sparse_regression.py --input %s --num_epochs 100 --batch_size 10_000 --num_samples 1000" % base_name)

    # Plotting
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 8))
    
    # True W-matrix plot
    ax0 = plt.subplot2grid((2, 3), (0, 0))
    W_plot = W
    vmax = np.max(np.abs(W_plot))
    im = ax0.imshow(W_plot, cmap='bwr', interpolation='nearest', vmin=-vmax, vmax=vmax)
    ax0.set_title("True W-matrix")
    ax0.set_ylabel("presyn. node index, source")
    ax0.set_xlabel("postsyn. node index, target")
    ax0.axhline(y=num_excite - 0.5, color='k', linestyle='--')
    ax0.text(num_neuron * 0.5, num_excite / 2, f'Excitatory ({num_excite})', color='red', ha='center', va='center')
    ax0.text(num_neuron * 0.5, num_excite + (num_neuron - num_excite) / 2, f'Inhibitory ({num_neuron - num_excite})', color='blue', ha='center', va='center')
    cbar = plt.colorbar(im, ax=ax0)
    cbar.set_label('coupling strength')
    ax0.set_aspect('equal', adjustable='box')
    ax0.grid(True)
    
    # Trajectories
    ax1 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    plot_indices = np.random.choice(num_neuron, K, replace=False)
    for i in plot_indices:
        ax1.plot(X[i, :1000], label="Variable %d" % (i+1))
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("State")
    title_str = "Trajectories of %d vars, tau=%.1f, %s" % (K, tau, base_name)
    ax1.set_title(title_str)
    ax1.legend()
    ax1.grid(True)

    # Diagonal elements histogram
    ax2 = plt.subplot2grid((2, 3), (1, 0))
    diag_W = np.diag(W)
    ax2.hist(diag_W, bins=20, color='green')
    ax2.set_xlabel("Diagonal Weight")
    ax2.set_ylabel("Count")
    ax2.set_title("Diagonal Elements (N=%d)" % len(diag_W))
    ax2.grid(True)
    ax2.locator_params(axis='x', nbins=5)

    # Excitatory off-diagonal elements histogram
    ax3 = plt.subplot2grid((2, 3), (1, 1))
    off_diag_mask = ~np.eye(num_neuron, dtype=bool)
    excite_W_flat = W[:num_excite, :]
    excite_off_diag_mask = off_diag_mask[:num_excite, :]
    excite_off_diag_W = excite_W_flat[excite_off_diag_mask & (excite_W_flat != 0)]
    ax3.hist(excite_off_diag_W, bins=20, color='salmon')
    ax3.set_xlabel("Excitatory Weight")
    ax3.set_ylabel("Count")
    ax3.set_title("Excitatory Off-diag (N=%d)" % len(excite_off_diag_W))
    ax3.grid(True)
    ax3.locator_params(axis='x', nbins=5)

    # Inhibitory off-diagonal elements histogram
    ax4 = plt.subplot2grid((2, 3), (1, 2))
    inhibit_W_flat = W[num_excite:, :]
    inhibit_off_diag_mask = off_diag_mask[num_excite:, :]
    inhibit_off_diag_W = inhibit_W_flat[inhibit_off_diag_mask & (inhibit_W_flat != 0)]
    ax4.hist(inhibit_off_diag_W, bins=20, color='blue')
    ax4.set_xlabel("Inhibitory Weight")
    ax4.set_ylabel("Count")
    ax4.set_title("Inhibitory Off-diag (N=%d)" % len(inhibit_off_diag_W))
    ax4.grid(True)
    ax4.locator_params(axis='x', nbins=5)
    
    plt.tight_layout()
    plt.savefig(png_filepath)
    if not args.noXterm:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-X',"--noXterm", action='store_true', default=False, help="Disable X-server for plotting")
    parser.add_argument('-v',"--verb", type=int, default=1, help="Verbosity level")
    parser.add_argument('-M', "--numNeuron", type=int, default=40, help="Number of neurons")
    parser.add_argument('-E', "--numExcite", type=int, default=20, help="Number of excitatory neurons")
    parser.add_argument("-T", type=int, default=int(1e4), help="Number of time steps")
    parser.add_argument("-K", type=int, default=6, help="Number of variables to plot")
    parser.add_argument("-tau", type=float, default=20.0, help="Tau value")
    parser.add_argument("--sigma", type=float, default=1.0, help="Standard deviation of the noise")
    parser.add_argument("--sparse", type=float, default=0.15, help="Sparsity level for W matrix")
    parser.add_argument("--simName", type=str, default=None, help="Optional base name for output files")
    args = parser.parse_args()

    if args.noXterm:
        if args.verb > 0: print('disable Xterm')
        mpl.use('Agg')
    else:
        mpl.use('TkAgg')
    
    generate_data_and_plot(args)

