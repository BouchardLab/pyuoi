#!/usr/bin/env python3
"""
gen_dale_data.py: Simulate and generate training data for a dynamic neural system.

This script creates a ground-truth "world" defined by a weight matrix W,
simulates how the system evolves over time according to a set of equations,
and then saves the resulting data and a visualization to files. The matrix
generation follows Dale's Principle.

Key Operations:
1.  generate_dale_matrix(num_neuron, num_excite, sparsity):
    - Creates a sparse weight matrix W for `num_neuron` neurons.
    - The first `num_excite` rows are excitatory (positive off-diagonal weights).
    - The remaining rows are inhibitory (negative off-diagonal weights).
    - Diagonal elements are all strongly negative.
    - Checks for system stability (Re(eig(W)) < 1) and rescales if needed.

2.  simulate_evolution(W, T, tau, sigma):
    - Simulates the state of M variables over T time steps using the equation:
      x_{t+1} = x_t + (1/tau) * (-x_t + W @ x_t) + noise

3.  Main Execution Block:
    - Orchestrates the generation of W and the simulation of the trajectory.
    - Saves a .npz file with W, sparsity mask E, time constant tau, trajectory,
      and dimension info `w_dims` = [num_neuron, num_excite, num_inhibit].
    - Saves a .png file visualizing the data, showing:
        - The true W-matrix structure.
        - Time-series trajectories of a few variables.
        - Histograms of diagonal, excitatory, and inhibitory weights.
    - Prints a suggested command to run the corresponding fitter script.

Command-line arguments allow for configuration of neuron counts,
simulation length, sparsity, noise, and more.
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
    W[:num_excite, :][excite_mask] = np.random.uniform(0.5, 1.5, size=num_excite_conns)

    # Inhibitory rows (all connections from these neurons are negative)
    inhibit_mask = is_connected[num_excite:, :]
    num_inhibit_conns = np.sum(inhibit_mask)
    W[num_excite:, :][inhibit_mask] = np.random.uniform(-1.5, -0.5, size=num_inhibit_conns)

    # Diagonal elements are set independently
    diag_weights = np.random.uniform(-2, -1, size=num_neuron)
    np.fill_diagonal(W, diag_weights)

    # Ensure stability of the continuous-time system dx/dt ~ (W-I)x , which requires Re(eig(W)) < 1
    w_rescale_factor = 1.0
    eigvals = np.linalg.eigvals(W)
    max_re_eig = np.max(np.real(eigvals))
    if max_re_eig >= 1:
        print("Warning: W matrix is unstable (max Re(eig)=%.3f >= 1). Rescaling W." % max_re_eig)
        w_rescale_factor = max_re_eig
        W = W / w_rescale_factor * 0.99
        # Recompute for verification
        eigvals_new = np.linalg.eigvals(W)
        max_re_eig_new = np.max(np.real(eigvals_new))
        print("Info: new W matrix is stable, max Re(eig)=%.3f" % max_re_eig_new)
    return W, w_rescale_factor

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

    W, w_rescale_factor = generate_dale_matrix(num_neuron, num_excite, sparsity)

    num_inhibit = num_neuron - num_excite
    w_dims = np.array([num_neuron, num_excite, num_inhibit])

    print("W-matrix (M=%d):" % num_neuron)
    with np.printoptions(precision=3, suppress=True, linewidth=400):
        print(W)

    X = simulate_evolution(W, T, tau, sigma)
    E = (W != 0).astype(int)

    # Generate hash for filename
    hash_object = hashlib.md5(str(W).encode())
    hash_value = hash_object.hexdigest()[:6]
    
    base_name = "dataM%d_e%d_%s" % (num_neuron, num_excite, hash_value)
    if args.simName is not None:
        base_name = args.simName
        
    filename = base_name + ".npz"
    filepath = os.path.join("data", filename)
    os.makedirs("data", exist_ok=True)

    np.savez_compressed(filepath, W=W, E=E, tau=tau, trajectory=X, w_rescale_factor=w_rescale_factor, w_dims=w_dims)

    # Print filenames and command before plotting
    png_filename = base_name + ".png"
    png_filepath = os.path.join("data", png_filename)

    print("output .npz file: %s" % filepath)
    print("output .png file: %s" % png_filepath)
    print("./fit_dale_model.py --input %s --epochs 100 --batch 256 --lr 0.01 " % base_name)

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
    ax0.text(num_neuron * 0.5, num_excite / 2, 'Excitatory', color='red', ha='center', va='center')
    ax0.text(num_neuron * 0.5, num_excite + (num_neuron - num_excite) / 2, 'Inhibitory', color='blue', ha='center', va='center')
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
    parser.add_argument("--verb", type=int, default=1, help="Verbosity level")
    parser.add_argument('-M', "--numNeuron", type=int, default=40, help="Number of neurons")
    parser.add_argument('-E', "--numExcite", type=int, default=20, help="Number of excitatory neurons")
    parser.add_argument("-T", type=int, default=int(1e4), help="Number of time steps")
    parser.add_argument("-K", type=int, default=4, help="Number of variables to plot")
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

