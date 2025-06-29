#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import hashlib
import os
import argparse
import torch

def generate_w_matrix(M, sparsity):
    """Generates a sparse, stable W matrix."""
    # Generate W matrix with negative diagonal and sparsity
    W = np.random.rand(M, M) * 2 - 1  # values in [-1,1] range
    
    # Preserve diagonal before applying sparsity
    diag_W = np.diag(W).copy()
    
    # Apply sparsity to the whole matrix
    W = W * (np.random.rand(M, M) < sparsity)
    
    # Restore the original diagonal, ensuring it is negative and non-zero
    np.fill_diagonal(W, -np.abs(diag_W))

    # Ensure stability of the continuous-time system dx/dt ~ (W-I)x , which requires Re(eig(W)) < 1
    eigvals = np.linalg.eigvals(W)
    max_re_eig = np.max(np.real(eigvals))
    if max_re_eig >= 1:
        print("Warning: W matrix is unstable (max Re(eig)=%.3f >= 1). Rescaling W." % max_re_eig)
        W = W / max_re_eig * 0.99
        # Recompute for verification
        eigvals_new = np.linalg.eigvals(W)
        max_re_eig_new = np.max(np.real(eigvals_new))
        print("Info: new W matrix is stable, max Re(eig)=%.3f" % max_re_eig_new)
    return W

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
    M, T, K, tau, sigma, sparsity = args.M, args.T, args.K, args.tau, args.sigma, args.sparse
    print("generate_data START, args:", args)

    W = generate_w_matrix(M, sparsity)

    print("W-matrix (M=%d):" % M)
    with np.printoptions(precision=3, suppress=True, linewidth=400):
        print(W)

    X = simulate_evolution(W, T, tau, sigma)
    E = (W != 0).astype(int)

    # Generate hash for filename
    hash_object = hashlib.md5(str(W).encode())
    hash_value = hash_object.hexdigest()[:6]
    
    base_name = "dataM%d_%s" % (M, hash_value)
    if args.simName is not None:
        base_name = args.simName
        
    filename = base_name + ".npz"
    filepath = os.path.join("data", filename)
    os.makedirs("data", exist_ok=True)

    np.savez_compressed(filepath, W=W, E=E, tau=tau, trajectory=X)

    # Print filenames and command before plotting
    png_filename = base_name + ".png"
    png_filepath = os.path.join("data", png_filename)

    print("output .npz file: %s" % filepath)
    print("output .png file: %s" % png_filepath)
    print("./fit_model.py --input %s --epochs 100 --batch 256 --lr 0.01 " % base_name)

    # Plotting
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 8))
    
    # Trajectories
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    plot_indices = np.random.choice(M, K, replace=False)
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
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    diag_W = np.diag(W)
    ax2.hist(diag_W, bins=20)
    ax2.set_xlabel("Diagonal Weight")
    ax2.set_ylabel("Count")
    ax2.set_title("Diagonal Elements of W (N=%d)" % len(diag_W))
    ax2.grid(True)

    # Off-diagonal elements histogram
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    off_diag_mask = ~np.eye(M, dtype=bool)
    off_diag_W = W[off_diag_mask & (W != 0)]
    ax3.hist(off_diag_W, bins=50)
    ax3.set_xlabel("Off-diagonal Weight")
    ax3.set_ylabel("Count")
    ax3.set_title("Non-zero Off-diagonal Elements (N=%d)" % len(off_diag_W))
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(png_filepath)
    if not args.noXterm:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-X',"--noXterm", action='store_true', default=False, help="Disable X-server for plotting")
    parser.add_argument("--verb", type=int, default=1, help="Verbosity level")
    parser.add_argument("-M", type=int, default=10, help="Number of variables")
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

