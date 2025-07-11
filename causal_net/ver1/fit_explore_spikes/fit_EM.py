#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
 salloc -q interactive -C gpu  -t 4:00:00 -A m2043 -N 1

 module load pytorch
"""

import numpy as np
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torch.amp import autocast, GradScaler  # type: ignore
import os
import time
import argparse

# Import HDF5 reading utility
import sys
sys.path.append('../ver2')  # Add path to toolbox
from toolbox.Util_H5io4 import read4_data_hdf5

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device setup for A100 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global variables for dimensions (will be set from data)
N_NEURONS = None  # Number of neurons (will be extracted from data)
T_TIMEBINS = None  # Number of time bins (will be extracted from data)
TAU = 10.0  # Known time constant (assumed value; adjust as needed)
SIGMA = 1.0  # Noise standard deviation
DT = 0.001  # Time step in seconds (1ms)
BATCH_SIZE_TIME = 10000  # Batch size for time bins to manage memory
SPARSITY_TARGET = 0.15  # Target sparsity for off-diagonal elements in W
L1_LAMBDA = 0.1  # Initial L1 penalty for sparsity (tune as needed)
MAX_EM_ITER = 20  # Maximum number of EM iterations
b = None  # Bias term per neuron (initialized in initialize_parameters)

# Load spike data from HDF5 file
def load_spike_data_hdf5(file_path, time_range=None, rebin_factor=1):
    """
    Load spike data from HDF5 file and return dimensions.
    Data is always loaded to CPU memory; GPU is used only for computations.
    time_range: [start, end] to limit time bins for memory management.
    rebin_factor: int, rebin data by summing over consecutive time bins.
    """
    print(f"Loading spike data from {file_path}...")
    
    # Read HDF5 data
    expD, expMD = read4_data_hdf5(file_path)
    
    # Extract spike data
    spike_data = expD['spikes_data']
    print(f"Original spike data shape: {spike_data.shape}, dtype: {spike_data.dtype}")
    
    # Apply time range if specified
    if time_range is not None:
        start_t, end_t = time_range
        print(f"Applying time range: [{start_t}, {end_t}]")
        spike_data = spike_data[:, start_t:end_t]
        print(f"Cropped spike data shape: {spike_data.shape}")
    
    # Apply rebinning if specified
    if rebin_factor > 1:
        print(f"Rebinning data by factor {rebin_factor}")
        original_time_bins = spike_data.shape[1]
        new_time_bins = original_time_bins // rebin_factor
        print(f"Original time bins: {original_time_bins}, new time bins: {new_time_bins}")
        
        clipped_time_bins = new_time_bins * rebin_factor
        spike_data = spike_data[:, :clipped_time_bins]
        print(f"Clipped {original_time_bins - clipped_time_bins} time bins from end")
        
        spike_data = spike_data.reshape(spike_data.shape[0], new_time_bins, rebin_factor)
        spike_data = spike_data.sum(axis=2)
        print(f"Rebinned spike data shape: {spike_data.shape}")
    
    # Get dimensions from actual data
    n_neurons = spike_data.shape[0]
    t_timebins = spike_data.shape[1]
    
    print(f"Set n_neurons = {n_neurons}, t_timebins = {t_timebins}")
    
    memory_gb = (n_neurons * t_timebins * 4) / (1024**3)
    print(f"Estimated memory usage for spike data: {memory_gb:.2f} GB")
    
    print("Loading data to CPU memory")
    spike_data_tensor = torch.from_numpy(spike_data.astype(np.float32)).to('cpu')
    
    return spike_data_tensor, expD, expMD, n_neurons, t_timebins

# Initialize hidden states X_t and connectivity matrix W
def initialize_parameters(n_neurons, t_timebins, device):
    X_t = torch.randn(n_neurons, t_timebins, device='cpu') * 0.1
    
    W = torch.randn(n_neurons, n_neurons, device=device) * 0.01
    with torch.no_grad():
        torch.diagonal(W).fill_(-1.0)
    
    b = torch.zeros(n_neurons, device=device)
    
    return X_t, W, b

# Compute dynamics update for X_t
def compute_dynamics(X_t, W, tau):
    """
    Compute the update term: Delta_X = (-X_t + W @ X_t) / tau
    """
    Delta_X = (-X_t + torch.matmul(W, X_t)) / tau
    return Delta_X

# Poisson log-likelihood for observed spikes
def poisson_log_likelihood(S_t, X_t, b, dt):
    """
    Compute log-likelihood of observed spikes under Poisson model
    log(lambda) = log(dt) + X_t + b (bias per neuron)
    """
    log_lambda = torch.log(torch.tensor(dt, device=b.device)) + X_t + b.view(-1, 1)
    lambda_rate = torch.exp(log_lambda)
    log_lik = S_t * log_lambda - lambda_rate
    return log_lik.sum()

# E-Step: Approximate posterior of X_t by optimization
def e_step(S_t, X_t, W, b, t_timebins, hparams, num_epochs=5, lr=0.01):
    """
    Approximate E-Step by optimizing X_t to maximize likelihood of S_t and dynamics.
    Returns updated X_t (on CPU) and the final loss of the last epoch.
    """
    X_t = X_t.clone().requires_grad_(True)
    optimizer = optim.Adam([X_t], lr=lr)
    scaler = GradScaler()
    last_loss = None
    
    batch_size_time = hparams['BATCH_SIZE_TIME']
    dt = hparams['DT']
    sigma = hparams['SIGMA']
    tau = hparams['TAU']
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        with autocast('cuda'):
            total_loss = 0.0
            for t_start in range(0, t_timebins, batch_size_time):
                t_end = min(t_start + batch_size_time, t_timebins)
                X_batch = X_t[:, t_start:t_end].to(device)
                S_batch = S_t[:, t_start:t_end].to(device)
                obs_loss = -poisson_log_likelihood(S_batch, X_batch, b, dt)
                if t_end < t_timebins:
                    X_next_batch = X_t[:, t_start+1:t_end+1].to(device)
                    Delta_X = compute_dynamics(X_batch[:, :-1], W, tau)
                    dyn_loss = 0.5 * torch.sum((X_next_batch[:, :-1] - X_batch[:, :-1] - Delta_X) ** 2) / (sigma ** 2)
                    total_loss += obs_loss + dyn_loss
                else:
                    total_loss += obs_loss
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        last_loss = total_loss.item()  # type: ignore
    
    return X_t.detach().cpu(), last_loss

# M-Step: Update W with sparsity constraint only (per-edge thresholding)
def m_step(X_t, W, n_neurons, t_timebins, hparams):
    """
    Update W by solving least squares with L1 penalty for sparsity
    Enforce per-edge thresholding only, without per-row sign enforcement
    """
    W_new = W.clone().requires_grad_(True)
    optimizer = optim.Adam([W_new], lr=0.01)
    scaler = GradScaler()
    
    batch_size_time = hparams['BATCH_SIZE_TIME']
    sigma = hparams['SIGMA']
    l1_lambda = hparams['L1_LAMBDA']
    tau = hparams['TAU']

    diag_mask = torch.eye(n_neurons, device=device).bool()

    for _ in range(10):
        optimizer.zero_grad()
        with autocast('cuda'):
            loss = 0.0
            for t_start in range(0, t_timebins-1, batch_size_time):
                t_end = min(t_start + batch_size_time, t_timebins-1)
                X_batch = X_t[:, t_start:t_end].to(device)
                X_next_batch = X_t[:, t_start+1:t_end+1].to(device)
                Delta_X_pred = compute_dynamics(X_batch, W_new, tau)
                residual = X_next_batch - X_batch - Delta_X_pred
                loss += 0.5 * torch.sum(residual ** 2) / (sigma ** 2)

            loss += l1_lambda * torch.sum(torch.abs(W_new[~diag_mask]))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            torch.diagonal(W_new).fill_(-1.0)

    return W_new.detach()

# Post-process W to binary +1/-1 with sparsity
def post_process_W(W, n_neurons, sparsity_target, device):
    """
    Threshold W to achieve target sparsity and set to +1/-1
    """
    W_processed = W.clone()
    with torch.no_grad():
        off_diag_mask = ~torch.eye(n_neurons, dtype=torch.bool, device=device)
        off_diag = W_processed[off_diag_mask]
        
        num_off_diag = off_diag.numel()
        num_nonzero_target = int(sparsity_target * num_off_diag)
        
        if num_nonzero_target > 0:
            abs_off_diag = torch.abs(off_diag)
            threshold = torch.kthvalue(abs_off_diag, num_off_diag - num_nonzero_target).values
            mask = abs_off_diag >= threshold
            off_diag[mask & (off_diag > 0)] = 1.0
            off_diag[mask & (off_diag < 0)] = -1.0
            off_diag[~mask] = 0.0
        else:
            off_diag.fill_(0.0)

        W_processed[off_diag_mask] = off_diag
        torch.diagonal(W_processed).fill_(-1.0)
    return W_processed

# Compare estimated W with ground truth
def compare_with_ground_truth(W_estimated, expD):
    """
    Compare estimated W matrix with ground truth from HDF5 data
    """
    # Extract ground truth matrix
    W_true = expD['true_network_matrix']
    print(f"Ground truth matrix shape: {W_true.shape}")
    print(f"Estimated matrix shape: {W_estimated.shape}")
    
    # Convert to numpy if needed
    if torch.is_tensor(W_estimated):
        W_est_np = W_estimated.cpu().numpy()
    else:
        W_est_np = W_estimated
    
    # Print basic statistics
    print(f"\n=== Matrix Comparison ===")
    print(f"Ground truth sparsity: {np.sum(W_true != 0) / W_true.size:.4f}")
    print(f"Estimated sparsity: {np.sum(W_est_np != 0) / W_est_np.size:.4f}")
    print(f"Ground truth range: [{W_true.min():.3f}, {W_true.max():.3f}]")
    print(f"Estimated range: [{W_est_np.min():.3f}, {W_est_np.max():.3f}]")
    
    # Edge detection metrics (binary: connection vs no connection)
    # Exclude diagonal for edge detection
    n_neurons = W_true.shape[0]
    off_diag_mask = ~np.eye(n_neurons, dtype=bool)
    
    true_edges = (W_true[off_diag_mask] != 0)
    est_edges = (W_est_np[off_diag_mask] != 0)
    
    # Calculate precision, recall, F1
    true_positives = np.sum(true_edges & est_edges)
    false_positives = np.sum(~true_edges & est_edges)
    false_negatives = np.sum(true_edges & ~est_edges)
    true_negatives = np.sum(~true_edges & ~est_edges)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(true_edges)
    
    print(f"\n=== Edge Detection Metrics ===")
    print(f"True edges: {np.sum(true_edges)}")
    print(f"Estimated edges: {np.sum(est_edges)}")
    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Sign prediction metrics (for detected edges only)
    sign_accuracy = 0.0
    if true_positives > 0:
        # Get signs of true positive edges
        true_positive_mask = true_edges & est_edges
        true_signs = np.sign(W_true[off_diag_mask][true_positive_mask])
        est_signs = np.sign(W_est_np[off_diag_mask][true_positive_mask])
        
        sign_accuracy = np.sum(true_signs == est_signs) / len(true_signs)
        excitatory_correct = np.sum((true_signs > 0) & (est_signs > 0))
        inhibitory_correct = np.sum((true_signs < 0) & (est_signs < 0))
        
        print(f"\n=== Sign Prediction Metrics (for detected edges) ===")
        print(f"Sign accuracy: {sign_accuracy:.4f}")
        print(f"Excitatory correctly identified: {excitatory_correct}")
        print(f"Inhibitory correctly identified: {inhibitory_correct}")
        
        # Count excitatory vs inhibitory in ground truth
        true_excitatory = np.sum(W_true[off_diag_mask] > 0)
        true_inhibitory = np.sum(W_true[off_diag_mask] < 0)
        est_excitatory = np.sum(W_est_np[off_diag_mask] > 0)
        est_inhibitory = np.sum(W_est_np[off_diag_mask] < 0)
        
        print(f"True excitatory connections: {true_excitatory}")
        print(f"True inhibitory connections: {true_inhibitory}")
        print(f"Estimated excitatory connections: {est_excitatory}")
        print(f"Estimated inhibitory connections: {est_inhibitory}")
    
    # Correlation metrics
    correlation = np.corrcoef(W_true.flatten(), W_est_np.flatten())[0, 1]
    print(f"\n=== Correlation Metrics ===")
    print(f"Pearson correlation: {correlation:.4f}")
    
    # MSE for non-zero elements
    nonzero_mask = (W_true != 0)
    if np.sum(nonzero_mask) > 0:
        mse_nonzero = np.mean((W_true[nonzero_mask] - W_est_np[nonzero_mask]) ** 2)
        print(f"MSE (non-zero elements): {mse_nonzero:.4f}")
    
    # Return metrics dictionary
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'sign_accuracy': sign_accuracy,
        'correlation': correlation,
        'true_edges': np.sum(true_edges),
        'estimated_edges': np.sum(est_edges),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }
    
    return metrics

# Main EM loop
def em_algorithm(S_t, n_neurons, t_timebins, hparams, device):
    # Keep spike data on CPU, process in chunks on GPU
    print("Spike data kept on CPU, will process in chunks on GPU...")
    initial_lr = 0.01
    max_iter = hparams['MAX_EM_ITER']
    dt = hparams['DT']
    
    X_t, W, b = initialize_parameters(n_neurons, t_timebins, device)
    for iter in range(max_iter):
        iter_start_time = time.time()
        
        if max_iter > 1:
            decay = iter / (max_iter - 1)
            lr_current = initial_lr * (1 - 0.9 * decay)
        else:
            lr_current = initial_lr
            
        print(f"EM Iteration {iter+1}/{max_iter}, E-step LR: {lr_current:.6f}")
        
        X_t, final_loss = e_step(S_t, X_t, W, b, t_timebins, hparams, lr=lr_current)
        print(f"E-Step final loss: {final_loss:.2f}")
        
        W = m_step(X_t, W, n_neurons, t_timebins, hparams)
        
        with torch.no_grad():
            sum_spikes = S_t.sum(dim=1)
            sum_exp = torch.exp(X_t).sum(dim=1)
            b_cpu = torch.log((sum_spikes + 1e-8) / (dt * sum_exp + 1e-8))
            b = b_cpu.to(device)
            
        W_bin = post_process_W(W, n_neurons, hparams['SPARSITY_TARGET'], device)
        off_bin = W_bin[~torch.eye(n_neurons, dtype=torch.bool, device=device)]
        pred_spars = torch.sum(off_bin != 0).float() / off_bin.numel()
        print(f"Predicted binary sparsity: {pred_spars.item():.3f}")  # type: ignore
        
        iter_elapsed_time = time.time() - iter_start_time
        print(f"EM Iteration {iter+1} completed in {iter_elapsed_time:.2f} seconds")
    
    W_final = post_process_W(W, n_neurons, hparams['SPARSITY_TARGET'], device)
    return W_final, X_t

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EM Algorithm for Neural Connectivity Inference")
    parser.add_argument("--data_path", type=str, default="/global/homes/b/balewski/prjs/bioDataVault2025/causalNet_tmp2/input_fitter", help="Path to the data directory")
    parser.add_argument("--file_name", type=str, default="daleM40-a03d69a-c1edd9d.spike.h5", help="Name of the input HDF5 file")
    args = parser.parse_args()

    print("=== EM Algorithm for Neural Connectivity Inference ===")
    print(f"Using device: {device}")
    
    # Hyperparameters and Configuration
    hparams = {
        'TAU': 10.0,
        'SIGMA': 1.0,
        'DT': 0.001,
        'BATCH_SIZE_TIME': 10000,
        'SPARSITY_TARGET': 0.15,
        'L1_LAMBDA': 0.1,
        'MAX_EM_ITER': 20,
    }

    # Data loading settings
    time_range = [0, 290_000]
    rebin_factor = 4
    full_path = os.path.join(args.data_path, args.file_name)

    print(f"Loading with time_range={time_range}, max_em_iterations={hparams['MAX_EM_ITER']}, rebin_factor={rebin_factor}")
    print("Data will be loaded to CPU memory, GPU used only for computations")

    # Load data
    S_t, expD, expMD, n_neurons, t_timebins = load_spike_data_hdf5(
        full_path, time_range=time_range, rebin_factor=rebin_factor
    )
    
    if rebin_factor > 1:
        hparams['DT'] *= rebin_factor
        print(f"Adjusted DT to {hparams['DT']:.4f} seconds due to rebinning")
    
    print(f"Data loaded: {n_neurons} neurons, {t_timebins} time bins")
    print(f"Data sparsity: {S_t.mean().item():.4f}")  # type: ignore
    
    # Run EM algorithm
    W_estimated, X_estimated = em_algorithm(S_t, n_neurons, t_timebins, hparams, device)
    
    # Compare with ground truth
    print("\n" + "="*60)
    print("COMPARING WITH GROUND TRUTH")
    print("="*60)
    metrics = compare_with_ground_truth(W_estimated, expD)
    
    # Save results
    W_estimated_np = W_estimated.cpu().numpy()
    X_estimated_np = X_estimated.cpu().numpy()
    
    expD['em_network_matrix'] = W_estimated_np
    expD['em_hidden_states'] = X_estimated_np
    expD['em_evaluation_metrics'] = metrics
    
    np.save("estimated_W_EM.npy", W_estimated_np)
    np.save("estimated_X_EM.npy", X_estimated_np)
    
    print(f"Estimated connectivity matrix W saved to 'estimated_W_EM.npy' with shape {W_estimated_np.shape}")
    print(f"Estimated hidden states X saved to 'estimated_X_EM.npy' with shape {X_estimated_np.shape}")

