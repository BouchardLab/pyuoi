#!/usr/bin/env python3
"""
Stage 1: Structure identification with L1 regularization.
Identifies which connections exist (sparsity pattern) - BINARY MASK ONLY.
Saves binary map of connections into xxx.struct.npy.

UNIFORM L1 REGULARIZATION:
The L1 regularization uses uniform weighting for all connections.
For each connection A[i,j] (from neuron i to neuron j), the L1 penalty is uniform:
  L1_weight[i,j] = 1.0

This means:
- All connections get the same L1 penalty weight
- Diagonal elements are excluded from L1 regularization
- Standard L1 regularization for sparsity induction

FIRING RATE WEIGHTED LOSS:
The loss function uses firing rates to weight the contribution of each neuron:
- Low firing neurons get higher weights in the loss calculation
- High firing neurons get lower weights in the loss calculation
- This helps balance the contribution of neurons with different activity levels

IMPLEMENTATION:
- Uses uniform L1 penalty matrix
- Excludes diagonal elements from regularization
- Standard thresholding based on learned weights
- Firing rate weighted Poisson negative log-likelihood loss
"""
'''
bash -c "source /usr/share/lmod/lmod/init/bash && module load pytorch && python3 fitXXX.py

 salloc -q interactive -C gpu  -t 4:00:00 -A m2043 -N 1
 module load pytorch 
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import sys
import os

import matplotlib.pyplot as plt
import argparse
from PlotterFitStruct import Plotter

# Check for GPU
if not torch.cuda.is_available():
    print("Error: This script requires a CUDA-enabled GPU environment. Aborting.")
    sys.exit(1)

# Use only GPU 0, even if more GPUs are available
device = torch.device('cuda:0')
torch.cuda.set_device(0)
print(f"Using device: {device}")
n_gpu = torch.cuda.device_count()
print(f"Number of GPUs available: {n_gpu}")
print(f"  GPU 0: {torch.cuda.get_device_name(0)} (selected)")
if n_gpu > 1:
    print(f"Note: {n_gpu-1} additional GPU(s) available but not used")

class PoissonGLMBlind(nn.Module):
    """Poisson GLM for neural connectivity estimation without Dale's principle."""
    def __init__(self, n_neurons, mask=None, init_A=None):
        super().__init__()
        self.n_neurons = n_neurons
        
        # Initialize parameters
        if init_A is not None:
            self.A = nn.Parameter(torch.tensor(init_A, dtype=torch.float32))
        else:
            # Standard initialization
            A_init = torch.randn(n_neurons, n_neurons) * 0.1
            self.A = nn.Parameter(A_init)
        
        # Bias terms
        self.B = nn.Parameter(torch.randn(n_neurons) * 0.1)
        
        # Mask for enforcing structure 
        if mask is not None:
            self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32))
        else:
            self.register_buffer('mask', torch.ones(n_neurons, n_neurons))
            
    def forward(self, Y_prev, dt=0.01):
        # Apply only mask constraints, no Dale's principle
        A_constrained = self.apply_constraints()
        
        # Compute spikes: lambda = exp(A @ Y_prev + B) * dt
        linear_pred = torch.addmm(self.B.unsqueeze(0), Y_prev, A_constrained.T)
        # Clip to prevent overflow
        linear_pred = torch.clamp(linear_pred, min=-10, max=10)
        spikes = torch.exp(linear_pred) * dt
        
        return spikes
    
    def apply_constraints(self):
        """Apply only mask constraints, no Dale's principle."""
        A_constrained = self.A * self.mask
        return A_constrained

def poisson_nll_loss(spikes, targets):
    """Negative log-likelihood for Poisson distribution."""
    # Avoid log(0) by adding small epsilon
    eps = 1e-8
    loss = -targets * torch.log(spikes + eps) + spikes
    return loss.mean()

def poisson_nll_loss_weighted(spikes, targets, firing_rates):
    """Negative log-likelihood for Poisson distribution with firing rate weighting."""
    # Avoid log(0) by adding small epsilon
    eps = 1e-8
    loss_per_neuron = -targets * torch.log(spikes + eps) + spikes
    
    # Weight by firing rates (inverse weighting: higher weight for low firing neurons)
    # Normalize firing rates to avoid extreme weights
    firing_rates_safe = torch.maximum(firing_rates, torch.tensor(0.1))
    weights = 1.0 / firing_rates_safe
    weights = weights / torch.mean(weights)  # Normalize to keep average weight = 1
    
    # Apply weights to each neuron's loss
    weighted_loss = loss_per_neuron * weights.unsqueeze(0)  # Broadcast weights to batch dimension
    
    return weighted_loss.mean()

def train_model_with_adaptive_L1(model, train_loader, val_loader, n_epochs, lr, L1_lambda=0.0, use_scheduler=False, firing_rates=None):
    """Train the Poisson GLM model with uniform L1 regularization and firing rate weighted loss."""
    batch_size = train_loader.batch_size
    print(f"Using 1 GPU for training with uniform L1, BS={batch_size}, target epochs={n_epochs}")
    
    # GPU utilization monitoring
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
    
    model.to(device)
    
    # Pre-allocate memory for better GPU utilization
    torch.cuda.empty_cache()
    print(f"GPU memory after model loading: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=n_epochs)

    # Create diagonal mask to exclude diagonal elements from L1 regularization
    n_neurons = model.n_neurons
    diag_mask = torch.eye(n_neurons, device=device).bool()
    off_diag_mask = ~diag_mask  # Mask for off-diagonal elements only

    # Use uniform L1 penalty (off-diagonal only)
    print("Using uniform L1 penalty")
    L1_weight_matrix = torch.ones(n_neurons, n_neurons).to(device)
    # Zero out diagonal elements
    L1_weight_matrix[diag_mask] = 0.0
    print(f"Diagonal elements excluded from L1 regularization")
    
    # Prepare firing rates for loss weighting
    if firing_rates is not None:
        firing_rates_tensor = torch.tensor(firing_rates, dtype=torch.float32).to(device)
        print(f"Using firing rate weighted loss")
        print(f"Firing rates range: [{torch.min(firing_rates_tensor):.3f}, {torch.max(firing_rates_tensor):.3f}]")
    else:
        firing_rates_tensor = None
        print("Using standard loss (no firing rate weighting)")

    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for Y_prev, Y_curr in train_loader:
            Y_prev = Y_prev.float().to(device)
            Y_curr = Y_curr.float().to(device)
            
            optimizer.zero_grad()
            spikes = model(Y_prev)
            
            # Use weighted loss if firing rates are provided
            if firing_rates_tensor is not None:
                loss = poisson_nll_loss_weighted(spikes, Y_curr, firing_rates_tensor)
            else:
                loss = poisson_nll_loss(spikes, Y_curr)
            
            # Add uniform L1 regularization (off-diagonal only)
            if L1_lambda > 0:
                A_constrained = model.apply_constraints()
                # Apply uniform L1 penalties only to off-diagonal elements
                weighted_L1_loss = L1_lambda * torch.sum(torch.abs(A_constrained) * L1_weight_matrix)
                loss = loss + weighted_L1_loss
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Y_prev, Y_curr in val_loader:
                Y_prev = Y_prev.float().to(device)
                Y_curr = Y_curr.float().to(device)
                spikes = model(Y_prev)
                
                # Use weighted loss if firing rates are provided
                if firing_rates_tensor is not None:
                    loss = poisson_nll_loss_weighted(spikes, Y_curr, firing_rates_tensor)
                else:
                    loss = poisson_nll_loss(spikes, Y_curr)
                    
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        if scheduler is not None:
            scheduler.step()
        
        # Print progress every 20 epochs with GPU monitoring
        if (epoch + 1) % 20 == 0:
            elapsed = time.time() - start_time
            gpu_mem = torch.cuda.memory_allocated(0) / 1e6 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{n_epochs}: Train={train_losses[-1]:.4f}, Val={val_losses[-1]:.4f}, LR={current_lr:.6f}, Time={elapsed:.1f}s, GPU={gpu_mem:.0f}MB")

    return train_losses, val_losses



def fit_structure_blind(Y_tensor, L1_lambda, n_epochs=100, target_sparsity=0.9, lr=0.001, batch_size=256, num_workers=4, firing_rates=None):
    """Stage 1: Identify network structure using L1 regularization without Dale's principle."""
    n_neurons = Y_tensor.shape[1]
    
    # Create dataset
    X = Y_tensor[:-1]
    y = Y_tensor[1:]
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Optimized DataLoader settings for GPU utilization
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=min(num_workers, 8), pin_memory=True, 
                             persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=min(num_workers, 4), pin_memory=True,
                           persistent_workers=True, prefetch_factor=2)
    
    # Train model with uniform L1 regularization
    print("\n--- Training Stage 1 model with uniform L1 regularization (NO Dale's principle) ---")
    print(f"DataLoader optimization: train_workers={min(num_workers, 8)}, val_workers={min(num_workers, 4)}, prefetch=4/2")
    
    model = PoissonGLMBlind(n_neurons).to(device)
    train_losses, val_losses = train_model_with_adaptive_L1(
        model, train_loader, val_loader, n_epochs, lr=lr, L1_lambda=L1_lambda, firing_rates=firing_rates, use_scheduler=True
    )
    
    # Extract learned matrix and bias terms
    A_learned = model.apply_constraints().detach().cpu().numpy()
    B_learned = model.B.detach().cpu().numpy()
    
    # Standard thresholding
    A_abs = np.abs(A_learned)
    n_total = n_neurons**2
    n_keep = int(n_total * (1 - target_sparsity))
    
    # Create diagonal mask
    diag_mask = np.eye(n_neurons, dtype=bool)
    off_diag_mask = ~diag_mask
    
    # Always include diagonal elements in the mask
    mask = diag_mask.copy()
    
    # For off-diagonal elements, find threshold that keeps top n_keep_off_diag connections
    n_off_diag = n_neurons * (n_neurons - 1)  # Number of off-diagonal elements
    n_keep_off_diag = n_keep - n_neurons  # Reserve n_neurons spots for diagonal elements
    
    if n_keep_off_diag > 0:
        # Get sorted off-diagonal weights (descending order)
        off_diag_weights = A_abs[off_diag_mask]
        sorted_off_diag = np.sort(off_diag_weights)[::-1]
        
        if n_keep_off_diag < len(sorted_off_diag):
            adaptive_threshold = sorted_off_diag[n_keep_off_diag-1]
        else:
            adaptive_threshold = 0.0
        
        # Apply threshold to off-diagonal elements
        off_diag_selected = A_abs >= adaptive_threshold
        mask = mask | off_diag_selected
    else:
        # If n_keep is less than n_neurons, only keep diagonal elements
        adaptive_threshold = 0.0
        print(f"Warning: n_keep ({n_keep}) < n_neurons ({n_neurons}), keeping only diagonal elements")
    
    actual_sparsity = 1 - np.sum(mask) / n_total
    print(f"Target sparsity: {target_sparsity*100:.1f}%, Achieved sparsity: {actual_sparsity*100:.1f}%")
    print(f"Adaptive threshold: {adaptive_threshold:.6f}")
    print(f"Identified {np.sum(mask)} non-zero connections out of {n_total} possible")
    print(f"  - Diagonal elements: {np.sum(diag_mask)} (always included)")
    print(f"  - Off-diagonal elements: {np.sum(mask & off_diag_mask)}")
    
    # Analyze detected connections by neuron index
    print(f"\nAnalyzing detected connections by neuron index:")
    outgoing_connections = np.sum(mask, axis=1)  # Total connections FROM each neuron
    incoming_connections = np.sum(mask, axis=0)  # Total connections TO each neuron
    
    for i in range(0, n_neurons, 5):  # Print every 5th neuron
        end_idx = min(i + 5, n_neurons)
        out_slice = outgoing_connections[i:end_idx]
        in_slice = incoming_connections[i:end_idx]
        conn_pairs = [f"{in_val:2d}/{out_val:2d}" for in_val, out_val in zip(in_slice, out_slice)]
        print(f"  Neurons {i:2d}-{end_idx-1:2d}: in/out: {conn_pairs}")
    
    # Check weight magnitudes by neuron region
    print(f"\nWeight magnitudes by neuron region:")
    for region_start in [0, 10, 20, 30]:
        if region_start >= n_neurons:
            break
        region_end = min(region_start + 10, n_neurons)
        region_weights = A_abs[region_start:region_end, :]
        max_weight = np.max(region_weights)
        mean_weight = np.mean(region_weights[region_weights > 0]) if np.any(region_weights > 0) else 0
        print(f"  Neurons {region_start:2d}-{region_end-1:2d}: max={max_weight:.6f}, mean={mean_weight:.6f}")
    
    print("--- End of Stage 1 ---")
    
    return mask, A_learned, B_learned, train_losses, val_losses

def preprocess_data_blind(Y):
    """Preprocess spike data without knowledge of excitatory/inhibitory identity."""
    print("\n=== Data Preprocessing (Blind) ===")
    print(f"Data shape: {Y.shape}")
    print(f"Total time steps: {Y.shape[0]}")
    n_samples = Y.shape[0] - 1
    print(f"Number of neurons: {Y.shape[1]}")
    print(f"Total number of training samples (time steps - 1): {n_samples}")
    
    # Calculate sparsity of data
    sparsity = 1 - np.count_nonzero(Y) / Y.size
    print(f"\nData sparsity: {sparsity*100:.1f}% zeros")
    
    return None

def load_data_and_rates(args):
    """
    Load spike data and firing rates from files.
    
    Args:
        args: Command line arguments containing dataName, dataPath, and num_samples
        
    Returns:
        dict: Dictionary containing:
            - Y: Spike data array (time_steps x n_neurons)
            - firing_rates: Array of firing rates for each neuron
            - n_neurons: Number of neurons
            - n_samples: Number of time samples used
    """
    # Construct file paths
    spikes_file = os.path.join(args.dataPath, f"{args.dataName}.spikes.npz")
    
    # Load spike data and firing rates
    print(f"Loading spike data from {spikes_file}")
    spike_data = np.load(spikes_file, allow_pickle=True)
    Y = spike_data['Y']
    firing_rates = spike_data['firing_rates']
    
    # Determine number of neurons
    n_neurons = Y.shape[1]
    original_steps = Y.shape[0]
    print(f"Loaded {original_steps} time steps for {n_neurons} neurons")
    print(f"Loaded firing rates: shape={firing_rates.shape}")
    
    # Clip data if num_samples is specified
    if args.num_samples is not None and args.num_samples > 0:
        if args.num_samples > original_steps:
            print(f"Warning: Requested {args.num_samples} samples but only {original_steps} available. Using all available data.")
            args.num_samples = original_steps
        else:
            Y = Y[:args.num_samples, :]
            print(f"Clipped data to {args.num_samples} time samples (from {original_steps})")
    
    n_samples = Y.shape[0]
    print(f"Using {n_samples} time steps for fitting")
    
    return {
        'Y': Y,
        'firing_rates': firing_rates,
        'n_neurons': n_neurons,
        'n_samples': n_samples
    }

def analyze_edge_detection(A_true, mask_detected, num_excite):
    """Analyze edge detection performance separately for excitatory and inhibitory connections."""
    
    # Create true edge mask (non-zero elements) - exclude diagonal
    mask_true = np.abs(A_true) > 1e-10
    diag_mask = np.eye(mask_true.shape[0], dtype=bool)
    mask_true[diag_mask] = False  # Exclude diagonal elements from evaluation
    
    # Create masks for excitatory and inhibitory neurons (rows)
    exc_mask = np.zeros_like(mask_true)
    exc_mask[:num_excite, :] = True
    inh_mask = np.zeros_like(mask_true)
    inh_mask[num_excite:, :] = True
    
    # Exclude diagonal from neuron type masks
    exc_mask[diag_mask] = False
    inh_mask[diag_mask] = False
    
    def calculate_metrics(true_edges, detected_edges):
        """Calculate TP, FP, TN, FN and derived metrics."""
        tp = np.sum(true_edges & detected_edges)
        fp = np.sum(~true_edges & detected_edges)
        tn = np.sum(~true_edges & ~detected_edges)
        fn = np.sum(true_edges & ~detected_edges)
        
        # Derived metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        
        return {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1_score': f1_score,
            'specificity': specificity, 'accuracy': accuracy
        }
    
    # Overall analysis (off-diagonal only)
    overall_metrics = calculate_metrics(mask_true, mask_detected)
    
    # Excitatory analysis (excitatory rows, off-diagonal only)
    exc_true = mask_true & exc_mask
    exc_detected = mask_detected & exc_mask
    exc_metrics = calculate_metrics(exc_true, exc_detected)
    
    # Debug: Print total excitatory weights for verification
    total_excit_weights = np.sum(np.abs(A_true[:num_excite, :]) > 1e-10)
    print(f"\nDebug: Total excitatory weights (threshold 1e-10): {total_excit_weights}")
    print(f"Debug: Excitatory true edges (off-diagonal): {np.sum(exc_true)}")
    print(f"Debug: Excitatory TP + FN = {exc_metrics['tp'] + exc_metrics['fn']}")
    
    # Inhibitory analysis (inhibitory rows, off-diagonal only)
    inh_true = mask_true & inh_mask
    inh_detected = mask_detected & inh_mask
    inh_metrics = calculate_metrics(inh_true, inh_detected)
    
    # Print results
    print("\n=== Edge Detection Analysis (Off-Diagonal Only) ===")
    print("Overall (off-diagonal):")
    print(f"  TP: {overall_metrics['tp']}, FP: {overall_metrics['fp']}, TN: {overall_metrics['tn']}, FN: {overall_metrics['fn']}")
    print(f"  Precision: {overall_metrics['precision']:.3f}, Recall: {overall_metrics['recall']:.3f}")
    print(f"  Accuracy: {overall_metrics['accuracy']:.3f}, F1 Score: {overall_metrics['f1_score']:.3f}")
    
    print("\nExcitatory rows (off-diagonal):")
    print(f"  TP: {exc_metrics['tp']}, FP: {exc_metrics['fp']}, TN: {exc_metrics['tn']}, FN: {exc_metrics['fn']}")
    print(f"  Precision: {exc_metrics['precision']:.3f}, Recall: {exc_metrics['recall']:.3f}")
    print(f"  Accuracy: {exc_metrics['accuracy']:.3f}, F1 Score: {exc_metrics['f1_score']:.3f}")
    
    print("\nInhibitory rows (off-diagonal):")
    print(f"  TP: {inh_metrics['tp']}, FP: {inh_metrics['fp']}, TN: {inh_metrics['tn']}, FN: {inh_metrics['fn']}")
    print(f"  Precision: {inh_metrics['precision']:.3f}, Recall: {inh_metrics['recall']:.3f}")
    print(f"  Accuracy: {inh_metrics['accuracy']:.3f}, F1 Score: {inh_metrics['f1_score']:.3f}")
    
    return {
        'overall': overall_metrics,
        'excitatory': exc_metrics,
        'inhibitory': inh_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Identify network structure from spike data")
    parser.add_argument("--dataName", type=str, default="dale_2aee70", help="Base name for input/output files")
    parser.add_argument("--dataPath", type=str, default="out/", help="Path to data directory")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of time samples to use (None=use all, positive=clip to this many)")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs for structure identification")
    parser.add_argument("--batch_size", type=int, default=2048*8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--L1_lambda", type=float, default=1.0e-06, help="L1 regularization parameter")
    parser.add_argument("--target_sparsity", type=float, default=0.8, help="Target sparsity level (0.9 = 90% zeros)")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of data loader workers")
    
    # Evaluation parameters
    parser.add_argument("--disable_evaluate", action="store_true", help="Disable evaluation against ground truth")
    
    # Plotting parameters
    parser.add_argument('-p',"--showPlots", type=str, default="ab", nargs='+', help="Which plots to show: a=structure_results, b=weight_distributions (rate_analysis moved to sim_dale_poissonV4.py)")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument("--formatVenue", type=str, default="prod", help="Plot format venue")
    parser.add_argument("--verb", type=int, default=1, help="Verbosity level")
    
    args = parser.parse_args() 
    args.showPlots=''.join(args.showPlots)

    print("\nStructure identification with configuration:")
    print(vars(args))
    print("")
    overall_start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(args.dataPath, exist_ok=True)
    
    # Construct file paths
    truth_file = os.path.join(args.dataPath, f"{args.dataName}.truth.npz")
    struct_file = os.path.join(args.dataPath, f"{args.dataName}.struct.npy")
    
    # Load data and rates using the factored function
    data_dict = load_data_and_rates(args)
    Y = data_dict['Y']
    n_neurons = data_dict['n_neurons']
    n_samples = data_dict['n_samples']
    
    
    # Preprocess and analyze data
    preprocess_data_blind(Y)
    print(f"Data loading and preprocessing took {time.time() - overall_start_time:.1f} seconds")
    
    # Convert to PyTorch tensor
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    
    # Use provided L1 parameter
    best_L1 = args.L1_lambda
    print(f"\nUsing L1 parameter: {best_L1:.2e}")
    
    # Structure identification
    print("\n=== Structure Identification ===")
    stage_start_time = time.time()
    mask, A_stage1, B_stage1, train_losses, val_losses = fit_structure_blind(
        Y_tensor, best_L1, n_epochs=args.n_epochs, target_sparsity=args.target_sparsity, lr=args.lr, batch_size=args.batch_size, num_workers=args.num_workers, firing_rates=data_dict['firing_rates']
    )
    stage1_time = time.time() - stage_start_time
    print(f"Structure identification completed in {stage1_time:.1f} seconds")
    
    # Save structure results
    save_dict = {
        'mask': mask,
        'A_stage1': A_stage1,
        'B_stage1': B_stage1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_L1': best_L1,
        'firing_rates': data_dict['firing_rates'],
        'args': vars(args)
    }
    
    np.save(struct_file, save_dict)
    print(f"\nStructure results saved to {struct_file}")
    
    # Evaluation against ground truth (if requested)
    if not args.disable_evaluate:
        print(f"\n=== Loading Ground Truth for Evaluation ===")
        print(f"Loading ground truth from {truth_file}")
        truth_data = np.load(truth_file, allow_pickle=True)
        A_true = truth_data['A']
        conf = truth_data['conf'].item()
        num_excite = conf['num_excite']
        
        # Analyze edge detection performance
        edge_analysis = analyze_edge_detection(A_true, mask, num_excite)
        
        # Plot results will be handled by PlotterFitStruct
        
        # Save evaluation results
        eval_file = os.path.join(args.dataPath, f"{args.dataName}_struct_eval.npz")
        eval_dict = {
            'edge_analysis': edge_analysis,
            'num_excite': num_excite
        }
        np.savez(eval_file, **eval_dict)
        print(f"Evaluation results saved to {eval_file}")
    
    # Final summary
    print("\n=== Final Summary ===")
    print(f"Total processing time: {time.time() - overall_start_time:.1f} seconds")
    print(f"Identified {np.sum(mask)} connections out of {n_neurons**2} possible")
    print(f"Structure file saved: {struct_file}")
    if not args.disable_evaluate:
        print(f"Evaluation completed and saved")

    #--------------------------------
    # ....  plotting ........
    # Initialize metadata for plotting
    num_excite_plot = 0
    A_true_plot = None
    if not args.disable_evaluate:
        num_excite_plot = num_excite
        A_true_plot = A_true
    
    MD={'num_excit_neur':num_excite_plot,'short_name':args.dataName,'num_samples':n_samples,
        'initial_lr':args.lr,'train_time_min':stage1_time/60.0,'num_epochs':args.n_epochs}
    args.prjName=args.dataName+'_struct'
    args.outPath=args.dataPath
    plot=Plotter(args)
    
    if 'a' in args.showPlots and not args.disable_evaluate:
        plot.structure_results(A_true_plot, mask, train_losses, val_losses, MD, figId=1, edge_analysis=edge_analysis)
    if 'b' in args.showPlots and not args.disable_evaluate:
        plot.weight_distributions(A_true_plot, mask, MD, figId=2)
    if 'c' in args.showPlots and not args.disable_evaluate:
        plot.weight_reconstruction_results(A_true_plot, A_stage1, train_losses, val_losses, MD, figId=3)
    if 'd' in args.showPlots and not args.disable_evaluate:
        plot.weight_category_analysis(A_true_plot, A_stage1, MD, figId=4)

    plot.display_all()
    print('M:done')

if __name__ == "__main__":
    main() 
