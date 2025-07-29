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

import argparse
from UtilFitPoisson import (
    load_data_and_rates, 
    preprocess_data_blind, 
    create_data_loaders, 
    save_structure_results
)

# Global device variable
device = None
n_gpu = 0

def check_gpu_availability():
    """Check GPU availability and set up device configuration."""
    global device, n_gpu
    
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
    def __init__(self, n_neurons):
        super().__init__()
        self.n_neurons = n_neurons
        
        # Standard initialization
        A_init = torch.randn(n_neurons, n_neurons) * 0.1
        self.A = nn.Parameter(A_init)
        
        # Bias terms
        self.B = nn.Parameter(torch.randn(n_neurons) * 0.1)
        
        # Mask for enforcing structure 
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

def poisson_nll_loss(spikes, targets, firing_rates):
    """
    Negative log-likelihood for Poisson distribution, with firing rate weighting.

    For each neuron i:
        loss_i = - w_i * targets_i * log(spikes_i + eps) + w_i * spikes_i

    where:
        - targets: [batch_size, n_neurons], observed spike counts
        - spikes:  [batch_size, n_neurons], predicted rates
        - firing_rates: [n_neurons], mean firing rate per neuron
        - w_i = 1 / max(firing_rates_i, 0.1), normalized so mean(w) = 1
        - eps: small constant for numerical stability

    Returns:
        Scalar loss (averaged over batch and neurons)
    """
    eps = 1e-10  # was 1e-8
    firing_rates_safe = torch.maximum(firing_rates, torch.tensor(0.1, device=firing_rates.device))
    weights = 1.0 / firing_rates_safe
    weights = weights / torch.mean(weights)
    weights = weights.unsqueeze(0)  # shape [1, n_neurons] for broadcasting
    loss = -weights * targets * torch.log(spikes + eps) + weights * spikes
    return loss.mean()

def train_model_with_adaptive_L1(model, train_loader, val_loader, n_epochs, lr, L1_alpha=0.0, use_scheduler=False, firing_rates=None):
    """Train the Poisson GLM model with uniform L1 regularization and firing rate weighted loss."""
    batch_size = train_loader.batch_size
    print(f"Using 1 GPU for training with uniform L1, BS={batch_size}, target epochs={n_epochs}")
    
    # GPU utilization monitoring
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB, Initial allocated: {torch.cuda.memory_allocated(0) / 1e6:.1f} MB")
    
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
            
            # Use combined loss function with optional firing rate weighting
            loss = poisson_nll_loss(spikes, Y_curr, firing_rates_tensor)
            
            # Add uniform L1 regularization (off-diagonal only)
            if L1_alpha > 0:
                A_constrained = model.apply_constraints()
                # Apply uniform L1 penalties only to off-diagonal elements
                weighted_L1_loss = L1_alpha * torch.sum(torch.abs(A_constrained) * L1_weight_matrix)
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
                
                # Use combined loss function with optional firing rate weighting
                loss = poisson_nll_loss(spikes, Y_curr, firing_rates_tensor)
                    
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        if scheduler is not None:
            scheduler.step()
        
        # Print progress every 20 epochs with GPU monitoring
        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            gpu_mem = torch.cuda.memory_allocated(0) / 1e6 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{n_epochs}: Train={train_losses[-1]:.4f}, Val={val_losses[-1]:.4f}, LR={current_lr:.6f}, Time={elapsed:.1f}s, GPU={gpu_mem:.0f}MB")

    return train_losses, val_losses



def fit_structure_blind(Y_tensor, L1_alpha, n_epochs=100, lr=0.001, batch_size=256, num_workers=4, firing_rates=None):
    """Stage 1: Identify network structure using L1 regularization without Dale's principle."""
    n_neurons = Y_tensor.shape[1]
    
    # Create data loaders using utility function
    train_loader, val_loader = create_data_loaders(Y_tensor, batch_size, num_workers)
    
    # Train model with uniform L1 regularization
    print("\n--- Training Stage 1 model with uniform L1 regularization (NO Dale's principle) ---")
    
    model = PoissonGLMBlind(n_neurons).to(device)
    train_losses, val_losses = train_model_with_adaptive_L1(
        model, train_loader, val_loader, n_epochs, lr=lr, L1_alpha=L1_alpha, firing_rates=firing_rates, use_scheduler=True
    )
    
    # Extract learned matrix and bias terms
    A_learned = model.apply_constraints().detach().cpu().numpy()
    B_learned = model.B.detach().cpu().numpy()
    
    print("--- End of Stage 1 ---")
    
    # Return results as bigD dictionary
    bigD = {
        'A_stage1': A_learned,
        'B_stage1': B_learned,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'firing_rates': firing_rates,
        'n_neurons': n_neurons
    }
    
    return bigD

#########################
#  MAIN
#########################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataName", type=str, default="dale_2aee70", help="Base name for input/output files")
    parser.add_argument("--dataPath", type=str, default="out/", help="Path to data directory")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of time samples to use (None=use all, positive=clip to this many)")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs for structure identification")
    parser.add_argument("--batch_size", type=int, default=2048*8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--L1_alpha", type=float, default=2.0e-07, help="L1 regularization parameter")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of data loader workers")
    
    args = parser.parse_args()

    print("\nStructure identification with configuration:")
    print(vars(args))
    print("")
    np.set_printoptions(precision=3, suppress=True)
    
    # Check GPU availability
    check_gpu_availability()
    
    overall_start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(args.dataPath, exist_ok=True)
    
    # Construct file paths
    truth_file = os.path.join(args.dataPath, f"{args.dataName}.truth.npz")
    struct_file = os.path.join(args.dataPath, f"{args.dataName}.struct.npy")
    
    # Load data and rates using the utility function
    data_dict = load_data_and_rates(args)
    Y = data_dict['Y']
    n_neurons = data_dict['n_neurons']
    n_samples = data_dict['n_samples']
    firing_rates=data_dict['firing_rates']
    
    # Preprocess and analyze data
    preprocess_data_blind(Y)
    print(f"Data loading and preprocessing took {time.time() - overall_start_time:.1f} seconds")
    
    # Convert to PyTorch tensor
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    
    # Structure identification
    print("\n=== Structure Identification ===")
    stage_start_time = time.time()
    stage1_bigD = fit_structure_blind(
        Y_tensor, args.L1_alpha, n_epochs=args.n_epochs, lr=args.lr, batch_size=args.batch_size, num_workers=args.num_workers, firing_rates=firing_rates )
    stage1_time = time.time() - stage_start_time
    print(f"Structure identification completed in {stage1_time:.1f} seconds")
    
    # Extract values from bigD for saving
    A_stage1 = stage1_bigD['A_stage1']
    B_stage1 = stage1_bigD['B_stage1']
    train_losses = stage1_bigD['train_losses']
    val_losses = stage1_bigD['val_losses']
    
    # Save structure results using utility function
    metadata = {
        'num_samples': n_samples,
        'initial_lr': args.lr,
        'train_time_min': stage1_time/60.0,
        'num_epochs': args.n_epochs
    }
    save_structure_results(struct_file, A_stage1, B_stage1, train_losses, val_losses, data_dict['firing_rates'], args, metadata)
    
    # Final summary
    print("\n=== Final Summary ===")
    print(f"Total processing time: {time.time() - overall_start_time:.1f} seconds")
    print(f"Structure file saved: {struct_file}")
    print("Model fitting completed successfully")

    print('\n  ./eval2_fitPoisson.py  --dataName %s ' % (args.dataName))
    
if __name__ == "__main__":
    main() 
