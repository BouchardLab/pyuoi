#!/usr/bin/env python3
"""
Inverse problem solver for recovering connectivity matrix A from spike data Y.
Uses a two-stage approach: structure identification with L1 regularization, 
followed by parameter refinement.

=== TWO-STAGE TRAINING APPROACH ===

STAGE 1: Structure Identification with L1 Regularization
- Purpose: Identify which connections exist (sparsity pattern) - BINARY MASK ONLY
- Loss: Poisson negative log-likelihood + L1 penalty on connection weights
  Loss = -sum(Y_t * log(λ_t) - λ_t) + λ_L1 * sum(|A_ij|)
  where λ_t = exp(A @ Y_{t-1} + B) * dt (Poisson rates)
- L1 regularization encourages sparsity by penalizing non-zero weights
- Learning rate: linear decay from --lr_stage1 to 10% over epochs
- Output: ONLY binary mask indicating which connections exist (ignore signs/magnitudes)

STAGE 2: Parameter Estimation with Fixed Structure
- Purpose: Estimate signs and magnitudes for connections identified in Stage 1
- Loss: Pure Poisson negative log-likelihood (no L1 penalty)
  Loss = -sum(Y_t * log(λ_t) - λ_t)
- Only weights allowed by Stage 1 binary mask are updated
- Learning rate: linear decay from --lr_stage2 to 10% over epochs
- Determines both signs and magnitudes of existing connections
- Output: Final connectivity matrix A_stage2 with proper signs/magnitudes

Clean separation: Stage 1 finds WHICH connections exist, Stage 2 finds their signs/magnitudes.
Both stages are blind to Dale's principle. Ground truth only used for final evaluation.
"""
'''
bash -c "source /usr/share/lmod/lmod/init/bash && module load pytorch && python3 fitXXX.py

 salloc -q interactive -C gpu  -t 4:00:00 -A m2043 -N 1
 module load pytorch 
 
 # Set to use only GPU 0 explicitly
CUDA_VISIBLE_DEVICES=0 ./fit_model.py --simuName daleM40may27_simu-lya37b --epochs 200 --learning_rate 0.001 --batch_size 512

'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import sys
import os
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import argparse

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
    def __init__(self, n_neurons, mask=None, init_A=None, firing_rates=None):
        super().__init__()
        self.n_neurons = n_neurons
        
        # Initialize parameters
        if init_A is not None:
            self.A = nn.Parameter(torch.tensor(init_A, dtype=torch.float32))
        else:
            # Firing-rate-aware initialization for better low-firing neuron representation
            if firing_rates is not None:
                # Initialize with larger weights for low-firing neurons
                firing_rates_np = firing_rates.cpu().numpy() if torch.is_tensor(firing_rates) else firing_rates
                # Create initialization scale matrix - higher for low firing rates
                median_rate = np.median(firing_rates_np)
                scale_matrix = np.sqrt(median_rate / np.maximum(firing_rates_np, 0.1))
                # Broadcast to (from_neuron, to_neuron) - scale by from_neuron firing rate
                init_scale = np.outer(scale_matrix, np.ones(n_neurons))
                A_init = torch.randn(n_neurons, n_neurons) * 0.1 * torch.tensor(init_scale, dtype=torch.float32)
                print(f"Firing-rate-aware initialization: scale range [{np.min(init_scale):.2f}, {np.max(init_scale):.2f}]")
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
        
        # Compute rates: lambda = exp(A @ Y_prev + B) * dt
        linear_pred = torch.addmm(self.B.unsqueeze(0), Y_prev, A_constrained.T)
        # Clip to prevent overflow
        linear_pred = torch.clamp(linear_pred, min=-10, max=10)
        rates = torch.exp(linear_pred) * dt
        
        return rates
    
    def apply_constraints(self):
        """Apply only mask constraints, no Dale's principle."""
        A_constrained = self.A * self.mask
        return A_constrained

class PoissonGLM(nn.Module):
    """Poisson GLM for neural connectivity estimation."""
    def __init__(self, n_neurons, num_excite, mask=None, init_A=None):
        super().__init__()
        self.n_neurons = n_neurons
        self.num_excite = num_excite
        self.num_inhib = n_neurons - num_excite
        
        # Initialize parameters
        if init_A is not None:
            self.A = nn.Parameter(torch.tensor(init_A, dtype=torch.float32))
        else:
            # Random initialization respecting Dale's principle
            A_init = torch.zeros(n_neurons, n_neurons)
            # Excitatory connections (positive)
            A_init[:num_excite, :] = torch.abs(torch.randn(num_excite, n_neurons) * 0.01)
            # Inhibitory connections (negative)
            A_init[num_excite:, :] = -torch.abs(torch.randn(self.num_inhib, n_neurons) * 0.01)
            self.A = nn.Parameter(A_init)
        
        # Bias terms
        self.B = nn.Parameter(torch.randn(n_neurons) * 0.1)
        
        # Mask for enforcing structure (used in stage 2)
        if mask is not None:
            self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32))
        else:
            self.register_buffer('mask', torch.ones(n_neurons, n_neurons))
            
    def forward(self, Y_prev, dt=0.01):
        # Apply mask and Dale's principle constraints
        A_constrained = self.apply_constraints()
        
        # Compute rates: lambda = exp(A @ Y_prev + B) * dt
        # Use more efficient matrix multiplication
        linear_pred = torch.addmm(self.B.unsqueeze(0), Y_prev, A_constrained.T)
        # Clip to prevent overflow
        linear_pred = torch.clamp(linear_pred, min=-10, max=10)
        rates = torch.exp(linear_pred) * dt
        
        return rates
    
    def apply_constraints(self):
        """Apply Dale's principle and mask constraints."""
        A_masked = self.A * self.mask
        
        # Enforce Dale's principle without in-place operations
        A_exc = torch.abs(A_masked[:self.num_excite, :])
        A_inh = -torch.abs(A_masked[self.num_excite:, :])
        
        A_constrained = torch.cat([A_exc, A_inh], dim=0)
        
        return A_constrained

def poisson_nll_loss(rates, targets):
    """Negative log-likelihood for Poisson distribution."""
    # Avoid log(0) by adding small epsilon
    eps = 1e-8
    loss = -targets * torch.log(rates + eps) + rates
    return loss.mean()

def train_model(model, train_loader, val_loader, n_epochs, lr, l1_lambda=0.0, use_scheduler=False):
    """Train the Poisson GLM model."""
    batch_size = train_loader.batch_size
    print(f"Using 1 GPU for training, BS={batch_size}, target epochs={n_epochs}")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=n_epochs)

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
            rates = model(Y_prev)
            loss = poisson_nll_loss(rates, Y_curr)
            
            # Add L1 regularization if specified
            if l1_lambda > 0:
                A_constrained = model.apply_constraints()
                l1_loss = l1_lambda * torch.abs(A_constrained).sum()
                loss = loss + l1_loss
            
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
                rates = model(Y_prev)
                loss = poisson_nll_loss(rates, Y_curr)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            log_msg = f"Epoch {epoch}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}"
            if scheduler:
                log_msg += f", LR = {optimizer.param_groups[0]['lr']:.6f}"
            log_msg += f", Elapsed={elapsed:.1f}s"
            print(log_msg)

        if scheduler:
            scheduler.step()
    
    return train_losses, val_losses

def train_model_with_adaptive_l1(model, train_loader, val_loader, n_epochs, lr, l1_lambda=0.0, firing_rates=None, use_scheduler=False, 
                                l1_rate_power=1.5, disable_adaptive_l1=False):
    """Train the Poisson GLM model with firing-rate-aware L1 regularization."""
    batch_size = train_loader.batch_size
    print(f"Using 1 GPU for training with adaptive L1, BS={batch_size}, target epochs={n_epochs}")
    
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

    # Prepare firing rate normalization for L1 penalty (off-diagonal only)
    if firing_rates is not None and not disable_adaptive_l1:
        firing_rates_tensor = firing_rates.to(device)
        # Create normalization matrix for L1 penalty: (from_neuron x to_neuron)
        rate_norm_matrix = torch.sqrt(torch.outer(firing_rates_tensor, firing_rates_tensor))
        rate_norm_matrix = torch.maximum(rate_norm_matrix, torch.tensor(0.1))  # Avoid division by very small numbers
        # Invert for penalty - low firing rate connections get much lower L1 penalty
        l1_weight_matrix = 1.0 / torch.pow(rate_norm_matrix, l1_rate_power)  # Use configurable power
        # Normalize so the average penalty remains the same
        l1_weight_matrix = l1_weight_matrix / torch.mean(l1_weight_matrix)
        

        
        # Zero out diagonal elements in L1 weight matrix (they won't be penalized)
        l1_weight_matrix[diag_mask] = 0.0
        
        # Show L1 penalty statistics (off-diagonal only)
        l1_weight_off_diag = l1_weight_matrix[off_diag_mask]
        l1_mean = torch.mean(l1_weight_off_diag).item()
        l1_min = torch.min(l1_weight_off_diag).item()
        l1_max = torch.max(l1_weight_off_diag).item()
        print(f"L1 penalty matrix (off-diagonal): mean={l1_mean:.3f}, range=[{l1_min:.3f}, {l1_max:.3f}]")
        print(f"Diagonal elements excluded from L1 regularization")
    else:
        if disable_adaptive_l1:
            print("Using uniform L1 penalty (firing-rate-aware regularization disabled)")
        else:
            print("No firing rates provided, using uniform L1 penalty")
        l1_weight_matrix = torch.ones(n_neurons, n_neurons).to(device)
        # Zero out diagonal elements
        l1_weight_matrix[diag_mask] = 0.0
        print(f"Diagonal elements excluded from L1 regularization")

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
            rates = model(Y_prev)
            loss = poisson_nll_loss(rates, Y_curr)
            
            # Add firing-rate-aware L1 regularization (off-diagonal only)
            if l1_lambda > 0:
                A_constrained = model.apply_constraints()
                # Apply neuron-specific L1 penalties only to off-diagonal elements
                weighted_l1_loss = l1_lambda * torch.sum(torch.abs(A_constrained) * l1_weight_matrix)
                loss = loss + weighted_l1_loss
            
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
                rates = model(Y_prev)
                loss = poisson_nll_loss(rates, Y_curr)
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

def cross_validate_l1_blind(Y_tensor, l1_values, batch_size, n_splits=5, n_epochs=50, num_workers=4, data_fraction=1.0):
    """Cross-validate to find optimal L1 regularization parameter without Dale's principle."""
    n_neurons = Y_tensor.shape[1]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_results = {l1: [] for l1 in l1_values}
    
    # Prepare full dataset for splitting
    X = Y_tensor[:-1]
    y = Y_tensor[1:]
    
    # Use only a fraction of the data for cross-validation
    if data_fraction < 1.0:
        n_samples = len(X)
        n_use = int(n_samples * data_fraction)
        print(f"Using {data_fraction*100:.1f}% of data for cross-validation: {n_use}/{n_samples} samples")
        # Randomly sample indices
        indices = torch.randperm(n_samples)[:n_use]
        X = X[indices]
        y = y[indices]
    
    for l1_lambda in l1_values:
        fold_losses = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            # Create datasets
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            
            # Train model without Dale's principle
            print(f"\n[CV] L1={l1_lambda:.2e}, Fold={fold+1}/{n_splits}")
            model = PoissonGLMBlind(n_neurons).to(device)
            _, val_losses = train_model(model, train_loader, val_loader, n_epochs, lr=0.001, l1_lambda=l1_lambda)
            
            # Store best validation loss
            fold_losses.append(min(val_losses))
        
        cv_results[l1_lambda] = fold_losses
    
    # Find best L1 value
    mean_scores = {l1: np.mean(scores) for l1, scores in cv_results.items()}
    best_l1 = min(mean_scores, key=mean_scores.get)
    
    return best_l1, cv_results

def cross_validate_l1(Y_tensor, num_excite, l1_values, batch_size, n_splits=5, n_epochs=50, num_workers=4):
    """Cross-validate to find optimal L1 regularization parameter."""
    n_neurons = Y_tensor.shape[1]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_results = {l1: [] for l1 in l1_values}
    
    # Prepare full dataset for splitting
    X = Y_tensor[:-1]
    y = Y_tensor[1:]
    
    for l1_lambda in l1_values:
        fold_losses = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            # Create datasets
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            
            # Train model
            print(f"\n[CV] L1={l1_lambda:.2e}, Fold={fold+1}/{n_splits}")
            model = PoissonGLM(n_neurons, num_excite).to(device)
            _, val_losses = train_model(model, train_loader, val_loader, n_epochs, lr=0.001, l1_lambda=l1_lambda)
            
            # Store best validation loss
            fold_losses.append(min(val_losses))
        
        cv_results[l1_lambda] = fold_losses
    
    # Find best L1 value
    mean_scores = {l1: np.mean(scores) for l1, scores in cv_results.items()}
    best_l1 = min(mean_scores, key=mean_scores.get)
    
    return best_l1, cv_results

def identify_structure_blind(Y_tensor, l1_lambda, n_epochs=100, target_sparsity=0.9, lr=0.001, batch_size=256, num_workers=4,
                           l1_rate_power=1.5, disable_adaptive_l1=False):
    """Stage 1: Identify network structure using L1 regularization without Dale's principle."""
    n_neurons = Y_tensor.shape[1]
    
    # Calculate firing rates for neuron-specific normalization
    firing_rates = torch.sum(Y_tensor, dim=0) / (Y_tensor.shape[0] * 0.01)  # 10ms bins
    
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
    
    # Train model with firing-rate-aware L1 regularization
    print("\n--- Training Stage 1 model with firing-rate-aware regularization (NO Dale's principle) ---")
    print(f"DataLoader optimization: train_workers={min(num_workers, 8)}, val_workers={min(num_workers, 4)}, prefetch=4/2")
    print(f"To monitor GPU utilization in real-time, run in another terminal:")
    print(f"  watch -n 1 nvidia-smi")
    print(f"  or: nvtop")
    model = PoissonGLMBlind(n_neurons, firing_rates=firing_rates).to(device)
    train_losses, val_losses = train_model_with_adaptive_l1(
        model, train_loader, val_loader, n_epochs, lr=lr, l1_lambda=l1_lambda, 
        firing_rates=firing_rates, use_scheduler=True,
        l1_rate_power=l1_rate_power, disable_adaptive_l1=disable_adaptive_l1
    )
    
    # Extract learned matrix
    A_learned = model.apply_constraints().detach().cpu().numpy()
    
    # Firing-rate-aware adaptive thresholding
    A_abs = np.abs(A_learned)
    n_total = n_neurons**2
    n_keep = int(n_total * (1 - target_sparsity))
    
    # Normalize weights by firing rates for fair comparison
    print("Applying firing-rate-aware adaptive thresholding...")
    firing_rates_np = firing_rates.cpu().numpy()
    
    # Create firing rate normalization matrix (from_neuron x to_neuron)
    rate_norm_matrix = np.sqrt(np.outer(firing_rates_np, firing_rates_np))
    rate_norm_matrix = np.maximum(rate_norm_matrix, 0.1)  # Avoid division by very small numbers
    
    # Normalize weights by firing rate product for fairer comparison
    A_normalized = A_abs / rate_norm_matrix
    
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
        off_diag_weights = A_normalized[off_diag_mask]
        sorted_off_diag = np.sort(off_diag_weights)[::-1]
        
        if n_keep_off_diag < len(sorted_off_diag):
            adaptive_threshold_norm = sorted_off_diag[n_keep_off_diag-1]
        else:
            adaptive_threshold_norm = 0.0
        
        # Apply threshold to off-diagonal elements
        off_diag_selected = A_normalized >= adaptive_threshold_norm
        mask = mask | off_diag_selected
    else:
        # If n_keep is less than n_neurons, only keep diagonal elements
        adaptive_threshold_norm = 0.0
        print(f"Warning: n_keep ({n_keep}) < n_neurons ({n_neurons}), keeping only diagonal elements")
    
    # Calculate equivalent threshold for original weights (for reporting)
    adaptive_threshold = adaptive_threshold_norm * np.mean(rate_norm_matrix)
    
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
        region_weights_norm = A_normalized[region_start:region_end, :]
        max_weight = np.max(region_weights)
        mean_weight = np.mean(region_weights[region_weights > 0]) if np.any(region_weights > 0) else 0
        max_weight_norm = np.max(region_weights_norm)
        mean_weight_norm = np.mean(region_weights_norm[region_weights_norm > 0]) if np.any(region_weights_norm > 0) else 0
        print(f"  Neurons {region_start:2d}-{region_end-1:2d}: raw_max={max_weight:.6f}, norm_max={max_weight_norm:.6f}")
        print(f"                           raw_mean={mean_weight:.6f}, norm_mean={mean_weight_norm:.6f}")
    
    # Show firing rate normalization effect
    print(f"\nFiring rate normalization summary:")
    print(f"  Normalization matrix range: [{np.min(rate_norm_matrix):.2f}, {np.max(rate_norm_matrix):.2f}]")
    print(f"  Original threshold (global): {np.max(A_abs[mask]):.6f}")
    print(f"  Normalized threshold: {adaptive_threshold_norm:.6f}")
    print(f"  Equivalent average threshold: {adaptive_threshold:.6f}")
    
    print("--- End of Stage 1 (Firing-Rate-Aware) ---")
    
    return mask, A_learned, train_losses, val_losses

def identify_structure(Y_tensor, num_excite, l1_lambda, n_epochs=100, threshold=1e-4, lr=0.001, batch_size=256, num_workers=4):
    """Stage 1: Identify network structure using L1 regularization."""
    n_neurons = Y_tensor.shape[1]
    
    # Create dataset
    X = Y_tensor[:-1]
    y = Y_tensor[1:]
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Train model with L1 regularization
    print("\n--- Training Stage 1 model with LR scheduler ---")
    model = PoissonGLM(n_neurons, num_excite).to(device)
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, n_epochs, lr=lr, l1_lambda=l1_lambda, use_scheduler=True
    )
    
    # Extract learned matrix and create mask
    A_learned = model.apply_constraints().detach().cpu().numpy()
    mask = np.abs(A_learned) > threshold
    
    print(f"Identified {np.sum(mask)} non-zero connections out of {n_neurons**2} possible")
    print(f"Sparsity: {100 * (1 - np.sum(mask) / (n_neurons**2)):.1f}%")
    print("--- End of Stage 1 ---")
    
    return mask, A_learned, train_losses, val_losses

def estimate_parameters_with_mask(Y_tensor, mask, A_stage1=None, n_epochs=100, lr=0.001, batch_size=256, num_workers=4):
    """Stage 2: Estimate signs and magnitudes for connections identified by binary mask."""
    n_neurons = Y_tensor.shape[1]
    
    # Create dataset
    X = Y_tensor[:-1]
    y = Y_tensor[1:]
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Warm-start Stage 2 with Stage 1 weights for better magnitude estimation
    if A_stage1 is not None:
        # Use Stage 1 weights as initialization, but respect the mask
        init_A = A_stage1 * mask
        print(f"Stage 2 warm-start: using Stage 1 weights for {np.sum(mask)} connections")
        print(f"  Stage 1 weight range: [{np.min(A_stage1[mask]):.4f}, {np.max(A_stage1[mask]):.4f}]")
    else:
        init_A = None
        print("Stage 2 cold-start: random initialization")
    
    model = PoissonGLMBlind(n_neurons, mask=mask, init_A=init_A).to(device)
    # Use minimal L1 regularization in Stage 2 to prevent overfitting while allowing proper magnitude estimation
    stage2_l1 = 0.1 * (1e-4)  # 10x smaller than typical Stage 1 L1
    train_losses, val_losses = train_model(model, train_loader, val_loader, n_epochs, lr=lr, l1_lambda=stage2_l1, use_scheduler=True)
    print(f"Stage 2 used minimal L1 regularization: {stage2_l1:.2e}")
    
    # Extract final parameters
    A_final = model.apply_constraints().detach().cpu().numpy()
    B_final = model.B.detach().cpu().numpy()
    
    return A_final, B_final, train_losses, val_losses

def refine_parameters_blind(Y_tensor, mask, init_A, n_epochs=100, lr=0.001, batch_size=256, num_workers=4):
    """Stage 2: Refine parameters with fixed structure (blind - no Dale's principle)."""
    n_neurons = Y_tensor.shape[1]
    
    # Create dataset
    X = Y_tensor[:-1]
    y = Y_tensor[1:]
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Train model without L1 regularization and without Dale's principle, with mask
    model = PoissonGLMBlind(n_neurons, mask=mask, init_A=init_A).to(device)
    train_losses, val_losses = train_model(model, train_loader, val_loader, n_epochs, lr=lr, l1_lambda=0.0, use_scheduler=True)
    
    # Extract final parameters
    A_final = model.apply_constraints().detach().cpu().numpy()
    B_final = model.B.detach().cpu().numpy()
    
    return A_final, B_final, train_losses, val_losses

def refine_parameters(Y_tensor, num_excite, mask, init_A, n_epochs=100, lr=0.001, batch_size=256, num_workers=4):
    """Stage 2: Refine parameters with fixed structure."""
    n_neurons = Y_tensor.shape[1]
    
    # Create dataset
    X = Y_tensor[:-1]
    y = Y_tensor[1:]
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Train model without L1 regularization, with mask
    model = PoissonGLM(n_neurons, num_excite, mask=mask, init_A=init_A).to(device)
    train_losses, val_losses = train_model(model, train_loader, val_loader, n_epochs, lr=lr, l1_lambda=0.0, use_scheduler=True)
    
    # Extract final parameters
    A_final = model.apply_constraints().detach().cpu().numpy()
    B_final = model.B.detach().cpu().numpy()
    
    return A_final, B_final, train_losses, val_losses

def analyze_results(A_true, A_estimated, num_excite):
    """Analyze the quality of reconstruction."""
    # Only compare non-zero elements in true matrix
    mask_true = np.abs(A_true) > 1e-6
    
    # Correlation between true and estimated (only for true connections)
    true_vals = A_true[mask_true]
    est_vals = A_estimated[mask_true]
    
    # Safe correlation calculation with handling for constant values
    def safe_corrcoef(x, y):
        """Calculate correlation with proper handling of constant arrays."""
        if len(x) < 2 or len(y) < 2:
            return np.nan
        if np.std(x) == 0 or np.std(y) == 0:
            # If either variable is constant, correlation is undefined
            return np.nan
        return np.corrcoef(x, y)[0, 1]
    
    correlation = safe_corrcoef(true_vals, est_vals)
    
    # Mean squared error
    mse = np.mean((A_true - A_estimated)**2)
    
    # Separate analysis for excitatory and inhibitory
    exc_mask = np.zeros_like(mask_true)
    exc_mask[:num_excite, :] = True
    inh_mask = np.zeros_like(mask_true)
    inh_mask[num_excite:, :] = True
    
    # Safe correlation for excitatory connections
    exc_true = A_true[mask_true & exc_mask]
    exc_est = A_estimated[mask_true & exc_mask]
    exc_corr = safe_corrcoef(exc_true, exc_est)
    
    # Safe correlation for inhibitory connections  
    inh_true = A_true[mask_true & inh_mask]
    inh_est = A_estimated[mask_true & inh_mask]
    inh_corr = safe_corrcoef(inh_true, inh_est)
    
    print("\n=== Reconstruction Analysis ===")
    print(f"Overall correlation: {correlation:.3f}" if not np.isnan(correlation) else "Overall correlation: N/A (insufficient variation)")
    print(f"Excitatory correlation: {exc_corr:.3f}" if not np.isnan(exc_corr) else "Excitatory correlation: N/A (insufficient variation)")
    print(f"Inhibitory correlation: {inh_corr:.3f}" if not np.isnan(inh_corr) else "Inhibitory correlation: N/A (insufficient variation)")
    print(f"Mean squared error: {mse:.6f}")
    print(f"Number of true connections: {len(true_vals)}")
    print(f"Excitatory connections: {len(exc_true)}, Inhibitory connections: {len(inh_true)}")
    
    return correlation, mse


def analyze_edge_detection(A_true, mask_detected, num_excite):
    """Analyze edge detection performance separately for excitatory and inhibitory connections."""
    
    # Create true edge mask (non-zero elements)
    mask_true = np.abs(A_true) > 1e-6
    
    # Create masks for excitatory and inhibitory neurons
    exc_mask = np.zeros_like(mask_true)
    exc_mask[:num_excite, :] = True
    inh_mask = np.zeros_like(mask_true)
    inh_mask[num_excite:, :] = True
    
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
    
    # Overall analysis
    overall_metrics = calculate_metrics(mask_true, mask_detected)
    
    # Excitatory analysis
    exc_true = mask_true & exc_mask
    exc_detected = mask_detected & exc_mask
    exc_metrics = calculate_metrics(exc_true, exc_detected)
    
    # Inhibitory analysis
    inh_true = mask_true & inh_mask
    inh_detected = mask_detected & inh_mask
    inh_metrics = calculate_metrics(inh_true, inh_detected)
    
    # Print results
    print("\n=== Edge Detection Analysis ===")
    print("Overall:")
    print(f"  TP: {overall_metrics['tp']}, FP: {overall_metrics['fp']}, TN: {overall_metrics['tn']}, FN: {overall_metrics['fn']}")
    print(f"  Precision: {overall_metrics['precision']:.3f}, Recall: {overall_metrics['recall']:.3f}")
    print(f"  F1 Score: {overall_metrics['f1_score']:.3f}, Accuracy: {overall_metrics['accuracy']:.3f}")
    
    print("\nExcitatory connections:")
    print(f"  TP: {exc_metrics['tp']}, FP: {exc_metrics['fp']}, TN: {exc_metrics['tn']}, FN: {exc_metrics['fn']}")
    print(f"  Precision: {exc_metrics['precision']:.3f}, Recall: {exc_metrics['recall']:.3f}")
    print(f"  F1 Score: {exc_metrics['f1_score']:.3f}, Accuracy: {exc_metrics['accuracy']:.3f}")
    
    print("\nInhibitory connections:")
    print(f"  TP: {inh_metrics['tp']}, FP: {inh_metrics['fp']}, TN: {inh_metrics['tn']}, FN: {inh_metrics['fn']}")
    print(f"  Precision: {inh_metrics['precision']:.3f}, Recall: {inh_metrics['recall']:.3f}")
    print(f"  F1 Score: {inh_metrics['f1_score']:.3f}, Accuracy: {inh_metrics['accuracy']:.3f}")
    
    return {
        'overall': overall_metrics,
        'excitatory': exc_metrics,
        'inhibitory': inh_metrics
    }

def preprocess_data_blind(Y):
    """Preprocess spike data without knowledge of excitatory/inhibitory identity."""
    print("\n=== Data Preprocessing (Blind) ===")
    print(f"Data shape: {Y.shape}")
    print(f"Total time steps: {Y.shape[0]}")
    n_samples = Y.shape[0] - 1
    print(f"Number of neurons: {Y.shape[1]}")
    print(f"Total number of training samples (time steps - 1): {n_samples}")
    
    # Calculate firing rates
    dt = 0.01  # 10ms bins
    firing_rates = np.sum(Y, axis=0) / (Y.shape[0] * dt)
    
    print(f"\nFiring rates (Hz):")
    print(f"  Overall: {np.mean(firing_rates):.2f} ± {np.std(firing_rates):.2f}")
    print(f"  Range: [{np.min(firing_rates):.2f}, {np.max(firing_rates):.2f}]")
    
    # Print detailed firing rates by neuron index to check for patterns
    print(f"\nDetailed firing rates by neuron index:")
    for i in range(0, Y.shape[1], 5):  # Print every 5th neuron
        end_idx = min(i + 5, Y.shape[1])
        rates_slice = firing_rates[i:end_idx]
        indices = list(range(i, end_idx))
        print(f"  Neurons {i:2d}-{end_idx-1:2d}: {[f'{r:.1f}' for r in rates_slice]} Hz")
    
    # Check for neurons with very different firing patterns
    median_rate = np.median(firing_rates)
    low_rate_neurons = np.where(firing_rates < median_rate * 0.1)[0]  # < 10% of median
    high_rate_neurons = np.where(firing_rates > median_rate * 10)[0]   # > 10x median
    
    if len(low_rate_neurons) > 0:
        print(f"\nLow firing neurons (< {median_rate*0.1:.1f} Hz): {low_rate_neurons}")
    if len(high_rate_neurons) > 0:
        print(f"High firing neurons (> {median_rate*10:.1f} Hz): {high_rate_neurons}")
    
    # Check for neurons with very low firing
    min_spikes = 100  # Minimum spikes needed for reliable estimation
    spike_counts = np.sum(Y, axis=0)
    low_firing = spike_counts < min_spikes
    
    if np.any(low_firing):
        print(f"\nWarning: {np.sum(low_firing)} neurons have fewer than {min_spikes} spikes")
        print(f"  Neuron indices: {np.where(low_firing)[0]}")
        print(f"  Their spike counts: {spike_counts[low_firing]}")
    
    # Calculate sparsity of data
    sparsity = 1 - np.count_nonzero(Y) / Y.size
    print(f"\nData sparsity: {sparsity*100:.1f}% zeros")
    
    return firing_rates, spike_counts

def preprocess_data(Y, num_excite):
    """Preprocess spike data and perform initial analysis."""
    print("\n=== Data Preprocessing ===")
    print(f"Data shape: {Y.shape}")
    print(f"Total time steps: {Y.shape[0]}")
    n_samples = Y.shape[0] - 1
    print(f"Number of neurons: {Y.shape[1]} ({num_excite} excitatory, {Y.shape[1]-num_excite} inhibitory)")
    print(f"Total number of training samples (time steps - 1): {n_samples}")
    
    # Calculate firing rates
    dt = 0.01  # 10ms bins
    firing_rates = np.sum(Y, axis=0) / (Y.shape[0] * dt)
    
    print(f"\nFiring rates (Hz):")
    print(f"  Overall: {np.mean(firing_rates):.2f} ± {np.std(firing_rates):.2f}")
    print(f"  Excitatory: {np.mean(firing_rates[:num_excite]):.2f} ± {np.std(firing_rates[:num_excite]):.2f}")
    print(f"  Inhibitory: {np.mean(firing_rates[num_excite:]):.2f} ± {np.std(firing_rates[num_excite:]):.2f}")
    
    # Check for neurons with very low firing
    min_spikes = 100  # Minimum spikes needed for reliable estimation
    spike_counts = np.sum(Y, axis=0)
    low_firing = spike_counts < min_spikes
    
    if np.any(low_firing):
        print(f"\nWarning: {np.sum(low_firing)} neurons have fewer than {min_spikes} spikes")
        print(f"  Neuron indices: {np.where(low_firing)[0]}")
        print(f"  Their spike counts: {spike_counts[low_firing]}")
    
    # Calculate sparsity of data
    sparsity = 1 - np.count_nonzero(Y) / Y.size
    print(f"\nData sparsity: {sparsity*100:.1f}% zeros")
    
    return firing_rates, spike_counts

def bootstrap_confidence_intervals(Y_tensor, num_excite, A_final, B_final, mask, n_bootstrap=20, n_epochs=20, batch_size=256, num_workers=4):
    """Estimate confidence intervals using bootstrap."""
    print("\n=== Bootstrap Confidence Intervals ===")
    n_neurons = Y_tensor.shape[1]
    n_params = np.sum(mask)
    
    # Store bootstrap estimates
    A_bootstrap = []
    B_bootstrap = []
    
    X = Y_tensor[:-1]
    y = Y_tensor[1:]
    
    for i in range(n_bootstrap):
        # Resample time indices with replacement
        n_samples = len(X)
        indices = torch.randperm(n_samples)[:int(0.8 * n_samples)]
        
        # Create bootstrap dataset
        X_boot, y_boot = X[indices], y[indices]
        dataset = torch.utils.data.TensorDataset(X_boot, y_boot)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        
        # Train model with fixed structure
        model = PoissonGLM(n_neurons, num_excite, mask=mask, init_A=A_final).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Quick training
        model.train()
        for epoch in range(n_epochs):
            for Y_prev, Y_curr in loader:
                Y_prev = Y_prev.float().to(device)
                Y_curr = Y_curr.float().to(device)
                
                optimizer.zero_grad()
                rates = model(Y_prev)
                loss = poisson_nll_loss(rates, Y_curr)
                loss.backward()
                optimizer.step()
        
        # Store estimates
        A_boot = model.apply_constraints().detach().cpu().numpy()
        B_boot = model.B.detach().cpu().numpy()
        A_bootstrap.append(A_boot)
        B_bootstrap.append(B_boot)
    
    # Calculate confidence intervals
    A_bootstrap = np.array(A_bootstrap)
    B_bootstrap = np.array(B_bootstrap)
    
    A_mean = np.mean(A_bootstrap, axis=0)
    A_std = np.std(A_bootstrap, axis=0)
    A_ci_lower = np.percentile(A_bootstrap, 2.5, axis=0)
    A_ci_upper = np.percentile(A_bootstrap, 97.5, axis=0)
    
    # Report CI for significant connections
    significant_connections = mask & (np.abs(A_final) > 0.01)
    n_sig = np.sum(significant_connections)
    
    print(f"\nFound {n_sig} significant connections")
    print("\nExample confidence intervals for strongest connections:")
    
    # Find strongest connections
    strength_order = np.argsort(np.abs(A_final).flatten())[::-1]
    count = 0
    for idx in strength_order:
        i, j = np.unravel_index(idx, A_final.shape)
        if significant_connections[i, j] and count < 10:
            print(f"  A[{i},{j}]: {A_final[i,j]:.4f} [{A_ci_lower[i,j]:.4f}, {A_ci_upper[i,j]:.4f}]")
            count += 1
    
    return A_ci_lower, A_ci_upper, A_std

def plot_results(A_true, A_estimated, train_losses_s1, val_losses_s1, train_losses_s2, val_losses_s2, save_png=False, output_prefix="connectivity_reconstruction", data_name=None):
    """Plot comparison of true vs estimated connectivity."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # True connectivity
    im1 = axes[0, 0].imshow(A_true, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    title = 'True Connectivity'
    if data_name:
        title += f' ({data_name})'
    axes[0, 0].set_title(title)
    axes[0, 0].set_xlabel('From neuron')
    axes[0, 0].set_ylabel('To neuron')
    plt.colorbar(im1, ax=axes[0, 0])
    axes[0, 0].grid(True, alpha=0.3)
    
    # Estimated connectivity
    im2 = axes[0, 1].imshow(A_estimated, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[0, 1].set_title('Estimated Connectivity')
    axes[0, 1].set_xlabel('From neuron')
    axes[0, 1].set_ylabel('To neuron')
    plt.colorbar(im2, ax=axes[0, 1])
    axes[0, 1].grid(True, alpha=0.3)
    
    # Difference
    diff = A_estimated - A_true
    im3 = axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    axes[0, 2].set_title('Difference (Est - True)')
    axes[0, 2].set_xlabel('From neuron')
    axes[0, 2].set_ylabel('To neuron')
    plt.colorbar(im3, ax=axes[0, 2])
    axes[0, 2].grid(True, alpha=0.3)
    
    # Scatter plot of true vs estimated
    mask = np.abs(A_true) > 1e-6
    axes[1, 0].scatter(A_true[mask], A_estimated[mask], alpha=0.5)
    axes[1, 0].plot([-0.5, 0.5], [-0.5, 0.5], 'r--')
    axes[1, 0].set_xlabel('True weight')
    axes[1, 0].set_ylabel('Estimated weight')
    axes[1, 0].set_title('Weight comparison')
    
    # Training curves
    axes[1, 1].plot(train_losses_s1, label='Train Stage 1', color='blue', linestyle='-')
    axes[1, 1].plot(val_losses_s1, label='Val Stage 1', color='blue', linestyle='--')
    axes[1, 1].plot(train_losses_s2, label='Train Stage 2', color='red', linestyle='-')
    axes[1, 1].plot(val_losses_s2, label='Val Stage 2', color='red', linestyle='--')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Training curves (both stages)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Remove empty subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_png:
        outF = f"{output_prefix}_fit.png"
        plt.savefig(outF, bbox_inches='tight')
        print(f"Plot saved as {outF}")
    
    plt.show()

def analyze_and_save_results(args, truth_file, mask, A_stage1, A_final, B_final, 
                            train_losses_s1, val_losses_s1, train_losses_s2, val_losses_s2,
                            best_l1, cv_results, firing_rates, spike_counts, overall_start_time):
    """
    Encapsulating function for all results analysis and postprocessing.
    Opens truth file internally and performs comprehensive evaluation.
    """
    
    # Load ground truth for evaluation
    print(f"\n=== Loading Ground Truth for Final Evaluation ===")
    print(f"Loading ground truth from {truth_file}")
    truth_data = np.load(truth_file, allow_pickle=True)
    A_true = truth_data['A']
    B_true = truth_data['B_intercept']
    conf = truth_data['conf'].item()
    num_excite = conf['num_excite']
    n_neurons = A_true.shape[0]
    
    # Analyze results
    correlation, mse = analyze_results(A_true, A_final, num_excite)
    
    # Post-processing: Edge analysis
    edge_analysis = analyze_edge_detection(A_true, mask, num_excite)
    
    # Bootstrap confidence intervals (optional)
    if args.bootstrap:
        stage_start_time = time.time()
        # Need to reconstruct Y_tensor for bootstrap
        spikes_file = os.path.join(args.dataPath, f"{args.dataName}.spikes.npz")
        spike_data = np.load(spikes_file, allow_pickle=True)
        Y = spike_data['Y']
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        
        A_ci_lower, A_ci_upper, A_std = bootstrap_confidence_intervals(
            Y_tensor, num_excite, A_final, B_final, mask, 
            n_bootstrap=args.n_bootstrap, n_epochs=50, batch_size=args.batch_size, num_workers=args.num_workers
        )
        print(f"Bootstrap completed in {time.time() - stage_start_time:.1f} seconds")
    
    # Plot results and save
    output_prefix = os.path.join(args.dataPath, args.dataName)
    plot_results(A_true, A_final, train_losses_s1, val_losses_s1, train_losses_s2, val_losses_s2, save_png=True, output_prefix=output_prefix, data_name=args.dataName)
    
    # Save results in the same directory as input data
    output_file = os.path.join(args.dataPath, f"{args.dataName}_results.npz")
    save_dict = {
        'A_true': A_true,
        'A_stage1': A_stage1,
        'A_stage2': A_final,
        'B_stage2': B_final,
        'mask_stage1': mask,
        'correlation': correlation,
        'mse': mse,
        'train_losses_stage1': train_losses_s1,
        'val_losses_stage1': val_losses_s1,
        'train_losses_stage2': train_losses_s2,
        'val_losses_stage2': val_losses_s2,
        'best_l1': best_l1,
        'cv_results': cv_results,
        'firing_rates': firing_rates,
        'spike_counts': spike_counts,
        'edge_analysis': edge_analysis,
        'args': vars(args)
    }
    
    if args.bootstrap:
        save_dict.update({
            'A_ci_lower': A_ci_lower,
            'A_ci_upper': A_ci_upper,
            'A_std': A_std
        })
    
    np.savez(output_file, **save_dict)
    print(f"\nResults saved to {output_file}")
    
    # Final summary
    print("\n=== Final Summary ===")
    print(f"Total processing time: {time.time() - overall_start_time:.1f} seconds")
    print(f"Correlation with true matrix: {correlation:.3f}")
    print(f"Identified {np.sum(mask)} / {np.sum(np.abs(A_true) > 1e-6)} true connections")
    print(f"False positive rate: {np.sum(mask & (np.abs(A_true) < 1e-6)) / np.sum(np.abs(A_true) < 1e-6):.3f}")
    print(f"True positive rate: {np.sum(mask & (np.abs(A_true) > 1e-6)) / np.sum(np.abs(A_true) > 1e-6):.3f}")

########################################################
#  Main function
########################################################
def main():
    parser = argparse.ArgumentParser(description="Recover connectivity matrix from spike data")
    parser.add_argument("--dataName", type=str, default="dale_M40_2M", help="Base name for input/output files")
    parser.add_argument("--dataPath", type=str, default="out/", help="Path to data directory")
    parser.add_argument("--n_epochs_stage1", type=int, default=50, help="Number of epochs for stage 1")
    parser.add_argument("--n_epochs_stage2", type=int, default=50, help="Number of epochs for stage 2")
    parser.add_argument("--batch_size", type=int, default=2048*8, help="Batch size for training")
    parser.add_argument("--lr_stage1", type=float, default=0.001, help="Learning rate for stage 1")
    parser.add_argument("--lr_stage2", type=float, default=0.001, help="Learning rate for stage 2")
    parser.add_argument("--l1_lambda", type=float, default=1.0e-04, help="L1 regularization parameter (set to 0 to enable L1 scan)")
    parser.add_argument("--target_sparsity", type=float, default=0.8, help="Target sparsity level (0.9 = 90% zeros)")
    parser.add_argument("--cv_folds", type=int, default=2, help="Number of cross-validation folds")
    parser.add_argument("--bootstrap", action="store_true", help="Compute bootstrap confidence intervals")
    parser.add_argument("--n_bootstrap", type=int, default=20, help="Number of bootstrap samples")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of data loader workers")
    
    # Firing-rate-aware L1 regularization parameters
    parser.add_argument("--l1_rate_power", type=float, default=1.0, help="Power for firing-rate-aware L1 penalty (1.0 = no bias, higher = stronger bias toward low-firing neurons)")
    parser.add_argument("--disable_adaptive_l1", action="store_true", help="Disable firing-rate-aware L1 regularization (use uniform L1 penalty)")
    
    args = parser.parse_args() 
    print("\nFit with configuration:")
    print(vars(args))
    print("")
    overall_start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(args.dataPath, exist_ok=True)
    
    # Construct file paths
    spikes_file = os.path.join(args.dataPath, f"{args.dataName}.spikes.npz")
    truth_file = os.path.join(args.dataPath, f"{args.dataName}.truth.npz")
    
    # Load spike data
    print(f"Loading spike data from {spikes_file}")
    spike_data = np.load(spikes_file, allow_pickle=True)
    Y = spike_data['Y']
    
    # Determine number of neurons and assume we don't know excitatory/inhibitory split
    n_neurons = Y.shape[1]
    print(f"Loaded {Y.shape[0]} time steps for {n_neurons} neurons")
    print("Note: Stage 1 will not use Dale's principle - only detecting edge existence")
    
    # Preprocess and analyze data (without Dale's principle knowledge)
    firing_rates, spike_counts = preprocess_data_blind(Y)
    print(f"Data loading and preprocessing took {time.time() - overall_start_time:.1f} seconds")
    
    # Convert to PyTorch tensor
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    
    # Stage 0: Cross-validation for L1 parameter (only if l1_lambda is 0)
    if args.l1_lambda == 0:
        print("\n=== Stage 0: Cross-validation for L1 parameter ===")
        stage_start_time = time.time()
        l1_values = np.logspace(-5, -4, 3)
        print(f"Testing L1 values: {l1_values}")
        
        best_l1, cv_results = cross_validate_l1_blind(Y_tensor, l1_values, 
                                               batch_size=args.batch_size, n_splits=args.cv_folds, n_epochs=50, num_workers=args.num_workers, data_fraction=0.1)
        print(f"\nBest L1 parameter: {best_l1:.2e}")
        print(f"Stage 0 (Cross-validation) completed in {time.time() - stage_start_time:.1f} seconds")
    else:
        best_l1 = args.l1_lambda
        print(f"\nUsing fixed L1 parameter: {best_l1:.2e}")
        cv_results = {}
    
    # Stage 1: Structure identification (without Dale's principle)
    print("\n=== Stage 1: Structure Identification (No Dale's Principle) ===")
    stage_start_time = time.time()
    mask, A_stage1, train_losses_s1, val_losses_s1 = identify_structure_blind(
        Y_tensor, best_l1, n_epochs=args.n_epochs_stage1, target_sparsity=args.target_sparsity, lr=args.lr_stage1, batch_size=args.batch_size, num_workers=args.num_workers,
        l1_rate_power=args.l1_rate_power, disable_adaptive_l1=args.disable_adaptive_l1
    )
    stage1_time = time.time() - stage_start_time
    print(f"Stage 1 completed in {stage1_time:.1f} seconds")
    
    # Stage 2: Parameter estimation using binary mask and warm-start from Stage 1
    print("\n=== Stage 2: Parameter Estimation (Warm-Started from Stage 1) ===")
    print("Note: Stage 2 uses binary mask from Stage 1 AND warm-starts with Stage 1 weight magnitudes")
    stage_start_time = time.time()
    A_final, B_final, train_losses_s2, val_losses_s2 = estimate_parameters_with_mask(
        Y_tensor, mask, A_stage1=A_stage1, n_epochs=args.n_epochs_stage2, lr=args.lr_stage2, batch_size=args.batch_size, num_workers=args.num_workers
    )
    stage2_time = time.time() - stage_start_time
    print(f"Stage 2 completed in {stage2_time:.1f} seconds")
    
    # Results analysis and postprocessing
    analyze_and_save_results(
        args, truth_file, mask, A_stage1, A_final, B_final, 
        train_losses_s1, val_losses_s1, train_losses_s2, val_losses_s2,
        best_l1, cv_results, firing_rates, spike_counts, 
        overall_start_time
    )

if __name__ == "__main__":
    main()
