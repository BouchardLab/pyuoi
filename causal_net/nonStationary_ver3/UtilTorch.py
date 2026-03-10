#!/usr/bin/env python3
"""
PyTorch utilities for GPU setup, data preprocessing and model training.

This module provides essential utilities for PyTorch-based neural network training,
specifically optimized for Poisson GLM fitting with distributed GPU support.
Main functionality includes:
- GPU availability checking and device configuration
- Data preprocessing with time decorrelation and shuffling options
- Custom dataset classes for paired neural data (Y_prev, Y_curr)
- Distributed training utilities with Poisson loss optimization
- Learning rate scheduling and training loop management

Designed to work with multi-GPU setups using DistributedDataParallel
for efficient training of large-scale neural connectivity models.
"""

import torch
import numpy as np
import time
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PoissonGLModel import poisson_nll_loss

def offdiag_soft_threshold_(A, lr, lam):
    """In-place off-diagonal soft-thresholding for a single square A matrix."""
    if lam <= 0.0:
        return
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"offdiag_soft_threshold_ expects square 2D A, got shape={tuple(A.shape)}")
    with torch.no_grad():
        N = A.shape[0]
        off_diag = ~torch.eye(N, dtype=torch.bool, device=A.device)
        t = lr * lam
        A_off = A[off_diag]
        A[off_diag] = A_off.sign() * (A_off.abs() - t).clamp(min=0.0)

def enforce_spectral_radius_(A, rho_max, eps=1e-12):
    """Scale A in-place only if spectral radius exceeds rho_max (GPU-friendly)."""
    if rho_max is None:
        return
    if rho_max <= 0.0:
        raise ValueError(f"rho_max must be > 0, got {rho_max}")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"enforce_spectral_radius_ expects square 2D A, got shape={tuple(A.shape)}")
    with torch.no_grad():
        # Keep computation on current device; no CPU transfer.
        rho = torch.linalg.eigvals(A).abs().max()
        if torch.isfinite(rho) and (rho > rho_max):
            A.mul_(float(rho_max) / float(rho + eps))

def check_gpu_availability():
    """Check GPU availability and set up device configuration."""
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA-enabled GPU environment.")
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    return device

def preprocess_data(Y, args):
    import random
    Nt, Nn = Y.shape
    is_main = (getattr(args, "rank", 0) == 0)
    
    # Apply time decorrelation if requested
    if args.desyncTime:
        if is_main: print("\n=== Applying Time Decorrelation, it shifts time for each neuron ===")
        seed = int(time.time() * 1000) % 1000000
        np.random.seed(seed)
        shift_amounts = np.random.randint(1, Nt//4, size=Nn, dtype=np.int32)
        if is_main: print('Generated random shifts with seed=%d'%(seed),shift_amounts[:10],'...',flush=True)
        Y_shifted = np.zeros_like(Y)
        for neuron_idx in range(Nn):
            shift_amount = int(shift_amounts[neuron_idx])
            Y_shifted[:, neuron_idx] = np.roll(Y[:, neuron_idx], shift_amount)
        if is_main: print(f"Applied time shifts, destroys temporal correlations between neurons")
        Y = Y_shifted
    
    # Create consecutive pairs
    max_pairs = Nt - 1
    num_samples = args.num_samples
    if num_samples is None or num_samples > max_pairs:
        num_samples = max_pairs
    XY = np.stack([Y[:num_samples], Y[1:num_samples + 1]], axis=1)
    
    # Apply random data dropping if requested
    if args.dropDataFrac > 0:
        n_pairs = XY.shape[0]
        drop_seed = (int(time.time() * 1000) + np.random.randint(0, 1000)) % (2**32)
        np.random.seed(drop_seed)
        random.seed(drop_seed)
        n_keep = int(n_pairs * (1.0 - args.dropDataFrac))
        keep_indices = np.random.choice(n_pairs, size=n_keep, replace=False)
        keep_indices = np.sort(keep_indices)
        XY = XY[keep_indices]
        if is_main: print(f"Dropped {args.dropDataFrac:.1%} of data, keeping {XY.shape[0]} samples (seed={drop_seed})")
    
    return XY


def train_Poisson_model(model, device, train_loader, n_epochs, lr, L1_alpha=0.0, use_scheduler=False,
                        firing_rates=None, train_sampler=None, print_every=20, apply_prox=False,
                        lr_end_factor=0.03, minW=1e-6, rho_max=0.99, prescale_m_step_4_ArhoMax=10, L1_prune_epoch=0):
    if L1_prune_epoch < 0:
        raise ValueError(f"L1_prune_epoch must be >= 0, got {L1_prune_epoch}")
    if prescale_m_step_4_ArhoMax < 1:
        raise ValueError(f"prescale_m_step_4_ArhoMax must be >= 1, got {prescale_m_step_4_ArhoMax}")
    use_fused = (isinstance(device, torch.device) and device.type=='cuda' and torch.cuda.is_available())
    assert use_fused
    optimizer = optim.Adam(model.parameters(), lr=lr, fused=True)
    mdl = model.module if hasattr(model, 'module') else model
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=lr_end_factor, total_iters=n_epochs
    ) if use_scheduler else None
    
    diag_mask = torch.eye(mdl.n_neurons, device=device).bool()
    off_diag_mask = ~diag_mask
    L1_weight_matrix = torch.ones(mdl.n_neurons, mdl.n_neurons, device=device)
    L1_weight_matrix[diag_mask] = 0.0
    
    firing_rates_tensor = torch.tensor(firing_rates, dtype=torch.float32, device=device) if firing_rates is not None else None
    
    train_losses_w_L1, train_losses_wo_L1, learning_rates = [], [], []
    sparsity_epoch, nz_offdiag_epoch, spectral_radius_epoch = [], [], []
    train_epochs = []
    start_time = time.time()

    for epoch in range(n_epochs):
        if train_sampler is not None:   train_sampler.set_epoch(epoch)
        model.train()
        train_loss_w_L1 = 0
        train_loss_wo_L1 = 0
        apply_rho = True
        apply_prune = (epoch >= L1_prune_epoch)
        for batch_idx, (Y_prev, Y_curr) in enumerate(train_loader):
            Y_prev, Y_curr = Y_prev.float().to(device, non_blocking=True), Y_curr.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            spikes = model(Y_prev)
            base_loss = poisson_nll_loss(spikes, Y_curr, firing_rates_tensor)
            loss_with_L1 = base_loss.clone()
            if L1_alpha > 0:
                loss_with_L1 += L1_alpha * torch.mean(torch.abs(mdl.A) * L1_weight_matrix)
            loss_with_L1.backward()
            optimizer.step()
            if apply_prune and apply_prox and L1_alpha > 0:
                offdiag_soft_threshold_(mdl.A, optimizer.param_groups[0]['lr'], L1_alpha)
            if apply_rho and (batch_idx % prescale_m_step_4_ArhoMax == 0):
                enforce_spectral_radius_(mdl.A, rho_max)
            train_loss_w_L1 += loss_with_L1.item()
            train_loss_wo_L1 += base_loss.item()
        
        # record train losses each epoch
        train_losses_w_L1.append(train_loss_w_L1 / len(train_loader))
        train_losses_wo_L1.append(train_loss_wo_L1 / len(train_loader))
        train_epochs.append(epoch + 1)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        with torch.no_grad():
            A_off = mdl.A[off_diag_mask]
            nz_off = int((A_off.abs() > minW).sum().item())
            n_off = int(off_diag_mask.sum().item())
            sparsity = 1.0 - nz_off / max(1, n_off)
            rho = float(torch.linalg.eigvals(mdl.A).abs().max().item())
        sparsity_epoch.append(sparsity)
        nz_offdiag_epoch.append(nz_off)
        spectral_radius_epoch.append(rho)
        
        if scheduler:
            scheduler.step()
           
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}/{n_epochs}:  Loss_Tot={train_losses_w_L1[-1]:.5g}, only_L1={(train_losses_w_L1[-1]-train_losses_wo_L1[-1]):.4g}, A_sparsity={sparsity:.3f}, nz_offdiag={nz_off}, rho(A)={rho:.4f}, Elapsed={(time.time() - start_time):.1f}s")
    
        
            
    return train_losses_w_L1, train_losses_wo_L1, learning_rates, train_epochs, sparsity_epoch, nz_offdiag_epoch, spectral_radius_epoch


class NumpyPairDataset(Dataset):
    def __init__(self, X_np, Y_np):
        self.X = X_np
        self.Y = Y_np
        self.n = X_np.shape[0]
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).to(dtype=torch.float32), torch.from_numpy(self.Y[idx]).to(dtype=torch.float32)

def make_loader(X, Yt, args, is_dist=False, shuffle=True):
    if is_dist:
        raise ValueError("Distributed loading was removed from UtilTorch.make_loader; use is_dist=False.")
    dataset = NumpyPairDataset(X, Yt)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, drop_last=shuffle, pin_memory=True, pin_memory_device='cuda', num_workers=8,
                      persistent_workers=True, prefetch_factor=8)
