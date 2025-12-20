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
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PoissonGLModel import poisson_nll_loss

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
    
    # Apply time decorrelation if requested
    if args.desyncTime:
        if args.rank==0: print("\n=== Applying Time Decorrelation, it shifts time for each neuron ===")
        seed = int(time.time() * 1000) % 1000000
        np.random.seed(seed)
        shift_amounts = np.random.randint(1, Nt//4, size=Nn, dtype=np.int32)
        if args.rank==0: print('Generated random shifts with seed=%d'%(seed),shift_amounts[:10],'...',flush=True)
        Y_shifted = np.zeros_like(Y)
        for neuron_idx in range(Nn):
            shift_amount = int(shift_amounts[neuron_idx])
            Y_shifted[:, neuron_idx] = np.roll(Y[:, neuron_idx], shift_amount)
        if args.rank==0: print(f"Applied time shifts, destroys temporal correlations between neurons")
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
        if args.rank==0: print(f"Dropped {args.dropDataFrac:.1%} of data, keeping {XY.shape[0]} samples (seed={drop_seed})")
    
    return XY


def train_Poisson_model(model, device, train_loader, n_epochs, lr, L1_alpha=0.0, use_scheduler=False, firing_rates=None, train_sampler=None, print_every=20):
    use_fused = (isinstance(device, torch.device) and device.type=='cuda' and torch.cuda.is_available())
    assert use_fused
    optimizer = optim.Adam(model.parameters(), lr=lr, fused=True)
    mdl = model.module if hasattr(model, 'module') else model
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=n_epochs) if use_scheduler else None
    
    diag_mask = torch.eye(mdl.n_neurons, device=device).bool()
    L1_weight_matrix = torch.ones(mdl.n_neurons, mdl.n_neurons, device=device)
    L1_weight_matrix[diag_mask] = 0.0
    
    firing_rates_tensor = torch.tensor(firing_rates, dtype=torch.float32, device=device) if firing_rates is not None else None
    
    train_losses_w_L1, train_losses_wo_L1, learning_rates = [], [], []
    train_epochs = []
    start_time = time.time()

    for epoch in range(n_epochs):
        if train_sampler is not None:   train_sampler.set_epoch(epoch)
        model.train()
        train_loss_w_L1 = 0
        train_loss_wo_L1 = 0
        for Y_prev, Y_curr in train_loader:
            Y_prev, Y_curr = Y_prev.float().to(device, non_blocking=True), Y_curr.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            spikes = model(Y_prev)
            base_loss = poisson_nll_loss(spikes, Y_curr, firing_rates_tensor)
            loss_with_L1 = base_loss.clone()
            if L1_alpha > 0:
                loss_with_L1 += L1_alpha * torch.mean(torch.abs(mdl.A) * L1_weight_matrix)
            loss_with_L1.backward()
            optimizer.step()
            train_loss_w_L1 += loss_with_L1.item()
            train_loss_wo_L1 += base_loss.item()
        
        # record train losses each epoch
        train_losses_w_L1.append(train_loss_w_L1 / len(train_loader))
        train_losses_wo_L1.append(train_loss_wo_L1 / len(train_loader))
        train_epochs.append(epoch + 1)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        if scheduler:
            scheduler.step()
           
        if (epoch + 1) % print_every == 0 and (not dist.is_initialized() or dist.get_rank()==0):
            
            print(f"Epoch {epoch+1}/{n_epochs}:  Loss_Tot={train_losses_w_L1[-1]:.5g}, only_L1={(train_losses_w_L1[-1]-train_losses_wo_L1[-1]):.4g}, Elapsed={(time.time() - start_time):.1f}s")
    
        
            
    return train_losses_w_L1, train_losses_wo_L1, learning_rates, train_epochs


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
    dataset = NumpyPairDataset(X, Yt)
    if is_dist:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        return DataLoader(dataset, batch_size=max(1, args.batch_size//world_size), sampler=sampler, shuffle=False, drop_last=shuffle,
                          pin_memory=True, pin_memory_device='cuda', num_workers=8, persistent_workers=True, prefetch_factor=8)
    else:
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, drop_last=shuffle, pin_memory=True, pin_memory_device='cuda', num_workers=8,
                          persistent_workers=True, prefetch_factor=8)


