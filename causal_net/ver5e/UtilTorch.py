import torch
import numpy as np
import time
import torch.optim as optim
import torch.distributed as dist
from PoissonGLModel import poisson_nll_loss

def check_gpu_availability():
    """Check GPU availability and set up device configuration."""
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA-enabled GPU environment.")
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    return device

def preprocess_data(Y, args, time_mask=None):
    T, M = Y.shape
    
    # Apply decorrelation if requested
    if hasattr(args, 'desync_time') and args.desync_time > 0:
        print("\n=== Applying Time Decorrelation ===")
        print(f"Shifting consecutive neurons by {args.desync_time} time bins")

        Y_decorr = Y.copy()
        for neuron_idx in range(M):
            shift_amount = neuron_idx * args.desync_time
            if shift_amount > 0:
                # Circular shift: move data to the right, wrap around
                Y_decorr[:, neuron_idx] = np.roll(Y[:, neuron_idx], shift_amount)

        print(f"Applied time shifts from 0 to {(M-1) * args.desync_time} bins")
        print(f"This destroys temporal correlations between neurons")
        Y = Y_decorr
    
    # Handle time masking to preserve causal structure
    if time_mask is not None:
        # Ensure mask doesn't exceed data length
        mask_len = min(len(time_mask), T)
        valid_pairs = []
        
        # Create pairs (t, t+1) only where both t and t+1 are not masked
        for t in range(mask_len - 1):
            if not time_mask[t] and not time_mask[t + 1]:
                valid_pairs.append(t)
        
        # Add remaining pairs if mask is shorter than data
        for t in range(mask_len, T - 1):
            valid_pairs.append(t)
        
        valid_pairs = np.array(valid_pairs)
        print(f"Time masking: keeping {len(valid_pairs)} valid consecutive pairs out of {T-1} possible pairs")
        
        max_pairs = len(valid_pairs)
        num_samples = args.num_samples
        if num_samples is None or num_samples > max_pairs:
            num_samples = max_pairs
        
        selected_pairs = valid_pairs[:num_samples]
        return Y[selected_pairs], Y[selected_pairs + 1]
    else:
        # Original logic when no masking
        max_pairs = T - 1
        num_samples = args.num_samples
        if num_samples is None or num_samples > max_pairs:
            num_samples = max_pairs
        return Y[:num_samples], Y[1:num_samples + 1]


def train_Poisson_model(model, device, train_loader, n_epochs, lr, L1_alpha=0.0, use_scheduler=False, firing_rates=None, train_sampler=None, print_every=10):
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
            
            print(f"Epoch {epoch+1}/{n_epochs}:  Loss_Tot={train_losses_w_L1[-1]:.4g}, only_L1={(train_losses_w_L1[-1]-train_losses_wo_L1[-1]):.3g}, Elapsed={(time.time() - start_time):.1f}s")
    
        
            
    return train_losses_w_L1, train_losses_wo_L1, learning_rates, train_epochs


