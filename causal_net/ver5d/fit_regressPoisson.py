#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
import torch.optim as optim
import torch
from torch.utils.data import TensorDataset, DataLoader

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
from PoissonGLM import PoissonGLModel, poisson_nll_loss

from UtilTorch import check_gpu_availability, preprocess_data
from fit_lassoPoisson import train_Poisson_model
from pprint import pprint
#
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--dataName", type=str, required=True, help="fitA NPZ file name")
    parser.add_argument("--dataPath", type=str, default="out/", help="path to input and output files")
    parser.add_argument("--num_samples", type=int, default=None, help="limit number of samples, None=all")
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--fitName", type=str, default=None, help="base name for output files")
    parser.add_argument("--noise_scale", type=float, default=0.2, help="Random noise scale for Stage B initialization (0.0=no noise, 0.1=moderate noise)")
    
    args = parser.parse_args()
    args.desync_time=0

    print("Initial configuration:", vars(args))
    device = check_gpu_availability()
    
    fit1FF = os.path.join(args.dataPath, f"{args.dataName}.lasso.npz")
    fitD, fitMD = read_data_npz(fit1FF)
    A_init = fitD['A_pass']
    B_init = fitD['B_lasso']

    pprint(fitMD)
    fmd=fitMD['fit_lasso']
    # Inherit hyperparams from stageA if not provided
    if args.n_epochs is None:
        args.n_epochs = fmd['n_epochs']
    if args.batch_size is None:
        args.batch_size = fmd['batch_size']
         
    print("Effective configuration:", vars(args))
    
    spike_data_name = fitMD['input_file']
    spikesFF = os.path.join(args.dataPath, f"{spike_data_name}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF)
    dataYield, dataRates = spikeD['spikes'], np.clip(spikeD['single_rates'], 0.1, 40.0)
    T, M = dataYield.shape
    step_size = spikeMD['dale_simu_stats']['time_step_sec']
    
    if args.fitName is None:
        import hashlib
        hash_str =  hashlib.md5(os.urandom(32)).hexdigest()[:6]
        args.fitName = f"{args.dataName}_{hash_str}"
  
    fit2FF = os.path.join(args.dataPath, f"{args.fitName}.regress.npz")
    
    X_np, Yt_np = preprocess_data(dataYield, args)
    n_pairs, train_split = X_np.shape[0], 0.8
    n_train = int(n_pairs * train_split)
    
    assert n_train >= args.batch_size, f"ERROR: Not enough training samples ({n_train}) for batch size ({args.batch_size})."

    def make_loader(X, Yt, shuffle=True):
        return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Yt, dtype=torch.float32)), 
                         batch_size=args.batch_size, shuffle=shuffle, drop_last=shuffle)
    
    train_loader = make_loader(X_np[:n_train], Yt_np[:n_train])
    val_loader = make_loader(X_np[n_train:], Yt_np[n_train:], shuffle=False)
    
    print(f"Loaded T={T}, M={M}, using {n_pairs} pairs ({n_train} train, {n_pairs-n_train} val), step_size={step_size}")

    # --- Set up model for stage B ---
    trainable_mask = fitD['mask.lasso.pass']
    print(f"Trainable mask: {trainable_mask.sum()} out of {trainable_mask.size} A elements are trainable")
    print(f"Trainable fraction: {trainable_mask.mean():.3f}")
    
    model = PoissonGLModel(M, A_init=A_init, B_init=B_init, trainable_mask=trainable_mask, noise_scale=args.noise_scale)
    
    # Debug: check if noise was actually applied
    if args.noise_scale > 0:
        print(f"Checking noise application...")
        A_reconstructed = model.A.detach().cpu().numpy()
        A_diff = np.abs(A_reconstructed - A_init).max()
        B_reconstructed = model.B.detach().cpu().numpy()
        B_diff = np.abs(B_reconstructed - B_init).max()
        print(f"Max A difference from init: {A_diff:.6f}")
        print(f"Max B difference from init: {B_diff:.6f}")
    
    # Debug: count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected_trainable = trainable_mask.sum() + M  # masked A elements + B vector
    print(f"Total parameters: {total_params}, PyTorch trainable: {trainable_params}, Expected trainable: {expected_trainable}")
    
    if hasattr(model, 'is_stage_b') and model.is_stage_b:
        print(f"Stage B efficient parameterization: C tensor has {model.C.numel()} parameters")
    else:
        print("Stage A standard parameterization")
    
    # Debug: check initial loss and parameter values
    model.to(device)
    model.train()
    
    # Check initial loss before training
    with torch.no_grad():
        for Y_prev, Y_curr in train_loader:
            Y_prev, Y_curr = Y_prev.float().to(device), Y_curr.float().to(device)
            spikes = model(Y_prev)
            initial_loss = poisson_nll_loss(spikes, Y_curr, torch.tensor(dataRates, dtype=torch.float32, device=device))
            print(f"Initial loss with noise: {initial_loss:.6f}")
            break
    
    # Check if C parameters actually changed from Stage A
    if hasattr(model, 'C'):
        C_values = model.C.detach().cpu()
        print(f"C parameter stats: min={C_values.min():.6f}, max={C_values.max():.6f}, std={C_values.std():.6f}")
    
    # Check gradients after one step
    for Y_prev, Y_curr in train_loader:
        Y_prev, Y_curr = Y_prev.float().to(device), Y_curr.float().to(device)
        spikes = model(Y_prev)
        loss = poisson_nll_loss(spikes, Y_curr, torch.tensor(dataRates, dtype=torch.float32, device=device))
        loss.backward()
        
        # Check gradients for efficient parameterization
        if hasattr(model, 'C') and model.C.grad is not None:
            C_grad_norm = model.C.grad.norm().item()
            C_grad_max = model.C.grad.abs().max().item()
            print(f"C gradient: norm={C_grad_norm:.6f}, max={C_grad_max:.6f}")
        break
    
    model.zero_grad()  # Clear gradients before actual training
    
    start_time = time.time()
    train_losses, val_losses, learning_rates = train_Poisson_model(
        model, device, train_loader, val_loader, args.n_epochs,
        lr=args.lr, L1_alpha=0.0, use_scheduler=True, firing_rates=dataRates
    )
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds")

    # --- Modern output saving from fit_stageA.py ---
    bigD = {
        'A_regress': model.A.detach().cpu().numpy(), 
        'B_regress': model.B.detach().cpu().numpy(),
        'train_losses': np.array(train_losses), 
        'val_losses': np.array(val_losses),
        'learning_rates': np.array(learning_rates),
        'firing_rates': dataRates
    }

    metaD = {
        'regressFit_output_name': args.fitName, 'regressFit_input_name': args.dataName,
        'lassoFit_input_name': spike_data_name,
        'batch_size': args.batch_size, 'num_samples_used': n_pairs, 'n_epochs': args.n_epochs,
        'num_train_samples': n_train, 'num_val_samples': n_pairs-n_train,
        'learning_rate': args.lr, 'step_size': step_size,
        'training_time_sec': total_time, 'num_neurons': M,
        'desync_time': args.desync_time,
    }
    fitMD['fit_regress']=metaD
    fitMD['short_name']=args.fitName
    write_data_npz(bigD, fit2FF, metaD=fitMD)

    print('\n  ./eval_fit.py  --dataName %s ' % (args.fitName))

if __name__ == "__main__":
    main()
