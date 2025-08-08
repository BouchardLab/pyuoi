#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
import torch.optim as optim
import torch
from torch.utils.data import TensorDataset, DataLoader

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
from PoissonGLModel import PoissonGLModel, poisson_nll_loss

from UtilTorch import check_gpu_availability, preprocess_data,  train_Poisson_model
from pprint import pprint

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

     # DDP init
    is_dist = (int(os.environ.get('WORLD_SIZE', '1')) > 1) or ('LOCAL_RANK' in os.environ) or ('RANK' in os.environ)
    if is_dist:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data.distributed import DistributedSampler
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0; rank = 0; world_size = 1
        device = check_gpu_availability()
    if rank==0:
        print("world_size=%d" % (world_size))
        print("Initial configuration:", vars(args))
   
    # enable fast matmul paths
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    gpu_name = torch.cuda.get_device_name(device) if isinstance(device, torch.device) and device.type=='cuda' else str(device)
    print("[rank %d] Using device %s : %s" % (rank, str(device), gpu_name))
    
    fit1FF = os.path.join(args.dataPath, f"{args.dataName}.lasso.npz")
    fitD, fitMD = read_data_npz(fit1FF,verb=rank==0)
    A_init = fitD['A_pass']
    B_init = fitD['B_lasso']

    if rank==0: pprint(fitMD)
    fmd=fitMD['fit_lasso']
    # Inherit hyperparams from stageA if not provided
    if args.n_epochs is None:
        args.n_epochs = fmd['n_epochs']
    if args.batch_size is None:
        args.batch_size = fmd['batch_size']
         
    if rank==0: print("Effective configuration:", vars(args))
    
    spike_data_name = fitMD['input_file']
    spikesFF = os.path.join(args.dataPath, f"{spike_data_name}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF,verb=rank==0)
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

    from torch.utils.data import Dataset
    class NumpyPairDataset(Dataset):
        def __init__(self, X_np, Y_np):
            self.X = X_np
            self.Y = Y_np
            self.n = X_np.shape[0]
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            return torch.from_numpy(self.X[idx]).to(dtype=torch.float32), torch.from_numpy(self.Y[idx]).to(dtype=torch.float32)

    def make_loader(X, Yt, shuffle=True):
        dataset = NumpyPairDataset(X, Yt)
        if is_dist:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            return DataLoader(dataset, batch_size=max(1, args.batch_size//world_size), sampler=sampler, shuffle=False, drop_last=shuffle,
                              pin_memory=True, pin_memory_device='cuda', num_workers=8, persistent_workers=True, prefetch_factor=8)
        else:
            return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, drop_last=shuffle, pin_memory=True, pin_memory_device='cuda', num_workers=8,
                              persistent_workers=True, prefetch_factor=8)
    
    train_loader = make_loader(X_np[:n_train], Yt_np[:n_train])
    val_loader = make_loader(X_np[n_train:], Yt_np[n_train:], shuffle=False)
    
    if rank==0:
        print(f"Loaded T={T}, M={M}, using {n_pairs} pairs ({n_train} train, {n_pairs-n_train} val), world_size={world_size}, per_gpu_bs={args.batch_size//max(1,world_size)}")

    # --- Set up model for stage B ---
    trainable_mask = fitD['mask.lasso.pass']
    if rank==0:
        print(f"Trainable mask: {trainable_mask.sum()} out of {trainable_mask.size} A elements are trainable")
        print(f"Trainable fraction: {trainable_mask.mean():.3f}")
    
    base_model = PoissonGLModel(M, A_init=A_init, B_init=B_init, trainable_mask=trainable_mask, noise_scale=args.noise_scale).to(device)
    model = DDP(base_model, device_ids=[local_rank]) if is_dist else base_model
    mdl = model.module if hasattr(model,'module') else model
    
    # Debug: check if noise was actually applied
    if args.noise_scale > 0:
        if rank==0: print(f"Checking noise application...")
        A_reconstructed = mdl.A.detach().cpu().numpy()
        A_diff = np.abs(A_reconstructed - A_init).max()
        B_reconstructed = mdl.B.detach().cpu().numpy()
        B_diff = np.abs(B_reconstructed - B_init).max()
        if rank==0:
            print(f"Max A difference from init: {A_diff:.6f}")
            print(f"Max B difference from init: {B_diff:.6f}")
    
    # Debug: count trainable parameters
    total_params = sum(p.numel() for p in mdl.parameters())
    trainable_params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    expected_trainable = trainable_mask.sum() + M  # masked A elements + B vector
    if rank==0:
        print(f"Total parameters: {total_params}, PyTorch trainable: {trainable_params}, Expected trainable: {expected_trainable}")
    
    if hasattr(mdl, 'is_stage_b') and mdl.is_stage_b:
        if rank==0: print(f"Stage B efficient parameterization: C tensor has {mdl.C.numel()} parameters")
    else:
        if rank==0: print("Stage A standard parameterization")
    
    # Debug: check initial loss and parameter values
    model.train()
    
    # Check initial loss before training
    with torch.no_grad():
        for Y_prev, Y_curr in train_loader:
            Y_prev, Y_curr = Y_prev.float().to(device), Y_curr.float().to(device)
            spikes = model(Y_prev)
            initial_loss = poisson_nll_loss(spikes, Y_curr, torch.tensor(dataRates, dtype=torch.float32, device=device))
            if rank==0: print(f"Initial loss with noise: {initial_loss:.6f}")
            break
    
    # Check if C parameters actually changed from Stage A
    if hasattr(mdl, 'C'):
        C_values = mdl.C.detach().cpu()
        if rank==0: print(f"C parameter stats: min={C_values.min():.6f}, max={C_values.max():.6f}, std={C_values.std():.6f}")
    
    # Check gradients after one step
    for Y_prev, Y_curr in train_loader:
        Y_prev, Y_curr = Y_prev.float().to(device), Y_curr.float().to(device)
        spikes = model(Y_prev)
        loss = poisson_nll_loss(spikes, Y_curr, torch.tensor(dataRates, dtype=torch.float32, device=device))
        loss.backward()
        
        # Check gradients for efficient parameterization
        if hasattr(mdl, 'C') and mdl.C.grad is not None:
            C_grad_norm = mdl.C.grad.norm().item()
            C_grad_max = mdl.C.grad.abs().max().item()
            if rank==0: print(f"C gradient: norm={C_grad_norm:.6f}, max={C_grad_max:.6f}")
        break
    
    model.zero_grad()  # Clear gradients before actual training
    
    start_time = time.time()
    train_losses, val_losses, learning_rates, train_epochs, val_epochs = train_Poisson_model(
        model, device, train_loader, val_loader, args.n_epochs, lr=args.lr, L1_alpha=0.0, use_scheduler=True, firing_rates=dataRates,
        train_sampler=train_loader.sampler if is_dist else None
    )
    total_time = time.time() - start_time
    if rank==0:
        print(f"Training completed in {total_time:.1f} seconds")

    # --- Modern output saving from fit_stageA.py ---
    if rank==0:
        mdl = model.module if hasattr(model,'module') else model
        bigD = { 'A_regress': mdl.A.detach().cpu().numpy(), 'B_regress': mdl.B.detach().cpu().numpy(), 'train_losses': np.array(train_losses), 'val_losses': np.array(val_losses), 'train_loss_epochs': np.array(train_epochs, dtype=np.int32), 'val_loss_epochs': np.array(val_epochs, dtype=np.int32), 'learning_rates': np.array(learning_rates), 'firing_rates': dataRates }
        metaD = { 'regressFit_output_name': args.fitName, 'regressFit_input_name': args.dataName, 'lassoFit_input_name': spike_data_name, 'batch_size': args.batch_size, 'num_samples_used': n_pairs, 'n_epochs': args.n_epochs, 'num_train_samples': n_train, 'num_val_samples': n_pairs-n_train, 'learning_rate': args.lr, 'step_size': step_size, 'training_time_sec': total_time, 'num_neurons': M, 'desync_time': args.desync_time }
        fitMD['fit_regress']=metaD
        fitMD['short_name']=args.fitName
        write_data_npz(bigD, fit2FF, metaD=fitMD)

    if rank==0:
        print('\n  ./eval_fit.py  --dataName %s -p a ' % (args.fitName))

    # ensure distributed shutdown to avoid resource leak warning
    if is_dist and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
