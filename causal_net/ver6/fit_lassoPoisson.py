#!/usr/bin/env python3
"""
Multi-GPU distributed training of Poisson GLM with LASSO regularization.

This script implements distributed training of a Poisson Generalized Linear Model
for neural connectivity inference with L1 (LASSO) regularization. Key features:
- Multi-GPU support using PyTorch DistributedDataParallel 
- LASSO regularization for sparse connectivity estimation
- Poisson negative log-likelihood loss optimized for spike count data
- Support for time decorrelation and data shuffling
- Efficient data loading with distributed sampling
- Automatic model checkpointing and metadata saving

Usage:
    Multi-GPU: OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 ./fit_lassoPoisson.py --dataName mydata --num_epochs 100
    Single GPU: ./fit_lassoPoisson.py --dataName mydata --num_epochs 100

Example SLURM execution:
    salloc -q interactive -C gpu -t 4:00:00 -A m2043 -N 1
    module load pytorch
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 ./fit_lassoPoisson.py --dataName daleM600_443813 --num_epochs 5 --dataPath $dataPath
"""

import os
import time
import random
import argparse
import numpy as np
from pprint import pprint
import torch.optim as optim
import torch
from torch.utils.data import TensorDataset, DataLoader

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
from PoissonGLModel import PoissonGLModel, poisson_nll_loss

from UtilTorch import check_gpu_availability, preprocess_data, train_Poisson_model, NumpyPairDataset, make_loader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset

#########################
#  MAIN
#########################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataName", type=str, default="dale_2aee70")
    parser.add_argument("--dataPath", type=str, default="/pscratch/sd/b/balewski/2025_causalNet_tmp/")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--L1_alpha", type=float, default=1e-3)
    parser.add_argument("--fitName", type=str, default=None)
    parser.add_argument("--Tmask", action='store_true', help="Use time mask to remove time bins from data")
    parser.add_argument("--shuffleTime", action='store_true', help="If true completely shuffle time axis for input data, independently for all channels")
    parser.add_argument("--desyncTime", action='store_true', help="If true completely shuffle time axis for input data, independently for all channels")
    parser.add_argument("--dropDataFrac", type=float, default=0.0, help="Fraction of training samples to randomly drop per rank (0.0=use all data, 0.3=drop 30%%)")
 
    args = parser.parse_args()
    
    # DDP init
    is_dist = (int(os.environ.get('WORLD_SIZE', '1')) > 1) or ('LOCAL_RANK' in os.environ) or ('RANK' in os.environ)
    #print('is_dist;',is_dist,os.environ.get('WORLD_SIZE', '1'),'RANK' in os.environ)
    
    if is_dist:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0; rank = 0; world_size = 1
        device = check_gpu_availability()
    args.rank=rank
    if rank==0:
        print("Configuration:", vars(args))
        print("world_size=%d" % (world_size))
    # enable fast matmul paths
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    gpu_name = torch.cuda.get_device_name(device) if isinstance(device, torch.device) and device.type=='cuda' else str(device)
    print("[rank %d] Using device %s : %s" % (rank, str(device), gpu_name))
    
    # --- data loading  ---
    spikesFF = os.path.join(args.dataPath, f"{args.dataName}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF,verb=rank==0)
    dataYield, dataRates = spikeD['spikes'], np.clip(spikeD['single_rates'], 0.1, 40.0)
    T, M = dataYield.shape
    pprint(spikeMD)
    step_size = spikeMD['time_step_sec']
   
    # --- time mask loading ---
    time_mask = None
    if args.Tmask:
        maskFF = os.path.join(args.dataPath, f"{args.dataName}.tmask.npz")
        maskD, maskMD = read_data_npz(maskFF, verb=rank==0)
        time_mask = maskD['time_mask']

        #1time_mask=~ time_mask  ; print('WARN burst-mask reversed')
        
        # Handle size mismatch - clip mask if it's longer than data
        if len(time_mask) > T:
            if rank==0:
                print(f"Time mask length ({len(time_mask)}) > data length ({T}), clipping mask")
            time_mask = time_mask[:T]
        elif len(time_mask) < T:
            if rank==0:
                print(f"Time mask length ({len(time_mask)}) < data length ({T}), using available mask")
            # Keep the mask as is, preprocess_data will handle the shorter length
 
    # Apply decorrelation if requested
    if args.desyncTime > 0:
        if rank==0: print("\n=== Applying Time Decorrelation, it shifts time for each neuron ===")
        
        # Rank 0 generates random shift amounts for all neurons
        if rank == 0:
            # Use current time + process info to create different shifts each run
            seed = int(time.time() * 1000) % 1000000  # Use millisecond timestamp as seed
            np.random.seed(seed)
            shift_amounts = np.random.randint(1, T//4, size=M, dtype=np.int32)  # Random shifts between 1 and T/4
            if rank == 0:
                print('Generated random shifts with seed=%d'%(seed),shift_amounts[:10],'...',flush=True)
        else:
            shift_amounts = np.zeros(M, dtype=np.int32)
        
        # Broadcast shift amounts from rank 0 to all other ranks
        if is_dist:
            # Ensure all ranks have the tensor on the same device and dtype
            shift_amounts_tensor = torch.tensor(shift_amounts, dtype=torch.int32, device='cuda')
            dist.broadcast(shift_amounts_tensor, src=0)
            shift_amounts = shift_amounts_tensor.cpu().numpy()
        
        # Apply the same shifts on all ranks
        #print('myrank=',rank,'shift_amounts=',shift_amounts[:10],flush=True)
        Y = np.zeros_like(dataYield)        
        for neuron_idx in range(M):
            shift_amount = int(shift_amounts[neuron_idx])
            # Circular shift: move data to the right, wrap around
            Y[:, neuron_idx] = np.roll(dataYield[:, neuron_idx], shift_amount)
        
        if rank==0: print(f"Applied synchronized time shifts across all ranks, destroys temporal correlations between neurons")
        dataYield = Y 
        
    X_np, Yt_np = preprocess_data(dataYield, args, time_mask=time_mask)
    n_pairs = X_np.shape[0]
    
    # --- Random data dropping per rank ---
    if args.dropDataFrac > 0:
        # Create different random seed for each rank based on rank + current time
        # Keep seed within valid numpy range (0 to 2^32 - 1)
        drop_seed = (int(time.time() * 1000) + rank * 1000 + np.random.randint(0, 1000)) % (2**32)
        np.random.seed(drop_seed)
        random.seed(drop_seed)
        
        # Calculate how many samples to keep
        n_keep = int(n_pairs * (1.0 - args.dropDataFrac))
        
        # Randomly select indices to keep (different for each rank)
        keep_indices = np.random.choice(n_pairs, size=n_keep, replace=False)
        keep_indices = np.sort(keep_indices)  # Sort for better memory access
        
        # Filter data
        X_np = X_np[keep_indices]
        Yt_np = Yt_np[keep_indices]
        n_pairs = X_np.shape[0]
        
        print(f"[rank {rank}] Dropped {args.dropDataFrac:.1%} of data, keeping {n_pairs} samples (seed={drop_seed})")
    
    assert n_pairs >= args.batch_size, f"ERROR: Not enough samples ({n_pairs}) for batch size ({args.batch_size}) after data dropping."

    train_loader = make_loader(X_np, Yt_np, args, is_dist=is_dist)
    
    if rank==0:
        print(f"Loaded pairs={n_pairs/1000}k, M={M}, using {n_pairs/1000}k pairs (all for training), world_size={world_size}, per_gpu_bs={args.batch_size//max(1,world_size)}")

    # --- Original training logic from fit_poissonV4.py ---
    base_model = PoissonGLModel(M).to(device)
    model = DDP(base_model, device_ids=[local_rank]) if is_dist else base_model
    start_time = time.time()
    losses_total, losses_wo_L1, learning_rates, train_epochs = train_Poisson_model(
        model, device, train_loader, args.num_epochs, lr=args.lr, L1_alpha=args.L1_alpha, firing_rates=dataRates, use_scheduler=True,
        train_sampler=train_loader.sampler if isinstance(train_loader.sampler, DistributedSampler) else None
    )
    total_time = time.time() - start_time
    if rank==0:
        print(f"Training completed in {total_time:.1f} seconds")
  
        if args.fitName is None:
            import string
            hash_str =  ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            fit_core = f"{args.dataName}-{hash_str}"
        else:
            fit_core = args.fitName

        # saving from fit_Lasso ---
        mdl = model.module if hasattr(model,'module') else model
        lassoD = { 'A_lasso': mdl.A.detach().cpu().numpy(), 'B_lasso': mdl.B.detach().cpu().numpy(), 'losses_total': np.array(losses_total), 'losses_wo_L1': np.array(losses_wo_L1), 'losses_epochs': np.array(train_epochs, dtype=np.int32), 'learning_rates': np.array(learning_rates), 'single_rates': dataRates }
        lassoMD = { 'lassoFit_output_name': fit_core, 'lassoFit_input_name': args.dataName, 'batch_size': args.batch_size, 'num_samples_used': n_pairs, 'n_epochs': args.num_epochs, 'num_train_samples': n_pairs, 'learning_rate': args.lr, 'L1_alpha': args.L1_alpha, 'step_size': step_size, 'training_time_sec': total_time, 'num_neurons': M, 'dropDataFrac': args.dropDataFrac }
        spikeMD['fit_type']='lasso'        
        spikeMD['fit_lasso']=lassoMD
        spikeMD['edge_selector']={'selector_type':'None'}
          
        fitFF = os.path.join(args.dataPath, f"{fit_core}.lassoFit.npz")
        write_data_npz(lassoD, fitFF, metaD=spikeMD)

    if rank==0:
        if spikeMD['data_type']=='simDale':         flags=' -p  a  c  '
        else:         flags=' -p a c  '
        print('\n  ./eval_fitLasso.py --dataPath $dataPath  --dataName %s  %s  ' % (fit_core,flags))
        print('\n  ./fit_regressPoisson.py  --dataName %s  ' % (fit_core))
        print('    --dataPath '+args.dataPath)
    
    # ensure distributed shutdown to avoid resource leak warning
    if is_dist and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
