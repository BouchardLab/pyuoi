#!/usr/bin/env python3
''' mutli GPU & 1 node execution
 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 ./fit_lassoPoisson.py --dataName daleM600_443813 --batch_size 16384 --n_epochs 5 --dataPath /pscratch/sd/b/balewski/tmp2 
'''

import os
import time
import argparse
import numpy as np
from pprint import pprint
import torch.optim as optim
import torch
from torch.utils.data import TensorDataset, DataLoader

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
from PoissonGLModel import PoissonGLModel, poisson_nll_loss

from UtilTorch import check_gpu_availability, preprocess_data, train_Poisson_model
from UtilDalePoissonV5 import select_eges_from_fitLasso

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset

VAL_EVERY = 10  # validate every N epochs

#########################
#  MAIN
#########################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataName", type=str, default="dale_2aee70")
    parser.add_argument("--dataPath", type=str, default="out/")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=2048*8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--L1_alpha", type=float, default=1e-3)
    parser.add_argument("--fitName", type=str, default=None)
    parser.add_argument("--desync_time", type=int, default=0, help="Time shift for decorrelation (0=disabled, >0=shift consecutive neurons by this many time bins)")
    parser.add_argument('-a',"--ampl_thres", type=float, default=0.05, help="minima amplitude of valid off-diagonal edge")

    args = parser.parse_args()

    # DDP init
    is_dist = (int(os.environ.get('WORLD_SIZE', '1')) > 1) or ('LOCAL_RANK' in os.environ) or ('RANK' in os.environ)
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
    if rank==0:
        print("Configuration:", vars(args))
        print("world_size=%d" % (world_size))
    # enable fast matmul paths
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    gpu_name = torch.cuda.get_device_name(device) if isinstance(device, torch.device) and device.type=='cuda' else str(device)
    print("[rank %d] Using device %s : %s" % (rank, str(device), gpu_name))
    
    # --- Modern data loading  ---
    spikesFF = os.path.join(args.dataPath, f"{args.dataName}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF,verb=rank==0)
    dataYield, dataRates = spikeD['spikes'], np.clip(spikeD['single_rates'], 0.1, 40.0)
    T, M = dataYield.shape
    step_size = spikeMD['dale_simu_stats']['time_step_sec']
    
    if args.fitName is None:
        import random, string
        hash_str =  ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        fit_core = f"{args.dataName}_{hash_str}"
    else:
        fit_core = args.fitName
    
    X_np, Yt_np = preprocess_data(dataYield, args)
    n_pairs, train_split = X_np.shape[0], 0.8
    n_train = int(n_pairs * train_split)
    
    assert n_train >= args.batch_size, f"ERROR: Not enough training samples ({n_train}) for batch size ({args.batch_size})."

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

    # --- Original training logic from fit_poissonV4.py ---
    base_model = PoissonGLModel(M).to(device)
    model = DDP(base_model, device_ids=[local_rank]) if is_dist else base_model
    start_time = time.time()
    train_losses, val_losses, learning_rates, train_epochs, val_epochs = train_Poisson_model(
        model, device, train_loader, val_loader, args.n_epochs, lr=args.lr, L1_alpha=args.L1_alpha, firing_rates=dataRates, use_scheduler=True,
        train_sampler=train_loader.sampler if isinstance(train_loader.sampler, DistributedSampler) else None
    )
    total_time = time.time() - start_time
    if rank==0:
        print(f"Training completed in {total_time:.1f} seconds")

    # --- Modern output saving from fit_stageA.py ---
    if rank==0:
        mdl = model.module if hasattr(model,'module') else model
        lassoD = { 'A_lasso': mdl.A.detach().cpu().numpy(), 'B_lasso': mdl.B.detach().cpu().numpy(), 'train_losses': np.array(train_losses), 'val_losses': np.array(val_losses), 'train_loss_epochs': np.array(train_epochs, dtype=np.int32), 'val_loss_epochs': np.array(val_epochs, dtype=np.int32), 'learning_rates': np.array(learning_rates), 'firing_rates': dataRates }
        lassoMD = { 'lassoFit_output_name': fit_core, 'lassoFit_input_name': args.dataName, 'batch_size': args.batch_size, 'num_samples_used': n_pairs, 'n_epochs': args.n_epochs, 'num_train_samples': n_train, 'num_val_samples': n_pairs-n_train, 'learning_rate': args.lr, 'L1_alpha': args.L1_alpha, 'step_size': step_size, 'training_time_sec': total_time, 'num_neurons': M }
        # ... select edges in A_fit matrix ...
        lassoD['ampl_thres']= args.ampl_thres
        spikeMD['fit_lasso']=lassoMD
        maskF=select_eges_from_fitLasso(lassoD,args.ampl_thres)
        for xx in maskF:
            lassoD['mask.lasso.'+xx]=maskF[xx]
        spikeMD['short_name']=fit_core
     
    if rank==0:
        fitFF = os.path.join(args.dataPath, f"{fit_core}.lasso.npz")
        write_data_npz(lassoD, fitFF, metaD=spikeMD)

    if rank==0:
        print('\n  ./eval_fit.py  --dataName %s   -p a b ' % (fit_core))
        print('\n  ./fit_regressPoisson.py  --dataName %s  ' % (fit_core))
        print(' --dataPath /pscratch/sd/b/balewski/tmp2')
    
    # ensure distributed shutdown to avoid resource leak warning
    if is_dist and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
