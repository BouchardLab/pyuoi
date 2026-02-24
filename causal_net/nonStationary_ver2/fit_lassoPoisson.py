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
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 ./fit_lassoPoisson.py --dataName daleM600_443813 --num_epochs 5 --basePath $dataPath
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
    parser.add_argument("--basePath", type=str, default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="head dir for input/output data")
    parser.add_argument("--inpPath", type=str, default=None, help="alternative location of input, takes precedence")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--L1_alpha", type=float, default=1e-3)
    parser.add_argument("--fitName", type=str, default=None)
    parser.add_argument("--desyncTime", action='store_true', help="If true completely shuffle time axis for input data, independently for all channels")
    parser.add_argument("--dropDataFrac", type=float, default=0.0, help="Fraction of training samples to randomly drop per rank (0.0=use all data, 0.3=drop 30%%)")

    args = parser.parse_args()
    if args.inpPath ==None:
        args.inpPath = os.path.join(args.basePath, 'spikesData')
    args.outPath = os.path.join(args.basePath, 'lassoFdrFit')
    os.makedirs(args.outPath, exist_ok=True)

    # DDP init
    is_dist = (int(os.environ.get('WORLD_SIZE', '1')) > 1) or ('LOCAL_RANK' in os.environ) or ('RANK' in os.environ)
    #print('is_dist;',is_dist,os.environ.get('WORLD_SIZE', '1'),'RANK' in os.environ)
    
    if is_dist:
        # set CUDA device first so init_process_group can pick it up
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        # PyTorch 2.8 warns if device id isn't provided; fall back for older versions
        try:
            dist.init_process_group(backend='nccl', device_id=local_rank)
        except TypeError:
            dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0; rank = 0; world_size = 1
        device = check_gpu_availability()
    args.rank=rank
    if rank==0:
        print("FitLasso Config:", vars(args))
        print("world_size=%d" % (world_size))
    # enable fast matmul paths
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    gpu_name = torch.cuda.get_device_name(device) if isinstance(device, torch.device) and device.type=='cuda' else str(device)
    print("[rank %d] Using device %s : %s" % (rank, str(device), gpu_name))
    
    # --- data loading and preprocessing (rank 0 only) ---
    if rank == 0:
        spikesFF = os.path.join(args.inpPath, f"{args.dataName}.spikes.npz")
        spikeD, spikeMD = read_data_npz(spikesFF, verb=True)
        pprint(spikeMD)
        dataYield = spikeD['spikes']
        dataRates = spikeD['single_rates']
        step_size = spikeMD['time_step_sec']
        if 'simDale' in spikeMD['data_type']:
            Mstate,Nt, Nn = dataYield.shape       
            #assert Mstate==1  # tmp, fix it for multi-mode
            dataYield=dataYield[0] 
            dataRates=dataRates[0]
        else:
            Nt, Nn = dataYield.shape   
        
        XY_np = preprocess_data(dataYield, args)
        n_pairs = XY_np.shape[0]
        print(f"Rank 0 preprocessed data: XY shape={XY_np.shape}, n_pairs={n_pairs}")
    else:
        XY_np = None
        Nn = None
        dataRates = None
        spikeMD = None
        step_size = None
    
    # --- Broadcast data from rank 0 to all other ranks ---
    if is_dist:
        Nn_tensor = torch.tensor([Nn if rank==0 else 0], dtype=torch.int32, device='cuda')
        dist.broadcast(Nn_tensor, src=0)
        Nn = Nn_tensor.item()
        if rank == 0:
            shape_info = torch.tensor([XY_np.shape[0], XY_np.shape[1], XY_np.shape[2]], dtype=torch.int64, device='cuda')
        else:
            shape_info = torch.zeros(3, dtype=torch.int64, device='cuda')
        dist.broadcast(shape_info, src=0)
        if rank != 0:
            XY_np = np.zeros((shape_info[0].item(), shape_info[1].item(), shape_info[2].item()), dtype=np.float32)
        else:
            XY_np = np.ascontiguousarray(XY_np, dtype=np.float32)
        XY_tensor = torch.tensor(XY_np, dtype=torch.float32, device='cuda').contiguous()
        dist.broadcast(XY_tensor, src=0)
        XY_np = XY_tensor.cpu().numpy()
        if rank != 0:
            dataRates = torch.zeros(Nn, dtype=torch.float32, device='cuda')
            step_size_tensor = torch.zeros(1, dtype=torch.float32, device='cuda')
        else:
            dataRates = torch.tensor(dataRates, dtype=torch.float32, device='cuda')
            step_size_tensor = torch.tensor([step_size], dtype=torch.float32, device='cuda')
        dist.broadcast(dataRates, src=0)
        dist.broadcast(step_size_tensor, src=0)
        dataRates = dataRates.cpu().numpy()
        step_size = step_size_tensor.item()
        if rank != 0:
            print(f"[rank {rank}] Received broadcasted data: XY shape={XY_np.shape}")
    
    # Split XY into X and Y for all ranks
    X_np = XY_np[:, 0, :]
    Yt_np = XY_np[:, 1, :]
    n_pairs = X_np.shape[0]
    
    assert n_pairs >= args.batch_size, f"ERROR: Not enough samples ({n_pairs}) for batch size ({args.batch_size}) after data dropping."

    train_loader = make_loader(X_np, Yt_np, args, is_dist=is_dist)
    
    if rank==0:
        print(f"Loaded pairs={n_pairs/1000}k, Nn={Nn}, using {n_pairs/1000}k pairs (all for training), world_size={world_size}, per_gpu_bs={args.batch_size//max(1,world_size)}")

    # --- Original training logic from fit_poissonV4.py ---
    base_model = PoissonGLModel(Nn).to(device)
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
        A_hat=mdl.A.detach().cpu().numpy()
        E_hat = (np.abs(A_hat) > 1e-5)
        lassoD = { 'A_lasso': A_hat, 'B_lasso': mdl.B.detach().cpu().numpy(), 'E_lasso':E_hat, 'losses_total': np.array(losses_total), 'losses_wo_L1': np.array(losses_wo_L1), 'losses_epochs': np.array(train_epochs, dtype=np.int32), 'learning_rates': np.array(learning_rates), 'single_rates': dataRates }
        lassoMD = { 'lassoFit_output_name': fit_core, 'lassoFit_input_name': args.dataName,  'lassoFit_input_path': args.inpPath ,'batch_size': args.batch_size, 'num_samples_used': n_pairs, 'num_epochs': args.num_epochs, 'num_train_samples': n_pairs, 'learning_rate': args.lr, 'L1_alpha': args.L1_alpha, 'step_size': step_size, 'training_time_sec': total_time, 'num_neurons': Nn, 'dropDataFrac': args.dropDataFrac }
        spikeMD['fit_type']='lasso'        
        spikeMD['fit_lasso']=lassoMD
        spikeMD['edge_selector']={'selector_type':'None'}
          
        fitFF = os.path.join(args.outPath, f"{fit_core}.lassoFit.npz")
        write_data_npz(lassoD, fitFF, metaD=spikeMD)

    if rank==0:
        if spikeMD['data_type']=='simDale':         flags=' -p  a  b c  '
        else:         flags=' -p a c  '
        print('    basePath='+args.basePath)
        print('  ./eval_fitLasso.py --basePath $basePath  --dataName %s  %s \n ' % (fit_core,flags))    
    # ensure distributed shutdown to avoid resource leak warning
    if is_dist and dist.is_initialized():
        try:
            dist.barrier(device_ids=[local_rank])
        except TypeError:
            dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
