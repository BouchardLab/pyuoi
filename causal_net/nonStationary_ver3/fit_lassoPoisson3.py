#!/usr/bin/env python3
"""
Single-GPU training of Poisson GLM with LASSO regularization.

This script fits a Poisson Generalized Linear Model for neural connectivity
inference with L1 (LASSO) regularization on a single GPU.

Usage:
    ./fit_lassoPoisson3.py --dataName mydata --num_epochs 100
"""

import os
import time
import secrets
import argparse


import numpy as np
from pprint import pprint
import torch

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
from PoissonGLModel import PoissonGLModel

from UtilTorch import check_gpu_availability, preprocess_data, train_Poisson_model, make_loader

#########################
#  MAIN
#########################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataName", type=str, default="dale_2aee70")
    parser.add_argument("--basePath", type=str, default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="head dir for input/output data")
    parser.add_argument("--inpPath", type=str, default=None, help="alternative location of input, takes precedence")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--L1_alpha", type=float, default=0.02, help="L1 regularization strength, higher=more sparse (0: disable soft-thresholding)")
    parser.add_argument("--rho_max", type=float, default=0.97, help="Maximum allowed spectral radius of A; projection applied after each batch")
    parser.add_argument("--rho_enforce_every_batch", type=int, default=20, help="Apply spectral-radius projection every N batches")
    parser.add_argument("--L1_prune_epoch", type=int, default=50, help="Delay L1 edge-pruning only; spectral-radius correction starts immediately")
    parser.add_argument("--minW", type=float, default=0.01, help="Threshold for A-matrix eval, not for fitting")
    parser.add_argument("--fitName", type=str, default=None)
    parser.add_argument("--desyncTime", action='store_true', help="If true completely shuffle time axis for input data, independently for all channels")
    parser.add_argument("--dropDataFrac", type=float, default=0.0, help="Fraction of training samples to randomly drop (0.0=use all data, 0.3=drop 30%%)")
    parser.add_argument("--verb", "-v", type=int, default=1, help="Verbosity level")

    args = parser.parse_args()
    if args.inpPath ==None:
        args.inpPath = os.path.join(args.basePath, 'spikesData')
    outPath = os.path.join(args.basePath, 'lassoFdrFit')
    assert os.path.exists(outPath) 

    device = check_gpu_availability()
    print("\nFitLasso3 Config:", vars(args,), "\n")
    # enable fast matmul paths
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    gpu_name = torch.cuda.get_device_name(device) if isinstance(device, torch.device) and device.type=='cuda' else str(device)
    print("Using device %s : %s" % (str(device), gpu_name))

    spikesFF = os.path.join(args.inpPath, f"{args.dataName}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF, verb=True)
    if args.verb > 1:
        pprint(spikeMD)
    dataYield = spikeD['spikes']
    dataRates = spikeD['single_rates']
    step_size = spikeMD['time_step_sec']
    if 'simDale' in spikeMD['data_type']:
        _, _, Nn = dataYield.shape
        dataYield = dataYield[0]
        dataRates = dataRates[0]
    else:
        _, Nn = dataYield.shape

    XY_np = preprocess_data(dataYield, args)
    n_pairs = XY_np.shape[0]
    print(f"Preprocessed data: XY shape={XY_np.shape}, n_pairs={n_pairs}")

    # Split XY into X and Y
    X_np = XY_np[:, 0, :]
    Yt_np = XY_np[:, 1, :]
    n_pairs = X_np.shape[0]
    
    assert n_pairs >= args.batch_size, f"ERROR: Not enough samples ({n_pairs}) for batch size ({args.batch_size}) after data dropping."

    train_loader = make_loader(X_np, Yt_np, args, is_dist=False)
    print(f"Loaded pairs={n_pairs/1000}k, Nn={Nn}, using {n_pairs/1000}k pairs (all for training), batch_size={args.batch_size}")

    model = PoissonGLModel(Nn).to(device)
    start_time = time.time()
    losses_total, losses_wo_L1, learning_rates, train_epochs, sparsity_epoch, nz_offdiag_epoch, spectral_radius_epoch = train_Poisson_model(
        model, device, train_loader, args.num_epochs, lr=args.lr, L1_alpha=args.L1_alpha, firing_rates=dataRates, use_scheduler=True,
        train_sampler=None, apply_prox=(args.L1_alpha > 0), minW=args.minW, rho_max=args.rho_max, rho_enforce_every_batch=args.rho_enforce_every_batch, L1_prune_epoch=args.L1_prune_epoch
    )

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds")
  
    if args.fitName is None:
        hash6 = secrets.token_hex(3) # 6 hex digits
        fit_core = f"{args.dataName}-{hash6}"
    else:
        fit_core = args.fitName

    # saving from fit_Lasso ---
    mdl = model
    A_hat=mdl.A.detach().cpu().numpy()
    E_hat = (np.abs(A_hat) > 1e-5)
    lassoD = { 'A_lasso': A_hat, 'B_lasso': mdl.B.detach().cpu().numpy(), 'E_lasso':E_hat, 'losses_total': np.array(losses_total), 'losses_wo_L1': np.array(losses_wo_L1), 'losses_epochs': np.array(train_epochs, dtype=np.int32), 'learning_rates': np.array(learning_rates), 'sparsity_epoch': np.array(sparsity_epoch, dtype=np.float32), 'nz_offdiag_epoch': np.array(nz_offdiag_epoch, dtype=np.int32), 'spectral_radius_epoch': np.array(spectral_radius_epoch, dtype=np.float32), 'single_rates': dataRates }
    lassoMD = {'batch_size': args.batch_size, 'num_samples_used': n_pairs, 'num_epochs': args.num_epochs, 'num_train_samples': n_pairs, 'learning_rate': args.lr, 'L1_alpha': args.L1_alpha, 'rho_max': args.rho_max, 'rho_enforce_every_batch': args.rho_enforce_every_batch, 'L1_prune_epoch': args.L1_prune_epoch, 'minW': args.minW, 'step_size': step_size, 'training_time_sec': total_time, 'num_neurons': Nn, 'dropDataFrac': args.dropDataFrac }
    #  'lassoFit_output_name': fit_core, 'lassoFit_input_name': args.dataName,  'lassoFit_input_path': args.inpPath ,
   
    outMD=spikeMD
    outMD['fit_type']='lasso'        
    outMD['fit_lasso']=lassoMD
    outMD['edge_selector']={'selector_type':'None'}
    outMD['provenance']['output_lasso_file']=fit_core

    if args.verb>1: pprint(outMD)
    fitFF = os.path.join(outPath, f"{fit_core}.lassoFit.npz")
    write_data_npz(lassoD, fitFF, metaD=outMD)

    if spikeMD['data_type']=='simDale':         flags=' -p  a  b c  '
    else:         flags=' -p a d e '
    print('    basePath='+args.basePath)
    print('  ./eval_fitLasso.py --basePath $basePath  --dataName %s  %s \n ' % (fit_core,flags))    

if __name__ == "__main__":
    main()
