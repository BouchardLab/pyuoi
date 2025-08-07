#!/usr/bin/env python3
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

from UtilTorch import check_gpu_availability, preprocess_data
from UtilDalePoissonV5 import select_eges_from_fitLasso

def train_Poisson_model(model, device, train_loader, val_loader, n_epochs, lr, L1_alpha=0.0, use_scheduler=False, firing_rates=None):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=n_epochs) if use_scheduler else None
     
    diag_mask = torch.eye(model.n_neurons, device=device).bool()
    L1_weight_matrix = torch.ones(model.n_neurons, model.n_neurons, device=device)
    L1_weight_matrix[diag_mask] = 0.0
    
    firing_rates_tensor = torch.tensor(firing_rates, dtype=torch.float32, device=device) if firing_rates is not None else None
    
    train_losses, val_losses, learning_rates = [], [], []
    start_time = time.time()
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for Y_prev, Y_curr in train_loader:
            Y_prev, Y_curr = Y_prev.float().to(device), Y_curr.float().to(device)
            optimizer.zero_grad()
            spikes = model(Y_prev)
            loss = poisson_nll_loss(spikes, Y_curr, firing_rates_tensor)
            if L1_alpha > 0:
                loss += L1_alpha * torch.mean(torch.abs(model.A) * L1_weight_matrix)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Y_prev, Y_curr in val_loader:
                Y_prev, Y_curr = Y_prev.float().to(device), Y_curr.float().to(device)
                spikes = model(Y_prev)
                val_loss += poisson_nll_loss(spikes, Y_curr, firing_rates_tensor).item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        if scheduler:
            scheduler.step()
           
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: TrainLoss={train_losses[-1]:.4f}, ValLoss={val_losses[-1]:.4f}, LR={learning_rates[-1]:.1e}, Elapsed={(time.time() - start_time):.1f}s")
        
            
    return train_losses, val_losses, learning_rates



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
    parser.add_argument('-a',"--ampl_thres", type=float, default=0.07, help="minima amplitude of valid off-diagonal edge")

    args = parser.parse_args()

    print("Configuration:", vars(args))
    device = check_gpu_availability()
    
    # --- Modern data loading from fit_stageA.py ---
    spikesFF = os.path.join(args.dataPath, f"{args.dataName}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF)
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

    def make_loader(X, Yt, shuffle=True):
        return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Yt, dtype=torch.float32)), 
                         batch_size=args.batch_size, shuffle=shuffle, drop_last=shuffle)
    
    train_loader = make_loader(X_np[:n_train], Yt_np[:n_train])
    val_loader = make_loader(X_np[n_train:], Yt_np[n_train:], shuffle=False)
    
    print(f"Loaded T={T}, M={M}, using {n_pairs} pairs ({n_train} train, {n_pairs-n_train} val), step_size={step_size}")

    # --- Original training logic from fit_poissonV4.py ---
    model = PoissonGLModel(M)
    start_time = time.time()
    train_losses, val_losses, learning_rates = train_Poisson_model(
        model, device, train_loader, val_loader, args.n_epochs, lr=args.lr, L1_alpha=args.L1_alpha, firing_rates=dataRates, use_scheduler=True
    )
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds")

    # --- Modern output saving from fit_stageA.py ---
    lassoD = {
        'A_lasso': model.A.detach().cpu().numpy(), 
        'B_lasso': model.B.detach().cpu().numpy(),
        'train_losses': np.array(train_losses), 
        'val_losses': np.array(val_losses),
        'learning_rates': np.array(learning_rates),
        'firing_rates': dataRates
    }
    
    lassoMD = {
        'lassoFit_output_name': fit_core, 'lassoFit_input_name': args.dataName,
        'batch_size': args.batch_size, 'num_samples_used': n_pairs, 'n_epochs': args.n_epochs,
        'num_train_samples': n_train, 'num_val_samples': n_pairs-n_train,
        'learning_rate': args.lr, 'L1_alpha': args.L1_alpha, 'step_size': step_size,
        'training_time_sec': total_time, 'num_neurons': M,
    }

    # ... select edges in A_fit matrix ...
    lassoD['ampl_thres']= args.ampl_thres
    spikeMD['fit_lasso']=lassoMD
    maskF=select_eges_from_fitLasso(lassoD,args.ampl_thres)
    
    for xx in maskF:
        lassoD['mask.lasso.'+xx]=maskF[xx]
    fitMD['short_name']=fit_core
     
    fitFF = os.path.join(args.dataPath, f"{fit_core}.lasso.npz")
    write_data_npz(lassoD, fitFF, metaD=spikeMD)

    print('\n  ./eval_fit.py  --dataName %s ' % (fit_core))
    print('\n  ./fit_regress.py  --dataName %s   -p a  ' % (fit_core))

if __name__ == "__main__":
    main()
