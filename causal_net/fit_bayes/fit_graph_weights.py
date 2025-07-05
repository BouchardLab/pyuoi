#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

'''
bash -c "source /usr/share/lmod/lmod/init/bash && module load pytorch && python3 debug_dale_law.py

 salloc -q interactive -C gpu  -t 4:00:00 -A m2043 -N 1
 module load pytorch 
 
 # Set to use only GPU 0 explicitly
CUDA_VISIBLE_DEVICES=0 ./fit_model.py --simuName daleM40may27_simu-lya37b --epochs 200 --learning_rate 0.001 --batch_size 512

'''

"""
fit_dale_model.py: Learn system parameters from simulated neural data
                   using a model that respects Dale's Principle.

This script learns the parameters of a dynamic system by training a model
that enforces Dale's Principle (excitatory/inhibitory neurons) at every step.
Instead of learning the weight matrix `W` directly, it learns a precursor
matrix `V` and computes `W` through a fixed transformation.

Key Operations:
1.  Data Loading:
    - Loads a .npz file, extracting the trajectory, true `W_true`, and
      the `w_dims` array which contains [num_neuron, num_excite, num_inhibit].

2.  Model Architecture (V -> W):
    - Initializes a dense learnable precursor matrix `V`.
    - The model's forward pass computes the effective `W` matrix as follows:
        1. V2 = V * V (element-wise square, making values non-negative)
        2. Diagonal weights: W(i, i) = -V2(i, i)
        3. Excitatory rows:  W(i, j) = +V2(i, j)
        4. Inhibitory rows: W(i, j) = -V2(i, j)
    - This ensures `W` always adheres to Dale's Principle.

3.  Training Loop:
    - Uses Mean Squared Error (MSE) loss between model predictions and targets.
    - Employs the Adam optimizer, learning rate scheduling, and early stopping.

4.  Evaluation and Plotting:
    - Saves the trained model state to a .pth file.
    - Uses eval_edge_weights module to generate comprehensive evaluation plots
      including training loss, weight correlations, residual analysis, and
      connectivity matrix visualization with Dale's principle structure.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib as mpl
import argparse
import os,hashlib
import time
from torch.utils.data import DataLoader, TensorDataset
from SparseDaleModel import SparseNetworkModel
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from pprint import pprint

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-X',"--noXterm", action='store_true', default=False, help="Disable X-server for plotting")

   
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=25, help="Patience for early stopping (should be > LR patience)")
    parser.add_argument("--max_lr_reductions", type=int, default=4, help="Maximum number of LR reductions")

    parser.add_argument('-v',"--verb", type=int, default=1, help="Verbosity level")

    parser.add_argument("--fitName",  default=None,help='fit name')

    parser.add_argument('--time_range' , default=[50, 40_000],  nargs=2,   type=int, help='fit data time range')

    parser.add_argument("--basePath",default='out',help="head dir for any results")
    parser.add_argument("--inpName",  required=True,help='name of input data')

    args = parser.parse_args()
    
    if args.noXterm:
        if args.verb > 0: print('disable Xterm')
        mpl.use('Agg')
    else:
        mpl.use('TkAgg')
    #args.dataPath=os.path.join(args.basePath,'input_uoi')    
    args.modelPath=os.path.join(args.basePath,'model_bayes')
   
    print( 'myArg-program:',parser.prog)
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    #assert os.path.exists(args.dataPath)
    assert os.path.exists(args.modelPath)
    if args.time_range!=None: assert args.time_range[0] < args.time_range[1] 
    
    return args

#...!...!....................
def buildWeightMeta(args,md):
    fim={};  md['fit_weight']=fim
    
    fim['hash']=hashlib.md5(os.urandom(32)).hexdigest()[:6]
    if args.fitName==None:
        md['short_name']='fitw-%s'%(fim['hash'])
    else:
        md['short_name']=args.fitName
    pprint(fim); print( flush=True)
   
def fit_model_and_plot(args,bigD,md):
    dmm=md['dale_truth']
    pmd=md['payload']
    
    W_true =bigD['true_network_matrix']
    E = bigD['bayes_edge_matrix']
    featData=bigD['all_features']
    
    #w_dims = data["w_dims"]
    num_neuron=dmm['num_any_neur']
    num_excite=dmm['num_excit_neur']
    num_inhibit = num_neuron - num_excite
    tau = pmd['tau_response']
    
    print(f"Loaded data with {num_neuron} neurons ({num_excite} excitatory, {num_inhibit} inhibitory)")
        
    
    trajectory = torch.tensor(featData.T, dtype=torch.float32)

    # Prepare data for PyTorch
    X = trajectory[:, :-1]
    y = trajectory[:, 1:]
    dataset = TensorDataset(X.T, y.T)  # Transpose for (samples, features)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize V_model as a dense tensor
    mean, std = -1.0, 0.3
    initial_V = torch.randn(E.shape) * std + mean
    torch.clamp_(initial_V, min=mean - 3*std, max=mean + 3*std)
    
    # E is the connectivity mask, also needs to be a tensor
    E_mask_tensor = torch.tensor(E, dtype=torch.float32)

    model = SparseNetworkModel(initial_V, E_mask_tensor, tau, num_neuron, num_excite)
    model.to(device)
    criterion = nn.MSELoss()
    # Using Adam since the model parameters are now dense
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=8)

    losses = []
    best_loss = float('inf')
    epochs_no_improve = 0
    lr_reductions = 0
    
    start_time = time.time()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)

        # Log progress
        if epoch < 5 or (epoch + 1) % 20 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            print(f'Epoch {epoch + 1:3d}, Loss: {epoch_loss:.4f}, loss-1: {epoch_loss - 1.:.2e}, '
                  f'Elapsed: {elapsed_time/60:.1f} min, Avg time/epoch: {avg_time_per_epoch:.1f}s')

        # Reduce LR on plateau
        scheduler.step(epoch_loss)
        old_lr = optimizer.param_groups[0]['lr']
        new_lr = scheduler.get_last_lr()[0]

        if new_lr < old_lr:
            print(f'Epoch {epoch + 1}: reducing learning rate to {new_lr:.1e}')
            lr_reductions += 1
            if lr_reductions >= args.max_lr_reductions:
                print(f"Epoch {epoch + 1}: Max LR reductions reached. Stopping training.")
                # The scheduler already updated the optimizer, so no need to restore old LR
                break

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= args.patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    fit_time = time.time() - start_time

    #Save the model
    model_path = os.path.join("model", "%s_model.pth" % md['short_name'])
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print('FiM: saved model:',model_path)

    # Evaluate and plot
    from eval_edge_weights import create_evaluation_plot, save_and_show_plot
    W_fitted = model.get_w().cpu().detach().numpy()
    num_samples_k = len(dataset) / 1000
    
    # Create comprehensive evaluation plot
    evaluation_results = create_evaluation_plot(
        W_true, W_fitted, losses, fit_time, args, md,
        num_samples_k, num_excite, num_inhibit
    )
    
    # Save and show plot
    save_and_show_plot(args, show_plot=not args.noXterm)

    print(f'Finished Training in {fit_time:.2f} seconds')

if __name__ == "__main__":

    args=setup_args()

    print("fit_model START, args:", args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print("CUDA is available. Using GPU: %s" % gpu_name)
    else:
        print("CUDA not available. Using CPU.")

    inpF=args.inpName+'.fitBayes.h5'
    expD,expMD=read4_data_hdf5(os.path.join(args.modelPath,inpF))
    if args.verb>=2:
        print('M:expMD:');  pprint(expMD)
        stop2
    
    buildWeightMeta(args,expMD)
    
    fit_model_and_plot(args,expD,expMD)

