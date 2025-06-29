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
    - Generates a detailed 2x4 panel plot to analyze fit quality, showing:
        - Training loss over epochs.
        - The final fitted W-matrix, visualized with its Dale's structure.
        - Correlation and residual plots for three separate categories:
          diagonal, excitatory off-diagonal, and inhibitory off-diagonal weights.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib as mpl
import argparse
import os
import time
from torch.utils.data import DataLoader, TensorDataset
from neural_daleNet_model import SparseNetworkModel

def fit_model_and_plot(args):
    print("fit_model START, args:", args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print("CUDA is available. Using GPU: %s" % gpu_name)
    else:
        print("CUDA not available. Using CPU.")

    input_path = os.path.join("data", args.input + ".npz")
    data = np.load(input_path)
    W_true = data["W"]
    E = data["E"]
    w_dims = data["w_dims"]
    num_neuron, num_excite, num_inhibit = w_dims[0], w_dims[1], w_dims[2]
    
    print(f"Loaded data with {num_neuron} neurons ({num_excite} excitatory, {num_inhibit} inhibitory)")
        
    tau = data["tau"]
    trajectory = torch.tensor(data["trajectory"], dtype=torch.float32)

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
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)

        # Log progress
        if epoch < 5 or (epoch + 1) % 20 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / (epoch + 1)
            print(f'Epoch {epoch + 1:3d}, Loss: {epoch_loss:.4f}, '
                  f'Elapsed: {elapsed_time/60:.1f} min, Avg time/epoch: {avg_time_per_epoch:.1f}s')

        # Reduce LR on plateau
        scheduler.step(epoch_loss)
        old_lr = optimizer.param_groups[0]['lr']
        new_lr = scheduler.get_last_lr()[0]

        if new_lr < old_lr:
            if lr_reductions < args.max_lr_reductions:
                lr_reductions += 1
                print(f'Epoch {epoch + 1}: reducing learning rate to {new_lr:.1e}')
                # The scheduler already updated the optimizer's LR
            else:
                print(f"Epoch {epoch + 1}: Max LR reductions reached. Stopping training.")
                # Restore old LR since we are not applying this reduction and stopping
                for param_group in optimizer.param_groups:
                    param_group['lr'] = old_lr
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
    model_path = os.path.join("model", "%s_model.pth" % args.input)
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # Evaluate and plot
    import matplotlib.pyplot as plt
    W_fitted = model.get_w().cpu().detach().numpy()
    plt.figure(figsize=(16, 8))
    rmsAxRng=0.10

    # Plot Loss
    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(losses)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.grid(True)

    final_loss = losses[-1] if losses else float('nan')
    num_epochs = len(losses)
    avg_time_per_epoch = fit_time / num_epochs if num_epochs > 0 else 0
    info_text = (f'End Loss: {final_loss:.4f}\n'
                 f'LR start: {args.lr:.1e}, Patience: {args.patience}\n'
                 f'Batch: {args.batch_size}, Samples: {len(dataset)}\n'
                 f'Fit time: {fit_time / 60:.1f} min\n'
                 f'Avg time/epoch: {avg_time_per_epoch:.2f}s')
    ax1.text(0.95, 0.95, info_text, transform=ax1.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

    # --- Analysis plots ---
    off_diag_mask = ~np.eye(num_neuron, dtype=bool)

    # 1. Diagonal elements
    diag_true = np.diag(W_true)
    diag_fitted = np.diag(W_fitted)
    diag_corr = np.corrcoef(diag_true, diag_fitted)[0, 1]
    diag_residuals = diag_fitted - diag_true
    diag_res_mean = np.mean(diag_residuals)
    diag_res_rmse = np.sqrt(np.mean(diag_residuals**2))

    ax2 = plt.subplot(2, 4, 2)
    ax2.scatter(diag_true, diag_fitted, s=10, alpha=0.6, color='green')
    ax2.set_title(f"Diagonal Weights\n(N={len(diag_true)})")
    ax2.text(0.1, 0.9, f"Corr: {diag_corr:.3f}", transform=ax2.transAxes)
    lims = [np.min([ax2.get_xlim(), ax2.get_ylim()]), np.max([ax2.get_xlim(), ax2.get_ylim()])]
    ax2.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax2.set_aspect('equal', adjustable='box')
    
    # 2. Excitatory off-diagonal elements
    excite_mask_true = off_diag_mask[:num_excite, :] & (W_true[:num_excite, :] != 0)
    excite_true = W_true[:num_excite, :][excite_mask_true]
    excite_fitted = W_fitted[:num_excite, :][excite_mask_true]
    excite_corr = np.corrcoef(excite_true, excite_fitted)[0, 1]
    excite_residuals = excite_fitted - excite_true
    excite_res_mean = np.mean(excite_residuals)
    excite_res_rmse = np.sqrt(np.mean(excite_residuals**2))

    ax3 = plt.subplot(2, 4, 3)
    ax3.scatter(excite_true, excite_fitted, s=10, alpha=0.6, color='salmon')
    ax3.set_title(f"Excitatory Weights\n(N={len(excite_true)})")
    ax3.text(0.1, 0.9, f"Corr: {excite_corr:.3f}", transform=ax3.transAxes)
    lims = [np.min([ax3.get_xlim(), ax3.get_ylim()]), np.max([ax3.get_xlim(), ax3.get_ylim()])]
    ax3.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax3.set_aspect('equal', adjustable='box')

    # 3. Inhibitory off-diagonal elements
    inhibit_mask_true = off_diag_mask[num_excite:, :] & (W_true[num_excite:, :] != 0)
    inhibit_true = W_true[num_excite:, :][inhibit_mask_true]
    inhibit_fitted = W_fitted[num_excite:, :][inhibit_mask_true]
    inhibit_corr = np.corrcoef(inhibit_true, inhibit_fitted)[0, 1]
    inhibit_residuals = inhibit_fitted - inhibit_true
    inhibit_res_mean = np.mean(inhibit_residuals)
    inhibit_res_rmse = np.sqrt(np.mean(inhibit_residuals**2))

    ax4 = plt.subplot(2, 4, 4)
    ax4.scatter(inhibit_true, inhibit_fitted, s=10, alpha=0.6, color='blue')
    ax4.set_title(f"Inhibitory Weights\n(N={len(inhibit_true)})")
    ax4.text(0.1, 0.9, f"Corr: {inhibit_corr:.3f}", transform=ax4.transAxes)
    lims = [np.min([ax4.get_xlim(), ax4.get_ylim()]), np.max([ax4.get_xlim(), ax4.get_ylim()])]
    ax4.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax4.set_aspect('equal', adjustable='box')

    # --- ROW 2 ---
    # 4. Fitted W-matrix plot (Dale's principle visualization)
    ax5 = plt.subplot(2, 4, 5)
    W_plot = W_fitted # Do not transpose, excitatory are rows
    vmax = np.max(np.abs(W_plot))
    im = ax5.imshow(W_plot, cmap='bwr', interpolation='nearest', vmin=-vmax, vmax=vmax)
    
    ax5.set_title("Fitted W-matrix")
    ax5.set_ylabel("presyn. node index, source")
    ax5.set_xlabel("postsyn. node index, target")
    
    # Add separator line and annotations
    ax5.axhline(y=num_excite - 0.5, color='k', linestyle='--')
    ax5.text(num_neuron * 0.5, num_excite / 2, 'Excitatory', color='red', ha='center', va='center')
    ax5.text(num_neuron * 0.5, num_excite + num_inhibit / 2, 'Inhibitory', color='blue', ha='center', va='center')
    
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('coupling strength')
    ax5.set_aspect('equal', adjustable='box')
    ax5.grid(True)

    ax6 = plt.subplot(2, 4, 6)
    ax6.hist(diag_residuals, bins=20, color='green')
    ax6.set_title("Diagonal Residuals")
    ax6.text(0.1, 0.8, f"Mean: {diag_res_mean:.3f}\nRMSE: {diag_res_rmse:.3f}", transform=ax6.transAxes)
    ax6.axvline(0, color='lime', linestyle='--')
    ax6.set_xlim(-rmsAxRng, rmsAxRng)

    ax7 = plt.subplot(2, 4, 7)
    ax7.hist(excite_residuals, bins=50, color='salmon')
    ax7.set_title("Excitatory Residuals")
    ax7.text(0.1, 0.8, f"Mean: {excite_res_mean:.3f}\nRMSE: {excite_res_rmse:.3f}", transform=ax7.transAxes)
    ax7.axvline(0, color='lime', linestyle='--')
    ax7.set_xlim(-rmsAxRng, rmsAxRng)

    ax8 = plt.subplot(2, 4, 8)
    ax8.hist(inhibit_residuals, bins=50, color='blue')
    ax8.set_title("Inhibitory Residuals")
    ax8.text(0.1, 0.8, f"Mean: {inhibit_res_mean:.3f}\nRMSE: {inhibit_res_rmse:.3f}", transform=ax8.transAxes)
    ax8.axvline(0, color='lime', linestyle='--')
    ax8.set_xlim(-rmsAxRng, rmsAxRng)

    print(f"\nFit results for {args.input}:")
    print(f"  Diagonal residuals RMS: {diag_res_rmse:.4f}")
    print(f"  Excitatory residuals RMS: {excite_res_rmse:.4f}")
    print(f"  Inhibitory residuals RMS: {inhibit_res_rmse:.4f}")
      
    fig = plt.gcf()
    numKsamples = len(dataset)/1000
    fig.suptitle(f'Fit for {args.input}, trained on {numKsamples:.0f}k samples for {len(losses)} epochs, took {fit_time:.1f} sec', fontsize=16)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.9, hspace=0.4, wspace=0.3)
    out_path = os.path.join("model", "%s_results.png" % args.input)
    plt.savefig(out_path)
    print("Saved plot to %s" % out_path)
    if not args.noXterm:
        plt.show()

    print(f'Finished Training in {fit_time:.2f} seconds')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-X',"--noXterm", action='store_true', default=False, help="Disable X-server for plotting")
    parser.add_argument("--verb", type=int, default=1, help="Verbosity level")
    parser.add_argument("--input", type=str, required=True, help="Input data file base name")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=25, help="Patience for early stopping (should be > LR patience)")
    parser.add_argument("--max_lr_reductions", type=int, default=4, help="Maximum number of LR reductions")
    args = parser.parse_args()
    
    if args.noXterm:
        if args.verb > 0: print('disable Xterm')
        mpl.use('Agg')
    else:
        mpl.use('TkAgg')

    fit_model_and_plot(args)

