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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import os
from torch.utils.data import DataLoader, TensorDataset
from neural_net_model import SparseNetworkModel

def fit_model(args):
    print("fit_model START, args:", args)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print("CUDA is available. Using GPU: %s" % gpu_name)
    else:
        print("CUDA not available. Using CPU.")

    input_path = os.path.join("data", args.input + ".npz")
    data = np.load(input_path)
    W_true = data["W"]
    E = data["E"]
    
    print("E-matrix (M=%d):" % E.shape[0])
    with np.printoptions(linewidth=400):
        print(E)
        
    tau = data["tau"]
    trajectory = torch.tensor(data["trajectory"], dtype=torch.float32)

    # Prepare data for PyTorch
    X = trajectory[:, :-1]
    y = trajectory[:, 1:]
    dataset = TensorDataset(X.T, y.T)  # Transpose for (samples, features)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    # Initialize W_model (same as before)
    num_nonzero = np.count_nonzero(E)
    nonzero_indices = torch.tensor(np.array(E.nonzero()))
    initial_weights = torch.rand(num_nonzero) * 2 - 1
    W_model = torch.sparse_coo_tensor(nonzero_indices, initial_weights, E.shape, dtype=torch.float32)
    W_model.requires_grad = True

    model = SparseNetworkModel(W_model, tau)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=8)

    losses = []
    best_loss = float('inf')
    epochs_no_improve = 0
    lr_reductions = 0
    
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, targets = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)
        print("Epoch %d, Loss: %.4f" % (epoch + 1, epoch_loss))
        
        if lr_reductions < args.max_lr_reductions:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(epoch_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print("Epoch %d: reducing learning rate to %.1e" % (epoch + 1, new_lr))
                lr_reductions += 1
        else:
            if epoch == 0 or (epoch + 1) % 10 == 0:  # Print only on first epoch and every 10 epochs
                 print("Epoch %d: Max LR reductions reached, not reducing further." % (epoch + 1))
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    #Save the model
    model_path = os.path.join("model", "%s_model.pth" % args.input)
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # Evaluate and plot
    W_fitted = model.W_model.to_dense().detach().numpy()
    plt.figure(figsize=(12, 8))
    
    # Plot Loss
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(losses)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("Training Loss for %s" % args.input)
    info_text = "Initial LR: %.1e\nBatch size: %d\nSamples: %d" % (args.lr, args.batch, len(dataset))
    ax1.text(0.5, 0.8, info_text, transform=ax1.transAxes)

    # Diagonal elements analysis
    diag_true = np.diag(W_true)
    diag_fitted = np.diag(W_fitted)
    diag_corr = np.corrcoef(diag_true, diag_fitted)[0, 1]
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(diag_true, diag_fitted)
    ax2.set_xlabel("True Diagonal")
    ax2.set_ylabel("Fitted Diagonal")
    ax2.set_title("Diagonal Correlation")
    ax2.text(0.1, 0.9, "Corr: %.3f" % diag_corr, transform=ax2.transAxes)
    lims = [np.min([ax2.get_xlim(), ax2.get_ylim()]), np.max([ax2.get_xlim(), ax2.get_ylim()])]
    ax2.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)

    # Diagonal residuals histogram
    diag_residuals = diag_true - diag_fitted
    diag_res_mean = np.mean(diag_residuals)
    diag_res_rmse = np.sqrt(np.mean(diag_residuals**2))
    
    ax5 = plt.subplot(2, 3, 5)
    n, bins, _ = ax5.hist(diag_residuals, bins=20)
    ax5.set_xlabel("Residuals (True - Fitted)")
    ax5.set_ylabel("Count")
    ax5.set_title("Diagonal Residuals")
    ax5.text(0.1, 0.8, "Mean: %.3f\nRMSE: %.3f" % (diag_res_mean, diag_res_rmse), transform=ax5.transAxes)
    y_pos = np.max(n) / 2
    ax5.errorbar(diag_res_mean, y_pos, xerr=diag_res_rmse, fmt='o', color='r', capsize=5)

    # Off-diagonal elements analysis
    off_diag_mask = E.astype(bool) & ~np.eye(E.shape[0], dtype=bool)
    off_diag_true = W_true[off_diag_mask]
    off_diag_fitted = W_fitted[off_diag_mask]
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title("Off-diagonal Correlation")
    if off_diag_true.size > 1:
        off_diag_corr = np.corrcoef(off_diag_true, off_diag_fitted)[0, 1]
        ax3.scatter(off_diag_true, off_diag_fitted)
        ax3.text(0.1, 0.9, "Corr: %.3f" % off_diag_corr, transform=ax3.transAxes)
        lims = [np.min([ax3.get_xlim(), ax3.get_ylim()]), np.max([ax3.get_xlim(), ax3.get_ylim()])]
        ax3.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        ax3.set_xlim(lims)
        ax3.set_ylim(lims)
    else:
        ax3.text(0.1, 0.5, "Not enough data for plot", transform=ax3.transAxes)

    ax3.set_xlabel("True Non-zero Off-diagonal")
    ax3.set_ylabel("Fitted Non-zero Off-diagonal")
    

    # Off-diagonal residuals histogram
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_title("Off-diagonal Residuals")
    if off_diag_true.size > 1:
        off_diag_residuals = off_diag_true - off_diag_fitted
        off_diag_res_mean = np.mean(off_diag_residuals)
        off_diag_res_rmse = np.sqrt(np.mean(off_diag_residuals**2))
        n, bins, _ = ax6.hist(off_diag_residuals, bins=50)
        ax6.text(0.1, 0.8, "Mean: %.3f\nRMSE: %.3f" % (off_diag_res_mean, off_diag_res_rmse), transform=ax6.transAxes)
        y_pos = np.max(n) / 2
        ax6.errorbar(off_diag_res_mean, y_pos, xerr=off_diag_res_rmse, fmt='o', color='r', capsize=5)
    else:
        ax6.text(0.1, 0.5, "Not enough data for plot", transform=ax6.transAxes)

    ax6.set_xlabel("Residuals (True - Fitted)")
    ax6.set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join("model", "%s_results.png" % args.input))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input data file base name")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=25, help="Patience for early stopping (should be > LR patience)")
    parser.add_argument("--max_lr_reductions", type=int, default=4, help="Maximum number of LR reductions")
    args = parser.parse_args()
    fit_model(args)

