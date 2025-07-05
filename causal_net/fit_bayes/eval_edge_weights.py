#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""
Evaluation and plotting functions for Dale model fitting results

This module contains all plotting and visualization functions for analyzing
the quality of fitted Dale models against ground truth connectivity matrices.

Key Functions:
- Weight correlation analysis (diagonal, excitatory, inhibitory)
- Residual distribution analysis
- Connectivity matrix visualization
- Training loss plotting
- Comprehensive results visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_training_loss(ax, losses, fit_time, args, num_samples_k):
    """Plot training loss over epochs with training info."""
    start_epoch_plot = 3  # do not show the first N epochs
    epochs_to_plot = np.arange(len(losses))
    ax.plot(epochs_to_plot[start_epoch_plot:], losses[start_epoch_plot:])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.grid(True)

    final_loss = losses[-1] if losses else float('nan')
    num_epochs = len(losses)
    avg_time_per_epoch = fit_time / num_epochs if num_epochs > 0 else 0
    info_text = (f'End Loss: {final_loss:.4f}\n'
                 f'LR start: {args.lr:.1e}, Patience: {args.patience}\n'
                 f'Batch: {args.batch_size}, Samples: {num_samples_k:.0f}k\n'
                 f'Fit time: {fit_time / 60:.1f} min\n'
                 f'Avg time/epoch: {avg_time_per_epoch:.2f}s')
    ax.text(0.95, 0.95, info_text, transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

def plot_weight_correlation(ax, true_weights, fitted_weights, title, color):
    """Plot correlation between true and fitted weights."""
    corr = np.corrcoef(true_weights, fitted_weights)[0, 1]
    ax.scatter(true_weights, fitted_weights, s=10, alpha=0.6, color=color)
    ax.set_title(f"{title}\n(N={len(true_weights)})")
    ax.text(0.1, 0.9, f"Corr: {corr:.3f}", transform=ax.transAxes)
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_aspect('equal', adjustable='box')
    return corr

def plot_weight_residuals(ax, residuals, title, color, rms_ax_range=0.10):
    """Plot histogram of weight residuals with statistics."""
    res_mean = np.mean(residuals)
    res_rmse = np.sqrt(np.mean(residuals**2))
    
    n, _, _ = ax.hist(residuals, bins=20 if 'Diagonal' in title else 50, color=color)
    ax.set_title(f"{title} Residuals")
    ax.text(0.1, 0.8, f"Mean: {res_mean:.3f}\nRMSE: {res_rmse:.3f}", transform=ax.transAxes)
    ax.axvline(0, color='lime', linestyle='--')
    ax.errorbar(res_mean, np.max(n) * 0.5, xerr=res_rmse, fmt='o', color='m', capsize=5)
    ax.set_xlim(-rms_ax_range, rms_ax_range)
    return res_mean, res_rmse

def plot_connectivity_matrix(ax, W_matrix, num_excite, num_inhibit, title):
    """Plot connectivity matrix with Dale's principle visualization."""
    num_neuron = W_matrix.shape[0]
    vmax = np.max(np.abs(W_matrix))
    im = ax.imshow(W_matrix, cmap='bwr', interpolation='nearest', vmin=-vmax, vmax=vmax)
    
    ax.set_title(title)
    ax.set_ylabel("presyn. node index, source")
    ax.set_xlabel("postsyn. node index, target")
    
    # Add separator line and annotations
    ax.axhline(y=num_excite - 0.5, color='k', linestyle='--')
    ax.text(num_neuron * 0.5, num_excite / 2, f'Excitatory ({num_excite})', color='red', ha='center', va='center')
    ax.text(num_neuron * 0.5, num_excite + num_inhibit / 2, f'Inhibitory ({num_inhibit})', color='blue', ha='center', va='center')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('coupling strength')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

def analyze_weight_categories(W_true, W_fitted, num_excite):
    """Analyze weights by category (diagonal, excitatory, inhibitory)."""
    num_neuron = W_true.shape[0]
    off_diag_mask = ~np.eye(num_neuron, dtype=bool)
    
    # 1. Diagonal elements
    diag_true = np.diag(W_true)
    diag_fitted = np.diag(W_fitted)
    diag_residuals = diag_fitted - diag_true
    
    # 2. Excitatory off-diagonal elements
    excite_mask_true = off_diag_mask[:num_excite, :] & (W_true[:num_excite, :] != 0)
    excite_true = W_true[:num_excite, :][excite_mask_true]
    excite_fitted = W_fitted[:num_excite, :][excite_mask_true]
    excite_residuals = excite_fitted - excite_true
    
    # 3. Inhibitory off-diagonal elements
    inhibit_mask_true = off_diag_mask[num_excite:, :] & (W_true[num_excite:, :] != 0)
    inhibit_true = W_true[num_excite:, :][inhibit_mask_true]
    inhibit_fitted = W_fitted[num_excite:, :][inhibit_mask_true]
    inhibit_residuals = inhibit_fitted - inhibit_true
    
    return {
        'diagonal': {'true': diag_true, 'fitted': diag_fitted, 'residuals': diag_residuals},
        'excitatory': {'true': excite_true, 'fitted': excite_fitted, 'residuals': excite_residuals},
        'inhibitory': {'true': inhibit_true, 'fitted': inhibit_fitted, 'residuals': inhibit_residuals}
    }

def create_evaluation_plot(W_true, W_fitted, losses, fit_time, args, md,num_samples_k, num_excite, num_inhibit):
    """Create comprehensive evaluation plot comparing fitted vs true weights."""
    
    # Create figure
    plt.figure(figsize=(16, 8))
    rms_ax_range = 0.10
    
    # Analyze weight categories
    weight_analysis = analyze_weight_categories(W_true, W_fitted, num_excite)
    
    # Row 1: Loss and correlation plots
    ax1 = plt.subplot(2, 4, 1)
    plot_training_loss(ax1, losses, fit_time, args, num_samples_k)
    
    ax2 = plt.subplot(2, 4, 2)
    diag_corr = plot_weight_correlation(ax2, weight_analysis['diagonal']['true'], 
                                      weight_analysis['diagonal']['fitted'], 
                                      "Diagonal Weights", 'green')
    
    ax3 = plt.subplot(2, 4, 3)
    excite_corr = plot_weight_correlation(ax3, weight_analysis['excitatory']['true'], 
                                        weight_analysis['excitatory']['fitted'], 
                                        "Excitatory Weights", 'salmon')
    
    ax4 = plt.subplot(2, 4, 4)
    inhibit_corr = plot_weight_correlation(ax4, weight_analysis['inhibitory']['true'], 
                                         weight_analysis['inhibitory']['fitted'], 
                                         "Inhibitory Weights", 'blue')
    
    # Row 2: Matrix visualization and residual plots
    ax5 = plt.subplot(2, 4, 5)
    plot_connectivity_matrix(ax5, W_fitted, num_excite, num_inhibit, "Fitted W-matrix")
    
    ax6 = plt.subplot(2, 4, 6)
    diag_res_mean, diag_res_rmse = plot_weight_residuals(ax6, weight_analysis['diagonal']['residuals'], 
                                                        "Diagonal", 'green', rms_ax_range)
    
    ax7 = plt.subplot(2, 4, 7)
    excite_res_mean, excite_res_rmse = plot_weight_residuals(ax7, weight_analysis['excitatory']['residuals'], 
                                                           "Excitatory", 'salmon', rms_ax_range)
    
    ax8 = plt.subplot(2, 4, 8)
    inhibit_res_mean, inhibit_res_rmse = plot_weight_residuals(ax8, weight_analysis['inhibitory']['residuals'], 
                                                             "Inhibitory", 'blue', rms_ax_range)

    shortN=md['short_name']
    # Print results summary
    print(f"\nFit results for {shortN}:")
    print(f"  Diagonal residuals RMS: {diag_res_rmse:.4f}")
    print(f"  Excitatory residuals RMS: {excite_res_rmse:.4f}")
    print(f"  Inhibitory residuals RMS: {inhibit_res_rmse:.4f}")
    
    # Final formatting and saving
    fig = plt.gcf()
    num_epochs = len(losses)
    fig.suptitle(f'Fit for {shortN}, trained on {num_samples_k:.0f}k samples for {num_epochs} epochs, took {fit_time:.1f} sec', fontsize=16)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.9, hspace=0.4, wspace=0.3)
    
    return {
        'correlations': {'diagonal': diag_corr, 'excitatory': excite_corr, 'inhibitory': inhibit_corr},
        'rmse': {'diagonal': diag_res_rmse, 'excitatory': excite_res_rmse, 'inhibitory': inhibit_res_rmse}
    }

def save_and_show_plot(args, show_plot=True):
    """Save the current plot and optionally show it."""
    out_path = os.path.join("model", "%s_results.png" % 'aa1')
    plt.savefig(out_path)
    print("Saved plot to %s" % out_path)
    if show_plot:
        plt.show() 
