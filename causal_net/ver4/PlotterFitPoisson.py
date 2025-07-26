__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from toolbox.PlotterBackbone import PlotterBackbone
from matplotlib import cm as cmap
import matplotlib.ticker as ticker
from pprint import pprint
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

#...!...!..................
def xplot_structure_matrix(fig,ax,W):
    a=0
    
#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

#...!...!..................
    def structure_results(self, A_true, mask_detected, train_losses, val_losses, md, figId=1, edge_analysis=None):
        """Plot structure identification results."""
        figId=self.smart_append(figId)        
        nrow,ncol=2,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,7))

        # Number of excitatory neurons must be provided
        num_excite = md['num_excit_neur']
        
        # True connectivity structure with weight magnitudes
        ax = self.plt.subplot(nrow,ncol,1)
        im1 = ax.imshow(A_true, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        title = 'True Weights (%s)' % md["short_name"]
        ax.set_title(title)
        ax.set_ylabel('From neuron')
        ax.set_xlabel('To neuron')
        fig.colorbar(im1, ax=ax)
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line separating excitatory from inhibitory neurons
        if num_excite > 0 and num_excite < A_true.shape[0]:
            ax.axhline(y=num_excite-0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.text(A_true.shape[1]-1, num_excite-1, 'Excitatory', ha='right', va='bottom', 
                   fontsize=10, fontweight='bold', color='red')
            ax.text(A_true.shape[1]-1, num_excite+1, 'Inhibitory', ha='right', va='top', 
                   fontsize=10, fontweight='bold', color='blue')
        
        # Detected structure
        ax = self.plt.subplot(nrow,ncol,3)
        im2 = ax.imshow(mask_detected, cmap='binary')
        ax.set_title('Fitted binary map, nSamp:%dk' % (md['num_samples']/1000))
        ax.set_ylabel('From neuron')
        ax.set_xlabel('To neuron')
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line separating excitatory from inhibitory neurons
        if num_excite > 0 and num_excite < A_true.shape[0]:
            ax.axhline(y=num_excite-0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.text(A_true.shape[1]-1, num_excite-1, 'Excitatory', ha='right', va='bottom', 
                   fontsize=10, fontweight='bold', color='red')
            ax.text(A_true.shape[1]-1, num_excite+1, 'Inhibitory', ha='right', va='top', 
                   fontsize=10, fontweight='bold', color='blue')
        
        # Training curves
        ax = self.plt.subplot(nrow,ncol,2)
        ax.plot(train_losses, label='Train', color='blue', linestyle='-')
        ax.plot(val_losses, label='Val', color='blue', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        
        # Create title with number of samples if available
        title = 'Training curves'
        if 'num_samples' in md:
            title += f' (N={md["num_samples"]})'
        ax.set_title(title)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Use edge_analysis results if provided, otherwise calculate metrics
        if edge_analysis is not None:
            # Use pre-calculated metrics from analyze_edge_detection
            overall_metrics = edge_analysis['overall']
            exc_metrics = edge_analysis['excitatory']
            inh_metrics = edge_analysis['inhibitory']
            
            tp = overall_metrics['tp']
            fp = overall_metrics['fp']
            fn = overall_metrics['fn']
            precision = overall_metrics['precision']
            recall = overall_metrics['recall']
            f1_score = overall_metrics['f1_score']
            
            exc_tp = exc_metrics['tp']
            exc_fp = exc_metrics['fp']
            exc_fn = exc_metrics['fn']
            exc_precision = exc_metrics['precision']
            exc_recall = exc_metrics['recall']
            exc_f1 = exc_metrics['f1_score']
            
            inh_tp = inh_metrics['tp']
            inh_fp = inh_metrics['fp']
            inh_fn = inh_metrics['fn']
            inh_precision = inh_metrics['precision']
            inh_recall = inh_metrics['recall']
            inh_f1 = inh_metrics['f1_score']
        else:
            # Fallback: calculate metrics manually (off-diagonal only)
            true_edges = np.abs(A_true) > 1e-10
            diag_mask = np.eye(true_edges.shape[0], dtype=bool)
            true_edges[diag_mask] = False  # Exclude diagonal
            mask_detected_off_diag = mask_detected.copy()
            mask_detected_off_diag[diag_mask] = False  # Exclude diagonal
            
            tp = np.sum(true_edges & mask_detected_off_diag)
            fp = np.sum(~true_edges & mask_detected_off_diag)
            tn = np.sum(~true_edges & ~mask_detected_off_diag)
            fn = np.sum(true_edges & ~mask_detected_off_diag)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate metrics for excitatory and inhibitory rows separately
            exc_mask = np.zeros_like(true_edges)
            exc_mask[:num_excite, :] = True
            exc_mask[diag_mask] = False
            
            inh_mask = np.zeros_like(true_edges)
            inh_mask[num_excite:, :] = True
            inh_mask[diag_mask] = False
            
            # Excitatory metrics
            exc_true = true_edges & exc_mask
            exc_detected = mask_detected_off_diag & exc_mask
            exc_tp = np.sum(exc_true & exc_detected)
            exc_fp = np.sum(~exc_true & exc_detected)
            exc_fn = np.sum(exc_true & ~exc_detected)
            exc_precision = exc_tp / (exc_tp + exc_fp) if (exc_tp + exc_fp) > 0 else 0
            exc_recall = exc_tp / (exc_tp + exc_fn) if (exc_tp + exc_fn) > 0 else 0
            exc_f1 = 2 * exc_precision * exc_recall / (exc_precision + exc_recall) if (exc_precision + exc_recall) > 0 else 0
            
            # Inhibitory metrics
            inh_true = true_edges & inh_mask
            inh_detected = mask_detected_off_diag & inh_mask
            inh_tp = np.sum(inh_true & inh_detected)
            inh_fp = np.sum(~inh_true & inh_detected)
            inh_fn = np.sum(inh_true & ~inh_detected)
            inh_precision = inh_tp / (inh_tp + inh_fp) if (inh_tp + inh_fp) > 0 else 0
            inh_recall = inh_tp / (inh_tp + inh_fn) if (inh_tp + inh_fn) > 0 else 0
            inh_f1 = 2 * inh_precision * inh_recall / (inh_precision + inh_recall) if (inh_precision + inh_recall) > 0 else 0
        
        # Display metrics
        ax = self.plt.subplot(nrow,ncol,4)
        x0, y0 = 0.05, 0.95  # Absolute position of the metrics block
        
        ax.text(x0, y0, 'Overall (off-diagonal):', fontsize=11, fontweight='bold')
        ax.text(x0, y0-0.07, f'TP: {tp}, FP: {fp}, FN: {fn}', fontsize=10)
        ax.text(x0, y0-0.13, f'Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}', fontsize=10)
        
        ax.text(x0, y0-0.25, 'Excitatory rows:', fontsize=11, fontweight='bold')
        ax.text(x0, y0-0.32, f'TP: {exc_tp}, FP: {exc_fp}, FN: {exc_fn}', fontsize=10)
        ax.text(x0, y0-0.38, f'Precision: {exc_precision:.3f}, Recall: {exc_recall:.3f}, F1: {exc_f1:.3f}', fontsize=10)
        
        ax.text(x0, y0-0.50, 'Inhibitory rows:', fontsize=11, fontweight='bold')
        ax.text(x0, y0-0.57, f'TP: {inh_tp}, FP: {inh_fp}, FN: {inh_fn}', fontsize=10)
        ax.text(x0, y0-0.63, f'Precision: {inh_precision:.3f}, Recall: {inh_recall:.3f}, F1: {inh_f1:.3f}', fontsize=10)
        
        # Add loss information
        y0-=0.03
        if len(train_losses) > 0 and len(val_losses) > 0:
            final_train_loss = train_losses[-1]
            final_val_loss = val_losses[-1]
            
            # Get validation loss from 10 epochs before the last (if available)
            if len(val_losses) > 10:
                val_loss_10_epochs_ago = val_losses[-11]  # -11 because indexing starts at 0
                ax.text(x0, y0-0.70, 'Final Losses:', fontsize=11, fontweight='bold')
                ax.text(x0, y0-0.77, f'Train: {final_train_loss:.4f}, Val: {final_val_loss:.4f}', fontsize=10)
                ax.text(x0, y0-0.83, f'Val (epoch -10): {val_loss_10_epochs_ago:.4f}', fontsize=10)
            else:
                ax.text(x0, y0-0.70, 'Final Losses:', fontsize=11, fontweight='bold')
                ax.text(x0, y0-0.77, f'Train: {final_train_loss:.4f}, Val: {final_val_loss:.4f}', fontsize=10)
        
        # Add training information
        if 'initial_lr' in md and 'train_time_min' in md and 'num_epochs' in md:
            ax.text(x0, y0-0.90, 'Training Info:', fontsize=11, fontweight='bold')
            ax.text(x0, y0-0.97, f'Initial LR: {md["initial_lr"]:.1e}', fontsize=10)
            ax.text(x0, y0-1.03, f'Train time: {md["train_time_min"]:.1f} min, Epochs: {md["num_epochs"]}', fontsize=10)
        else:
            ax.text(x0, y0-0.90, 'Note: Diagonal elements excluded', fontsize=9, style='italic', color='gray')
        
        ax.set_title('Structure Detection (%s)' % md["short_name"])
        ax.axis('off')

# Rate plotting moved to PlotterSimPoisson.py

#...!...!..................
    def weight_distributions(self, A_true, mask_detected, md, figId=3):
        """Plot histograms of true weight distributions for different categories."""
        figId=self.smart_append(figId)        
        nrow,ncol=2,3
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,6))

        # Number of excitatory neurons must be provided
        num_excite = md['num_excit_neur']
        
        # Create masks for different categories
        diag_mask = np.eye(A_true.shape[0], dtype=bool)
        
        # Excitatory and inhibitory masks (rows)
        exc_mask = np.zeros_like(A_true, dtype=bool)
        exc_mask[:num_excite, :] = True
        inh_mask = np.zeros_like(A_true, dtype=bool)
        inh_mask[num_excite:, :] = True
        
        # True edges (non-zero weights)
        true_edges = np.abs(A_true) > 1e-10
        
        # Categories for histograms
        categories = {
            'excitatory_tp': A_true[true_edges & exc_mask & ~diag_mask & mask_detected],
            'excitatory_fn': A_true[true_edges & exc_mask & ~diag_mask & ~mask_detected],
            'diagonal': A_true[diag_mask],
            'inhibitory_tp': A_true[true_edges & inh_mask & ~diag_mask & mask_detected],
            'inhibitory_fn': A_true[true_edges & inh_mask & ~diag_mask & ~mask_detected]
        }
        
        # Calculate common x-axis limits for non-diagonal histograms
        off_diag_weights = A_true[~diag_mask]
        if len(off_diag_weights) > 0:
            global_min = np.min(off_diag_weights)
            global_max = np.max(off_diag_weights)
            # Add some padding to the range
            range_padding = (global_max - global_min) * 0.05
            x_min = global_min - range_padding
            x_max = global_max + range_padding
        else:
            x_min, x_max = -1, 1  # Default range if no off-diagonal weights
        
        # Plot histograms
        plot_config = {
            'diagonal': ('Diagonal Weights', 'blue'),
            'excitatory_tp': ('Excitatory TP', 'green'), 
            'excitatory_fn': ('Excitatory FN', 'red'),
            'inhibitory_tp': ('Inhibitory TP', 'purple'),
            'inhibitory_fn': ('Inhibitory FN', 'orange')
        }
        
        for i, (category, weights) in enumerate(categories.items()):
            row = i // 3
            col = i % 3
            
            ax = self.plt.subplot(nrow,ncol,row*3 + col + 1)
            if len(weights) > 0:
                title, color = plot_config[category]
                ax.hist(weights, bins=30, alpha=0.7, color=color, edgecolor='black')
                ax.set_title(f'{title} (n={len(weights)})')
                ax.set_xlabel('Weight Value')
                ax.set_ylabel('Count')
                ax.grid(True, alpha=0.3)
                
                # Set common x-axis limits only for non-diagonal histograms
                if category != 'diagonal':
                    ax.set_xlim(x_min, x_max)
                # Diagonal histogram will auto-scale
                
                # Add statistics
                mean_val = np.mean(weights)
                std_val = np.std(weights)
                ax.text(0.05, 0.95, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                      transform=ax.transAxes, fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                      verticalalignment='top')
            else:
                title, color = plot_config[category]
                ax.text(0.5, 0.5, f'No {title}', 
                      transform=ax.transAxes, ha='center', va='center',
                      fontsize=12, style='italic')
                ax.set_title(f'{title} (n=0)')
                # Set common x-axis limits even for empty plots (except diagonal)
                if category != 'diagonal':
                    ax.set_xlim(x_min, x_max)
        
        # Remove the last subplot (bottom-right)
        ax = self.plt.subplot(nrow,ncol,6)
        ax.remove()
        
        # Add overall title
        
        fig.suptitle(f'True Weight, sorted by struct_fit - {md["short_name"]}', fontsize=14, fontweight='bold')
        
#...!...!..................
    def weight_reconstruction_results(self, A_true, A_estimated, train_losses, val_losses, md, figId=4):
        """Plot weight reconstruction results."""
        figId=self.smart_append(figId)        
        nrow,ncol=2,3
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,8))

        # True connectivity structure with weight magnitudes
        ax = self.plt.subplot(nrow,ncol,1)
        im1 = ax.imshow(A_true, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        title = 'True Weights (%s)' % md["short_name"]
        ax.set_title(title)
        ax.set_ylabel('From neuron')
        ax.set_xlabel('To neuron')
        fig.colorbar(im1, ax=ax)
        ax.grid(True, alpha=0.3)
        
        # Estimated connectivity
        ax = self.plt.subplot(nrow,ncol,2)
        im2 = ax.imshow(A_estimated, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax.set_title('Estimated Weights')
        ax.set_xlabel('From neuron')
        ax.set_ylabel('To neuron')
        fig.colorbar(im2, ax=ax)
        ax.grid(True, alpha=0.3)
        
        # Difference
        diff = A_estimated - A_true
        ax = self.plt.subplot(nrow,ncol,3)
        im3 = ax.imshow(diff, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        ax.set_title('Difference (Est - True)')
        ax.set_xlabel('From neuron')
        ax.set_ylabel('To neuron')
        fig.colorbar(im3, ax=ax)
        ax.grid(True, alpha=0.3)
        
        # Scatter plot of true vs estimated
        mask = np.abs(A_true) > 1e-6
        ax = self.plt.subplot(nrow,ncol,4)
        ax.scatter(A_true[mask], A_estimated[mask], alpha=0.5)
        ax.plot([-0.5, 0.5], [-0.5, 0.5], 'r--')
        ax.set_xlabel('True weight')
        ax.set_ylabel('Estimated weight')
        ax.set_title('Weight comparison')
        ax.grid(True, alpha=0.3)
        
        # Training curves
        ax = self.plt.subplot(nrow,ncol,5)
        ax.plot(train_losses, label='Train', color='red', linestyle='-')
        ax.plot(val_losses, label='Val', color='red', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training curves (Stage 1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Weight distribution comparison
        ax = self.plt.subplot(nrow,ncol,6)
        ax.hist(A_true[mask].flatten(), bins=30, alpha=0.5, label='True', density=True)
        ax.hist(A_estimated[mask].flatten(), bins=30, alpha=0.5, label='Estimated', density=True)
        ax.set_xlabel('Weight value')
        ax.set_ylabel('Density')
        ax.set_title('Weight distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.plt.tight_layout()

#...!...!..................
    def weight_category_analysis(self, A_true, A_estimated, md, figId=5):
        """Plot correlation and residuals analysis by weight categories."""
        figId=self.smart_append(figId)        
        nrow,ncol=2,3
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,7))

        # Number of excitatory neurons must be provided
        num_excite = md['num_excit_neur']
        
        # Create masks for different categories
        diag_mask = np.eye(A_true.shape[0], dtype=bool)
        
        # Excitatory and inhibitory masks (rows)
        exc_mask = np.zeros_like(A_true, dtype=bool)
        exc_mask[:num_excite, :] = True
        inh_mask = np.zeros_like(A_true, dtype=bool)
        inh_mask[num_excite:, :] = True
        
        # True edges (non-zero weights)
        true_edges = np.abs(A_true) > 1e-10
        
        # Categories for analysis
        categories = {
            'diagonal': A_true[diag_mask],
            'excitatory': A_true[true_edges & exc_mask & ~diag_mask],
            'inhibitory': A_true[true_edges & inh_mask & ~diag_mask]
        }
        
        estimated_categories = {
            'diagonal': A_estimated[diag_mask],
            'excitatory': A_estimated[true_edges & exc_mask & ~diag_mask],
            'inhibitory': A_estimated[true_edges & inh_mask & ~diag_mask]
        }
        
        titles = ['Diagonal Weights', 'Excitatory Weights', 'Inhibitory Weights']
        colors = ['blue', 'red', 'green']
        
        # Upper row: Correlation plots
        for i, (category, true_weights) in enumerate(categories.items()):
            ax = self.plt.subplot(nrow,ncol,i+1)
            est_weights = estimated_categories[category]
            
            if len(true_weights) > 0:
                ax.scatter(true_weights, est_weights, alpha=0.6, color=colors[i])
                # Add diagonal line
                min_val = min(np.min(true_weights), np.min(est_weights))
                max_val = max(np.max(true_weights), np.max(est_weights))
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                
                # Calculate correlation
                correlation = np.corrcoef(true_weights, est_weights)[0, 1]
                ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                      transform=ax.transAxes, fontsize=10,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                      verticalalignment='top')
                
                ax.set_xlabel('True weight')
                ax.set_ylabel('Estimated weight')
                ax.set_title(f'{titles[i]} (n={len(true_weights)})')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No {titles[i]}', 
                      transform=ax.transAxes, ha='center', va='center',
                      fontsize=12, style='italic')
                ax.set_title(f'{titles[i]} (n=0)')
        
        # Bottom row: Residual histograms
        for i, (category, true_weights) in enumerate(categories.items()):
            ax = self.plt.subplot(nrow,ncol,i+4)
            est_weights = estimated_categories[category]
            
            if len(true_weights) > 0:
                residuals = est_weights - true_weights
                ax.hist(residuals, bins=30, alpha=0.7, color=colors[i], edgecolor='black')
                
                # Calculate statistics
                mean_residual = np.mean(residuals)
                rms_residual = np.sqrt(np.mean(residuals**2))
                
                ax.text(0.05, 0.95, f'Mean: {mean_residual:.4f}\nRMS: {rms_residual:.4f}', 
                      transform=ax.transAxes, fontsize=10,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                      verticalalignment='top')
                
                ax.set_xlabel('Residual (Est - True)')
                ax.set_ylabel('Count')
                ax.set_title(f'{titles[i]} Residuals (n={len(true_weights)})')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No {titles[i]}', 
                      transform=ax.transAxes, ha='center', va='center',
                      fontsize=12, style='italic')
                ax.set_title(f'{titles[i]} Residuals (n=0)')
        
        self.plt.tight_layout()