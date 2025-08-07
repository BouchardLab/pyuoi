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
    
    def add_histogram_stats(self, ax, data, color='black', x0=None):
        """Add mean, std, and vertical line at x=0 to histogram."""
        if len(data) > 0:
            mean_val = np.mean(data)
            std_val = np.std(data)
            
            # Draw vertical line at x=0
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            
            # Draw circle with error bar at half height
            ylim = ax.get_ylim()
            y_pos = (ylim[0] + ylim[1]) / 2
            
            # Draw horizontal error bar
            ax.errorbar(mean_val, y_pos, xerr=std_val, fmt='o', 
                       markersize=8, color='black', capsize=5, capthick=2, linewidth=2)
            
            # Add text with mean and std, and sigma/x0 ratio if x0 is provided
            if x0 is not None and abs(x0) > 1e-10:
                sigma_x0_ratio = std_val / abs(x0)
                ax.text(0.05, 0.92, f'μ={mean_val:.3f}\nσ={std_val:.3f}\nσ/|x0|={sigma_x0_ratio:.3f}', 
                       transform=ax.transAxes, fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top')
            else:
                ax.text(0.05, 0.92, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                       transform=ax.transAxes, fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top')

#...!...!..................
    def structure_results(self, bigD, md, figId=1):
        """Plot structure identification results."""
        # Unpack arrays from bigD
        A_true = bigD['A_true']
        mask_detected = bigD['mask']
        train_losses = bigD['train_losses']
        val_losses = bigD['val_losses']
        edge_analysis = bigD.get('edge_analysis', None)
        
        figId=self.smart_append(figId)        
        nrow,ncol=2,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,8))

        # Number of excitatory neurons must be provided
        num_excite = md['num_excit_neur']
        
        # True connectivity structure with weight magnitudes
        ax = self.plt.subplot(nrow,ncol,1)
        im1 = ax.imshow(A_true, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        title = 'True Weights (%s)' % md["short_name"]
        ax.set(title=title, ylabel='From neuron', xlabel='To neuron')
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
        ax.set(title='Fitted binary map, nSamp:%dk' % (md['num_samples']/1000), 
               ylabel='From neuron', xlabel='To neuron')
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
        
        # Create title with number of samples if available
        title = 'Training curves'
        if 'num_samples' in md:
            title += f' (N={md["num_samples"]})'
        ax.set(xlabel='Epoch', ylabel='Loss', title=title)
        
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
        
        ax.set(title='Structure Detection (%s)' % md["short_name"])
        ax.axis('off')

# Rate plotting moved to PlotterSimPoisson.py

#...!...!..................
    def weight_distributions(self, bigD, md, figId=3):
        """Plot histograms of true weight distributions for different categories."""
        # Unpack arrays from bigD
        A_true = bigD['A_true']
        mask_detected = bigD['mask']
        
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
                ax.hist(weights, bins=30, alpha=0.7, color=color)
                ax.set(title=f'{title} (n={len(weights)})', xlabel='Weight Value', ylabel='Count')
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
                ax.set(title=f'{title} (n=0)')
                # Set common x-axis limits even for empty plots (except diagonal)
                if category != 'diagonal':
                    ax.set_xlim(x_min, x_max)
        
        # Remove the last subplot (bottom-right)
        ax = self.plt.subplot(nrow,ncol,6)
        ax.remove()
        
        # Add overall title
        
        fig.suptitle(f'True Weight, sorted by struct_fit - {md["short_name"]}', fontsize=14, fontweight='bold')
        
#...!...!..................
    def weight_reconstruction_results(self, bigD, md, figId=4):
        """Plot weight reconstruction results."""
        # Unpack arrays from bigD
        A_true = bigD['A_true']
        A_estimated = bigD['A_stage1']
        B_true = bigD['B_true']
        B_estimated = bigD['B_stage1']
        train_losses = bigD['train_losses']
        val_losses = bigD['val_losses']
        
        # Debug: Check if matrices are different
        if A_true is not None and A_estimated is not None:
            diff_norm = np.linalg.norm(A_true - A_estimated)
            print(f"DEBUG: ||A_true - A_estimated|| = {diff_norm:.6f}")
            print(f"DEBUG: A_true range: [{np.min(A_true):.6f}, {np.max(A_true):.6f}]")
            print(f"DEBUG: A_estimated range: [{np.min(A_estimated):.6f}, {np.max(A_estimated):.6f}]")
            if diff_norm < 1e-10:
                print("WARNING: A_true and A_estimated are nearly identical!")
        else:
            print("WARNING: A_true or A_estimated is None!")
        
        figId=self.smart_append(figId)        
        nrow,ncol=2,3
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,8))

        # True connectivity structure with weight magnitudes
        ax = self.plt.subplot(nrow,ncol,1)
        im1 = ax.imshow(A_true, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        title = 'True Weights (%s)' % md["short_name"]
        ax.set(title=title, ylabel='From neuron', xlabel='To neuron')
        fig.colorbar(im1, ax=ax)
        ax.grid(True, alpha=0.3)
        
        # Estimated connectivity
        ax = self.plt.subplot(nrow,ncol,2)
        im2 = ax.imshow(A_estimated, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax.set(title='Estimated Weights * true mask', xlabel='From neuron', ylabel='To neuron')
        fig.colorbar(im2, ax=ax)
        ax.grid(True, alpha=0.3)
        
        # Scatter plot of true vs estimated weights
        ax = self.plt.subplot(nrow,ncol,3)
        mask = np.abs(A_true) > 1e-6
        ax.scatter(A_true[mask], A_estimated[mask], alpha=0.5)
        ax.plot([-0.5, 0.5], [-0.5, 0.5], 'r--')
        ax.set(xlabel='True weight', ylabel='Estimated weight', title='Weight comparison')
        ax.grid(True, alpha=0.3)
        
        # Training curves
        ax = self.plt.subplot(nrow,ncol,4)
        ax.plot(train_losses, label='Train', color='red', linestyle='-')
        ax.plot(val_losses, label='Val', color='red', linestyle='--')
        ax.set(xlabel='Epoch', ylabel='Loss', title='Training curves (Stage 1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Weight distribution comparison
        ax = self.plt.subplot(nrow,ncol,5)
        ax.hist(A_true[mask].flatten(), bins=30, alpha=0.5, label='True', density=True)
        ax.hist(A_estimated[mask].flatten(), bins=30, alpha=0.5, label='Estimated', density=True)
        ax.set(xlabel='Weight value', ylabel='Density', title='Weight distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        

        
        

#...!...!..................
    def weight_category_analysis(self, bigD, md, figId=5):
        """Plot correlation and residuals analysis by weight categories."""
        # Unpack arrays from bigD
        A_true = bigD['A_true']
        A_estimated = bigD['A_stage1']
        edge_analysis = bigD.get('edge_analysis', None)
        mask_detected = bigD['mask']
        
        figId=self.smart_append(figId)        
        nrow,ncol=3,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(20,12))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        # Number of excitatory neurons must be provided
        num_excite = md['num_excit_neur']
        
        # Use masks from edge_analysis (always available)
        true_masks = edge_analysis['true_masks']
        diag_mask = true_masks['diagonal']
        exc_mask = true_masks['excitatory']
        inh_mask = true_masks['inhibitory']
        
        # True edges (non-zero weights)
        true_edges = np.abs(A_true) > 1e-10
        
        # Categories for analysis using the exported masks
        categories = {
            'excitatory': A_true[true_edges & exc_mask & ~diag_mask],
            'inhibitory': A_true[true_edges & inh_mask & ~diag_mask],
            'diagonal': A_true[diag_mask]
        }
        
        estimated_categories = {
            'excitatory': A_estimated[true_edges & exc_mask & ~diag_mask],
            'inhibitory': A_estimated[true_edges & inh_mask & ~diag_mask],
            'diagonal': A_estimated[diag_mask]
        }
        
        # Row 1: Excitatory analysis
        # Column 1: Correlation for found excitatory weights (TP only)
        ax = self.plt.subplot(nrow,ncol,1)
        # Use TP value from edge_analysis dictionary
        exc_tp = edge_analysis['excitatory']['tp']
        # Get True Positives: excitatory connections that were correctly detected
        # Use the detected mask from the analysis
        exc_true_edges = true_edges & exc_mask & ~diag_mask
        exc_detected = mask_detected & exc_mask & ~diag_mask
        exc_tp_mask = exc_true_edges & exc_detected
        true_exc = A_true[exc_tp_mask]
        est_exc = A_estimated[exc_tp_mask]
        
        if len(true_exc) > 0:
            ax.scatter(true_exc, est_exc, alpha=0.6, color='red')
            min_val = min(np.min(true_exc), np.min(est_exc))
            max_val = max(np.max(true_exc), np.max(est_exc))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            # Calculate center of gravity
            x0 = np.mean(true_exc)
            y0 = np.mean(est_exc)
            
            # Draw large black cross at center of gravity
            cross_size = (max_val - min_val) * 0.1
            ax.plot([x0 - cross_size, x0 + cross_size], [y0, y0], 'k-', linewidth=3)
            ax.plot([x0, x0], [y0 - cross_size, y0 + cross_size], 'k-', linewidth=3)
            
            correlation = np.corrcoef(true_exc, est_exc)[0, 1]
            ax.text(0.05, 0.92, f'Correlation: {correlation:.3f}\nx0: {x0:.3f}, y0: {y0:.3f}', 
                  transform=ax.transAxes, fontsize=10,
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                  verticalalignment='top')
            ax.set(xlabel='True weight', ylabel='Estimated weight', 
                   title=f'Excitatory Found (n={exc_tp})')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        else:
            ax.text(0.5, 0.5, 'No excitatory weights found', 
                  transform=ax.transAxes, ha='center', va='center',
                  fontsize=12, style='italic')
            ax.set(title='Excitatory Found (n=0)')
            ax.set_aspect('equal')
        
        # Column 2: Histogram of residuals for found excitatory weights (TP only)
        ax = self.plt.subplot(nrow,ncol,2)
        if len(true_exc) > 0:
            residuals = est_exc - true_exc
            ax.hist(residuals, bins=30, alpha=0.7, color='red')
            ax.set(xlabel='Residual (Est - True)', ylabel='Count', 
                   title=f'Excitatory Residuals (n={exc_tp})')
            ax.grid(True, alpha=0.3)
            self.add_histogram_stats(ax, residuals, color='red', x0=x0)
        else:
            ax.text(0.5, 0.5, 'No excitatory residuals', 
                  transform=ax.transAxes, ha='center', va='center',
                  fontsize=12, style='italic')
            ax.set(title='Excitatory Residuals (n=0)')
        
        # Column 3: Histogram of not found excitatory weights
        ax = self.plt.subplot(nrow,ncol,3)
        # Use edge_analysis data (always available)
        exc_fn = edge_analysis['excitatory']['fn']
        exc_fp = edge_analysis['excitatory']['fp']
        # Get the actual data for plotting using the detected mask
        exc_true_edges = true_edges & exc_mask & ~diag_mask
        exc_detected = mask_detected & exc_mask & ~diag_mask
        exc_not_found = A_true[exc_true_edges & ~exc_detected]
        exc_fp_weights = A_estimated[~exc_true_edges & exc_detected]
        
        if len(exc_not_found) > 0:
            ax.hist(exc_not_found, bins=30, alpha=0.7, color='orange')
            ax.set(xlabel='True weight', ylabel='Count', 
                   title=f'Excitatory Not Found (n={exc_fn})')
            ax.grid(True, alpha=0.3)
            self.add_histogram_stats(ax, exc_not_found, color='orange')
        else:
            ax.text(0.5, 0.5, 'No excitatory weights missed', 
                  transform=ax.transAxes, ha='center', va='center',
                  fontsize=12, style='italic')
            ax.set(title=f'Excitatory Not Found (n={exc_fn})')
        
        # Column 4: Histogram of false positive excitatory weights
        ax = self.plt.subplot(nrow,ncol,4)
        if len(exc_fp_weights) > 0:
            ax.hist(exc_fp_weights, bins=30, alpha=0.7, color='red')
            ax.set(xlabel='Estimated weight', ylabel='Count', 
                   title=f'Excitatory False Positives (n={exc_fp})')
            ax.grid(True, alpha=0.3)
            self.add_histogram_stats(ax, exc_fp_weights, color='red')
        else:
            ax.text(0.5, 0.5, 'No excitatory false positives', 
                  transform=ax.transAxes, ha='center', va='center',
                  fontsize=12, style='italic')
            ax.set(title=f'Excitatory False Positives (n={exc_fp})')
        
        # Row 2: Inhibitory analysis
        # Column 1: Correlation for found inhibitory weights (TP only)
        ax = self.plt.subplot(nrow,ncol,5)
        # Use TP value from edge_analysis dictionary
        inh_tp = edge_analysis['inhibitory']['tp']
        # Get True Positives: inhibitory connections that were correctly detected
        # Use the detected mask from the analysis
        inh_true_edges = true_edges & inh_mask & ~diag_mask
        inh_detected = mask_detected & inh_mask & ~diag_mask
        inh_tp_mask = inh_true_edges & inh_detected
        true_inh = A_true[inh_tp_mask]
        est_inh = A_estimated[inh_tp_mask]
        
        if len(true_inh) > 0:
            ax.scatter(true_inh, est_inh, alpha=0.6, color='blue')
            min_val = min(np.min(true_inh), np.min(est_inh))
            max_val = max(np.max(true_inh), np.max(est_inh))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            # Calculate center of gravity
            x0 = np.mean(true_inh)
            y0 = np.mean(est_inh)
            
            # Draw large black cross at center of gravity
            cross_size = (max_val - min_val) * 0.1
            ax.plot([x0 - cross_size, x0 + cross_size], [y0, y0], 'k-', linewidth=3)
            ax.plot([x0, x0], [y0 - cross_size, y0 + cross_size], 'k-', linewidth=3)
            
            correlation = np.corrcoef(true_inh, est_inh)[0, 1]
            ax.text(0.05, 0.92, f'Correlation: {correlation:.3f}\nx0: {x0:.3f}, y0: {y0:.3f}', 
                  transform=ax.transAxes, fontsize=10,
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                  verticalalignment='top')
            ax.set(xlabel='True weight', ylabel='Estimated weight', 
                   title=f'Inhibitory Found (n={inh_tp})')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        else:
            ax.text(0.5, 0.5, 'No inhibitory weights found', 
                  transform=ax.transAxes, ha='center', va='center',
                  fontsize=12, style='italic')
            ax.set(title='Inhibitory Found (n=0)')
            ax.set_aspect('equal')
        
        # Column 2: Histogram of residuals for found inhibitory weights (TP only)
        ax = self.plt.subplot(nrow,ncol,6)
        if len(true_inh) > 0:
            residuals = est_inh - true_inh
            ax.hist(residuals, bins=30, alpha=0.7, color='blue')
            ax.set(xlabel='Residual (Est - True)', ylabel='Count', 
                   title=f'Inhibitory Residuals (n={inh_tp})')
            ax.grid(True, alpha=0.3)
            self.add_histogram_stats(ax, residuals, color='blue', x0=x0)
        else:
            ax.text(0.5, 0.5, 'No inhibitory residuals', 
                  transform=ax.transAxes, ha='center', va='center',
                  fontsize=12, style='italic')
            ax.set(title='Inhibitory Residuals (n=0)')
        
        # Column 3: Histogram of not found inhibitory weights
        ax = self.plt.subplot(nrow,ncol,7)
        # Use edge_analysis data (always available)
        inh_fn = edge_analysis['inhibitory']['fn']
        inh_fp = edge_analysis['inhibitory']['fp']
        # Get the actual data for plotting using the detected mask
        inh_true_edges = true_edges & inh_mask & ~diag_mask
        inh_detected = mask_detected & inh_mask & ~diag_mask
        inh_not_found = A_true[inh_true_edges & ~inh_detected]
        inh_fp_weights = A_estimated[~inh_true_edges & inh_detected]
        
        if len(inh_not_found) > 0:
            ax.hist(inh_not_found, bins=30, alpha=0.7, color='cyan')
            ax.set(xlabel='True weight', ylabel='Count', 
                   title=f'Inhibitory Not Found (n={inh_fn})')
            ax.grid(True, alpha=0.3)
            self.add_histogram_stats(ax, inh_not_found, color='cyan')
        else:
            ax.text(0.5, 0.5, 'No inhibitory weights missed', 
                  transform=ax.transAxes, ha='center', va='center',
                  fontsize=12, style='italic')
            ax.set(title=f'Inhibitory Not Found (n={inh_fn})')
        
        # Column 4: Histogram of false positive inhibitory weights
        ax = self.plt.subplot(nrow,ncol,8)
        if len(inh_fp_weights) > 0:
            ax.hist(inh_fp_weights, bins=30, alpha=0.7, color='blue')
            ax.set(xlabel='Estimated weight', ylabel='Count', 
                   title=f'Inhibitory False Positives (n={inh_fp})')
            ax.grid(True, alpha=0.3)
            self.add_histogram_stats(ax, inh_fp_weights, color='blue')
        else:
            ax.text(0.5, 0.5, 'No inhibitory false positives', 
                  transform=ax.transAxes, ha='center', va='center',
                  fontsize=12, style='italic')
            ax.set(title=f'Inhibitory False Positives (n={inh_fp})')
        
        # Row 3: Diagonal analysis and stats
        # Column 1: Correlation for diagonal weights
        ax = self.plt.subplot(nrow,ncol,9)
        true_diag = categories['diagonal']
        est_diag = estimated_categories['diagonal']
        if len(true_diag) > 0:
            ax.scatter(true_diag, est_diag, alpha=0.6, color='green')
            min_val = min(np.min(true_diag), np.min(est_diag))
            max_val = max(np.max(true_diag), np.max(est_diag))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            # Calculate center of gravity
            x0 = np.mean(true_diag)
            y0 = np.mean(est_diag)
            
            # Draw large black cross at center of gravity
            cross_size = (max_val - min_val) * 0.1
            ax.plot([x0 - cross_size, x0 + cross_size], [y0, y0], 'k-', linewidth=3)
            ax.plot([x0, x0], [y0 - cross_size, y0 + cross_size], 'k-', linewidth=3)
            
            correlation = np.corrcoef(true_diag, est_diag)[0, 1]
            ax.text(0.05, 0.92, f'Correlation: {correlation:.3f}\nx0: {x0:.3f}, y0: {y0:.3f}', 
                  transform=ax.transAxes, fontsize=10,
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                  verticalalignment='top')
            ax.set(xlabel='True weight', ylabel='Estimated weight', 
                   title=f'Diagonal Found (n={len(true_diag)})')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        else:
            ax.text(0.5, 0.5, 'No diagonal weights found', 
                  transform=ax.transAxes, ha='center', va='center',
                  fontsize=12, style='italic')
            ax.set(title='Diagonal Found (n=0)')
            ax.set_aspect('equal')
        
        # Column 2: Histogram of residuals for diagonal weights
        ax = self.plt.subplot(nrow,ncol,10)
        if len(true_diag) > 0:
            residuals = est_diag - true_diag
            ax.hist(residuals, bins=30, alpha=0.7, color='green')
            ax.set(xlabel='Residual (Est - True)', ylabel='Count', 
                   title=f'Diagonal Residuals (n={len(true_diag)})')
            ax.grid(True, alpha=0.3)
            self.add_histogram_stats(ax, residuals, color='green', x0=x0)
        else:
            ax.text(0.5, 0.5, 'No diagonal residuals', 
                  transform=ax.transAxes, ha='center', va='center',
                  fontsize=12, style='italic')
            ax.set(title='Diagonal Residuals (n=0)')
        
        # Column 3: Stats printout
        ax = self.plt.subplot(nrow,ncol,11)
        ax.axis('off')
        
        # Use edge_analysis data (always available)
        exc_metrics = edge_analysis['excitatory']
        inh_metrics = edge_analysis['inhibitory']
        
        exc_tp = exc_metrics['tp']
        exc_fp = exc_metrics['fp']
        exc_fn = exc_metrics['fn']
        exc_tn = exc_metrics['tn']
        exc_precision = exc_metrics['precision']
        exc_recall = exc_metrics['recall']
        exc_f1 = exc_metrics['f1_score']
        
        inh_tp = inh_metrics['tp']
        inh_fp = inh_metrics['fp']
        inh_fn = inh_metrics['fn']
        inh_tn = inh_metrics['tn']
        inh_precision = inh_metrics['precision']
        inh_recall = inh_metrics['recall']
        inh_f1 = inh_metrics['f1_score']
        
        # Get final loss and training time
        final_loss = md.get('final_loss', 'N/A')
        train_time = md.get('train_time_min', 'N/A')
        
        # Create stats text with tables
        stats_text = f"""
 Excitatory:
 TP: {exc_tp:3d}  FP: {exc_fp:3d}  FN: {exc_fn:3d}  TN: {exc_tn:3d}
 Precision: {exc_precision:.3f}  Recall: {exc_recall:.3f}  F1: {exc_f1:.3f}

 Inhibitory:
 TP: {inh_tp:3d}  FP: {inh_fp:3d}  FN: {inh_fn:3d}  TN: {inh_tn:3d}
 Precision: {inh_precision:.3f}  Recall: {inh_recall:.3f}  F1: {inh_f1:.3f}

 Final Loss: {final_loss}
 Training Time: {train_time:.1f} min

 Input File: {md.get('short_name', 'N/A')}
 Num Samples: {md.get('num_samples', 'N/A')}"""
        
        ax.text(0.0, 1.05, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
        

        