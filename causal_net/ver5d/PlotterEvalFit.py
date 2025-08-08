__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from toolbox.PlotterBackbone import PlotterBackbone
from matplotlib import cm as cmap
import matplotlib.ticker as ticker
from pprint import pprint
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gennorm
   
#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)         

#...!...!..................
    def correl_after_thresh(self, trueD,fitD, md, fitType,figId=1):
        #pprint(md)
        fmd=md['fit_'+fitType]
        
        figId=self.smart_append(figId)        
        nrow,ncol=2,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(13,7))

        # Unpack arrays from bigD
        A_true = trueD['A_true']
        B_true = trueD['B_true']
        A_fit = fitD['A_'+fitType]
        B_fit = fitD['B_'+fitType]
        
        for j,ntype in enumerate(['exc','inh']):
            gmask=trueD['mask.geom.'+ntype]
            tmask=trueD['mask.true.'+ntype]
            fmask=fitD['mask.lasso.%s'%(ntype)]
        
            ax = self.plt.subplot(nrow,ncol,1+j)
            title = 'fit (%s)' % ntype
            plot_Afit(fig,ax,A_fit,fmask,title=title)
          
            ax = self.plt.subplot(nrow,ncol,1+ncol+j)
            plot_correl_offdiag(fig,ax,A_true,A_fit,tmask,fmask,title=title)
       
        # ... diagonal
        gexc_1d= trueD['mask.geom.exc_idx']
        mask_diag= trueD['mask.geom.diag']
        title = 'fit (diagonal)'
        ax = self.plt.subplot(nrow,ncol,3+ncol)
        
        plot_correl_diag(fig,ax,A_true[mask_diag],A_fit[mask_diag],gexc_1d,title=title)
       
        # ...  B-term
        title = 'fit (B-term)'
        ax = self.plt.subplot(nrow,ncol,4+ncol)        
        plot_correl_diag(fig,ax,B_true,B_fit,gexc_1d,title=title)
       
        ''' use it later
        gmask=maskG['diag']  # fmask is the same as tmask
        plot_histo(ax,A_true[gmask],A_fit[gmask],title=title)
        '''
            
        #...... Training curves
        ax = self.plt.subplot(nrow,ncol,3)
        plot_trainingCurves(ax,fitD,md,fitType)
                            
        #..... all values of A
        ax = self.plt.subplot(nrow,ncol,4)
        ax.hist(A_fit[~mask_diag], bins=100, alpha=0.7)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.set_title(" Off-Diagonal Weights")
        ax.grid(True)
        ax.set_yscale('log')



#...!...!..................
    def daleA_and_eigen(self, bigD,maskD, md, figId=1):
        pprint(md)
        fmd=md['fit']
        maskF=maskD['fitL1']
        
        figId=self.smart_append(figId)        
        nrow,ncol=2,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,7))

        # Unpack arrays from bigD
        A_true = bigD['A_true']
        A_pass = bigD['A_pass']

        # truth
        ax = self.plt.subplot(nrow,ncol,1)
        plot_dale_matrix(fig,ax,A_true)
        ax.set_title('True Dale :%s'%( md["short_name"]))
        ax = self.plt.subplot(nrow,ncol,1+nrow)
        Eigen=np.linalg.eigvals(A_true)
        plot_dale_eigen(ax,Eigen)
        ax.set_title('True eigen values')
        
        # Fit
        ax = self.plt.subplot(nrow,ncol,2)
        plot_dale_matrix(fig,ax,A_pass)
        #1ax.set_title('Fit spikes, Dale, amplThres=%.2f'%(md['post']['target_density']))
        ax = self.plt.subplot(nrow,ncol,2+nrow)
        Eigen=np.linalg.eigvals(A_pass)
        plot_dale_eigen(ax,Eigen)
        ax.set_title('Fit eigen values')
        

#...!...!..................
    def plot_slicedA_histos(self, fitD, md, fitType,figId=1, k=5):
        #pprint(md)
        fmd=md['fit_'+fitType]

        
        figId=self.smart_append(figId)        
        nrow,ncol=k,2  # Add 1 extra row for the neuron stats plot
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,10))

        # Unpack arrays from bigD
        A_fit = fitD['A_'+fitType]

        num_neurons = A_fit.shape[0]
        wzoomMx=0.05        
        # Remove diagonal elements by setting them to NaN
        A_fit_no_diag = A_fit.copy()
        np.fill_diagonal(A_fit_no_diag, np.nan)
        
        # Left column: 2D histogram of A-matrix (top 2 rows)
        ax = self.plt.subplot2grid((nrow, ncol), (0, 0), rowspan=2)
        ax.axvline(0, linestyle='--', color='r', linewidth=1)
        
        # Create 2D histogram: x-axis is value, y-axis is row index
        A_flat = A_fit_no_diag.flatten()
        row_indices = np.repeat(np.arange(num_neurons), num_neurons)
        
        # Remove NaN values (diagonal elements)
        valid_mask = ~np.isnan(A_flat)
        A_flat = A_flat[valid_mask]
        row_indices = row_indices[valid_mask]
        
        # Use ax.hist2d() directly with log scale
        H, xedges, yedges, im = ax.hist2d(A_flat, row_indices, bins=[50, num_neurons], cmap='Greys', vmax=6)
        ax.set_xlabel('A-matrix value')
        ax.set_ylabel('neuron index')
        ax.set_title(f'2D non-diag A-matrix (Fit), epochs={fmd["n_epochs"]}')
        fig.colorbar(im, ax=ax)
        
        # Add horizontal lines to mark K block boundaries
        add_k_block_lines(ax, num_neurons, k)
        
        # Second 2D histogram (zoomed) in the left column, bottom 2 rows
        ax = self.plt.subplot2grid((nrow, ncol), (2, 0), rowspan=2)
        ax.axvline(0, linestyle='--', color='r', linewidth=1)
        # Filter data to narrow x-range (A_flat already has diagonal removed)
        mask = (A_flat >= -wzoomMx) & (A_flat <= wzoomMx)
        A_flat_narrow = A_flat[mask]
        row_indices_narrow = row_indices[mask]
        
        # Use ax.hist2d() directly with log scale and narrowed range
        H2, xedges2, yedges2, im2 = ax.hist2d(A_flat_narrow, row_indices_narrow, bins=[51, num_neurons], cmap='Purples', norm=colors.LogNorm())
        ax.set_xlabel('A-matrix value')
        ax.set_ylabel('Neuron index')
        ax.set_title(f'Input: {md["short_name"]}  zoom-in')
        ax.set_xlim(-wzoomMx,wzoomMx)
        fig.colorbar(im2, ax=ax)
        
        # Add horizontal lines to mark K block boundaries
        add_k_block_lines(ax, num_neurons, k)
        
        # Right column: K 1D histograms in separate rows
        rows_per_group = num_neurons // k
        color_list = [ 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive','red',]
        
        for i in range(k):
            # Reverse the order by using (k-1-i) for positioning
            ax = self.plt.subplot(nrow,ncol,2+(k-1-i)*ncol)
            
            start_row = i * rows_per_group
            end_row = (i + 1) * rows_per_group if i < k - 1 else num_neurons
            
            # Extract values from rows in this group (excluding diagonal)
            group_values = A_fit_no_diag[start_row:end_row, :].flatten()
            # Remove NaN values (diagonal elements)
            group_values = group_values[~np.isnan(group_values)]
            
            # Plot 1D histogram with statistics
            plot_1d_histo_with_stats(ax, group_values, start_row, end_row, i, 
                                   color_list[i % len(color_list)], wzoomMx)
        
        # Neuron statistics plot at bottom right
        ax = self.plt.subplot(nrow,ncol,1+(k-1)*ncol)
        plot_neuron_stats(ax, A_flat_narrow, row_indices_narrow, num_neurons)
        
    
#............................
#............................
#............................
   
def plot_trainingCurves(ax,fitD,md,fitType,title='aa3'):
    fmd=md['fit_'+fitType]
            
    train_losses = fitD['train_losses']
    val_losses = fitD['val_losses']
    epochsT=fitD['train_loss_epochs']
    epochsV=fitD['val_loss_epochs']
 
    epoch0=3 # skip intial loss values
    epochs = np.arange(len(train_losses))
    ax.plot(epochsT[epoch0:], train_losses[epoch0:], label='Train '+fitType, color='blue', linestyle='-')
    ax.plot(epochsV, val_losses, label='Val', color='red', linestyle='--')
    
    title = md["short_name"]
    ax.set(xlabel='Epoch', ylabel='Loss', title=title)
    
    titl='%dk samp'%(fmd["num_samples_used"]/1000)
    ax.legend(title=titl)
    ax.grid(True, alpha=0.3)
  
def plot_Afit(fig,ax,A,mask,title='aa'):
    Am=A.copy()
    Am[~mask]=0
 
    vmin = Am.min()-0.3
    vmax = Am.max()+0.3
    print('mm',vmin,vmax,title)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im1=ax.imshow(Am, cmap='bwr', norm=norm, origin='lower')  # 'RdBu_r'

    masked_values = Am[mask]
    vmin = masked_values.min()
    vmax = masked_values.max()
    
    title='%s n=%d'%(title,1)
    ax.set(title=title, ylabel='From neuron', xlabel='To neuron')
    fig.colorbar(im1, ax=ax)
    ax.grid(True, alpha=0.5)
    add_x45_lins(ax, only45=True)

  
def add_x45_lins(ax, only45=False):
    # Add dashed lines through (0,0)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # 45 degree line y=x
    ax.plot(lims, lims, '--', color='m', linewidth=1)
    if  only45: return

    ax.axvline(0, linestyle='--', color='k', linewidth=1)
    ax.axhline(0, linestyle='--', color='k', linewidth=1)

    
def plot_correl_offdiag(fig,ax,At,Af,tmask,fmask,title='aa2'):    
    FP = ~tmask &  fmask  # False Positive: predicted True, actually False
    TP =  tmask &  fmask  # True Positive: predicted True, actually True
    TN = ~tmask & ~fmask  # True Negative: predicted False, actually False
    FN = tmask & ~fmask  # False Negative: predicted False, actually True

    ax.scatter(At[TP], Af[TP], alpha=0.6, color='green',label='TP: %d'%np.sum(TP), facecolors='none')
    ax.scatter(At[FN], Af[FN], alpha=0.6, color='red',label='FN: %d'%np.sum(FN))
    ax.scatter(At[FP], Af[FP], alpha=0.6, color='orange',label='FP: %d'%np.sum(FP))

    ax.set(aspect=1. ,xlabel='true weight',ylabel='fitted',title=title)
    ax.grid(True, alpha=0.5)

    add_x45_lins(ax)
    if np.any(TP):  # Only plot if there are TP points
        x_cg = np.mean(At[TP])
        y_cg = np.mean(Af[TP])
        ax.scatter(x_cg, y_cg, marker='+', s=200, color='k', linewidths=2) #, label='TP avr')
    ax.legend()


def plot_correl_diag(fig,ax,Bt,Bf,exc_mask,title='aa2'):
    BtFl=Bt.flatten()
    BfFl=Bf.flatten()
  
    ax.scatter(BtFl[~exc_mask], BfFl[~exc_mask], alpha=0.6, color='blue',label='inh: %d'%np.sum(~exc_mask), facecolors='none')
    ax.scatter(BtFl[exc_mask], BfFl[exc_mask], alpha=0.6, color='salmon',label='exc: %d'%np.sum(exc_mask), facecolors='none')
    add_x45_lins(ax, only45=True)
    ax.set(aspect=1. ,xlabel='true value',ylabel='fitted',title=title)
    ax.grid(True, alpha=0.5)
    ax.legend()



#...!...!..................
def plot_1d_histo_with_stats(ax, group_values, start_row, end_row, group_idx, color, wzoomMx):
    """Plot 1D histogram with statistics for a single group"""
    
    # Create histogram for this group with log scale
    counts, bins, _ = ax.hist(group_values, bins=100, alpha=0.7, color=color)
    
    # Draw vertical lines at ±wzoomMx and compute statistics in that range
    ax.axvline(x=-wzoomMx, color='black', linestyle='--', linewidth=1, alpha=0.8)
    ax.axvline(x=wzoomMx, color='black', linestyle='--', linewidth=1, alpha=0.8)
    ax.axvline(x=0, color='black', linestyle=':', linewidth=1, alpha=0.8)

    # Compute mean and RMSE in the range ±wzoomMx
    mask_range = (group_values >= -wzoomMx) & (group_values <= wzoomMx)
    values_in_range = group_values[mask_range]
    
    if len(values_in_range) > 0:
        mean_val = np.mean(values_in_range)
        rmse_val = np.sqrt(np.mean(values_in_range**2))
        
        # Fit Generalized Gaussian Distribution (GGD)
        try:
            # Fit GGD without any constraints to allow all parameters to be estimated freely
            beta, loc, scale = gennorm.fit(values_in_range)
            
            # Format with scientific notation for very small values
            if abs(loc) < 1e-3:
                loc_str = f'{loc:.2e}'
            else:
                loc_str = f'{loc:.4f}'
            
            if abs(scale) < 1e-3:
                scale_str = f'{scale:.2e}'
            else:
                scale_str = f'{scale:.4f}'
                
            ggd_text = f'β={beta:.4f}\nx₀={loc_str}\nμ={scale_str}'
            
            # Plot GGD fit curve
            x_fit = np.linspace(-wzoomMx, wzoomMx, 100)
            # Scale the PDF to match histogram counts
            ggd_pdf = gennorm.pdf(x_fit, beta, loc, scale)
            # Scale to match histogram by finding the maximum bin count in range
            bin_centers = (bins[:-1] + bins[1:]) / 2
            mask_fit_bins = (bin_centers >= -wzoomMx) & (bin_centers <= wzoomMx)
            if np.any(mask_fit_bins):
                max_count_in_range = np.max(counts[mask_fit_bins])
                max_pdf = np.max(ggd_pdf)
                if max_pdf > 0:
                    scaled_pdf = ggd_pdf * max_count_in_range / max_pdf
                    ax.plot(x_fit, scaled_pdf, '--', color='magenta', linewidth=2, 
                           label='GGD fit', zorder=8)
            
        except Exception as e:
            print(f"GGD fit failed for group {group_idx+1}: {e}")
            beta, loc, scale = np.nan, np.nan, np.nan
            ggd_text = f'β=failed\nx₀=failed\nμ=failed'
        
        # Add text with statistics
        ax.text(0.25, 0.85, f'mean={mean_val:.4f}\nRMSE={rmse_val:.4f}\n{ggd_text}', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top')
        
        # Add circle for mean with horizontal error bar for RMSE at half height
        max_count = np.max(counts)
        half_height = max_count / 20
        
        # Draw horizontal error bar for RMSE
        ax.errorbar(mean_val, half_height, xerr=rmse_val, fmt='o',
                   color='black', capsize=8, capthick=1, linewidth=1, zorder=9)
    
    ax.set_yscale('log')  # Set y-axis to log scale
    ax.set_xlabel('off-diagonal weight value')
    ax.set_ylabel('count')
    ax.set_title(f'Fitted weights, group {group_idx+1}: neurons {start_row}-{end_row-1} (n={len(group_values)})')
    ax.grid(True, alpha=0.3)
    
    # Limit y-range from 0.5 to max count
    max_count = np.max(counts)
    ax.set_ylim(0.5, max_count * 1.1)  # Add 10% margin above max


#...!...!..................
def add_k_block_lines(ax, num_neurons, k):
    """Add horizontal lines to mark K block boundaries"""
    rows_per_group = num_neurons // k
    for i in range(1, k):  # Don't draw line at the very top or bottom
        y_line = i * rows_per_group - 0.5  # Position between blocks
        ax.axhline(y=y_line, color='black', linestyle='--', linewidth=0.5, alpha=1.0)

#...!...!..................
def plot_neuron_stats(ax, A_flat_narrow, row_indices_narrow, num_neurons):
    """Plot mean and RMSE for each neuron index"""
    
    means = []
    rmses = []
    neuron_indices = []
    
    # Compute statistics for each neuron
    for neuron_idx in range(num_neurons):
        # Get values for this specific neuron
        mask = row_indices_narrow == neuron_idx
        values = A_flat_narrow[mask]
        
        if len(values) > 0:
            mean_val = np.mean(values)
            rmse_val = np.sqrt(np.mean(values**2))
            
            means.append(mean_val)
            rmses.append(rmse_val)
            neuron_indices.append(neuron_idx)
    
    # Convert to numpy arrays
    means = np.array(means)
    rmses = np.array(rmses)
    neuron_indices = np.array(neuron_indices)
    
    # Plot mean values with RMSE as error bars
    ax.errorbar(neuron_indices, means, yerr=rmses, fmt='o', 
               color='blue', capsize=3, capthick=1, linewidth=1, markersize=3)
    
    ax.set_xlabel('Neuron index')
    ax.set_ylabel('Mean ± RMSE')
    ax.set_title('Per-neuron zero-values stats')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
