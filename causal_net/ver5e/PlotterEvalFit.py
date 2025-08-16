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
from brokenaxes import brokenaxes

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
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,7))

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
            
        #...... Training curves
        ax = self.plt.subplot(nrow,ncol,3)
        plot_trainingCurves(ax,fitD,md,fitType)
                            
        #..... all values of A
        ax = self.plt.subplot(nrow,ncol,4)
        ax.hist(A_fit[~mask_diag], bins=100, alpha=0.7)
        ax.set_xlabel("Off-Diagonal Weights")
        ax.set_ylabel("edge count")
        ax.set_title("Fit "+fitType)
        ax.grid(True)
        ax.set_yscale('log')


#...!...!..................
    def slicedA_histos(self, fitD, md, fitType,figId=1, k=5):
        #pprint(md); aa67
        fmd=md['fit_'+fitType]
        ampl_thres=fmd['ampl_thres']
        
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
        ax.set_xlabel('non-diag weigts')
        ax.set_ylabel('neuron index')
        ax.set_title(f'Fit {fitType}, epochs={fmd["n_epochs"]}')
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
        ax.set_title(f'Input: {md["short_name"]},  zoom-in')
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
            plot_1d_weight_histo_with_stats(ax, group_values, start_row, end_row, i, 
                                            color_list[i % len(color_list)], ampl_thres/2)
        
        # Neuron statistics plot at bottom right
        ax = self.plt.subplot(nrow,ncol,1+(k-1)*ncol)
        plot_neuron_stats(ax, A_flat_narrow, row_indices_narrow, num_neurons)
        
#...!...!..................
    def residuals(self, trueD,fitD, md, fitType,figId=1):
        #pprint(md)
        fmd=md['fit_'+fitType]
        
        figId=self.smart_append(figId)        
        nrow,ncol=2,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,6))

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
            valT,valF=plot_correl_offdiag(fig,ax,A_true,A_fit,tmask,fmask,title=title)

            ax = self.plt.subplot(nrow,ncol,1+j+ncol)
            plot_1D_residuals(ax, valT,valF,lab='TP off-diag '+ntype,col='green')

            if ntype=='exc':
                txt=md["short_name"]
            else:
                txt='fit: '+fitType
            ax.text(0.05, 0.2, txt, transform=ax.transAxes, fontsize=8)

        # ... diagonal
        gexc_1d= trueD['mask.geom.exc_idx']
        mask_diag= trueD['mask.geom.diag']
        title = 'fit (diagonal)'
        ax = self.plt.subplot(nrow,ncol,3)
        vecT=A_true[mask_diag].flatten()
        vecF=A_fit[mask_diag].flatten()
        plot_correl_diag(fig,ax,vecT,vecF,gexc_1d,title=title)
        
        ax = self.plt.subplot(nrow,ncol,3+ncol)        
        plot_1D_residuals(ax, vecT[~gexc_1d],vecF[~gexc_1d],lab='inh diag',col='blue')
        plot_1D_residuals(ax, vecT[gexc_1d],vecF[gexc_1d],lab='exc diag',col='tomato',first=False)

        # ...  B-term
        title = 'fit (B-term)'
        ax = self.plt.subplot(nrow,ncol,4)
        vecT=B_true
        vecF=B_fit
        plot_correl_diag(fig,ax,vecT,vecF,gexc_1d,title=title)
        
        ax = self.plt.subplot(nrow,ncol,4+ncol)        
        plot_1D_residuals(ax, vecT[~gexc_1d],vecF[~gexc_1d],lab='inh B-term',col='blue')
        plot_1D_residuals(ax, vecT[gexc_1d],vecF[gexc_1d],lab='exc B-term',col='tomato',first=False)

#...!...!..................
    def compare_eigen(self, trueD,fitD, md, fitType,figId=4):
        # Unpack arrays from bigD
        A_true = trueD['A_true']
        B_true = trueD['B_true']
        A_fit = fitD['A_'+fitType]
        B_fit = fitD['B_'+fitType]
        gexc_1d= trueD['mask.geom.exc_idx']
         
        figId=self.smart_append(figId)        
        nrow,ncol=2,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(5,10))

        ax = self.plt.subplot(nrow,ncol,1)
        ax.set_title(f'{fitType}: {md["short_name"]}')
        eigT=np.linalg.eigvals(A_true)
        eigF=np.linalg.eigvals(A_fit)
        plot_both_eigen(ax,eigT,eigF)

        ax = self.plt.subplot(nrow,ncol,2)
        title = 'fit (B-term)'
        
        vecT=B_true
        vecF=B_fit
        plot_correl_diag(fig,ax,vecT,vecF,gexc_1d,title=title)
        
#...!...!..................
    def experiment_eigen(self, fitD, md, fitType,figId=4):
        
        # Unpack arrays from bigD
        A_fit = fitD['A_'+fitType]
        B_fit = fitD['B_'+fitType]
         
        figId=self.smart_append(figId)        
        nrow,ncol=1,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,4))

        ax = self.plt.subplot(nrow,ncol,1)
        plot_trainingCurves(ax,fitD,md,fitType)

        ax = self.plt.subplot(nrow,ncol,2)
        ax.set_title(f'{fitType}: {md["short_name"]}')
        eigF=np.linalg.eigvals(A_fit)
        reF = np.real(eigF)
        imF = np.imag(eigF)

        ax.scatter(reF, imF, color='red', marker='o', facecolors='none',label='fit',s=20)
        ax.set_ylim(-0.1,)
        #1ax.set_xlim(right=1)
        ax.axhline(0, linestyle='--', color='k', linewidth=1)
        ax.axvline(0, linestyle='--', color='k', linewidth=1)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")
        ax.legend()

        
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
    ax.plot(epochsV, val_losses, label='Val NLL', color='red', linestyle='--')
    
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
    ax.plot(lims, lims, '--', color='k', linewidth=0.8)
    if  only45: return

    ax.axvline(0, linestyle='--', color='k', linewidth=1)
    ax.axhline(0, linestyle='--', color='k', linewidth=1)

    
def plot_correl_offdiag(fig,ax,At,Af,tmask,fmask,title='aa2'):    
    FP = ~tmask &  fmask  # False Positive: predicted True, actually False
    TP =  tmask &  fmask  # True Positive: predicted True, actually True
    TN = ~tmask & ~fmask  # True Negative: predicted False, actually False
    FN = tmask & ~fmask  # False Negative: predicted False, actually True

    ax.scatter(At[TP], Af[TP], alpha=0.6, color='green',label='TP: %d'%np.sum(TP),marker='.',s=5) #, facecolors='none')
    ax.scatter(At[FN], Af[FN], alpha=0.6, color='red',s=5,label='FN: %d'%np.sum(FN))
    ax.scatter(At[FP], Af[FP], alpha=0.6, color='orange',s=5,label='FP: %d'%np.sum(FP))

    ax.set(aspect=1. ,xlabel='true weight',ylabel='fitted',title=title)
    ax.grid(True, alpha=0.5)

    add_x45_lins(ax)
    if np.any(TP):  # Only plot if there are TP points
        x_cg = np.mean(At[TP])
        y_cg = np.mean(Af[TP])
        ax.scatter(x_cg, y_cg, marker='+', s=200, color='k', linewidths=2) #, label='TP avr')
    ax.legend()
    return At[TP], Af[TP]

def plot_correl_diag(fig,ax,Bt,Bf,exc_mask,title='aa2'):
    BtFl=Bt.flatten()
    BfFl=Bf.flatten()
  
    ax.scatter(BtFl[~exc_mask], BfFl[~exc_mask], alpha=0.6, color='blue',label='inh: %d'%np.sum(~exc_mask), marker='.',s=5) #facecolors='none')
    ax.scatter(BtFl[exc_mask], BfFl[exc_mask], alpha=0.6, color='tomato',label='exc: %d'%np.sum(exc_mask), marker='.',s=5) #, facecolors='none')
    add_x45_lins(ax, only45=True)
    ax.set(aspect=1. ,xlabel='true value',ylabel='fitted',title=title)
    ax.grid(True, alpha=0.5)
    ax.legend()


#...!...!..................
def plot_1d_weight_histo_with_stats(ax, group_values, start_row, end_row, group_idx, color, ggd_range):
    """Plot 1D histogram with statistics for a single group"""
    
    # Create histogram for this group with log scale
    counts, bins, _ = ax.hist(group_values, bins=100, alpha=0.7, color=color)
    
    ax.axvline(x=-ggd_range, color='black', linestyle='--', linewidth=1, alpha=0.8)
    ax.axvline(x=ggd_range, color='black', linestyle='--', linewidth=1, alpha=0.8)
    ax.axvline(x=0, color='black', linestyle=':', linewidth=1, alpha=0.8)

    # Compute mean and RMSE in the range ±ggd_range
    mask_range = (group_values >= -ggd_range) & (group_values <= ggd_range)
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
            ggd_plot_range=2*ggd_range
            x_fit = np.linspace(-ggd_plot_range, ggd_plot_range, 100)
            # Scale the PDF to match histogram counts
            ggd_pdf = gennorm.pdf(x_fit, beta, loc, scale)
            
            # Scale to match histogram by finding the maximum bin count in range
            bin_centers = (bins[:-1] + bins[1:]) / 2
            mask_fit_bins = (bin_centers >= -ggd_plot_range) & (bin_centers <= ggd_plot_range)
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
        half_height = max_count / 200
        
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


#...!...!..................
def plot_1D_residuals(ax, valT,valF,lab,col,first=True):
    fac=100
    res=fac*(valT-valF)
    
    # Plot histogram of residuals
    n, bins, patches = ax.hist(res, bins=30, color=col, edgecolor='none', alpha=0.7, density=True )

    # Compute statistics: mean, standard deviation, and RMSE
    mean_val = np.mean(res)
    std_val = np.std(res)
    n_entries = len(valT)

    # Determine half of the maximum bin height to position the error bar
    half_height = np.max(n) / 2.0

    # Draw horizontal error bar for RMSE at the computed mean and half-height
    ax.errorbar(mean_val, half_height, xerr=std_val, fmt='o',
                color='black', capsize=8, capthick=1, linewidth=1, zorder=9)

    # Add text to the figure displaying the mean, standard deviation, and RMSE
    textstr = f"{lab}\nN    = {n_entries}\nMean = {mean_val:.2f}\nStd  = {std_val:.2f}"

    if first:
        x0=0.95
    else:
        x0=0.4
    ax.text(x0, 0.95, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='right')

    # Add a black vertical line at x = 0
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set(xlabel='residuals x %d'%fac, ylabel='density/bin')


#...!...!..................
def plot_both_eigen(ax, eigT,eigF):
    reT = np.real(eigT)
    imT = np.imag(eigT)

    ax.scatter(reT, imT, color='blue', marker='o',label='True',s=10)

    reF = np.real(eigF)
    imF = np.imag(eigF)

    ax.scatter(reF, imF, color='red', marker='o', facecolors='none',label='fit',s=20)
    ax.set_ylim(-0.1,)
    ax.set_xlim(right=1)
    ax.axhline(0, linestyle='--', color='k', linewidth=1)
    ax.axvline(0, linestyle='--', color='k', linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.legend()
 
