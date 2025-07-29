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

#...!...!..................
def plot_dale_matrix(fig,ax,W):
    normMap = colors.TwoSlopeNorm(vmin=W.min(), vcenter=0, vmax=W.max())

    im=ax.imshow(W, aspect='auto', origin='upper', cmap='bwr', norm=normMap, interpolation='nearest')
    ax.set( xlabel='presyn. node index, source', ylabel='postsyn. node index, target')
    n_nonzero = np.count_nonzero(W)
    ax.text(0.02, 0.98, f'Nonzero: {n_nonzero}', color='black', fontsize=10,
        ha='left', va='top', transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax.set_aspect(1.0)
    ax.grid()
    # Create the colorbar.
    cbar = fig.colorbar(im, ax=ax, extend="both")
    #cbar.set_label('Dal-Matrix: coupling strength')

#...!...!..................
def plot_dale_eigen(ax,Eigen):
    real_parts = np.real(Eigen)
    imag_parts = np.imag(Eigen)
    ax.scatter(real_parts, imag_parts, color='blue', marker='o')
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.grid(True)

    ax.axvline(0,color='red', linestyle='--')

    
def plot_image(fig,ax,A,mask,title='aa'):
    Am=A.copy()
    Am[~mask]=0
 
    vmin = Am.min()-0.3
    vmax = Am.max()+0.3
    print('mm',vmin,vmax,title)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im1=ax.imshow(Am, cmap='bwr', norm=norm)  # 'RdBu_r'

    masked_values = Am[mask]
    vmin = masked_values.min()
    vmax = masked_values.max()
    
    title='%s n=%d'%(title,np.sum(mask))
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
    ax.plot(lims, lims, '--', color='m', linewidth=1, label='y=x')
    if  only45: return
    # Vertical x=0
    ax.axvline(0, linestyle='--', color='k', linewidth=1)
    # Horizontal y=0
    ax.axhline(0, linestyle='--', color='k', linewidth=1)

    
def plot_correl_offdiag(fig,ax,At,Af,tmask,fmask,title='aa2'):    
    FP = ~tmask &  fmask  # False Positive: predicted True, actually False
    TP =  tmask &  fmask  # True Positive: predicted True, actually True
    TN = ~tmask & ~fmask  # True Negative: predicted False, actually False
    FN = tmask & ~fmask  # False Negative: predicted False, actually True

    ax.scatter(At[TP], Af[TP], alpha=0.6, color='green',label='TP: %d'%np.sum(TP))
    ax.scatter(At[FN], Af[FN], alpha=0.6, color='red',label='FN: %d'%np.sum(FN))
    ax.scatter(At[FP], Af[FP], alpha=0.6, color='orange',label='FP: %d'%np.sum(FP))

    ax.set(aspect=1. ,xlabel='true weight',ylabel='fitted',title=title)
    ax.grid(True, alpha=0.5)

    add_x45_lins(ax)
    if np.any(TP):  # Only plot if there are TP points
        x_cg = np.mean(At[TP])
        y_cg = np.mean(Af[TP])
        ax.scatter(x_cg, y_cg, marker='+', s=200, color='k', linewidths=2, label='TP center')
    ax.legend()

def plot_correl_diag(fig,ax,Bt,Bf,exc_mask,title='aa2'):
    #print('rr',Bt.shape())
    BtFl=Bt.flatten()
    BfFl=Bf.flatten()
    #print('rr2',BtFl.shape())
    ax.scatter(BtFl[exc_mask], BfFl[exc_mask], alpha=0.6, color='salmon',label='exc: %d'%np.sum(exc_mask))
    ax.scatter(BtFl[~exc_mask], BfFl[~exc_mask], alpha=0.6, color='blue',label='inh: %d'%np.sum(~exc_mask))
    add_x45_lins(ax, only45=True)
    ax.set(aspect=1. ,xlabel='true value',ylabel='fitted',title=title)
    ax.grid(True, alpha=0.5)
    ax.legend()

def plot_histo(ax,At,Af,title='aa2'):
    resV = At-Af
    ax.hist(resV, bins=30, alpha=0.7, color='red')
    ax.set(xlabel='Residual (Est - True)', ylabel='Count',
        title=title)
    ax.grid(True, alpha=0.3)

    if len(resV) > 0:
        mean_val = np.mean(resV)
        std_val = np.std(resV)
            
        # Draw vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            
        # Draw circle with error bar at half height
        ylim = ax.get_ylim()
        y_pos = (ylim[0] + ylim[1]) / 2

        # Draw horizontal error bar
        ax.errorbar(mean_val, y_pos, xerr=std_val, fmt='o',
                    markersize=8, color='black', capsize=5, capthick=2, linewidth=2)
        
        # Add text with mean and std, and sigma/x0 ratio if x0 is provided
       
        # \nσ/|x0|={sigma_x0_ratio:.3f}
        ax.text(0.05, 0.92, f'μ={mean_val:.3f}\nσ={std_val:.3f}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
    
#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)         

#...!...!..................
    def correl_after_thresh(self, bigD, maskD,md, figId=1):
        pprint(md)
        fmd=md['fit']
        
        figId=self.smart_append(figId)        
        nrow,ncol=2,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(15,8))

        # Unpack arrays from bigD
        A_true = bigD['A_true']
        A_fit = bigD['A_fit']
        B_true = bigD['B_true']
        B_fit = bigD['B_fit']
        train_losses = bigD['train_losses']
        val_losses = bigD['val_losses']
                
        # md["short_name"]
        
        maskT=maskD['true']
        maskF=maskD['fit']
        maskG=maskD['geom']
       
        for j,ntype in enumerate(['exc','inh']):
            tmask=maskT[ntype]
            fmask=maskF[ntype]
            ax = self.plt.subplot(nrow,ncol,1+j)
            title = 'fit (%s)' % ntype           
            plot_image(fig,ax,A_fit,fmask,title=title)

            ax = self.plt.subplot(nrow,ncol,1+ncol+j)
            plot_correl_offdiag(fig,ax,A_true,A_fit,tmask,fmask,title=title)

        
        # ... diagonal
        exc_1d=maskG['exc_1d']
        gmask=maskG['diag']  # fmask is the same as tmask
        title = 'fit (diagonal)'
        ax = self.plt.subplot(nrow,ncol,3+ncol)
        
        plot_correl_diag(fig,ax,A_true[gmask],A_fit[gmask],exc_1d,title=title)
    
        # ...  B-term
        title = 'fit (B-term)'
        ax = self.plt.subplot(nrow,ncol,4+ncol)        
        plot_correl_diag(fig,ax,B_true,B_fit,exc_1d,title=title)

        ''' use it later
        gmask=maskG['diag']  # fmask is the same as tmask
        plot_histo(ax,A_true[gmask],A_fit[gmask],title=title)
        '''
            
        #...... Training curves
        ax = self.plt.subplot(nrow,ncol,3)
        epoch0=3 # skip intial loss values
        epochs = np.arange(len(train_losses))
        ax.plot(epochs[epoch0:], train_losses[epoch0:], label='Train', color='blue', linestyle='-')
        ax.plot(epochs[epoch0:], val_losses[epoch0:], label='Val', color='blue', linestyle='--')

        ax.set_xlim(0,)  # x-axis starts at 0
 
        # Create title with number of samples if available
        title = 'Training curves'
        
        title += f' (N={fmd["num_samples"]})'
        ax.set(xlabel='Epoch', ylabel='Loss', title=title)
        
        ax.legend()
        ax.grid(True, alpha=0.3)


#...!...!..................
    def daleA_and_eigen(self, bigD,maskD, md, figId=1):
        pprint(md)
        fmd=md['fit']
        maskF=maskD['fit']
        
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
        ax.set_title('Fit spikes, Dale, targ_dens=%.2f'%(md['post']['target_density']))
        ax = self.plt.subplot(nrow,ncol,2+nrow)
        Eigen=np.linalg.eigvals(A_pass)
        plot_dale_eigen(ax,Eigen)
        ax.set_title('Fit eigen values')
        
