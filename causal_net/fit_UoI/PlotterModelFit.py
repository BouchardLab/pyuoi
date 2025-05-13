__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from toolbox.PlotterBackbone import PlotterBackbone
from matplotlib import cm as cmap
import matplotlib.ticker as ticker
from pprint import pprint
import numpy as np
import matplotlib.gridspec as gridspec

from matplotlib.colors import LinearSegmentedColormap
    
#...!...!....................
def summary_column(md):
    #pprint(md)
    pmd=md['payload']
    sem=md['selector']
    
    txt=md['short_name']
    txt+='\ninput '+sem['input_name']
    txt+='\nsampFreq %d Hz'%(sem['sampling_freq'])
    #txt+='\ndecay:%d ms,  len:%d ms '%(pmd['tau_decay'][0]*1000., pmd['tau_decay'][1]*1000.)
    txt+='\nsel time [%.1f %.1f] s'%(sem['time_range'][0],sem['time_range'][1])
    txt+='\nsel features %d'%(sem['num_feature'])
    
    return txt

    
#...!...!....................
def plot_diagonal_and_violins(A,plt,figId,tit0, eps=1e-5):
    """
    Plots the diagonal elements of the matrix A as a 1D line plot (top row) and 
    vertical violin plots for each row (bottom row), excluding diagonal elements 
    and values with amplitude smaller than eps.

    Parameters:
        A (2D sparse matrix or array): Sparse matrix of size (nfeat, nfeat).
        eps (float): Minimum value threshold to include in violin plots.
    """
    nfeat = A.shape[0]

    # Create the canvas with two rows
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 2]},   num=figId)
    
    
    # Top Row: Plotting the diagonal elements
    diagV = np.diag(A)
    ax1.plot(diagV, marker='o', linestyle='-', color='blue', label='Diagonal Elements')
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Diagonal Value')
    ax1.set_title('Diagonal Elements of A')
    ax1.grid(True)
    ax1.set_ylim(0,1.1*max(diagV))
    ax1.set_xlim(-0.5,nfeat+0.5)
    ax1.set_title('%s   Auto-correlation' % (tit0))

    # Customizing x-axis
    tickL=5
    x_ticks = np.arange(0, nfeat, tickL)
    ax1.set_xticks(x_ticks)
    
    # Bottom Row: Vertical violin plots
    violin_data = []
    for i in range(nfeat):
        # Exclude diagonal and values with amplitude smaller than eps
        row_values = np.delete(A[i, :], i)  # Exclude diagonal element
        row_values = row_values[np.abs(row_values) >= eps]  # Filter values by amplitude
        if len(row_values) > 0:
            violin_data.append(row_values)
        else:
            violin_data.append([0])  # Add a placeholder to maintain alignment

    # Plotting vertical violin plots
    ax2.violinplot(violin_data, showmeans=True, showmedians=True)

    # Customize the x-axis and labels
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Off-diagonal Values')
    ax2.set_title('Vertical Violin Plots for Each Row of A')
    ax2.grid(True)
    ax2.set_xlim(-0.5,nfeat+0.5)
    ax2.set_xticks(x_ticks)
    ax2.axhline(0,lw=1.,ls='--',c='k')
    return ax2

 
#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

#...!...!..................
    def A_matrix(self,bigD,md,figId=3,lag=0):

        pmd=md['payload']
        tit=md['short_name']+' M-matrix[lag=%d]'%(lag)

        figId=self.smart_append(figId)        
        nrow,ncol=1,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,7))
        ax = self.plt.subplot(nrow,ncol,1)

        A0=bigD['fit_A_model'][lag]        
        nfeat = A0.shape[0]
        A=np.copy(A0)
        # Set diagonal values of A to 0
        np.fill_diagonal(A, 0)
        
        # Find the maximum absolute value in A
        max_val = np.max(np.abs(A))/2.
        
        # Plot the sparse matrix using imshow
        im = ax.imshow(A.T, aspect='auto', origin='lower', cmap='bwr', vmin=-max_val, vmax=max_val)
        ax.grid()
        ax.plot([0,nfeat],[0,nfeat],'--',lw=0.5)
        ax.set_xlim(-0.5,nfeat+0.5)
        ax.set_ylim(-0.5,nfeat+0.5)
        ax.set_aspect(1.0)

        # Create the colorbar.
        cbar = fig.colorbar(im, ax=ax, extend="both")
        cbar.set_label('UoI coupling strength')
        
        tit='UoI fit matrix, %d neurons, name=%s'%(nfeat,md['short_name'])
        ax.set(title=tit, xlabel='presyn. node index, source', ylabel='postsyn. node index, target')
        
#...!...!..................
    def Aper_row(self,bigD,md,figId=3,lag=0):
        figId=self.smart_append(figId)
        pmd=md['payload']
        tit=md['short_name']
        tit=md['short_name']
        
        AV=bigD['fit_A_model']

        ax=plot_diagonal_and_violins( AV[lag],self.plt,figId,tit)
        
        txt=summary_column(md)
        ax.text(0.6, 0.95, txt, fontsize=10, color='blue', ha='left', va='top',transform=ax.transAxes)
        
