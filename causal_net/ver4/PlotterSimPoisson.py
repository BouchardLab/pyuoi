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
def xplot_dale_matrix(fig,ax,W):
    a=0
    
#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

#...!...!..................
    def Dale_matrix_and_eigen(self,W,md,figId=3):
        
        figId=self.smart_append(figId)        
        nrow,ncol=1,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,5))

        #.... left ......
        ax = self.plt.subplot(nrow,ncol,1)
        normMap = colors.TwoSlopeNorm(vmin=W.min(), vcenter=0, vmax=W.max())
    
        im=ax.imshow(W, aspect=1., origin='upper', cmap='bwr', norm=normMap, interpolation='nearest')
        ax.set( xlabel='presyn. node index, source', ylabel='postsyn. node index, target')

        ax.set_aspect(1.0)
        ax.grid()
        # Create the colorbar.
        cbar = fig.colorbar(im, ax=ax, extend="both")
        
        tit='True Dale, M%d, %s'%(W.shape[0],md['short_name'])
        ax.set(title=tit)
        numExc=md['num_excit_neur']
        ax.axhline(numExc-0.5,color='k',ls='--')
        
        # Calculate non-zero weight counts (off-diagonal only)
        num_inhib = W.shape[0] - numExc
        
        # Create diagonal mask
        diag_mask = np.eye(W.shape[0], dtype=bool)
        off_diag_mask = ~diag_mask
        
        # Excitatory weights (rows 0 to num_excite-1, off-diagonal only)
        excit_weights = W[:numExc, :]
        excit_off_diag = excit_weights[off_diag_mask[:numExc, :]]
        excit_nonzero = np.sum(np.abs(excit_off_diag) > 1e-10)
        excit_total = excit_off_diag.size
        
        # Inhibitory weights (rows num_excite to num_neurons-1, off-diagonal only)
        inhib_weights = W[numExc:, :]
        inhib_off_diag = inhib_weights[off_diag_mask[numExc:, :]]
        inhib_nonzero = np.sum(np.abs(inhib_off_diag) > 1e-10)
        inhib_total = inhib_off_diag.size
        
        ax.text(0.2, 0.90, f'Excitatory ({excit_nonzero}/{excit_total})', size=18,color='r',transform=ax.transAxes)
        ax.text(0.2, 0.12, f'Inhibitory ({inhib_nonzero}/{inhib_total})', size=18,color='b',transform=ax.transAxes)
        #..... right......
        ax = self.plt.subplot(nrow,ncol,2)
        Eigen=np.linalg.eigvals(W)
        real_parts = np.real(Eigen)
        imag_parts = np.imag(Eigen)
        ax.scatter(real_parts, imag_parts, color='blue', marker='o')
        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")
        ax.set_title(tit)
        ax.axhline(0, color='black', lw=0.5)
        ax.axvline(0, color='black', lw=0.5)
        ax.grid(True)
    
        ax.axvline(0,color='red', linestyle='--')

        
#...!...!..................
    def histo_true_weights(self,W1,md,figId=3):        
        figId=self.smart_append(figId)        
        nrow,ncol=1,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,3))

        data_name=md['short_name']
        # Prepare weight data
        nnAny = W1.shape[0]
        nnExcit = md['num_excit_neur']
        nnInhib = nnAny - nnExcit
    
        # Use all weights (including diagonal)
        # Mask excitatory and inhibitory columns (outgoing connections)
        W_excit = W1[:nnExcit, :].flatten()  # From excitatory neurons
        W_inhib = W1[nnExcit:, :].flatten()  # From inhibitory neurons
    
        # Skip 0's
        W_excit = W_excit[W_excit != 0]
        W_inhib = W_inhib[W_inhib != 0]
    
        #.... left: Excitatory weights
        ax = self.plt.subplot(nrow,ncol,1)
        ax.hist(W_excit, bins=50, color='tab:red', alpha=0.7, edgecolor='black')
        ax.set_title(data_name + ' Excitatory Weights')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.text(0.1, 0.6, 'diagonal', transform=ax.transAxes, rotation=45, fontsize=10)
        ax.text(0.7, 0.8, 'off-diagonal', transform=ax.transAxes, fontsize=10)

        # .... right: Inhibitory weights
        ax = self.plt.subplot(nrow,ncol,2)
        ax.hist(W_inhib, bins=50, color='tab:blue', alpha=0.7, edgecolor='black')
        ax.set_title('Inhibitory Weights')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.text(0.1, 0.6, 'diagonal', transform=ax.transAxes, rotation=45, fontsize=10)
        ax.text(0.4, 0.8, 'off-diagonal', transform=ax.transAxes, fontsize=10)

#...!...!..................
    def rate_analysis(self, firing_rates, firing_rate_errors, coincidence_rates, coincidence_rate_errors, md, figId=3):
        """Plot firing rates and coincidence rates analysis."""
        figId=self.smart_append(figId)        
        nrow,ncol=2,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,7))

        n_neurons = len(firing_rates)
        
        # 1D plot: Firing rates per neuron
        ax = self.plt.subplot(nrow,ncol,1)
        neuron_indices = np.arange(n_neurons)
        ax.errorbar(neuron_indices, firing_rates, yerr=firing_rate_errors, 
                   fmt='o-', capsize=3, capthick=1, markersize=4)
        ax.set_xlabel('Neuron index')
        ax.set_ylabel('Firing rate (Hz)')
        data_name = md['short_name']
        ax.set_title(f'Single Neuron Firing Rates ({data_name})')
        ax.grid(True, alpha=0.3)
        numExc=md['num_excit_neur']
        ax.axvline(numExc-0.5,color='k',ls='--')
        
        # 2D plot: Coincidence rates matrix
        ax = self.plt.subplot(nrow,ncol,3)
        im = ax.imshow(coincidence_rates, cmap='viridis', aspect=1.)
        ax.set_xlabel('To neuron')
        ax.set_ylabel('From neuron')
        ax.set_title('Pairwise Coincidence Rates (Hz)')
        fig.colorbar(im, ax=ax)
        ax.grid(True, alpha=0.3)
        
        # 2D plot: Coincidence rate errors matrix
        ax = self.plt.subplot(nrow,ncol,4)
        im2 = ax.imshow(coincidence_rate_errors, cmap='plasma', aspect=1.)
        ax.set_xlabel('To neuron')
        ax.set_ylabel('From neuron')
        ax.set_title('Coincidence Rate Standard Errors (Hz)')
        fig.colorbar(im2, ax=ax)
        ax.grid(True, alpha=0.3)
        
        # Scatter plot: Firing rate vs coincidence rate (off-diagonal only)
        ax = self.plt.subplot(nrow,ncol,2)
        off_diag_mask = ~np.eye(n_neurons, dtype=bool)
        firing_rates_from = np.outer(firing_rates, np.ones(n_neurons))[off_diag_mask]
        firing_rates_to = np.outer(np.ones(n_neurons), firing_rates)[off_diag_mask]
        coincidence_rates_off_diag = coincidence_rates[off_diag_mask]
        
        # Expected coincidence rate under independence
        dt = 0.01  # Default time bin size
        expected_coincidence = firing_rates_from * firing_rates_to * dt
        
        ax.scatter(expected_coincidence, coincidence_rates_off_diag, alpha=0.6, s=20)
        ax.plot([0, np.max(expected_coincidence)], [0, np.max(expected_coincidence)], 'r--', alpha=0.7)
        ax.set_xlabel('Expected coincidence rate (independence)')
        ax.set_ylabel('Observed coincidence rate')
        ax.set_title('Coincidence vs Independence')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')  # Fix aspect ratio to 1:1
        
        # Add statistics
        correlation = np.corrcoef(expected_coincidence, coincidence_rates_off_diag)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
