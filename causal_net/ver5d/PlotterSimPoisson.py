__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from toolbox.PlotterBackbone import PlotterBackbone
from matplotlib import cm as cmap
import matplotlib.ticker as ticker
from pprint import pprint
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
    
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

        dmd=md['dale_conf']
        numExc=dmd['num_excite']
        numNeur=dmd['num_neurons']
        
        #.... left ......
        ax = self.plt.subplot(nrow,ncol,1)
        normMap = colors.TwoSlopeNorm(vmin=W.min(), vcenter=0, vmax=W.max())

        im=ax.imshow(W, aspect=1., origin='lower', cmap='bwr', norm=normMap, interpolation='nearest')
        ax.set( xlabel='presyn. node index, source', ylabel='postsyn. node index, target')

        ax.set_aspect(1.0)
        ax.grid()
        # Create the colorbar.
        cbar = fig.colorbar(im, ax=ax, extend="both")

        tit='True Dale, M%d, %s'%(W.shape[0],md['short_name'])
        ax.set(title=tit)
        ax.axhline(numExc-0.5,color='k',ls='--')
 
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

        #.... right 
        ax = self.plt.subplot(nrow,ncol,2)
      
        
#...!...!..................
    def histo_weights_rates(self,trueD,spikeD,md,figId=3):        
        figId=self.smart_append(figId)        
        nrow,ncol=2,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,8))        
        data_name=md['short_name']
        dmd=md['dale_conf']
        numExc=dmd['num_excite']
        numNeur=dmd['num_neurons']
        step_size=md['evol_conf']['step_size']
        
        A=trueD['A_true']     
        single_rates = spikeD['single_rates']
        coincidence_rates = spikeD['coincidence_rates']
              
        m_diag=trueD['mask.geom.diag']
        m_exc=trueD['mask.true.exc']
        m_inh=trueD['mask.true.inh']

        #.... subplot(2,2,1):  sorted weights
        ax = self.plt.subplot(nrow,ncol,1)
        binX=30
        ax.hist(A[m_exc], bins=binX, color='red', alpha=0.7, edgecolor=None,label='exc:%d'%np.sum(m_exc))
        ax.hist(A[m_inh], bins=binX, color='blue', alpha=0.7, edgecolor=None,label='inh:%d'%np.sum(m_inh))
        ax.hist(A[m_diag], bins=binX, color='green', alpha=0.7, edgecolor=None,label='diag:%d'%np.sum(m_diag))

        ax.set_yscale('log')
        ax.legend(loc='upper left')
        tit='True Dale, M%d, %s'%(A.shape[0],md['short_name'])
        ax.set(title=tit, xlabel='Weight value')

        ax.axvline(0,color='k',ls='--')
        ax.grid(True, alpha=0.3)
        
        #.... : firing rates
        ax = self.plt.subplot(nrow,ncol,2)
        x_vals = np.arange(numNeur)
        ax.fill_between(x_vals, single_rates, step='mid', color='darkviolet', alpha=0.7)
        ax.set_xlabel('Neuron index')
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_ylim(0,)
        ax.grid(True, alpha=0.3)
        ax.set_title('Single Neuron Firing Rates')

        #.... : rho_true vs neuron index
        ax = self.plt.subplot(nrow,ncol,4)
        rho_true = trueD['rho_true']
        x_vals = np.arange(numNeur)
        
        # Create the filled step plot
        ax.fill_between(x_vals, rho_true, step='mid', color='salmon', alpha=0.7)
        #1ax.plot(x_vals, rho_true, drawstyle='steps-mid', color='black', lw=0.5)
                
        ax.set_xlabel('Neuron index')
        ax.set_ylabel('num true edges')
        ax.set_ylim(0,)
        ax.grid(True, alpha=0.3)
        probLo, probHi = dmd['edge_prob']
        rho_title = f'edgeCount_true, edge_prob=[{probLo:.2f}, {probHi:.2f}]'
        ax.set_title(rho_title)

         # Scatter plot: Firing rate vs coincidence rate (off-diagonal only)
        ax = self.plt.subplot(nrow,ncol,3)
        off_diag_mask = ~np.eye(numNeur, dtype=bool)
        firing_rates_from = np.outer(single_rates, np.ones(numNeur))[off_diag_mask]
        firing_rates_to = np.outer(np.ones(numNeur), single_rates)[off_diag_mask]
        coincidence_rates_off_diag = coincidence_rates[off_diag_mask]

        # Expected coincidence rate under independence
        plot_coincidence_vs_independence_with_power_fit(
            ax,
            firing_rates_from,
            firing_rates_to,
            coincidence_rates_off_diag,
            dt=step_size
        )



#............................
#............................
#............................

#...!...!..................
def plot_coincidence_vs_independence_with_power_fit(
        ax,
        firing_rates_from,
        firing_rates_to,
        coincidence_rates_off_diag,
        dt
    ):
    """
    Plots expected vs observed coincidence rates and fits a power function.

    Parameters:
        ax: matplotlib.axes.Axes object to plot on
        firing_rates_from: np.ndarray, firing rates of 'from' population
        firing_rates_to: np.ndarray, firing rates of 'to' population
        coincidence_rates_off_diag: np.ndarray, observed coincidence rates (off-diagonal)
        dt: float, time bin size 
    """
    # Expected coincidence rate under independence
    expected_coincidence = firing_rates_from * firing_rates_to * dt

    # Scatter plot
    ax.scatter(expected_coincidence, coincidence_rates_off_diag, alpha=0.6, s=20)

    
    # Identity line
    max_val = max(np.max(expected_coincidence), np.max(coincidence_rates_off_diag))
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.7)

 
  # Power function fit
    def power_func(x, a, b):
        return a * x**b

    # Only fit to positive x values to avoid issues with log(0)
    mask = expected_coincidence > 0
    x_fit = expected_coincidence[mask]
    y_fit = coincidence_rates_off_diag[mask]
    try:
        popt, pcov = curve_fit(power_func, x_fit, y_fit, p0=(1, 1), maxfev=5000)
        a_fit, b_fit = popt
        # Plot the fit
        x_curve = np.linspace(0, max_val, 200)
        y_curve = power_func(x_curve, a_fit, b_fit)
        ax.plot(x_curve, y_curve, color='red', lw=1, label=f'power fit: $y={a_fit:.2f}x^{{{b_fit:.2f}}}$')
        ax.legend()
    except Exception as e:
        print("Could not fit power function: %s" % e)
            
    # Labels and formatting
    ax.set_xlabel('Expected coincidence rate (independence)')
    ax.set_ylabel('Observed coincidence rate')
    ax.set_title('Coincidence vs Independence')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Add statistics
    correlation = np.corrcoef(expected_coincidence, coincidence_rates_off_diag)[0, 1]
    ax.text(0.05, 0.85, f'Correlation: {correlation:.3f}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top')
# Mark center of gravity
    x_mean = np.mean(expected_coincidence)
    y_mean = np.mean(coincidence_rates_off_diag)
    ax.plot(x_mean, y_mean, '+', color='lime', markersize=15, markeredgewidth=3)
