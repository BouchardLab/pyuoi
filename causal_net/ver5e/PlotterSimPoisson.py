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
        nrow,ncol=1,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(15,4))        
        data_name=md['short_name']
        dmd=md['dale_conf']
        numExc=dmd['num_excite']
        numNeur=dmd['num_neurons']
        step_size=md['evol_conf']['step_size']
        
        A=trueD['A_true']     
        single_rates = spikeD['single_rates']
                  
        m_diag=trueD['mask.geom.diag']
        m_exc=trueD['mask.true.exc']
        m_inh=trueD['mask.true.inh']

        #....   sorted weights
        ax = self.plt.subplot(nrow,ncol,1)
        binX=30
        ax.hist(A[m_exc], bins=binX, color='red', alpha=0.7, edgecolor=None,label='exc:%d'%np.sum(m_exc))
        ax.hist(A[m_inh], bins=binX, color='blue', alpha=0.7, edgecolor=None,label='inh:%d'%np.sum(m_inh))
       #1 ax.hist(A[m_diag], bins=binX, color='green', alpha=0.7, edgecolor=None,label='diag:%d'%np.sum(m_diag))

        #ax.set_yscale('log')
        ax.legend(loc='upper left')
        tit='True Dale, M%d, %s'%(A.shape[0],md['short_name'])
        ax.set(title=tit, xlabel='Weight value')

        ax.axvline(0,color='k',ls='--')
        ax.grid(True, alpha=0.3)
        
        #.... : rho_true vs neuron index
        ax = self.plt.subplot(nrow,ncol,2)
        edgeTV = trueD['edge_cnt_true']
        x_vals = np.arange(numNeur)
        
        # Create the filled step plot
        ax.fill_between(x_vals, edgeTV, step='mid', color='salmon', alpha=0.7)
                
        ax.set_xlabel('Neuron index')
        ax.set_ylabel('num true edges')
        ax.set_ylim(0,)
        ax.grid(True, alpha=0.3)
        probLo, probHi = dmd['edge_prob']
        rho_title = f'outgoing edges, true, edge_prob=[{probLo:.2f}, {probHi:.2f}]'
        ax.set_title(rho_title)

        #.... : firing rates
        ax = self.plt.subplot(nrow,ncol,3)
        x_vals = np.arange(numNeur)
        ax.fill_between(x_vals, single_rates, step='mid', color='darkviolet', alpha=0.7)
        ax.set_xlabel('Neuron index')
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_ylim(0,)
        ax.grid(True, alpha=0.3)
        ax.set_title('Single Neuron Firing Rates')

        #.... : histogram of rates
        ax = self.plt.subplot(nrow,ncol,4)
        ax.hist(single_rates, bins=20, log=True)
        x_vals = np.arange(numNeur)
        ax.set_xlabel('num neurons')
        ax.set_xlabel('Firing rate (Hz)')
        ax.grid(True, alpha=0.3)
        ax.set_title('Single rates spectrum')
        median_val = np.median(single_rates)
        ax.axvline(median_val, color='r', linestyle='--', linewidth=1.5)
        y_max = ax.get_ylim()[1]
        median_text = f"median: {median_val:.2f} (Hz), N={single_rates.shape[0]}"
        ax.text( x=median_val * 1.1,  y=y_max * 0.7, s=median_text,  color='red')

 

#............................
#............................
#............................
