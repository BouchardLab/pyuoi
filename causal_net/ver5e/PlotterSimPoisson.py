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
    def histo_weights_rates(self,trueD,spikeD,md,byFreq=False,figId=3):        
        figId=self.smart_append(figId)        
        nrow,ncol=1,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(15,4))        
        data_name=md['short_name']
        dmd=md['dale_conf']
        numExc=dmd['num_excite']
        numNeur=dmd['num_neurons']
        step_size=md['evol_conf']['step_size']
                
        A=trueD['A_true']

        if byFreq:
            # Data is already frequency-sorted, display as-is
            neurIdx=np.arange(numNeur)
            neurXlab='freq sorted neurons index'
        else:
            # Reorder frequency-sorted data back to natural neuron order
            neur_natIdx = trueD['neur_natIdx']
            neurIdx = neur_natIdx  # Maps natural position → frequency-sorted position
            neurXlab='natural index neurons'
            
        single_rates = spikeD['single_rates']
                  
        m_diagA=trueD['mask.geom.diagA']
        m_excA=trueD['mask.true.excA']
        m_inhA=trueD['mask.true.inhA']
        m_inh1d=trueD['mask.geom.inh_idx']
        #print('iinn',np.sum(m_inh1d), m_inh1d.shape); aa 
        
        #....   sorted weights
        ax = self.plt.subplot(nrow,ncol,1)
        binX=30
        ax.hist(A[m_excA], bins=binX, color='red', alpha=0.7, edgecolor=None,label='exc:%d'%np.sum(m_excA))
        ax.hist(A[m_inhA], bins=binX, color='blue', alpha=0.7, edgecolor=None,label='inh:%d'%np.sum(m_inhA))
       #1 ax.hist(A[m_diag], bins=binX, color='green', alpha=0.7, edgecolor=None,label='diag:%d'%np.sum(m_diag))

        ax.legend(loc='upper left')
        tit='True Dale, M%d, %s'%(A.shape[0],md['short_name'])
        ax.set(title=tit, xlabel='Weight value')

        ax.axvline(0,color='k',ls='--')
        ax.grid(True, alpha=0.3)
        
        #.... : rho_true vs neuron index
        ax = self.plt.subplot(nrow,ncol,2)
        edgeTV = trueD['edge_cnt_true']
        x_vals = np.arange(numNeur)
        
        # Reorder edge counts based on byFreq flag (same logic as firing rates)
        sortedEdgeTV = edgeTV[neurIdx]
        
        #  plot edge count per neuron 
        ax.fill_between(x_vals, sortedEdgeTV, step='mid', color='salmon', alpha=0.7)
                
        ax.set_xlabel(neurXlab)
        ax.set_ylabel('num true edges')
        ax.set_ylim(0,)
        ax.grid(True, alpha=0.3)
        probLo, probHi = dmd['edge_prob']
        rho_title = f'outgoing edges, true, prob=[{probLo:.2f}, {probHi:.2f}]'
        ax.set_title(rho_title)

        #....  firing rates ..... 
        ax = self.plt.subplot(nrow,ncol,3)
        chanW=0.9

        # this is to complicated for a human, but seems to work
        sortSR=single_rates[neurIdx]
        
        # Create masks for inhibitory and excitatory neurons in sorted order
        inh_mask_sorted = m_inh1d[neurIdx]
        exc_mask_sorted = ~m_inh1d[neurIdx]
        
        # Plot inhibitory neurons in blue
        ax.bar(x_vals[inh_mask_sorted], sortSR[inh_mask_sorted], width=chanW, color='blue', align='center', alpha=0.7, label='Inhibitory')
                
        # Plot excitatory neurons in red
        ax.bar(x_vals[exc_mask_sorted], sortSR[exc_mask_sorted], width=chanW, color='red', align='center', alpha=0.7, label='Excitatory')
        
        ax.set_xlabel(neurXlab)
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_ylim(0,)
        ax.grid(True, alpha=0.3)
        ax.set_title('Single Neuron Firing Rates')
        ax.legend()

        #.... : histogram of rates
        ax = self.plt.subplot(nrow,ncol,4)
        yLog= md['evol_conf']['expRate'] 
        ax.hist(single_rates, bins=20, log=yLog)
        x_vals = np.arange(numNeur)
        ax.set_xlabel('num neurons')
        ax.set_xlabel('Firing rate (Hz)')
        ax.grid(True, alpha=0.3)
        ax.set_title('Single rates spectrum')
        ax.set_xlim(0,)
        median_val = np.median(single_rates)
        ax.axvline(median_val, color='r', linestyle='--', linewidth=1.5)
        y_max = ax.get_ylim()[1]
        median_text = f"median: {median_val:.2f} (Hz), N={single_rates.shape[0]}"
        ax.text( x=median_val * 1.1,  y=y_max * 0.7, s=median_text,  color='red')

 

#............................
#............................
#............................
