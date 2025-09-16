#!/usr/bin/env python3
"""
Plotting utilities for simulated Poisson process visualization.

This module provides specialized plotting capabilities for analyzing
simulated Dale's principle neural networks. The Plotter class extends
PlotterBackbone to create visualizations including:
- Dale connectivity matrix plots with excitatory/inhibitory separation
- Eigenvalue analysis and network stability visualization
- Poisson process statistics and firing rate distributions
- Network dynamics and temporal evolution plots

Designed specifically for validating and analyzing simulated neural
networks that follow Dale's principle with Poisson spiking dynamics.
"""

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
    def Dale_matrix_and_eigen(self,W_freq,md,trueD,figId=3):
        
        figId=self.smart_append(figId)        
        nrow,ncol=1,3
        fig=self.plt.figure(figId,facecolor='white', figsize=(15,3.5))

        dmd=md['dale_conf']
        numExc=dmd['num_excite']
        numNeur=dmd['num_neurons']
        
        # Reconstruct natural-order matrix from frequency-sorted matrix
        neur_freqIdx = trueD['neur_freqIdx']  # natural_index → freq_sorted_position
        W_natural = W_freq[np.ix_(neur_freqIdx, neur_freqIdx)]
        
        # Common normalization for both plots
        vmin = min(W_freq.min(), W_natural.min())
        vmax = max(W_freq.max(), W_natural.max())
        normMap = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        #.... Position 1: Natural order Dale matrix ......
        ax = self.plt.subplot(nrow,ncol,1)
        im=ax.imshow(W_natural, aspect=1., origin='lower', cmap='bwr', norm=normMap, interpolation='nearest')
        ax.set( xlabel='presyn. neuron index (natural idx)', ylabel='postsyn. neuron index (natural idx)')
        ax.set_aspect(1.0)
        ax.grid()
        cbar = fig.colorbar(im, ax=ax, extend="both", shrink=0.5)
        
        tit_natural='True Dale (natural), M%d, %s'%(W_natural.shape[0],md['short_name'])
        ax.set(title=tit_natural)
        ax.axhline(numExc-0.5,color='k',ls='--', label='E/I boundary')
        ax.axvline(numExc-0.5,color='k',ls='--')
        
        #.... Position 2: Frequency-sorted Dale matrix ......
        ax = self.plt.subplot(nrow,ncol,2)
        im=ax.imshow(W_freq, aspect=1., origin='lower', cmap='bwr', norm=normMap, interpolation='nearest')
        ax.set( xlabel='presyn. neuron index (freq-sorted)', ylabel='postsyn. neuron index (freq-sorted)')
        ax.set_aspect(1.0)
        ax.grid()
        cbar = fig.colorbar(im, ax=ax, extend="both", shrink=0.7)

        tit_freq='True Dale (freq-sorted), M%d, %s'%(W_freq.shape[0],md['short_name'])
        ax.set(title=tit_freq)
        # Note: In frequency-sorted order, excitatory/inhibitory neurons are mixed, so no simple boundary line
 
        #..... Position 3: Eigenvalues......
        ax = self.plt.subplot(nrow,ncol,3)
        Eigen=np.linalg.eigvals(W_freq)  # Use freq-sorted matrix for eigenvalues
        real_parts = np.real(Eigen)
        imag_parts = np.imag(Eigen)
        ax.scatter(real_parts, imag_parts, color='blue', marker='o')
        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")
        tit3='Eigenvalues, M%d, %s'%(W_freq.shape[0],md['short_name'])
        ax.set_title(tit3)
        ax.axhline(0, color='black', lw=0.5)
        ax.axvline(0, color='black', lw=0.5)
        ax.axvline(0,color='red', linestyle='--')
        ax.grid(True)
      
        
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

        # Data is now stored in frequency-sorted order by default
        single_rates = spikeD['single_rates']  # Already in freq-sorted order
        edgeTV = trueD['edge_cnt_true']        # Already in freq-sorted order
        m_diagA = trueD['mask.geom.diagA']     # Already in freq-sorted order
        m_excA = trueD['mask.true.excA']       # Already in freq-sorted order
        m_inhA = trueD['mask.true.inhA']       # Already in freq-sorted order
        m_inh1d = trueD['mask.geom.inh_idx']   # Already in freq-sorted order
        
        #....   weights histogram (always use freq-sorted data)
        ax = self.plt.subplot(nrow,ncol,1)
        binX=30
        ax.hist(A[m_excA], bins=binX, color='red', alpha=0.7, edgecolor=None,label='exc:%d'%np.sum(m_excA))
        ax.hist(A[m_inhA], bins=binX, color='blue', alpha=0.7, edgecolor=None,label='inh:%d'%np.sum(m_inhA))

        ax.legend(loc='upper left')
        tit='True Dale, M%d, %s'%(A.shape[0],md['short_name'])
        ax.set(title=tit, xlabel='Weight value',ylabel='num edges')
        ax.axvline(0,color='k',ls='--')
        ax.grid(True, alpha=0.3)
        
        # Prepare data for neuron-indexed plots based on display order
        if byFreq:
            # Display data in frequency-sorted order (as stored)
            display_single_rates = single_rates
            display_edgeTV = edgeTV
            display_inh_mask = m_inh1d
            neurXlab = 'freq sorted neurons index'
        else:
            # Convert to natural neuron order for display
            neur_revFreqIdx = trueD['neur_revFreqIdx']  # freq_sorted_position → natural_index
            
            # Create arrays in natural order
            display_single_rates = np.zeros_like(single_rates)
            display_edgeTV = np.zeros_like(edgeTV)
            
            # Map firing rates and edge counts from freq-sorted back to natural order
            display_single_rates[neur_revFreqIdx] = single_rates
            display_edgeTV[neur_revFreqIdx] = edgeTV
            
            # For natural order, create inhibitory mask based on original neuron types
            # First num_excite neurons are excitatory, rest are inhibitory
            display_inh_mask = np.zeros(numNeur, dtype=bool)
            display_inh_mask[numExc:] = True  # Inhibitory neurons start at index numExc
            neurXlab = 'natural indexed neurons'
        
        x_vals = np.arange(numNeur)

        #.... : histogram of rates
        ax = self.plt.subplot(nrow,ncol,2)
        yLog= md['evol_conf']['expRate'] 
        ax.hist(single_rates, bins=20)#, log=yLog)
        x_vals = np.arange(numNeur)
        ax.set_xlabel('num neurons')
        ax.set_xlabel('Firing rate (Hz)')
        ax.set_ylabel('num neurons')
        ax.grid(True, alpha=0.3)
        ax.set_title('Single rates spectrum')
        ax.set_xlim(0,)
        median_val = np.median(single_rates)
        ax.axvline(median_val, color='r', linestyle='--', linewidth=1.5)
        y_max = ax.get_ylim()[1]
        median_text = f"median: {median_val:.2f} (Hz)\n N={single_rates.shape[0]}"
        ax.text( x=median_val * 1.1,  y=y_max * 0.7, s=median_text,  color='red')

        
        #.... : rho_true vs neuron index
        ax = self.plt.subplot(nrow,ncol,3)
        ax.fill_between(x_vals, display_edgeTV, step='mid', color='salmon', alpha=0.7)
        ax.set_xlabel(neurXlab)
        ax.set_ylabel('num true edges')
        ax.set_ylim(0,)
        ax.grid(True, alpha=0.3)
        probLo, probHi = dmd['edge_prob']
        rho_title = f'outgoing edges, true, prob=[{probLo:.2f}, {probHi:.2f}]'
        ax.set_title(rho_title)

        #....  firing rates ..... 
        ax = self.plt.subplot(nrow,ncol,4)
        chanW=0.9        
        # Create masks for inhibitory and excitatory neurons
        inh_mask_display = display_inh_mask
        exc_mask_display = ~display_inh_mask
        
        ax.bar(x_vals[inh_mask_display], display_single_rates[inh_mask_display], width=chanW, color='blue', align='center', alpha=0.7, label='Inhibitory')
                
        ax.bar(x_vals[exc_mask_display], display_single_rates[exc_mask_display], width=chanW, color='red', align='center', alpha=0.7, label='Excitatory')
        
        ax.set_xlabel(neurXlab)
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_ylim(0,)
        ax.grid(True, alpha=0.3)
        ax.set_title('Single Neuron Firing Rates')
        ax.legend()
 

#............................
#............................
#............................
