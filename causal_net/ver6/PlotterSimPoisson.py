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
  
        from UtilSelectFDR import get_offdiag_triplets
        # output:  np.column_stack([i_indices, j_indices, values])
        EposT=get_offdiag_triplets(A,isPos=True)
        EnegT=get_offdiag_triplets(A,isPos=False)
        print('True0 num edges  pos=%d  neg=%d'%(EposT.shape[0],EnegT.shape[0]))

        wMin=np.min(EnegT[:,2])
        wMax=np.max(EposT[:,2])
                
        def count_elements(E, Nn=numNeur):
            i_indices = E[:, 0].astype(int)
            counts = np.bincount(i_indices, minlength=Nn)
            return counts
        
        edgeCount=count_elements(EnegT)  + count_elements(EposT)
        m_inh1d=count_elements(EnegT)>0
        #print('num inh neur=',np.sum(m_inh1d), ' num inh edges:',np.sum(edgeCount))
        
        #....   weights histogram 
        ax = self.plt.subplot(nrow,ncol,1)
        binX= np.linspace(wMin, wMax, 100)
        ax.hist(EnegT[:,2], bins=binX, color='blue', alpha=0.7, edgecolor=None,label='neg:%d'%EnegT.shape[0])
        ax.hist(EposT[:,2], bins=binX, color='red', alpha=0.7, edgecolor=None,label='pos:%d'%EposT.shape[0])

        ax.legend(loc='upper left')
        tit='True Dale, M%d, %s'%(A.shape[0],md['short_name'])
        ax.set(title=tit, xlabel='True weight value',ylabel='num edges')
        ax.axvline(0,color='k',ls='--')
        ax.grid(True, alpha=0.3)

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

        # Prepare data for neuron-indexed plots based on display order
        if byFreq:
            # Display data in frequency-sorted order (as stored)
            display_single_rates = single_rates
            display_edgeTV = edgeCount
            display_inh_mask = m_inh1d
            neurXlab = 'freq sorted neurons index'
        else:
            # Convert to natural neuron order for display
            neur_revFreqIdx = trueD['neur_revFreqIdx']  # freq_sorted_position → natural_index
            
            # Create arrays in natural order
            display_single_rates = np.zeros_like(single_rates)
            display_edgeTV = np.zeros_like(edgeCount)
            
            # Map firing rates and edge counts from freq-sorted back to natural order
            display_single_rates[neur_revFreqIdx] = single_rates
            display_edgeTV[neur_revFreqIdx] = edgeCount
            
            # For natural order, create inhibitory mask based on original neuron types
            # First num_excite neurons are excitatory, rest are inhibitory
            display_inh_mask = np.zeros(numNeur, dtype=bool)
            display_inh_mask[numExc:] = True  # Inhibitory neurons start at index numExc
            neurXlab = 'natural indexed neurons'
        
        x_vals = np.arange(numNeur)         
        #....  edge count
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
               
        ax.bar(x_vals[exc_mask_display], display_single_rates[exc_mask_display], width=chanW, color='red', align='center', alpha=0.7, label='Excitatory')
        ax.bar(x_vals[inh_mask_display], display_single_rates[inh_mask_display], width=chanW, color='blue', align='center', alpha=0.7, label='Inhibitory')

        ax.set_xlabel(neurXlab)
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_ylim(0,)
        ax.grid(True, alpha=0.3)
        ax.set_title('Single Neuron Firing Rates')
        ax.legend()
 

#............................
#............................
#............................
    def rates_study(self,trueD,spikeD,md,figId=4):
        figId=self.smart_append(figId)
        nrow,ncol=1,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(15,4))
        data_name=md['short_name']
        dmd=md['dale_conf']
        numExc=dmd['num_excite']
        numNeur=dmd['num_neurons']
        
        # Data are frequency-sorted in primary storage
        B_idle = trueD['B_true']                 # freq-sorted bias
        single_rates = spikeD['single_rates']    # freq-sorted rates (Hz)
        single_rates_snr = spikeD['sigle_rates_snr']  # freq-sorted SNR (dimensionless)
        neur_revFreqIdx = trueD['neur_revFreqIdx']    # freq_sorted_position → natural_index
        
        # Build excitatory/inhibitory masks in freq-sorted order using natural index split
        nat_index = neur_revFreqIdx
        exc_mask = nat_index < numExc
        inh_mask = ~exc_mask
        
        # 1) Scatter: x=B_idle, y=log(single_rates)
        ax = self.plt.subplot(nrow,ncol,1)
        ax.scatter(B_idle[exc_mask], single_rates[exc_mask], s=14, alpha=0.6, marker='^', facecolors='none', edgecolors='red', label='Excitatory')
        ax.scatter(B_idle[inh_mask], single_rates[inh_mask], s=14, alpha=0.6, marker='^', facecolors='none', edgecolors='blue', label='Inhibitory')
        ax.set_xlabel('true B_idle ')
        ax.set_ylabel('single_rates (Hz)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        tit='True Dale, M%d, %s'%(numNeur, data_name)
        ax.set_title(tit)
       
        # reference line: y = exp(B), clipped to central 80% of B range
        bmin = float(np.min(B_idle))
        bmax = float(np.max(B_idle))
        bLo = bmin + 0.1 * (bmax - bmin)
        bHi = bmax - 0.1 * (bmax - bmin)
        bx = np.linspace(bLo, bHi, 100)
        by = np.exp(bx)
        ax.plot(bx, by, linestyle='--', color='black', linewidth=0.8, label='y=exp(x)')
        ax.legend()

        # 2) Scatter: x=B_idle, y=single SNR (from spikeD)
        ax = self.plt.subplot(nrow,ncol,2)
        ax.scatter(B_idle[exc_mask], single_rates_snr[exc_mask], s=14, alpha=0.6, facecolors='none', edgecolors='red', label='Excitatory')
        ax.scatter(B_idle[inh_mask], single_rates_snr[inh_mask], s=14, alpha=0.6, facecolors='none', edgecolors='blue', label='Inhibitory')
        ax.set_xlabel('true B_idle')
        ax.set_ylabel('single SNR (rate^2/var)')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_title('single SNR vs B_idle')
        ax.legend()
        
        # 3) Histogram: single_rates for excitatory
        ax3 = self.plt.subplot(nrow,ncol,3)
        exc_vals = single_rates[exc_mask]
        inh_vals = single_rates[inh_mask]
        # compute common bins and range (start at 0)
        exc_max = np.max(exc_vals) if exc_vals.size > 0 else 0.0
        inh_max = np.max(inh_vals) if inh_vals.size > 0 else 0.0
        x_max = max(exc_max, inh_max)
        if x_max <= 0:  x_max = 1.0
        #common_bins = np.linspace(0, x_max, 21)
        common_bins = np.linspace(0, x_max, int(2*x_max))
        ax3.hist(exc_vals, bins=common_bins, color='red', alpha=0.7, edgecolor=None)
        ax3.set_xlabel('single_rates (Hz)')
        ax3.set_ylabel('num neurons')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Rates: Excitatory (N=%d)' % (numExc))
        
        # 4) Histogram: single_rates for inhibitory
        ax4 = self.plt.subplot(nrow,ncol,4)
        ax4.hist(inh_vals, bins=common_bins, color='blue', alpha=0.7, edgecolor=None)
        ax4.set_xlabel('single_rates (Hz)')
        ax4.set_ylabel('num neurons')
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Rates: Inhibitory (N=%d)' % (numNeur-numExc))
        # unify x-range starting at 0 for both histograms
        ax3.set_xlim(0, x_max)
        ax4.set_xlim(0, x_max)

