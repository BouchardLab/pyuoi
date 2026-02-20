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
from Util_pseudospectra import compute_pseudospectrum
from UtilDalePoisson import get_offdiag_triplets
        
#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

#...!...!..................
    def Dale_matrix_and_eigen(self,A,md,trueD,figId=3):
        
        figId=self.smart_append(figId)        
        nrow,ncol=1,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,5.))

        dmd=md['dale_conf']
        numExc=dmd['num_excite']
        numNeur=dmd['num_neurons']
        R_sel = md['sel_spect_radius']
        R_tag = ', R=%.3f' % R_sel if R_sel is not None else ''
        print('A.shape=',A.shape)
        vmin = A.min()
        vmax = A.max()
        normMap = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        #.... Position 1: Dale matrix (natural order) ......
        ax = self.plt.subplot(nrow,ncol,1)
        im=ax.imshow(A, aspect=1., origin='lower', cmap='bwr', norm=normMap, interpolation='nearest')
        ax.set( xlabel='presyn. neuron index', ylabel='postsyn. neuron index')
        ax.set_aspect(1.0)
        ax.grid()
        cbar = fig.colorbar(im, ax=ax, extend="both", shrink=0.7)
        
        tit='True Dale, N%d%s, %s'%(A.shape[0], R_tag, md['short_name'])
        ax.set(title=tit)
        ax.axhline(numExc-0.5,color='k',ls='--', label='E/I boundary')
        ax.axvline(numExc-0.5,color='k',ls='--')
 
        #..... Position 2: Eigenvalues......
        ax = self.plt.subplot(nrow,ncol,2)
        Eigen=np.linalg.eigvals(A)
        real_parts = np.real(Eigen)
        imag_parts = np.imag(Eigen)
        ax.scatter(real_parts, imag_parts, color='blue', marker='o')
        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")
        tit3='Eigenvalues, N%d%s, %s'%(A.shape[0], R_tag, md['short_name'])
        ax.set_title(tit3)
        ax.axhline(0, color='black', lw=0.5)
        ax.axvline(0, color='black', lw=0.5)
        ax.axvline(0,color='red', linestyle='--')
        ax.set_aspect('equal')
        if R_sel is not None:
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(R_sel*np.cos(theta), R_sel*np.sin(theta), color='magenta', linestyle='--', lw=1.5, label='R=%.3f'%R_sel)
            ax.legend()
        ax.grid(True)
      
    def Dale_matrix_pseudospectra(self, A, md, trueD, figId=5):  # p=e
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(8, 6))
        
        npts = 80
        minY = -0.2
        epsMin = 0.03

        X, Y, sigma_grid, eigs = compute_pseudospectrum(A, npts, minY)
        
        ax = self.plt.subplot(1, 1, 1)
        R_sel = md['sel_spect_radius']
        R_tag = ', R=%.3f' % R_sel if R_sel is not None else ''
        title = 'Pseudospectra, true N%d%s, %s' % (A.shape[0], R_tag, md['short_name'])
        
        levels = np.logspace(-2.5, -0.5, 10)
        contour = ax.contour(X, Y, sigma_grid, levels=levels, cmap='viridis', linewidths=0.8)
        ax.clabel(contour, inline=True, fontsize=8, fmt='ε=%.3f')
        ax.text(0.05, 0.85,   'green: ε<%.3f'%epsMin, transform=ax.transAxes) 
        mask = sigma_grid > epsMin
        sigma_grid_masked = np.ma.array(sigma_grid, mask=mask)

        contour_fill = ax.contourf(X, Y, sigma_grid_masked, levels=np.linspace(0, epsMin, 10), 
                                    colors=['lightgreen'], alpha=0.5)
        
        ax.scatter(np.real(eigs), np.imag(eigs), color='red', s=15, zorder=3, label='Eigenvalues')
        
        ax.axvline(0, color='black', linestyle='--', lw=1.5)
        ax.axhline(0, color='black', linestyle='--', lw=1.5)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        ax.set_ylim(minY,1.1)
        ax.set_xlim(-1.1,1.1)
        ax.set_aspect(1.)
        ax.axhline(1,color='m',linestyle='--')
        ax.axvline(1,color='m',linestyle='--')
        ax.axvline(-1,color='m',linestyle='--')

#...!...!..................
    def histo_weights_rates(self,trueD,spikeD,md,figId=3):        
        figId=self.smart_append(figId)        
        nrow,ncol=1,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(15,3.5))        
        data_name=md['short_name']
        dmd=md['dale_conf']
        numExc=dmd['num_excite']
        numNeur=dmd['num_neurons']
        R_sel = md['sel_spect_radius']
        R_tag = ', R=%.2f' % R_sel 
                
        A=trueD['A_true']
        single_rates = spikeD['single_rates']
  
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
        
        #....   weights histogram 
        ax = self.plt.subplot(nrow,ncol,1)
        binX= np.linspace(wMin, wMax, 100)
        ax.hist(EnegT[:,2], bins=binX, color='blue', alpha=0.7, edgecolor=None,label='neg:%d'%EnegT.shape[0])
        ax.hist(EposT[:,2], bins=binX, color='red', alpha=0.7, edgecolor=None,label='pos:%d'%EposT.shape[0])

        ax.legend(loc='upper left')
        tit='True Dale, N%d%s, %s'%(A.shape[0], R_tag, md['short_name'])
        ax.set(title=tit, xlabel='True weight value',ylabel='num edges')
        ax.axvline(0,color='k',ls='--')
        ax.grid(True, alpha=0.3)

        #.... : histogram of rates
        ax = self.plt.subplot(nrow,ncol,3)
        ax.hist(single_rates, bins=20)
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

        # Natural neuron indexing: first numExc are excitatory, rest inhibitory
        inh_mask = np.zeros(numNeur, dtype=bool)
        inh_mask[numExc:] = True
        exc_mask = ~inh_mask
        neurXlab = 'neuron index'
        
        x_vals = np.arange(numNeur)         
        #....  edge count
        ax = self.plt.subplot(nrow,ncol,2)
        ax.fill_between(x_vals, edgeCount, step='mid', color='salmon', alpha=0.7)
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
               
        ax.bar(x_vals[exc_mask], single_rates[exc_mask], width=chanW, color='red', align='center', alpha=0.7, label='Excitatory')
        ax.bar(x_vals[inh_mask], single_rates[inh_mask], width=chanW, color='blue', align='center', alpha=0.7, label='Inhibitory')

        ax.set_xlabel(neurXlab)
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_ylim(0,)
        ax.grid(True, alpha=0.3)
        ax.set_title('Single Neurons, R=%.2f' % R_sel)
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
        R_sel = md['sel_spect_radius']
        R_tag = ', R=%.3f' % R_sel if R_sel is not None else ''
        
        B_idle = trueD['B_true']
        single_rates = spikeD['single_rates']
        
        # Natural indexing: first numExc are excitatory, rest inhibitory
        exc_mask = np.zeros(numNeur, dtype=bool)
        exc_mask[:numExc] = True
        inh_mask = ~exc_mask
        
        # 1) Scatter: x=B_idle, y=log(single_rates)
        ax = self.plt.subplot(nrow,ncol,1)
        ax.scatter(B_idle[exc_mask], single_rates[exc_mask], s=14, alpha=0.6, marker='^', facecolors='none', edgecolors='red', label='Excitatory')
        ax.scatter(B_idle[inh_mask], single_rates[inh_mask], s=14, alpha=0.6, marker='^', facecolors='none', edgecolors='blue', label='Inhibitory')
        ax.set_xlabel('true B_idle ')
        ax.set_ylabel('single_rates (Hz)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        tit='True Dale, N%d%s, %s'%(numNeur, R_tag, data_name)
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

        # 2) Keep this panel intentionally empty (SNR plot removed)
        ax = self.plt.subplot(nrow,ncol,4)
        ax.set_axis_off()
        
        # 3) Histogram: single_rates for excitatory
        ax3 = self.plt.subplot(nrow,ncol,2)
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
        ax4 = self.plt.subplot(nrow,ncol,3)
        ax4.hist(inh_vals, bins=common_bins, color='blue', alpha=0.7, edgecolor=None)
        ax4.set_xlabel('single_rates (Hz)')
        ax4.set_ylabel('num neurons')
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Rates: Inhibitory (N=%d)' % (numNeur-numExc))
        # unify x-range starting at 0 for both histograms
        ax3.set_xlim(0, x_max)
        ax4.set_xlim(0, x_max)
