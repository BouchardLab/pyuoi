#!/usr/bin/env python3
"""
Plotting utilities for LASSO fit evaluation and visualization.

This module provides comprehensive plotting capabilities for analyzing
LASSO Poisson model fitting results. The Plotter class extends PlotterBackbone
to create specialized visualizations including:
- Connectivity matrix heatmaps with frequency-sorted neurons
- Weight distribution histograms and statistical summaries
- Network structure plots showing excitatory/inhibitory connections  
- Reconstruction quality plots comparing fitted vs observed rates
- Comparative analysis plots for simulation validation

The plots support both simulated and experimental data with automatic
adaptation based on available metadata.
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
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gennorm
from matplotlib.colors import LogNorm


#............................
#............................
#............................
class Plotter(PlotterBackbone): 
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)         

#...!...!..................
    def summary_fitLasso(self,fitD, md,byFreq=False, figId=1):  # p=a
        figId=self.smart_append(figId)        
        nrow,ncol=2,5
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,6))

        #pprint(md)
        fmd=md['fit_lasso']
        esmd=md['edge_selector']
        Nn=fmd['num_neurons']

        # Create diagonal mask
        diagM = np.eye(Nn, dtype=bool)
        #!off_diagM = ~diagM
        
        fitType=md['fit_type']
        eselType=esmd['selector_type']
        isExp = md.get('data_type') == 'bioExp'  # Automatically detect experimental data
        # Unpack arrays from bigD
        A_fit = fitD['A_'+fitType]
        B = fitD['B_'+fitType]
        Freq=fitD['single_rates']
                 
        #...... Training curves
        ax = self.plt.subplot(nrow,ncol,1)
        plot_trainingCurves(ax,fitD,md)

        #.... : histogram of rates
        ax = self.plt.subplot(nrow,ncol,1+ncol)  
        ax.hist(Freq, bins=20) 
        x_vals = np.arange(Nn)
        frLab='Firing rate (Hz)'
        ax.set(ylabel='num neurons',xlabel=frLab,title='Single rates, %d neurons'%Nn)
        ax.grid(True, alpha=0.4)
        ax.set_xlim(0,)
        # Compute median
        median_val = np.median(Freq)
        txtM= f'median rate: {median_val:.2f} Hz'
        ax.text( median_val,ax.get_ylim()[1]*0.8,txtM, 
                 color='red', ha='left', va='bottom', fontsize=10)
        ax.axvline(median_val, color='red', linestyle='--', linewidth=1)

        # ... A off-diagonal
        xLab='edge value'
        frLab='log10( Firing rate/ Hz )'
        Aedg=A_fit[~diagM]
        i_indices, j_indices = np.where(~diagM)
        Freq_expanded = Freq[i_indices]
        # Filter out 0 values
        valid_mask =np.abs(Aedg)>1e-10
        Aedg_clean = Aedg[valid_mask]
        Freq_expanded_clean = Freq_expanded[valid_mask]
        # Count non-zero edges (non-NaN values)
        n_edges = len(Aedg_clean)

        ax = self.plt.subplot(nrow,ncol,2)
        ax.hist(Aedg_clean, bins=100,color='g',alpha=0.8)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.4)
        ax.set(ylabel='edges',xlabel=xLab,title='A off-diagonal')
        txt='edge sel meth: %s \n acc frac=%.2f' %(  eselType,n_edges/Nn/(Nn-1))
        ax.text(0.05, 0.75,   txt, transform=ax.transAxes, fontsize=10)
        # Plot 2D histogram
        ax = self.plt.subplot(nrow,ncol,2+ncol)
        h = ax.hist2d(Aedg_clean, np.log10(Freq_expanded_clean), bins=30, cmap='viridis', norm=LogNorm())
        cbar = fig.colorbar(h[3],ax=ax)
        ax.set(ylabel=frLab,xlabel=xLab,title='accept %d of %d edges '%(n_edges,Nn*(Nn-1)))
        ax.grid(True, alpha=0.4)
        
        # ... A sparsity vs epoch .....
        ax = self.plt.subplot(nrow,ncol,3)
        epochs = fitD.get('losses_epochs', None)
        sp = np.asarray(fitD['sparsity_epoch'])
        if epochs is None:
            epochs = np.arange(1, len(sp)+1, dtype=np.int32)
        else:
            epochs = np.asarray(epochs)
        ax.plot(epochs, sp, color='tab:purple', linewidth=1.5)
        add_delay_markers(ax, fmd)
        ax.set_ylim(0.0, 1.02)
        ax.set(ylabel='fraction', xlabel='epoch', title='A sparsity (off-diag)')
        ax.grid(True, alpha=0.4)

        # ... spectral radius vs epoch .....
        ax = self.plt.subplot(nrow,ncol,4)
        rho_ep = fitD.get('spectral_radius_epoch', None)
        if rho_ep is not None:
            rho_ep = np.asarray(rho_ep)
            if epochs is None:
                rho_epochs = np.arange(1, len(rho_ep)+1, dtype=np.int32)
            else:
                rho_epochs = np.asarray(epochs)
            ax.plot(rho_epochs, rho_ep, color='tab:red', linewidth=1.5)
            add_delay_markers(ax, fmd)
            if 'rho_max' in fmd:
                ax.axhline(float(fmd['rho_max']), color='k', linestyle='--', linewidth=1.0)
            ax.set(ylabel='radius', xlabel='epoch', title='Spectral radius(A)')
        else:
            rho_now = float(np.max(np.abs(np.linalg.eigvals(A_fit))))
            ax.plot([0, 1], [rho_now, rho_now], color='tab:red', linewidth=1.5)
            ax.set(ylabel='radius', xlabel='epoch', title='Spectral radius(A, const)')
        ax.grid(True, alpha=0.4)
        
        # ... B-term .....
        xLab='B-term value'
        ax = self.plt.subplot(nrow,ncol,5)
        ax.hist(B, bins=30,color='darkviolet')
        ax.grid(True, alpha=0.4)
        ax.set(ylabel='neurons',xlabel=xLab,title='B-term')
        ax.axvline(0, linestyle='--', color='lime', linewidth=1)
        # Plot 2D histogram
        ax = self.plt.subplot(nrow,ncol,4+ncol)
        h = ax.hist2d(B,np.log10(Freq),  bins=30, cmap='Grays',vmax=2.1)
        cbar = fig.colorbar(h[3],ax=ax)
        ax.set(ylabel=frLab,xlabel=xLab,title='num neurons')
        ax.grid(True, alpha=0.4)
        ax.axvline(0, linestyle='--', color='lime', linewidth=1)

        short_name = md.get('short_name', '')
        if short_name:
            fig.suptitle(f"Summary LASSO: {short_name}", fontsize=14)
 
 #...!...!..................
    def residuals(self, evalD,md, figId=1):
        #pprint(md)
        fitType=md['fit_type']
        fmd=md['fit_'+fitType]
        
        figId=self.smart_append(figId)        
        nrow,ncol=2,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,6))

        # Unpack arrays from bigD
        for j,etype in enumerate(['neg','pos']):
            ax = self.plt.subplot(nrow,ncol,1+j)
            
            title = 'edges: %s' % etype
            valT,valF=plot_correl_offdiag(fig,ax,evalD[etype])
            ax.set_title(title)
        
            ax = self.plt.subplot(nrow,ncol,1+j+ncol)
            plot_1D_residuals(ax, valT,valF,lab='TP off-diag '+etype,col='green')

            if etype=='pos':
                txt=md["short_name"]
            else:
                txt='fit: '+fitType
            ax.text(0.05, 0.2, txt, transform=ax.transAxes, fontsize=8)

        # ... diagonal
        title = 'fit (diagonal)'
        ax = self.plt.subplot(nrow,ncol,3)
        V=evalD['diag']
        dCol='salmon'
        ax.scatter(V[:,1], V[:,0], alpha=0.6, color=dCol,marker='.',s=5)
        ax.set(title=title,xlabel='truth',ylabel='fitted')
        add_x45_lins(ax, only45=True)
        ax.grid(True, alpha=0.5)
        
        ax = self.plt.subplot(nrow,ncol,3+ncol)        
        plot_1D_residuals(ax,V[:,1],V[:,0],lab='diag',col=dCol)


        # ...  B-term
        title = 'fit (B-term)'
        ax = self.plt.subplot(nrow,ncol,4)
        dCol='darkviolet'
        V=evalD['bterm']
        ax.scatter(V[:,1], V[:,0], alpha=0.6, color=dCol,marker='.',s=5)
        ax.set(title=title,xlabel='truth',ylabel='fitted')
        add_x45_lins(ax, only45=True)
        ax.grid(True, alpha=0.5)
        
        ax = self.plt.subplot(nrow,ncol,4+ncol)        
        plot_1D_residuals(ax,V[:,1],V[:,0],lab='diag',col=dCol)

        short_name = md.get('short_name', '')
        if short_name:
            fig.suptitle(f"Residuals: {short_name}", fontsize=14)
 

#...!...!..................
    def A_histos(self, fitD, md, spikeD, figId=1, k=6):
        #pprint(md); aa67
        fitType=md['fit_type']

        fmd=md['fit_'+fitType]
        
        #1isExp = md.get('data_type') == 'bioExp'  # Automatically detect experimental data
        figId=self.smart_append(figId)        
        nrow,ncol=k,2  # Add 1 extra row for the neuron stats plot
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,12))

        # Unpack arrays from bigD
        A_fit = fitD['A_'+fitType].copy()
        Freq=fitD['single_rates']
        
        # Multiply each row of A_fit by the corresponding element of F
        A_fit=A_fit * np.sqrt(Freq[:,None])
          
        # convert 0's to Nan
        A_fit[ A_fit==0.]=np.nan

        num_neurons = A_fit.shape[0]
        wzoomMx=0.05        
        # Remove diagonal elements by setting them to NaN
        A_fit_no_diag = A_fit.copy()

        short_name = md['short_name']
        
        np.fill_diagonal(A_fit_no_diag, np.nan)
        A_flat = A_fit_no_diag.flatten()
        xLab=r' weights * $\sqrt{ rate}$'
        
        #------- top left plot w/ 1D histo fo weights
        ax = self.plt.subplot(nrow,ncol,1)
        ax.hist(A_flat, bins=200, color='green', alpha=0.7, edgecolor=None)
        ax.axvline(0, linestyle='--', color='lime', linewidth=1)
        ax.grid()
        method_name = md.get('edge_selection_method', 'unknown')
        ax.set(xlabel=xLab,ylabel='count',title=f'All non-diag weights, Fit {fitType}, Method: {method_name}')
        ax.set_yscale('log')

        # Left column: 2D histogram of A-matrix (mutiple rows)
        ax = self.plt.subplot2grid((nrow, ncol), (1, 0), rowspan=5)
        ax.axvline(0, linestyle='--', color='lime', linewidth=1)

        # Create 2D histogram: x-axis is value, y-axis is row index
        row_indices = np.repeat(np.arange(num_neurons), num_neurons)

        # Remove NaN values (diagonal elements)
        valid_mask = ~np.isnan(A_flat)
        A_flat = A_flat[valid_mask]
        row_indices = row_indices[valid_mask]
        
        # Use ax.hist2d() directly with log scale
        H, xedges, yedges, im = ax.hist2d(A_flat, row_indices, bins=[50, num_neurons], cmap='Greys', vmax=6)

        cbar = fig.colorbar(im, ax=ax,
                            orientation='horizontal',  # Make it horizontal
                            location='bottom',          # Place it below
                            shrink=0.7,                 # Make it 70% width
                            pad=0.1)                    # Add some padding from plot
        cbar.set_label('num edges')

        ax.set_xlabel(xLab)
        ax.set_ylabel('freq-sorted neuron index')

        pprint(md['fit_lasso'])
        pprint(md)
        lasso_name = md['provenance']['output_lasso_file']
        title_text = f'{lasso_name},  Fit {fitType}, epochs={fmd["num_epochs"]}, sampl/k={fmd["num_samples_used"]/1000}'

        ax.set_title(title_text)

        # Compute row indices and collect data for global binning
        single_rates = spikeD['single_rates']
        idxOff = (num_neurons // k) // 2 + 2
        row_indices = [min(i * (num_neurons // k) + idxOff, num_neurons - 1) for i in range(k)]
        selected_rows_data = [A_fit_no_diag[idx, :][~np.isnan(A_fit_no_diag[idx, :])] for idx in row_indices]
        all_valid_data = np.concatenate([data for data in selected_rows_data if len(data) > 0])
        global_bin_edges = np.linspace(np.min(all_valid_data), np.max(all_valid_data), 51) if len(all_valid_data) > 0 else None

        # Add frequency annotations and plot histograms in single loop
        xlim_offset = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        for i in range(k):
            rowIdx = row_indices[i]
            ax.axhline(rowIdx, color='yellow', linewidth=2, alpha=0.8)
            ax.text(ax.get_xlim()[0] + xlim_offset, rowIdx, f'{single_rates[rowIdx]:.1f} Hz', verticalalignment='center', horizontalalignment='left', fontsize=12)
            ax_hist = self.plt.subplot(nrow, ncol, 2 + i * ncol)
            reversed_rowIdx = row_indices[k - 1 - i]
            #xLab1 = xLab if (i == k - 1) else None
            plot_row_histogram(ax_hist, A_fit_no_diag, reversed_rowIdx, single_rates, global_bin_edges, xLab, isExp=True)


        
  
        if short_name:
            fig.suptitle(f"Freq-sorted weights: {short_name}", fontsize=14)

#...!...!..................
    def edges_fitLasso(self, fitD, md, minW=0,figId=1):
        """Compare binary edge mask to ground truth."""
        fitType = md['fit_type']
        E_hat = fitD.get('E_mask')
        if E_hat is None:
            E_hat = fitD.get(f'E_{fitType}')
        
        if minW>0:
            A_hat = fitD.get(f'A_{fitType}')           
            if A_hat is None:
                A_hat = fitD.get('A_avr')
            E_hat = (np.abs(A_hat) > minW)
                
        E_true = fitD['E_true']
        E_hat = np.array(E_hat, dtype=bool)
        E_true = np.array(E_true, dtype=bool)
        assert E_hat.shape == E_true.shape
    
        N = E_true.shape[0]
        diag = np.eye(N, dtype=bool)
        E_hat = E_hat.copy(); E_hat[diag] = False
        E_true = E_true.copy(); E_true[diag] = False

        TP = E_hat & E_true
        FP = E_hat & (~E_true)
        FN = (~E_hat) & E_true
        TN = (~E_hat) & (~E_true)

        counts = {
            'TP': int(TP.sum()),
            'FP': int(FP.sum()),
            'FN': int(FN.sum()),
            'TN': int(TN.sum()),
        }
        denom_pos = max(E_true.sum(), 1)
        denom_pred = max(E_hat.sum(), 1)
        recall = counts['TP'] / denom_pos
        precision = counts['TP'] / denom_pred
        f1 = 2*precision*recall / max(precision+recall, 1e-12)
        acc = (counts['TP'] + counts['TN']) / max(E_true.size - N, 1)  # exclude diag

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(16,3.5))
        gs = gridspec.GridSpec(1,4, width_ratios=[1,1,1,1.1])

        ax = self.plt.subplot(gs[0])
        A_true = fitD['A_true']
        if A_true is not None and A_true.ndim == 3:
            A_true = A_true[0]
        if A_true is not None:
            vmin = A_true.min()
            vmax = A_true.max()
            normMap = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            im = ax.imshow(A_true, aspect=1., origin='lower', cmap='bwr', norm=normMap, interpolation='nearest')
            dmd = md.get('dale_conf', {})
            numExc = dmd.get('num_excite')
            numNeur = dmd.get('num_neurons', N)
            R_sel = md.get('sel_spect_radius')
            R_tag = f", R={R_sel:.3f}" if R_sel is not None else ''
            if 'E_true' in md:
                E_true = md['E_true']
                n_diag = int(np.trace(E_true))
                n_off = int(np.sum(E_true) - n_diag)
            else:
                n_diag = int(np.count_nonzero(np.diag(A_true)))
                n_off = int(np.count_nonzero(A_true) - n_diag)
            tit = f"True Dale, N{A_true.shape[0]}{R_tag}, nEdges={n_diag}+{n_off}"
            ax.set(title=tit)
            if numExc is not None:
                ax.axhline(numExc-0.5, color='k', ls='--')
                ax.axvline(numExc-0.5, color='k', ls='--')
            ax.plot([0,numNeur],[0,numNeur],'--',lw=0.8,color='magenta')
        else:
            im = ax.imshow(E_true, cmap='Greys', vmin=0, vmax=1, origin='lower')
            ax.set(title=f'True edges (n={int(E_true.sum())})')
            ax.plot([0,N],[0,N],'--',lw=0.8,color='magenta')
        self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_aspect(1.0)
        ax.grid()
        ax.set_xlim(-0.5,N+0.5);     ax.set_ylim(-0.5,N+0.5)
        ax.set( xlabel='presyn. neuron index (output)', ylabel='postsyn. neuron index (input)')
        ax.xaxis.labelpad = 8

        ax = self.plt.subplot(gs[1])
        im = ax.imshow(E_hat, cmap='Greys', vmin=0, vmax=1, origin='lower')
        ax.set(title=f'Predicted edges (n={int(E_hat.sum())}), minW={minW}')
        self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        dmd = md.get('dale_conf', {})
        numExc = dmd.get('num_excite')
        if numExc is not None:
            ax.axhline(numExc-0.5, color='k', ls='--')
            ax.axvline(numExc-0.5, color='k', ls='--')
        ax.plot([0,N],[0,N],'--',lw=0.8,color='magenta')
        ax.set_aspect(1.0)
        ax.grid()
        ax.set_xlim(-0.5,N+0.5);     ax.set_ylim(-0.5,N+0.5)
        ax.set( xlabel='presyn. neuron index (output)', ylabel='postsyn. neuron index (input)')
        ax.xaxis.labelpad = 8
        ax.set( xlabel='presyn. neuron index', ylabel='postsyn. neuron index')

        conf_map = np.zeros_like(E_true, dtype=int)
        conf_map[TP] = 3
        conf_map[FP] = 2
        conf_map[FN] = 1
        cmap_conf = colors.ListedColormap(['white',  'magenta','red', 'green'])
        bounds = [-0.5,0.5,1.5,2.5,3.5]
        norm = colors.BoundaryNorm(bounds, cmap_conf.N)
        ax = self.plt.subplot(gs[2])
        im = ax.imshow(conf_map, cmap=cmap_conf, norm=norm, origin='lower')
        ax.set(title='Confusion map')
        cbar = self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0,1,2,3])
        cbar.set_ticklabels(['TN','FN','FP','TP'])
        ax.plot([0,N],[0,N],'--k',lw=0.8)
        ax.set_aspect(1.0)
        ax.grid()
        ax.set_xlim(-0.5,N+0.5);     ax.set_ylim(-0.5,N+0.5)
        ax.set( xlabel='presyn. neuron index', ylabel='postsyn. neuron index')

        ax = self.plt.subplot(gs[3])
        bar_names = ['TP','FP','FN']
        vals = [counts[k] for k in bar_names]
        ax.bar(bar_names, vals, color=['green','red','magenta','grey'])
        ax.set_ylabel('count')
        y_pos = 0.25 * max(1, max(vals))
        for name, val in zip(bar_names, vals):
            ax.text(name, y_pos, f"{val}", ha='center', va='center', fontsize=10)
        txt=f'precision={precision:.3f}\nrecall={recall:.3f}\nf1={f1:.3f}\nacc={acc:.3f}'
        ax.text(0.55, 0.75,  txt, transform=ax.transAxes)

        ax.set_title('stats')
        ax.grid(axis='y', alpha=0.4)

        fig.subplots_adjust(bottom=0.18)
        short_name = md.get('short_name', '')
        tag = f", short_name={short_name}" if short_name else ""
        fig.suptitle(f"Edge detection vs truth, minW={minW}{tag}", fontsize=14)
        return

#...!...!..................
    def summary_network(self, fitD, edgeD, md, procFrac=0.8,figId=5):
        figId=self.smart_append(figId)        
        nrow,ncol=1,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,3.5))

        fitType=md['fit_type']

        # Unpack arrays from bigD
        A_fit = fitD['A_'+fitType]
        Neu_sum=edgeD['edge_sum']
        Neu_edg=edgeD['edge_vals']
        
        # .... A-matrix  ....
        ax = self.plt.subplot(nrow,ncol,1)
        plot_A2D(fig,ax,A_fit)

        # .... A-eigen  ....
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

        # .... pos vs. neg count  ....
        ax = self.plt.subplot(nrow,ncol,3)
        # Extract positive and negative counts
        n_pos = Neu_sum[:, 1]
        n_neg = Neu_sum[:, 2]
        # Create 2D histogram
        h = ax.hist2d(n_pos, n_neg, bins=20, cmap='Greys', cmin=1,cmax=5)
        # Add colorbar
        self.plt.colorbar(h[3], ax=ax, label='Neurons')
        ax.set_xlim(0,)
        ax.set_ylim(0,)
        ax.grid(True, alpha=0.3)
        
        # Add diagonal line (y=x)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'b--', alpha=0.5, linewidth=1.5, label='y=x')
        ax.set(title='Edge type correlation, %d neurons'%(A_fit.shape[0]),xlabel='num pos',ylabel='num neg')
        
        # .... edge std vs. value ....
        ax = self.plt.subplot(nrow,ncol,4)
        ax.grid(True, alpha=0.3)
        ax.set(title='Edge value accuracy',xlabel='edge val',ylabel='edge std')

        # Extract average weights and standard deviations
        avg_weights = Neu_edg[:, 2]
        std_devs = Neu_edg[:, 3]

        # Compute percentile range to contain procFrac of data by magnitude
        abs_weights = np.abs(avg_weights)
        percentile_cutoff = procFrac * 100
        threshold = np.percentile(abs_weights, percentile_cutoff)
         
        # Filter data within range
        mask = abs_weights <= threshold
        avg_weights_accepted = avg_weights[mask]
        std_devs_accepted = std_devs[mask]

        short_name = md.get('short_name', '')
        if short_name:
            fig.suptitle(f"Summary network: {short_name}", fontsize=14)
    
        # Count accepted and rejected
        n_total = len(avg_weights)
        n_accepted = len(avg_weights_accepted)
        n_rejected = n_total - n_accepted
    
        # Create 2D histogram
        h = ax.hist2d(avg_weights_accepted, std_devs_accepted, bins=30, cmap='Greys', cmin=1)
        
        # Add colorbar
        self.plt.colorbar(h[3], ax=ax, label='edges')
        ax.axvline(0, linestyle='--', color='lime', linewidth=1)
        
        # Add text box with statistics
        stats_text = (f'display fract = {procFrac:.2f}\n'
                  f'Accepted: {n_accepted:,} ({n_accepted/n_total*100:.1f}%)\n'
                  f'Rejected: {n_rejected:,} ({n_rejected/n_total*100:.1f}%)\n'
                  f'abs(x) cutoff: ±{threshold:.3f}')
    
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top')
       

        
#............................
#............................
#............................
   
def plot_trainingCurves(ax,fitD,md,title='aa3'):
    fitType=md['fit_type']
    fmd=md['fit_'+fitType]

    epoch0=10 # skip intial loss values
    lossTot = fitD['losses_total'][epoch0:]
    epochsT=fitD['losses_epochs'][epoch0:]
 
    # Plot total loss on left y-axis
    ax.plot(epochsT, lossTot, label='total '+fitType, color='blue', linestyle='-')

    titl='%dk samp'%(fmd["num_samples_used"]/1000)
    
    if fitType=='lasso':  # Create second y-axis for L1 loss
        lossL1= lossTot - fitD['losses_wo_L1'][epoch0:]  
        ax2 = ax.twinx()
        ax2.plot(epochsT, lossL1, label='L1 loss', color='red', linestyle='--')
        ax2.set_ylabel('L1 loss', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        # Move y-axis ticks and labels inside
        ax2.tick_params(axis='y', direction='in', pad=-60, labelsize=10)
        ax2.yaxis.set_label_position('right')

        # Format second y-axis ticks in scientific notation
        from matplotlib.ticker import FuncFormatter
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1e}'))
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, title=titl)
    else:
        ax.legend( title=titl)
            
    title = 'Loss: '+md["short_name"]
    
    add_delay_markers(ax, fmd)
    ax.set(xlabel='Epoch', title=title)
    ax.set_ylabel('Loss', color='blue')
    
    # Color the y-axis labels to match the lines
    ax.tick_params(axis='y', labelcolor='blue')
    ax.grid(True, alpha=0.3)


def add_delay_markers(ax, fmd):
    """Add dashed black vertical lines at delayed-constraint boundaries."""
    delay = fmd.get('delay_epoch', None)
    if delay is None:
        return
    delay = int(delay)
    if delay < 0:
        return
    # Training logic applies pruning starting at 0-based epoch>=delay,
    # which corresponds to displayed epoch delay+1.
    ax.axvline(delay + 1, color='k', linestyle='--', linewidth=1.0)
      
def plot_A2D(fig,ax,A,title='aa',byFreq=True,trueD=None):
    Am=A.copy()
   
    
    # If byFreq=False, reorder matrix to natural neuron indexing
    if not byFreq and trueD is not None and 'neur_revFreqIdx' in trueD:
        neur_revFreqIdx = trueD['neur_revFreqIdx']  # freq_sorted_position → natural_index
        # Reorder both rows and columns from frequency-sorted to natural order
        Am = Am[np.ix_(neur_revFreqIdx, neur_revFreqIdx)]
        mask = mask[np.ix_(neur_revFreqIdx, neur_revFreqIdx)]
        Am[~mask]=0  # Re-apply mask after reordering
 
    vmin = Am.min()-0.3
    vmax = Am.max()+0.3
    
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im1=ax.imshow(Am, cmap='bwr', norm=norm, origin='lower')  # 'RdBu_r'

    #masked_values = Am[mask]
    #vmin = masked_values.min()
    #vmax = masked_values.max()
    
    nval=np.sum(Am!=0.)
    title='%s n=%d'%(title,nval)
    # Simplified axis labels based on display order
    if byFreq:
        ylabel_text = 'From neuron (freq-sorted)'
        xlabel_text = 'To neuron (freq-sorted)'
    else:
        ylabel_text = 'From neuron (natural idx)'
        xlabel_text = 'To neuron (natural idx)'
    ax.set(title=title, ylabel=ylabel_text, xlabel=xlabel_text)
    fig.colorbar(im1, ax=ax, shrink=0.7)
    ax.grid(True, alpha=0.5)
    #add_x45_lins(ax, only45=True)

  
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


def plot_correl_offdiag(fig,ax,tripV):
    TP,FP,FN=tripV
    #print('ss',TP.shape)
    ax.scatter(TP[:,3], TP[:,2], alpha=0.6, color='green',label='TP: %d'%TP.shape[0],marker='.',s=5)
    n=FN.shape[0]
    ax.scatter(FN[:,2], [0]*n, alpha=0.6, color='red',s=5,label='FN: %d'%FN.shape[0])

    n=FP.shape[0]
    ax.scatter([0]*n,FP[:,2],  alpha=0.6, color='blue',s=5,label='FP: %d'%FP.shape[0])
    
    ax.set(aspect=1. ,xlabel='true weight',ylabel='fitted')
    ax.grid(True, alpha=0.5)

    add_x45_lins(ax)
    if np.any(TP):  # Only plot if there are TP points
        x_cg = np.mean(TP[:,3])
        y_cg = np.mean(TP[:,2])
        ax.scatter(x_cg, y_cg, marker='+', s=200, color='k', linewidths=2) #, label='TP avr')
    ax.legend()

    return TP[:,3], TP[:,2]
  


#...!...!..................
def XXXplot_1d_weight_histo_with_stats(ax, group_values, start_row, end_row, group_idx, color, ggd_rangeL, isExp=False):
    """Plot 1D histogram with statistics for a single group"""

    ggd_range=(ggd_rangeL[0]+ggd_rangeL[1])/2.  # TMP
    # Create histogram for this group with log scale
    counts, bins, _ = ax.hist(group_values, bins=100, alpha=0.7, color=color)
    
    ax.axvline(x=ggd_range, color='black', linestyle='--', linewidth=1, alpha=0.8)
    ax.axvline(x=-ggd_range, color='black', linestyle='--', linewidth=1, alpha=0.8)
    ax.axvline(0, linestyle='--', color='lime', linewidth=1)

    # Compute mean and RMSE in the range ±ggd_range, excluding values too close to zero
    mask_range = (np.abs(group_values) < ggd_range) & (np.abs(group_values) > ggd_range/5)
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
    
    ax.set_xlabel('Neuron freq-sorted index')
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
 
       
#...!...!..................
def plot_row_histogram(ax, A_fit_no_diag, rowIdx, single_rates=None, global_bin_edges=None, xLab=None, isExp=False):
    """Plot histogram of a specific row from the 2D matrix with statistics"""
    
    # Extract row data (exclude diagonal element which is NaN)
    row_data = A_fit_no_diag[rowIdx, :]
    valid_data = row_data[~np.isnan(row_data)]
    
    if len(valid_data) == 0:
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, ha='center', va='center')
        return
    
    # Plot histogram with consistent binning
    if global_bin_edges is not None:
        counts, bin_edges, _ = ax.hist(valid_data, bins=global_bin_edges, color='chocolate')
    else:
        neve_happens
        n_bins = min(60, len(valid_data)//3)  # Adaptive bin count fallback
        counts, bin_edges, _ = ax.hist(valid_data, bins=n_bins, alpha=0.7, color='skyblue')
    
    # Calculate median value
    xMedian = np.median(valid_data)
    
    # Define slice range around median
    xDel = 0.1
    slice_mask = (valid_data >= (xMedian - xDel)) & (valid_data <= (xMedian + xDel))
    sliced_data = valid_data[slice_mask]
    
    # Compute standard deviation of sliced data
    if len(sliced_data) > 1:
        std_val = np.std(sliced_data)
        x0 = np.mean(sliced_data)
    else:
        std_val = 0.0
        x0 = xMedian
    
    # Draw x=0 reference line (always shown)
    ax.axvline(0, linestyle='--', color='lime', linewidth=1)
    
    # Draw median line (always shown)
    ax.axvline(xMedian, linestyle='-', color='k', linewidth=2, label='Median')
    
    # Draw dashed lines for the ±0.1 range (only for simulation data)
    if not isExp:
        ax.axvline(xMedian - xDel, linestyle='--', color='k', alpha=0.7)
        ax.axvline(xMedian + xDel, linestyle='--', color='k', alpha=0.7)
    else:
        ax.set_yscale('log')

    # Add merged text with statistics and row/frequency info inside the plot
    freq_text = f', {single_rates[rowIdx]:.1f}Hz' if single_rates is not None else ''
    info_text = f'Row {rowIdx}{freq_text}\nmedian={xMedian:.3f}, std={std_val:.3f}'
    
    ax.text(0.98, 0.95, info_text, transform=ax.transAxes, fontsize=8, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Set consistent x-axis limits when using global binning
    if global_bin_edges is not None:
        ax.set_xlim(global_bin_edges[0], global_bin_edges[-1])
    
    if xLab!=None:   ax.set_xlabel(xLab)

    ax.set_ylabel('Count')  # Restore y-axis label
    ax.grid(True, alpha=0.3)
