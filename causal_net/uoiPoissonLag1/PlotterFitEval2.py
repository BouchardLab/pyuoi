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
from Util_poissonFdr import qa_Bfit, qa_Afit

#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)


#...!...!..................
    def summary_fitUoI(self,fitD, trueD,md,byFreq=False, figId=1):
        figId=self.smart_append(figId)
        nrow,ncol=1,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(14,3))

        fmd=md['fit_uoi']
        cmd=md['conf_uoi']
        Nn=fmd['num_neurons']
        fitType=md['fit_type']
        isExp = md.get('data_type') == 'bioExp'  # Automatically detect experimental data

        # Unpack arrays from bigD
        A_fit = fitD['A_'+fitType]
        B = fitD['B_'+fitType]
        Freq=fitD['single_rates']
        A_true = trueD['A_true']
              
        #...... Training curves
        ax = self.plt.subplot(nrow,ncol,1)
        iterSkip=3 # is x10
        l1_loss=fitD['l1_loss_sel'][:,iterSkip:]
        ax.plot(l1_loss[0] ,l1_loss[1], marker='o')
        ax.grid(True)
        # Title and run summary
        samples_k = int(fmd.get('num_samples_used', 0))//1000
        tit='UoI %s  samples=%dk' % (md.get('short_name',''), samples_k)
        ax.set(title=tit,
               xlabel='UoI Iteration',
               ylabel='UoI L1 Loss')
        slurm_nodes = fmd.get('slurm_nodes', 0)
        slurm_ranks = fmd.get('slurm_ranks', 0)
        fit_min = fmd.get('training_time_sec', 0)/60.0
        n_boots_sel = cmd.get('n_boots_sel', fmd.get('n_boots', '?'))
        n_boots_est = cmd.get('n_boots_est', fmd.get('n_boots', '?'))
        info_txt = 'Slurm N=%s ranks=%s\n fit %.1f min\n nBoot %s, %s' % (str(slurm_nodes), str(slurm_ranks), fit_min, str(n_boots_sel), str(n_boots_est))
        ax.text(0.5, 0.95, info_txt, transform=ax.transAxes, va='top', ha='left')
        
        #...... off diagonal distribution
        ax = self.plt.subplot(nrow,ncol,2)
        # 1D histogram of non-zero off-diagonal A_fit values
        N = A_fit.shape[0]
        offdiag_mask = ~np.eye(N, dtype=bool)
        A_off = A_fit[offdiag_mask]
        nz = A_off[np.abs(A_off) > 0]
        ax.hist(nz, bins=100, color='green', alpha=0.8, edgecolor=None)
        tit='A_fit off-diagonal (N=%d)' % nz.size
        ax.set(title=tit,
               xlabel='A_fit value (off-diagonal, non-zero)',
               ylabel='count')
        ax.grid(True)

        #...... diagonal(A_fit) distribution
        ax = self.plt.subplot(nrow,ncol,3)
        diag_vals = np.diag(A_fit)
        ax.hist(diag_vals, bins=100, color='brown', alpha=0.8, edgecolor=None)
        tit='A_fit diagonal (N=%d)' % diag_vals.size
        ax.set(title=tit,
               xlabel='A_fit diag value',
               ylabel='count')
        ax.grid(True)

        #...... B (bias) distribution
        ax = self.plt.subplot(nrow,ncol,4)
        ax.hist(B, bins=100, color='salmon', alpha=0.8, edgecolor=None)
        tit='B_fit values (N=%d)' % B.size
        ax.set(title=tit,
               xlabel='B_fit value',
               ylabel='count')
        ax.grid(True)

        # Figure title with short_name, samples, and fdr rate
        fdr_rate = fmd.get('fdr_rate', md.get('conf_uoi', {}).get('fdr_rate', None))
        if fdr_rate is None:
            fig_tit = '%s  samples=%dk' % (md.get('short_name',''), int(fmd.get('num_samples_used', 0))//1000)
        else:
            fig_tit = '%s  samples=%dk   fdr=%.3f' % (md.get('short_name',''), int(fmd.get('num_samples_used', 0))//1000, float(fdr_rate))
        self.plt.suptitle(fig_tit)
        self.plt.tight_layout(rect=[0,0,1,0.92])

#...!...!..................
    def correlations(self, fitD, trueD, md, figId=1):
        figId=self.smart_append(figId)
        nrow,ncol=3,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(14,9))
        
        # Metadata
        fmd = md['fit_uoi']
        data_short = md.get('short_name','')
        samples_k = int(fmd.get('num_samples_used', 0))//1000
        fdr_rate = fmd.get('fdr_rate', md.get('conf_uoi', {}).get('fdr_rate', None))
        fig_tit = 'UoI-Poisson-FDR   %s  samples=%dk   fdr=%.3f' % (data_short, samples_k, float(fdr_rate))
        # Extract arrays
        fitType = md['fit_type']
        A_fit = fitD['A_'+fitType]
        B_fit = fitD['B_'+fitType]
        A_true = trueD['A_true']
        B_true = trueD['B_true']
        
        # Build QA stats, borrowing style from sparePlot
        qaD = qa_Afit(A_true, A_fit)
        qaD['bterm'] = qa_Bfit(B_true, B_fit)
        
        colors = {'exc': 'red', 'inh': 'blue', 'diag': 'brown', 'bterm': 'salmon'}
        names = ['inh', 'exc', 'diag', 'bterm']
        
        for i, name in enumerate(names):
            ax = self.plt.subplot(nrow, ncol, i+1)
            stats = qaD[name]
            color = colors[name]
            tval = stats['tval']
            fval = stats['fval']
            
            if tval.size > 0:
                ax.scatter(tval, fval, alpha=0.6, s=10, c=color)
                # y=x line
                lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),
                    np.max([ax.get_xlim(), ax.get_ylim()]),
                ]
                ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
                ax.set_aspect('equal', 'box')
                ax.set_xlim(lims)
                ax.set_ylim(lims)
                # center of gravity cross
                ax.plot(np.mean(tval), np.mean(fval), '+', c='black', markersize=18, markeredgewidth=3)
            
            ax.grid(True, alpha=0.3)
            sub_tit=name.capitalize()
            ax.set(title=sub_tit, xlabel='True Value', ylabel='Fitted Value')
            
            # mean/std box
            ax.text(0.95, 0.07, 'N=%d\nmean=%.3f\nstd=%.3f'%(tval.shape[0], stats['res_mean'], stats['res_std']),
                    transform=ax.transAxes, fontsize=9, va='bottom', ha='right')
            if name in ['exc','inh']:
                ax.text(0.05, 0.95, 'TP=%d\nFP=%d\nFN=%d'%(stats['TP'], stats['FP'], stats['FN']),
                        transform=ax.transAxes, fontsize=9, va='top', ha='left')
        
        # Second row: residuals vs single_rates with adaptive x-scale, lighter colors
        rates = fitD.get('single_rates', None)
        use_log_x = False
        if rates is not None and np.any(rates > 0):
            rmin = float(np.min(rates[rates > 0]))
            rmax = float(np.max(rates))
            ratio = rmin / rmax if rmax > 0 else 1.0
            # If small dynamic range (min/max > 0.02), use linear; otherwise log
            use_log_x = not (ratio > 0.02)
        for i, name in enumerate(names):
            ax = self.plt.subplot(nrow, ncol, ncol + i + 1)
            stats = qaD[name]
            color = colors[name]
            tval = stats['tval']
            fval = stats['fval']
            resid = fval - tval
            if tval.size > 0 and rates is not None:
                if name in ['exc','inh']:
                    diag_m = np.eye(A_true.shape[0], dtype=bool)
                    if name == 'exc':
                        true_m = (A_true > 0) & (~diag_m)
                        pred_m = (A_fit > 0) & (~diag_m)
                    else:
                        true_m = (A_true < 0) & (~diag_m)
                        pred_m = (A_fit < 0) & (~diag_m)
                    tp_m = true_m & pred_m
                    _, cols = np.where(tp_m)
                    xvals = rates[cols]
                elif name == 'diag':
                    xvals = rates  # one per neuron
                else:  # bterm
                    xvals = rates
                epsx = 1e-6
                m = (xvals > epsx) if use_log_x else np.ones_like(xvals, dtype=bool)
                ax.scatter(xvals[m], resid[:m.sum()], alpha=0.4, s=10, c=color)
                ax.set_xscale('log' if use_log_x else 'linear')
                ax.axhline(0.0, linestyle='--', color='gray', linewidth=0.8)
            ax.grid(True, alpha=0.3)
            sub_tit='%s residuals vs rate'%(name.capitalize())
            ax.set(title=sub_tit,
                   xlabel='single_rates (Hz)',
                   ylabel='fit - true')
        
        # Third row: histograms of residuals with stats and x=0 line
        for i, name in enumerate(names):
            ax = self.plt.subplot(nrow, ncol, 2*ncol + i + 1)
            stats = qaD[name]
            tval = stats['tval']
            fval = stats['fval']
            if tval.size > 0:
                resid = fval - tval
                ax.hist(resid, bins=100, color=colors[name], alpha=0.6, edgecolor=None)
                ax.axvline(0.0, linestyle='--', color='black', linewidth=0.8)
                # stats box (reuse top-row mean/std)
                ax.text(0.95, 0.90, 'N=%d\nmean=%.3f\nstd=%.3f'%(tval.shape[0], stats['res_mean'], stats['res_std']),
                        transform=ax.transAxes, fontsize=9, va='top', ha='right')
            ax.grid(True, alpha=0.3)
            sub_tit='%s residuals'%(name.capitalize())
            ax.set(title=sub_tit,
                   xlabel='fit - true',
                   ylabel='count')
        
        # Title for the figure
        self.plt.suptitle(fig_tit)
        self.plt.tight_layout(rect=[0,0,1,0.95])

#...!...!..................
    def correlation_for_kris(self, fitD, trueD, md, figId=1):
        figId=self.smart_append(figId)
        nrow,ncol=2,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(14,6))
        
        # Metadata and title
        fmd = md['fit_uoi']
        data_short = md.get('short_name','')
        samples_k = int(fmd.get('num_samples_used', 0))//1000
        fdr_rate = fmd.get('fdr_rate', md.get('conf_uoi', {}).get('fdr_rate', None))
        fig_tit = 'UoI-Poisson fit   %s  samples=%dk   fdr=%.3f' % (data_short, samples_k, float(fdr_rate))
        
        # Extract arrays
        fitType = md['fit_type']
        A_fit = fitD['A_'+fitType]
        B_fit = fitD['B_'+fitType]
        A_true = trueD['A_true']
        B_true = trueD['B_true']
        
        # QA stats
        qaD = qa_Afit(A_true, A_fit)
        qaD['bterm'] = qa_Bfit(B_true, B_fit)
        
        colors = {'exc': 'red', 'inh': 'blue', 'diag': 'brown', 'bterm': 'salmon'}
        names = ['inh', 'exc', 'diag', 'bterm']
        
        # Row 1: true vs fitted
        for i, name in enumerate(names):
            ax = self.plt.subplot(nrow, ncol, i+1)
            stats = qaD[name]
            color = colors[name]
            tval = stats['tval']; fval = stats['fval']
            if tval.size > 0:
                ax.scatter(tval, fval, alpha=0.6, s=10, c=color)
                lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
                ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
                ax.set_aspect('equal', 'box'); ax.set_xlim(lims); ax.set_ylim(lims)
                ax.plot(np.mean(tval), np.mean(fval), '+', c='black', markersize=18, markeredgewidth=3)
            ax.grid(True, alpha=0.3)
            sub_tit=name.capitalize()
            ax.set(title=sub_tit, xlabel='True Value', ylabel='Fitted Value')
            # lightweight stats text (no frame)
            ax.text(0.95, 0.07, 'N=%d\nmean=%.3f\nstd=%.3f'%(tval.shape[0], stats['res_mean'], stats['res_std']),
                    transform=ax.transAxes, fontsize=9, va='bottom', ha='right')
            if name in ['exc','inh']:
                ax.text(0.05, 0.95, 'TP=%d\nFP=%d\nFN=%d'%(stats['TP'], stats['FP'], stats['FN']),
                        transform=ax.transAxes, fontsize=9, va='top', ha='left')
        
        # Row 2: residuals vs true B_idle
        for i, name in enumerate(names):
            ax = self.plt.subplot(nrow, ncol, ncol + i + 1)
            stats = qaD[name]
            color = colors[name]
            tval = stats['tval']; fval = stats['fval']
            resid = fval - tval
            if tval.size > 0:
                if name in ['exc','inh']:
                    diag_m = np.eye(A_true.shape[0], dtype=bool)
                    if name == 'exc':
                        true_m = (A_true > 0) & (~diag_m)
                        pred_m = (A_fit > 0) & (~diag_m)
                    else:
                        true_m = (A_true < 0) & (~diag_m)
                        pred_m = (A_fit < 0) & (~diag_m)
                    tp_m = true_m & pred_m
                    _, cols = np.where(tp_m)
                    xvals = B_true[cols]
                elif name == 'diag':
                    xvals = B_true
                else:
                    xvals = B_true
                ax.scatter(xvals, resid[:xvals.shape[0]], alpha=0.5, s=12, c=color)
                ax.axhline(0.0, linestyle='--', color='gray', linewidth=0.8)
            ax.grid(True, alpha=0.3)
            sub_tit='%s residuals vs true B'%(name.capitalize())
            ax.set(title=sub_tit, xlabel='true B_idle', ylabel='fit - true')
        
        self.plt.suptitle(fig_tit)
        self.plt.tight_layout(rect=[0,0,1,0.94])


#............................
#............................
#............................


 
    
    
    
