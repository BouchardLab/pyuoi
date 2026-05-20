#!/usr/bin/env python3
"""
Plotting utilities for biological experiment data visualization.

This module provides specialized plotting capabilities for experimental
neural data analysis. The Plotter class extends PlotterBackbone to create
visualizations tailored for biological neural recordings including:
- Time series plots of neural activity patterns
- Statistical summaries and distribution analysis  
- Data quality assessment plots
- Comparative analysis between experimental conditions

Designed specifically for processing and visualizing data from biological
neural experiments, with automatic adaptation to experimental metadata.
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from toolbox.PlotterBackbone import PlotterBackbone
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

#...!...!....................
def summary_column(md):
    return 'fix-me 32678'
    #print(sorted(md))
    rs=md['rate_summary']
    ds=md['data_selector']
    #pprint(md)
    txt='dataset: '+md['short_name']
    txt += '\nnum acc neurons: %d' % md['num_neurons']
    txt += '\ndrop neur: %d <%.1f Hz,  %d >%.1f Hz' % (ds['drop_neur_by_freq_range'][0],ds['freq_range'][0],ds['drop_neur_by_freq_range'][1],ds['freq_range'][1])
    txt+='\ndata type: %s\ntime_step=%.2f sec'%(md['data_type'],md['time_step_sec']) 
    txt += '\nMedian rate: %.1f Hz' % rs['median_spike_rate']
    txt += '\nrate range [%.1f, %.1f Hz]' % (rs['min_spike_rate'], rs['max_spike_rate'])
    txt += '\nAvg Rate: %.1f±%.1f Hz' % (rs['avg_spike_rate'], rs['std_spike_rate'])
    txt += '\nAvg Fano: %.2f±%.2f' % (rs['avg_fano_factor'], rs['std_fano_factor'])

    return txt
  
#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)
        
#...!...!..................
    def freq_histo(self,spikeD,md,figId=1):
        
        tit='dataset '+md['short_name']
        
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,4))
        
        # Create gridspec with 7:3 width ratio
        gs = gridspec.GridSpec(1, 2, width_ratios=[7, 3])

        dataYield, dataRates = spikeD['spikes'], spikeD['single_rates']

        # Compute median
        median_val = np.median(dataRates)
        txtM= f'median : {median_val:.2f} Hz'
        
        #.... freq per channel
        ax = self.plt.subplot(gs[0, 0])
        chanV=np.arange(dataRates.shape[0])
        ax.bar(chanV, dataRates , width=0.8, color='g', align='center', alpha=0.7)
         
        ax.axhline(median_val, color='red', linestyle='--', linewidth=1)
        if md['data_type']=='simDale':
            ax.set_yscale('log')
        ax.grid()
        ax.set(xlabel='input neuron index', ylabel='avr frequency (Hz)',title=tit)
        ax.text( ax.get_xlim()[1]*0.1,median_val,txtM, 
                 color='red', ha='center', va='bottom', fontsize=10)

        txt=summary_column(md)
        ax.text(0.05, 0.3,   txt, transform=ax.transAxes, fontsize=12,color='b')
        
        #........ freq histo .....
        ax = self.plt.subplot(gs[0, 1])
        ax.hist(dataRates,bins=20)
        #ax.set_yscale('log')
        ax.set(ylabel='num channels', xlabel='avr frequency (Hz)',title=tit)
        ax.axvline(median_val, color='red', linestyle='--', linewidth=1)
        # Add median text annotation
        ax.text(median_val+1, ax.get_ylim()[1]*0.5,txtM, 
                color='red', ha='left', va='bottom', fontsize=10,rotation=90)
        ax.text(0.25, 0.3,   txt, transform=ax.transAxes, fontsize=11,color='m')
      
        ax.grid()
      
#...!...!..................
    def freq_vs_time(self,rebD,md,figId=2,S_true=None,S_oracle=None):
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,11))
        if S_oracle is None and isinstance(rebD, dict) and 'S_oracle' in rebD:
            S_oracle = rebD['S_oracle']

        tit='dataset '+md['short_name']
        time_step=rebD['time_step2']
        R_sel = md['sel_spect_radius']
        state_tag = md['sel_state']
         
        # clip time data for display      
        tL,tR=md['plot']['time_rangeLR']
        itL,itR=(md['plot']['time_rangeLR']/time_step).astype(int)
        ntime_tot = rebD['rate2D'].shape[0]
        itL = max(0, min(itL, ntime_tot - 1))
        itR = max(itL + 1, min(itR, ntime_tot))
        print('iTL,R', itL,itR)

        # unpack data and clip the time range
        rate2D=rebD['rate2D'][itL:itR]
        timeV=rebD['timeV'][itL:itR]

        _,nchan=rate2D.shape
        Tbin = time_step
        xL = float(timeV[0])
        xR = float(timeV[-1] + Tbin)
        if isinstance(rebD, dict) and 'pop_rate_hz' in rebD:
            pop_rate_hz = np.asarray(rebD['pop_rate_hz'][itL:itR], dtype=np.float64)
        else:
            pop_rate_hz = np.sum(rate2D, axis=1)
        medRateDisp = float(np.median(rate2D))
        cntAboveMed = np.sum(rate2D > medRateDisp, axis=1)
           
        tit0='dataset: %s  state=%d  R=%.3f    nchan=%d  Tbin=%.2f sec'%(md['short_name'], state_tag, R_sel, nchan, time_step)
        
        # Layout: top-count trace, heatmap, synchronicity trace, target-state, oracle-state.
        gs = fig.add_gridspec(5, 1, height_ratios=[0.14, 0.52, 0.14, 0.10, 0.10], hspace=0.36)

        # ..... top plot
        ax = fig.add_subplot(gs[0, 0])
        ax.bar(timeV+Tbin*.5, cntAboveMed, width=Tbin, color='teal', align='center', alpha=0.7)
        ax.set(ylabel='neurons > median', title=f'neurons above displayed-time median rate ({medRateDisp:.2f} Hz)')
        ax.set_xlim(xL,xR)
        ax.tick_params(axis='x', labelbottom=False)
        ax.grid()
        
        # ....... main heatmap
        ax = fig.add_subplot(gs[1, 0])
        #cmap='tab20c', 'Oranges'
        # - - -  Get the colormap and modify it ---
        original_cmap = self.plt.get_cmap('tab20c')
        custom_cmap = original_cmap.copy()
        custom_cmap.set_under('white')

        cax = ax.imshow(rate2D.T,
                aspect='auto',
                cmap=custom_cmap,      # Use our modified colormap       
                origin='lower',
                extent=[xL, xR, 0, nchan],
                interpolation='nearest',
                vmin=1,              # Set the lower bound of the colormap
                vmax=rate2D.max())     # Optional: ensure the upper bound is set

        ax.set_ylabel('Neuron index')
        ax.set_title(tit0   )
        ax.tick_params(axis='x', labelbottom=False)
        cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', pad=0.02, label=f'Instantanous freq  (Hz) per neuron over Tbin={time_step:.1f} sec', shrink=0.55)
        cbar.ax.tick_params(labelsize=10)  
        ax.grid()

        # ..... synchronicity (yellow) below heatmap
        ax = fig.add_subplot(gs[2, 0])
        ax.bar(timeV+Tbin*.5, pop_rate_hz, width=Tbin, color='gold', align='center', alpha=0.8)
        ax.set(ylabel='synchronicity (Hz)', title=f'Network synchronicity from {nchan} neurons, Tbin={Tbin:.2f} sec')
        ax.set_xlim(xL,xR)
        ax.tick_params(axis='x', labelbottom=True)
        ax.grid()

        # ..... target state
        ax = fig.add_subplot(gs[3, 0])
        ax.set_xlim(xL, xR)
        if S_true is not None:
            assert S_true.ndim == 1, f"S_true must be 1D, got shape={S_true.shape}"
            dt0 = float(md['time_step_sec'])
            t_state = np.arange(S_true.shape[0], dtype=float) * dt0
            sel = (t_state >= xL) & (t_state < xR + dt0)
            if np.any(sel):
                t_sel = t_state[sel]
                s_sel = S_true[sel]
                ax.step(t_sel, s_sel, where='post', color='k', linewidth=1.0)
                smin = int(np.min(s_sel))
                smax = int(np.max(s_sel))
                ax.set_ylim(smin - 0.5, smax + 0.5)
                ax.set_yticks(np.arange(smin, smax + 1, 1))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylabel('state S')
            ax.set_title(f"target state — {md['short_name']}")
            ax.grid()
        else:
            ax.set_axis_off()
        ax.tick_params(axis='x', labelbottom=False)

        # ..... oracle state
        ax = fig.add_subplot(gs[4, 0])
        ax.set_xlim(xL, xR)
        if S_oracle is not None:
            assert S_oracle.ndim == 1, f"S_oracle must be 1D, got shape={S_oracle.shape}"
            score_txt = f", avr_score={md['oracle_score']:.3f}"
            dt0 = float(md['time_step_sec'])
            t_state = np.arange(S_oracle.shape[0], dtype=float) * dt0
            sel = (t_state >= xL) & (t_state < xR + dt0)
            if np.any(sel):
                t_sel = t_state[sel]
                s_sel = S_oracle[sel]
                ax.step(t_sel, s_sel, where='post', color='g', linewidth=1.0)
                smin = int(np.min(s_sel))
                smax = int(np.max(s_sel))
                ax.set_ylim(smin - 0.5, smax + 0.5)
                ax.set_yticks(np.arange(smin, smax + 1, 1))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylabel('oracle S')
            ax.set_title(f"oracle state{score_txt}")
            ax.grid()
        else:
            ax.set_axis_off()
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_xlabel('Time (s)')
