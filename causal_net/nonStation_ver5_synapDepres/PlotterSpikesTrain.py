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
    def freq_vs_time(self,rebD,md,figId=2):
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(16, 7.7))

        time_step=rebD['time_step2']
        R_sel = md['sel_spect_radius']

        # clip time data for display
        itL, itR = (md['plot']['time_rangeLR'] / time_step).astype(int)
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
        # Median of per-neuron time-averaged rate (global median of all T×N bins is ~0 when activity is sparse).
        mean_per_neuron_hz = np.mean(rate2D, axis=0)
        medRateDisp = float(np.median(mean_per_neuron_hz))
        cntAboveMed = np.sum(rate2D > medRateDisp, axis=1)
        med_sync_hz = float(np.median(pop_rate_hz))

        tit0 = 'dataset: %s  R=%.3f    nchan=%d  Tbin=%.2f sec' % (
            md['short_name'],
            R_sel,
            nchan,
            time_step,
        )
        if (
            md.get('spike_model') == 'B'
            and 'mem_Q' in md
            and 'mem_tau' in md
        ):
            tit0 += '  Q=%.4g  tau/sec=%.4g' % (
                float(md['mem_Q']),
                float(md['mem_tau']),
            )
        tit0 += '  placement HxL=(%gx%g)' % (
            float(md['placement_H']),
            float(md['placement_L']),
        )

        # Layout: top-count trace, heatmap, synchronicity trace.
        gs = fig.add_gridspec(3, 1, height_ratios=[0.15, 0.55, 0.22], hspace=0.36)

        # ..... top plot
        ax = fig.add_subplot(gs[0, 0])
        ax.bar(timeV+Tbin*.5, cntAboveMed, width=Tbin, color='teal', align='center', alpha=0.7)
        ax.set(
            ylabel='neurons > median',
            title=f'neurons above median mean rate ({medRateDisp:.2f} Hz)',
        )
        ax.set_xlim(xL,xR)
        ax.set_xlabel('Time (s)')
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
        ax.set_xlabel('Time (s)')
        ax.tick_params(axis='x', labelbottom=False)
        cbar = fig.colorbar(
            cax, ax=ax, orientation='horizontal', pad=0.08, shrink=0.55,
            aspect=40, anchor=(0.0, 0.5)
        )
        cbar.ax.text(
            1.02, 0.5, f'Instantaneous freq  (Hz) per neuron over Tbin={time_step:.2f} sec',
            transform=cbar.ax.transAxes, va='center', ha='left', fontsize=10
        )
        cbar.ax.tick_params(labelsize=10)  
        ax.grid()

        # ..... synchronicity (yellow) below heatmap
        ax = fig.add_subplot(gs[2, 0])
        ax.bar(timeV+Tbin*.5, pop_rate_hz, width=Tbin, color='gold', align='center', alpha=0.8)
        ax.axhline(med_sync_hz, color='red', linestyle='--', linewidth=1.0, zorder=3)
        y_rng = float(np.max(pop_rate_hz) - np.min(pop_rate_hz))
        y_pad = max(0.02 * y_rng, 0.01 * max(float(np.max(pop_rate_hz)), 1.0))
        ax.text(
            xL + 0.01 * (xR - xL),
            med_sync_hz + y_pad,
            'median',
            va='bottom',
            ha='left',
            color='red',
            fontsize=9,
            zorder=4,
        )
        ax.set(
            ylabel='synchronicity (Hz)',
            title=(
                f'Network synchronicity from {nchan} neurons, Tbin={Tbin:.2f} sec, '
                f'median rate={med_sync_hz:.2f} Hz'
            ),
        )
        ax.set_xlim(xL,xR)
        ax.tick_params(axis='x', labelbottom=True)
        ax.grid()
        ax.set_xlabel('Time (s)')

#...!...!..................
    def state_x_vs_time(self, rebD, md, figId=3):
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(16, 7.7))

        time_step = rebD['time_step2']
        state2D_all = np.asarray(rebD['state2D'], dtype=np.float64)

        # clip time data for display, matching freq_vs_time()
        itL, itR = (md['plot']['time_rangeLR'] / time_step).astype(int)
        ntime_tot = state2D_all.shape[0]
        itL = max(0, min(itL, ntime_tot - 1))
        itR = max(itL + 1, min(itR, ntime_tot))
        print('x-state iTL,R', itL, itR)

        x2D = state2D_all[itL:itR]
        timeV = rebD['timeV'][itL:itR]

        _, nchan = x2D.shape
        Tbin = time_step
        xL = float(timeV[0])
        xR = float(timeV[-1] + Tbin)

        mean_x = np.mean(x2D, axis=1)
        median_x = np.median(x2D, axis=1)
        pct_lo_x = np.percentile(x2D, 10, axis=1)
        pct_hi_x = np.percentile(x2D, 90, axis=1)
        med_x = float(np.median(x2D))
        cnt_above_med = np.sum(x2D > med_x, axis=1)

        R_sel = md['sel_spect_radius']
        tit0 = 'dataset: %s  R=%.3f    nchan=%d  Tbin=%.2f sec' % (
            md['short_name'], float(R_sel), nchan, time_step
        )
        tit0 += r'  STD resource $x_j(t)$'

        gs = fig.add_gridspec(3, 1, height_ratios=[0.15, 0.55, 0.22], hspace=0.36)

        # ..... top plot: count above global displayed median
        ax = fig.add_subplot(gs[0, 0])
        ax.bar(timeV + Tbin * .5, cnt_above_med, width=Tbin, color='lightskyblue',
               align='center', alpha=0.85)
        ax.set(
            ylabel='neurons > median',
            title=r'neurons with $x_j$ above displayed median %.3f' % med_x,
        )
        ax.set_xlim(xL, xR)
        ax.set_xlabel('Time (s)')
        ax.tick_params(axis='x', labelbottom=False)
        ax.grid()

        # ....... main heatmap
        ax = fig.add_subplot(gs[1, 0])
        cax = ax.imshow(
            x2D.T,
            aspect='auto',
            cmap='viridis',
            origin='lower',
            extent=[xL, xR, 0, nchan],
            interpolation='nearest',
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_ylabel('Neuron index')
        ax.set_title(tit0)
        ax.set_xlabel('Time (s)')
        ax.tick_params(axis='x', labelbottom=False)
        cbar = fig.colorbar(
            cax, ax=ax, orientation='horizontal', pad=0.08, shrink=0.55,
            aspect=40, anchor=(0.0, 0.5)
        )
        cbar.ax.text(
            1.02, 0.5, r'rebinned STD resource $x_j$',
            transform=cbar.ax.transAxes, va='center', ha='left', fontsize=10
        )
        cbar.ax.tick_params(labelsize=10)
        ax.grid()

        # ..... bottom plot: population median with central 40 percentile envelope
        ax = fig.add_subplot(gs[2, 0])
        ax.bar(timeV + Tbin * .5, median_x, width=Tbin, color='salmon',
               align='center', alpha=0.65, label=r'median $x$')
        ax.plot(timeV + Tbin * .5, pct_hi_x, color='firebrick',
                linestyle='-', linewidth=0.8, label=r'10-90 percentile')
        ax.plot(timeV + Tbin * .5, pct_lo_x, color='firebrick',
                linestyle='-', linewidth=0.8)
        ax.axhline(med_x, color='black', linestyle='--', linewidth=1.0,
                   label='median %.3f' % med_x)
        ax.set(
            ylabel=r'population $x$',
            title=r'Population STD resource, median %.3f' % (
                med_x
            ),
        )
        ax.set_xlim(xL, xR)
        ax.set_ylim(0.0, 1.02)
        ax.tick_params(axis='x', labelbottom=True)
        ax.grid()
        ax.legend(loc='best', fontsize=9)
        ax.set_xlabel('Time (s)')
