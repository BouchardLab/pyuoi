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
from UtilBioExp import clip_rebD_time
from matplotlib import cm as cmap
import matplotlib.ticker as ticker
from pprint import pprint
import numpy as np
import matplotlib.gridspec as gridspec

from matplotlib.colors import LinearSegmentedColormap

#...!...!....................
def summary_column(md):
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
      
    def plot_bioexp_sum_rate(self, ax, rebD, clip):
        timeV = clip["timeV"]
        rate1D = clip["rate1D"]
        highChanMask = clip["highChanMask"]
        Tbin = clip["Tbin"]
        tL, tR = clip["tL"], clip["tR"]
        nchan = clip["nchan"]
        ax.bar(timeV + Tbin * 0.5, rate1D, width=Tbin, color='orange', align='center', alpha=0.7)
        ax.bar(timeV + Tbin * 0.5, rate1D * highChanMask, width=Tbin,
               color='red', align='center', alpha=0.7)
        ax.set(ylabel='sum rate (Hz)', title=f'sum rate from all {nchan} neurons')
        ax.set_xlim(tL, tR)
        ax.grid()

    def plot_bioexp_heatmap(self, fig, ax, rebD, dataset_name, clip):
        rate2D = clip["rate2D"]
        tL, tR = clip["tL"], clip["tR"]
        nchan = clip["nchan"]
        time_step = clip["time_step"]
        tit0 = f'dataset: {dataset_name}    nchan={nchan}  Tbin={time_step:.2f} sec'

        original_cmap = self.plt.get_cmap('tab20c')
        custom_cmap = original_cmap.copy()
        custom_cmap.set_under('white')

        cax = ax.imshow(
            rate2D.T, aspect='auto', cmap=custom_cmap, origin='lower',
            extent=[tL, tR, 0, nchan], interpolation='nearest',
            vmin=1, vmax=rate2D.max(),
        )
        ax.set_ylabel('Neuron index')
        ax.set_xlabel('Time (sec)')
        ax.set_title(tit0)
        cbar = fig.colorbar(
            cax, ax=ax, orientation='horizontal', pad=0.17,
            label=f'Instantanous freq  (Hz) per neuron over Tbin={time_step:.1f} sec',
            shrink=0.5,
        )
        cbar.ax.tick_params(labelsize=10)
        ax.grid()

    def freq_vs_time(self,rebD,md,figId=2):
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,11))

        rateThr2=rebD['rate_thres2']
        highCntThr=rebD['high_cnt_thres']
        clip = clip_rebD_time(rebD, md['plot']['time_rangeLR'])
        print('iTL,R', clip['itL'], clip['itR'])

        highChanCnt = rebD['highChanCnt'][clip['itL']:clip['itR']]
        highChanMask = clip['highChanMask']
        timeV = clip['timeV']
        Tbin = clip['Tbin']
        tL, tR = clip['tL'], clip['tR']
        nchan = clip['nchan']

        gs = fig.add_gridspec(4, 1, height_ratios=[0.15, 0.15, 0.54, 0.01])

        ax = fig.add_subplot(gs[0, 0])
        ax.bar(timeV+Tbin*.5, highChanCnt, width=Tbin, color='forestgreen', align='center', alpha=0.7)
        ax.bar(timeV+Tbin*.5, highChanCnt*highChanMask, width=Tbin, color='red', align='center', alpha=0.7)
        ax.axhline(highCntThr, c='g', lw=1, ls='--')
        tit = (f'num neurons with instantaneous rate > thres={rateThr2:.0f} (Hz), '
               f'cnt thres={highCntThr}, usable time frac:{rebD["usable_time_fract"]:.3f}  nchan={nchan}')
        ax.set(ylabel='num neurons', title=tit)
        ax.set_xlim(tL, tR)
        ax.grid()

        ax = fig.add_subplot(gs[1, 0])
        self.plot_bioexp_sum_rate(ax, rebD, clip)

        ax = fig.add_subplot(gs[2, 0])
        self.plot_bioexp_heatmap(fig, ax, rebD, md['short_name'], clip)


