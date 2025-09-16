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
from matplotlib import cm as cmap
import matplotlib.ticker as ticker
from pprint import pprint
import numpy as np
import matplotlib.gridspec as gridspec

from matplotlib.colors import LinearSegmentedColormap

#...!...!....................
def summary_column(md):
    #pprint(md)
    txt='dataset: '+md['short_name']
    txt += '\nnum neurons: %d' % md['num_neurons']
    txt+='\ndata type: %s\n time_step=%.2f sec'%(md['data_type'],md['time_step_sec']) 
    txt += '\nAvg Rate: %.2f±%.2f Hz' % (md['avg_spike_rate'], md['std_spike_rate'])
    txt += '\nAvg Fano: %.2f±%.2f' % (md['avg_fano_factor'], md['std_fano_factor'])
    txt += '\nMedian rate: %.2f Hz' % md['median_spike_rate']
    txt += '\nMin/Max rate: %.2f/%.2f Hz' % (md['min_spike_rate'], md['max_spike_rate'])

    return txt
  
   
  
#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)
        
#...!...!..................
    def freq_histo(self,spikeD,md,figId=1):
        pprint(md)
 
        tit='dataset '+md['short_name']
        
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,4))
        
        # Create gridspec with 7:3 width ratio
        gs = gridspec.GridSpec(1, 2, width_ratios=[7, 3])

        dataYield, dataRates = spikeD['spikes'], spikeD['single_rates']

        # Compute median
        median_val = np.median(dataRates)
        txtM= f'median rate: {median_val:.2f} Hz'
        
        #.... freq per channel
        ax = self.plt.subplot(gs[0, 0])
        chanV=np.arange(dataRates.shape[0])
        ax.bar(chanV, dataRates , width=0.8, color='g', align='center', alpha=0.7)
         
        ax.axhline(median_val, color='red', linestyle='--', linewidth=1)
        if md['data_type']=='simDale':
            ax.set_yscale('log')
        ax.grid()
        ax.set(xlabel='input channel', ylabel='avr frequency (Hz)',title=tit)
        ax.text( ax.get_xlim()[1]*0.1,median_val,txtM, 
                 color='red', ha='center', va='bottom', fontsize=10)

        txt=summary_column(md)
        ax.text(0.05, 0.5,   txt, transform=ax.transAxes, fontsize=10,color='b')
        
        #........ freq histo .....
        ax = self.plt.subplot(gs[0, 1])
        ax.hist(dataRates,bins=20)
        #ax.set_yscale('log')
        ax.set(ylabel='num channels', xlabel='avr frequency (Hz)')
        ax.axvline(median_val, color='red', linestyle='--', linewidth=1)
        # Add median text annotation
        ax.text(median_val, ax.get_ylim()[1]*0.7,txtM, 
                color='red', ha='left', va='bottom', fontsize=10)

        ax.grid()
      
#...!...!..................
    def freq_vs_time(self,rebD,md,figId=2):
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,11))

        tit='dataset '+md['short_name']
        time_step=rebD['time_step2']
        rateThr2=rebD['rate_thres2']
        highCntThr=rebD['high_cnt_thres']
         
        # clip time data for display      
        tL,tR=md['plot']['time_rangeLR']
        itL,itR=(md['plot']['time_rangeLR']/time_step).astype(int)
        print('iTL,R', itL,itR)

        # unpack data and clip the time range
        rate2D=rebD['rate2D'][itL:itR]
        highChanCnt=rebD['highChanCnt'][itL:itR]
        highChanSm=rebD['highChanSmooth'][itL:itR]
        highChanMask=rebD['highChanMask'][itL:itR]
         
        rate1D=rebD['rate1D'][itL:itR]
        timeV=rebD['timeV'][itL:itR]

        _,nchan=rate2D.shape
        Tbin = (timeV[1] - timeV[0])  
           
        tit0='dataset: %s    nchan=%d  Tbin=%.2f sec'%(md['short_name'],nchan,time_step)
        
        # Create gridspec with 2 rows in a 0.6:0.4 ratio
        gs = fig.add_gridspec(5, 1, height_ratios=[0.15,0.15,0.15, 0.54,0.01])        
        
        # ..... top plot
        ax = fig.add_subplot(gs[1, 0])
        ax.bar(timeV+Tbin*.5, highChanSm, width=Tbin, color='slateblue', align='center', alpha=0.7)
        ax.bar(timeV+Tbin*.5, highChanSm*highChanMask , width=Tbin, color='red', align='center', alpha=0.7)
        tit=f'num neurons w/ smooth kernel {rebD["smooth_kernel"]} ,  rate  thres> {rateThr2:.0f} (Hz),  usable time frac:{rebD["usable_time_fract"]:.3f}   nchan={nchan}'
        ax.set(ylabel='num neurons', title=tit)
        ax.axhline( highCntThr,c='g',lw=1,ls='--')
        ax.set_xlim(tL,tR)
        ax.grid()
        
        # ..... middle1 plot
        ax = fig.add_subplot(gs[2, 0])
      
        ax.bar(timeV+Tbin*.5, highChanCnt, width=Tbin, color='forestgreen', align='center', alpha=0.7)

        ax.bar(timeV+Tbin*.5, highChanCnt*highChanMask , width=Tbin, color='red', align='center', alpha=0.7)
        ax.set(xlabel='Time (s)', ylabel='num neurons', title=f'num neurons with instantanous rate  thres= {rateThr2:.0f} (Hz)')
        ax.set_xlim(tL,tR)
        ax.grid()
        
        # ..... middle2 plot
        ax = fig.add_subplot(gs[0, 0])
        ax.bar(timeV+Tbin*.5, rate1D, width=Tbin, color='orange', align='center', alpha=0.7)
        ax.bar(timeV+Tbin*.5, rate1D*highChanMask , width=Tbin, color='red', align='center', alpha=0.7)
        ax.set( ylabel='sum rate (Hz)', title=f'sum rate from all {nchan} neurons')
        ax.set_xlim(tL,tR)
        ax.grid()
        
        # ....... bottom row: 2D histogram
        ax = fig.add_subplot(gs[3, 0])
        #cmap='tab20c', 'Oranges'
        # - - -  Get the colormap and modify it ---
        original_cmap = self.plt.get_cmap('tab20c')
        custom_cmap = original_cmap.copy()
        custom_cmap.set_under('white')

        cax = ax.imshow(rate2D.T,
                aspect='auto',
                cmap=custom_cmap,      # Use our modified colormap       
                origin='lower',
                extent=[tL, tR, 0, nchan],
                interpolation='nearest',
                vmin=1,              # Set the lower bound of the colormap
                vmax=rate2D.max())     # Optional: ensure the upper bound is set

        ax.set_ylabel('Neuron index')
        ax.set_xlabel('Time (sec)')
        ax.set_title(tit0   )
        cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', pad=0.17, label=f'Instantanous freq  (Hz) per neuron over Tbin={time_step:.1f} sec', shrink=0.5)
        cbar.ax.tick_params(labelsize=10)  
        ax.grid()


