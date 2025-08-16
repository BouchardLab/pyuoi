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
    pmd=md['payload']
    smd=md['submit']
    tmd=md['transpile']
    pom=md['postproc']
    txt=md['short_name']
    txt+='\nback: %s'%smd['backend']
    txt+='\nshots/addr : %d'%(smd['num_shots']/pmd['num_addr'])
    txt+='\nshots/img : %d k'%(smd['num_shots']/1000)
    txt+='\nnum sample %d'%(pmd['num_sample'])
    txt+='\nsample size: %d'%(pmd['seq_len'])
    txt+='\nnum addr: %d'%pmd['num_addr']
    txt+='\nqubits: %d'%pmd['num_qubit']
    if 'ibm' in smd['backend']:  txt+='  RC: %r'%smd['random_compilation']
    txt+='\nnum 2q gates: %d'%tmd['2q_gate_count']
    txt+='\n2q gates depth: %d'%tmd['2q_gate_depth']

    #txt+='\nhwCalib: %s'%pom['hw_calib']
    #if pom['hw_calib']: txt+=' fac: %.2f'%pom['ampl_fact']
    return txt
    if 'noise_model' in smd:
        txt+='\nfake : %s'%(smd['noise_model'])       
 
   
  
#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)
        
#...!...!..................
    def freq_histo(self,spikeD,md,figId=1):
        pprint(md)
 
        tit='session '+md['short_name']
        
        figId=self.smart_append(figId)        
        nrow,ncol=2,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,6))

        dataYield, dataRates = spikeD['spikes'], spikeD['single_rates']

        # Compute median
        median_val = np.median(dataRates)

        #.... freq per channel
        ax = self.plt.subplot(nrow,ncol,1)
        ax.plot(dataRates)
        ax.axhline(median_val, color='red', linestyle='--', linewidth=1)
        ax.set_yscale('log')
        ax.grid()
        ax.set(xlabel='input channel', ylabel='avr frequency (Hz)',title=tit)

        #........ freq histo .....
        ax = self.plt.subplot(nrow,ncol,2)
        ax.hist(dataRates,bins=20)
        ax.set_yscale('log')
        ax.set(ylabel='num channels', xlabel='avr frequency (Hz)')
        ax.axvline(median_val, color='red', linestyle='--', linewidth=1)
        # Add median text annotation
        ax.text(median_val, ax.get_ylim()[1]*0.9, f'median: {median_val:.2f}', 
                color='red', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.grid()
        #ax.set(xlabel=xLab, ylabel='min spike del time (10 kHz bin) ',title=tit)

        # ax.text(0.05, 0.2, txt, transform=ax.transAxes, fontsize=8)
        return

#...!...!..................
    def freq_vs_time(self,rebD,md,figId=2):
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(14,10))

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
           
        tit='session: %s    nchan=%d  Tbin=%.2f sec'%(md['short_name'],nchan,time_step)
 
       
        # Create gridspec with 2 rows in a 0.6:0.4 ratio
        gs = fig.add_gridspec(4, 1, height_ratios=[0.15,0.15,0.15, 0.55])        
        
        # ..... top plot
        ax = fig.add_subplot(gs[0, 0])
        ax.bar(timeV+Tbin*.5, highChanSm, width=Tbin, color='slateblue', align='center', alpha=0.7)
        ax.bar(timeV+Tbin*.5, highChanSm*highChanMask , width=Tbin, color='red', align='center', alpha=0.7)
        tit=f'num neurons w/ smooth kernel {rebD["smooth_kernel"]} ,  rate  thres= {rateThr2:d} (Hz),  usable time frac:{rebD["usable_time_fract"]:.3f}   nchan={nchan}'
        ax.set(xlabel='Time (s)', ylabel='num neurons', title=tit)
        ax.axhline( highCntThr,c='r',lw=1,ls='--')
        ax.set_xlim(tL,tR)
        ax.grid()
        
        # ..... middle1 plot
        ax = fig.add_subplot(gs[1, 0])
      
        ax.bar(timeV+Tbin*.5, highChanCnt, width=Tbin, color='forestgreen', align='center', alpha=0.7)

        ax.bar(timeV+Tbin*.5, highChanCnt*highChanMask , width=Tbin, color='red', align='center', alpha=0.7)
        ax.set(xlabel='Time (s)', ylabel='num neurons', title=f'num neurons with instantanous rate  thres= {rateThr2:d} (Hz)')
        ax.set_xlim(tL,tR)
        ax.grid()
        
        # ..... middle2 plot
        ax = fig.add_subplot(gs[2, 0])
        ax.bar(timeV+Tbin*.5, rate1D, width=Tbin, color='orange', align='center', alpha=0.7)
        ax.bar(timeV+Tbin*.5, rate1D*highChanMask , width=Tbin, color='red', align='center', alpha=0.7)
        ax.set(xlabel='Time (s)', ylabel='sum rate (Hz)', title=f'sum rate from all {nchan} neurons')
        ax.set_xlim(tL,tR)
        ax.grid()
        
        # ....... bottom row: 2D histogram
        ax = fig.add_subplot(gs[3, 0])
        # - - -  Get the colormap and modify it ---
        original_cmap = self.plt.get_cmap('Oranges')
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
        #ax.set_xlabel('Time (sec)')
        ax.set_title(f'Rate (Hz)   over Tbin={time_step:.1f} sec')
        cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', pad=0.10, label='Instantanous freq per neuron (Hz)', shrink=0.4)
        cbar.ax.tick_params(labelsize=10)  
        ax.grid()


