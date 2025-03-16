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
 
#...!...!..................
def plot_spike_frequencies(spike_freq, Twindow,plt,figId,tit0):
    """
    Plots a 2D color map (upper plot) and a mean frequency line plot with ±1 std dev shading (bottom plot).
    
    Parameters:
    - spike_freq: 2D NumPy array of shape (nfeat, ntime)
    - Twindow: Scaling factor for the x-axis in the heatmap
    """
    nfeat, ntime = spike_freq.shape

    # Create 3 subplots: (color bar, heatmap, mean freq line plot)
    fig, axes = plt.subplots(nrows=3, figsize=(10, 8), gridspec_kw={'height_ratios': [0.2, 4, 1]}, 
                             num=figId)  # Removed sharex=True

    # Mask values where spike_freq <= 0.5
    min_freq = 0.5
    spike_freq_masked = np.where(spike_freq > min_freq, spike_freq, np.nan)

    # Middle plot: 2D colormap (heatmap) with swapped axes
    vmin, vmax = np.nanmin(spike_freq_masked), np.nanmax(spike_freq_masked)  # Get value range
    cax = axes[1].imshow(spike_freq_masked.T, aspect='auto', cmap='inferno_r', origin='lower',  
                          extent=[0, nfeat, 0, ntime * Twindow], vmin=vmin, vmax=vmax)

    axes[1].set_xlabel('Feature Index')  # X-axis is feature index
    axes[1].set_ylabel('Time (sec)')  # Y-axis is time
    axes[1].set_title('%s   Spike Frequency, Integration T-Window %d sec' % (tit0, Twindow))

    # Bottom plot: Line plot with ±1 std deviation shading
    mean_spike_freq = spike_freq.mean(axis=1)  # Mean per feature over time
    std_spike_freq = spike_freq.std(axis=1)    # Std dev per feature over time

    x_values = np.arange(nfeat)  # Feature index

    axes[2].plot(x_values, mean_spike_freq, color='blue', label='Mean Frequency')
    axes[2].fill_between(x_values, mean_spike_freq - std_spike_freq, mean_spike_freq + std_spike_freq,
                         color='salmon', alpha=0.5, label='±1 Std Dev')

    axes[2].set_xlabel('Feature Index')  # X-axis is aligned for both plots
    axes[2].set_ylabel('Averaged Frequency')
    axes[2].set_title('Mean Spike Frequency per Feature with ±1 Std Dev')
    axes[2].legend()

    # Top plot: Color bar in a dedicated axis
    cbar = fig.colorbar(cax, cax=axes[0], orientation='horizontal')

    # Ensure the color bar's x-axis labels & title are visible
    cbar.ax.xaxis.set_ticks_position('bottom')  # Move ticks to top
    cbar.ax.xaxis.set_label_position('bottom')  # Move label to top
    cbar.ax.tick_params(axis='x', direction='out')  # Make ticks clearly visible

    # Set label for color bar with padding    
    cbar.ax.set_title("Frequency (Hz) , cut-off thres=%.1f (Hz)"%min_freq)

    # Manually set tick labels for better visibility
    tick_positions = np.linspace(vmin, vmax, num=5)  # 5 evenly spaced ticks
    cbar.set_ticks(tick_positions)
    cbar.ax.set_xticklabels([f"{t:.2f}" for t in tick_positions], fontsize=10)  # Ensure tick labels are shown

    
   
#...!...!..................
def plot_histogram(ax, data, percentile_low=5, percentile_high=95):
    """Plot histogram of the difference and annotate mean, median, and percentiles."""

    ax.hist(data, bins=30, color='salmon', alpha=0.7)

    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)

    # Compute standard error of the standard deviation estimator
    N = data.shape[0]
    se_s = std / np.sqrt(2 * (N - 1))

    # Compute percentiles
    p_low = np.percentile(data, percentile_low)
    p_high = np.percentile(data, percentile_high)

    # Add vertical lines for mean, median, and percentiles
    ax.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
    ax.axvline(median, color='b', linestyle='dashed', linewidth=1, label='Median')
    ax.axvline(p_low, color='g', linestyle='dotted', linewidth=1, label=f'{percentile_low}th Percentile')
    ax.axvline(p_high, color='g', linestyle='dotted', linewidth=1, label=f'{percentile_high}th Percentile')

    # Annotate statistics
    txt = f"Mean: {mean:.3f}\nMedian: {median:.3f}\nRMSE: {std:.3f} ± {se_s:.3f}\n"
    txt += f"{percentile_low}th: {p_low:.3f}\n{percentile_high}th: {p_high:.3f}"
    ax.annotate(txt, xy=(0.35, 0.75), color='black', xycoords='axes fraction')

    ax.legend()
 
#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)
        
#...!...!..................
    def input_features(self,bigD,md,figId=1):
        pprint(md)
        pmd=md['payload']
        plm=md['plot']
        nfeat=min(10,pmd['num_feature'])
        ntime=pmd['num_time_bin']
        
        figId=self.smart_append(figId)        
        nrow,ncol=nfeat,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,1.2*nrow))

        timeV=bigD['time']
        
        width =0.0005 
        for k in range(nrow):
            ax = self.plt.subplot(nrow,ncol,1+k)
            j=k
            featV=bigD['feature'][j]
            spikeV=bigD['spike'][j].astype(float)
            # Plot Exponential Decay as Filled Area
            ax.fill_between(timeV, 0,featV , color='red', alpha=0.3, label='Exponential Decay')
            
            #.... decorations
            ax.grid()
            if 'time_rangeLR' in plm:  ax.set_xlim(tuple(plm['time_rangeLR']))

            if k==nrow-1: ax.set_xlabel('Time (s)')
            ax.set_ylabel('F=%d'%j)
            print('draw F=',j)
        return

        # .... decorations ....
        # Overlay the text on top of the plots
        txt=summary_column(md)
        ax.text(0.88, 0.95, txt, fontsize=10, color='m', ha='left', va='top',transform=ax.transAxes)

#...!...!..................
    def global_qa(self,bigD,md,figId=3):
        #pprint(md)
        pmd=md['payload']
        
        tit='session '+md['short_name']
        freqV=bigD['avr_spike_freq']
        
        figId=self.smart_append(figId)        
        nrow,ncol=2,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(8,8))

        ax = self.plt.subplot(nrow,ncol,1)
        xLab='input feature index'
        ax.plot(freqV)
        ax.set(xlabel=xLab, ylabel='spike frequency (Hz)',title=tit)
        ax.grid()

        ax = self.plt.subplot(nrow,ncol,2)
        plot_histogram(ax, freqV)
        ax.set_yscale('log')
        ax.set(xlabel='spike frequency (Hz)', ylabel='num features',title=tit)
        ax.grid()
        
#...!...!..................
    def detailed_qa(self,bigD,md,figId=3):

        pmd=md['payload']
        tit=md['short_name']+' session, '

        figId=self.smart_append(figId)        
        nrow,ncol=1,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,14))
        
        plot_spike_frequencies(bigD['spike_freq'],pmd['qa_twindow_sec'],self.plt,figId,tit)
    

        
#...!...!..................
    def xyz(self,bigD,md,figId=3):
        #pprint(md)
        pmd=md['payload']
        smd=md['submit']
        tmd=md['transpile']

        figId=self.smart_append(figId)        
        nrow,ncol=1,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,4))

        make_it_work

#...!...!..................
    def input_features_dense(self,bigD,md,figId=1):
        pprint(md)
        pmd=md['payload']
        plm=md['plot']
        nfeat=min(9,pmd['num_feature'])
        ntime=pmd['num_time_bin']
        nrow,ncol=nfeat,1
         
        axes=self.blank_share2D(nrow=nrow,ncol=ncol, figsize=(12,1.5*nrow),figId=figId)
        #axes=self.blank_share2D(nrow=nrow,ncol=ncol, figsize=(20,0.6*nrow),figId=figId)  

        timeV=bigD['time']
        featIdL=bigD['feature_id']
        width =0.0005 
        for k in range(nrow):
            ax = axes[k]
            j=k+1
            featV=bigD['feature'][j]
            fid=featIdL[k]
            spikeV=bigD['spike'][j].astype(float)
            # Plot Exponential Decay as Filled Area
            ax.fill_between(timeV, 0,featV , color='red', alpha=0.3, label='feature=%d'%fid)

            # .... decorations ....
            ax.legend()
            ax.set_ylim(0,2.1)
            ax.set_ylabel('Ampl')
            if k==0:ax.set_title('Spikes with Exponential Decay, data=%s'%md['short_name'] )
            
            
        # common
        if 'time_rangeLR' in plm:  ax.set_xlim(tuple(plm['time_rangeLR']))
        ax.set_xlabel('Time (s)')
