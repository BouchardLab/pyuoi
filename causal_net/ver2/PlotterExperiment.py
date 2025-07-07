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
def add_time_scale_marker(ax):
    # Length of the horizontal line (in x-axis units)
    tLen = 30

    # Position as a fraction of the plot dimensions (10% from left, 50% from bottom)
    x_pos = 0.1
    y_pos = 0.7

    # Get the adjusted x-axis limits
    x_min, x_max = ax.get_xlim()

    # Extracting y-axis limits from the plot
    y_min, y_max = ax.get_ylim()
    
    # Calculating actual positions based on the adjusted x-axis limits
    x_start = x_min + x_pos * (x_max - x_min)
    x_end = x_start + tLen
    y_coord = y_min + y_pos * (y_max - y_min)


    #print('x a,b:',x_start,x_end)
    #print('y :',y_coord)
    # Adding the horizontal line
    ax.plot([x_start, x_end], [y_coord, y_coord], color='blue', linewidth=2)

    # Adding the text just above the blue line
    txt='%d (ms)'%(tLen)
    ax.text((x_start + x_end) / 2, y_coord + (y_max - y_min) * 0.02, txt, 
            color='blue', fontsize=10, ha='center', va='bottom')

    
#...!...!....................
def summary_column(md):
    #pprint(md)
    pmd=md['dataset']
    txt=md['short_name']
    txt+='\nsampFreq %d Hz'%(pmd['sampling_freq'])
    
    return txt
 
#...!...!..................
def plot_2Dspike_session(spike_freq, Twindow,plt,figId,tit0):
    """
    Plots a 2D color map (upper plot) and a mean frequency line plot with ±1 std dev shading (bottom plot).
    
    Parameters:
    - spike_freq: 2D NumPy array of shape (nfeat, ntime)
    - Twindow: Scaling factor for the x-axis in the heatmap
    """
    nfeat, ntime = spike_freq.shape

    # Create 3 subplots: (color bar, heatmap, mean freq line plot)
    fig, axes = plt.subplots(nrows=3, figsize=(10, 8), gridspec_kw={'height_ratios': [0.2, 4, 1]},   num=figId)  # Removed sharex=True

    # Mask values where spike_freq <= 0.5
    min_freq = 0.5
    spike_freq_masked = np.where(spike_freq > min_freq, spike_freq, np.nan)

    # Middle plot: 2D colormap (heatmap) with swapped axes
    vmin, vmax = np.nanmin(spike_freq_masked), np.nanmax(spike_freq_masked)  # Get value range
    cax = axes[1].imshow(spike_freq_masked, aspect='auto', cmap='inferno_r', origin='lower',  
                          extent=[ 0, ntime * Twindow,0, nfeat], vmin=vmin, vmax=vmax)

    axes[1].set_ylabel('Feature Index')  # X-axis is feature index
    axes[1].set_xlabel('Time (sec)')  # Y-axis is time
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
def plot_histogram(ax, data, percentile_low=30, percentile_high=70):
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
    def input_features(self,bigD,md,figId=1,mxFeat=10):
        pprint(md)
        pmd=md['dataset']
        plm=md['plot']
        nfeat=min(mxFeat,pmd['num_feature'])
        ntime=pmd['num_time_bin']
        
        figId=self.smart_append(figId)        
        nrow,ncol=nfeat,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(10,1.2*nrow))

        timeV=bigD['time_ms']
        tit='session '+md['short_name']
        
        for k in range(nrow):
            ax = self.plt.subplot(nrow,ncol,1+k)
            j=k+30
            featId=bigD['exp_feature_id'][j]
            spikeV=bigD['spikes_data'][j].astype(float)
            # Plot Exponential Decay as Filled Area
            #ax.fill_between(timeV, 0,featV , color='red', alpha=0.3, label='Exponential Decay')
            ax.plot(timeV, spikeV , color='red', label='featExponential Decay')
            
            #.... decorations
            ax.grid()
            if 'time_rangeLR' in plm:  ax.set_xlim(tuple(plm['time_rangeLR']))

            if k==nrow-1: ax.set_xlabel('Time (ms)')
            ax.set_ylabel('F=%d'%featId)
            print('draw F=',j)

            
        add_time_scale_marker(ax)

        # .... decorations ....
        # Overlay the text on top of the plots
        txt=summary_column(md)
        ax.text(0.6, 0.95, txt, fontsize=10, color='blue', ha='left', va='top',transform=ax.transAxes)

#...!...!..................
    def global_qa(self,bigD,md,figId=3):
        #pprint(md)
        pmd=md['dataset']
        
        tit='session '+md['short_name']
        freqV=bigD['qa_avr_spike_freq']
        
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

        pmd=md['dataset']
        tit=md['short_name']

        figId=self.smart_append(figId)        
        #fig=self.plt.figure(figId,facecolor='white', figsize=(12,14))
        
        plot_2Dspike_session(bigD['qa_spike_freq'],pmd['qa_twindow_sec'],self.plt,figId,tit)
    


#...!...!..................
    def input_features_dense(self,bigD,md,figId=1,mxFeat=9):
        pprint(md)
        pmd=md['dataset']
        plm=md['plot']
        nfeat=min(mxFeat,pmd['num_feature'])
        ntime=pmd['num_time_bin']
        nrow,ncol=nfeat,1
         
        axes=self.blank_share2D(nrow=nrow,ncol=ncol, figsize=(12,1.5*nrow),figId=figId)
        #axes=self.blank_share2D(nrow=nrow,ncol=ncol, figsize=(20,0.6*nrow),figId=figId)  

        timeV=bigD['time_ms']
        featIdL=bigD['exp_feature_id']
        spikeVV=bigD['spikes_data']  # is bool_ type
        
        if 'time_rangeLR' in plm:
            ta,tb=plm['time_rangeLR']
            timeV=timeV[ta:tb]
            spikeVV=spikeVV[:,ta:tb]
            #print('tt',spikeVV.dtype)
            #print('tt2',timeV[-10:])
            
        for k in range(nrow):
            ax = axes[k]
            j=k+1
            spikeV=spikeVV[j]
            fid=featIdL[j]
            ax.plot(timeV, spikeV , color='red',label='feature=%d'%fid)
            
            
            # .... decorations ....
            ax.legend()
            ax.set_ylim(0,2.1)
            ax.set_ylabel('spike')
            if k==0:ax.set_title('Spikes, data=%s'%md['short_name'] )
            
            
        # common
        #if 'time_rangeLR' in plm:  ax.set_xlim(tuple(plm['time_rangeLR']))
        ax.set_xlabel('Time (ms)')

#...!...!..................
    def cox_autocov_fit(self, expD, expMD, figId=5):
        #pprint(expMD)
        tit = 'Cox Process Analysis, dataset: ' + expMD['short_name']
                
        figId=self.smart_append(figId)        
        nrow,ncol=2,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(6,7))

        # A) Top plot: Fano factor stored in expD['qa_spike_moments']
        ax1 = self.plt.subplot(nrow,ncol,1)
        
        # Extract Fano factor data from spike moments
        spike_moments = expD['qa_spike_moments']
        # spike_moments is array with columns: [window_size, mean, variance, fano_factor]
        window_sizes = spike_moments[:, 0]  # window sizes in time bins
        fano_factors = spike_moments[:, 3]  # fano factors
        
        ax1.semilogx(window_sizes, fano_factors, 'bo-', label='Fano factor')
        ax1.axhline(y=1.0, color='red', linestyle='--', label='Poisson (F=1)')
        ax1.set_xlabel('Window size (time bins)')
        ax1.set_ylabel('Fano factor')
        
        # Calculate average spike rate from last row of qa_spike_moments
        last_row = spike_moments[-1]  # [window_size, mean, variance, fano_factor]
        mean_spike_count = last_row[1]  # mean spikes per window
        window_size_bins = last_row[0]  # window size in time bins
        sampling_freq = expMD['dataset']['sampling_freq']  # Hz
        
        # Convert to spike rate (spikes/sec)
        window_duration_sec = window_size_bins / sampling_freq
        avg_spike_rate = mean_spike_count / window_duration_sec
        
        ax1.set_title(f'Fano Factor (avg spike rate: {avg_spike_rate:.1f} Hz)')
        ax1.legend()
        ax1.grid(True)

        # B) Bottom plot: Cross-covariance plot from expD['cross_cov_data'] and expMD['cross_cov_fit']
        ax2 = self.plt.subplot(nrow,ncol,2)
        
        # Extract cross-covariance data and fit parameters
        # Extract times and C from the 2D array
        cross_cov_data = expD['cross_cov_data']
        times = cross_cov_data[0]  # First row: times
        C = cross_cov_data[1]      # Second row: covariance values
           
        fit_params = expMD['cross_cov_fit']
        
        # Get fit parameters from dictionary
        tau_c = fit_params['tau']
        A = fit_params['A']
        B = fit_params['B']
        fit_start_ms = fit_params['fit_start_ms']
        
        # convert cutoff to index
        dt_s = times[1] - times[0]
        start_idx = int(round(fit_start_ms / (dt_s*1000.0)))
        if start_idx < 1:  start_idx = 1

        # data for fit‐line
        t_fit = times[start_idx:]
        c_fit = A * np.exp(-t_fit / tau_c) +B
        #yA=A * np.exp(-t_fit / tau_c)
        #c_fit = np.sqrt(yA**2+B**2)

        # plot empirical
        ioff=10
        data_shape = fit_params['spikes_data_shape']
        dLab=f'data: {data_shape[0]}f  × {data_shape[1]}t '
        ax2.plot(times[ioff:], C[ioff:], 'k.', label=dLab)  # skip 0-time

        # Calculate B/A ratio as percentage
        B_over_A_ratio = (B/A) * 100 if A != 0 else 0
        
        # plot fit (only in fitted region)
        ax2.plot(t_fit, c_fit, 'r-', label=f'fit τc={tau_c:.3f}s, B/A={B_over_A_ratio:.0f}%')

        # vertical line for cutoff
        ax2.axvline(fit_start_ms/1000.0, color='gray', linestyle=':', 
                   label=f'start at {fit_start_ms}ms')

        
        ax2.set_xlabel('lag(s)')
        ax2.set_ylabel('cross-covariance')
        ax2.set_title(expMD["short_name"]+' - Cross-covariance')
        ax2.legend(loc='best')
        ax2.grid(True)
        
        # Add formula text to the plot
        formula_text = f'C(t) = A·exp(-t/τc) + B'
        ax2.text(0.05, 0.95, formula_text, transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        
