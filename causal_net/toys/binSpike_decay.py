#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
if 0:
    print('disable Xterm')
    mpl.use('Agg')  # to plot w/o X-server
else:
    mpl.use('TkAgg')
  
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# This function returns binary spike locations as a numpy array.
#...!...!....................

def gen_spike_data(sampling_rate=10_000,tmax_sec=2, spikeFreq=50):
    """
    Generates synthetic binary spike data.
    - sampling_rate: Sampling rate in Hz (e.g., 10000 for 10 kHz)
    - tmax_sec: Total time duration in seconds
    - spikeFreq: Frequency of spikes per second
    """
 
    # Simulated binary spike data (Replace with your actual data)
    np.random.seed(42)
    nsamp=int(tmax_sec*sampling_rate)
    spikeProb=spikeFreq/sampling_rate
    time = np.linspace(0, tmax_sec,  nsamp)  # 1 second at 10 kHz
    binary_data = (np.random.rand(nsamp) < spikeProb).astype(int)  # Simulated spikes
    return time, binary_data

# Function to Add Exponential Decay to Spikes
#...!...!....................

def add_spike_decay(binary_data, sampling_rate=10000, tau_decay=0.01,num_tau=5):
    """
    Adds exponential decay to each binary spike.
    - binary_data: Binary spike data (0s and 1s)
    - sampling_rate: Sampling rate in Hz
    - tau_decay: Decay constant in seconds
    """
    y_pred = np.zeros_like(binary_data,dtype=np.float64)
    decay_samples = num_tau*int(tau_decay * sampling_rate)
    
    # Apply exponential decay to each spike
    for i in range(len(binary_data)):
        if binary_data[i] == 1:
            # Create an exponential decay curve
            decay_curve = np.exp(-np.arange(decay_samples) / (tau_decay * sampling_rate))
            end = min(i + decay_samples, len(binary_data))
            y_pred[i:end] += decay_curve[:end-i]

    return y_pred


# Visualization Function
#...!...!....................
def visualize(time, binary_data, y_pred):
    fig, ax = plt.subplots(figsize=(15, 5))

 
    # Plot Exponential Decay as Filled Area
    ax.fill_between(time, 0, y_pred, color='blue', alpha=0.3, label='Exponential Decay')
    
    # Plot Binary Input Data
    # Ensure at least 4 pixel wide bars  -not working
    width = max(time[1] - time[0], ax.transData.inverted().transform([(10, 0), (0, 0)])[0][0])
    ax.bar(time, binary_data, width=width, color='black', label='Binary Spike Locations')

    # Labels and Title
    ax.set_title('Spike Shape with Exponential Decay')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    outF='out/activation.png'
    plt.savefig(outF)
    print('saved:',outF)
    plt.show()
    

#=================================
#  M A I N 
#=================================

# Run the main function
if __name__ == "__main__":
   # Parameters
    sampling_rate = 10000  # 10 kHz
    tmax_sec = 0.1         # 0.5 seconds duration
    spikeFreq = 200        # 50 spikes per second
    tau_decay = 0.001      # Exponential decay constant (2 ms)
    num_tau=10           # decay cut-off at 20 ms
    
    # Generate Synthetic Spike Data
    time, binary_data = gen_spike_data(sampling_rate, tmax_sec, spikeFreq)

    # Add Exponential Decay to Spikes
    y_pred = add_spike_decay(binary_data, sampling_rate, tau_decay,num_tau)

    # Visualization
    visualize(time, binary_data, y_pred)
    
