#!/usr/bin/env python3



import pickle
import numpy as np
np.set_printoptions(precision=3)

filepath = "/dataVault2025/causalNet_tmp/HET_80k_1_spike_dict.pkl"

# Load the dictionary from the .pkl file
with open(filepath, "rb") as f:
    spike_dict = pickle.load(f)



max_value = max([max(v) for v in spike_dict.values()])

num_neurons = len(spike_dict)

print('M: max_value=',max_value)
print('M: num_neurons',num_neurons)

sampling_freq = 10000  # Hz
target_fs = 1000
duration = max_value/sampling_freq
# Calculate the number of neurons and bins
num_neurons = len(spike_dict)
num_bins = int(duration * target_fs)

# Create an empty binary matrix for the downsampled data
spike_matrix = np.zeros((num_neurons, num_bins), dtype=int)

print('M: spike_matrix:',spike_matrix.shape)

# Define the bin edges based on the target sampling frequency
bin_edges = np.linspace(0, duration, num_bins + 1)  # num_bins + 1 to include the last edge

print('M: bin_edges',bin_edges.shape) 

print('M: spike agregation')
print(' neuron_idx,  id, num_samp  spikes_in_seconds[:10]')
# Iterate over each neuron and populate the matrix
for neuron_idx, (neuron_id, spikes) in enumerate(spike_dict.items()):
    # Convert spike times to seconds
    spikes_in_seconds = np.array(spikes )/ sampling_freq

    if  neuron_idx%100==0 or neuron_idx==num_neurons-1:
        rec=spikes_in_seconds
        print(' idx:',neuron_idx, ' id:',neuron_id, rec.shape, rec[:10])
        
    
    # Digitize the spike times into bins
    bin_indices = np.digitize(spikes_in_seconds, bin_edges) - 1  # Convert to 0-based index
    bin_indices = bin_indices[(bin_indices >= 0) & (bin_indices < num_bins)]  # Remove out-of-range indices
    
    # Mark the bins where spikes occurred
    spike_matrix[neuron_idx, np.unique(bin_indices)] = 1  # Use np.unique to avoid duplicate marking

    
    
print("Binary Spike Matrix:")
print(spike_matrix)

file_path = ""  #add filepath

# Save the matrix
np.save(file_path, spike_matrix)

print(f"Spike matrix saved to {file_path}")
