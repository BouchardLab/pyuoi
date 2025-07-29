#!/usr/bin/env python3
"""
Utility functions for Dale Poisson simulation data processing and file I/O.

This module contains functions for:
1. Evaluating spike statistics and printing detailed reports
2. Computing firing rates and coincidence rates with errors
3. Saving simulation data to files in organized formats
4. Loading and processing saved simulation data

Functions:
- eval_spikes_stats(): Evaluates and prints statistics of generated spike data
- estimate_rates_with_errors(): Estimates firing rates and coincidence rates with statistical errors
- save_simulation_data(): Saves spike data and truth data to separate files
- load_simulation_data(): Loads saved simulation data from files
"""

import numpy as np
import os
from pprint import pprint

def eval_spikes_stats(Y, dt, num_excite, mxNn=5):
    """Evaluates and prints statistics of the generated spike data."""
    num_steps_sim, Nn_sim = Y.shape
    num_inhib = Nn_sim - num_excite
    time_evol = num_steps_sim * dt
    print('steps num_steps=%d, time_evol=%.1f sec, Nn=%d (%d Excit, %d Inhib)' % (num_steps_sim, time_evol, Nn_sim, num_excite, num_inhib))

    spike_counts = np.sum(Y, axis=0)
    spike_rates = spike_counts / time_evol
    mean_counts_per_bin = np.mean(Y, axis=0)
    spike_variance = np.var(Y, axis=0)
    # Fano Factor can be undefined if mean is zero
    fano_factor = np.divide(spike_variance, mean_counts_per_bin, out=np.zeros_like(spike_variance), where=mean_counts_per_bin!=0)

    mxE = min(mxNn, num_excite)
    mxI = min(mxNn, num_inhib)

    print('\n--- Stats for first %d Excitatory Neurons ---' % mxE)
    np.set_printoptions(precision=2)
    print('Total Spike Counts:                   %s' % spike_counts[:mxE])
    print('Mean Firing Rate (Hz):                %s' % spike_rates[:mxE])
    print('Mean Spike Count per bin (dt=%.3fs): %s' % (dt, mean_counts_per_bin[:mxE]))
    print('Spike Count Variance per bin:         %s' % spike_variance[:mxE])
    print('Fano Factor (Var/Mean):               %s' % fano_factor[:mxE])

    if num_inhib > 0:
        print('\n--- Stats for first %d Inhibitory Neurons ---' % mxI)
        np.set_printoptions(precision=2)
        inhib_slice = slice(num_excite, num_excite + mxI)
        print('Total Spike Counts:                   %s' % spike_counts[inhib_slice])
        print('Mean Firing Rate (Hz):                %s' % spike_rates[inhib_slice])
        print('Mean Spike Count per bin (dt=%.3fs): %s' % (dt, mean_counts_per_bin[inhib_slice]))
        print('Spike Count Variance per bin:         %s' % spike_variance[inhib_slice])
        print('Fano Factor (Var/Mean):               %s' % fano_factor[inhib_slice])

    # --- Summary Stats ---
    print('\n--- Population Summary Statistics ---')
    # All neurons
    avg_rate_all = np.mean(spike_rates)
    std_rate_all = np.std(spike_rates)
    avg_fano_all = np.mean(fano_factor)
    std_fano_all = np.std(fano_factor)
    print('All    (%d neurons): Avg Rate=%.2f±%.2f Hz, Avg Fano=%.2f±%.2f' % (Nn_sim, avg_rate_all, std_rate_all, avg_fano_all, std_fano_all))

    # Excitatory neurons
    avg_rate_e = np.mean(spike_rates[:num_excite])
    std_rate_e = np.std(spike_rates[:num_excite])
    avg_fano_e = np.mean(fano_factor[:num_excite])
    std_fano_e = np.std(fano_factor[:num_excite])
    print('Excit (%d neurons): Avg Rate=%.2f±%.2f Hz, Avg Fano=%.2f±%.2f' % (num_excite, avg_rate_e, std_rate_e, avg_fano_e, std_fano_e))

    # Inhibitory neurons
    if num_inhib > 0:
        avg_rate_i = np.mean(spike_rates[num_excite:])
        std_rate_i = np.std(spike_rates[num_excite:])
        avg_fano_i = np.mean(fano_factor[num_excite:])
        std_fano_i = np.std(fano_factor[num_excite:])
        print('Inhib (%d neurons): Avg Rate=%.2f±%.2f Hz, Avg Fano=%.2f±%.2f' % (num_inhib, avg_rate_i, std_rate_i, avg_fano_i, std_fano_i))

    print('')

def estimate_rates_with_errors(Y, dt=0.01):
    """
    Estimate single neuron firing rates and pairwise coincidence rates with statistical errors.
    
    Args:
        Y: Spike data array (time_steps x n_neurons)
        dt: Time bin size in seconds
    
    Returns:
        firing_rates: Array of firing rates (Hz) for each neuron
        firing_rate_errors: Standard errors of firing rate estimates
        coincidence_rates: Matrix of coincidence rates (Hz) for each neuron pair
        coincidence_rate_errors: Standard errors of coincidence rate estimates
    """
    n_time_steps, n_neurons = Y.shape
    total_time = n_time_steps * dt
    
    # Estimate single neuron firing rates
    spike_counts = np.sum(Y, axis=0)  # Total spikes per neuron
    firing_rates = spike_counts / total_time  # Hz
    
    # Estimate firing rate standard errors (assuming Poisson process)
    # For Poisson process, variance = mean, so SE = sqrt(mean/N)
    firing_rate_errors = np.sqrt(firing_rates / n_time_steps)
    
    # Estimate pairwise coincidence rates
    coincidence_rates = np.zeros((n_neurons, n_neurons))
    coincidence_rate_errors = np.zeros((n_neurons, n_neurons))
    
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i == j:
                # Self-coincidence is just the firing rate
                coincidence_rates[i, j] = firing_rates[i]
                coincidence_rate_errors[i, j] = firing_rate_errors[i]
            else:
                # Count simultaneous spikes (coincidences)
                coincidences = np.sum(Y[:, i] & Y[:, j])
                coincidence_rates[i, j] = coincidences / total_time
                
                # Estimate standard error for coincidence rate
                # For small coincidence rates, use Poisson approximation
                if coincidences > 0:
                    coincidence_rate_errors[i, j] = np.sqrt(coincidences) / total_time
                else:
                    # For zero coincidences, use upper bound based on firing rates
                    coincidence_rate_errors[i, j] = np.sqrt(firing_rates[i] * firing_rates[j] / n_time_steps)
                
    return firing_rates, firing_rate_errors, coincidence_rates, coincidence_rate_errors

def save_simulation_data(Y, A, B_intercept, dale_conf, evol_conf, firing_rates, firing_rate_errors, 
                        coincidence_rates, coincidence_rate_errors, dataName, outPath):
    """
    Save simulation data to organized files.
    
    Args:
        Y: Spike data array (time_steps x n_neurons)
        A: Connectivity matrix
        B_intercept: Bias vector
        dale_conf: Dale configuration dictionary
        evol_conf: Evolution configuration dictionary
        firing_rates: Array of firing rates
        firing_rate_errors: Array of firing rate errors
        coincidence_rates: Matrix of coincidence rates
        coincidence_rate_errors: Matrix of coincidence rate errors
        dataName: Base name for output files
        outPath: Output directory path
    """
    # Split output into two files
    # 1. Spike trains and rates
    spikes_file = os.path.join(outPath, dataName+'.spikes.npz')
    # Convert to uint8 and clip at max value
    Y_uchar = np.clip(Y, 0, 255).astype(np.uint8)
    np.savez(spikes_file, Y=Y_uchar, firing_rates=firing_rates, firing_rate_errors=firing_rate_errors, 
             coincidence_rates=coincidence_rates, coincidence_rate_errors=coincidence_rate_errors)
    print('Spike data and rates saved to %s (uint8, clipped at 255)' % spikes_file)
    
    # 2. All other truth data
    truth_file = os.path.join(outPath, dataName+'.truth.npz')
    np.savez(truth_file, A=A, B_intercept=B_intercept, conf=dale_conf, evol_conf=evol_conf)
    print('Truth data saved to %s' % truth_file)

def load_simulation_data(dataName, outPath):
    """
    Load simulation data from saved files.
    
    Args:
        dataName: Base name for input files
        outPath: Input directory path
        
    Returns:
        Dictionary containing loaded data:
        - Y: Spike data array
        - A: Connectivity matrix
        - B_intercept: Bias vector
        - dale_conf: Dale configuration
        - evol_conf: Evolution configuration
        - firing_rates: Firing rates array
        - firing_rate_errors: Firing rate errors array
        - coincidence_rates: Coincidence rates matrix
        - coincidence_rate_errors: Coincidence rate errors matrix
    """
    # Load spike data and rates
    spikes_file = os.path.join(outPath, dataName+'.spikes.npz')
    spikes_data = np.load(spikes_file)
    Y = spikes_data['Y'].astype(np.int32)  # Convert back to int32
    firing_rates = spikes_data['firing_rates']
    firing_rate_errors = spikes_data['firing_rate_errors']
    coincidence_rates = spikes_data['coincidence_rates']
    coincidence_rate_errors = spikes_data['coincidence_rate_errors']
    
    # Load truth data
    truth_file = os.path.join(outPath, dataName+'.truth.npz')
    truth_data = np.load(truth_file)
    A = truth_data['A']
    B_intercept = truth_data['B_intercept']
    dale_conf = truth_data['conf'].item() if hasattr(truth_data['conf'], 'item') else truth_data['conf']
    evol_conf = truth_data['evol_conf'].item() if hasattr(truth_data['evol_conf'], 'item') else truth_data['evol_conf']
    
    return {
        'Y': Y,
        'A': A,
        'B_intercept': B_intercept,
        'dale_conf': dale_conf,
        'evol_conf': evol_conf,
        'firing_rates': firing_rates,
        'firing_rate_errors': firing_rate_errors,
        'coincidence_rates': coincidence_rates,
        'coincidence_rate_errors': coincidence_rate_errors
    }

def print_simulation_summary(dataName, outPath):
    """
    Print a summary of saved simulation data.
    
    Args:
        dataName: Base name for input files
        outPath: Input directory path
    """
    try:
        data = load_simulation_data(dataName, outPath)
        
        print(f"\n=== Simulation Data Summary for '{dataName}' ===")
        print(f"Spike data shape: {data['Y'].shape}")
        print(f"Connectivity matrix shape: {data['A'].shape}")
        print(f"Number of excitatory neurons: {data['dale_conf']['num_excite']}")
        print(f"Total number of neurons: {data['dale_conf']['num_neurons']}")
        print(f"Simulation time: {data['evol_conf']['evol_time']:.1f} seconds")
        print(f"Time step: {data['evol_conf']['step_size']:.3f} seconds")
        print(f"Mean firing rate: {np.mean(data['firing_rates']):.2f} ± {np.std(data['firing_rates']):.2f} Hz")
        print(f"Mean coincidence rate: {np.mean(data['coincidence_rates']):.4f} ± {np.std(data['coincidence_rates']):.4f} Hz")
        print("=" * 50)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find simulation data files for '{dataName}' in '{outPath}'")
        print(f"Details: {e}")
    except Exception as e:
        print(f"Error loading simulation data: {e}") 