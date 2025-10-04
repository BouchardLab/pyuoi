"""
Utility functions for Dale Poisson simulation data processing 

"""

import numpy as np
import os
import time
from pprint import pprint

def compute_consecutive_coincidence_rate(Y,time_evol):
    """
    Compute the frequency of coincidences for 2 consecutive time bins for 2 different channels.
    
    Args:
        Y (np.ndarray): Spike data array (time_steps x n_neurons).
    
    Returns:
        float: conc_rate_all - the overall coincidence rate
    """
    num_steps, num_neurons = Y.shape
    
    # We need at least 2 time steps for consecutive bins
    if num_steps < 2:
        return 0.0
    
    # Get consecutive time slices using vectorized operations
    Y_t = Y[:-1, :]  # Y[0:T-1, :] - current time bins
    Y_t_plus_1 = Y[1:, :]  # Y[1:T, :] - next time bins
    
    # Create boolean masks for non-zero values
    mask_t = (Y_t != 0)  # Shape: (T-1, N)
    mask_t_plus_1 = (Y_t_plus_1 != 0)  # Shape: (T-1, N)
    
    # Use broadcasting to compute all channel pairs at once
    # mask_t[:, :, None] has shape (T-1, N, 1)
    # mask_t_plus_1[:, None, :] has shape (T-1, 1, N)
    # Broadcasting gives shape (T-1, N, N) for all pairs
    coincidences = mask_t[:, :, None] & mask_t_plus_1[:, None, :]
    
    # Remove diagonal (same channel pairs) using boolean indexing
    diagonal_mask = np.eye(num_neurons, dtype=bool)
    coincidences[:, diagonal_mask] = False
    
    # Count total coincidences across all time steps
    coincidence_count = np.sum(coincidences)
    
    conc_rate = coincidence_count / time_evol/num_neurons
    return conc_rate  # Hz, per neuron

def estimate_rates(Y, dt, num_excite, max_samples, mxNn=5):
    """
    Evaluates spike statistics and estimates firing/coincidence rates.

    Args:
        Y (np.ndarray): Spike data array (time_steps x n_neurons).
        dt (float): Time bin size in seconds.
        num_excite (int): Number of excitatory neurons.
        max_samples_for_rates (int): The maximum number of time samples to use for calculation.
        mxNn (int, optional): Max number of neurons to show in detailed stats. Defaults to 5.

    Returns:
        tuple: A tuple containing two dictionaries:
            - stats_dict (dict): Contains detailed spike statistics.
            - rates_dict (dict): Contains firing rates and coincidence rates.
    """
    # 1. Clip data to max_samples_for_rates
    print("\n=== Estimating Rates & Stats ===")
    if Y.shape[0] > max_samples:
        print("Using %d samples (out of %d) for rate computation" % (max_samples, Y.shape[0]))
        Y = Y[:max_samples]

    # Part 1: Compute all basic statistics once
    num_steps_sim, Nn_sim = Y.shape
    num_inhib = Nn_sim - num_excite
    time_evol = num_steps_sim * dt
    print('steps num_steps=%d, time_evol=%.1f sec, Nn=%d (%d Excit, %d Inhib)' % (num_steps_sim, time_evol, Nn_sim, num_excite, num_inhib))

    # Compute raw arrays
    spike_counts = np.sum(Y, axis=0)
    spike_rates = spike_counts / time_evol
    mean_counts_per_bin = np.mean(Y, axis=0)
    spike_variance = np.var(Y, axis=0)
    fano_factor = np.divide(spike_variance, mean_counts_per_bin, out=np.zeros_like(spike_variance), where=mean_counts_per_bin != 0)
    
    # Compute consecutive coincidence rate
    start_time = time.time()
    conc_rate_per_neuron = compute_consecutive_coincidence_rate(Y,time_evol)
    elapsed_time = time.time() - start_time
    print('Coincidence rate %.2g Hz, Y.shape=%s elaT %.3f sec' % (conc_rate_per_neuron, Y.shape, elapsed_time))
    
    # Compute all population statistics once
    med_rate_all = np.median(spike_rates)
    avg_rate_all = float(np.mean(spike_rates))
    std_rate_all = float(np.std(spike_rates))
    avg_fano_all = float(np.mean(fano_factor))
    std_fano_all = float(np.std(fano_factor))
    avg_rate_excit = float(np.mean(spike_rates[:num_excite]))
    std_rate_excit = float(np.std(spike_rates[:num_excite]))
    avg_fano_excit = float(np.mean(fano_factor[:num_excite]))
    std_fano_excit = float(np.std(fano_factor[:num_excite]))
    avg_rate_inhib = float(np.mean(spike_rates[num_excite:]))
    std_rate_inhib = float(np.std(spike_rates[num_excite:]))
    avg_fano_inhib = float(np.mean(fano_factor[num_excite:]))
    std_fano_inhib = float(np.std(fano_factor[num_excite:]))
    
    # Build stats dictionary with computed values
    stats_dict = {
        'num_steps': num_steps_sim,
        'time_evol_sec': time_evol,
        'time_step_sec': dt,
        'num_neurons': Nn_sim,
        'num_excitatory': num_excite,
        'num_inhibitory': num_inhib,
        'avg_spike_rate_all': avg_rate_all,
        'std_spike_rate_all': std_rate_all,
        'avg_fano_factor_all': avg_fano_all,
        'std_fano_factor_all': std_fano_all,
        'avg_spike_rate_excit': avg_rate_excit,
        'std_spike_rate_excit': std_rate_excit,
        'avg_fano_factor_excit': avg_fano_excit,
        'std_fano_factor_excit': std_fano_excit,
        'avg_spike_rate_inhib': avg_rate_inhib,
        'std_spike_rate_inhib': std_rate_inhib,
        'avg_fano_factor_inhib': avg_fano_inhib,
        'std_fano_factor_inhib': std_fano_inhib,
        'conc_rate_per_neuron': float(conc_rate_per_neuron),
        'median_spike_rate_all': med_rate_all
    }

    # Printing detailed stats for individual neurons
    mxE = min(mxNn, num_excite)
    mxI = min(mxNn, num_inhib)

    print('\n--- Stats for first %d Excitatory Neurons ---' % mxE)
    np.set_printoptions(precision=2)
    print('Total Spike Counts:                   %s' % spike_counts[:mxE])
    print('Mean Firing Rate (Hz):                %s' % spike_rates[:mxE])
    print('Mean Spike Count per bin (dt=%.3fs): %s' % (dt, mean_counts_per_bin[:mxE]))
    print('Spike Count Variance per bin:         %s' % spike_variance[:mxE])
    print('Fano Factor (Var/Mean):               %s' % fano_factor[:mxE])

    print('\n--- Stats for first %d Inhibitory Neurons ---' % mxI)
    np.set_printoptions(precision=2)
    inhib_slice = slice(num_excite, num_excite + mxI)
    print('Total Spike Counts:                   %s' % spike_counts[inhib_slice])
    print('Mean Firing Rate (Hz):                %s' % spike_rates[inhib_slice])
    print('Mean Spike Count per bin (dt=%.3fs): %s' % (dt, mean_counts_per_bin[inhib_slice]))
    print('Spike Count Variance per bin:         %s' % spike_variance[inhib_slice])
    print('Fano Factor (Var/Mean):               %s' % fano_factor[inhib_slice])

    # Print population summary using dictionary values
    print('\n--- Population Summary Statistics ---')
    print('All   (%d neurons): Avg Rate=%.2f±%.2f Hz, Avg Fano=%.2f±%.2f' % (Nn_sim, stats_dict['avg_spike_rate_all'], stats_dict['std_spike_rate_all'], stats_dict['avg_fano_factor_all'], stats_dict['std_fano_factor_all']))
    print('Excit (%d neurons): Avg Rate=%.2f±%.2f Hz, Avg Fano=%.2f±%.2f' % (num_excite, stats_dict['avg_spike_rate_excit'], stats_dict['std_spike_rate_excit'], stats_dict['avg_fano_factor_excit'], stats_dict['std_fano_factor_excit']))
    if num_inhib > 0:
        print('Inhib (%d neurons): Avg Rate=%.2f±%.2f Hz, Avg Fano=%.2f±%.2f' % (num_inhib, stats_dict['avg_spike_rate_inhib'], stats_dict['std_spike_rate_inhib'], stats_dict['avg_fano_factor_inhib'], stats_dict['std_fano_factor_inhib']))
    
    # Print key summary values using dictionary
    summary_keys = ['conc_rate_per_neuron', 'median_spike_rate_all']
    for key in summary_keys:
        if key == 'conc_rate_per_neuron':
            print('Coincidence rate per neuron %.2g Hz' % stats_dict[key])
        elif key == 'median_spike_rate_all':
            print('Median rate  %.2f Hz\n' % stats_dict[key])

    # Part 2: from estimate_rates_with_errors (computes rates, no errors)
    n_time_steps, n_neurons = Y.shape
    total_time = n_time_steps * dt

    # Compute index of neurons sorted by frequency (from lowest to highest )
    neur_freq_index = np.argsort(spike_rates)

    rates_dict = {
        'single_rates': spike_rates,
        'neur_freqIdx':neur_freq_index
    }

    return stats_dict, rates_dict, neur_freq_index



def  do_neuron_classifier(A):  # ???
    #  
    thrMaj=0.9  # edge count for:  pure | majority
    thrMix=0.5  # edge count for:   majority  | mix

    # Create mask for off-diagonal elements
    offdiag_mask = ~np.eye(A.shape[0], A.shape[1], dtype=bool)
    posEdge_mask= A>0
    negEdge_mask= A<0

    # combine masks

    mask = posEdge_mask & offdiag_mask
    i_indices, j_indices = np.where(mask)
    values = A[i_indices, j_indices]
    not_used_yet
    return np.column_stack([i_indices, j_indices, values])
