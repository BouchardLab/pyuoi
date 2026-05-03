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

def estimate_rates(Y, dt, tau, max_samples, spect_radius, varTwindow=5, mxNn=5, verb=1):
    """
    Evaluates spike statistics and estimates firing rates and Fano factors.

    tau[i] == 0 excitatory, tau[i] == 1 inhibitory; must match Y columns (same order as A_true / B).

    Args:
        Y (np.ndarray): Spike data array (time_steps x n_neurons).
        dt (float): Time bin size in seconds.
        tau (np.ndarray): shape (N,), int; 0 = excitatory, 1 = inhibitory.
        max_samples (int): Max time samples to use for rate computation.
        spect_radius (float): Reported in summary (required).
        varTwindow (float): Window length (seconds) for variance / Fano.
        mxNn (int): Max neurons per type in detailed printout.
        verb (int): Verbose diagnostics.
    """
    if verb > 0:
        print("\n=== Estimating Rates & Stats ===")
    if Y.shape[0] > max_samples:
        if verb > 0:
            print("Using %d samples (out of %d) for rate computation" % (max_samples, Y.shape[0]))
        Y = Y[:max_samples]

    num_steps_sim, Nn_sim = Y.shape
    tau = np.asarray(tau).reshape(-1)
    if tau.shape[0] != Nn_sim:
        raise ValueError("tau length %d != Y.shape[1] %d" % (tau.shape[0], Nn_sim))
    if tau.dtype.kind not in "iu":
        raise ValueError("tau must have integer dtype (0=exc, 1=inh)")
    tau_i = tau.astype(np.int64, copy=False)
    if np.any((tau_i != 0) & (tau_i != 1)):
        raise ValueError("tau entries must be 0 (excitatory) or 1 (inhibitory)")
    exc_mask = tau_i == 0
    inh_mask = tau_i == 1
    if not np.any(exc_mask) or not np.any(inh_mask):
        raise ValueError("tau must contain at least one excitatory and one inhibitory neuron")
    num_excite = int(np.sum(exc_mask))
    num_inhib = int(np.sum(inh_mask))

    time_evol = num_steps_sim * dt
    if verb > 0:
        print('steps num_steps=%d, time_evol=%.1f sec, Nn=%d (%d Excit, %d Inhib)' % (num_steps_sim, time_evol, Nn_sim, num_excite, num_inhib))

    # Compute windowed spike counts and statistics over non-overlapping time windows of length varTwindow (sec), per neuron
    window_size = max(1, int(round(varTwindow / dt)))
    num_windows = num_steps_sim // window_size
    if verb > 0:
        print('variance computation: num_windows=%d, window_size=%d' % (num_windows, window_size))
    if num_windows <= 1:
        raise ValueError("varTwindow=%s is too large for data (num_windows=%d). Need at least 2 windows for variance." % (varTwindow, num_windows))
    Y_trim = Y[:num_windows * window_size]
    Y_win = Y_trim.reshape(num_windows, window_size, Nn_sim)
    
    # Sum spike counts over each window: shape (num_windows, n_neurons)
    window_spike_counts = np.sum(Y_win, axis=1)
    
    # Fano Factor: Var[spike count] / Mean[spike count] over windows
    mean_window_spike_counts = np.mean(window_spike_counts, axis=0)
    var_window_spike_counts = np.var(window_spike_counts, axis=0)
    fano_factor = np.divide(var_window_spike_counts, mean_window_spike_counts, 
                           out=np.zeros_like(var_window_spike_counts), 
                           where=mean_window_spike_counts != 0)
    
    # Convert window spike counts to rates (Hz) for rate variance
    window_rates = window_spike_counts / (window_size * dt)  # shape: (num_windows, n_neurons), Hz
    single_rates_var = np.var(window_rates, axis=0)  # variance of rate across windows, per neuron (Hz^2)

    # Compute raw arrays (full time series)
    spike_counts = np.sum(Y, axis=0)
    spike_rates = spike_counts / time_evol
    mean_counts_per_bin = np.mean(Y, axis=0)
    
    # Compute consecutive coincidence rate
    start_time = time.time()
    conc_rate_per_neuron = compute_consecutive_coincidence_rate(Y,time_evol)
    elapsed_time = time.time() - start_time
    if verb > 0:
        print('Coincidence rate %.2g Hz, Y.shape=%s elaT %.3f sec' % (conc_rate_per_neuron, Y.shape, elapsed_time))
    
    # Compute all population statistics once
    med_rate_all = np.median(spike_rates)
    avg_rate_all = float(np.mean(spike_rates))
    std_rate_all = float(np.std(spike_rates))
    avg_fano_all = float(np.mean(fano_factor))
    std_fano_all = float(np.std(fano_factor))
    avg_rate_excit = float(np.mean(spike_rates[exc_mask]))
    std_rate_excit = float(np.std(spike_rates[exc_mask]))
    avg_fano_excit = float(np.mean(fano_factor[exc_mask]))
    std_fano_excit = float(np.std(fano_factor[exc_mask]))
    avg_rate_inhib = float(np.mean(spike_rates[inh_mask]))
    std_rate_inhib = float(np.std(spike_rates[inh_mask]))
    avg_fano_inhib = float(np.mean(fano_factor[inh_mask]))
    std_fano_inhib = float(np.std(fano_factor[inh_mask]))
    
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

    exc_show = np.where(exc_mask)[0][:mxE]
    inh_show = np.where(inh_mask)[0][:mxI]

    if verb > 0:
        print('\n--- Stats for %d Excitatory Neurons (by index order) ---' % len(exc_show))
        np.set_printoptions(precision=2)
        print('Total Spike Counts:                   %s' % spike_counts[exc_show])
        print('Mean Firing Rate (Hz):                %s' % spike_rates[exc_show])
        print('Mean Spike Count per bin (dt=%.3fs): %s' % (dt, mean_counts_per_bin[exc_show]))
        print('Rate Variance (window=%.1fs):        %s' % (varTwindow, single_rates_var[exc_show]))
        print('Fano Factor (Var/Mean):               %s' % fano_factor[exc_show])

        print('\n--- Stats for %d Inhibitory Neurons (by index order) ---' % len(inh_show))
        np.set_printoptions(precision=2)
        print('Total Spike Counts:                   %s' % spike_counts[inh_show])
        print('Mean Firing Rate (Hz):                %s' % spike_rates[inh_show])
        print('Mean Spike Count per bin (dt=%.3fs): %s' % (dt, mean_counts_per_bin[inh_show]))
        print('Rate Variance (window=%.1fs):        %s' % (varTwindow, single_rates_var[inh_show]))
        print('Fano Factor (Var/Mean):               %s' % fano_factor[inh_show])

    R_tag = ', R=%.3f' % float(spect_radius)
    print('\n--- Population Summary Statistics%s ---' % R_tag)
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

    # Part 2: Compute frequency sorting
    neur_freq_index = np.argsort(spike_rates)

    rates_dict = {
        'single_rates': spike_rates,
        'neur_freqIdx':neur_freq_index,
        'sigle_rates_var': single_rates_var,
        'single_fano_fact': fano_factor
    }

    return stats_dict, rates_dict, neur_freq_index



def get_offdiag_triplets(A, isPos=True):
    """
    Returns positive/negative, off-diagonal elements of 2D array A as array of [i, j, value] triplets.

    Parameters:
    A : 2D numpy array
    isPos : bool, if True only return positive values (>0), if False return all negative values

    Returns:
    Array of shape (n, 3) where n is the number of valid off-diagonal elements
    Each row is [row_index, col_index, value]
    """
    # Create mask for off-diagonal elements
    offdiag_mask = ~np.eye(A.shape[0], A.shape[1], dtype=bool)

    # Add value condition based on isPos
    if isPos:
        value_mask = A > 0
    else:
        value_mask = A < 0

    # Combine masks
    mask = value_mask & offdiag_mask

    i_indices, j_indices = np.where(mask)
    values = A[i_indices, j_indices]

    return np.column_stack([i_indices, j_indices, values])
