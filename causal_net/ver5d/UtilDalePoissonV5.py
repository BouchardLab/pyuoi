#!/usr/bin/env python3
"""
Utility functions for Dale Poisson simulation data processing 

"""

import numpy as np
import os
from pprint import pprint

def estimate_rates(Y, dt, num_excite, max_samples_for_rates, mxNn=5):
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
    if Y.shape[0] > max_samples_for_rates:
        print("Using %d samples (out of %d) for rate computation" % (max_samples_for_rates, Y.shape[0]))
        Y_for_rates = Y[:max_samples_for_rates]
    else:
        Y_for_rates = Y
        print("Using all %d samples for rate computation" % (Y.shape[0],))

    # Part 1: from eval_spikes_stats
    num_steps_sim, Nn_sim = Y_for_rates.shape
    num_inhib = Nn_sim - num_excite
    time_evol = num_steps_sim * dt
    print('steps num_steps=%d, time_evol=%.1f sec, Nn=%d (%d Excit, %d Inhib)' % (num_steps_sim, time_evol, Nn_sim, num_excite, num_inhib))

    spike_counts = np.sum(Y_for_rates, axis=0)
    spike_rates = spike_counts / time_evol
    mean_counts_per_bin = np.mean(Y_for_rates, axis=0)
    spike_variance = np.var(Y_for_rates, axis=0)
    fano_factor = np.divide(spike_variance, mean_counts_per_bin, out=np.zeros_like(spike_variance), where=mean_counts_per_bin != 0)

    stats_dict = {
        'num_steps': num_steps_sim,
        'time_evol_sec': time_evol,
        'time_step_sec': dt,
        'num_neurons': Nn_sim,
        'num_excitatory': num_excite,
        'num_inhibitory': num_inhib,
        'avg_spike_rate_all': np.mean(spike_rates),
        'std_spike_rate_all': np.std(spike_rates),
        'avg_fano_factor_all': np.mean(fano_factor),
        'std_fano_factor_all': np.std(fano_factor),
        'avg_spike_rate_excit': np.mean(spike_rates[:num_excite]),
        'std_spike_rate_excit': np.std(spike_rates[:num_excite]),
        'avg_fano_factor_excit': np.mean(fano_factor[:num_excite]),
        'std_fano_factor_excit': np.std(fano_factor[:num_excite])
    }
    
    
    stats_dict.update({
            'avg_spike_rate_inhib': np.mean(spike_rates[num_excite:]),
            'std_spike_rate_inhib': np.std(spike_rates[num_excite:]),
            'avg_fano_factor_inhib': np.mean(fano_factor[num_excite:]),
            'std_fano_factor_inhib': np.std(fano_factor[num_excite:])
        })

    # Printing part from eval_spikes_stats
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

    print('\n--- Population Summary Statistics ---')
    avg_rate_all = np.mean(spike_rates)
    std_rate_all = np.std(spike_rates)
    avg_fano_all = np.mean(fano_factor)
    std_fano_all = np.std(fano_factor)
    print('All    (%d neurons): Avg Rate=%.2f±%.2f Hz, Avg Fano=%.2f±%.2f' % (Nn_sim, avg_rate_all, std_rate_all, avg_fano_all, std_fano_all))

    avg_rate_e = np.mean(spike_rates[:num_excite])
    std_rate_e = np.std(spike_rates[:num_excite])
    avg_fano_e = np.mean(fano_factor[:num_excite])
    std_fano_e = np.std(fano_factor[:num_excite])
    print('Excit (%d neurons): Avg Rate=%.2f±%.2f Hz, Avg Fano=%.2f±%.2f' % (num_excite, avg_rate_e, std_rate_e, avg_fano_e, std_fano_e))

    if num_inhib > 0:
        avg_rate_i = np.mean(spike_rates[num_excite:])
        std_rate_i = np.std(spike_rates[num_excite:])
        avg_fano_i = np.mean(fano_factor[num_excite:])
        std_fano_i = np.std(fano_factor[num_excite:])
        print('Inhib (%d neurons): Avg Rate=%.2f±%.2f Hz, Avg Fano=%.2f±%.2f' % (num_inhib, avg_rate_i, std_rate_i, avg_fano_i, std_fano_i))
    print('')

    # Part 2: from estimate_rates_with_errors (computes rates, no errors)
    n_time_steps, n_neurons = Y_for_rates.shape
    total_time = n_time_steps * dt

    # Firing rates are already computed as `spike_rates`
    firing_rates = spike_rates

    # Estimate pairwise coincidence rates
    coincidence_rates = np.zeros((n_neurons, n_neurons))
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i == j:
                coincidence_rates[i, j] = firing_rates[i]
            else:
                coincidences = np.sum(Y_for_rates[:, i] & Y_for_rates[:, j])
                coincidence_rates[i, j] = coincidences / total_time
    
    rates_dict = {
        'single_rates': firing_rates,
        'coincidence_rates': coincidence_rates
    }
    
    print("Coincidence rates: mean=%.4f ± %.4f Hz" % (np.mean(rates_dict['coincidence_rates']), np.std(rates_dict['coincidence_rates'])))

    return stats_dict, rates_dict


def geom_edges_mask(md):
    dale_conf=md['dale_conf']
    Nn=dale_conf['num_neurons']
    Ne=dale_conf['num_excite']
    Ni=Nn-Ne

    # Create diagonal mask
    diag_mask = np.eye(Nn, dtype=bool)
    # exc_mask: first Ne rows, all columns, except diagonal
    exc_mask = np.zeros((Nn, Nn), dtype=bool)
    exc_mask[:Ne, :] = True
    exc_mask = exc_mask & (~diag_mask)  # remove diagonal

    # inh_mask: next Ni rows, all columns, except diagonal
    inh_mask = np.zeros((Nn, Nn), dtype=bool)
    inh_mask[Ne:, :] = True
    inh_mask = inh_mask & (~diag_mask)  # remove diagonal

    # create 1d masks for exc & inh
    exc_1d= np.zeros((Nn), dtype=bool)
    exc_1d[:Ne] = True
    inh_1d= np.zeros((Nn), dtype=bool)
    inh_1d[Ne:] = True

    maskG={'diag':diag_mask, 'exc':exc_mask,'inh':inh_mask,'exc_idx':exc_1d,'inh_idx':inh_1d}
    maskD={'geom':maskG}
    return maskD

def true_edges_mask(maskD, A_true):
    #print('\ntrue_edge_mask')
    maskD['true']=maskT={}
    maskG=maskD['geom']
    A_abs = np.abs(A_true)
    for ntype in ['exc','inh']:
        gmask=maskG[ntype]
        tmask = gmask & (A_abs>1e-8)
        nGeom=np.sum(gmask)
        nTrue=np.sum(tmask)
        maskT[ntype]=tmask


def select_eges_from_fitLasso( bigD, amplThres=0.2):
    print('\nselect_eges_from_fitL1 amplThres=%.2f' % amplThres)
    maskF = {}
    #maskG = maskD['geom']
    A_fit = bigD['A_lasso']
    A_abs = np.abs(A_fit)
    n = A_fit.shape[0]
    offdiag = ~np.eye(n, dtype=bool)  # mask for off-diagonal elements

    # Select elements where abs(A_fit) > amplThres, only off-diagonal
    fmask = (A_abs > amplThres) & offdiag

    # Initialize 1D masks classyfuing neurons
    exc_1d = np.zeros(n, dtype=bool)
    inh_1d = np.zeros(n, dtype=bool)
    iso_1d = np.zeros(n, dtype=bool)

    for i in range(n):
        sel = fmask[i, :]
        if np.any(sel):
            avg = np.mean(A_fit[i, sel])
            if avg > 0:
                exc_1d[i] = True
            else:
                inh_1d[i] = True
        else:
            iso_1d[i] = True

    # Convert 1D masks to 2D masks by broadcasting over columns
    exc_mask = (exc_1d[:, None]) & (A_fit > amplThres) & offdiag
    inh_mask = (inh_1d[:, None]) & (A_fit < -amplThres) & offdiag

    #... collect masks ....
    maskF['above'] = fmask
    maskF['exc'] = exc_mask
    maskF['inh'] = inh_mask

    maskF['exc_idx'] = exc_1d
    maskF['inh_idx'] = inh_1d
    maskF['iso_idx'] = iso_1d

    # Optionally, you can combine all for a 'pass' mask:
    maskF['pass'] = exc_mask | inh_mask | ~offdiag
    bigD['A_pass'] = np.where(maskF['pass'], A_fit, 0)

    nGeom = np.sum(offdiag)
    nFit = np.sum( exc_mask | inh_mask  )
    print('fit mask', nGeom, nFit, 'amplThres=%.3f' % amplThres)
    return maskF

