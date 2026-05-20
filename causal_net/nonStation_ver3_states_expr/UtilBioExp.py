#!/usr/bin/env python3
"""
Utility functions for biological experiment data processing.

This module provides specialized functions for analyzing biological neural data,
particularly focused on identifying and processing clusters of neural activity.
Main functionality includes:
- Cluster detection in time series data using connectivity analysis
- Threshold-based cluster validation and masking
- Signal processing utilities for experimental neural recordings

Used primarily in conjunction with experimental data preprocessing and
analysis pipelines for biological neural network studies.
"""

import numpy as np
from scipy.ndimage import label

def create_clusters_mask(X, th):
    """
    Creates a boolean mask for a 1D NumPy array.

    A contiguous cluster of non-zero values is marked as True if any value
    within that cluster is greater than or equal to the threshold 'th'.
    Clusters are assumed to be separated by zero values.

    Parameters:
    X (np.ndarray): Input 1D NumPy array.
    th (float): The threshold value.

    Returns:
    np.ndarray: A boolean mask with the same shape as X.
    """
    # Step 1: Find all contiguous clusters of non-zero values.
    # The `label` function assigns a unique integer to each cluster.
    # e.g., [1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3]
    labeled_array, num_clusters = label(X > 0)
    
    # Step 2: Initialize the final mask to all False.
    final_mask = np.zeros_like(X, dtype=bool)
    
    # Step 3: Iterate through each cluster found by the label function.
    # We start from 1 because 0 is the background (the zeros in X).
    for i in range(1, num_clusters + 1):
        # Create a boolean mask for the current cluster only.
        current_cluster_mask = (labeled_array == i)
        
        # Step 4: Check if ANY value within this specific cluster meets the threshold.
        if np.any(X[current_cluster_mask] >= th):
            # Step 5: If the condition is met, mark this entire cluster as True in our final mask.
            final_mask[current_cluster_mask] = True
            
    return final_mask


def detect_spike_bursts(spikeD, spikeMD, time_rebin2, burst_freq_thres, burst_chan_thres):
    """Rebin spikes in time; return rate panels for bioExp freq-vs-time plots."""
    rateThr2 = float(burst_freq_thres)
    spikeYield = spikeD["spikes"]
    time_step = float(spikeMD["time_step_sec"])
    tReb2 = int(time_rebin2)
    assert tReb2 < 101

    ntime, nchan = spikeYield.shape
    if ntime % tReb2 != 0:
        ntime_c = ntime - (ntime % tReb2)
        spikeYield = spikeYield[:ntime_c]

    spikeYieldR = np.sum(spikeYield.reshape(-1, tReb2, nchan), axis=1)
    time_step2 = time_step * tReb2

    rate2D = spikeYieldR / time_step2
    mask2D = rate2D > rateThr2
    highChan = np.sum(mask2D, axis=1)

    mCnt = int(burst_chan_thres)
    XM = create_clusters_mask(highChan, th=mCnt)
    usableFrac = 1 - np.sum(XM) / XM.shape[0]

    rebD = {
        "time_step2": time_step2,
        "rate_thres2": rateThr2,
        "high_cnt_thres": mCnt,
        "usable_time_fract": usableFrac,
        "rate2D": rate2D,
        "mask2D": mask2D,
        "highChanCnt": highChan,
        "highChanMask": XM,
        "rate1D": np.sum(rate2D, axis=1),
    }
    ntime = spikeYieldR.shape[0]
    rebD["timeV"] = np.linspace(0, (ntime - 1) * time_step2, ntime)

    print("usable time frac:%.3f  nchan=%d  thr=%.1f Hz"
          % (rebD["usable_time_fract"], nchan, rateThr2))
    return rebD


def clip_rebD_time(rebD, time_range_lr):
    """Clip rebD arrays to a time window in seconds."""
    time_step = rebD["time_step2"]
    tL, tR = [float(x) for x in time_range_lr]
    itL, itR = (np.asarray(time_range_lr, dtype=np.float64) / time_step).astype(int)
    itR = min(itR, rebD["rate2D"].shape[0])
    if itR <= itL:
        raise ValueError("time_range_lr leaves no rebinned bins")

    rate2D = rebD["rate2D"][itL:itR]
    timeV = rebD["timeV"][itL:itR]
    rate1D = rebD["rate1D"][itL:itR]
    highChanMask = rebD["highChanMask"][itL:itR]
    _, nchan = rate2D.shape
    Tbin = float(timeV[1] - timeV[0])
    return {
        "tL": tL, "tR": tR, "itL": itL, "itR": itR,
        "time_step": time_step, "Tbin": Tbin, "nchan": nchan,
        "rate2D": rate2D, "timeV": timeV, "rate1D": rate1D,
        "highChanMask": highChanMask,
    }


if __name__=="__main__":

    # Example usage
    X = np.array([1, 2, 3, 0, 2, 6, 2, 0, 1, 2, 8, 0, 0])
    threshold = 5
    mask = create_clusters_mask(X, threshold)
    
    # Print the output in two columns
    print(f"Input  | Mask , thr:{threshold}")
    print("-" * 25)
    for x_val, mask_val in zip(X, mask):
        print(f"{x_val:<4} | {mask_val}")
