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


def detect_spike_bursts(spikeD, spikeMD, time_rebin2, burst_freq_thres):
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

    rebD = {
        "time_step2": time_step2,
        "rate_thres2": rateThr2,
        "rate2D": rate2D,
        "mask2D": mask2D,
        "highChanCnt": highChan,
        "rate1D": np.sum(rate2D, axis=1),
    }
    ntime = spikeYieldR.shape[0]
    rebD["timeV"] = np.linspace(0, (ntime - 1) * time_step2, ntime)

    print("nchan=%d  thr=%.1f Hz" % (nchan, rateThr2))
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
    _, nchan = rate2D.shape
    Tbin = float(timeV[1] - timeV[0])
    return {
        "tL": tL, "tR": tR, "itL": itL, "itR": itR,
        "time_step": time_step, "Tbin": Tbin, "nchan": nchan,
        "rate2D": rate2D, "timeV": timeV, "rate1D": rate1D,
    }


if __name__=="__main__":
    print("Utility module for biological experiment spike-rate rebinning.")
