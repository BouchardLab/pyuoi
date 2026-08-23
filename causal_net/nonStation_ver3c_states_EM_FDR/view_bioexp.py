#!/usr/bin/env python3
"""
Visualization tool for biological experiment input features and data quality.

This script provides comprehensive visualization and analysis of experimental
neural data, focusing on data quality assessment and feature exploration.
Main functionality includes:
- Time series visualization of neural recordings with customizable time ranges
- Data quality metrics and statistical summaries
- Cluster detection and activity pattern analysis
- Interactive plotting with configurable display options

Used primarily for exploratory data analysis of biological neural recordings
before further processing and connectivity analysis.
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os

from pprint import pprint
import numpy as np
from PlotterBioExp import Plotter
from toolbox.Util_NumpyIOv2 import read_data_npz
from UtilBioExp import detect_spike_bursts
import argparse


def _require_bioexp(bioD, bioMD):
    assert isinstance(bioMD, dict), "bioExp.npz is missing schema-v2 metadata"
    for key in ("metrics_curated", "MEA_idx", "single_rates"):
        assert key in bioD, f"bioExp.npz missing {key!r}; rerun prep_bioexp3c.py"
    for key in ("metrics_curated_columns", "short_name", "data_selector", "rate_summary"):
        assert key in bioMD, f"bioExp metadata missing {key!r}"
    assert "freq_range" in bioMD["data_selector"]

    assert int(bioMD.get("bioexp_schema_version", -1)) == 3, (
        "bioExp data require bioexp_schema_version=3; rerun prep_bioexp3c.py"
    )
    assert bioMD.get("waveforms_available") is True, (
        "bioExp metadata must declare waveforms_available=true"
    )
    waveform_keys = (
        "raw_mean_templates",
        "waveform_num_samples",
        "waveform_unit_ids",
        "waveform_channel_ids",
        "waveform_ms_before",
        "waveform_ms_after",
        "waveform_n_spikes_used",
        "waveform_grid_distance",
        "waveform_is_multichannel",
    )
    for key in waveform_keys:
        assert key in bioD, f"bioExp.npz is missing required record {key!r}"
    templates = np.asarray(bioD["raw_mean_templates"])
    num_neurons = np.asarray(bioD["MEA_idx"]).size
    assert templates.ndim == 2 and templates.shape[0] == num_neurons, (
        "raw_mean_templates must have shape (num_neurons, num_samples)"
    )
    for key in waveform_keys[1:]:
        assert np.asarray(bioD[key]).size == num_neurons, (
            f"{key} must contain one value per neuron"
        )
    assert np.asarray(bioD["waveform_is_multichannel"]).dtype == np.bool_, (
        "waveform_is_multichannel must have Boolean dtype"
    )
    return True


#...!...!....................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int,
                        help="increase output verbosity", default=1, dest="verb")
    parser.add_argument("-p", "--showPlots", default="a", nargs="+",
                        help="plots: a=freq histo, b=freq vs time, c=MEA layout, "
                             "d=metrics histograms, e=metrics correlations")

    parser.add_argument("-X", "--noXterm", action="store_true",
                        help="Disable X terminal for plotting")

    parser.add_argument("--dataPath",
                        default="/pscratch/sd/b/balewski/2025_causalNet_tmp/",
                        help="head dir for any further data processing")

    parser.add_argument("-T", "--time_range", default=[0., 60], nargs=2, type=float,
                        help="display data time range in seconds")
    parser.add_argument("--burst_freq_thres", default=5., type=float,
                        help="tags high freq channels for burst detection")
    parser.add_argument("--dataName", default="HET_80k_1-fc62ef",
                        help="preprocessed session name")

    parser.add_argument("-R", "--time_rebin2", default=10, type=int,
                        help="rebin current time axis for burst panels")

    args = parser.parse_args()
    args.outPath = "out/"
    args.showPlots = "".join(args.showPlots)

    print("myArg-program:", parser.prog)
    for arg in vars(args):
        print("myArg:", arg, getattr(args, arg))

    assert args.time_range[0] < args.time_range[1]
    assert os.path.exists(args.dataPath)
    assert os.path.exists(args.outPath)
    return args


#=================================
#  M A I N
#=================================
if __name__ == "__main__":
    args = get_parser()
    np.set_printoptions(precision=3)

    bioFF = os.path.join(args.dataPath, f"{args.dataName}.bioExp.npz")
    print("bioExp:", bioFF)
    bioD, bioMD = read_data_npz(bioFF)
    if args.verb > 1: pprint(bioMD)
    waveforms_available = _require_bioexp(bioD, bioMD)
    print("raw mean waveforms:", "available" if waveforms_available else "not available")

    plotMD = dict(bioMD)
    args.prjName = bioMD["short_name"]

    spikeD = None
    rebD = None
    if "a" in args.showPlots or "b" in args.showPlots:
        spikesFF = os.path.join(args.dataPath, f"{args.dataName}.spikes.npz")
        spikeD, spikeMD = read_data_npz(spikesFF)
        '''
        for key in ("spikes", "single_rates", "provenance", "data_type",
                    "time_step_sec", "num_neurons"):
            assert key in spikeMD, f"spikes.npz metadata missing {key!r}"
        '''
        assert "experiment_name" in spikeMD["provenance"]
        plotMD = {**bioMD, **spikeMD}
        args.prjName = spikeMD["provenance"]["experiment_name"]
        if args.verb > 1:
            pprint(spikeMD)

    if "b" in args.showPlots:
        plotMD["plot"] = {"time_rangeLR": np.array(args.time_range)}
        rebD = detect_spike_bursts(
            spikeD, spikeMD, args.time_rebin2,
            args.burst_freq_thres,
        )

    plot = Plotter(args)

    if "a" in args.showPlots:
        plot.freq_histo(spikeD, plotMD, figId=1)
    if "b" in args.showPlots:
        plot.freq_vs_time(rebD, plotMD, figId=2)
    if "c" in args.showPlots:
        plot.neuron_spatial(bioD, plotMD, figId=3)
    if "d" in args.showPlots:
        plot.metrics_histograms(bioD, plotMD, figId=4)
    if "e" in args.showPlots:
        plot.metrics_correlations(bioD, plotMD, figId=5)

    plot.display_all()
    print("M:done")

  
