#!/usr/bin/env python3
"""
Visualization tool for in-silico (NetPyNE synthetic network) input features
and data quality.

Reads the two outputs of prep_inSilico3c.py:
  <dataName>.spikes.npz      -- binned spike counts + single_rates
  <dataName>.prismTruth.npz  -- node geometry/type ground truth

Main functionality includes:
- Per-neuron firing-rate histograms
- Rebinned freq-vs-time raster/heatmap for burst-activity inspection
- Spatial layout of neuron positions colored by firing rate
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os

from pprint import pprint
import numpy as np
from PlotterInSilico import Plotter
from toolbox.Util_NumpyIOv2 import read_data_npz
from UtilBioExp import detect_spike_bursts
import argparse


def _require_inSilico(spikeD, spikeMD, truthD, truthMD):
    for key in ("spikes", "single_rates"):
        assert key in spikeD, f"spikes.npz missing {key!r}; rerun prep_inSilico3c.py"
    for key in ("data_type", "time_step_sec", "num_neurons", "provenance"):
        assert key in spikeMD, f"spikes.npz metadata missing {key!r}"
    for key in ("node_positions", "node_is_inhibitory"):
        assert key in truthD, f"prismTruth.npz missing {key!r}; rerun prep_inSilico3c.py"
    for key in ("short_name", "data_selector", "rate_summary", "dale_conf"):
        assert key in truthMD, f"prismTruth metadata missing {key!r}"
    assert spikeMD["data_type"] == "silico", f"unexpected data_type {spikeMD['data_type']!r}"
    return True


#...!...!....................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int,
                        help="increase output verbosity", default=1, dest="verb")
    parser.add_argument("-p", "--showPlots", default="a", nargs="+",
                        help="plots: a=freq histo, b=freq vs time, c=spatial layout")

    parser.add_argument("--plotFormat", choices=("png", "pdf"), default="png",
                        help="Output format for saved plots")
    parser.add_argument("-X", "--noXterm", action="store_true",
                        help="Disable X terminal for plotting")

    parser.add_argument("--dataPath",
                        default="/pscratch/sd/b/balewski/2025_causalNet_tmp/",
                        help="head dir for any further data processing")

    parser.add_argument("-T", "--time_range", default=[0., 60], nargs=2, type=float,
                        help="display data time range in seconds")
    parser.add_argument("--burst_freq_thres", default=5., type=float,
                        help="tags high freq channels for burst detection")
    parser.add_argument("--dataName", default=None, required=True,
                        help="preprocessed session short name")

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

    truthFF = os.path.join(args.dataPath, f"{args.dataName}.prismTruth.npz")
    print("prismTruth:", truthFF)
    truthD, truthMD = read_data_npz(truthFF)
    if args.verb > 1: pprint(truthMD)

    spikesFF = os.path.join(args.dataPath, f"{args.dataName}.spikes.npz")
    print("spikes:", spikesFF)
    spikeD, spikeMD = read_data_npz(spikesFF)
    if args.verb > 1: pprint(spikeMD)

    _require_inSilico(spikeD, spikeMD, truthD, truthMD)

    plotMD = {**truthMD, **spikeMD}
    args.prjName = spikeMD["provenance"]["state_transition_file"]

    rebD = None
    if "b" in args.showPlots:
        plotMD["plot"] = {"time_rangeLR": np.array(args.time_range)}
        rebD = detect_spike_bursts(
            spikeD, spikeMD, args.time_rebin2,
            args.burst_freq_thres,
        )

    plot = Plotter(args)

    if "a" in args.showPlots:
        plot.freq_histo(spikeD, truthD, plotMD, figId=1)
    if "b" in args.showPlots:
        plot.freq_vs_time(rebD, plotMD, figId=2)
    if "c" in args.showPlots:
        plot.neuron_spatial(truthD, spikeD, plotMD, figId=3)

    plot.display_all()
    print("M:done")
