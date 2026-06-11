#!/usr/bin/env python3
"""
Visualization tool for BSSM-STD network data.

Loads simulation output (simTruth.npz and spikes.npz) produced by
gen5_BSSM_STD_spikes.py. Neuron order follows placement; use
node_is_inhibitory in simTruth for E/I (not index blocks).

Available plots (-p flag):
  a  Recurrent weight matrix W (color-coded) + eigenvalue scatter
     with spectral-radius circle
  b  Weight histogram, in/out-degree counts, empirical firing-rate histogram,
     and baseline-vs-empirical rate scatter
  c  Latent BSSM-STD summaries: population resource trace, release/filter trace,
     mean spike-probability vs empirical rate, and mean resource vs rate
  d  2D neuron placement: triangles=excitatory, squares=inhibitory;
     red=outgoing excitatory edges, blue=outgoing inhibitory edges (presynaptic)
  e  Connection-length histogram, lag-M synaptic kernel, and STD recovery curve
  f  Pseudospectral contour plot with eigenvalue overlay
  g  Topology overview: signed off-diagonal matrix, distance histogram,
     and placement topology on one canvas

Usage:
    ./view_daleMatrix5.py --dataName daleN100_9fbe7f -p a b c d e f g
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import os
from pprint import pprint
import numpy as np
from PlotterDaleMatrix import Plotter
from toolbox.Util_NumpyIO import read_data_npz
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Visualize simulated BSSM-STD network data")
    parser.add_argument("-v", "--verbosity", type=int, help="increase output verbosity", default=1, dest="verb")
    parser.add_argument(
        "-p",
        "--showPlots",
        default="a b",
        nargs="+",
        help="plot letters: a=W+eigen, b=weights/rates, c=latent summaries, d=placement, e=dist+kernel+STD, f=pseudospectra, g=topo overview",
    )
    parser.add_argument("-X", "--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument("--basePath", default="/private/tmp/2025_causalNet_tmp/", help="head dir for input data")
    parser.add_argument("--dataName", default="daleN150_448b86", help="simulated Dale network base name")

    args = parser.parse_args()

    args.inpPath = os.path.join(args.basePath, "truthDale")
    args.outPath = os.path.join(args.basePath, "plots")
    args.showPlots = "".join(args.showPlots)

    print("myArg-program:", parser.prog)
    for arg in vars(args):
        print("myArg:", arg, getattr(args, arg))

    assert os.path.exists(args.basePath)
    return args


if __name__ == "__main__":
    args = get_parser()
    np.set_printoptions(precision=3)

    truthFF = os.path.join(args.inpPath, f"{args.dataName}.simTruth.npz")
    trueD, trueMD = read_data_npz(truthFF, verb=args.verb > 0)
    assert trueD["b_true"].ndim == 1, "b_true must be shape (N,) from gen5_BSSM_STD_spikes"
    assert trueMD["dale_conf"]["model_name"] == "BSSM_STD"
    if args.verb > 1:
        print("\nSimulation Truth Metadata:")
        pprint(trueMD)

    spikesFF = os.path.join(args.inpPath, f"{args.dataName}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF, verb=args.verb > 0)
    if args.verb > 1:
        print("\nSpike Data Metadata:")
        pprint(spikeMD)

    trueMD["short_name"] = args.dataName

    R_sel = trueMD["dale_conf"]["spectral_radius"]
    print(f"\nSpectral radius: R={R_sel:.3f}")

    trueMD["sel_spect_radius"] = R_sel

    args.prjName = args.dataName + "_view"
    plot = Plotter(args)

    if "a" in args.showPlots:
        plot.Dale_matrix_and_eigen(trueD["W_true"], trueMD, trueD, figId=1)

    if "b" in args.showPlots:
        plot.histo_weights_rates(trueD, spikeD, trueMD, figId=2)

    if "c" in args.showPlots:
        plot.rates_study(trueD, spikeD, trueMD, figId=3)

    if "d" in args.showPlots:
        plot.plot_placement_topology(trueD, trueMD, figId=4)

    if "e" in args.showPlots:
        plot.connection_length_kernel_hist(trueD, trueMD, figId=5)

    if "f" in args.showPlots:
        plot.Dale_matrix_pseudospectra(trueD["W_true"], trueMD, trueD, figId=6)

    if "g" in args.showPlots:
        plot.topo_overview(trueD, trueMD, figId=7)


    plot.display_all()
    print("M:done - view_daleMatrix5 completed successfully!")
