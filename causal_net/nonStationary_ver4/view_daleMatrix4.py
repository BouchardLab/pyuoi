#!/usr/bin/env python3
"""
Visualization tool for simulated Dale Poisson network data.

Loads simulation output (simTruth.npz and spikes.npz) produced by
gen_daleMatrices4.py. Neuron order follows placement; use node_is_inhibitory
in simTruth for E/I (not index blocks).

Available plots (-p flag):
  a  Dale connectivity matrix (color-coded) + eigenvalue scatter
     with spectral-radius circle
  b  Weight histogram, firing-rate histogram, outgoing-edge count
     per neuron, and per-neuron firing-rate bar chart
  c  B_idle vs firing rate / SNR scatter, plus excitatory and
     inhibitory rate histograms (rates_study)
  d  Pseudospectral contour plot with eigenvalue overlay
  e  2D neuron placement: circles=excitatory, triangles=inhibitory;
     red=outgoing excitatory edges, blue=outgoing inhibitory edges (presynaptic)
  f  Off-diagonal distance histogram, offdiag_kernel histogram, empty panel

Usage:
    ./view_daleMatrix4.py --dataName daleN100_9fbe7f -p a b c d e f
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
    parser = argparse.ArgumentParser(description="Visualize simulated Dale Poisson network data")
    parser.add_argument("-v", "--verbosity", type=int, help="increase output verbosity", default=1, dest="verb")
    parser.add_argument(
        "-p",
        "--showPlots",
        default="a b",
        nargs="+",
        help="plot letters: a=Dale+eigen, b=histograms, c=rates_study, d=pseudospectra, e=placement, f=dist/kernel hist",
    )
    parser.add_argument("-X", "--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument("--basePath", default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="head dir for input data")
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
    assert trueD["B_true"].ndim == 1, "B_true must be shape (N,) from gen_daleMatrices4"
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

    trueD_r = {
        "A_true": trueD["A_true"],
        "B_true": trueD["B_true"],
        "E_true": trueD["E_true"],
        "node_is_inhibitory": trueD["node_is_inhibitory"],
    }
    spikeD_r = dict(spikeD)

    trueMD["sel_spect_radius"] = R_sel

    args.prjName = args.dataName + "_view"
    plot = Plotter(args)

    if "a" in args.showPlots:
        plot.Dale_matrix_and_eigen(trueD_r["A_true"], trueMD, trueD_r, figId=1)

    if "b" in args.showPlots:
        plot.histo_weights_rates(trueD_r, spikeD_r, trueMD, figId=2)

    if "c" in args.showPlots:
        plot.rates_study(trueD_r, spikeD_r, trueMD, figId=3)

    if "d" in args.showPlots:
        plot.Dale_matrix_pseudospectra(trueD_r["A_true"], trueMD, trueD_r, figId=4)

    if "e" in args.showPlots:
        plot.plot_placement_topology(trueD, trueMD, figId=5)

    if "f" in args.showPlots:
        plot.offdiag_distance_kernel_hist(trueD, trueMD, figId=6)

    plot.display_all()
    print("M:done - view_dalePoisson completed successfully!")
