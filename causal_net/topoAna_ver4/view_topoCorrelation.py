#!/usr/bin/env python3
"""
Inspect how saved topology-analysis metrics vary with placement_ker_delta.

Usage:
    ./view_topoCorrelation.py --nameTemplate aN200 -p a
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import argparse
import glob
import os

import numpy as np

from PlotterTopoCorrel import Plotter
from toolbox.Util_NumpyIO import read_data_npz


METRIC_SPECS = (
    ("assortativity", "r_assortativity", "r_assort"),
    ("jaccard_index", "mean_jaccard", "jaccard"),
    ("cycle_decay", "cycle_decay", "cycle_decay"),
    ("homology_k1", "homology_k1", "hom_k1"),
)


def get_parser():
    parser = argparse.ArgumentParser(description="Visualize metric-vs-kernel correlations from *.topoAna.npz")
    parser.add_argument("-v", "--verbosity", type=int, help="increase output verbosity", default=1, dest="verb")
    parser.add_argument(
        "-p",
        "--showPlots",
        default="a",
        nargs="+",
        help="plot letters: a=metric inspector",
    )
    parser.add_argument("-X", "--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument("--basePath", default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="head dir for input data")
    parser.add_argument("--inpPath", default=None, help="directory with *.topoAna.npz; defaults to basePath/topoAna")
    parser.add_argument("--outPath", default=None, help="directory for plots; defaults to basePath/plots")
    parser.add_argument("--nameTemplate", default="aN200", help="load files matching inpPath/[nameTemplate]*.topoAna.npz")

    args = parser.parse_args()

    if args.inpPath is None:
        args.inpPath = os.path.join(args.basePath, "topoAna")
    if args.outPath is None:
        args.outPath = os.path.join(args.basePath, "plots")

    args.showPlots = "".join(args.showPlots)

    print("myArg-program:", parser.prog)
    for arg in vars(args):
        print("myArg:", arg, getattr(args, arg))

    assert os.path.exists(args.inpPath), "missing input path: %s" % args.inpPath
    os.makedirs(args.outPath, exist_ok=True)
    return args


def _data_name_from_file(inpFF, inpMD):
    provD = inpMD.get("provenance", {})
    if "A-input" in provD:
        return provD["A-input"]
    base = os.path.basename(inpFF)
    return base.replace(".topoAna.npz", "")


def harvest_topology_metrics(args):
    inpPattern = os.path.join(args.inpPath, "%s*.topoAna.npz" % args.nameTemplate)
    inpFL = sorted(glob.glob(inpPattern))
    if not inpFL:
        raise FileNotFoundError("no files found for pattern: %s" % inpPattern)

    if args.verb > 0:
        print("\nScanning %d files from pattern:\n  %s" % (len(inpFL), inpPattern))

    harvestD = {metric_name: {} for metric_name, _, _ in METRIC_SPECS}

    print("\n#Summary: dataName, ker_delta, r_assort, jaccard, cycle_decay, hom_k1")
    j=0
    for inpFF in inpFL:
        _, inpMD = read_data_npz(inpFF, verb=j==0)
        j += 1

        dmd = inpMD["dale_conf"]
        methodsD = inpMD["methods"]
        placement_L = int(round(float(dmd["placement_L"])))
        ker_delta = float(dmd["placement_ker_delta"])
        data_name = _data_name_from_file(inpFF, inpMD)

        metric_values = {}
        for method_name, observable_name, _ in METRIC_SPECS:
            metric_val = float(methodsD[method_name][observable_name])
            metric_values[method_name] = metric_val
            harvestD[method_name].setdefault(placement_L, []).append((ker_delta, metric_val, data_name))

        print(
            "#Values: %s  %.2f  %.6f  %.6f  %.6f  %.6f"
            % (
                data_name,
                ker_delta,
                metric_values["assortativity"],
                metric_values["jaccard_index"],
                metric_values["cycle_decay"],
                metric_values["homology_k1"],
            )
        )
    print("\nHarvested data for %d files." % j)
    for metric_name in harvestD:
        for placement_L in harvestD[metric_name]:
            harvestD[metric_name][placement_L] = sorted(harvestD[metric_name][placement_L], key=lambda rec: (rec[0], rec[2]))

    return harvestD


if __name__ == "__main__":
    args = get_parser()
    np.set_printoptions(precision=3)

    harvestD = harvest_topology_metrics(args)

    args.prjName = args.nameTemplate + "_topoCorr"
    plot = Plotter(args)

    if "a" in args.showPlots:
        plot.metrics_inspector(harvestD, args.nameTemplate, figId=1)

    plot.display_all()
    print("M:done - view_topoCorrelation completed successfully!")
