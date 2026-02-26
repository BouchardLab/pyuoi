#!/usr/bin/env python3
"""
Evaluation and plotting for prism E-step results.
"""

import os
import argparse
from pprint import pprint

from toolbox.Util_NumpyIO import read_data_npz
from PlotterPrismEstep import Plotter


def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot prism E-step results")
    parser.add_argument("--dataName", type=str, required=True, help="Base name for prismEstep file")
    parser.add_argument("--basePath", type=str, default="/pscratch/sd/b/balewski/2026_causalNet_tmp2/", help="head dir for input/output data")
    parser.add_argument("-p", "--showPlots", type=str, nargs='+', default="a", help="Plot types: a=training summary")
    parser.add_argument("--time_bin_merge", type=int, default=20, help="Merge factor for loss(time) x-axis")
    parser.add_argument("-X", "--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument("-v", "--verb", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()

    args.inpPath = os.path.join(args.basePath, "prismFit")
    args.outPath = os.path.join(args.basePath, "plots")
    os.makedirs(args.outPath, exist_ok=True)
    args.showPlots = ''.join(args.showPlots)

    print(vars(args))

    fitFF = os.path.join(args.inpPath, f"{args.dataName}.prismEstep.npz")
    fitD, fitMD = read_data_npz(fitFF)
    assert isinstance(fitMD, dict), "Expected metadata dict in prismEstep file"

    if args.verb > 1:
        pprint(fitMD)

    MD = {**fitMD, "short_name": args.dataName}

    args.prjName = args.dataName
    plot = Plotter(args)

    if 'a' in args.showPlots:
        plot.summary_prismEstep(fitD, MD, figId=1, time_bin_merge=args.time_bin_merge)

    plot.display_all()


if __name__ == "__main__":
    main()
