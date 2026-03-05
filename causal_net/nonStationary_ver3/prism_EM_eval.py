#!/usr/bin/env python3
"""
Evaluation and plotting for prism EM results.
"""

import os
import argparse
from pprint import pprint
from toolbox.Util_NumpyIO import read_data_npz
from PlotterPrismEM import Plotter


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and plot prism EM results")
    parser.add_argument("--dataName", type=str, required=True,
                        help="Base name for prismEM file")
    parser.add_argument("--basePath", type=str,
                        default="/pscratch/sd/b/balewski/2026_causalNet_tmp3/",
                        help="Head dir for input/output data")
    parser.add_argument("-p", "--showPlots", type=str, nargs='+',
                        default="a",
                        help="Plot types: a=EM convergence summary")
    parser.add_argument("--minW", type=float, default=0.02,
                        help="Threshold for A-matrix edge eval")
    parser.add_argument("--timeReb", type=int, default=20,
                        help="Time rebin factor for time-axis plots")
    parser.add_argument("-X", "--noXterm", action="store_true",
                        help="Disable X terminal for plotting")
    parser.add_argument("-v", "--verb", type=int, default=1,
                        help="Verbosity level")
    args = parser.parse_args()

    args.inpPath = os.path.join(args.basePath, "prismFit")
    args.outPath = os.path.join(args.basePath, "plots")
    os.makedirs(args.outPath, exist_ok=True)
    args.showPlots = ''.join(args.showPlots)

    print(vars(args))

    # ── load EM fit ──────────────────────────────────────────────────
    fitFF = os.path.join(args.inpPath, f"{args.dataName}.prismEM.npz")
    fitD, fitMD = read_data_npz(fitFF)
    assert isinstance(fitMD, dict), "Expected metadata dict in prismEM file"

    if args.verb > 1:
        pprint(fitMD)

    MD = {**fitMD, "short_name": args.dataName}
   
    # ── load ground truth if available ───────────────────────────────
    prov = fitMD.get("provenance", {})
    truth_name = prov.get("state_model_file")
    if truth_name:
        truthPath = os.path.join(args.basePath, "truthDale")
        truthFF = os.path.join(truthPath, f"{truth_name}.simTruth.npz")
        if os.path.isfile(truthFF):
            trueD, trueMD = read_data_npz(truthFF, verb=args.verb > 1)
            MD.update(trueMD)
            MD["A_true"] = trueD["A_true"]
            MD["B_true"] = trueD["B_true"]
            MD["E_true"] = trueD["E_true"]

    st_name = prov.get("state_transition_file")
    if st_name:
        ptFF = os.path.join(args.basePath, "spikesData",
                            f"{st_name}.prismTruth.npz")
        if os.path.isfile(ptFF):
            trD, _ = read_data_npz(ptFF, verb=args.verb > 1)
            MD["S_true"] = trD["S_true"]
            MD["C_true"] = trD["C_true"]

    MD["short_name"] = args.dataName

    # ── plot ──────────────────────────────────────────────────────────
    args.prjName = args.dataName
    plot = Plotter(args)

    if 'a' in args.showPlots:
        plot.summary_prismEM(fitD, MD, figId=1)

    plot.display_all()


if __name__ == "__main__":
    main()
