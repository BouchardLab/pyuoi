#!/usr/bin/env python3
"""
Evaluation and plotting for prism M-step results.
"""

import os
import argparse
from pprint import pprint
from toolbox.Util_NumpyIO import read_data_npz
from PlotterPrismMstep import Plotter


def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot prism M-step results")
    parser.add_argument("--dataName", type=str, required=True, help="Base name for prismMstep file")
    parser.add_argument("--basePath", type=str, default="/pscratch/sd/b/balewski/2026_causalNet_tmp2/", help="head dir for input/output data")
    parser.add_argument(
        "-p",
        "--showPlots",
        type=str,
        nargs='+',
        default="a",
        help=(
            "Plot types: a=training summary, b=2D correlations (A_true vs A_hat, B_true vs B_hat), "
            "c=B_true vs B_hat, d=state correlations (fit), e=state correlations (truth), "
            "f=edge detection vs truth, g=A_true vs A_hat"
        ),
    )
    parser.add_argument("--minW", type=float, default=0.02, help="Threshold for A-correlation sub-regions in scatter plots")
    parser.add_argument("--divideB", type=float, default=2.5, help="B-data x-axis divisor for correlation plots")
    parser.add_argument("-X", "--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument("-v", "--verb", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()

    args.inpPath = os.path.join(args.basePath, "prismFit")
    args.outPath = os.path.join(args.basePath, "plots")
    os.makedirs(args.outPath, exist_ok=True)
    args.showPlots = ''.join(args.showPlots)

    print(vars(args))

    fitFF = os.path.join(args.inpPath, f"{args.dataName}.prismMstep.npz")
    fitD, fitMD = read_data_npz(fitFF)
    assert isinstance(fitMD, dict), "Expected metadata dict in prismMstep file"

    if args.verb > 1:
        pprint(fitMD)

    MD = {**fitMD, "short_name": args.dataName}

    # Load ground truth (A_true/B_true) if available
    prov = fitMD.get("provenance", {})
    truth_name = prov.get("state_model_file")
    if truth_name:
        truthPath = os.path.join(args.basePath, "truthDale")
        truthFF = os.path.join(truthPath, f"{truth_name}.simTruth.npz")
        if os.path.exists(truthFF):
            trueD, trueMD = read_data_npz(truthFF)
            MD.update(trueMD)
            MD["A_true"] = trueD.get("A_true")
            MD["B_true"] = trueD.get("B_true")
            MD["E_true"] = trueD.get("E_true")
            MD["short_name"] = args.dataName
        else:
            if args.verb > 0:
                print(f"Warning: missing truth file: {truthFF}")
    else:
        if args.verb > 0:
            print("Warning: provenance missing state_model_file; skipping truth load")

    args.prjName = args.dataName
    plot = Plotter(args)

    if 'a' in args.showPlots: 
        plot.summary_prismMstep(fitD, MD, figId=1)

    if 'b' in args.showPlots:
        plot.eval_ABcorr_prismMstep(fitD, MD, figId=2)

    if 'c' in args.showPlots:
        free1

    if 'd' in args.showPlots:
        plot.state_ABcorr_prismMstep(fitD, MD, type="fit", figId=4)

    if 'e' in args.showPlots:
        plot.state_ABcorr_prismMstep(fitD, MD, type="truth", figId=4)

    if 'f' in args.showPlots:  # E-hat  per state
        plot.edge_state_prismMstep(fitD, MD, figId=6)

 
    plot.display_all()


if __name__ == "__main__":
    main()
