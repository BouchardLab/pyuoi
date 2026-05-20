#!/usr/bin/env python3
"""
Evaluation and plotting for prism M-step results.
"""

import os
import argparse
from pprint import pprint
import numpy as np
from toolbox.Util_NumpyIO import read_data_npz
from PlotterPrismMstep import Plotter


def compute_edge_eval(A_hat, E_true, minW):
    """Compute edge-detection metrics/masks once; plotter only displays them."""
    A_hat = np.asarray(A_hat)
    E_true = np.asarray(E_true).astype(bool)

    if A_hat.ndim == 2:
        A_hat = A_hat[None, :, :]
    if E_true.ndim == 2:
        E_true = np.repeat(E_true[None, :, :], A_hat.shape[0], axis=0)

    n_states, N, N2 = A_hat.shape
    assert N == N2, "A_hat must be square per state"
    assert E_true.shape == (n_states, N, N), "E_true shape must match A_hat"

    off_diag = ~np.eye(N, dtype=bool)
    out = []
    for m in range(n_states):
        E_hat = (np.abs(A_hat[m]) > minW) & off_diag
        E_t = E_true[m] & off_diag

        TP = E_t & E_hat
        FP = (~E_t) & E_hat
        FN = E_t & (~E_hat)
        TN = (~E_t) & (~E_hat)

        tp = int(TP.sum())
        fp = int(FP.sum())
        fn = int(FN.sum())
        tn = int(TN.sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        acc = (tp + tn) / max(1, tp + tn + fp + fn)

        conf_map = np.zeros((N, N), dtype=np.int8)
        conf_map[FN] = 1
        conf_map[FP] = 2
        conf_map[TP] = 3

        out.append({
            "state": int(m),
            "minW": float(minW),
            "n_hat": int(E_hat.sum()),
            "n_true": int(E_t.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "acc": float(acc),
            "E_hat": E_hat.astype(np.uint8),
            "E_true_offdiag": E_t.astype(np.uint8),
            "conf_map": conf_map,
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot prism M-step results")
    parser.add_argument("--dataName", type=str, required=True, help="Base name for prismMstep file")
    parser.add_argument("--basePath", type=str, default="/pscratch/sd/b/balewski/2026_causalNet_tmp2/", help="head dir for input/output data")
    parser.add_argument("-p", "--showPlots", type=str, nargs='+', default="a", help=("Plot types: a=training summary, b=A-matrix edge evaluation (1-row: E_true|A_hat|confusion|stats), c=2D correlations (A_true vs A_hat, B_true vs B_hat), d=state correlations (fit), e=state correlations (truth), f=edge detection vs truth"))
    parser.add_argument("--minW", type=float, default=0.02, help="Threshold for A-matrix eval, not for fitting")
    parser.add_argument("--timeReb", type=int, default=20, help="Time rebin factor for plots with time axis")
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
    prov = fitMD["provenance"]
    truth_name = prov["state_model_file"]
    truthPath = os.path.join(args.basePath, "truthDale")
    truthFF = os.path.join(truthPath, f"{truth_name}.simTruth.npz")
    trueD, trueMD = read_data_npz(truthFF)
    if args.verb > 1:
        print("\nsimTruth metadata:"); pprint(trueMD)
        
    MD.update(trueMD)
    MD["A_true"] = trueD["A_true"]
    MD["B_true"] = trueD["B_true"]
    MD["E_true"] = trueD["E_true"]
    MD["short_name"] = args.dataName

    st_name = prov["state_transition_file"]
    prismTruthFF = os.path.join(args.basePath, "spikesData", f"{st_name}.prismTruth.npz")
    trD, trMD = read_data_npz(prismTruthFF, verb=args.verb > 1)
    MD["S_true"] = trD["S_true"]
    MD["C_true"] = trD["C_true"]
    edge_eval_states = compute_edge_eval(fitD["A_hat"], MD["E_true"], args.minW)
    MD["edge_eval_states"] = edge_eval_states
    MD["edge_eval_main"] = edge_eval_states[0]

    
    args.prjName = args.dataName
    plot = Plotter(args)

    if 'a' in args.showPlots:
        plot.summary_prismMstep(fitD, MD, figId=1)

    if 'b' in args.showPlots:
        plot.eval_Amatrix_prismMstep(fitD, MD, figId=2)

    if 'c' in args.showPlots:
        plot.eval_ABcorr_prismMstep(fitD, MD, figId=3)

    if 'd' in args.showPlots:
        plot.state_Bcorr_prismMstep(fitD, MD, figId=4)

 
    plot.display_all()


if __name__ == "__main__":
    main()
