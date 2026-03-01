#!/usr/bin/env python3
"""
Evaluation and plotting for prism E-step results.
"""

import os
import argparse
import numpy as np
from pprint import pprint

from toolbox.Util_NumpyIO import read_data_npz
from PlotterPrismEstep import Plotter


def compute_ll_gap(spikes, A_true, B_true, t0_bin, t1_bin, dt, eta_clip):
    spikes_sub = spikes[t0_bin : t1_bin + 1].astype(np.float64)
    y_prev = spikes_sub[:-1]
    y_curr = spikes_sub[1:]
    T_pairs = y_prev.shape[0]
    M = B_true.shape[0]
    ll_gap = np.zeros((T_pairs,), dtype=np.float64)
    if M == 1:
        return ll_gap
    for t in range(T_pairs):
        yp = y_prev[t]
        yc = y_curr[t]
        base = A_true @ yp
        eta = base[None, :] + B_true
        eta_c = np.minimum(eta, eta_clip)
        lam = np.exp(eta_c) * dt
        scores = np.sum(yc * eta_c - lam, axis=1)
        top2 = np.partition(scores, -2)[-2:]
        ll_gap[t] = top2[-1] - top2[-2]
    return ll_gap


def eval_estep_metrics(fitD, md, spikes):
    trainMD = md["train"]
    t0_bin, t1_bin = trainMD["time_range_bins"]
    s_true = md["S_true"][t0_bin : t1_bin + 1]
    s_hat = fitD["S_hat"]
    s_hat_cl = fitD["S_hat_CL"]
    n_cmp = min(s_true.shape[0], s_hat.shape[0])
    if n_cmp > 0:
        acc = float((s_true[:n_cmp] == s_hat[:n_cmp]).mean())
    else:
        acc = float("nan")

    # Per-state average S_hat_CL and accuracy
    M = int(trainMD["num_states"])
    avg_cl_per_state = []
    acc_per_state = []
    for m in range(M):
        mask = s_hat[:n_cmp] == m
        cnt = int(mask.sum())
        if cnt > 0:
            avg_cl_per_state.append(float(s_hat_cl[:n_cmp][mask].mean()))
            acc_per_state.append(float((s_true[:n_cmp][mask] == m).mean()))
        else:
            avg_cl_per_state.append(float("nan"))
            acc_per_state.append(float("nan"))

    # Store per-state metrics in decode_eval metadata
    md["decode_eval"]["avg_cl_per_state"] = avg_cl_per_state
    md["decode_eval"]["acc_per_state"] = acc_per_state

    A_true = md["A_true"]
    B_true = md["B_true"]
    dt = float(trainMD["time_step_sec"])
    eta_clip = float(trainMD["eta_clip"])
    ll_gap = compute_ll_gap(spikes, A_true, B_true, t0_bin, t1_bin, dt, eta_clip)
    return {
        "acc": acc,
        "ll_gap": ll_gap,
        "ll_gap_mean": float(np.mean(ll_gap)) if ll_gap.size else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot prism E-step results")
    parser.add_argument("--dataName", type=str, required=True, help="Base name for prismEstep file")
    parser.add_argument("--basePath", type=str, default="/pscratch/sd/b/balewski/2026_causalNet_tmp2/", help="head dir for input/output data")
    parser.add_argument("-p", "--showPlots", type=str, nargs='+', default="a", help="Plot types: a=training summary, b=state sequence (fit vs truth)")
    parser.add_argument("--time_bin_merge", type=int, default=2, help="Merge factor for loss(time) x-axis")
    parser.add_argument("-X", "--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument("-v", "--verb", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()


    truthPath = os.path.join(args.basePath,"spikesData")
    args.inpPath = os.path.join(args.basePath, "prismFit")
    args.outPath = os.path.join(args.basePath, "plots")
    os.makedirs(args.outPath, exist_ok=True)
    args.showPlots = ''.join(args.showPlots)

    print(vars(args))

    fitFF = os.path.join(args.inpPath, f"{args.dataName}.prismEstep.npz")
    fitD, fitMD = read_data_npz(fitFF)
    if args.verb > 1:    pprint(fitMD)

    truthF = fitMD["provenance"]['state_transition_file']
    truthFF = os.path.join(truthPath, f"{truthF}.prismTruth.npz")
    trueD, trueMD = read_data_npz(truthFF, verb=args.verb > 0)
    A_true = trueD["A_true"]
    B_true = trueD["B_true"]
    C_true = trueD["C_true"]
    S_true = trueD["S_true"]
    if args.verb > 1:    pprint(trueMD)
    
    spikesFF = os.path.join(truthPath, f"{truthF}.spikes.npz")
    spikesD, spikesMD = read_data_npz(spikesFF, verb=args.verb > 0)
    spikes = spikesD["spikes"]

    MD = {**fitMD, "short_name": args.dataName}
    MD["A_true"] = A_true
    MD["B_true"] = B_true
    MD["C_true"] = C_true
    MD["S_true"] = S_true
    MD["eval_Estep"] = eval_estep_metrics(fitD, MD, spikes) 
    decE = MD["decode_eval"]
    print(
        f"E-step eval avr acc {MD['eval_Estep']['acc']:.3f}, {args.dataName}"
    )
    print(f"  {'state':>5s}  {'CL':>6s}  {'acc':>5s}")
    print(f"  {'-----':>5s}  {'------':>6s}  {'-----':>5s}")
    for m, (cl, ac) in enumerate(zip(decE["avg_cl_per_state"], decE["acc_per_state"])):
        print(f"  {m:5d}  {cl:6.3f}  {ac:5.3f}")

    #pprint(decE)
    args.prjName = args.dataName
    plot = Plotter(args) 

    if 'a' in args.showPlots:
        plot.summary_prismEstep(fitD, MD, figId=1, time_bin_merge=args.time_bin_merge)

    if 'b' in args.showPlots:
        plot.state_seq_prismEstep(fitD, MD, figId=2, time_reb=args.time_bin_merge)

    plot.display_all()


if __name__ == "__main__":
    main()
