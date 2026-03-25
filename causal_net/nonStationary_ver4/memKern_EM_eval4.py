#!/usr/bin/env python3
"""
Evaluation and plotting for prism EM results.
"""

import os
import argparse
import numpy as np
from pprint import pprint
from toolbox.Util_NumpyIO import read_data_npz
from PlotterMemKernEM import Plotter


def compute_ll_gap(spikes_sub, A_true, B_true, dt, eta_clip):
    """Per-time LL gap between best and 2nd-best true-state likelihood."""
    y_prev = np.asarray(spikes_sub[:-1], dtype=np.float64)
    y_curr = np.asarray(spikes_sub[1:], dtype=np.float64)
    a_true = np.asarray(A_true, dtype=np.float64)
    b_true = np.asarray(B_true, dtype=np.float64)

    t_pairs = y_prev.shape[0]
    m_states = b_true.shape[0]
    ll_gap = np.zeros((t_pairs,), dtype=np.float64)
    if m_states <= 1:
        return ll_gap

    for t in range(t_pairs):
        yp = y_prev[t]
        yc = y_curr[t]
        base = a_true @ yp
        eta = base[None, :] + b_true
        eta_c = np.minimum(eta, float(eta_clip))
        lam = np.exp(eta_c) * float(dt)
        scores = np.sum(yc[None, :] * eta_c - lam, axis=1)
        top2 = np.partition(scores, -2)[-2:]
        ll_gap[t] = float(top2[-1] - top2[-2])
    return ll_gap


def eval_em_metrics_time(fitD, md, spikes):
    """Compute per-time metrics for -p f canvas."""
    trainMD = md["train"]
    t0_bin, t1_bin = [int(x) for x in trainMD["time_range_bins"]]
    dt = float(trainMD["time_step_sec"])
    eta_clip = float(trainMD["eta_clip"])
    lambda2 = float(trainMD["lambda2"])

    spikes_sub = np.asarray(spikes[t0_bin : t1_bin + 1], dtype=np.float64)
    yp = spikes_sub[:-1]
    yc = spikes_sub[1:]

    # Model B uses different keys
    a_hat = fitD.get("A_hat", fitD.get("A_off_hat"))
    b_hat = fitD.get("B_hat")
    c_hat = fitD.get("c_hat")
    
    if a_hat is None or b_hat is None or c_hat is None:
        return {}

    a_hat = np.asarray(a_hat, dtype=np.float64)
    b_hat = np.asarray(b_hat, dtype=np.float64)
    c_hat = np.asarray(c_hat, dtype=np.float64)

    n_pairs = spikes_sub.shape[0] - 1
    assert c_hat.shape[0] == spikes_sub.shape[0], "c_hat and spikes_sub must have matching time bins"
    c_pairs = c_hat[1:]
    c_prev = c_hat[:-1]

    rates = np.asarray(fitD["single_rates"], dtype=np.float64)
    w = 1.0 / np.maximum(rates, 0.1)
    w /= w.mean()

    eta = yp @ a_hat.T + c_pairs @ b_hat
    eta_c = np.minimum(eta, eta_clip)
    lam = np.exp(eta_c) * dt
    log_dt = np.log(dt)
    nll_t = np.sum(w[None, :] * (lam - yc * (eta + log_dt)), axis=1)

    dc = c_pairs - c_prev
    l2_t = lambda2 * np.sum(dc * dc, axis=1)

    ll_gap = compute_ll_gap(spikes_sub, md["A_true"], md["B_true"], dt, eta_clip)

    s_true = np.asarray(md["S_true"])[t0_bin : t1_bin + 1]
    s_hat = np.asarray(fitD["S_hat"])
    assert s_true.shape[0] == s_hat.shape[0], "S_true and S_hat must have matching length"
    acc = float((s_true == s_hat).mean())

    return {
        "acc": acc,
        "loss_nll_time": nll_t,
        "loss_l2_time": l2_t,
        "ll_gap": ll_gap,
        "ll_gap_mean": float(np.mean(ll_gap)),
    }


def eval_state_recovery(fitD, md):
    """Compute state recovery table on full training window."""
    trainMD = md["train"]
    t0_bin, t1_bin = [int(x) for x in trainMD["time_range_bins"]]
    s_true = np.asarray(md["S_true"], dtype=np.int64)[t0_bin : t1_bin + 1]
    s_hat = np.asarray(fitD["S_hat"], dtype=np.int64)
    s_hat_cl = np.asarray(fitD["S_hat_CL"], dtype=np.float64)
    m_states = int(trainMD.get("num_states", 1))

    assert s_true.shape[0] == s_hat.shape[0] == s_hat_cl.shape[0], \
        "S_true/S_hat/S_hat_CL length mismatch on training window"

    avg_acc = float((s_true == s_hat).mean())
    state_acc_cl = []
    bins_hat = np.bincount(s_hat, minlength=m_states).astype(np.int64)
    enter_hat = np.zeros(m_states, dtype=np.int64)
    if s_hat.shape[0] > 1:
        enter_idx = np.where(s_hat[1:] != s_hat[:-1])[0] + 1
        if enter_idx.size > 0:
            enter_hat = np.bincount(s_hat[enter_idx], minlength=m_states).astype(np.int64)

    trans_hat = np.zeros((m_states, m_states), dtype=np.int64)
    if s_hat.shape[0] > 1:
        np.add.at(trans_hat, (s_hat[:-1], s_hat[1:]), 1)

    for m in range(m_states):
        mask = (s_hat == m)
        cnt = int(mask.sum())
        if cnt > 0:
            acc_m = float((s_true[mask] == m).mean())
            cl_m = float(s_hat_cl[mask].mean())
        else:
            acc_m = float("nan")
            cl_m = float("nan")
        state_acc_cl.append([acc_m, cl_m, int(bins_hat[m]), int(enter_hat[m])])  # [acc, CL, bins_hat, enter_hat]

    return {
        "avg_acc": avg_acc,
        "state_acc_cl": state_acc_cl,
        "trans_hat": trans_hat,
    }


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
                        help="Plot types: a=EM convergence summary, b=init-vs-truth states, c=A_init-vs-truth, d=A_hat-vs-truth, e=A_hat edge recovery, f=state sequence, g=2D correlations (A/B), h=A_init TP quality (diag/exc/inh)")
    parser.add_argument("--minW", type=float, default=0.02,
                        help="Threshold for A-matrix edge eval")
    parser.add_argument("--timeReb", type=int, default=20,
                        help="Time rebin factor for time-axis plots")
    g = parser.add_argument_group("data")
    g.add_argument("-T", "--time_range_sec", default=[0.0, 15.0],
                   nargs=2, type=float,
                   help="Time window [t0, t1] in seconds")
    parser.add_argument("-X", "--noXterm", action="store_true",
                        help="Disable X terminal for plotting")
    parser.add_argument("-v", "--verb", type=int, default=1,
                        help="Verbosity level")
    args = parser.parse_args()

    args.inpPath = os.path.join(args.basePath, "memKernFit")
    args.outPath = os.path.join(args.basePath, "plots")
    os.makedirs(args.outPath, exist_ok=True)
    
    show_req = ''.join(args.showPlots)
    args.showPlots = ""
    for c in show_req:
        if c in "abcdefgh" and c not in args.showPlots:
            args.showPlots += c

    print("EM-eval args:",  vars(args), "\n")

    # ── load EM fit ──────────────────────────────────────────────────
    fitFF = os.path.join(args.inpPath, f"{args.dataName}.memKernEM.npz")
    fitD, fitMD = read_data_npz(fitFF)
    assert isinstance(fitMD, dict), "Expected metadata dict in prismEM file"

    if args.verb > 1:  pprint(fitMD)

    MD = {**fitMD}  #, "short_name": args.dataName}

    #--- load spikes ----
    prov = fitMD["provenance"]
    spikesF=prov['spiksData_file']
    spikesFF = os.path.join(args.basePath, "truthDale", f"{spikesF}.spikes.npz")  # should be; "spikesData"
    spikeD, _ = read_data_npz(spikesFF, verb=args.verb > 1)
    spikes = spikeD["spikes"]
 
    # ── load ground truth if available ───────────────────────────────
    prov = fitMD["provenance"]
    trueF=spikesF
    trueFF = os.path.join(args.basePath, "truthDale", f"{trueF}.simTruth.npz")
    trueD, trueMD = read_data_npz(trueFF, verb=args.verb > 1)
    if args.verb > 1:  pprint(trueMD)
    
    has_truth = False
    for xx in [ 'dale_conf', 'evol_conf']:
        MD[xx]=trueMD[xx]
    
    
    MD["A_off_true"] = trueD["A_off_true"]
    MD["A_diag_true"] = trueD["A_diag_true"]
    MD["B_true"] = trueD["B_true"]
    MD["E_true"] = trueD["E_true"]
    MD["offdiag_kernel"] = trueD["offdiag_kernel"]
    MD["A_true"] = MD["A_off_true"] + np.diag(MD["A_diag_true"])
    MD["eval_f"] = eval_em_metrics_time(fitD, MD, spikes)
    
    #MD["short_name"] = args.dataName

    # ── plot ──────────────────────────────────────────────────────────
    args.prjName = args.dataName
    plot = Plotter(args)

    if 'a' in args.showPlots:
        plot.summary_memKerEM(fitD, MD, figId=1)

    if 'b' in args.showPlots:
        plot.correl_fit_truth(fitD, MD, figId=2)

  
    plot.display_all()


if __name__ == "__main__":
    main()
