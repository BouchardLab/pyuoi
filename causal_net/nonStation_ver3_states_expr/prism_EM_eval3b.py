#!/usr/bin/env python3
"""
Evaluation and plotting for prism EM results.
"""

import os
import argparse
import numpy as np
from pprint import pprint
from toolbox.Util_NumpyIO import read_data_npz
from PlotterPrismEM import Plotter
from UtilBioExp import detect_spike_bursts


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

    a_hat = np.asarray(fitD["A_hat"], dtype=np.float64)
    b_hat = np.asarray(fitD["B_hat"], dtype=np.float64)
    c_hat = np.asarray(fitD["c_hat"], dtype=np.float64)
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


def eval_true_state_recovery(fitD, md):
    """Compute state recovery table on full training window."""
    trainMD = md["train"]
    t0_bin, t1_bin = [int(x) for x in trainMD["time_range_bins"]]
    s_true = np.asarray(md["S_true"], dtype=np.int64)[t0_bin : t1_bin + 1]
    s_hat = np.asarray(fitD["S_hat"], dtype=np.int64)
    s_hat_cl = np.asarray(fitD["S_hat_CL"], dtype=np.float64)
    m_states = int(trainMD["num_states"])

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


def compute_neuron_type(fitD, minW):
    """Compute and store neuron_type vector: -1=inh, 0=und, +1=exc."""
    A_hat = np.asarray(fitD["A_hat"], dtype=np.float64)
    assert A_hat.ndim == 2 and A_hat.shape[0] == A_hat.shape[1], "A_hat must be square"
    N = A_hat.shape[0]
    minW = float(minW)
    off_mask = ~np.eye(N, dtype=bool)
    A_thr = A_hat.copy()
    A_thr[off_mask & (np.abs(A_thr) < minW)] = 0.0
    np.fill_diagonal(A_thr, 0.0)
    Sedge = A_thr.sum(axis=1)
    neuron_type = np.zeros((N,), dtype=np.int8)
    neuron_type[Sedge > minW] = 1
    neuron_type[Sedge < -minW] = -1
    fitD["neuron_Sedge"] = Sedge.astype(np.float32)
    fitD["neuron_type"] = neuron_type
    return neuron_type


def compute_A_prune(fitD, neuron_type, minW):
    """Compute and store A_prune from A_hat and neuron_type."""
    A_hat = np.asarray(fitD["A_hat"], dtype=np.float64)
    neuron_type = np.asarray(neuron_type, dtype=np.int8)
    assert neuron_type.shape[0] == A_hat.shape[0], "neuron_type length must match A_hat rows"
    minW = float(minW)
    N = A_hat.shape[0]
    off_mask = ~np.eye(N, dtype=bool)
    A_prune = A_hat.copy()
    A_prune[off_mask & (np.abs(A_prune) < minW)] = 0.0
    diag_A = np.diag(A_hat).copy()
    exc_rows = neuron_type > 0
    inh_rows = neuron_type < 0
    A_prune[exc_rows, :] = np.where(A_prune[exc_rows, :] > 0, A_prune[exc_rows, :], 0.0)
    A_prune[inh_rows, :] = np.where(A_prune[inh_rows, :] < 0, A_prune[inh_rows, :], 0.0)
    np.fill_diagonal(A_prune, diag_A)
    fitD["A_prune"] = A_prune.astype(np.float32)
    return fitD["A_prune"]


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
                        help="Plot types: a=EM convergence summary, b=init-vs-truth states, c=A_init-vs-truth, d=A_hat-vs-truth, e=A_hat edge recovery, f=state sequence, g=2D correlations (A/B), h=A_init TP quality (diag/exc/inh), i=A_init/A_hat fitted-only, j=state+bioExp rates (no truth), k=bioExp spatial A_hat edges, m=Nedge/Sedge node stats, n=bioExp A_prune pos/neg edges")
    parser.add_argument("--minW", type=float, default=0.02,
                        help="Threshold for A-matrix edge eval")
    parser.add_argument("--timeReb", type=int, default=20,
                        help="Time rebin factor for time-axis plots")
    g = parser.add_argument_group("data")
    g.add_argument("-T", "--time_range_sec", default=[0.0, 65.0],
                   nargs=2, type=float,
                   help="Time window [t0, t1] in seconds")
    g.add_argument("-R", "--time_rebin2", default=50, type=int,
                   help="bioExp burst panels: rebin factor on spike time axis")
    g.add_argument("--burst_freq_thres", default=5.0, type=float,
                   help="bioExp burst panels: per-neuron rate threshold (Hz)")
    parser.add_argument("-X", "--noXterm", action="store_true",
                        help="Disable X terminal for plotting")
    parser.add_argument("-v", "--verb", type=int, default=1,
                        help="Verbosity level")
    args = parser.parse_args()

    args.inpPath = os.path.join(args.basePath, "prismFit")
    args.outPath = os.path.join(args.basePath, "plots")
    os.makedirs(args.outPath, exist_ok=True)
    args.showPlots = ''.join(args.showPlots)

    print("EM-eval args:",  vars(args), "\n")

    # ── load EM fit ──────────────────────────────────────────────────
    fitFF = os.path.join(args.inpPath, f"{args.dataName}.prismEM.npz")
    fitD, fitMD = read_data_npz(fitFF)
    assert isinstance(fitMD, dict), "Expected metadata dict in prismEM file"

    if args.verb > 1:  pprint(fitMD)

    MD = {**fitMD, "short_name": args.dataName}

    is_bioexp = fitMD.get("data_type") == "bioExp"
    prov = fitMD["provenance"]
    spikesPath = os.path.join(args.basePath, "spikesData")

    if is_bioexp:
        st_name = prov["experiment_name"]
        spikesFF = os.path.join(spikesPath, f"{st_name}.spikes.npz")
        print(f"bioExp data: skipped simTruth/prismTruth, spikes={spikesFF}")
        spikeD, spikeMD = read_data_npz(spikesFF, verb=args.verb > 1)
        spikes = spikeD["spikes"]
        bioFF = spikesFF.replace('spikes.npz', 'bioExp.npz')
        bioD, bioMD = read_data_npz(bioFF, verb=args.verb > 1)
        bio_plot_md = {**spikeMD, **bioMD}
        if args.verb > 1: pprint(bioMD)
    else:
        truth_name = prov["state_model_file"]
        truthPath = os.path.join(args.basePath, "truthDale")
        truthFF = os.path.join(truthPath, f"{truth_name}.simTruth.npz")
        trueD, trueMD = read_data_npz(truthFF, verb=args.verb > 1)
        if args.verb > 1:
            pprint(trueMD)
        MD.update(trueMD)
        MD["A_true"] = trueD["A_true"]
        MD["B_true"] = trueD["B_true"]
        MD["E_true"] = trueD["E_true"]

        st_name = prov["state_transition_file"]
        ptFF = os.path.join(spikesPath, f"{st_name}.prismTruth.npz")
        trD, _ = read_data_npz(ptFF, verb=args.verb > 1)
        MD["S_true"] = trD["S_true"]
        MD["C_true"] = trD["C_true"]

        spikesFF = os.path.join(spikesPath, f"{st_name}.spikes.npz")
        spikeD, spikeMD = read_data_npz(spikesFF, verb=args.verb > 1)
        spikes = spikeD["spikes"]

        MD["eval_f"] = eval_em_metrics_time(fitD, MD, spikes)

    if not is_bioexp:
        reco = eval_true_state_recovery(fitD, MD)
        MD["states_recovery_eval"]["avg_acc"] = reco["avg_acc"]
        MD["states_recovery_eval"]["state_acc_cl"] = reco["state_acc_cl"]

        print(f"state reco avr acc {reco['avg_acc']:.3f}, {args.dataName}")
        print(f"  {'state':>5s}  {'acc':>5s}  {'CL':>6s}  {'bins_hat':>8s}  {'enter_hat':>9s}")
        print(f"  {'-----':>5s}  {'-----':>5s}  {'------':>6s}  {'--------':>8s}  {'---------':>9s}")
        for m, (acc_m, cl_m, bins_m, enter_m) in enumerate(reco["state_acc_cl"]):
            print(f"  {m:5d}  {acc_m:5.3f}  {cl_m:6.3f}  {bins_m:8d}  {enter_m:9d}")

        trans_hat = np.asarray(reco["trans_hat"], dtype=np.int64)
        n_states = trans_hat.shape[0]
        print("\nS_hat transition counts (from row -> to col):")
        hdr = " from\\to" + "".join(f"{j:8d}" for j in range(n_states))
        print(hdr)
        print(" " + "-" * (len(hdr) - 1))
        for i in range(n_states):
            row = f"{i:8d}" + "".join(f"{int(trans_hat[i, j]):8d}" for j in range(n_states))
            print(row)

    MD["short_name"] = args.dataName
    neuron_type = compute_neuron_type(fitD, args.minW)
    compute_A_prune(fitD, neuron_type, args.minW)

    # ── plot ───────────────────────────
    if is_bioexp:
        assert not any(c in args.showPlots for c in "bcdefgh"), \
            "bioExp: plots b-h require ground truth; use -p a i j k m n"

    args.prjName = args.dataName
    plot = Plotter(args)

    if 'a' in args.showPlots:
        plot.summary_prismEM(fitD, MD, figId=1)

    if 'b' in args.showPlots:  # need truth
        plot.state_init_prismEM(fitD, MD, figId=2, time_reb=args.timeReb)

    if 'c' in args.showPlots:
        plot.matrix_init_prismEM(fitD, MD, figId=3, est_key="A_init", est_label="A_init")

    if 'd' in args.showPlots:
        plot.matrix_init_prismEM(fitD, MD, figId=3, est_key="A_hat", est_label="A_hat")

    if 'e' in args.showPlots:
        plot.edge_recovery_prismEM(
            fitD, MD, minW=args.minW, figId=4, est_key="A_hat", est_label="A_hat"
        )

    if 'f' in args.showPlots:
        plot.state_seq_prismEM(
            fitD, MD, figId=5, time_range_sec=args.time_range_sec
        )

    if 'g' in args.showPlots:
        plot.eval_ABcorr_prismEM(fitD, MD, figId=6)

    if 'h' in args.showPlots:
        plot.initA_quality_prismEM(fitD, MD, minW=args.minW, figId=7)

    if 'i' in args.showPlots:
        plot.A_fitted_prismEM(
            fitD, MD, spikeD["single_rates"], minW=args.minW, figId=8
        )

    if 'j' in args.showPlots:
        assert is_bioexp, "plot j requires bioExp data (experiment_name in provenance)"
        rebD = detect_spike_bursts(
            spikeD, spikeMD, args.time_rebin2,
            args.burst_freq_thres,
        )
        plot.state_seq_fitonly_prismEM(
            fitD, MD, rebD, bio_plot_md, figId=9,
            time_range_sec=args.time_range_sec,
        )

    if 'k' in args.showPlots:
        assert is_bioexp, "plot k requires bioExp data (experiment_name in provenance)"
        plot.neuron_spatial_Ahat_edges(
            fitD, bioD, bio_plot_md, minW=args.minW, figId=10
        )

    if 'm' in args.showPlots:
        plot.node_outgoing_edge_stats_prismEM(
            fitD, MD, spikeD["single_rates"], neuron_type,
            minW=args.minW, figId=11,
            est_key="A_hat", est_label="A_hat",
        )

    if 'n' in args.showPlots:
        assert is_bioexp, "plot n requires bioExp data (experiment_name in provenance)"
        plot.neuron_spatial_Ahat_edges_split(
            fitD, bioD, bio_plot_md, neuron_type, fitD["neuron_Sedge"],
            minW=args.minW, maxNeurons=24, figId=12
        )

    plot.display_all()


if __name__ == "__main__":
    main()
