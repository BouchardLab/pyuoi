#!/usr/bin/env python3
"""Evaluation and plotting for prism EM 3c results."""

import argparse
import os
import re
from pprint import pprint

import numpy as np

from toolbox.Util_NumpyIO import read_data_npz
from PlotterPrismEM3c import Plotter
from UtilBioExp import detect_spike_bursts


PLOT_FIG_ID = {chr(ord("a") + i): chr(ord("a") + i) for i in range(18)}
SIM_ONLY_PLOTS = set("mnopqr")
IMPLEMENTED_PLOTS = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "m", "n", "o", "p", "r"}


def real_fit_metadata(md):
    if is_stage_c_metadata(md):
        return md
    if "bagsFDR_stageA" in md and "real_fit" in md["bagsFDR_stageA"]:
        return md["bagsFDR_stageA"]["real_fit"]
    return md


def is_stage_c_metadata(md):
    return md.get("fit_type") == "prismEM_deBias_stageC" or "deBias_stageC" in md


def display_fit_data(fitD, fitMD):
    """Map Stage (c) output arrays onto legacy plotting keys without mutating input."""
    if not is_stage_c_metadata(fitMD):
        return fitD
    for key in ("A_debias", "B_debias"):
        assert key in fitD, f"Stage (c) display requires {key}"

    disp = dict(fitD)
    for key in ("A_hat", "A_prune", "A_init", "B_hat", "neuron_Sedge",
                "A_diag_mean", "A_diag_stderr"):
        if key in fitD:
            disp["stageB_" + key] = fitD[key]

    A_debias = np.asarray(fitD["A_debias"])
    disp["A_hat"] = fitD["A_debias"]
    disp["B_hat"] = fitD["B_debias"]
    disp["A_prune"] = fitD["A_debias"]
    disp["A_init"] = fitD["A_debias_init"] if "A_debias_init" in fitD else fitD["A_hat"]
    if "neuron_Sedge_debias" in fitD:
        disp["neuron_Sedge"] = fitD["neuron_Sedge_debias"]
    else:
        off_mask = ~np.eye(A_debias.shape[0], dtype=bool)
        disp["neuron_Sedge"] = (A_debias * off_mask).sum(axis=0).astype(np.float32)
    if "A_diag_debias" in fitD:
        diag = np.asarray(fitD["A_diag_debias"], dtype=np.float32)
    else:
        diag = np.diag(A_debias).astype(np.float32)
    disp["A_diag_mean"] = diag
    disp["A_diag_stderr"] = np.zeros_like(diag, dtype=np.float32)
    return disp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and plot prism EM 3c results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataName", type=str, required=True,
                        help="Base name for prismEM file in basePath/prismFit")
    parser.add_argument("--basePath", type=str,
                        default="/pscratch/sd/b/balewski/2026_causalNet_tmp3/",
                        help="Head dir for input/output data")
    parser.add_argument("-p", "--showPlots", type=str, nargs="+", default="a",
                        help="Plot letters: a-l for experiments and simulations; m-r simulations only")
    parser.add_argument("-T", "--time_range_sec", default=None,
                        nargs=2, type=float,
                        help="Time window [t0, t1] in seconds for time-axis plots")
    parser.add_argument("-R", "--time_rebin2", default=20, type=int,
                        help="Burst panels: rebin factor on spike time axis")
    parser.add_argument("--burst_freq_thres", default=5.0, type=float,
                        help="Burst panels: per-neuron rate threshold (Hz)")
    parser.add_argument("--maxNeurons", default=24, type=int,
                        help="Spatial split panels: max source neurons per panel")
    parser.add_argument("-X", "--noXterm", action="store_true",
                        help="Disable X terminal for plotting")
    parser.add_argument("-v", "--verb", type=int, default=1,
                        help="Verbosity level")
    return parser.parse_args()


def resolve_fit_file(base_path, data_name):
    if re.search(r"\.bag\d{3}(?:$|\.)", data_name):
        if data_name.endswith(".prismFDRbag.npz"):
            fname = data_name
        elif data_name.endswith(".prismFDRbag"):
            fname = f"{data_name}.npz"
        elif data_name.endswith(".npz"):
            fname = data_name
        else:
            fname = f"{data_name}.prismFDRbag.npz"
        fit_f = fname if os.path.isabs(fname) else os.path.join(base_path, "prismFDR", fname)
        return fit_f, "prismFDR"

    fit_f = os.path.join(base_path, "prismFit", f"{data_name}.prismEM.npz")
    return fit_f, "prismFit"


def load_fit(base_path, data_name, verb=1):
    fit_f, fit_source = resolve_fit_file(base_path, data_name)
    fitD, fitMD = read_data_npz(fit_f, verb=verb > 1)
    assert isinstance(fitMD, dict), "Expected metadata dict in prismEM file"
    return fitD, fitMD, fit_f, fit_source


def default_plot_time_range_sec(md, duration_sec=60.0):
    trainMD = real_fit_metadata(md)["train"]
    t0_sec = float(trainMD["time_range_sec"][0])
    t1_sec = min(float(trainMD["time_range_sec"][1]), t0_sec + float(duration_sec))
    return [t0_sec, t1_sec]


def load_simu_truth(base_path, fitD, fitMD, md, verb=1):
    if "S_true" in fitD and "C_true" in fitD:
        md["S_true"] = fitD["S_true"]
        md["C_true"] = fitD["C_true"]
        return md

    prov = fitMD["provenance"]
    spikes_path = os.path.join(base_path, "spikesData")

    st_name = prov["state_transition_file"]
    pt_f = os.path.join(spikes_path, f"{st_name}.prismTruth.npz")
    trD, _ = read_data_npz(pt_f, verb=verb > 1)
    md["S_true"] = trD["S_true"]
    md["C_true"] = trD["C_true"]
    return md


def load_simu_static_truth(base_path, fitMD, md, verb=1):
    prov = fitMD["provenance"]
    truth_name = prov["state_model_file"]
    truth_f = os.path.join(base_path, "truthDale", f"{truth_name}.simTruth.npz")
    trueD, trueMD = read_data_npz(truth_f, verb=verb > 1)
    for key in ("A_true", "B_true", "E_true"):
        assert key in trueD, f"{truth_f} must contain {key}"
        md[key] = trueD[key]
    if "dale_conf" in trueMD:
        md["dale_conf"] = trueMD["dale_conf"]
    return md


def source_spike_name(fitMD):
    prov = fitMD["provenance"]
    if fitMD.get("data_type") == "bioExp":
        return prov["experiment_name"]
    return prov["state_transition_file"]


def load_source_spikes(base_path, fitMD, verb=1):
    spike_name = source_spike_name(fitMD)
    spike_f = os.path.join(base_path, "spikesData", f"{spike_name}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spike_f, verb=verb > 1)
    return spikeD, {**spikeMD, "short_name": spike_name}, spike_f


def load_node_metadata(base_path, fitMD, data_name, verb=1):
    prov = fitMD["provenance"]
    if fitMD.get("data_type") == "bioExp":
        node_name = prov["experiment_name"]
        node_f = os.path.join(base_path, "spikesData", f"{node_name}.bioExp.npz")
        nodeD, nodeMD = read_data_npz(node_f, verb=verb > 1)
        nodeMD = {**nodeMD, "short_name": data_name, "node_source_name": node_name}
    else:
        node_name = prov["state_model_file"]
        node_f = os.path.join(base_path, "truthDale", f"{node_name}.simTruth.npz")
        nodeD, nodeMD = read_data_npz(node_f, verb=verb > 1)
        nodeMD = {
            **nodeMD,
            "short_name": data_name,
            "data_type": fitMD.get("data_type", "simPrism"),
            "node_source_name": node_name,
        }

    assert "node_positions" in nodeD, (
        f"{node_f} must contain node_positions; regenerate the source metadata file"
    )
    node_pos = np.asarray(nodeD["node_positions"], dtype=np.float64)
    assert node_pos.ndim == 2 and node_pos.shape[1] == 2, (
        f"node_positions in {node_f} must have shape (N, 2), got {node_pos.shape}"
    )
    return nodeD, nodeMD, node_f


def normalize_plot_letters(show_plots):
    letters = "".join(show_plots)
    unknown = sorted(set(letters) - set(PLOT_FIG_ID))
    assert not unknown, f"Unknown plot letters: {''.join(unknown)}"
    missing = sorted(set(letters) - IMPLEMENTED_PLOTS)
    assert not missing, f"Plots not implemented yet in prism_EM_eval3c.py: {''.join(missing)}"
    return letters


def _binned_acceptance(x, accepted, bins):
    total, _ = np.histogram(x, bins=bins)
    passed, _ = np.histogram(x[accepted], bins=bins)
    prob = np.full(total.shape, np.nan, dtype=np.float64)
    m = total > 0
    prob[m] = passed[m].astype(np.float64) / total[m].astype(np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, prob, total.astype(np.int64), passed.astype(np.int64)


def compute_fdr_acceptance_truth(fitD, md):
    assert "bagsFDR_stageB" in md, "Plot o requires a Stage (b) FDR aggregate"
    for key in ("A_prune", "single_rates"):
        assert key in fitD, f"Plot o requires {key} in fit data"
    for key in ("A_true", "E_true", "dale_conf"):
        assert key in md, f"Plot o requires simulation truth metadata {key}"

    A_true = np.asarray(md["A_true"], dtype=np.float64)
    if A_true.ndim == 3:
        A_true = A_true[0]
    E_true = np.asarray(md["E_true"]) != 0
    if E_true.ndim == 3:
        E_true = E_true[0] != 0
    A_prune = np.asarray(fitD["A_prune"], dtype=np.float64)
    assert A_prune.shape == A_true.shape and A_true.ndim == 2, (
        f"A_prune shape {A_prune.shape} must match 2D A_true shape {A_true.shape}"
    )
    assert E_true.shape == A_true.shape, f"E_true shape {E_true.shape} must match A_true {A_true.shape}"
    n_neur = A_true.shape[0]
    rates = np.asarray(fitD["single_rates"], dtype=np.float64).reshape(-1)
    assert rates.shape[0] == n_neur, "single_rates length must match A_true columns"

    num_exc = int(md["dale_conf"]["num_excite"])
    assert 0 < num_exc < n_neur, "dale_conf.num_excite must split excitatory/inhibitory source columns"

    off_mask = ~np.eye(n_neur, dtype=bool)
    true_edge_mask = off_mask & E_true
    assert np.any(true_edge_mask), "Plot o requires at least one true off-diagonal edge"
    src_idx = np.broadcast_to(np.arange(n_neur, dtype=np.int64)[None, :], A_true.shape)
    x_w = A_true[true_edge_mask]
    accepted = (np.abs(A_prune) > 1e-12)[true_edge_mask]
    src = src_idx[true_edge_mask]
    src_rate = rates[src]
    src_is_exc = src < num_exc

    w_step = 0.01
    w_lo = np.floor(float(np.min(x_w)) / w_step) * w_step
    w_hi = np.ceil(float(np.max(x_w)) / w_step) * w_step
    if np.isclose(w_lo, w_hi):
        w_hi = w_lo + w_step
    w_bins = np.arange(w_lo, w_hi + 1.5 * w_step, w_step, dtype=np.float64)
    w_center, w_prob, w_total, w_pass = _binned_acceptance(x_w, accepted, w_bins)

    r_min = float(np.min(src_rate))
    r_max = float(np.max(src_rate))
    n_rate_bins = 10
    r_pos = src_rate[src_rate > 0.0]
    if r_min > 0.0 and r_pos.size > 0 and r_max / float(np.min(r_pos)) > 20.0:
        r_lo = float(np.min(r_pos))
        r_bins = np.geomspace(r_lo, r_max, n_rate_bins + 1)
        r_scale = "log"
    else:
        if np.isclose(r_min, r_max):
            r_max = r_min + 1.0
        r_bins = np.linspace(r_min, r_max, n_rate_bins + 1)
        r_scale = "linear"

    rate_by_type = {}
    for label, mask in (("exc", src_is_exc), ("inh", ~src_is_exc)):
        c, p, t, a = _binned_acceptance(src_rate[mask], accepted[mask], r_bins)
        rate_by_type[label] = {
            "center": c,
            "prob": p,
            "total": t,
            "passed": a,
        }

    return {
        "weight": {
            "bin_edges": w_bins,
            "center": w_center,
            "prob": w_prob,
            "total": w_total,
            "passed": w_pass,
            "bin_width": w_step,
        },
        "rate": {
            "bin_edges": r_bins,
            "scale": r_scale,
            "exc": rate_by_type["exc"],
            "inh": rate_by_type["inh"],
        },
        "summary": {
            "num_candidates": int(x_w.size),
            "num_accepted": int(np.sum(accepted)),
            "num_exc": num_exc,
            "num_inh": int(n_neur - num_exc),
        },
    }


def main():
    args = parse_args()
    args.showPlots = normalize_plot_letters(args.showPlots)
    fit_f, fit_source = resolve_fit_file(args.basePath, args.dataName)
    args.inpPath = os.path.dirname(fit_f)
    args.outPath = os.path.join(args.basePath, "plots")
    os.makedirs(args.outPath, exist_ok=True)

    print("EM-eval3c args:", vars(args), "\n")

    fitD, fitMD, fit_f, fit_source = load_fit(args.basePath, args.dataName, verb=args.verb)
    if args.verb > 0:
        print(f"Loaded fit ({fit_source}): {fit_f}")
    if args.verb > 1:
        pprint(fitMD)
    plotD = display_fit_data(fitD, fitMD)
    stage_c = is_stage_c_metadata(fitMD)
    primary_A_label = "A_debias" if stage_c else "A_hat"

    is_sim = fitMD.get("data_type") != "bioExp"
    requested_sim_only = sorted(set(args.showPlots) & SIM_ONLY_PLOTS)
    assert is_sim or not requested_sim_only, (
        "Plots m-r require simulation truth; requested "
        f"{''.join(requested_sim_only)} for data_type={fitMD.get('data_type')!r}"
    )
    md = {**fitMD, "short_name": args.dataName}
    if any(c in args.showPlots for c in "nopr"):
        md = load_simu_static_truth(args.basePath, fitMD, md, verb=args.verb)
    if "m" in args.showPlots:
        md = load_simu_truth(args.basePath, fitD, fitMD, md, verb=args.verb)
    if any(c in args.showPlots for c in "cdm") and args.time_range_sec is None:
        args.time_range_sec = default_plot_time_range_sec(md)
        if args.verb > 0:
            print(f"default --time_range_sec from fitted data start: {args.time_range_sec}")

    args.prjName = args.dataName
    plot = Plotter(args)

    if "a" in args.showPlots:
        plot.summary(plotD, md, figId=PLOT_FIG_ID["a"])
    if "b" in args.showPlots:
        plot.A_fitted(plotD, md, plotD["single_rates"], figId=PLOT_FIG_ID["b"])
    if "c" in args.showPlots:
        plot.state_seq_fit(plotD, md, figId=PLOT_FIG_ID["c"], time_range_sec=args.time_range_sec)
    if "d" in args.showPlots:
        spikeD, spikeMD, spike_f = load_source_spikes(args.basePath, fitMD, verb=args.verb)
        if args.verb > 0:
            print(f"Loaded source spikes: {spike_f}")
        rebD = detect_spike_bursts(
            spikeD, spikeMD, args.time_rebin2, args.burst_freq_thres
        )
        plot.spike_bursts(rebD, spikeMD, figId=PLOT_FIG_ID["d"], time_range_sec=args.time_range_sec)
    if "e" in args.showPlots:
        plot.node_outgoing_edge_stats(
            plotD, md, plotD["single_rates"], plotD["neuron_type"],
            figId=PLOT_FIG_ID["e"], est_key="A_hat", est_label=primary_A_label,
        )
    if "f" in args.showPlots:
        nodeD, nodeMD, node_f = load_node_metadata(args.basePath, fitMD, args.dataName, verb=args.verb)
        if args.verb > 0:
            print(f"Loaded node metadata: {node_f}")
        plot.neuron_spatial_edges_split(
            plotD, nodeD, nodeMD, plotD["neuron_type"], plotD["neuron_Sedge"],
            maxNeurons=args.maxNeurons, figId=PLOT_FIG_ID["f"],
        )
    if "g" in args.showPlots:
        plot.fdr_selection_summary(fitD, md, figId=PLOT_FIG_ID["g"])
    if "h" in args.showPlots:
        plot.final_weight_distributions(plotD, md, figId=PLOT_FIG_ID["h"])
    if "i" in args.showPlots:
        plot.offdiag_weight_investigation(plotD, md, figId=PLOT_FIG_ID["i"])

    #....  SIM only plots ....
    if "m" in args.showPlots:
        plot.state_seq_simu(plotD, md, figId=PLOT_FIG_ID["m"], time_range_sec=args.time_range_sec)
    if "n" in args.showPlots:
        plot.edge_stats_and_ABcorr(plotD, md, figId=PLOT_FIG_ID["n"])
    if "o" in args.showPlots:
        plot.matrix_init(fitD, md, figId=PLOT_FIG_ID["o"])
    if "p" in args.showPlots:
        plot.matrix_init(plotD, md, figId=PLOT_FIG_ID["p"], est_key="A_hat", est_label=primary_A_label)
    if "r" in args.showPlots:
        acceptD = compute_fdr_acceptance_truth(fitD, md)
        plot.fdr_acceptance_truth(acceptD, md, figId=PLOT_FIG_ID["r"])

    plot.display_all()


if __name__ == "__main__":
    main()
