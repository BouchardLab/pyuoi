#!/usr/bin/env python3
"""
Strict evaluation and plotting for fit5_BSSM_STD_blocks.py outputs.

The evaluator intentionally requires the current BSSM-STD NPZ schema.  Missing
records are errors, not silently inferred defaults.
"""

import argparse
import os
from pprint import pprint

import numpy as np

from PlotterBSSM_fit import Plotter
from toolbox.Util_NumpyIO import read_data_npz


FIT_REQUIRED = [
    "W_fit", "B_fit", "W_init", "B_init",
    "init_p_mean", "init_p_plus", "init_p_minus", "W_init_abs_mean",
    "initW_random", "W_init_rebin_steps", "W_init_rebin_bins",
    "outer_idx", "U_outer", "tau_rec_outer",
    "nll_after_blockA", "nll_after_std_grid", "nll_outer",
    "spectral_radius_outer", "nz_weight_outer", "blockB_executed",
    "blockA_outer", "blockA_epoch", "blockA_bce_loss", "blockA_l1_loss", "blockA_loss", "blockA_lr",
    "joint_search_outer", "joint_search_U", "joint_search_tau_rec", "joint_search_nll",
    "joint_search_refine", "joint_search_eval", "joint_search_eval_total",
    "kappa_fit", "alpha_fit", "burn_bins", "time_range_bins", "requested_time_range_bins",
    "fit_time_range_bins", "init_time_range_bins", "time_step_sec",
    "num_neurons", "num_input_bins", "num_eval_bins", "num_init_bins",
    "U_init", "tau_rec_init", "u_bounds", "tau_bounds",
    "u_grid_points", "tau_grid_points", "grid_refine", "grid_shrink",
    "delay_epoch_4_blockB", "eta_clip", "lr_w", "lr_end_factor",
    "lambda_l1", "rho_max", "weight_threshold", "h_chunk_steps",
]

SPIKE_REQUIRED = ["spikes", "single_rates", "single_rates_var", "single_fano_fact"]


def abort(msg):
    raise SystemExit("ERROR: %s" % msg)


def require_records(data, keys, label):
    missing = [key for key in keys if key not in data]
    if missing:
        abort("%s missing required records: %s" % (label, ", ".join(missing)))


def scalar(arr, dtype=float):
    val = np.asarray(arr).reshape(-1)[0]
    return dtype(val)


def stable_sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def std_h_forward_numpy(spikes, U, tau_rec, dt, alpha, mem_lag_steps, burn_bins):
    spikes = np.asarray(spikes, dtype=np.float64)
    assert spikes.ndim == 2, "spikes window must be 2D"
    assert 0.0 < float(U) < 0.999, "U must satisfy 0 < U < 0.999"
    assert float(tau_rec) > 0.0, "tau_rec must be positive"
    assert int(mem_lag_steps) >= 1, "mem_lag_steps must be positive"
    assert 0 <= int(burn_bins) < spikes.shape[0], "bad burn_bins"

    T, N = spikes.shape
    M = int(mem_lag_steps)
    H = np.empty((T - int(burn_bins), N), dtype=np.float32)
    x = np.ones((N,), dtype=np.float64)
    h = np.zeros((N,), dtype=np.float64)
    u_hist = np.zeros((M, N), dtype=np.float64)

    rec = float(np.exp(-float(dt) / float(tau_rec)))
    gain = 1.0 - float(alpha)
    tail_gain = gain * (float(alpha) ** M)

    for t in range(T):
        hist_idx = t % M
        old_u = u_hist[hist_idx].copy()
        if t >= int(burn_bins):
            H[t - int(burn_bins)] = h.astype(np.float32)

        s_t = spikes[t]
        u_t = float(U) * x * s_t
        x_depleted = x * (1.0 - float(U) * s_t)
        x = 1.0 - (1.0 - x_depleted) * rec
        x = np.clip(x, 0.0, 1.0)

        h = float(alpha) * h + gain * u_t - tail_gain * old_u
        h = np.maximum(h, 0.0)
        u_hist[hist_idx] = u_t

    return H


def validate_fit_shapes(fitD):
    W = np.asarray(fitD["W_fit"])
    B = np.asarray(fitD["B_fit"])
    W0 = np.asarray(fitD["W_init"])
    B0 = np.asarray(fitD["B_init"])
    assert W.ndim == 2 and W.shape[0] == W.shape[1], "W_fit must be square"
    assert W0.shape == W.shape, "W_init/W_fit shape mismatch"
    assert B.shape == (W.shape[0],), "B_fit shape mismatch"
    assert B0.shape == B.shape, "B_init/B_fit shape mismatch"
    assert int(scalar(fitD["num_neurons"], int)) == W.shape[0], "num_neurons disagrees with W shape"
    assert np.allclose(np.diag(W0), 0.0), "Eq.17 requires zero diagonal in W_init"
    for key in ["init_p_mean", "init_p_plus", "init_p_minus"]:
        assert np.asarray(fitD[key]).shape == B.shape, "%s shape mismatch" % key

    n_outer = fitD["outer_idx"].shape[0]
    for key in ["U_outer", "tau_rec_outer", "nll_after_blockA", "nll_after_std_grid",
                "nll_outer", "spectral_radius_outer", "nz_weight_outer", "blockB_executed"]:
        assert fitD[key].shape[0] == n_outer, "%s length mismatch" % key

    n_block = fitD["blockA_outer"].shape[0]
    for key in ["blockA_epoch", "blockA_bce_loss", "blockA_l1_loss", "blockA_loss", "blockA_lr"]:
        assert fitD[key].shape[0] == n_block, "%s length mismatch" % key

    n_search = fitD["joint_search_outer"].shape[0]
    for key in ["joint_search_U", "joint_search_tau_rec", "joint_search_nll",
                "joint_search_refine", "joint_search_eval", "joint_search_eval_total"]:
        assert fitD[key].shape[0] == n_search, "%s length mismatch" % key


def build_diag_base(fitD, args, fit_file, spikes_file):
    validate_fit_shapes(fitD)
    raw_bins = np.asarray(fitD["time_range_bins"], dtype=np.int64)
    fit_bins = np.asarray(fitD["fit_time_range_bins"], dtype=np.int64)
    init_bins = np.asarray(fitD["init_time_range_bins"], dtype=np.int64)
    assert raw_bins.shape == fit_bins.shape == init_bins.shape == (2,), "time-bin records must be length 2"
    assert raw_bins[0] <= fit_bins[0] <= fit_bins[1] <= raw_bins[1], "fit bins must lie inside raw bins"
    assert fit_bins[0] <= init_bins[0] <= init_bins[1] <= fit_bins[1], "init bins must lie inside fit bins"
    
    return {
        "short_name": args.dataName,
        "data_name": args.dataName,
        "fit_file": fit_file,
        "spikes_file": spikes_file,
    }


def compute_observation_diagnostics(fitD, spikeD, obs_batch):
    spikes = np.asarray(spikeD["spikes"])
    assert spikes.ndim == 2, "spikes record must be 2D"
    assert np.max(spikes) <= 1, "spikes must be binary"

    raw0, raw1 = np.asarray(fitD["time_range_bins"], dtype=np.int64)
    assert 0 <= raw0 <= raw1 < spikes.shape[0], "fit raw bins exceed spikes record"
    spikes_w = spikes[raw0:raw1 + 1]

    burn_bins = scalar(fitD["burn_bins"], int)
    dt = scalar(fitD["time_step_sec"], float)
    alpha = scalar(fitD["alpha_fit"], float)
    M = int(np.asarray(fitD["kappa_fit"]).size)
    U = float(np.asarray(fitD["U_outer"], dtype=np.float64)[-1])
    tau_rec = float(np.asarray(fitD["tau_rec_outer"], dtype=np.float64)[-1])
    W = np.asarray(fitD["W_fit"], dtype=np.float64)
    B = np.asarray(fitD["B_fit"], dtype=np.float64)
    eta_clip = scalar(fitD["eta_clip"], float)
    n_eval = scalar(fitD["num_eval_bins"], int)

    H = std_h_forward_numpy(spikes_w, U, tau_rec, dt, alpha, M, burn_bins)
    Y = np.asarray(spikes_w[burn_bins:], dtype=np.float64)
    assert H.shape == Y.shape, "H/Y shape mismatch"
    assert H.shape[0] == n_eval, "computed H length disagrees with num_eval_bins"
    assert H.shape[1] == W.shape[0], "H/W neuron count mismatch"

    N = W.shape[0]
    sum_nll_neuron = np.zeros((N,), dtype=np.float64)
    sum_prob_neuron = np.zeros((N,), dtype=np.float64)
    total_nll = 0.0
    total_count = 0

    p_edges = np.linspace(0.0, 1.0, 41)
    p_hist_count = np.zeros((p_edges.size - 1,), dtype=np.int64)
    calib_sum_p = np.zeros_like(p_hist_count, dtype=np.float64)
    calib_sum_y = np.zeros_like(p_hist_count, dtype=np.float64)

    batch = int(obs_batch)
    assert batch > 0, "--obs_batch must be positive"
    for i0 in range(0, H.shape[0], batch):
        i1 = min(H.shape[0], i0 + batch)
        eta = np.asarray(H[i0:i1], dtype=np.float64) @ W.T + B
        eta = np.clip(eta, -eta_clip, eta_clip)
        p = stable_sigmoid(eta)
        y = Y[i0:i1]
        p_safe = np.clip(p, 1e-12, 1.0 - 1e-12)
        nll = -(y * np.log(p_safe) + (1.0 - y) * np.log1p(-p_safe))
        sum_nll_neuron += np.sum(nll, axis=0)
        sum_prob_neuron += np.sum(p, axis=0)
        total_nll += float(np.sum(nll))
        total_count += int(nll.size)

        flat_p = p.reshape(-1)
        flat_y = y.reshape(-1)
        bins = np.searchsorted(p_edges, flat_p, side="right") - 1
        bins = np.clip(bins, 0, p_hist_count.size - 1)
        p_hist_count += np.bincount(bins, minlength=p_hist_count.size)
        calib_sum_p += np.bincount(bins, weights=flat_p, minlength=p_hist_count.size)
        calib_sum_y += np.bincount(bins, weights=flat_y, minlength=p_hist_count.size)

    obs_rate_hz = np.mean(Y, axis=0) / dt
    model_rate_hz = sum_prob_neuron / float(Y.shape[0]) / dt
    nll_neuron = sum_nll_neuron / float(Y.shape[0])
    valid = p_hist_count > 0
    calib_p = np.zeros_like(calib_sum_p)
    calib_obs = np.zeros_like(calib_sum_y)
    calib_p[valid] = calib_sum_p[valid] / p_hist_count[valid]
    calib_obs[valid] = calib_sum_y[valid] / p_hist_count[valid]

    single_rates = np.asarray(spikeD["single_rates"], dtype=np.float64)
    assert single_rates.shape == obs_rate_hz.shape, "single_rates shape mismatch"

    return {
        "obs_rate_hz": obs_rate_hz,
        "model_rate_hz": model_rate_hz,
        "nll_neuron": nll_neuron,
        "calib_p": calib_p,
        "calib_obs": calib_obs,
        "calib_count": p_hist_count,
        "p_hist_edges": p_edges,
        "p_hist_count": p_hist_count,
        "single_rates_hz": single_rates,
        "nll_eval": total_nll / float(total_count),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot BSSM-STD fit results")
    prs=parser.add_argument
    prs("--dataName", required=True, help="Spike data base name, e.g. daleN100_861b17")
    prs("--basePath", default="/pscratch/sd/b/balewski/2026_causalNet_tmp",
                        help="Root containing truthDale/ and plots/")
    prs("-p", "--showPlots", nargs="+", default=["a", "b", "c", "d"],
                        help="Plots: a=two-block summary, b=B/W initialization vs fit, c=Block B grid, d=Bernoulli observation diagnostics")
    prs("--obs_batch", type=int, default=65536,
                        help="Time-bin batch size for Bernoulli observation diagnostics")
    prs("-X", "--noXterm", action="store_true", help="Disable X display and save PNG only")
    prs("-v", "--verb", type=int, default=1)
    args = parser.parse_args()
    if args.verb > 0:
        print("BSSM-fit eval args:");    pprint(vars(args))

    inpPath = os.path.join(args.basePath, "fitBssmStd")
    args.outPath = os.path.join(args.basePath, "plots")

    # ── load BSSM-STD fit ────────────────────────────────────────────
    fitFF = os.path.join(inpPath, f"{args.dataName}.fitBSSMSTD.npz")
    fitD, fitMD = read_data_npz(fitFF)
    require_records(fitD, FIT_REQUIRED, "fit NPZ")

    if args.verb > 1:  pprint(fitMD)
 
    #--- load spikes ----
    if "provenance" not in fitMD or "spikesData_file" not in fitMD["provenance"]:
        abort("fit metadata missing provenance.spikesData_file")
    prov = fitMD["provenance"]
    spikesF=prov['spikesData_file']
    spikesFF = os.path.join(args.basePath, "truthDale", f"{spikesF}.spikes.npz")  
    spikeD, _ = read_data_npz(spikesFF, verb=args.verb > 0)
    require_records(spikeD, SPIKE_REQUIRED, "spikes NPZ")

    args.showPlots =''.join(args.showPlots)
    
     # ── plot ───────────────────────────────
    args.prjName = args.dataName
    plot = Plotter(args)
    diag = build_diag_base(fitD, args, fitFF, spikesFF)

    if "d" in args.showPlots:
        obsD = compute_observation_diagnostics(fitD, spikeD, args.obs_batch)
        diag.update(obsD)
        if args.verb > 0:
            print("  recomputed Bernoulli observation NLL=%.9f" % diag["nll_eval"])

    plot = Plotter(args)
    if "a" in args.showPlots:
        plot.two_block_summary(fitD, diag, figId=1)
    if "b" in args.showPlots:
        plot.parameter_init_vs_fit(fitD, diag, figId=2)
    if "c" in args.showPlots:
        plot.blockB_joint_grid(fitD, diag, figId=3)
    if "d" in args.showPlots:
        plot.bernoulli_observation_diagnostics(fitD, diag, figId=4)
    plot.display_all()


if __name__ == "__main__":
    main()
