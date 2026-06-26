#!/usr/bin/env python3
"""Aggregate EM-FDR-bagging Stage (a) outputs into one eval-compatible fit."""

import argparse
import itertools
import os
import secrets
import time

import numpy as np

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz


BASE_FIT_KEYS = (
    "A_init",
    "A_hat",
    "A_prune",
    "neuron_type",
    "neuron_Sedge",
    "B_init",
    "B_hat",
    "freq_h1d",
    "c_init",
    "c_hat",
    "S_init",
    "S_hat",
    "S_hat_CL",
    "single_rates",
    "e_nll_em",
    "m_loss_epoch",
    "m_nll_epoch",
    "m_l1_epoch",
    "rho_epoch",
    "rho_correction_strength_epoch",
    "nz_edges_epoch",
    "learning_rates",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PRISM-EM FDR bagging Stage (b): aggregate bag files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--basePath", required=True,
                        help="Run directory containing prismFDR/ and prismFit/")
    parser.add_argument("--dataName", required=True,
                        help="Stage (a) bag input stem in prismFDR/")
    parser.add_argument("--outAgrName", default=None,
                        help="Output aggregate fit stem written to prismFit/; default is dataName plus random _hash4")
    parser.add_argument("--num_bags", type=int, required=True,
                        help="Number of bag files to aggregate; e.g. 2 means bag000 and bag001")
    parser.add_argument("--per_bag_quantile", type=float, default=0.99,
                        help="Per-source-column null magnitude quantile for bag selection")
    parser.add_argument("--stab_sel_thresh", type=float, default=0.7,
                        help="Cross-bag stability selection frequency threshold")
    parser.add_argument("--fdr_out_dir", default=None,
                        help="Directory containing Stage (a) bag files")
    parser.add_argument("-v", "--verb", type=int, default=1)
    return parser.parse_args()


def bag_file_name(data_name, bag_idx):
    return f"{data_name}.bag{int(bag_idx):03d}.prismFDRbag.npz"


def bag_indices_from_count(num_bags):
    return list(range(int(num_bags)))


def source_type_prune(A_hat, min_posW, max_negW):
    """Compute source-neuron signs and Dale-style pruned A using columns."""
    A_hat = np.asarray(A_hat, dtype=np.float32)
    n_neuron = A_hat.shape[0]
    A_thr = A_hat.copy()
    np.fill_diagonal(A_thr, 0.0)

    neuron_sedge = A_thr.sum(axis=0)
    neuron_type = np.zeros((n_neuron,), dtype=np.int8)
    neuron_type[neuron_sedge > float(min_posW)] = 1
    neuron_type[neuron_sedge < float(max_negW)] = -1

    A_prune = A_hat.copy()
    diag_A = np.diag(A_hat).copy()
    exc_cols = neuron_type > 0
    inh_cols = neuron_type < 0
    A_prune[:, exc_cols] = np.where(A_prune[:, exc_cols] > 0, A_prune[:, exc_cols], 0.0)
    A_prune[:, inh_cols] = np.where(A_prune[:, inh_cols] < 0, A_prune[:, inh_cols], 0.0)
    np.fill_diagonal(A_prune, diag_A)
    return A_prune.astype(np.float32), neuron_type, neuron_sedge.astype(np.float32)


def selected_weight_thresholds(A_hat, selected_mask):
    A = np.asarray(A_hat, dtype=np.float64)
    mask = np.asarray(selected_mask, dtype=bool)
    pos = A[mask & (A > 0.0)]
    neg = A[mask & (A < 0.0)]
    min_posW = float(np.min(pos)) if pos.size else float("nan")
    max_negW = float(np.max(neg)) if neg.size else float("nan")
    return min_posW, max_negW


def load_bags(args):
    inp_dir = args.fdr_out_dir
    if inp_dir is None:
        inp_dir = os.path.join(args.basePath, "prismFDR")

    bag_data = []
    bag_meta = []
    bag_files = []
    for bag_idx in bag_indices_from_count(args.num_bags):
        inp_f = os.path.join(inp_dir, bag_file_name(args.dataName, bag_idx))
        if not os.path.exists(inp_f):
            raise FileNotFoundError(f"Missing Stage (a) bag file: {inp_f}")
        data_d, meta_d = read_data_npz(inp_f, verb=args.verb > 1)
        if meta_d is None:
            raise ValueError(f"Bag file has no metadata: {inp_f}")
        bag_data.append(data_d)
        bag_meta.append(meta_d)
        bag_files.append(inp_f)
        if args.verb > 0:
            print(f"loaded bag {bag_idx:03d}: {inp_f}")
    return bag_data, bag_meta, bag_files, inp_dir


def validate_bags(bag_data):
    if not bag_data:
        raise ValueError("No bags loaded")
    n0 = np.asarray(bag_data[0]["A_hat"]).shape[0]
    b0 = np.asarray(bag_data[0]["B_hat"])
    if b0.ndim == 1:
        b0 = b0[None, :]
    m0 = b0.shape[0]
    for ib, data_d in enumerate(bag_data):
        a_hat = np.asarray(data_d["A_hat"])
        a_null = np.asarray(data_d["A_null"])
        b_hat = np.asarray(data_d["B_hat"])
        if b_hat.ndim == 1:
            b_hat = b_hat[None, :]
        if a_hat.shape != (n0, n0):
            raise ValueError(f"bag {ib}: A_hat shape {a_hat.shape} != {(n0, n0)}")
        if a_null.ndim != 3 or a_null.shape[1:] != (n0, n0):
            raise ValueError(f"bag {ib}: A_null shape {a_null.shape} incompatible with N={n0}")
        if b_hat.shape != (m0, n0):
            raise ValueError(f"bag {ib}: B_hat shape {b_hat.shape} != {(m0, n0)}")
    return n0, m0


def align_state_rows_to_reference(b_hat_stack):
    """Align B state rows to bag 0 by minimum squared distance."""
    b_hat_stack = np.asarray(b_hat_stack, dtype=np.float64)
    n_bag, n_state, _ = b_hat_stack.shape
    ref = b_hat_stack[0]
    aligned = np.empty_like(b_hat_stack)
    perms = np.zeros((n_bag, n_state), dtype=np.int64)
    aligned[0] = ref
    perms[0] = np.arange(n_state, dtype=np.int64)

    all_perms = None
    if n_state <= 8:
        all_perms = list(itertools.permutations(range(n_state)))

    for ib in range(1, n_bag):
        cur = b_hat_stack[ib]
        if all_perms is not None:
            best_perm = None
            best_score = None
            for perm in all_perms:
                diff = ref - cur[np.asarray(perm)]
                score = float(np.sum(diff * diff))
                if best_score is None or score < best_score:
                    best_score = score
                    best_perm = perm
            perm_arr = np.asarray(best_perm, dtype=np.int64)
        else:
            remaining = set(range(n_state))
            perm = []
            for m in range(n_state):
                best_j = min(
                    remaining,
                    key=lambda j: float(np.sum((ref[m] - cur[j]) ** 2)),
                )
                perm.append(best_j)
                remaining.remove(best_j)
            perm_arr = np.asarray(perm, dtype=np.int64)
        aligned[ib] = cur[perm_arr]
        perms[ib] = perm_arr
    return aligned.astype(np.float32), perms


def sd_or_nan(x, axis):
    x = np.asarray(x, dtype=np.float64)
    if x.shape[axis] < 2:
        return np.full(x.shape[:axis] + x.shape[axis + 1:], np.nan, dtype=np.float64)
    return np.std(x, axis=axis, ddof=1)


def aggregate_edges(a_bag, a_null, per_bag_quantile, stab_sel_thresh):
    """Return Stage (b) matrices and diagnostics."""
    a_bag = np.asarray(a_bag, dtype=np.float64)
    a_null = np.asarray(a_null, dtype=np.float64)
    n_bag, n, n2 = a_bag.shape
    if n != n2:
        raise ValueError("A_hat stack must be square")

    off_mask = ~np.eye(n, dtype=bool)
    tau = np.zeros((n_bag, n), dtype=np.float64)
    null_mean_bag = np.zeros((n_bag, n), dtype=np.float64)
    null_sd_bag = np.zeros((n_bag, n), dtype=np.float64)

    for ib in range(n_bag):
        for j in range(n):
            vals = a_null[ib, :, :, j][:, off_mask[:, j]].reshape(-1)
            tau[ib, j] = float(np.quantile(np.abs(vals), per_bag_quantile))
            null_mean_bag[ib, j] = float(np.mean(vals))
            null_sd_bag[ib, j] = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan

    sel = (np.abs(a_bag) > tau[:, None, :]) & off_mask[None, :, :]
    sel_count = sel.sum(axis=0).astype(np.int64)
    sel_freq = sel_count.astype(np.float64) / float(n_bag)
    final_sel = (sel_freq >= float(stab_sel_thresh)) & off_mask

    a_mean_all = np.mean(a_bag, axis=0)
    a_sd_all = sd_or_nan(a_bag, axis=0)

    a_sum_sel = np.sum(np.where(sel, a_bag, 0.0), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        a_mean_sel = a_sum_sel / sel_count
    a_mean_sel[sel_count == 0] = np.nan

    a_sd_sel = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            vals = a_bag[sel[:, i, j], i, j]
            if vals.size > 1:
                a_sd_sel[i, j] = float(np.std(vals, ddof=1))
            elif vals.size == 1:
                a_sd_sel[i, j] = 0.0

    src_null_mean = np.zeros((n,), dtype=np.float64)
    src_null_sd = np.zeros((n,), dtype=np.float64)
    for j in range(n):
        vals = a_null[:, :, :, j][:, :, off_mask[:, j]].reshape(-1)
        src_null_mean[j] = float(np.mean(vals))
        src_null_sd[j] = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan

    a_hat_final = np.zeros((n, n), dtype=np.float64)
    a_hat_final[final_sel] = a_mean_sel[final_sel]
    diag = np.diag(a_mean_all).copy()
    np.fill_diagonal(a_hat_final, diag)

    z_null = np.full((n, n), np.nan, dtype=np.float64)
    denom = src_null_sd[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        z_null = (a_mean_sel - src_null_mean[None, :]) / denom
    z_null[~off_mask] = np.nan

    selected_edges_per_bag_src = sel.sum(axis=1).astype(np.int64)
    selected_edges_per_bag = sel.sum(axis=(1, 2)).astype(np.int64)
    p_cand = int(n * (n - 1))
    q_lambda = float(np.mean(selected_edges_per_bag))
    if float(stab_sel_thresh) > 0.5:
        stability_false_edge_bound = q_lambda * q_lambda / (
            float(p_cand) * (2.0 * float(stab_sel_thresh) - 1.0)
        )
    else:
        stability_false_edge_bound = np.inf

    return {
        "A_hat_final": a_hat_final,
        "selected_mask": final_sel,
        "selected_in_bag": sel,
        "selection_count": sel_count,
        "selection_frequency": sel_freq,
        "A_mean_selected": a_mean_sel,
        "A_sd_selected": a_sd_sel,
        "A_mean_all": a_mean_all,
        "A_sd_all": a_sd_all,
        "src_null_tau_bag": tau,
        "src_null_tau_mean": np.mean(tau, axis=0),
        "src_null_mean": src_null_mean,
        "src_null_sd": src_null_sd,
        "src_null_mean_bag": null_mean_bag,
        "src_null_sd_bag": null_sd_bag,
        "z_null": z_null,
        "selected_edges_per_bag_src": selected_edges_per_bag_src,
        "selected_edges_per_bag": selected_edges_per_bag,
        "q_lambda": q_lambda,
        "p_cand": p_cand,
        "stability_false_edge_bound": float(stability_false_edge_bound),
    }


def edge_table(final_sel, agg):
    edge_i, edge_j = np.where(final_sel)
    return {
        "edge_i": edge_i.astype(np.int64),
        "edge_j": edge_j.astype(np.int64),
        "edge_sel_freq": agg["selection_frequency"][edge_i, edge_j].astype(np.float32),
        "edge_A_mean": agg["A_mean_selected"][edge_i, edge_j].astype(np.float32),
        "edge_A_sd_boot": agg["A_sd_selected"][edge_i, edge_j].astype(np.float32),
        "edge_z_null": agg["z_null"][edge_i, edge_j].astype(np.float32),
        "edge_A_mean_all": agg["A_mean_all"][edge_i, edge_j].astype(np.float32),
        "edge_A_sd_all": agg["A_sd_all"][edge_i, edge_j].astype(np.float32),
        "edge_src_null_mean": agg["src_null_mean"][edge_j].astype(np.float32),
        "edge_src_null_sd": agg["src_null_sd"][edge_j].astype(np.float32),
        "edge_src_tau_mean": agg["src_null_tau_mean"][edge_j].astype(np.float32),
    }


def real_fit_metadata(md):
    if "bagsFDR_stageA" not in md:
        raise KeyError("metadata missing 'bagsFDR_stageA' — was this file produced by prism_FDR_Bags_train3c.py?")
    if "real_fit" not in md["bagsFDR_stageA"]:
        raise KeyError("bagsFDR_stageA missing 'real_fit' — re-run prism_FDR_Bags_train3c.py to regenerate bag files")
    return md["bagsFDR_stageA"]["real_fit"]


def eval_loss_time(fitD, md, spikes):
    """Precompute per-time fit-loss diagnostics for plotting."""
    trainMD = real_fit_metadata(md)["train"]
    t0_bin, t1_bin = [int(x) for x in trainMD["time_range_bins"]]
    dt = float(trainMD["time_step_sec"])
    eta_clip = float(trainMD["eta_clip"])
    lambda2 = float(trainMD["lambda2"])

    spikes_sub = np.asarray(spikes[t0_bin:t1_bin + 1], dtype=np.float64)
    yp = spikes_sub[:-1]
    yc = spikes_sub[1:]

    a_hat = np.asarray(fitD["A_hat"], dtype=np.float64)
    b_hat = np.asarray(fitD["B_hat"], dtype=np.float64)
    if b_hat.ndim == 1:
        b_hat = b_hat[None, :]
    c_hat = np.asarray(fitD["c_hat"], dtype=np.float64)
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

    return {
        "loss_nll_time": nll_t.astype(np.float32),
        "loss_l2_time": l2_t.astype(np.float32),
    }


def eval_true_state_recovery(fitD, md):
    """Precompute state-recovery table for plotting."""
    trainMD = real_fit_metadata(md)["train"]
    t0_bin, t1_bin = [int(x) for x in trainMD["time_range_bins"]]
    s_true = np.asarray(md["S_true"], dtype=np.int64)[t0_bin:t1_bin + 1]
    s_hat = np.asarray(fitD["S_hat"], dtype=np.int64)
    s_hat_cl = np.asarray(fitD["S_hat_CL"], dtype=np.float64)
    n_state = int(trainMD["num_states"])

    assert s_true.shape[0] == s_hat.shape[0] == s_hat_cl.shape[0], \
        "S_true/S_hat/S_hat_CL length mismatch on training window"

    state_acc_cl = []
    bins_hat = np.bincount(s_hat, minlength=n_state).astype(np.int64)
    enter_hat = np.zeros(n_state, dtype=np.int64)
    if s_hat.shape[0] > 1:
        enter_idx = np.where(s_hat[1:] != s_hat[:-1])[0] + 1
        if enter_idx.size > 0:
            enter_hat = np.bincount(s_hat[enter_idx], minlength=n_state).astype(np.int64)

    for m in range(n_state):
        mask = (s_hat == m)
        if int(mask.sum()) > 0:
            acc_m = float((s_true[mask] == m).mean())
            cl_m = float(s_hat_cl[mask].mean())
        else:
            acc_m = float("nan")
            cl_m = float("nan")
        state_acc_cl.append([acc_m, cl_m, int(bins_hat[m]), int(enter_hat[m])])

    return {
        "avg_acc": float((s_true == s_hat).mean()),
        "state_acc_cl": state_acc_cl,
    }


def source_spike_file(base_path, out_md):
    prov = out_md["provenance"]
    if out_md.get("data_type") == "bioExp":
        data_name = prov["experiment_name"]
    else:
        data_name = prov["state_transition_file"]
    return os.path.join(base_path, "spikesData", f"{data_name}.spikes.npz")


def add_fit_eval_metadata(out_d, out_md, base_path, verb=1):
    """Attach display metrics to the aggregate output metadata."""
    spike_f = source_spike_file(base_path, out_md)
    spike_d, _ = read_data_npz(spike_f, verb=verb > 1)

    eval_f = eval_loss_time(out_d, out_md, spike_d["spikes"])
    out_d["loss_nll_time"] = eval_f.pop("loss_nll_time")
    out_d["loss_l2_time"] = eval_f.pop("loss_l2_time")
    out_md["eval_f"] = {
        **eval_f,
        "loss_nll_time_key": "loss_nll_time",
        "loss_l2_time_key": "loss_l2_time",
    }

    if out_md.get("data_type") == "bioExp":
        return out_md

    prov = out_md["provenance"]
    st_name = prov["state_transition_file"]
    truth_f = os.path.join(base_path, "spikesData", f"{st_name}.prismTruth.npz")

    truth_d, _ = read_data_npz(truth_f, verb=verb > 1)

    md_eval = dict(out_md)
    md_eval["S_true"] = truth_d["S_true"]
    md_eval["C_true"] = truth_d["C_true"]

    srec = real_fit_metadata(out_md)["states_recovery_eval"]
    reco = eval_true_state_recovery(out_d, md_eval)
    srec.update(reco)
    out_md["eval_f"]["acc"] = reco["avg_acc"]
    return out_md


def main():
    args = parse_args()
    if args.outAgrName is None:
        args.outAgrName = f"{args.dataName}_{secrets.token_hex(2)}"
    if args.num_bags < 1:
        raise ValueError("--num_bags must be at least 1")
    if not (0.0 < float(args.per_bag_quantile) < 1.0):
        raise ValueError("--per_bag_quantile must be between 0 and 1")
    if not (0.0 < float(args.stab_sel_thresh) <= 1.0):
        raise ValueError("--stab_sel_thresh must be in (0, 1]")
    if args.verb > 0:
        print("\nFDR-bag aggregate args:", vars(args), "\n")

    t0 = time.perf_counter()
    bag_data, bag_meta, bag_files, inp_dir = load_bags(args)
    n_neuron, n_state = validate_bags(bag_data)

    a_bag = np.stack([np.asarray(d["A_hat"], dtype=np.float32) for d in bag_data], axis=0)
    a_null = np.stack([np.asarray(d["A_null"], dtype=np.float32) for d in bag_data], axis=0)
    b_bag = np.stack([
        np.asarray(d["B_hat"], dtype=np.float32)
        if np.asarray(d["B_hat"]).ndim == 2
        else np.asarray(d["B_hat"], dtype=np.float32)[None, :]
        for d in bag_data
    ], axis=0)
    b_aligned = b_bag
    state_perms = np.tile(np.arange(n_state, dtype=np.int64), (int(args.num_bags), 1))

    agg = aggregate_edges(
        a_bag, a_null,
        per_bag_quantile=float(args.per_bag_quantile),
        stab_sel_thresh=float(args.stab_sel_thresh),
    )

    a_hat = agg["A_hat_final"].astype(np.float32)
    min_posW, max_negW = selected_weight_thresholds(a_hat, agg["selected_mask"])
    a_prune, neuron_type, neuron_sedge = source_type_prune(a_hat, min_posW, max_negW)
    b_hat = np.mean(b_aligned, axis=0).astype(np.float32)

    ref_d = bag_data[0]
    ref_md = bag_meta[0]
    out_d = {}
    for key in BASE_FIT_KEYS:
        if key in ref_d:
            out_d[key] = ref_d[key]

    out_d["A_hat"] = a_hat
    out_d["A_prune"] = a_prune
    out_d["neuron_type"] = neuron_type
    out_d["neuron_Sedge"] = neuron_sedge
    out_d["B_hat"] = b_hat

    out_d.update({
        "A_bag": a_bag.astype(np.float32),
        "B_bag": b_aligned.astype(np.float32),
        "state_permutation_to_ref": state_perms.astype(np.int64),
        "selected_mask": agg["selected_mask"].astype(np.bool_),
        "selected_in_bag": agg["selected_in_bag"].astype(np.bool_),
        "selection_count": agg["selection_count"].astype(np.int64),
        "selection_frequency": agg["selection_frequency"].astype(np.float32),
        "A_mean_selected": agg["A_mean_selected"].astype(np.float32),
        "A_sd_selected": agg["A_sd_selected"].astype(np.float32),
        "A_mean_all": agg["A_mean_all"].astype(np.float32),
        "A_sd_all": agg["A_sd_all"].astype(np.float32),
        "A_diag_mean": np.diag(agg["A_mean_all"]).astype(np.float32),
        "A_diag_std": np.diag(agg["A_sd_all"]).astype(np.float32),
        "A_diag_stderr": (np.diag(agg["A_sd_all"]) / np.sqrt(float(args.num_bags))).astype(np.float32),
        "src_null_tau_bag": agg["src_null_tau_bag"].astype(np.float32),
        "src_null_tau_mean": agg["src_null_tau_mean"].astype(np.float32),
        "src_null_mean": agg["src_null_mean"].astype(np.float32),
        "src_null_sd": agg["src_null_sd"].astype(np.float32),
        "src_null_mean_bag": agg["src_null_mean_bag"].astype(np.float32),
        "src_null_sd_bag": agg["src_null_sd_bag"].astype(np.float32),
        "z_null": agg["z_null"].astype(np.float32),
        "selected_edges_per_bag_src": agg["selected_edges_per_bag_src"].astype(np.int64),
        "selected_edges_per_bag": agg["selected_edges_per_bag"].astype(np.int64),
    })
    out_d.update(edge_table(agg["selected_mask"], agg))

    out_name = args.outAgrName
    out_dir = os.path.join(args.basePath, "prismFit")
    os.makedirs(out_dir, exist_ok=True)
    out_f = os.path.join(out_dir, f"{out_name}.prismEM.npz")

    out_md = dict(ref_md)
    out_md["fit_type"] = "prismEM_FDRbags_stageB"
    out_md["bagsFDR_stageB"] = {
        "program": "prism_EM_FDR_Bags_aggregate3c.py",
        "dataName": args.dataName,
        "input_dataName": args.dataName,
        "outAgrName": out_name,
        "output_name": out_name,
        "num_bags": int(args.num_bags),
        "expected_bag_indices": bag_indices_from_count(args.num_bags),
        "per_bag_quantile": float(args.per_bag_quantile),
        "stab_sel_thresh": float(args.stab_sel_thresh),
        "fdr_out_dir": inp_dir,
        "input_files": bag_files,
        "reference_state_source": "reference_EM_fit_locked_in_stageA",
        "reference_time_fields": [
            "A_init", "B_init", "freq_h1d", "c_init", "c_hat",
            "S_init", "S_hat", "S_hat_CL", "EM histories",
        ],
        "null_threshold_axis": "source_column",
        "A_hat_meaning": "selection_conditional_bagged_mean_for_stable_edges_zero_elsewhere",
        "A_hat_diagonal": "all_bag_mean_diagonal",
        "B_hat_meaning": "mean_across_bags_in_reference_state_order",
        "state_permutation_to_ref": state_perms.tolist(),
        "num_neurons": int(n_neuron),
        "num_states": int(n_state),
        "num_final_edges": int(np.sum(agg["selected_mask"])),
        "min_posW": min_posW,
        "max_negW": max_negW,
        "q_lambda_mean_edges_selected_per_bag": float(agg["q_lambda"]),
        "p_cand": int(agg["p_cand"]),
        "stability_false_edge_bound": float(agg["stability_false_edge_bound"]),
        "elapsed_sec": float(time.perf_counter() - t0),
    }

    prov = dict(out_md.get("provenance", {}))
    prov["EMtrain_file"] = out_name
    prov["bagsFDR_stageB_file"] = out_name
    prov["bagsFDR_stageB_dataName"] = args.dataName
    prov["bagsFDR_stageB_outAgrName"] = out_name
    out_md["provenance"] = prov
    out_md = add_fit_eval_metadata(out_d, out_md, args.basePath, verb=args.verb)

    if args.verb > 0:
        print(
            f"Stage (b): bags={args.num_bags} N={n_neuron} "
            f"q={args.per_bag_quantile:g} stab_sel_thresh={args.stab_sel_thresh:g}"
        )
        print(
            f"  selected final edges: {int(np.sum(agg['selected_mask']))} / "
            f"{int(agg['p_cand'])}"
        )
        print(
            f"  q_lambda={agg['q_lambda']:.3f} "
            f"stability_false_edge_bound={agg['stability_false_edge_bound']:.3f}"
        )
    write_data_npz(out_d, out_f, metaD=out_md, verb=args.verb > 1)
    if args.verb > 0:
        print(f"\nSaved Stage (b) aggregate: {out_f}")
        print(
            f"  ./prism_EM_eval3c.py --basePath $basePath "
            f"--dataName {out_name} -p  e f  h i   g"
        )


if __name__ == "__main__":
    main()
