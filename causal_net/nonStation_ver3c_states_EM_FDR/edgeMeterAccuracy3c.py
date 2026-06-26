#!/usr/bin/env python3
"""Absolute graph-recovery metrics for final prism EM-FDR aggregate fits."""

import argparse
import os
import secrets
import time

import numpy as np

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz


LABEL_ORDER = np.asarray([-1, 0, 1], dtype=np.int8)
SOURCE_RECO_ORDER = np.asarray([1, -1, 0], dtype=np.int8)
SOURCE_RECO_NAMES = np.asarray(["exc", "inh", "und"], dtype=object)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute absolute graph-recovery metrics for synthetic prism EM-FDR aggregate fits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--basePath", required=True,
                        help="Run directory containing prismFit/, truthDale/, edgeReco/, plots/")
    parser.add_argument("--fitNameTrunk", required=True,
                        help="Common prefix of aggregate fit files")
    parser.add_argument("--fitTags", nargs="+", required=True,
                        help="Fit suffixes, e.g. 'em103c_fdr103c_agr103c emfaec_fdrfaec_agrfaec'")
    parser.add_argument("--outName", default=argparse.SUPPRESS,
                        help="Output metric stem in edgeReco/; default is fitNameTrunk_ermHASH4")
    parser.add_argument("--epsilon", type=float, default=1e-4,
                        help="Presence tolerance for A_prune")
    parser.add_argument("-p", "--showPlots", nargs="+", default="abc",
                        help="Plot letters/groups, e.g. -p a b")
    parser.add_argument("-X", "--noXterm", action="store_true",
                        help="Disable X terminal for plotting")
    parser.add_argument("-v", "--verb", type=int, default=1,
                        help="Verbosity level")
    args = parser.parse_args()
    if not hasattr(args, "outName"):
        args.outName = f"{args.fitNameTrunk}_erm{secrets.token_hex(2)}"
    return args


def flatten_words(items):
    words = []
    for item in items:
        words.extend(str(item).split())
    return words


def normalize_plot_letters(items):
    letters = []
    for token in flatten_words(items):
        letters.extend(list(token))
    return letters


def full_fit_name(trunk, tag):
    tag = str(tag)
    if tag.startswith(trunk):
        return tag
    return f"{trunk}_{tag}"


def fit_file(base_path, fit_name):
    return os.path.join(base_path, "prismFit", f"{fit_name}.prismEM.npz")


def truth_file(base_path, truth_name):
    return os.path.join(base_path, "truthDale", f"{truth_name}.simTruth.npz")


def load_fit(base_path, fit_name, verb=1):
    inp_f = fit_file(base_path, fit_name)
    fit_d, fit_md = read_data_npz(inp_f, verb=verb > 1)
    if not isinstance(fit_md, dict):
        raise ValueError(f"Fit file has no metadata: {inp_f}")
    if "A_prune" not in fit_d:
        raise KeyError(f"{inp_f} must contain A_prune")
    if "neuron_type" not in fit_d:
        raise KeyError(f"{inp_f} must contain neuron_type")
    if "provenance" not in fit_md or "state_model_file" not in fit_md["provenance"]:
        raise KeyError(f"{inp_f} metadata must contain provenance.state_model_file")
    return fit_d, fit_md, inp_f


def load_truth_once(base_path, truth_name, verb=1):
    inp_f = truth_file(base_path, truth_name)
    truth_d, truth_md = read_data_npz(inp_f, verb=verb > 1)
    for key in ("A_true", "E_true", "node_is_inhibitory"):
        if key not in truth_d:
            raise KeyError(f"{inp_f} must contain {key}; regenerate synthetic truth")
    return truth_d, truth_md, inp_f


def as_static_2d(arr, name):
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2D matrix, got {arr.shape}")
    return arr


def safe_div(num, den):
    den = float(den)
    if den == 0.0:
        return float("nan")
    return float(num) / den


def safe_f1(precision, recall):
    if not np.isfinite(precision) or not np.isfinite(recall):
        return float("nan")
    den = precision + recall
    if den == 0.0:
        return float("nan")
    return float(2.0 * precision * recall / den)


def matthews_corrcoef(tp, fp, fn, tn):
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den <= 0:
        return float("nan")
    return float((tp * tn - fp * fn) / np.sqrt(den))


def graph_labels(A_true, E_true, A_prune, epsilon):
    A_true = as_static_2d(A_true, "A_true").astype(np.float64)
    E_true = as_static_2d(E_true, "E_true") != 0
    A_prune = as_static_2d(A_prune, "A_prune").astype(np.float64)
    if A_prune.shape != A_true.shape or E_true.shape != A_true.shape:
        raise ValueError(
            f"Shape mismatch: A_true={A_true.shape} E_true={E_true.shape} A_prune={A_prune.shape}"
        )

    n = A_true.shape[0]
    off_mask = ~np.eye(n, dtype=bool)
    true_edge = E_true & off_mask
    true_edge_sign = A_true[true_edge]
    if np.any(true_edge_sign == 0.0):
        raise ValueError("Every off-diagonal E_true edge must have nonzero A_true sign")

    true_lab_2d = np.zeros((n, n), dtype=np.int8)
    true_lab_2d[true_edge & (A_true > 0.0)] = 1
    true_lab_2d[true_edge & (A_true < 0.0)] = -1

    est_lab_2d = np.zeros((n, n), dtype=np.int8)
    est_lab_2d[off_mask & (A_prune > float(epsilon))] = 1
    est_lab_2d[off_mask & (A_prune < -float(epsilon))] = -1

    return true_lab_2d[off_mask], est_lab_2d[off_mask], true_lab_2d, est_lab_2d, off_mask


def edge_confusion_3x3(true_lab, est_lab):
    cm = np.zeros((3, 3), dtype=np.int64)
    idx = {-1: 0, 0: 1, 1: 2}
    for t, e in zip(true_lab, est_lab):
        cm[idx[int(t)], idx[int(e)]] += 1
    return cm


def multiclass_mcc(confusion):
    """Matthews correlation coefficient for a multiclass confusion matrix."""
    cm = np.asarray(confusion, dtype=np.float64)
    n = float(np.sum(cm))
    if n <= 0.0:
        return float("nan")
    c = float(np.trace(cm))
    row_sum = np.sum(cm, axis=1)
    col_sum = np.sum(cm, axis=0)
    num = c * n - float(np.dot(row_sum, col_sum))
    den = np.sqrt(
        (n * n - float(np.dot(col_sum, col_sum))) *
        (n * n - float(np.dot(row_sum, row_sum)))
    )
    if den == 0.0:
        return float("nan")
    return float(num / den)


def presence_metrics(true_lab, est_lab):
    t = true_lab != 0
    e = est_lab != 0
    tp = int(np.count_nonzero(t & e))
    fp = int(np.count_nonzero(~t & e))
    fn = int(np.count_nonzero(t & ~e))
    tn = int(np.count_nonzero(~t & ~e))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": safe_f1(precision, recall),
        "mcc": matthews_corrcoef(tp, fp, fn, tn),
        "fdp": safe_div(fp, tp + fp),
        "num_true_edges": int(tp + fn),
        "num_reco_edges": int(tp + fp),
        "num_candidates": int(true_lab.size),
        "density_true": safe_div(tp + fn, true_lab.size),
        "density_reco": safe_div(tp + fp, true_lab.size),
    }


def signed_metrics(true_lab, est_lab):
    true_present = true_lab != 0
    est_present = est_lab != 0
    correct = true_present & est_present & (true_lab == est_lab)
    sign_flip = true_present & est_present & (true_lab != est_lab)
    spurious = ~true_present & est_present
    missed = true_present & ~est_present

    signed_tp = int(np.count_nonzero(correct))
    signed_fp = int(np.count_nonzero(spurious) + np.count_nonzero(sign_flip))
    signed_fn = int(np.count_nonzero(missed) + np.count_nonzero(sign_flip))
    recovered_true = int(np.count_nonzero(true_present & est_present))
    n_flip = int(np.count_nonzero(sign_flip))
    precision = safe_div(signed_tp, signed_tp + signed_fp)
    recall = safe_div(signed_tp, signed_tp + signed_fn)
    return {
        "signed_TP": signed_tp,
        "signed_FP": signed_fp,
        "signed_FN": signed_fn,
        "sign_flip_count": n_flip,
        "signed_precision": precision,
        "signed_recall": recall,
        "signed_f1": safe_f1(precision, recall),
        "sign_flip_rate": safe_div(n_flip, recovered_true),
    }


def true_source_type_from_truth(truth_d, A_true, E_true):
    inhib = np.asarray(truth_d["node_is_inhibitory"]).astype(bool)
    n = A_true.shape[0]
    if inhib.shape != (n,):
        raise ValueError(f"node_is_inhibitory shape {inhib.shape} incompatible with N={n}")
    true_type = np.where(inhib, -1, 1).astype(np.int8)

    off_mask = ~np.eye(n, dtype=bool)
    E_true = E_true != 0
    for j in range(n):
        vals = A_true[off_mask[:, j] & E_true[:, j], j]
        if vals.size == 0:
            continue
        if true_type[j] == 1 and np.any(vals < 0.0):
            raise ValueError(f"Truth source column {j} marked excitatory but has negative true edges")
        if true_type[j] == -1 and np.any(vals > 0.0):
            raise ValueError(f"Truth source column {j} marked inhibitory but has positive true edges")
    return true_type


def source_type_metrics(true_source_type, reco_source_type, true_lab_2d, est_lab_2d, off_mask):
    reco_source_type = np.asarray(reco_source_type, dtype=np.int8)
    n = true_source_type.shape[0]
    if reco_source_type.shape != (n,):
        raise ValueError(f"neuron_type shape {reco_source_type.shape} incompatible with N={n}")

    counts = {
        "num_reco_exc_neurons": int(np.count_nonzero(reco_source_type == 1)),
        "num_reco_inh_neurons": int(np.count_nonzero(reco_source_type == -1)),
        "num_reco_und_neurons": int(np.count_nonzero(reco_source_type == 0)),
        "frac_reco_und_neurons": safe_div(np.count_nonzero(reco_source_type == 0), n),
    }

    source_conf = np.zeros((2, 3), dtype=np.int64)
    row_idx = {1: 0, -1: 1}
    col_idx = {1: 0, -1: 1, 0: 2}
    for j in range(n):
        source_conf[row_idx[int(true_source_type[j])], col_idx[int(reco_source_type[j])]] += 1

    src_idx = np.broadcast_to(np.arange(n, dtype=np.int64)[None, :], (n, n))
    true_edge = (true_lab_2d != 0) & off_mask
    reco_edge = (est_lab_2d != 0) & off_mask
    true_und = true_edge & (reco_source_type[src_idx] == 0)
    reco_und = reco_edge & (reco_source_type[src_idx] == 0)
    counts.update({
        "frac_true_edges_from_reco_und_src": safe_div(np.count_nonzero(true_und), np.count_nonzero(true_edge)),
        "frac_reco_edges_from_reco_und_src": safe_div(np.count_nonzero(reco_und), np.count_nonzero(reco_edge)),
        "source_type_confusion": source_conf,
    })
    return counts


def real_fit_metadata(md):
    if "bagsFDR_stageA" not in md:
        raise KeyError("fit metadata missing 'bagsFDR_stageA'; was this file produced by prism_EM_FDR_Bags_aggregate3c.py?")
    if "real_fit" not in md["bagsFDR_stageA"]:
        raise KeyError("fit metadata missing 'bagsFDR_stageA.real_fit'")
    return md["bagsFDR_stageA"]["real_fit"]


def metric_row(base_path, fit_name, fit_tag, fit_d, fit_md, truth_d, truth_md, truth_f, epsilon):
    A_true = as_static_2d(truth_d["A_true"], "A_true").astype(np.float64)
    E_true = as_static_2d(truth_d["E_true"], "E_true") != 0
    A_prune = as_static_2d(fit_d["A_prune"], "A_prune").astype(np.float64)
    true_lab, est_lab, true_lab_2d, est_lab_2d, off_mask = graph_labels(
        A_true, E_true, A_prune, epsilon
    )
    pmet = presence_metrics(true_lab, est_lab)
    smet = signed_metrics(true_lab, est_lab)
    cm3 = edge_confusion_3x3(true_lab, est_lab)
    smet["signed_mcc"] = multiclass_mcc(cm3)

    true_source_type = true_source_type_from_truth(truth_d, A_true, E_true)
    src_met = source_type_metrics(
        true_source_type,
        fit_d["neuron_type"],
        true_lab_2d,
        est_lab_2d,
        off_mask,
    )

    train_md = real_fit_metadata(fit_md)["train"]
    stage_a = fit_md["bagsFDR_stageA"]
    stage_b = fit_md["bagsFDR_stageB"]

    time_range_sec = np.asarray(train_md["time_range_sec"], dtype=np.float64)
    duration_sec = float(time_range_sec[1] - time_range_sec[0])
    num_reco_edges = int(pmet["num_reco_edges"])
    fdr_bound = float(stage_b["stability_false_edge_bound"])

    row = {
        "fit_name": fit_name,
        "fit_tag": fit_tag,
        "fit_file": fit_file(base_path, fit_name),
        "truth_file": truth_f,
        "num_neurons": int(A_true.shape[0]),
        "num_time_bins": int(train_md["num_time_bins"]),
        "time_range_sec": time_range_sec.astype(np.float64),
        "duration_sec": duration_sec,
        "duration_min": duration_sec / 60.0,
        "num_bags": int(stage_b["num_bags"]),
        "bag_frac": float(stage_a["bag_frac"]),
        "per_bag_quantile": float(stage_b["per_bag_quantile"]),
        "stab_sel_thresh": float(stage_b["stab_sel_thresh"]),
        "epsilon": float(epsilon),
        "stability_false_edge_bound": fdr_bound,
        "stability_bound_fdp": safe_div(fdr_bound, num_reco_edges),
        "confusion3": cm3,
        **pmet,
        **smet,
        **src_met,
    }
    return row


def assemble_metric_arrays(rows):
    keys_float = (
        "duration_sec", "duration_min", "bag_frac", "per_bag_quantile",
        "stab_sel_thresh", "epsilon", "stability_false_edge_bound",
        "stability_bound_fdp", "precision", "recall", "f1", "mcc", "fdp",
        "density_true", "density_reco", "signed_precision", "signed_recall",
        "signed_f1", "signed_mcc", "sign_flip_rate", "frac_reco_und_neurons",
        "frac_true_edges_from_reco_und_src", "frac_reco_edges_from_reco_und_src",
    )
    keys_int = (
        "num_neurons", "num_time_bins", "num_bags", "TP", "FP", "FN", "TN",
        "num_true_edges", "num_reco_edges", "num_candidates", "signed_TP",
        "signed_FP", "signed_FN", "sign_flip_count", "num_reco_exc_neurons",
        "num_reco_inh_neurons", "num_reco_und_neurons",
    )

    out = {
        "fit_name": np.asarray([r["fit_name"] for r in rows], dtype=object),
        "fit_tag": np.asarray([r["fit_tag"] for r in rows], dtype=object),
        "fit_file": np.asarray([r["fit_file"] for r in rows], dtype=object),
        "truth_file": np.asarray([r["truth_file"] for r in rows], dtype=object),
        "time_range_sec": np.stack([r["time_range_sec"] for r in rows], axis=0).astype(np.float64),
        "confusion3": np.stack([r["confusion3"] for r in rows], axis=0).astype(np.int64),
        "source_type_confusion": np.stack([r["source_type_confusion"] for r in rows], axis=0).astype(np.int64),
        "label_order": LABEL_ORDER.copy(),
        "source_truth_order": np.asarray(["exc", "inh"], dtype=object),
        "source_reco_order": SOURCE_RECO_NAMES.copy(),
    }
    out["confusion3_sum"] = np.sum(out["confusion3"], axis=0).astype(np.int64)
    out["source_type_confusion_sum"] = np.sum(out["source_type_confusion"], axis=0).astype(np.int64)
    for key in keys_float:
        out[key] = np.asarray([r[key] for r in rows], dtype=np.float64)
    for key in keys_int:
        out[key] = np.asarray([r[key] for r in rows], dtype=np.int64)
    return out


def main():
    args = parse_args()
    if args.epsilon < 0.0:
        raise ValueError("--epsilon must be non-negative")

    t0 = time.perf_counter()
    fit_tags = flatten_words(args.fitTags)
    if not fit_tags:
        raise ValueError("--fitTags produced an empty list")

    out_name = args.outName
    edge_dir = os.path.join(args.basePath, "edgeReco")
    plot_dir = os.path.join(args.basePath, "plots")
    os.makedirs(edge_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    rows = []
    truth_d = None
    truth_md = None
    truth_f = None
    truth_name_ref = None
    fit_names = [full_fit_name(args.fitNameTrunk, tag) for tag in fit_tags]

    for fit_tag, fit_name in zip(fit_tags, fit_names):
        fit_d, fit_md, inp_f = load_fit(args.basePath, fit_name, verb=args.verb)
        truth_name = fit_md["provenance"]["state_model_file"]
        if truth_name_ref is None:
            truth_name_ref = truth_name
            truth_d, truth_md, truth_f = load_truth_once(args.basePath, truth_name, verb=args.verb)
        elif truth_name != truth_name_ref:
            raise ValueError(
                f"All fits must use the same truth. First={truth_name_ref}, {fit_name}={truth_name}"
            )
        row = metric_row(
            args.basePath, fit_name, fit_tag,
            fit_d, fit_md, truth_d, truth_md, truth_f,
            epsilon=args.epsilon,
        )
        rows.append(row)
        if args.verb > 0:
            print(
                f"metric {fit_tag}: TP={row['TP']} FP={row['FP']} "
                f"FN={row['FN']} precision={row['precision']:.4f} "
                f"recall={row['recall']:.4f} MCC={row['mcc']:.4f}"
            )

    out_d = assemble_metric_arrays(rows)
    out_md = {
        "program": "edgeMeterAccuracy3c.py",
        "fitNameTrunk": args.fitNameTrunk,
        "fitTags": fit_tags,
        "fit_names": fit_names,
        "truth_name": truth_name_ref,
        "truth_file": truth_f,
        "metric_target": "A_prune",
        "candidate_universe": "all_off_diagonal_directed_pairs",
        "edge_presence_rule": "A_prune != 0 using epsilon tolerance",
        "truth_presence": "E_true != 0 on off-diagonal",
        "truth_sign": "sign(A_true) on E_true edges",
        "source_type_truth": "node_is_inhibitory from simTruth",
        "source_type_reco": "neuron_type from aggregate fit; 0 is undecided abstention",
        "epsilon": float(args.epsilon),
        "label_order": LABEL_ORDER.tolist(),
        "source_truth_order": ["exc", "inh"],
        "source_reco_order": SOURCE_RECO_NAMES.tolist(),
        "elapsed_sec": float(time.perf_counter() - t0),
    }

    out_f = os.path.join(edge_dir, f"{out_name}.npz")
    write_data_npz(out_d, out_f, metaD=out_md, verb=args.verb > 1)
    if args.verb > 0:
        print(f"\nSaved edge-meter metrics: {out_f}")

    plot_letters = normalize_plot_letters(args.showPlots)
    if plot_letters:
        from PlotterEdgeMeterAccuracy import Plotter

        unknown = sorted(set(plot_letters) - set("abc"))
        if unknown:
            raise ValueError(f"Unknown plot letters for edgeMeterAccuracy3c.py: {unknown}")
        args.prjName = out_name
        args.outPath = plot_dir
        args.formatVenue = "prod"
        plot = Plotter(args)
        if "a" in plot_letters:
            plot.recovery_summary(out_d, out_md, figId="a")
        if "b" in plot_letters:
            plot.count_summary(out_d, out_md, figId="b")
        if "c" in plot_letters:
            plot.confusion_summary(out_d, out_md, tagIdxL=[2, -1], figId="c")
        plot.display_all()


if __name__ == "__main__":
    main()
