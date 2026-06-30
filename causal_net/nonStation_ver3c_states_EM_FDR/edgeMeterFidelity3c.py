#!/usr/bin/env python3
"""Stability metrics for PRISM-FDR reconstructed connectivity graphs.

Compares K >= 2 aggregate fit files (data subsets / time slices) without
requiring ground truth.  Two comparison modes:
  consecutive  -- compare adjacent ordered pairs (k vs k+1)   [default]
  last         -- compare each subset against the last subset as reference
"""

import argparse
import os
import secrets
import time

import numpy as np

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stability metrics for PRISM-FDR aggregate fits (no ground truth)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--basePath", required=True,
                        help="Run directory containing prismFit/, edgeFidelity/, and plots/")
    parser.add_argument("--fitNameTrunk", required=True,
                        help="Common prefix of aggregate fit files")
    parser.add_argument("--fitTags", nargs="+", required=True,
                        help="Ordered tags identifying each data subset")
    parser.add_argument("--compare_mode", default="consecutive",
                        choices=["consecutive", "last"],
                        help="consecutive: compare adjacent pairs (x=midpoint duration); "
                             "last: compare each subset against the last subset (x=subset duration)")
    parser.add_argument("--outName", default=argparse.SUPPRESS,
                        help="Output metric stem in edgeFidelity/; default is fitNameTrunk_efmHASH4")
    parser.add_argument("-p", "--showPlots", nargs="+", default=["a"],
                        help="Plot groups: a=edge fidelity summary b=weight stats")
    parser.add_argument("-X", "--noXterm", action="store_true",
                        help="Disable X terminal for plotting")
    parser.add_argument("-v", "--verb", type=int, default=1,
                        help="Verbosity level")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def load_fit(base_path, fit_name, verb=1):
    inp_f = fit_file(base_path, fit_name)
    fit_d, fit_md = read_data_npz(inp_f, verb=verb > 1)
    if "A_prune" not in fit_d:
        raise KeyError(f"{inp_f} must contain A_prune")
    if "A_sd_selected" not in fit_d:
        raise KeyError(f"{inp_f} must contain A_sd_selected (bootstrap SD matrix)")
    if "neuron_type" not in fit_d:
        raise KeyError(f"{inp_f} must contain neuron_type")
    return fit_d, fit_md, inp_f


def as_2d(arr, name):
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square 2D, got {arr.shape}")
    return arr


def safe_div(num, den):
    den = float(den)
    return float("nan") if den == 0.0 else float(num) / den


# ---------------------------------------------------------------------------
# build per-subset edge arrays
# ---------------------------------------------------------------------------

def build_subsets(fit_records, verb=1):
    """Extract A_prune, A_sd_selected, neuron_type, diagonal per subset.

    Returns a list of dicts, one per data subset.
    """
    subsets = []
    for rec in fit_records:
        fit_d, fit_md, inp_f, fit_name, fit_tag = rec
        A_prune = as_2d(fit_d["A_prune"], "A_prune").astype(np.float64)
        A_sd_selected = as_2d(fit_d["A_sd_selected"], "A_sd_selected").astype(np.float64)
        # NaN means only 1 bag selected the edge (ddof=1 undefined) — treat as 0 uncertainty
        np.nan_to_num(A_sd_selected, nan=0.0, copy=False)
        neuron_type = np.asarray(fit_d["neuron_type"], dtype=np.int8).ravel()
        single_rates = np.asarray(fit_d["single_rates"], dtype=np.float64).ravel()
        n = A_prune.shape[0]
        if neuron_type.shape != (n,):
            raise ValueError(f"neuron_type shape {neuron_type.shape} != N={n}")
        if single_rates.shape != (n,):
            raise ValueError(f"single_rates shape {single_rates.shape} != N={n}")

        # parse duration from tag string (e.g. "30min" -> 30.0), fall back to metadata
        duration_min = float("nan")
        tag_str = str(fit_tag)
        if tag_str.endswith("min"):
            try:
                duration_min = float(tag_str[:-3])
            except ValueError:
                pass  # tag doesn't encode duration; plotter will use index as x-axis
        if not np.isfinite(duration_min):
            if not fit_md:
                raise ValueError(f"fit_tag '{fit_tag}' has no parseable duration and fit_md is empty")
            if "bagsFDR_stageA" not in fit_md:
                raise KeyError(f"fit_tag '{fit_tag}': fit_md missing 'bagsFDR_stageA' — re-run pipeline to regenerate files")
            if "real_fit" not in fit_md["bagsFDR_stageA"]:
                raise KeyError(f"fit_tag '{fit_tag}': bagsFDR_stageA missing 'real_fit' — re-run prism_FDR_Bags_train3c.py")
            tr = fit_md["bagsFDR_stageA"]["real_fit"]["train"]["time_range_sec"]
            tr = np.asarray(tr, dtype=np.float64)
            duration_min = float(tr[1] - tr[0]) / 60.0

        stage_b = fit_md.get("bagsFDR_stageB", {})
        min_posW = float(stage_b["min_posW"]) if "min_posW" in stage_b else float("nan")
        max_negW = float(stage_b["max_negW"]) if "max_negW" in stage_b else float("nan")

        off = A_prune.copy()
        np.fill_diagonal(off, 0.0)
        source_edge_counts = (np.abs(off) > 1e-12).sum(axis=0).astype(np.float64)
        source_edge_counts[neuron_type == 0] = 0.0
        weight_ranges = {}
        edge_counts = {}
        for cls, key in ((1, "exc"), (-1, "inh"), (0, "und")):
            cols = neuron_type == cls
            if np.any(cols):
                vals = off[:, cols]
                nz = vals[np.abs(vals) > 1e-12]
            else:
                nz = np.array([])
            weight_ranges[key] = (
                float(nz.min()) if nz.size else float("nan"),
                float(nz.max()) if nz.size else float("nan"),
                float(np.median(nz)) if nz.size else float("nan"),
            )
            edge_counts[key] = int(nz.size)

        subsets.append({
            "fit_name": fit_name,
            "fit_tag": fit_tag,
            "fit_file": inp_f,
            "A_prune": A_prune,
            "A_sd_selected": A_sd_selected,
            "neuron_type": neuron_type,
            "single_rates": single_rates,
            "source_edge_counts": source_edge_counts,
            "diag": np.diag(A_prune).copy(),
            "n": n,
            "duration_min": duration_min,
            "min_posW": min_posW,
            "max_negW": max_negW,
            "weight_ranges": weight_ranges,
            "edge_counts": edge_counts,
        })
    return subsets


# ---------------------------------------------------------------------------
# epsilon and E_global
# ---------------------------------------------------------------------------

def compute_epsilon(subsets):
    """Minimum non-zero absolute off-diagonal weight across all subsets."""
    vals = []
    for s in subsets:
        A = s["A_prune"]
        n = s["n"]
        off = A.copy()
        np.fill_diagonal(off, 0.0)
        nz = np.abs(off[off != 0.0])
        if nz.size > 0:
            vals.append(nz.min())
    if not vals:
        raise ValueError("All A_prune matrices are entirely zero; cannot compute epsilon")
    return float(np.min(vals))


def compute_e_global(subsets, eps):
    """Union of active off-diagonal edges across all subsets."""
    n = subsets[0]["n"]
    union = np.zeros((n, n), dtype=bool)
    off_mask = ~np.eye(n, dtype=bool)
    for s in subsets:
        union |= (np.abs(s["A_prune"]) >= eps) & off_mask
    return union   # (n, n) bool, diagonal always False




# ---------------------------------------------------------------------------
# off-diagonal pairwise metrics
# ---------------------------------------------------------------------------

def edge_sets(A, e_global, eps):
    """Return positive and negative edge sets as boolean arrays over e_global flat."""
    idx = np.where(e_global)
    vals = A[idx]
    E_pos = vals > eps
    E_neg = vals < -eps
    return E_pos, E_neg


def jaccard(a, b):
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return safe_div(inter, union)


def compute_topology_metrics(A1, A2, e_global, eps, include_smr=True):
    """J+, J-, and optionally SMR for one pair of weight matrices."""
    E1p, E1n = edge_sets(A1, e_global, eps)
    E2p, E2n = edge_sets(A2, e_global, eps)

    jp = jaccard(E1p, E2p)
    jn = jaccard(E1n, E2n)
    out = {"J_pos": jp, "J_neg": jn}

    if include_smr:
        both_present = (E1p | E1n) & (E2p | E2n)
        sign_agree = ((E1p & E2p) | (E1n & E2n))
        out["SMR"] = safe_div(np.count_nonzero(sign_agree), np.count_nonzero(both_present))

    return out


def _spearman_weighted(X, Y, W=None):
    """Weighted Spearman correlation of two 1D arrays.

    W=None gives the unweighted version.
    """
    n = len(X)
    if n < 2:
        return float("nan")
    # ranks (scipy-style average for ties)
    from scipy.stats import rankdata
    Rx = rankdata(X).astype(np.float64)
    Ry = rankdata(Y).astype(np.float64)

    if W is None:
        W = np.ones(n, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    Ws = W.sum()
    if Ws == 0.0:
        return float("nan")
    mRx = (W * Rx).sum() / Ws
    mRy = (W * Ry).sum() / Ws
    dx = Rx - mRx
    dy = Ry - mRy
    num = (W * dx * dy).sum()
    den = np.sqrt((W * dx * dx).sum() * (W * dy * dy).sum())
    return safe_div(num, den)


def compute_magnitude_metrics(A1, sigma1, A2, sigma2, e_global, eps):
    """r_s and r_s^w for one pair."""
    idx = np.where(e_global)
    X = np.abs(A1[idx])
    Y = np.abs(A2[idx])
    S1 = np.nan_to_num(sigma1[idx], nan=0.0)
    S2 = np.nan_to_num(sigma2[idx], nan=0.0)
    W = 1.0 / ((S1 + S2) / 2.0 + eps)

    rs = _spearman_weighted(X, Y, W=None)
    rsw = _spearman_weighted(X, Y, W=W)
    return {"r_spearman": rs, "r_spearman_w": rsw}


def metrics_for_scope(A1, sigma1, A2, sigma2, e_global, eps, neuron_type1, neuron_type2, scope):
    """Compute all off-diagonal metrics for a given source-type scope.

    scope: 'all', 'exc', 'inh'
    """
    if scope == "all":
        mask = e_global
    else:
        sign = 1 if scope == "exc" else -1
        n = A1.shape[0]
        src_mask = np.zeros((n, n), dtype=bool)
        for j in range(n):
            if neuron_type1[j] == sign or neuron_type2[j] == sign:
                src_mask[:, j] = True
        mask = e_global & src_mask

    if not np.any(mask):
        nan4 = {"J_pos": float("nan"), "J_neg": float("nan")}
        if scope == "all":
            nan4["SMR"] = float("nan")
            return nan4
        nan2 = {"r_spearman": float("nan"), "r_spearman_w": float("nan")}
        return {**nan4, **nan2}

    top = compute_topology_metrics(A1, A2, mask, eps, include_smr=(scope == "all"))
    if scope == "all":
        return top
    mag = compute_magnitude_metrics(A1, sigma1, A2, sigma2, mask, eps)
    return {**top, **mag}


# ---------------------------------------------------------------------------
# diagonal metrics
# ---------------------------------------------------------------------------

def compute_diagonal_metrics_pair(d1, d2):
    """Mean absolute change between two diagonal vectors."""
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)
    if d1.size == 0 or d1.shape != d2.shape:
        return float("nan")
    return float(np.mean(np.abs(d1 - d2)))


# ---------------------------------------------------------------------------
# run comparisons
# ---------------------------------------------------------------------------

SCOPES = ["all", "exc", "inh"]


def compare_pair(s1, s2, e_global, eps):
    """All metrics for one ordered pair of subsets."""
    result = {}
    for scope in SCOPES:
        m = metrics_for_scope(
            s1["A_prune"], s1["A_sd_selected"],
            s2["A_prune"], s2["A_sd_selected"],
            e_global, eps,
            s1["neuron_type"], s2["neuron_type"],
            scope,
        )
        result[scope] = m

    result["diag_mae"] = compute_diagonal_metrics_pair(s1["diag"], s2["diag"])
    return result


def run_consecutive(subsets, e_global, eps):
    """Strategy B: compare (k, k+1) pairs."""
    comparisons = []
    for i in range(len(subsets) - 1):
        s1, s2 = subsets[i], subsets[i + 1]
        res = compare_pair(s1, s2, e_global, eps)
        res["label"] = f"{s1['fit_tag']} vs {s2['fit_tag']}"
        res["x"] = 0.5 * (s1["duration_min"] + s2["duration_min"])
        comparisons.append(res)
    return comparisons


def run_last(subsets, e_global, eps):
    """Compare each subset (except the last) against the last subset as reference.

    x-axis is each subset's own duration, so caller can pass e.g. [20min,40min], [10min,50min],
    [0min,60min] and study how shrinking the time window degrades results vs. the full window.
    The last subset itself is the reference and is not compared against itself.
    """
    ref = subsets[-1]
    comparisons = []
    for s in subsets[:-1]:
        res = compare_pair(s, ref, e_global, eps)
        res["label"] = f"{s['fit_tag']} vs {ref['fit_tag']}"
        res["x"] = s["duration_min"]
        comparisons.append(res)
    return comparisons


# ---------------------------------------------------------------------------
# assemble output arrays
# ---------------------------------------------------------------------------

def assemble_output(comparisons, subsets, mode, eps):
    """Pack all results into numpy arrays for the Plotter."""
    nc = len(comparisons)
    ns = len(subsets)
    all_rates = np.concatenate([s["single_rates"] for s in subsets])
    all_rates = all_rates[np.isfinite(all_rates)]
    if all_rates.size:
        n_rate_bins = min(30, int(all_rates.size))
        r0, r1 = float(np.min(all_rates)), float(np.max(all_rates))
        if r0 == r1:
            pad = max(0.5, abs(r0) * 0.05)
            r0, r1 = r0 - pad, r1 + pad
        rate_edges = np.linspace(r0, r1, n_rate_bins + 1, dtype=np.float64)
        edge_rate_hist = np.zeros((n_rate_bins, ns), dtype=np.float64)
        for isub, s in enumerate(subsets):
            hist, _ = np.histogram(
                s["single_rates"], bins=rate_edges, weights=s["source_edge_counts"]
            )
            edge_rate_hist[:, isub] = hist
    else:
        rate_edges = np.array([0.0, 1.0], dtype=np.float64)
        edge_rate_hist = np.zeros((1, ns), dtype=np.float64)

    out = {
        "compare_mode": np.array([mode], dtype=object),
        "epsilon": np.array([eps]),
        "x": np.array([c["x"] for c in comparisons], dtype=np.float64),
        "labels": np.array([c["label"] for c in comparisons], dtype=object),
        "subset_tags": np.array([s["fit_tag"] for s in subsets], dtype=object),
        "subset_x": np.array([s["duration_min"] for s in subsets], dtype=np.float64),
        "diag_mae": np.array([c["diag_mae"] for c in comparisons], dtype=np.float64),
        "min_posW": np.array([s["min_posW"] for s in subsets], dtype=np.float64),
        "max_negW": np.array([s["max_negW"] for s in subsets], dtype=np.float64),
        "w_min_exc": np.array([s["weight_ranges"]["exc"][0] for s in subsets], dtype=np.float64),
        "w_max_exc": np.array([s["weight_ranges"]["exc"][1] for s in subsets], dtype=np.float64),
        "w_med_exc": np.array([s["weight_ranges"]["exc"][2] for s in subsets], dtype=np.float64),
        "w_min_inh": np.array([s["weight_ranges"]["inh"][0] for s in subsets], dtype=np.float64),
        "w_max_inh": np.array([s["weight_ranges"]["inh"][1] for s in subsets], dtype=np.float64),
        "w_med_inh": np.array([s["weight_ranges"]["inh"][2] for s in subsets], dtype=np.float64),
        "w_min_und": np.array([s["weight_ranges"]["und"][0] for s in subsets], dtype=np.float64),
        "w_max_und": np.array([s["weight_ranges"]["und"][1] for s in subsets], dtype=np.float64),
        "n_edges_exc": np.array([s["edge_counts"]["exc"] for s in subsets], dtype=np.int64),
        "n_edges_inh": np.array([s["edge_counts"]["inh"] for s in subsets], dtype=np.int64),
        "n_edges_und": np.array([s["edge_counts"]["und"] for s in subsets], dtype=np.int64),
        "rate_bin_edges": rate_edges,
        "edge_count_rate_hist": edge_rate_hist,
    }

    # off-diagonal metrics per scope
    for scope in SCOPES:
        metric_keys = ["J_pos", "J_neg", "SMR"] if scope == "all" else [
            "J_pos", "J_neg", "r_spearman", "r_spearman_w"
        ]
        for key in metric_keys:
            out[f"{scope}_{key}"] = np.array(
                [c[scope][key] for c in comparisons], dtype=np.float64
            )

    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    t0 = time.perf_counter()

    fit_tags = flatten_words(args.fitTags)
    if len(fit_tags) < 2:
        raise ValueError("Need at least 2 --fitTags")

    if not hasattr(args, "outName"):
        args.outName = f"{args.fitNameTrunk}_efm{secrets.token_hex(2)}"

    fidelity_dir = os.path.join(args.basePath, "edgeFidelity")
    plot_dir = os.path.join(args.basePath, "plots")
    os.makedirs(fidelity_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    fit_names = [full_fit_name(args.fitNameTrunk, tag) for tag in fit_tags]

    fit_records = []
    for fit_tag, fit_name in zip(fit_tags, fit_names):
        fit_d, fit_md, inp_f = load_fit(args.basePath, fit_name, verb=args.verb)
        fit_records.append((fit_d, fit_md, inp_f, fit_name, fit_tag))
        if args.verb > 0:
            print(f"loaded: {fit_name}")

    subsets = build_subsets(fit_records, verb=args.verb)
    eps = compute_epsilon(subsets)
    e_global = compute_e_global(subsets, eps)

    if args.verb > 0:
        print(f"epsilon={eps:.3e}  |E_global|={np.count_nonzero(e_global)}")

    if args.compare_mode == "consecutive":
        comparisons = run_consecutive(subsets, e_global, eps)
    else:
        comparisons = run_last(subsets, e_global, eps)

    if args.verb > 0:
        for i, c in enumerate(comparisons):
            print(
                f"comparison {c['label']}: "
                f"J+={c['all']['J_pos']:.3f}  "
                f"J-={c['all']['J_neg']:.3f}  "
                f"SMR={c['all']['SMR']:.3f}  "
                f"rs_exc={c['exc']['r_spearman']:.3f}  "
                f"rs_inh={c['inh']['r_spearman']:.3f}  "
                f"rsw_exc={c['exc']['r_spearman_w']:.3f}  "
                f"rsw_inh={c['inh']['r_spearman_w']:.3f}  "
                f"diag_mae={c['diag_mae']:.3g}"
            )

    out_d = assemble_output(comparisons, subsets, args.compare_mode, eps)
    out_md = {
        "program": "edgeMeterFidelity3c.py",
        "fitNameTrunk": args.fitNameTrunk,
        "fitTags": fit_tags,
        "compare_mode": args.compare_mode,
        "epsilon": float(eps),
        "elapsed_sec": float(time.perf_counter() - t0),
    }

    out_f = os.path.join(fidelity_dir, f"{args.outName}.npz")
    write_data_npz(out_d, out_f, metaD=out_md, verb=args.verb > 1)
    if args.verb > 0:
        print(f"\nSaved fidelity metrics: {out_f}")

    plot_letters = normalize_plot_letters(args.showPlots)
    if plot_letters:
        from PlotterEdgeMeterFidelity import Plotter

        unknown = sorted(set(plot_letters) - set("ab"))
        if unknown:
            raise ValueError(f"Unknown plot letters: {unknown}")

        args.prjName = args.outName
        args.outPath = plot_dir
        args.formatVenue = "prod"
        plot = Plotter(args)
        if "a" in plot_letters:
            plot.plot_jaccard_smr(out_d, out_md, figId="a")
        if "b" in plot_letters:
            plot.plot_weight_thresholds(out_d, out_md, figId="b")
        plot.display_all()


if __name__ == "__main__":
    main()
