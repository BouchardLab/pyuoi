#!/usr/bin/env python3
"""
PRISM Stage 3 - Dictionary Update (eval/plots).

Compares A_hat, B_hat to A_true, B_true and evaluates forward model
quality using the optimized dictionaries with C_hat from Stage 2.

Pass criteria:
  - A correlation (per state) > 0.90
  - B correlation (per state) > 0.90
  - Forward NLL with A_hat,B_hat within 5% of Stage 1 NLL

Produces two PNG canvases:
  <dataName>_s3_canvas1.png  : weight matrices A_true vs A_hat per state
  <dataName>_s3_canvas2.png  : training curves, B scatter, summary

Reads:
  <basePath>/prismFit/<dataName>.stage3.npz
"""

import os
import argparse
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import precision_recall_curve, auc
from itertools import permutations
from scipy.stats import pearsonr
from toolbox.Util_NumpyIO import read_data_npz

THRESH_CORR_A = 0.90
THRESH_CORR_B = 0.90
THRESH_NLL_FRAC = 0.05   # A_hat NLL within 5% of A_true NLL


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        dest="verb")
    parser.add_argument("--basePath",
                        default="/dataVault2026/neurodata_tmp2")
    parser.add_argument("--dataName", default=None,
                        help="Stage 3 output name.")
    parser.add_argument("-p", "--plotFmt", default="b")
    parser.add_argument("-X", "--noBlock", action="store_true")
    parser.add_argument("--prune_thresh", type=float, default=0.04,
                    help="Threshold below which |A_hat| is pruned to zero.")

    args = parser.parse_args()
    args.inpFit   = os.path.join(args.basePath, "prismFit")
    args.outPlots = os.path.join(args.basePath, "plots")

    print("myArg-program:", parser.prog)
    for arg in vars(args):
        print("myArg:", arg, getattr(args, arg))

    assert args.dataName is not None
    os.makedirs(args.outPlots, exist_ok=True)
    return args


def pearson_r(x, y):
    mask  = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2: return np.nan
    xm    = x[mask] - x[mask].mean()
    ym    = y[mask] - y[mask].mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    return float(np.dot(xm, ym) / denom) if denom > 0 else np.nan


def pass_label(value, threshold, higher_is_better=True):
    ok = (value >= threshold) if higher_is_better else (value <= threshold)
    return ("PASS" if ok else "FAIL"), ("green" if ok else "red")


def _format_lr_from_meta(fitMD):
    """Format all lr_* keys from metadata for display."""
    parts = [f"{k}={fitMD[k]}" for k in sorted(fitMD.keys()) if k.startswith("lr_")]
    return "  ".join(parts)


def make_title(tag, fitMD):
    return (f"PRISM Stage 3 — {tag}\n"
            f"{fitMD['short_name']}   "
            f"N={fitMD['num_neurons']}  "
            f"M={fitMD['num_states']}  "
            f"{_format_lr_from_meta(fitMD)}  "
            f"n_epoch={fitMD['n_epoch']}  "
            f"λ_ρ={fitMD['lam_rho']}  "
            f"ρ_max={fitMD['rho_max']}")


# ================================================================
#  Canvas 1 : weight matrices  (M rows x 4 cols)
#    cols: A_true_m | A_hat_m | A_hat - A_true | B scatter
# ================================================================
def make_canvas1(fitMD, A_true, A_hat, B_true, B_hat):
    M, N, _ = A_true.shape

    # shared color scale across all states
    a_abs_max = max(np.abs(A_true).max(), np.abs(A_hat).max())
    diff_max  = np.abs(A_hat - A_true).max()

    fig = plt.figure(figsize=(16, 4 * M))
    fig.suptitle(make_title("Weight Matrices", fitMD),
                 fontsize=10, y=1.01)

    gs = gridspec.GridSpec(M, 4, figure=fig,
                           hspace=0.35, wspace=0.35)

    for m in range(M):
        r_A = pearson_r(A_true[m].ravel(), A_hat[m].ravel())
        r_B = pearson_r(B_true[m].ravel(), B_hat[m].ravel())

        # col 0: A_true
        ax0 = fig.add_subplot(gs[m, 0])
        im0 = ax0.imshow(A_true[m], cmap="RdBu_r", aspect="auto",
                         vmin=-a_abs_max, vmax=a_abs_max)
        plt.colorbar(im0, ax=ax0, pad=0.02)
        ax0.set_title(f"A_true  state {m}")
        ax0.set_xlabel("pre-syn j"); ax0.set_ylabel("post-syn i")

        # col 1: A_hat
        ax1 = fig.add_subplot(gs[m, 1])
        im1 = ax1.imshow(A_hat[m], cmap="RdBu_r", aspect="auto",
                         vmin=-a_abs_max, vmax=a_abs_max)
        plt.colorbar(im1, ax=ax1, pad=0.02)
        lbl, col = pass_label(r_A, THRESH_CORR_A)
        ax1.set_title(f"A_hat  state {m}   r={r_A:.4f}  {lbl}",
                      color=col)
        ax1.set_xlabel("pre-syn j"); ax1.set_ylabel("post-syn i")

        # col 2: A_hat - A_true
        ax2 = fig.add_subplot(gs[m, 2])
        im2 = ax2.imshow(A_hat[m] - A_true[m], cmap="RdBu_r",
                         aspect="auto",
                         vmin=-diff_max, vmax=diff_max)
        plt.colorbar(im2, ax=ax2, pad=0.02)
        rmse = float(np.sqrt(((A_hat[m] - A_true[m])**2).mean()))
        ax2.set_title(f"A_hat − A_true  state {m}   RMSE={rmse:.4f}")
        ax2.set_xlabel("pre-syn j"); ax2.set_ylabel("post-syn i")

        # col 3: B scatter
        ax3 = fig.add_subplot(gs[m, 3])
        ax3.scatter(B_true[m], B_hat[m], s=18, alpha=0.7,
                    color="steelblue")
        lim = max(np.abs(B_true[m]).max(),
                  np.abs(B_hat[m]).max()) * 1.1
        ax3.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        ax3.set_xlabel("B_true"); ax3.set_ylabel("B_hat")
        lbl, col = pass_label(r_B, THRESH_CORR_B)
        ax3.set_title(f"Bias B  state {m}   r={r_B:.4f}  {lbl}",
                      color=col)

    return fig


# ================================================================
#  Canvas 2 : training curves + summary  (2 rows x 3 cols)
# ================================================================
def make_canvas2(fitMD, A_true, A_hat, B_true, B_hat,
                 loss_epoch, nll_epoch, pen_epoch):
    M, N, _ = A_true.shape

    # per-state metrics
    r_A_per = [pearson_r(A_true[m].ravel(), A_hat[m].ravel())
               for m in range(M)]
    r_B_per = [pearson_r(B_true[m].ravel(), B_hat[m].ravel())
               for m in range(M)]
    rmse_A  = [float(np.sqrt(((A_hat[m]-A_true[m])**2).mean()))
               for m in range(M)]

    # spectral radii
    def spec_rad(A_np):
        return [float(np.max(np.abs(np.linalg.eigvals(A_np[m]))))
                for m in range(M)]

    rho_true = spec_rad(A_true)
    rho_hat  = spec_rad(A_hat)

    # NLL improvement: final vs initial (epoch 0)
    nll_init  = float(nll_epoch[0])
    nll_final = float(nll_epoch[-1])
    nll_delta = (nll_final - nll_init) / max(abs(nll_init), 1e-6)

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(make_title("Statistics Panels", fitMD),
                 fontsize=10, y=1.00)
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.38)

    # ---- (0,0) Training loss curves ----
    ax_l = fig.add_subplot(gs[0, 0])
    epochs = np.arange(1, len(loss_epoch) + 1)
    ax_l.plot(epochs, loss_epoch, lw=1.5, color="royalblue",
              label="total loss")
    ax_l.plot(epochs, nll_epoch,  lw=1.2, color="steelblue",
              ls="--", label="NLL")
    ax_l.plot(epochs, pen_epoch,  lw=1.0, color="darkorange",
              ls=":", label="ρ penalty")
    ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Loss")
    ax_l.set_title(f"Training curves\n"
                   f"NLL: {nll_init:.3f} → {nll_final:.3f}  "
                   f"(Δ={nll_delta*100:+.2f}%)")
    ax_l.legend(fontsize=8)

    # ---- (0,1) A_true vs A_hat scatter (all states pooled) ----
    ax_s = fig.add_subplot(gs[0, 1])
    colors = plt.cm.tab10(np.linspace(0, 0.5, M))
    for m in range(M):
        ax_s.scatter(A_true[m].ravel(), A_hat[m].ravel(),
                     s=3, alpha=0.25, color=colors[m],
                     label=f"m={m} r={r_A_per[m]:.3f}")
    amax = max(np.abs(A_true).max(), np.abs(A_hat).max())
    ax_s.plot([-amax, amax], [-amax, amax], "k--", lw=1)
    ax_s.set_xlabel("A_true"); ax_s.set_ylabel("A_hat")
    ax_s.set_title("A scatter  (all states pooled)")
    ax_s.legend(fontsize=7, markerscale=4)

    # ---- (0,2) Spectral radius comparison ----
    ax_r = fig.add_subplot(gs[0, 2])
    x = np.arange(M)
    w = 0.35
    ax_r.bar(x - w/2, rho_true, w, color="steelblue",
             alpha=0.75, label="ρ(A_true)")
    ax_r.bar(x + w/2, rho_hat,  w, color="tomato",
             alpha=0.75, label="ρ(A_hat)")
    rho_max_raw = fitMD["rho_max"]
    if isinstance(rho_max_raw, (list, np.ndarray)):
        rho_max_list = rho_max_raw if isinstance(rho_max_raw, list) \
            else list(np.atleast_1d(rho_max_raw).astype(float))
        if len(rho_max_list) == 1:
            rho_max_list = np.full(M, float(rho_max_list[0]))
        for m in range(M):
            if m < len(rho_max_list):
                ax_r.hlines(rho_max_list[m], x[m] - w/2, x[m] + w/2,
                            color="black", lw=1.2, ls="--",
                            label="ρ_max" if m == 0 else None)
    else:
        rho_max = float(rho_max_raw)
        ax_r.axhline(rho_max, color="black", lw=1.2, ls="--",
                     label=f"ρ_max={rho_max}")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([f"state {m}" for m in range(M)])
    ax_r.set_ylabel("Spectral radius")
    ax_r.set_title("Spectral radius\nA_true vs A_hat")
    ax_r.legend(fontsize=8)

    # ---- (1,0) Per-state A correlation bar ----
    ax_ac = fig.add_subplot(gs[1, 0])
    ax_ac.bar(x, r_A_per, color=[colors[m] for m in range(M)],
              alpha=0.85)
    ax_ac.axhline(THRESH_CORR_A, color="red", lw=1.2,
                  ls="--", label=f"thresh={THRESH_CORR_A}")
    ax_ac.set_xticks(x)
    ax_ac.set_xticklabels([f"state {m}" for m in range(M)])
    ax_ac.set_ylim(0, 1.05)
    ax_ac.set_ylabel("Pearson r")
    ax_ac.set_title("A correlation per state\nA_true vs A_hat")
    ax_ac.legend(fontsize=8)

    # ---- (1,1) RMSE of A per state ----
    ax_rm = fig.add_subplot(gs[1, 1])
    ax_rm.bar(x, rmse_A, color=[colors[m] for m in range(M)],
              alpha=0.85)
    ax_rm.set_xticks(x)
    ax_rm.set_xticklabels([f"state {m}" for m in range(M)])
    ax_rm.set_ylabel("RMSE")
    ax_rm.set_title("A matrix RMSE per state\n||A_hat − A_true||_F / N")

    # ---- (1,2) Pass/fail summary ----
    ax_pf = fig.add_subplot(gs[1, 2])
    ax_pf.axis("off")

    rows = [
        ("PRISM Stage 3 Summary",          "black", 11, True),
        ("",                                "black",  9, False),
        (f"N={fitMD['num_neurons']}  "
         f"M={fitMD['num_states']}",  "black", 9, False),
        (f"{_format_lr_from_meta(fitMD)}  "
         f"n_epoch={fitMD['n_epoch']}","black", 9, False),
        (f"lam_rho={fitMD['lam_rho']}  "
         f"rho_max={fitMD['rho_max']}","black", 9, False),
        ("",                                "black",  9, False),
        (f"NLL init  : {nll_init:.4f}",     "black",  9, False),
        (f"NLL final : {nll_final:.4f}  "
         f"({nll_delta*100:+.2f}%)",         "black",  9, False),
        ("",                                "black",  9, False),
        ("A correlation (per state):",       "black",  9, False),
    ]
    for m in range(M):
        lbl2, col2 = pass_label(r_A_per[m], THRESH_CORR_A)
        rows.append((f"  state {m}: {r_A_per[m]:.4f}  "
                     f"ρ: {rho_true[m]:.3f}→{rho_hat[m]:.3f}",
                     col2, 9, True))
    rows += [
        ("",                                "black",  9, False),
        ("B correlation (per state):",       "black",  9, False),
    ]
    for m in range(M):
        lbl2, col2 = pass_label(r_B_per[m], THRESH_CORR_B)
        rows.append((f"  state {m}: {r_B_per[m]:.4f}",
                     col2, 9, True))

    y = 0.97
    for txt, color, fs, bold in rows:
        ax_pf.text(0.05, y, txt,
                   transform=ax_pf.transAxes,
                   fontsize=fs, color=color,
                   fontweight="bold" if bold else "normal",
                   verticalalignment="top",
                   family="monospace")
        y -= 0.068

    return fig, r_A_per, r_B_per


# ================================================================
#  L1 Diagnostics
# ================================================================
def print_l1_diagnostics(A_true, A_hat, modes=None, lam_l1=None,
                          prune_thresh=0.01, n_bins=5, save_path=None):
    """
    Comprehensive diagnostics for understanding L1 edge loss vs edge recovery.

    Args:
        A_true:       np.array shape (M,N,N) — true edge weights
        A_hat:        np.array shape (M,N,N) — predicted edge weights
        modes:        np.array of mode indices per edge (flattened), or None
        lam_l1:       float, L1 lambda used in training (for labeling only)
        prune_thresh: float, threshold below which A_hat is considered pruned to zero
        n_bins:       int, number of bins for edge weight stratification
        save_path:    str or None, if given saves the figure there
    """
    label  = f"L1={lam_l1}" if lam_l1 is not None else "this run"
    A_true = np.array(A_true).ravel()
    A_hat  = np.array(A_hat).ravel()

    nonzero_mask = np.abs(A_true) > 0
    zero_mask    = ~nonzero_mask
    pruned_mask  = np.abs(A_hat) < prune_thresh

    # ------------------------------------------------------------------ #
    # 1. OVERALL RECALL / PRECISION SUMMARY
    # ------------------------------------------------------------------ #
    n_true_edges      = nonzero_mask.sum()
    n_true_zeros      = zero_mask.sum()
    true_edges_lost   = (nonzero_mask & pruned_mask).sum()
    true_zeros_pruned = (zero_mask    & pruned_mask).sum()

    print("=" * 60)
    print(f"  L1 EDGE DIAGNOSTICS  ({label}, prune_thresh={prune_thresh})")
    print("=" * 60)
    print(f"\n[1] Overall edge counts")
    print(f"    True non-zero edges  : {n_true_edges}")
    print(f"    True zero edges      : {n_true_zeros}")
    print(f"\n[2] After pruning (|A_hat| < {prune_thresh})")
    print(f"    True edges LOST      : {true_edges_lost}  "
          f"({100*true_edges_lost/max(n_true_edges,1):.1f}% of true edges)")
    print(f"    True zeros pruned    : {true_zeros_pruned}  "
          f"({100*true_zeros_pruned/max(n_true_zeros,1):.1f}% of true zeros)")

    recall_overall    = 1.0 - true_edges_lost / max(n_true_edges, 1)
    precision_overall = (nonzero_mask & ~pruned_mask).sum() / max((~pruned_mask).sum(), 1)
    print(f"\n    Recall  (true edges surviving) : {recall_overall:.3f}")
    print(f"    Precision (surviving = true)   : {precision_overall:.3f}")

    # ------------------------------------------------------------------ #
    # 2. RECALL STRATIFIED BY TRUE EDGE WEIGHT BIN
    # ------------------------------------------------------------------ #
    print(f"\n[3] Recall by |A_true| bin (non-zero edges only)")
    A_true_nz   = A_true[nonzero_mask]
    A_hat_nz    = A_hat[nonzero_mask]
    abs_true_nz = np.abs(A_true_nz)
    bin_edges   = np.unique(
        np.percentile(abs_true_nz, np.linspace(0, 100, n_bins + 1))
    )

    recall_by_bin = []
    for i in range(len(bin_edges) - 1):
        lo, hi   = bin_edges[i], bin_edges[i+1]
        in_bin   = (abs_true_nz >= lo) & (abs_true_nz < hi)
        if i == len(bin_edges) - 2:
            in_bin = (abs_true_nz >= lo) & (abs_true_nz <= hi)
        n_bin    = in_bin.sum()
        if n_bin == 0:
            continue
        lost_bin = (in_bin & (np.abs(A_hat_nz) < prune_thresh)).sum()
        recall   = 1.0 - lost_bin / n_bin
        recall_by_bin.append((lo, hi, n_bin, lost_bin, recall))
        print(f"    |A_true| in [{lo:.3f}, {hi:.3f}] : "
              f"n={n_bin:5d}  lost={lost_bin:5d}  recall={recall:.3f}")

    # ------------------------------------------------------------------ #
    # 3. PER-MODE BREAKDOWN
    # ------------------------------------------------------------------ #
    if modes is not None:
        modes        = np.array(modes).ravel()
        unique_modes = np.unique(modes)
        print(f"\n[4] Per-mode recall and correlation")
        for m in unique_modes:
            m_mask  = modes == m
            nz_m    = nonzero_mask & m_mask
            n_nz_m  = nz_m.sum()
            if n_nz_m == 0:
                continue
            lost_m   = (nz_m & pruned_mask).sum()
            recall_m = 1.0 - lost_m / n_nz_m
            r_m      = (np.corrcoef(A_true[nz_m], A_hat[nz_m])[0, 1]
                        if n_nz_m > 1 else np.nan)
            print(f"    Mode {m}: n_true_edges={n_nz_m:5d}  "
                  f"lost={lost_m:5d}  recall={recall_m:.3f}  r={r_m:.3f}")

    # ------------------------------------------------------------------ #
    # 4. NLL PROXY DECOMPOSITION
    # ------------------------------------------------------------------ #
    print(f"\n[5] NLL proxy decomposition  (0.5 * (A_true - A_hat)^2)")
    nll_all     = 0.5 * (A_true - A_hat)**2
    nll_nonzero = nll_all[nonzero_mask].mean()
    nll_zero    = nll_all[zero_mask].mean()
    nll_lost    = (nll_all[nonzero_mask & pruned_mask].mean()
                   if (nonzero_mask & pruned_mask).any() else np.nan)
    print(f"    Mean NLL on true non-zero edges : {nll_nonzero:.4f}")
    print(f"    Mean NLL on true zero edges     : {nll_zero:.4f}")
    print(f"    Mean NLL on LOST true edges     : {nll_lost:.4f}  "
          f"(collapsed to ~0)")

    # ------------------------------------------------------------------ #
    # 5. FIGURES
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"L1 Diagnostics  —  {label}  prune_thresh={prune_thresh}", fontsize=13)

    # 5a. Precision-Recall curve
    ax = axes[0, 0]
    y_true_binary = nonzero_mask.astype(int)
    prec, rec, _ = precision_recall_curve(y_true_binary, np.abs(A_hat))
    pr_auc = auc(rec, prec)
    ax.plot(rec, prec)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall  (AUC={pr_auc:.3f})")
    ax.axvline(recall_overall, color='red', linestyle='--',
               label=f'current thresh recall={recall_overall:.2f}')
    ax.legend(fontsize=8)

    # 5b. A_hat histogram for true non-zero edges
    ax = axes[0, 1]
    ax.hist(A_hat[nonzero_mask], bins=60, color='steelblue',
            alpha=0.7, label='A_hat | Atrue≠0')
    ax.axvline( prune_thresh, color='red', linestyle='--',
               label=f'±thresh={prune_thresh}')
    ax.axvline(-prune_thresh, color='red', linestyle='--')
    ax.set_xlabel("A_hat")
    ax.set_title("A_hat distribution\n(true non-zero edges only)")
    ax.legend(fontsize=8)

    # 5c. A_hat histogram for true zero edges
    ax = axes[0, 2]
    ax.hist(A_hat[zero_mask], bins=60, color='orange',
            alpha=0.7, label='A_hat | Atrue=0')
    ax.axvline( prune_thresh, color='red', linestyle='--')
    ax.axvline(-prune_thresh, color='red', linestyle='--')
    ax.set_xlabel("A_hat")
    ax.set_title("A_hat distribution\n(true zero edges only)")
    ax.legend(fontsize=8)

    # 5d. Recall by bin bar chart
    ax = axes[1, 0]
    if recall_by_bin:
        bin_labels = [f"[{lo:.2f},{hi:.2f}]" for lo, hi, *_ in recall_by_bin]
        recalls    = [r for *_, r in recall_by_bin]
        ax.bar(range(len(recalls)), recalls,
               tick_label=bin_labels, color='steelblue')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Recall"); ax.set_xlabel("|A_true| bin")
        ax.set_title("True edge recall by weight bin")
        ax.tick_params(axis='x', labelrotation=30)

    # 5e. Scatter coloured by survived / lost
    ax = axes[1, 1]
    survived = nonzero_mask & ~pruned_mask
    lost     = nonzero_mask &  pruned_mask
    ax.scatter(A_true[zero_mask], A_hat[zero_mask],
               s=2, alpha=0.3, color='grey',  label='true zero')
    ax.scatter(A_true[survived],  A_hat[survived],
               s=4, alpha=0.5, color='green', label='survived true edge')
    ax.scatter(A_true[lost],      A_hat[lost],
               s=8, alpha=0.8, color='red',   label='LOST true edge')
    lim = np.max(np.abs(A_true)) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.8)
    ax.set_xlabel("A_true"); ax.set_ylabel("A_hat")
    ax.set_title("Scatter: survived vs lost true edges")
    ax.legend(fontsize=7, markerscale=2)

    # 5f. Per-mode recall bar
    ax = axes[1, 2]
    if modes is not None:
        mode_recalls, mode_labels = [], []
        for m in unique_modes:
            m_mask = modes == m
            nz_m   = nonzero_mask & m_mask
            if nz_m.sum() == 0:
                continue
            lost_m = (nz_m & pruned_mask).sum()
            mode_recalls.append(1.0 - lost_m / nz_m.sum())
            mode_labels.append(f"m={m}")
        ax.bar(range(len(mode_recalls)), mode_recalls,
               tick_label=mode_labels, color='steelblue')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Recall")
        ax.set_title("True edge recall per mode")
    else:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Figure saved to {save_path}")
    plt.show()
    print("=" * 60)


# ================================================================
#  Main
# ================================================================
def main():
    args = get_parser()
    np.set_printoptions(precision=3, suppress=True)

    fitFF = os.path.join(args.inpFit, f"{args.dataName}.stage3.npz")
    fitD, fitMD = read_data_npz(fitFF, verb=args.verb > 0)
    if args.verb > 1:
        print("\nstage3 metadata:"); pprint(fitMD)

    A_true     = fitD["A_true"].astype(np.float32)
    A_hat      = fitD["A_hat"].astype(np.float32)
    B_true     = fitD["B_true"].astype(np.float32)
    B_hat      = fitD["B_hat"].astype(np.float32)
    loss_epoch = fitD["loss_epoch"].astype(np.float32)
    nll_epoch  = fitD["nll_epoch"].astype(np.float32)
    pen_epoch  = fitD["pen_epoch"].astype(np.float32)

    M, N, _ = A_true.shape
    assert A_hat.shape[0]==M  # no handling of different sizes (what may happen)
    best_perm = max(
        permutations(range(M)),
        key=lambda p: sum(
            pearsonr(A_hat[p[m]].flatten(), A_true[m].flatten())[0]
            for m in range(M)
        )
    )
    if list(best_perm) != list(range(M)):
        print(f"  WARNING: reordering A_hat states {list(best_perm)} -> [0,1,...,M-1]")
    A_hat = A_hat[list(best_perm)]  # reorder to best match A_true


    if args.verb > 0:
        for m in range(M):
            r_A = pearson_r(A_true[m].ravel(), A_hat[m].ravel())
            r_B = pearson_r(B_true[m].ravel(), B_hat[m].ravel())
            rho_t = float(np.max(np.abs(np.linalg.eigvals(A_true[m]))))
            rho_h = float(np.max(np.abs(np.linalg.eigvals(A_hat[m]))))
            print(f"  state {m}: r_A={r_A:.4f}  r_B={r_B:.4f}  "
                  f"ρ_true={rho_t:.3f}  ρ_hat={rho_h:.3f}")

    # ---- canvas 1 ----
    fig1 = make_canvas1(fitMD, A_true, A_hat, B_true, B_hat)
    if args.plotFmt == "b":
        out1 = os.path.join(args.outPlots,
                             f"{args.dataName}_s3_canvas1.png")
        fig1.savefig(out1, bbox_inches="tight")
        print(f"  display {out1}")
    else:
        plt.show(block=not args.noBlock)
    plt.close(fig1)

    # ---- canvas 2 ----
    fig2, r_A_per, r_B_per = make_canvas2(
        fitMD, A_true, A_hat, B_true, B_hat,
        loss_epoch, nll_epoch, pen_epoch)
    if args.plotFmt == "b":
        out2 = os.path.join(args.outPlots,
                             f"{args.dataName}_s3_canvas2.png")
        fig2.savefig(out2, bbox_inches="tight")
        print(f"  display {out2}")
    else:
        plt.show(block=not args.noBlock)
    plt.close(fig2)

    # ---- L1 diagnostics ----
    # build pooled flat arrays and per-edge mode labels
    A_true_flat  = A_true.reshape(-1)           # (M*N*N,)
    A_hat_flat   = A_hat.reshape(-1)
    modes_flat   = np.concatenate(              # mode index for every edge
        [np.full(N * N, m, dtype=int) for m in range(M)]
    )
    lam_l1 = fitMD.get("lam_l1", None)         # None if not stored in metadata
    out_l1 = os.path.join(args.outPlots,
                           f"{args.dataName}_s3_l1diag.png")
    print_l1_diagnostics(
        A_true       = A_true_flat,
        A_hat        = A_hat_flat,
        modes        = modes_flat,
        lam_l1       = lam_l1,
        prune_thresh = args.prune_thresh,
        save_path    = out_l1,
    )

    # ---- terminal pass/fail ----
    print(f"\n{'='*54}")
    print(f" PRISM Stage 3 Pass/Fail  —  {args.dataName}")
    print(f"{'='*54}")
    all_pass = True
    for m in range(M):
        for label, val, thresh in [
            (f"A corr state {m}", r_A_per[m], THRESH_CORR_A),
            (f"B corr state {m}", r_B_per[m], THRESH_CORR_B),
        ]:
            ok  = val >= thresh
            all_pass &= ok
            sym = "✓" if ok else "✗"
            print(f"  {sym} {label}: {val:.5f}  (≥{thresh})")
    print(f"{'='*54}")
    print(f"  {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print(f"{'='*54}")


if __name__ == "__main__":
    main()
