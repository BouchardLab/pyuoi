#!/usr/bin/env python3
"""
PRISM Stage 1 - Forward Model Validation (eval/plots).

Reads stage1 fit results and compares predicted rates to observed spikes.
All comparisons use ground truth C_true, so failures here indicate
problems with the generative model or numerical implementation,
not with the optimizer.

Pass criteria:
  - Mean firing rate correlation (pred vs obs) > 0.95
  - Fano factor correlation                    > 0.80
  - Fraction of eta values clipped             < 0.01

Reads:
  <basePath>/prismFit/<dataName>_s1.stage1.npz
  <basePath>/spikesData/<dataName>.spikes.npz
  <basePath>/spikesData/<dataName>.prismTruth.npz

Writes:
  <basePath>/plots/<dataName>_s1.png   (single canvas)
"""

import os
import argparse
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from toolbox.Util_NumpyIO import read_data_npz


# Pass/fail thresholds
THRESH_RATE_CORR = 0.95
THRESH_FANO_CORR = 0.80
THRESH_CLIP_FRAC = 0.01


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        dest="verb", help="Verbosity level.")
    parser.add_argument("--basePath",
                        default="/dataVault2026/neurodata_tmp2",
                        help="Head dir for all data.")
    parser.add_argument("--dataName", default=None,
                        help="Spikes base name, e.g. daleN100_46f1c4_b90619")
    parser.add_argument("-p", "--plotFmt", default="b",
                        help="Plot format: b=batch(save png), s=screen.")
    parser.add_argument("-X", "--noBlock", action="store_true",
                        help="Non-blocking show (for batch runs).")

    args = parser.parse_args()
    args.inpFit    = os.path.join(args.basePath, "prismFit")
    args.inpSpikes = os.path.join(args.basePath, "spikesData")
    args.outPlots  = os.path.join(args.basePath, "plots")

    print("myArg-program:", parser.prog)
    for arg in vars(args):
        print("myArg:", arg, getattr(args, arg))

    assert args.dataName is not None, "must provide --dataName"
    assert os.path.exists(args.basePath), f"missing basePath: {args.basePath}"
    os.makedirs(args.outPlots, exist_ok=True)
    return args


def pearson_r(x, y):
    """Pearson correlation ignoring NaNs."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    xm = x[mask] - x[mask].mean()
    ym = y[mask] - y[mask].mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    return float(np.dot(xm, ym) / denom) if denom > 0 else np.nan


def compute_fano(spikes, min_rate=1e-6):
    """Per-neuron Fano factor: Var / Mean."""
    mu   = spikes.mean(axis=0)
    var  = spikes.var(axis=0)
    return np.where(mu > min_rate, var / mu, np.nan)


def pass_label(value, threshold, higher_is_better=True):
    """Return (label_str, color) for annotation."""
    ok = (value >= threshold) if higher_is_better else (value <= threshold)
    return ("PASS" if ok else "FAIL"), ("green" if ok else "red")


def make_canvas(fitD, fitMD, spikes, S_true, C_true, eta_clip, T_show=500):
    """
    Build single figure with 8 panels arranged in a 3-row grid:

    Row 0 (height 2): [raster observed | raster predicted | state trace]
    Row 1 (height 1): [rate scatter | fano scatter | deviance/time]
    Row 2 (height 1): [eta histogram | deviance/neuron hist | pass/fail text]
    """
    eta_t      = fitD["eta_t"]       # (T, N)
    lambda_t   = fitD["lambda_t"]    # (T, N)
    deviance_t = fitD["deviance_t"]  # (T,)
    deviance_n = fitD["deviance_n"]  # (N,)
    pred_rates = fitD["pred_rates"]  # (N,)
    obs_rates  = fitD["obs_rates"]   # (N,)

    T, N    = spikes.shape
    T_show  = min(T_show, T)
    n_show  = min(40, N)             # neurons shown in raster

    obs_fano  = compute_fano(spikes)
    pred_fano = compute_fano(lambda_t)
    r_rates   = pearson_r(pred_rates, obs_rates)
    r_fano    = pearson_r(pred_fano,  obs_fano)
    frac_clip = float((np.abs(eta_t.ravel()) >= eta_clip * 0.999).sum()) / eta_t.size

    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(17, 12))
    fig.suptitle(
        f"PRISM Stage 1 — Forward Model Validation\n{fitMD.get('short_name','')}   "
        f"T={T}  N={N}  M={fitMD.get('num_states','')}  "
        f"eta_clip=±{eta_clip}  dt={fitMD.get('time_step_sec','')}",
        fontsize=11, y=1.00
    )

    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[2.2, 1, 1],
                              hspace=0.42)

    # ---- Row 0: raster panels (3 columns) --------------------------
    row0 = gridspec.GridSpecFromSubplotSpec(
        3, 3, subplot_spec=outer[0],
        hspace=0.05, wspace=0.25,
        height_ratios=[3, 3, 0.8]
    )

    # Observed raster
    ax_robs = fig.add_subplot(row0[0, 0])
    t_idx, n_idx = np.where(spikes[:T_show, :n_show] > 0)
    ax_robs.scatter(t_idx, n_idx, s=1.2, color="black", alpha=0.55)
    ax_robs.set_xlim(0, T_show); ax_robs.set_ylim(-0.5, n_show - 0.5)
    ax_robs.set_ylabel("Neuron"); ax_robs.set_title("Observed spikes")
    ax_robs.set_xticklabels([])

    # Predicted rate heatmap
    ax_rpred = fig.add_subplot(row0[0, 1])
    im = ax_rpred.imshow(lambda_t[:T_show, :n_show].T,
                         aspect="auto", origin="lower",
                         extent=[0, T_show, 0, n_show],
                         cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax_rpred, label="λ (sp/bin)", pad=0.02)
    ax_rpred.set_title("Predicted rate λ_t")
    ax_rpred.set_xticklabels([])

    # Eta heatmap
    ax_reta = fig.add_subplot(row0[0, 2])
    im2 = ax_reta.imshow(eta_t[:T_show, :n_show].T,
                         aspect="auto", origin="lower",
                         extent=[0, T_show, 0, n_show],
                         cmap="RdBu_r", interpolation="nearest",
                         vmin=-eta_clip, vmax=eta_clip)
    plt.colorbar(im2, ax=ax_reta, label="η", pad=0.02)
    ax_reta.set_title("Internal potential η_t")
    ax_reta.set_xticklabels([])

    # Predicted rate below observed
    ax_pred2 = fig.add_subplot(row0[1, 0])
    lm_show = lambda_t[:T_show, :n_show]
    ax_pred2.imshow(lm_show.T, aspect="auto", origin="lower",
                    extent=[0, T_show, 0, n_show],
                    cmap="hot", interpolation="nearest")
    ax_pred2.set_ylabel("Neuron"); ax_pred2.set_xticklabels([])

    # C_true mixing coefficients
    ax_coef = fig.add_subplot(row0[1, 1])
    M = C_true.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 0.5, M))
    for m in range(M):
        ax_coef.plot(C_true[:T_show, m], lw=0.8,
                     color=colors[m], label=f"c_{m}", alpha=0.85)
    ax_coef.set_xlim(0, T_show)
    ax_coef.set_ylim(-0.05, 1.05)
    ax_coef.set_title("C_true mixing coefficients")
    ax_coef.legend(fontsize=7, loc="upper right", ncol=M)
    ax_coef.set_xticklabels([])

    # Empty top-right (placeholder)
    ax_empty = fig.add_subplot(row0[1, 2])
    ax_empty.axis("off")

    # State trace (spans all 3 columns of bottom row0 strip)
    ax_state = fig.add_subplot(row0[2, :])
    ax_state.step(np.arange(T_show), S_true[:T_show],
                  where="mid", color="steelblue", lw=1)
    ax_state.set_ylabel("State", fontsize=8)
    ax_state.set_xlabel(f"Time step  (first {T_show} of {T})", fontsize=8)
    ax_state.set_xlim(0, T_show)
    ax_state.yaxis.set_major_locator(MaxNLocator(integer=True))

    # ---- Row 1: scatter + deviance over time -----------------------
    row1 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1], wspace=0.35
    )

    # Rate scatter
    ax_rs = fig.add_subplot(row1[0])
    ax_rs.scatter(obs_rates, pred_rates, s=18, alpha=0.6, color="steelblue")
    lim = max(obs_rates.max(), pred_rates.max()) * 1.08
    ax_rs.plot([0, lim], [0, lim], "k--", lw=1)
    ax_rs.set_xlabel("Observed rate (sp/bin)")
    ax_rs.set_ylabel("Predicted rate (sp/bin)")
    ax_rs.set_title(f"Mean firing rate\nr = {r_rates:.4f}")
    lbl, col = pass_label(r_rates, THRESH_RATE_CORR)
    ax_rs.text(0.05, 0.90, lbl, transform=ax_rs.transAxes,
               color=col, fontweight="bold", fontsize=9)

    # Fano scatter
    ax_fs = fig.add_subplot(row1[1])
    ax_fs.scatter(obs_fano, pred_fano, s=18, alpha=0.6, color="darkorange")
    lim_f = np.nanmax([obs_fano, pred_fano]) * 1.08
    ax_fs.plot([0, lim_f], [0, lim_f], "k--", lw=1)
    ax_fs.set_xlabel("Observed Fano factor")
    ax_fs.set_ylabel("Predicted Fano factor")
    ax_fs.set_title(f"Fano factor\nr = {r_fano:.4f}")
    lbl, col = pass_label(r_fano, THRESH_FANO_CORR)
    ax_fs.text(0.05, 0.90, lbl, transform=ax_fs.transAxes,
               color=col, fontweight="bold", fontsize=9)

    # Deviance over time
    ax_dt = fig.add_subplot(row1[2])
    ax_dt.plot(deviance_t, lw=0.6, color="royalblue", alpha=0.8)
    ax_dt.set_xlabel("Time step t")
    ax_dt.set_ylabel("Poisson deviance")
    ax_dt.set_title(f"Deviance / time step\nmean={deviance_t.mean():.3f}  "
                    f"std={deviance_t.std():.3f}")

    # ---- Row 2: histograms + pass/fail text ------------------------
    row2 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[2], wspace=0.35
    )

    # Eta distribution
    ax_eh = fig.add_subplot(row2[0])
    eta_flat = eta_t.ravel()
    ax_eh.hist(eta_flat, bins=80, color="mediumpurple",
               edgecolor="white", alpha=0.85)
    ax_eh.axvline( eta_clip, color="red", lw=1.5, ls="--",
                  label=f"+clip={eta_clip}")
    ax_eh.axvline(-eta_clip, color="red", lw=1.5, ls="--",
                  label=f"-clip={eta_clip}")
    ax_eh.set_xlabel("η (internal potential)")
    ax_eh.set_ylabel("Count")
    ax_eh.set_title(f"η distribution\n{frac_clip*100:.3f}% clipped")
    ax_eh.legend(fontsize=7)
    lbl, col = pass_label(frac_clip, THRESH_CLIP_FRAC,
                          higher_is_better=False)
    ax_eh.text(0.05, 0.90, lbl, transform=ax_eh.transAxes,
               color=col, fontweight="bold", fontsize=9)

    # Per-neuron deviance histogram
    ax_dn = fig.add_subplot(row2[1])
    dev_per_step = deviance_n / T
    ax_dn.hist(dev_per_step, bins=30, color="royalblue",
               edgecolor="white", alpha=0.85)
    ax_dn.set_xlabel("Mean deviance/step (per neuron)")
    ax_dn.set_ylabel("Neuron count")
    ax_dn.set_title(f"Per-neuron deviance\nmean={dev_per_step.mean():.4f}  "
                    f"std={dev_per_step.std():.4f}")

    # Pass/fail summary text panel
    ax_pf = fig.add_subplot(row2[2])
    ax_pf.axis("off")

    total_dev = float(deviance_t.sum())
    lines = [
        ("PRISM Stage 1 Summary", None, "black", 11, True),
        ("", None, "black", 9, False),
        (f"T={T}  N={N}  M={fitMD.get('num_states','')}",
         None, "black", 9, False),
        (f"eta_clip = ±{eta_clip}",
         None, "black", 9, False),
        (f"Total deviance: {total_dev:.1f}",
         None, "black", 9, False),
        (f"Deviance/step:  {total_dev/T:.4f}",
         None, "black", 9, False),
        (f"Deviance/N/step: {total_dev/(T*N):.5f}",
         None, "black", 9, False),
        ("", None, "black", 9, False),
        (f"Rate corr:  {r_rates:.4f}  (≥{THRESH_RATE_CORR})",
         *pass_label(r_rates, THRESH_RATE_CORR), 9, True),
        (f"Fano corr:  {r_fano:.4f}  (≥{THRESH_FANO_CORR})",
         *pass_label(r_fano,  THRESH_FANO_CORR), 9, True),
        (f"Clip frac:  {frac_clip:.4f}  (≤{THRESH_CLIP_FRAC})",
         *pass_label(frac_clip, THRESH_CLIP_FRAC,
                     higher_is_better=False), 9, True),
    ]
    y = 0.97
    for row in lines:
        txt, color, fs, bold = row[0], row[1], row[2], row[3]
        fw = "bold" if bold else "normal"
        ax_pf.text(0.05, y, txt, transform=ax_pf.transAxes,
                   fontsize=fs, color=color, fontweight=fw,
                   verticalalignment="top", family="monospace")
        y -= 0.10

    return fig, r_rates, r_fano, frac_clip


def main():
    args = get_parser()
    np.set_printoptions(precision=3, suppress=True)

    # ---- load stage1 fit ----
    fitFF = os.path.join(args.inpFit, f"{args.dataName}_s1.stage1.npz")
    fitD, fitMD = read_data_npz(fitFF, verb=args.verb > 0)
    if args.verb > 1:
        print("\nstage1 metadata:"); pprint(fitMD)

    # ---- load spikes ----
    spikesFF = os.path.join(args.inpSpikes, f"{args.dataName}.spikes.npz")
    spikesD, spikesMD = read_data_npz(spikesFF, verb=args.verb > 0)

    # ---- load prismTruth ----
    prismTruthFF = os.path.join(args.inpSpikes,
                                f"{args.dataName}.prismTruth.npz")
    prismTruthD, prismTruthMD = read_data_npz(prismTruthFF,
                                              verb=args.verb > 0)

    spikes = spikesD["spikes"].astype(np.float32)   # (T, N)
    S_true = prismTruthD["S_true"]                  # (T,)
    C_true = prismTruthD["C_true"]                  # (T, M)
    eta_clip = float(spikesMD["poisson_eta_clip"])

    T, N = spikes.shape
    if args.verb > 0:
        print(f"\nLoaded: T={T}, N={N}  eta_clip={eta_clip}")
        print(f"Total deviance : {fitMD['total_deviance']:.4f}")
        print(f"Per step       : {fitMD['total_deviance']/T:.4f}")
        print(f"Per neuron*step: {fitMD['total_deviance']/(T*N):.6f}")

    # ---- convert fit arrays to numpy ----
    fitD_np = {k: v if isinstance(v, np.ndarray) else np.array(v)
               for k, v in fitD.items()}

    # ---- build canvas ----
    fig, r_rates, r_fano, frac_clip = make_canvas(
        fitD_np, fitMD, spikes, S_true, C_true,
        eta_clip=eta_clip, T_show=500
    )

    # ---- save or show ----
    if args.plotFmt == "b":
        outFF = os.path.join(args.outPlots, f"{args.dataName}_s1.png")
        fig.savefig(outFF, bbox_inches="tight", dpi=130)
        print(f"\n  saved: {outFF}")
    else:
        plt.show(block=not args.noBlock)
    plt.close(fig)

    # ---- terminal pass/fail ----
    print(f"\n{'='*50}")
    print(f" PRISM Stage 1 Pass/Fail  —  {args.dataName}")
    print(f"{'='*50}")
    for label, val, thresh, higher in [
        ("Rate corr ", r_rates,   THRESH_RATE_CORR, True),
        ("Fano corr ", r_fano,    THRESH_FANO_CORR, True),
        ("Clip frac ", frac_clip, THRESH_CLIP_FRAC, False),
    ]:
        ok  = (val >= thresh) if higher else (val <= thresh)
        sym = "✓" if ok else "✗"
        print(f"  {sym} {label}: {val:.5f}  "
              f"({'≥' if higher else '≤'}{thresh})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
