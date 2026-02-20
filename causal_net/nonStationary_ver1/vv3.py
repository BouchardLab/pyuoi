#!/usr/bin/env python3
"""
PRISM Stage 1 - Forward Model Validation (eval/plots).

Produces two PNG canvases:
  <dataName>_s1_canvas1.png  : time-domain panels (rasters, traces)
  <dataName>_s1_canvas2.png  : statistics panels  (scatters, histograms, summary)

Reads:
  <basePath>/prismFit/<dataName>_s1.stage1.npz
  <basePath>/spikesData/<dataName>.spikes.npz
  <basePath>/spikesData/<dataName>.prismTruth.npz
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
    parser.add_argument("--T_show", type=int, default=1000,
                        help="Number of time steps shown in raster panels.")
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


def compute_fano(arr, min_rate=1e-6):
    """Per-neuron Fano factor: Var / Mean."""
    mu  = arr.mean(axis=0)
    var = arr.var(axis=0)
    return np.where(mu > min_rate, var / mu, np.nan)


def pass_label(value, threshold, higher_is_better=True):
    """Return (label_str, color) for annotation."""
    ok = (value >= threshold) if higher_is_better else (value <= threshold)
    return ("PASS" if ok else "FAIL"), ("green" if ok else "red")


def make_title(tag, fitMD, eta_clip):
    """Title without total T - shows only displayed range."""
    return (f"PRISM Stage 1 — {tag}\n"
            f"{fitMD.get('short_name','')}   "
            f"N={fitMD.get('num_neurons','')}  "
            f"M={fitMD.get('num_states','')}  "
            f"eta_clip=±{eta_clip}  "
            f"dt={fitMD.get('time_step_sec','')} sec")

# ================================================================
#  Canvas 1 : time-domain panels
# ================================================================
def make_canvas1(fitD, fitMD, spikes, S_true, C_true, eta_clip, T_show):
    """
    Layout (4 rows):
      Row 0 (tall) : observed spike raster  - all neurons
      Row 1 (tall) : predicted rate heatmap | eta heatmap  - all neurons
      Row 2 (med)  : C_true mixing coefficients
      Row 3 (thin) : true state trace
    All panels share the same time axis [0, T_show].
    Heatmap z-ranges are locked and IDENTICAL in scale:
      lambda_t : [0,  lam_vmax]       95th percentile
      eta_t    : [-eta_clip, +eta_clip]
    Both colorbars use the same number of ticks for visual consistency.
    """
    lambda_t = fitD["lambda_t"]   # (T, N)
    eta_t    = fitD["eta_t"]      # (T, N)

    T, N   = spikes.shape
    T_show = min(T_show, T)
    M      = C_true.shape[1]
    dt     = float(fitMD.get("time_step_sec", 0.01))

    # ---- locked z-ranges ----------------------------------------
    # lambda: physical range [0, 95th pct] - no negative values
    lam_vmax  = float(np.percentile(lambda_t[:T_show, :], 95))
    lam_vmin  = 0.0
    # eta: symmetric, locked exactly to ±eta_clip
    eta_vmin  = -eta_clip
    eta_vmax  =  eta_clip
    # shared colorbar tick count
    N_TICKS   = 5

    # ---- shared x-tick positions --------------------------------
    x_ticks = np.linspace(0, T_show, 6, dtype=int)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(make_title("Time-Domain Panels", fitMD, eta_clip),
                 fontsize=10, y=1.00)

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           height_ratios=[2.6, 2.6, 1.3, 0.65],
                           hspace=0.50, wspace=0.28)

    # ----------------------------------------------------------------
    # Row 0: observed raster (spans both columns, all neurons)
    # ----------------------------------------------------------------
    ax_obs = fig.add_subplot(gs[0, :])
    t_idx, n_idx = np.where(spikes[:T_show, :N] > 0)
    ax_obs.scatter(t_idx, n_idx, s=0.8, color="black", alpha=0.45)
    ax_obs.set_xlim(0, T_show)
    ax_obs.set_ylim(-0.5, N - 0.5)
    ax_obs.set_ylabel("Neuron index")
    ax_obs.set_title(f"Observed spikes  (all {N} neurons)")
    # E/I boundary heuristic
    mean_rate = spikes[:T_show, :].mean(axis=0)
    boundary  = int(np.sum(mean_rate > mean_rate.mean()))
    ax_obs.axhline(boundary, color="red", lw=0.8, ls="--", alpha=0.6,
                   label=f"rate boundary ~{boundary}")
    ax_obs.legend(fontsize=7, loc="upper right")
    ax_obs.set_xticks(x_ticks)
    ax_obs.set_xticklabels([str(v) for v in x_ticks], fontsize=8)
    ax_obs.set_xlabel("Time step", fontsize=8)

    # ----------------------------------------------------------------
    # Row 1 left: predicted rate heatmap - locked z [lam_vmin, lam_vmax]
    # ----------------------------------------------------------------
    ax_lam = fig.add_subplot(gs[1, 0])
    im1 = ax_lam.imshow(lambda_t[:T_show, :N].T,
                        aspect="auto", origin="lower",
                        extent=[0, T_show, 0, N],
                        cmap="hot", interpolation="nearest",
                        vmin=lam_vmin, vmax=lam_vmax)
    cb1 = plt.colorbar(im1, ax=ax_lam, label="λ (sp/bin)", pad=0.02)
    cb1.set_ticks(np.linspace(lam_vmin, lam_vmax, N_TICKS))
    cb1.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax_lam.set_ylabel("Neuron index")
    ax_lam.set_title(f"Predicted rate  λ_t   "
                     f"z=[{lam_vmin:.2f}, {lam_vmax:.2f}]  (95th pct)")
    ax_lam.set_xticks(x_ticks)
    ax_lam.set_xticklabels([str(v) for v in x_ticks], fontsize=8)
    ax_lam.set_xlabel("Time step", fontsize=8)

    # ----------------------------------------------------------------
    # Row 1 right: eta heatmap - locked to [-eta_clip, +eta_clip]
    # ----------------------------------------------------------------
    ax_eta = fig.add_subplot(gs[1, 1])
    im2 = ax_eta.imshow(eta_t[:T_show, :N].T,
                        aspect="auto", origin="lower",
                        extent=[0, T_show, 0, N],
                        cmap="RdBu_r", interpolation="nearest",
                        vmin=eta_vmin, vmax=eta_vmax)
    cb2 = plt.colorbar(im2, ax=ax_eta, label="η", pad=0.02)
    cb2.set_ticks(np.linspace(eta_vmin, eta_vmax, N_TICKS))
    cb2.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax_eta.set_ylabel("Neuron index")
    ax_eta.set_title(f"Internal potential  η_t   "
                     f"z=[{eta_vmin:.1f}, {eta_vmax:.1f}]  (locked ±eta_clip)")
    ax_eta.set_xticks(x_ticks)
    ax_eta.set_xticklabels([str(v) for v in x_ticks], fontsize=8)
    ax_eta.set_xlabel("Time step", fontsize=8)

    # ----------------------------------------------------------------
    # Row 2: mixing coefficients (spans both columns)
    # ----------------------------------------------------------------
    ax_c = fig.add_subplot(gs[2, :])
    colors = plt.cm.tab10(np.linspace(0, 0.5, M))
    for m in range(M):
        ax_c.plot(C_true[:T_show, m], lw=1.1,
                  color=colors[m], label=f"c_{m}", alpha=0.85)
    ax_c.set_xlim(0, T_show)
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.set_ylabel("Coefficient")
    ax_c.set_title("C_true  mixing coefficients")
    ax_c.legend(fontsize=8, loc="upper right", ncol=M)
    ax_c.set_xticks(x_ticks)
    ax_c.set_xticklabels([str(v) for v in x_ticks], fontsize=8)
    ax_c.set_xlabel("Time step", fontsize=8)

    # ----------------------------------------------------------------
    # Row 3: true state trace (spans both columns)
    # ----------------------------------------------------------------
    ax_s = fig.add_subplot(gs[3, :])
    ax_s.step(np.arange(T_show), S_true[:T_show],
              where="mid", color="steelblue", lw=1.2)
    ax_s.set_xlim(0, T_show)
    ax_s.set_ylim(-0.3, int(S_true.max()) + 0.5)
    ax_s.set_ylabel("State", fontsize=8)
    ax_s.set_xticks(x_ticks)
    ax_s.set_xticklabels([str(v) for v in x_ticks], fontsize=8)
    ax_s.set_xlabel(
        f"Time step  "
        f"(showing {T_show} bins = {T_show * dt:.1f} sec  "
        f"of {T} total)",
        fontsize=9)
    ax_s.yaxis.set_major_locator(MaxNLocator(integer=True))

    # secondary axis in seconds
    ax_s2 = ax_s.twiny()
    ax_s2.set_xlim(0, T_show * dt)
    ax_s2.set_xlabel("Time (sec)", fontsize=8)
    ax_s2.xaxis.set_major_locator(plt.MaxNLocator(6))

    return fig
# ================================================================
#  Canvas 2 : statistics panels
# ================================================================
def make_canvas2(fitD, fitMD, spikes, eta_clip):
    """
    Layout (2 rows x 3 cols):
      Row 0: rate scatter | fano scatter | deviance over time
      Row 1: eta histogram | per-neuron deviance hist | pass/fail text
    """
    lambda_t   = fitD["lambda_t"]    # (T, N)
    deviance_t = fitD["deviance_t"]  # (T,)
    deviance_n = fitD["deviance_n"]  # (N,)
    pred_rates = fitD["pred_rates"]  # (N,)
    obs_rates  = fitD["obs_rates"]   # (N,)
    eta_t      = fitD["eta_t"]       # (T, N)

    T, N = spikes.shape

    obs_fano  = compute_fano(spikes)
    pred_fano = compute_fano(lambda_t)
    r_rates   = pearson_r(pred_rates, obs_rates)
    r_fano    = pearson_r(pred_fano,  obs_fano)
    frac_clip = float(
        (np.abs(eta_t.ravel()) >= eta_clip * 0.999).sum()
    ) / eta_t.size

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(make_title("Statistics Panels", fitMD, eta_clip),
                 fontsize=10, y=1.00)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.42, wspace=0.35)

    # ---- (0,0) Rate scatter ----
    ax_rs = fig.add_subplot(gs[0, 0])
    ax_rs.scatter(obs_rates, pred_rates, s=20, alpha=0.65,
                  color="steelblue", zorder=3)
    lim = max(obs_rates.max(), pred_rates.max()) * 1.08
    ax_rs.plot([0, lim], [0, lim], "k--", lw=1)
    ax_rs.set_xlabel("Observed rate (sp/bin)")
    ax_rs.set_ylabel("Predicted rate (sp/bin)")
    ax_rs.set_title(f"Mean firing rate\nr = {r_rates:.4f}")
    lbl, col = pass_label(r_rates, THRESH_RATE_CORR)
    ax_rs.text(0.05, 0.90, lbl, transform=ax_rs.transAxes,
               color=col, fontweight="bold", fontsize=10)

    # ---- (0,1) Fano scatter ----
    ax_fs = fig.add_subplot(gs[0, 1])
    ax_fs.scatter(obs_fano, pred_fano, s=20, alpha=0.65,
                  color="darkorange", zorder=3)
    lim_f = np.nanmax([obs_fano, pred_fano]) * 1.08
    ax_fs.plot([0, lim_f], [0, lim_f], "k--", lw=1)
    ax_fs.set_xlabel("Observed Fano factor")
    ax_fs.set_ylabel("Predicted Fano factor")
    ax_fs.set_title(f"Fano factor\nr = {r_fano:.4f}")
    lbl, col = pass_label(r_fano, THRESH_FANO_CORR)
    ax_fs.text(0.05, 0.90, lbl, transform=ax_fs.transAxes,
               color=col, fontweight="bold", fontsize=10)

    # ---- (0,2) Deviance over time ----
    ax_dt = fig.add_subplot(gs[0, 2])
    ax_dt.plot(deviance_t, lw=0.6, color="royalblue", alpha=0.8)
    ax_dt.set_xlabel("Time step t")
    ax_dt.set_ylabel("Poisson deviance")
    ax_dt.set_title(f"Deviance / time step\n"
                    f"mean={deviance_t.mean():.3f}  "
                    f"std={deviance_t.std():.3f}")

    # ---- (1,0) Eta histogram ----
    ax_eh = fig.add_subplot(gs[1, 0])
    eta_flat = eta_t.ravel()
    ax_eh.hist(eta_flat, bins=80, color="mediumpurple",
               edgecolor="white", alpha=0.85)
    ax_eh.axvline( eta_clip, color="red", lw=1.8, ls="--",
                  label=f"+clip={eta_clip}")
    ax_eh.axvline(-eta_clip, color="red", lw=1.8, ls="--",
                  label=f"-clip={eta_clip}")
    ax_eh.set_xlabel("η (internal potential)")
    ax_eh.set_ylabel("Count")
    ax_eh.set_title(f"η distribution\n{frac_clip*100:.3f}% clipped")
    ax_eh.legend(fontsize=8)
    lbl, col = pass_label(frac_clip, THRESH_CLIP_FRAC,
                          higher_is_better=False)
    ax_eh.text(0.05, 0.90, lbl, transform=ax_eh.transAxes,
               color=col, fontweight="bold", fontsize=10)

    # ---- (1,1) Per-neuron deviance histogram ----
    ax_dn = fig.add_subplot(gs[1, 1])
    dev_per_step = deviance_n / T
    ax_dn.hist(dev_per_step, bins=30, color="royalblue",
               edgecolor="white", alpha=0.85)
    ax_dn.set_xlabel("Mean deviance/step (per neuron)")
    ax_dn.set_ylabel("Neuron count")
    ax_dn.set_title(f"Per-neuron deviance\n"
                    f"mean={dev_per_step.mean():.4f}  "
                    f"std={dev_per_step.std():.4f}")

    # ---- (1,2) Pass/fail text ----
    ax_pf = fig.add_subplot(gs[1, 2])
    ax_pf.axis("off")

    total_dev = float(deviance_t.sum())
    rows = [
        # (text, color, fontsize, bold)
        ("PRISM Stage 1 Summary",         "black", 11, True),
        ("",                               "black",  9, False),
        (f"T={T}  N={N}  "
         f"M={fitMD.get('num_states','')}","black",  9, False),
        (f"eta_clip = ±{eta_clip}",        "black",  9, False),
        (f"dt = {fitMD.get('time_step_sec','')} sec",
                                           "black",  9, False),
        ("",                               "black",  9, False),
        (f"Total deviance : {total_dev:.1f}",
                                           "black",  9, False),
        (f"Deviance/step  : {total_dev/T:.4f}",
                                           "black",  9, False),
        (f"Dev/N/step     : {total_dev/(T*N):.5f}",
                                           "black",  9, False),
        ("",                               "black",  9, False),
        (f"Rate corr : {r_rates:.4f}  "
         f"(≥{THRESH_RATE_CORR})",
         pass_label(r_rates, THRESH_RATE_CORR)[1], 9, True),
        (f"Fano corr : {r_fano:.4f}  "
         f"(≥{THRESH_FANO_CORR})",
         pass_label(r_fano,  THRESH_FANO_CORR)[1], 9, True),
        (f"Clip frac : {frac_clip:.4f}  "
         f"(≤{THRESH_CLIP_FRAC})",
         pass_label(frac_clip, THRESH_CLIP_FRAC,
                    higher_is_better=False)[1], 9, True),
    ]
    y = 0.97
    for txt, color, fs, bold in rows:
        ax_pf.text(0.05, y, txt,
                   transform=ax_pf.transAxes,
                   fontsize=fs, color=color,
                   fontweight="bold" if bold else "normal",
                   verticalalignment="top",
                   family="monospace")
        y -= 0.09

    return fig, r_rates, r_fano, frac_clip


# ================================================================
#  Main
# ================================================================
def main():
    args = get_parser()
    np.set_printoptions(precision=3, suppress=True)

    # ---- load files ----
    fitFF = os.path.join(args.inpFit,
                         f"{args.dataName}_s1.stage1.npz")
    fitD, fitMD = read_data_npz(fitFF, verb=args.verb > 0)
    if args.verb > 1:
        print("\nstage1 metadata:"); pprint(fitMD)

    spikesFF = os.path.join(args.inpSpikes,
                             f"{args.dataName}.spikes.npz")
    spikesD, spikesMD = read_data_npz(spikesFF, verb=args.verb > 0)

    prismTruthFF = os.path.join(args.inpSpikes,
                                 f"{args.dataName}.prismTruth.npz")
    prismTruthD, _ = read_data_npz(prismTruthFF, verb=args.verb > 0)

    spikes   = spikesD["spikes"].astype(np.float32)  # (T, N)
    S_true   = prismTruthD["S_true"]                 # (T,)
    C_true   = prismTruthD["C_true"]                 # (T, M)
    eta_clip = float(spikesMD["poisson_eta_clip"])

    T, N = spikes.shape
    if args.verb > 0:
        print(f"\nLoaded: T={T}, N={N}  eta_clip={eta_clip}")
        print(f"Total deviance : {fitMD['total_deviance']:.4f}")
        print(f"Per step       : {fitMD['total_deviance']/T:.4f}")
        print(f"Per neuron*step: {fitMD['total_deviance']/(T*N):.6f}")

    # ensure numpy
    fitD_np = {k: (v if isinstance(v, np.ndarray) else np.array(v))
               for k, v in fitD.items()}

    # ---- Canvas 1 : time-domain ----
    fig1 = make_canvas1(fitD_np, fitMD, spikes, S_true, C_true,
                        eta_clip, args.T_show)
    if args.plotFmt == "b":
        out1 = os.path.join(args.outPlots,
                             f"{args.dataName}_s1_canvas1.png")
        fig1.savefig(out1, bbox_inches="tight")
        print(f"  saved: {out1}")
    else:
        plt.show(block=not args.noBlock)
    plt.close(fig1)

    # ---- Canvas 2 : statistics ----
    fig2, r_rates, r_fano, frac_clip = make_canvas2(
        fitD_np, fitMD, spikes, eta_clip)
    if args.plotFmt == "b":
        out2 = os.path.join(args.outPlots,
                             f"{args.dataName}_s1_canvas2.png")
        fig2.savefig(out2, bbox_inches="tight")
        print(f"  saved: {out2}")
    else:
        plt.show(block=not args.noBlock)
    plt.close(fig2)

    # ---- terminal pass/fail ----
    print(f"\n{'='*52}")
    print(f" PRISM Stage 1 Pass/Fail  —  {args.dataName}")
    print(f"{'='*52}")
    for label, val, thresh, higher in [
        ("Rate corr ", r_rates,   THRESH_RATE_CORR, True),
        ("Fano corr ", r_fano,    THRESH_FANO_CORR, True),
        ("Clip frac ", frac_clip, THRESH_CLIP_FRAC, False),
    ]:
        ok  = (val >= thresh) if higher else (val <= thresh)
        sym = "✓" if ok else "✗"
        print(f"  {sym} {label}: {val:.5f}  "
              f"({'≥' if higher else '≤'}{thresh})")
    

if __name__ == "__main__":
    main()
