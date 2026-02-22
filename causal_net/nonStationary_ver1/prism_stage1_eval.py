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
THRESH_CLIP_FRAC = 0.02

# Stable-state threshold: a time step is stable if max(C_true[t]) > this
STABLE_THRESHOLD = 0.95


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
    parser.add_argument("--stable_thresh", type=float,
                        default=STABLE_THRESHOLD,
                        help="Min max(C_true[t]) to be considered stable.")
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
    """Per-neuron Fano factor: Var / Mean.  arr shape: (T, N)"""
    mu  = arr.mean(axis=0)
    var = arr.var(axis=0)
    return np.where(mu > min_rate, var / mu, np.nan)


def compute_stable_mask(C_true, threshold):
    """
    Return boolean mask of shape (T,) marking stable (non-transitioning) steps.

    A time step t is stable if the dominant coefficient exceeds `threshold`:
        stable[t] = max_m C_true[t, m] > threshold

    Args:
        C_true    : (T, M) float array
        threshold : float in (0, 1], e.g. 0.95

    Returns:
        stable_mask : (T,) bool array
        trans_mask  : (T,) bool array  (complement)
    """
    stable_mask = C_true.max(axis=1) > threshold
    trans_mask  = ~stable_mask
    return stable_mask, trans_mask


def pass_label(value, threshold, higher_is_better=True):
    """Return (label_str, color) for annotation."""
    ok = (value >= threshold) if higher_is_better else (value <= threshold)
    return ("PASS" if ok else "FAIL"), ("green" if ok else "red")


def make_title(tag, fitMD, eta_clip):
    """Title without total T."""
    return (f"PRISM Stage 1 — {tag}\n"
            f"{fitMD['short_name']}   "
            f"N={fitMD['num_neurons']}  "
            f"M={fitMD['num_states']}  "
            f"eta_clip=±{eta_clip}  "
            f"dt={fitMD['time_step_sec']} sec")


# ================================================================
#  Canvas 1 : time-domain panels
# ================================================================
def make_canvas1(fitD, fitMD, spikes, S_true, C_true,
                 stable_mask, eta_clip, T_show):
    """
    Layout (4 rows):
      Row 0 (tall) : observed spike raster  - all neurons
                     transitioning bins shaded in background
      Row 1 (tall) : predicted rate heatmap | eta heatmap
      Row 2 (med)  : C_true mixing coefficients + transition shading
      Row 3 (thin) : true state trace
    """
    lambda_t = fitD["lambda_t"]   # (T, N)
    eta_t    = fitD["eta_t"]      # (T, N)

    T, N   = spikes.shape
    T_show = min(T_show, T)
    M      = C_true.shape[1]
    dt     = float(fitMD["time_step_sec"])

    lam_vmax = float(np.percentile(lambda_t[:T_show, :], 95))
    lam_vmin = 0.0
    eta_vmin = -eta_clip
    eta_vmax =  eta_clip
    N_TICKS  = 5
    x_ticks  = np.linspace(0, T_show, 6, dtype=int)

    # transition intervals for shading (contiguous runs of trans_mask)
    trans_shown = ~stable_mask[:T_show]

    def shade_transitions(ax):
        """Shade transitioning bins in light gray on any axis."""
        in_trans  = False
        t_start   = 0
        for t in range(T_show):
            if trans_shown[t] and not in_trans:
                t_start  = t
                in_trans = True
            elif not trans_shown[t] and in_trans:
                ax.axvspan(t_start, t, color="gray", alpha=0.18, lw=0)
                in_trans = False
        if in_trans:
            ax.axvspan(t_start, T_show, color="gray", alpha=0.18, lw=0)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(make_title("Time-Domain Panels", fitMD, eta_clip),
                 fontsize=10, y=1.00)

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           height_ratios=[2.6, 2.6, 1.3, 0.65],
                           hspace=0.50, wspace=0.28)

    # ---- Row 0: observed raster ----
    ax_obs = fig.add_subplot(gs[0, :])
    t_idx, n_idx = np.where(spikes[:T_show, :N] > 0)
    ax_obs.scatter(t_idx, n_idx, s=0.8, color="black", alpha=0.45, zorder=2)
    shade_transitions(ax_obs)
    ax_obs.set_xlim(0, T_show)
    ax_obs.set_ylim(-0.5, N - 0.5)
    ax_obs.set_ylabel("Neuron index")
    n_trans = int(trans_shown.sum())
    ax_obs.set_title(
        f"Observed spikes  (all {N} neurons)   "
        f"gray = transitioning bins  "
        f"({n_trans}/{T_show} = {100*n_trans/T_show:.1f}%)")
    mean_rate = spikes[:T_show, :].mean(axis=0)
    boundary  = int(np.sum(mean_rate > mean_rate.mean()))
    ax_obs.axhline(boundary, color="red", lw=0.8, ls="--", alpha=0.6,
                   label=f"rate boundary ~{boundary}", zorder=3)
    ax_obs.legend(fontsize=7, loc="upper right")
    ax_obs.set_xticks(x_ticks)
    ax_obs.set_xticklabels([str(v) for v in x_ticks], fontsize=8)
    ax_obs.set_xlabel("Time step", fontsize=8)

    # ---- Row 1 left: lambda heatmap ----
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
    shade_transitions(ax_lam)
    ax_lam.set_ylabel("Neuron index")
    ax_lam.set_title(f"Predicted rate  λ_t   "
                     f"z=[{lam_vmin:.2f}, {lam_vmax:.2f}]  (95th pct)")
    ax_lam.set_xticks(x_ticks)
    ax_lam.set_xticklabels([str(v) for v in x_ticks], fontsize=8)
    ax_lam.set_xlabel("Time step", fontsize=8)

    # ---- Row 1 right: eta heatmap ----
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
    shade_transitions(ax_eta)
    ax_eta.set_ylabel("Neuron index")
    ax_eta.set_title(f"Internal potential  η_t   "
                     f"z=[{eta_vmin:.1f}, {eta_vmax:.1f}]  (locked ±eta_clip)")
    ax_eta.set_xticks(x_ticks)
    ax_eta.set_xticklabels([str(v) for v in x_ticks], fontsize=8)
    ax_eta.set_xlabel("Time step", fontsize=8)

    # ---- Row 2: mixing coefficients ----
    ax_c = fig.add_subplot(gs[2, :])
    colors = plt.cm.tab10(np.linspace(0, 0.5, M))
    for m in range(M):
        ax_c.plot(C_true[:T_show, m], lw=1.1,
                  color=colors[m], label=f"c_{m}", alpha=0.85)
    # threshold line showing stable criterion
    ax_c.axhline(STABLE_THRESHOLD, color="gray", lw=0.8, ls=":",
                 label=f"stable thresh={STABLE_THRESHOLD}")
    shade_transitions(ax_c)
    ax_c.set_xlim(0, T_show)
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.set_ylabel("Coefficient")
    ax_c.set_title("C_true  mixing coefficients")
    ax_c.legend(fontsize=7, loc="upper right", ncol=M + 1)
    ax_c.set_xticks(x_ticks)
    ax_c.set_xticklabels([str(v) for v in x_ticks], fontsize=8)
    ax_c.set_xlabel("Time step", fontsize=8)

    # ---- Row 3: state trace ----
    ax_s = fig.add_subplot(gs[3, :])
    ax_s.step(np.arange(T_show), S_true[:T_show],
              where="mid", color="steelblue", lw=1.2)
    shade_transitions(ax_s)
    ax_s.set_xlim(0, T_show)
    ax_s.set_ylim(-0.3, int(S_true.max()) + 0.5)
    ax_s.set_ylabel("State", fontsize=8)
    ax_s.set_xticks(x_ticks)
    ax_s.set_xticklabels([str(v) for v in x_ticks], fontsize=8)
    ax_s.set_xlabel(
        f"Time step  "
        f"(showing {T_show} bins = {T_show * dt:.1f} sec  "
        f"of {fitMD['num_steps']} total)",
        fontsize=9)
    ax_s.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_s2 = ax_s.twiny()
    ax_s2.set_xlim(0, T_show * dt)
    ax_s2.set_xlabel("Time (sec)", fontsize=8)
    ax_s2.xaxis.set_major_locator(plt.MaxNLocator(6))

    return fig


# ================================================================
#  Canvas 2 : statistics panels
# ================================================================
def make_canvas2(fitD, fitMD, spikes, C_true, stable_mask,
                 eta_clip, stable_thresh):
    """
    Layout (2 rows x 3 cols):
      Row 0: rate scatter (all | stable) | fano scatter (all | stable)
             | deviance over time with transition shading
      Row 1: eta histogram | per-neuron deviance hist | pass/fail text

    Statistics are computed twice:
      - ALL bins
      - STABLE bins only (non-transitioning)
    Both are shown on the scatter plots as different marker styles.
    Pass/fail is judged on stable-only metrics.
    """
    lambda_t   = fitD["lambda_t"]    # (T, N)
    deviance_t = fitD["deviance_t"]  # (T,)
    deviance_n = fitD["deviance_n"]  # (N,)
    pred_rates = fitD["pred_rates"]  # (N,)  - full T average
    obs_rates  = fitD["obs_rates"]   # (N,)  - full T average
    eta_t      = fitD["eta_t"]       # (T, N)

    T, N = spikes.shape
    M    = C_true.shape[1]

    # ---- stable-only arrays ----
    # rates and fano recomputed on stable bins only
    lam_stable   = lambda_t[stable_mask]     # (T_s, N)
    spk_stable   = spikes[stable_mask]       # (T_s, N)
    T_stable     = int(stable_mask.sum())
    T_trans      = T - T_stable
    frac_stable  = T_stable / T

    pred_rates_st = lam_stable.mean(axis=0)
    obs_rates_st  = spk_stable.mean(axis=0)

    obs_fano_all  = compute_fano(spikes)
    pred_fano_all = compute_fano(lambda_t)
    obs_fano_st   = compute_fano(spk_stable)
    pred_fano_st  = compute_fano(lam_stable)

    r_rates_all = pearson_r(pred_rates,    obs_rates)
    r_rates_st  = pearson_r(pred_rates_st, obs_rates_st)
    r_fano_all  = pearson_r(pred_fano_all, obs_fano_all)
    r_fano_st   = pearson_r(pred_fano_st,  obs_fano_st)

    frac_clip = float(
        (np.abs(eta_t.ravel()) >= eta_clip * 0.999).sum()
    ) / eta_t.size

    if True:  # always print
        print(f"\n  Stable bins : {T_stable}/{T} "
              f"= {100*frac_stable:.1f}%  "
              f"(thresh={stable_thresh})")
        print(f"  Trans  bins : {T_trans}/{T} "
              f"= {100*(1-frac_stable):.1f}%")
        print(f"  Rate corr   all={r_rates_all:.4f}  "
              f"stable={r_rates_st:.4f}  "
              f"delta={r_rates_st - r_rates_all:+.4f}")
        print(f"  Fano corr   all={r_fano_all:.4f}  "
              f"stable={r_fano_st:.4f}  "
              f"delta={r_fano_st - r_fano_all:+.4f}")

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(make_title("Statistics Panels", fitMD, eta_clip),
                 fontsize=10, y=1.00)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.35)

    # ---- (0,0) Rate scatter: all + stable overlaid ----
    ax_rs = fig.add_subplot(gs[0, 0])
    ax_rs.scatter(obs_rates, pred_rates,
                  s=18, alpha=0.35, color="steelblue",
                  label=f"all  r={r_rates_all:.4f}", zorder=2)
    ax_rs.scatter(obs_rates_st, pred_rates_st,
                  s=22, alpha=0.80, color="navy",
                  marker="D", label=f"stable r={r_rates_st:.4f}", zorder=3)
    lim = max(obs_rates.max(), pred_rates.max(),
              obs_rates_st.max(), pred_rates_st.max()) * 1.08
    ax_rs.plot([0, lim], [0, lim], "k--", lw=1)
    ax_rs.set_xlabel("Observed rate (sp/bin)")
    ax_rs.set_ylabel("Predicted rate (sp/bin)")
    ax_rs.set_title("Mean firing rate")
    ax_rs.legend(fontsize=7, loc="upper left")
    lbl, col = pass_label(r_rates_st, THRESH_RATE_CORR)
    ax_rs.text(0.05, 0.78, f"{lbl} (stable)", transform=ax_rs.transAxes,
               color=col, fontweight="bold", fontsize=9)

    # ---- (0,1) Fano scatter: all + stable overlaid ----
    ax_fs = fig.add_subplot(gs[0, 1])
    ax_fs.scatter(obs_fano_all, pred_fano_all,
                  s=18, alpha=0.35, color="darkorange",
                  label=f"all  r={r_fano_all:.4f}", zorder=2)
    ax_fs.scatter(obs_fano_st, pred_fano_st,
                  s=22, alpha=0.80, color="saddlebrown",
                  marker="D", label=f"stable r={r_fano_st:.4f}", zorder=3)
    lim_f = np.nanmax([obs_fano_all, pred_fano_all]) * 1.08
    ax_fs.plot([0, lim_f], [0, lim_f], "k--", lw=1)
    ax_fs.set_xlabel("Observed Fano factor")
    ax_fs.set_ylabel("Predicted Fano factor")
    ax_fs.set_title("Fano factor")
    ax_fs.legend(fontsize=7, loc="upper left")
    lbl, col = pass_label(r_fano_st, THRESH_FANO_CORR)
    ax_fs.text(0.05, 0.78, f"{lbl} (stable)", transform=ax_fs.transAxes,
               color=col, fontweight="bold", fontsize=9)

    # ---- (0,2) Deviance over time with transition shading ----
    ax_dt = fig.add_subplot(gs[0, 2])
    ax_dt.plot(deviance_t, lw=0.6, color="royalblue", alpha=0.8, zorder=2)
    # shade transitioning bins
    trans_mask = ~stable_mask
    in_trans = False
    t_start  = 0
    for t in range(T):
        if trans_mask[t] and not in_trans:
            t_start  = t
            in_trans = True
        elif not trans_mask[t] and in_trans:
            ax_dt.axvspan(t_start, t, color="gray", alpha=0.25, lw=0)
            in_trans = False
    if in_trans:
        ax_dt.axvspan(t_start, T, color="gray", alpha=0.25, lw=0)
    # mean lines
    dev_all_mean = float(deviance_t.mean())
    dev_st_mean  = float(deviance_t[stable_mask].mean())
    ax_dt.axhline(dev_all_mean, color="royalblue", lw=1.2, ls="--",
                  label=f"mean all={dev_all_mean:.2f}")
    ax_dt.axhline(dev_st_mean,  color="navy",      lw=1.2, ls="-.",
                  label=f"mean stable={dev_st_mean:.2f}")
    ax_dt.set_xlabel("Time step t")
    ax_dt.set_ylabel("Poisson deviance")
    ax_dt.set_title("Deviance / time step\n(gray = transitioning)")
    ax_dt.legend(fontsize=7)

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

    # ---- (1,1) Per-neuron deviance histogram: all vs stable ----
    ax_dn = fig.add_subplot(gs[1, 1])
    dev_n_all = deviance_n / T
    # recompute stable deviance per neuron
    lam_st  = lambda_t[stable_mask]
    spk_st  = spikes[stable_mask]
    dev_n_st = (lam_st.sum(axis=0)
                - (spk_st * np.log(lam_st + 1e-10)).sum(axis=0)) / T_stable
    ax_dn.hist(dev_n_all, bins=25, color="royalblue",
               edgecolor="white", alpha=0.55, label=f"all  T={T}")
    ax_dn.hist(dev_n_st,  bins=25, color="navy",
               edgecolor="white", alpha=0.75, label=f"stable T={T_stable}")
    ax_dn.set_xlabel("Mean deviance/step (per neuron)")
    ax_dn.set_ylabel("Neuron count")
    ax_dn.set_title("Per-neuron deviance\nall vs stable bins")
    ax_dn.legend(fontsize=8)

    # ---- (1,2) Pass/fail text ----
    ax_pf = fig.add_subplot(gs[1, 2])
    ax_pf.axis("off")

    total_dev    = float(deviance_t.sum())
    total_dev_st = float(deviance_t[stable_mask].sum())

    rows = [
        ("PRISM Stage 1 Summary",          "black", 11, True),
        ("",                                "black",  9, False),
        (f"T={T}  N={N}  "
         f"M={fitMD['num_states']}","black",  9, False),
        (f"eta_clip = ±{eta_clip}",         "black",  9, False),
        (f"dt = {fitMD['time_step_sec']} sec",
                                            "black",  9, False),
        ("",                                "black",  9, False),
        (f"Stable bins : {T_stable}/{T} "
         f"({100*frac_stable:.1f}%)",       "black",  9, False),
        (f"Trans  bins : {T_trans}/{T} "
         f"({100*(1-frac_stable):.1f}%)",   "black",  9, False),
        ("",                                "black",  9, False),
        (f"Dev/step all   : {total_dev/T:.4f}",
                                            "black",  9, False),
        (f"Dev/step stable: {total_dev_st/T_stable:.4f}",
                                            "black",  9, False),
        ("",                                "black",  9, False),
        (f"Rate corr all   : {r_rates_all:.4f}",
                                            "gray",   9, False),
        (f"Rate corr stable: {r_rates_st:.4f} "
         f"(≥{THRESH_RATE_CORR})",
         pass_label(r_rates_st, THRESH_RATE_CORR)[1], 9, True),
        (f"Fano corr all   : {r_fano_all:.4f}",
                                            "gray",   9, False),
        (f"Fano corr stable: {r_fano_st:.4f} "
         f"(≥{THRESH_FANO_CORR})",
         pass_label(r_fano_st, THRESH_FANO_CORR)[1], 9, True),
        (f"Clip frac: {frac_clip:.4f} "
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
        y -= 0.072

    return fig, r_rates_st, r_fano_st, frac_clip


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

    # ---- compute stable mask from C_true ----
    stable_mask, trans_mask = compute_stable_mask(C_true, args.stable_thresh)
    T_stable = int(stable_mask.sum())
    if args.verb > 0:
        print(f"\nLoaded: T={T}, N={N}  eta_clip={eta_clip}")
        print(f"Total deviance : {fitMD['total_deviance']:.4f}")
        print(f"Per step       : {fitMD['total_deviance']/T:.4f}")
        print(f"Per neuron*step: {fitMD['total_deviance']/(T*N):.6f}")
        print(f"Stable bins    : {T_stable}/{T} "
              f"= {100*T_stable/T:.1f}%  "
              f"(thresh={args.stable_thresh})")

    # ensure numpy
    fitD_np = {k: (v if isinstance(v, np.ndarray) else np.array(v))
               for k, v in fitD.items()}

    # ---- Canvas 1 : time-domain ----
    fig1 = make_canvas1(fitD_np, fitMD, spikes, S_true, C_true,
                        stable_mask, eta_clip, args.T_show)
    if args.plotFmt == "b":
        out1 = os.path.join(args.outPlots,
                             f"{args.dataName}_s1_canvas1.png")
        fig1.savefig(out1, bbox_inches="tight")
        print(f"  saved: {out1}")
    else:
        plt.show(block=not args.noBlock)
    plt.close(fig1)

    # ---- Canvas 2 : statistics ----
    fig2, r_rates_st, r_fano_st, frac_clip = make_canvas2(
        fitD_np, fitMD, spikes, C_true,
        stable_mask, eta_clip, args.stable_thresh)
    if args.plotFmt == "b":
        out2 = os.path.join(args.outPlots,
                             f"{args.dataName}_s1_canvas2.png")
        fig2.savefig(out2, bbox_inches="tight")
        print(f"  saved: {out2}")
    else:
        plt.show(block=not args.noBlock)
    plt.close(fig2)

    # ---- terminal pass/fail (judged on stable bins) ----
    print(f"\n{'='*54}")
    print(f" PRISM Stage 1 Pass/Fail  —  {args.dataName}")
    print(f" Judged on stable bins only  (thresh={args.stable_thresh})")
    print(f"{'='*54}")
    for label, val, thresh, higher in [
        ("Rate corr (stable)", r_rates_st, THRESH_RATE_CORR, True),
        ("Fano corr (stable)", r_fano_st,  THRESH_FANO_CORR, True),
        ("Clip frac          ", frac_clip, THRESH_CLIP_FRAC, False),
    ]:
        ok  = (val >= thresh) if higher else (val <= thresh)
        sym = "✓" if ok else "✗"
        print(f"  {sym} {label}: {val:.5f}  "
              f"({'≥' if higher else '≤'}{thresh})")
    print(f"{'='*54}")


if __name__ == "__main__":
    main()
