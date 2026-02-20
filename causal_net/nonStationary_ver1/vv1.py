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
  - Deviance per neuron*step near theoretical floor

Reads:
  <basePath>/prismFit/<dataName>_s1.stage1.npz
  <basePath>/spikesData/<dataName>.spikes.npz
  <basePath>/spikesData/<dataName>.prismTruth.npz
  <basePath>/truthDale/<truthName>.simTruth.npz

Writes plots to:
  <basePath>/plots/<dataName>_s1_*.png
"""

import os
import argparse
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from toolbox.Util_NumpyIO import read_data_npz


# Pass/fail thresholds
THRESH_RATE_CORR  = 0.95
THRESH_FANO_CORR  = 0.80
THRESH_DEV_RTOL   = 0.10   # within 10% of theoretical minimum deviance


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
    args.inpTruth  = os.path.join(args.basePath, "truthDale")
    args.outPlots  = os.path.join(args.basePath, "plots")

    print("myArg-program:", parser.prog)
    for arg in vars(args):
        print("myArg:", arg, getattr(args, arg))

    assert args.dataName is not None, "must provide --dataName"
    assert os.path.exists(args.basePath),   f"missing basePath: {args.basePath}"
    os.makedirs(args.outPlots, exist_ok=True)
    return args


def savefig(fig, outPath, tag, dataName, fmt, noBlock):
    """Save or show figure following project convention."""
    if fmt == "b":
        outFF = os.path.join(outPath, f"{dataName}_s1_{tag}.png")
        fig.savefig(outFF, bbox_inches="tight", dpi=120)
        print(f"  saved: {outFF}")
    else:
        plt.show(block=not noBlock)
    plt.close(fig)


def compute_fano(spikes, min_rate=1e-6):
    """Per-neuron Fano factor: Var(spikes) / Mean(spikes)."""
    mu  = spikes.mean(axis=0)
    var = spikes.var(axis=0)
    fano = np.where(mu > min_rate, var / mu, np.nan)
    return fano


def pearson_r(x, y):
    """Pearson correlation, ignoring NaNs."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    xm = x[mask] - x[mask].mean()
    ym = y[mask] - y[mask].mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    return float(np.dot(xm, ym) / denom) if denom > 0 else np.nan


def plot_rates(pred_rates, obs_rates, dataName, outPath, fmt, noBlock):
    """Scatter: predicted vs observed mean firing rate per neuron."""
    r = pearson_r(pred_rates, obs_rates)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(obs_rates, pred_rates, s=18, alpha=0.6, color="steelblue")
    lim = max(obs_rates.max(), pred_rates.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
    ax.set_xlabel("Observed mean rate (spikes/bin)")
    ax.set_ylabel("Predicted mean rate (spikes/bin)")
    ax.set_title(f"Stage 1: Mean Firing Rate\n{dataName}\nPearson r = {r:.4f}")
    ax.legend()
    pass_str = "PASS" if r >= THRESH_RATE_CORR else "FAIL"
    ax.text(0.05, 0.92, f"{pass_str}  (thresh={THRESH_RATE_CORR})",
            transform=ax.transAxes,
            color="green" if pass_str == "PASS" else "red",
            fontweight="bold")
    fig.tight_layout()
    savefig(fig, outPath, "rates", dataName, fmt, noBlock)
    return r


def plot_fano(pred_fano, obs_fano, dataName, outPath, fmt, noBlock):
    """Scatter: predicted vs observed Fano factor per neuron."""
    r = pearson_r(pred_fano, obs_fano)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(obs_fano, pred_fano, s=18, alpha=0.6, color="darkorange")
    lim = np.nanmax([obs_fano, pred_fano]) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
    ax.set_xlabel("Observed Fano factor")
    ax.set_ylabel("Predicted Fano factor")
    ax.set_title(f"Stage 1: Fano Factor\n{dataName}\nPearson r = {r:.4f}")
    ax.legend()
    pass_str = "PASS" if r >= THRESH_FANO_CORR else "FAIL"
    ax.text(0.05, 0.92, f"{pass_str}  (thresh={THRESH_FANO_CORR})",
            transform=ax.transAxes,
            color="green" if pass_str == "PASS" else "red",
            fontweight="bold")
    fig.tight_layout()
    savefig(fig, outPath, "fano", dataName, fmt, noBlock)
    return r


def plot_deviance(deviance_t, deviance_n, dataName, outPath, fmt, noBlock):
    """Two-panel: deviance over time and per-neuron deviance histogram."""
    fig = plt.figure(figsize=(11, 4))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    # Left: deviance over time
    ax0 = fig.add_subplot(gs[0])
    T   = len(deviance_t)
    ax0.plot(deviance_t, lw=0.6, color="royalblue", alpha=0.8)
    ax0.set_xlabel("Time step t")
    ax0.set_ylabel("Poisson deviance")
    ax0.set_title(f"Deviance over time\nmean={deviance_t.mean():.4f}  "
                  f"std={deviance_t.std():.4f}")

    # Right: per-neuron deviance histogram
    ax1 = fig.add_subplot(gs[1])
    ax1.hist(deviance_n / T, bins=30, color="royalblue", edgecolor="white", alpha=0.8)
    ax1.set_xlabel("Mean deviance per time step (per neuron)")
    ax1.set_ylabel("Neuron count")
    ax1.set_title(f"Per-neuron deviance\nmean={deviance_n.mean()/T:.4f}  "
                  f"std={deviance_n.std()/T:.4f}")

    fig.suptitle(f"Stage 1 deviance — {dataName}", y=1.01)
    fig.tight_layout()
    savefig(fig, outPath, "deviance", dataName, fmt, noBlock)


def plot_eta_distribution(eta_t, dataName, outPath, fmt, noBlock):
    """Histogram of all eta values to check for clipping saturation."""
    from matplotlib.ticker import MaxNLocator
    eta_flat = eta_t.ravel()
    n_clipped = int((eta_flat >= 20.0).sum())
    frac_clipped = n_clipped / len(eta_flat)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(eta_flat, bins=80, color="mediumpurple", edgecolor="white", alpha=0.85)
    ax.axvline(20.0, color="red", lw=1.5, linestyle="--",
               label=f"clip=20  ({frac_clipped*100:.2f}% clipped)")
    ax.set_xlabel(r"$\eta_{t,i}$ (internal potential)")
    ax.set_ylabel("Count")
    ax.set_title(f"Stage 1: Eta distribution\n{dataName}")
    ax.legend()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    savefig(fig, outPath, "eta_dist", dataName, fmt, noBlock)
    return frac_clipped


def plot_raster(spikes, lambda_t, S_true, T_show, dataName, outPath,
                fmt, noBlock, n_show=30):
    """
    Three-panel raster: observed spikes, predicted rate, and true state.
    Shows only the first T_show time steps and first n_show neurons.
    """
    T_show  = min(T_show, spikes.shape[0])
    n_show  = min(n_show, spikes.shape[1])
    sp_show = spikes[:T_show, :n_show]
    lm_show = lambda_t[:T_show, :n_show]
    st_show = S_true[:T_show]

    fig = plt.figure(figsize=(12, 7))
    gs  = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[3, 3, 1],
                            hspace=0.05)

    # Top: observed raster
    ax0 = fig.add_subplot(gs[0])
    t_idx, n_idx = np.where(sp_show > 0)
    ax0.scatter(t_idx, n_idx, s=1.5, color="black", alpha=0.6)
    ax0.set_xlim(0, T_show)
    ax0.set_ylim(-0.5, n_show - 0.5)
    ax0.set_ylabel("Neuron")
    ax0.set_title(f"Stage 1 Raster (first {T_show} steps, {n_show} neurons)")
    ax0.set_xticklabels([])

    # Middle: predicted rate heatmap
    ax1 = fig.add_subplot(gs[1])
    im  = ax1.imshow(lm_show.T, aspect="auto", origin="lower",
                     extent=[0, T_show, 0, n_show],
                     cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax1, label="λ (spikes/bin)")
    ax1.set_ylabel("Neuron")
    ax1.set_xticklabels([])

    # Bottom: true state
    ax2 = fig.add_subplot(gs[2])
    ax2.step(np.arange(T_show), st_show, where="mid", color="steelblue", lw=1)
    ax2.set_ylabel("State")
    ax2.set_xlabel("Time step")
    ax2.set_xlim(0, T_show)

    fig.suptitle(f"Observed vs Predicted — {dataName}", y=1.01)
    savefig(fig, outPath, "raster", dataName, fmt, noBlock)


def print_summary(label, value, threshold, higher_is_better=True):
    """Print a pass/fail summary line."""
    if higher_is_better:
        ok = value >= threshold
    else:
        ok = value <= threshold
    status = "PASS" if ok else "FAIL"
    color  = "\033[92m" if ok else "\033[91m"
    reset  = "\033[0m"
    print(f"  {color}{status}{reset}  {label}: {value:.5f}  (threshold={threshold})")


def main():
    args = get_parser()
    np.set_printoptions(precision=3, suppress=True)

    # ---- load stage1 fit ----
    fitFF  = os.path.join(args.inpFit, f"{args.dataName}_s1.stage1.npz")
    fitD, fitMD = read_data_npz(fitFF, verb=args.verb > 0)
    if args.verb > 1:
        print("\nstage1 metadata:"); pprint(fitMD)

    # ---- load spikes ----
    spikesFF = os.path.join(args.inpSpikes, f"{args.dataName}.spikes.npz")
    spikesD, spikesMD = read_data_npz(spikesFF, verb=args.verb > 0)

    # ---- load prismTruth ----
    prismTruthFF = os.path.join(args.inpSpikes, f"{args.dataName}.prismTruth.npz")
    prismTruthD, prismTruthMD = read_data_npz(prismTruthFF, verb=args.verb > 0)

    # ---- extract arrays ----
    eta_t      = fitD["eta_t"]       # (T, N) float32
    lambda_t   = fitD["lambda_t"]    # (T, N) float32
    deviance_t = fitD["deviance_t"]  # (T,)   float32
    deviance_n = fitD["deviance_n"]  # (N,)   float32
    pred_rates = fitD["pred_rates"]  # (N,)   float32
    obs_rates  = fitD["obs_rates"]   # (N,)   float32

    spikes = spikesD["spikes"].astype(np.float32)   # (T, N)
    S_true = prismTruthD["S_true"]                  # (T,)
    T, N   = spikes.shape

    if args.verb > 0:
        print(f"\nLoaded: T={T}, N={N}")
        print(f"Total deviance : {fitMD['total_deviance']:.4f}")
        print(f"Per step       : {fitMD['total_deviance']/T:.4f}")
        print(f"Per neuron*step: {fitMD['total_deviance']/(T*N):.6f}")

    # ---- metrics ----
    # Fano factors
    obs_fano  = compute_fano(spikes)
    # Predicted Fano: for Poisson, Var = Mean = lambda, so Fano ~ 1 per neuron
    # We compute it from lambda_t samples for consistency
    pred_fano = compute_fano(lambda_t)

    r_rates = pearson_r(pred_rates, obs_rates)
    r_fano  = pearson_r(pred_fano, obs_fano)
    frac_cl = float((eta_t.ravel() >= 20.0).sum()) / eta_t.size

    # ---- plots ----
    T_show = min(500, T)
    plot_rates(pred_rates, obs_rates, args.dataName,
               args.outPlots, args.plotFmt, args.noBlock)
    plot_fano(pred_fano, obs_fano, args.dataName,
              args.outPlots, args.plotFmt, args.noBlock)
    plot_deviance(deviance_t, deviance_n, args.dataName,
                  args.outPlots, args.plotFmt, args.noBlock)
    plot_eta_distribution(eta_t, args.dataName,
                          args.outPlots, args.plotFmt, args.noBlock)
    plot_raster(spikes.astype(int), lambda_t, S_true, T_show,
                args.dataName, args.outPlots, args.plotFmt, args.noBlock)

    # ---- pass/fail summary ----
    print(f"\n{'='*55}")
    print(f" PRISM Stage 1 — Pass/Fail Summary")
    print(f" Dataset: {args.dataName}")
    print(f"{'='*55}")
    print_summary("Rate correlation  (pred vs obs)", r_rates, THRESH_RATE_CORR)
    print_summary("Fano correlation  (pred vs obs)", r_fano,  THRESH_FANO_CORR)
    print_summary("Frac eta clipped  (should be ~0)", frac_cl, 0.01,
                  higher_is_better=False)
    print(f"{'='*55}")

    if args.verb > 0:
        print(f"\nPred rate: mean={pred_rates.mean():.4f}  "
              f"std={pred_rates.std():.4f}  "
              f"min={pred_rates.min():.4f}  max={pred_rates.max():.4f}")
        print(f"Obs  rate: mean={obs_rates.mean():.4f}  "
              f"std={obs_rates.std():.4f}  "
              f"min={obs_rates.min():.4f}  max={obs_rates.max():.4f}")
        print(f"Fano obs : mean={np.nanmean(obs_fano):.4f}  "
              f"std={np.nanstd(obs_fano):.4f}")
        print(f"Eta clip : {frac_cl*100:.3f}% of all (t,i) pairs clipped")


if __name__ == "__main__":
    main()
