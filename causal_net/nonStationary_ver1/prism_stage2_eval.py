#!/usr/bin/env python3
"""
PRISM Stage 2 - Coefficient Inference (eval/plots).

Compares inferred C_hat to ground-truth C_true and S_true.
Pass criterion: state classification accuracy > 90% on stable bins.

Produces two PNG canvases:
  <dataName>_s2_canvas1.png  : time-domain  (C traces, state, error)
  <dataName>_s2_canvas2.png  : statistics   (accuracy, confusion, loss, scatter)

Reads:
  <basePath>/prismFit/<dataName>_s2.stage2.npz
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

THRESH_ACC_STABLE = 0.90
STABLE_THRESHOLD  = 0.95


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        dest="verb", help="Verbosity level.")
    parser.add_argument("--basePath",
                        default="/dataVault2026/neurodata_tmp2",
                        help="Head dir for all data.")
    parser.add_argument("--dataName", default=None,
                        help="Spikes base name.")
    parser.add_argument("--T_show", type=int, default=1000,
                        help="Time steps shown in time-domain canvas.")
    parser.add_argument("--stable_thresh", type=float,
                        default=STABLE_THRESHOLD,
                        help="Min max(C_true[t]) for stable label.")
    parser.add_argument("-p", "--plotFmt", default="b",
                        help="b=save png, s=screen.")
    parser.add_argument("-X", "--noBlock", action="store_true",
                        help="Non-blocking show.")

    args = parser.parse_args()
    args.fitPath    = os.path.join(args.basePath, "prismFit")
    args.spikesPath    = os.path.join(args.basePath, "spikesData")
    args.outPlots  = os.path.join(args.basePath, "plots")

    print("myArg-program:", parser.prog)
    for arg in vars(args):
        print("myArg:", arg, getattr(args, arg))

    assert args.dataName is not None, "must provide --dataName"
    assert os.path.exists(args.basePath), f"missing basePath: {args.basePath}"
    os.makedirs(args.outPlots, exist_ok=True)
    return args


def pearson_r(x, y):
    mask  = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    xm    = x[mask] - x[mask].mean()
    ym    = y[mask] - y[mask].mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    return float(np.dot(xm, ym) / denom) if denom > 0 else np.nan


def compute_stable_mask(C_true, threshold):
    stable = C_true.max(axis=1) > threshold
    return stable, ~stable


def pass_label(value, threshold, higher_is_better=True):
    ok = (value >= threshold) if higher_is_better else (value <= threshold)
    return ("PASS" if ok else "FAIL"), ("green" if ok else "red")


def make_title(tag, fitMD):
    return (f"PRISM Stage 2 — {tag}\n"
            f"{fitMD['short_name']}   "
            f"N={fitMD['num_neurons']}  "
            f"M={fitMD['num_states']}  "
            f"λ₂={fitMD['lam2']}  "
            f"lr={fitMD['lr']}  "
            f"n_inner={fitMD['n_inner']}")


def shade_transitions(ax, trans_mask, T_show, alpha=0.18):
    """Shade transitioning bins in light gray."""
    in_t, t0 = False, 0
    for t in range(T_show):
        if trans_mask[t] and not in_t:
            t0, in_t = t, True
        elif not trans_mask[t] and in_t:
            ax.axvspan(t0, t, color="gray", alpha=alpha, lw=0)
            in_t = False
    if in_t:
        ax.axvspan(t0, T_show, color="gray", alpha=alpha, lw=0)


# ================================================================
#  Canvas 1 : time-domain  (6 panels)
# ================================================================
def make_canvas1(fitMD, C_true, C_hat, S_true, state_hat,
                 stable_mask, T_show):
    """
    6 panels, 3 rows x 2 cols:
      (0,0) C_true traces        (0,1) C_hat traces
      (1,0) |C_true - C_hat| per state heatmap
                                 (1,1) pointwise max-coeff error ||c_t-c_hat_t||_inf
      (2,:) true state vs inferred state  (shared x-axis)
    """
    T   = C_true.shape[0]
    M   = C_true.shape[1]
    dt  = float(fitMD["time_step_sec"])
    T_show = min(T_show, T)

    trans_mask = ~stable_mask
    x_ticks    = np.linspace(0, T_show, 6, dtype=int)
    colors     = plt.cm.tab10(np.linspace(0, 0.5, M))

    err_coeff  = np.abs(C_true[:T_show] - C_hat[:T_show])  # (T_show, M)
    linf_err   = err_coeff.max(axis=1)                       # (T_show,)

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(make_title("Time-Domain Panels", fitMD),
                 fontsize=10, y=1.00)
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[2, 2, 1.2],
                           hspace=0.48, wspace=0.28)

    # ---- (0,0) C_true ----
    ax0 = fig.add_subplot(gs[0, 0])
    for m in range(M):
        ax0.plot(C_true[:T_show, m], lw=0.9,
                 color=colors[m], label=f"c_{m}", alpha=0.85)
    ax0.axhline(STABLE_THRESHOLD, color="gray", lw=0.7,
                ls=":", label=f"thresh={STABLE_THRESHOLD}")
    shade_transitions(ax0, trans_mask, T_show)
    ax0.set_xlim(0, T_show); ax0.set_ylim(-0.05, 1.05)
    ax0.set_ylabel("Coefficient"); ax0.set_title("C_true  (ground truth)")
    ax0.legend(fontsize=7, loc="upper right", ncol=M + 1)
    ax0.set_xticks(x_ticks)
    ax0.set_xticklabels([str(v) for v in x_ticks], fontsize=8)
    ax0.set_xlabel("Time step", fontsize=8)

    # ---- (0,1) C_hat ----
    ax1 = fig.add_subplot(gs[0, 1])
    for m in range(M):
        ax1.plot(C_hat[:T_show, m], lw=0.9,
                 color=colors[m], label=f"ĉ_{m}", alpha=0.85)
    ax1.axhline(STABLE_THRESHOLD, color="gray", lw=0.7, ls=":")
    shade_transitions(ax1, trans_mask, T_show)
    ax1.set_xlim(0, T_show); ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Coefficient"); ax1.set_title("C_hat  (inferred)")
    ax1.legend(fontsize=7, loc="upper right", ncol=M)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([str(v) for v in x_ticks], fontsize=8)
    ax1.set_xlabel("Time step", fontsize=8)

    # ---- (1,0) per-state error heatmap ----
    ax2 = fig.add_subplot(gs[1, 0])
    im = ax2.imshow(err_coeff.T, aspect="auto", origin="lower",
                    extent=[0, T_show, -0.5, M - 0.5],
                    cmap="hot_r", interpolation="nearest",
                    vmin=0, vmax=1)
    plt.colorbar(im, ax=ax2, label="|C_true - C_hat|", pad=0.02)
    ax2.set_ylabel("State m"); ax2.set_title("|C_true − C_hat|  per state")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([str(v) for v in x_ticks], fontsize=8)
    ax2.set_xlabel("Time step", fontsize=8)

    # ---- (1,1) L-inf error over time ----
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(linf_err, lw=0.7, color="firebrick", alpha=0.8)
    shade_transitions(ax3, trans_mask, T_show)
    # running mean
    w = 50
    if T_show > w:
        rm = np.convolve(linf_err, np.ones(w)/w, mode="same")
        ax3.plot(rm, lw=1.5, color="darkred", label=f"running mean w={w}")
        ax3.legend(fontsize=7)
    ax3.set_xlim(0, T_show)
    ax3.set_ylabel("max_m |c_m - ĉ_m|")
    ax3.set_title("L∞ coefficient error  (gray = transitioning)")
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels([str(v) for v in x_ticks], fontsize=8)
    ax3.set_xlabel("Time step", fontsize=8)

    # ---- (2,:) state comparison ----
    ax4 = fig.add_subplot(gs[2, :])
    ax4.step(np.arange(T_show), S_true[:T_show],
             where="mid", color="steelblue", lw=1.5,
             label="S_true", alpha=0.9)
    ax4.step(np.arange(T_show), state_hat[:T_show],
             where="mid", color="tomato", lw=1.0, ls="--",
             label="state_hat", alpha=0.9)
    shade_transitions(ax4, trans_mask, T_show)
    ax4.set_xlim(0, T_show)
    ax4.set_ylim(-0.4, int(S_true.max()) + 0.6)
    ax4.set_ylabel("State"); ax4.legend(fontsize=8, loc="upper right")
    ax4.set_title("True vs inferred state")
    ax4.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax4.set_xticks(x_ticks)
    ax4.set_xticklabels([str(v) for v in x_ticks], fontsize=8)
    ax4.set_xlabel(
        f"Time step  (showing {T_show} bins = "
        f"{T_show*dt:.1f} sec  of {T} total)", fontsize=9)

    # secondary seconds axis
    ax4b = ax4.twiny()
    ax4b.set_xlim(0, T_show * dt)
    ax4b.set_xlabel("Time (sec)", fontsize=8)
    ax4b.xaxis.set_major_locator(plt.MaxNLocator(6))

    return fig


# ================================================================
#  Canvas 2 : statistics  (5 panels)
# ================================================================
def make_canvas2(fitMD, C_true, C_hat, S_true, state_hat,
                 loss_t, smooth_t, stable_mask, stable_thresh):
    """
    5 panels, 2 rows x 3 cols  (last slot = summary text):
      (0,0) confusion matrix (stable bins)
      (0,1) per-state accuracy bar chart (all vs stable)
      (0,2) C_true vs C_hat scatter (all M states pooled)
      (1,0) NLL + smoothness loss over full T
      (1,1) rolling accuracy over time
      (1,2) pass/fail summary
    """
    T, M   = C_true.shape
    dt     = float(fitMD["time_step_sec"])
    trans_mask = ~stable_mask
    T_stable   = int(stable_mask.sum())

    # ---- accuracy ----
    correct_all    = (state_hat == S_true)
    correct_stable = correct_all[stable_mask]
    acc_all        = correct_all.mean()
    acc_stable     = correct_stable.mean()

    # per-state accuracy
    acc_per_state_all    = []
    acc_per_state_stable = []
    for m in range(M):
        mask_m     = S_true == m
        mask_m_st  = mask_m & stable_mask
        acc_per_state_all.append(
            correct_all[mask_m].mean() if mask_m.sum() > 0 else np.nan)
        acc_per_state_stable.append(
            correct_all[mask_m_st].mean() if mask_m_st.sum() > 0 else np.nan)

    # ---- confusion matrix (stable only) ----
    conf = np.zeros((M, M), dtype=int)
    for t in range(T):
        if stable_mask[t]:
            conf[S_true[t], state_hat[t]] += 1

    # ---- C scatter (sample every 10th point for speed) ----
    step   = max(1, T // 2000)
    c_true_flat = C_true[::step].ravel()
    c_hat_flat  = C_hat[::step].ravel()
    r_coeff     = pearson_r(c_true_flat, c_hat_flat)

    # ---- rolling accuracy (window=200 steps) ----
    w   = 200
    rol = np.convolve(correct_all.astype(float),
                      np.ones(w) / w, mode="same")

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(make_title("Statistics Panels", fitMD),
                 fontsize=10, y=1.00)
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.38)

    # ---- (0,0) Confusion matrix ----
    ax_cm = fig.add_subplot(gs[0, 0])
    im = ax_cm.imshow(conf, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax_cm, label="count", pad=0.02)
    for i in range(M):
        for j in range(M):
            ax_cm.text(j, i, str(conf[i, j]),
                       ha="center", va="center",
                       fontsize=9,
                       color="white" if conf[i, j] > conf.max() * 0.5
                       else "black")
    ax_cm.set_xticks(range(M)); ax_cm.set_yticks(range(M))
    ax_cm.set_xlabel("Predicted state"); ax_cm.set_ylabel("True state")
    ax_cm.set_title(f"Confusion matrix\n(stable bins only  T={T_stable})")

    # ---- (0,1) Per-state accuracy bar chart ----
    ax_ba = fig.add_subplot(gs[0, 1])
    x     = np.arange(M)
    w_bar = 0.35
    ax_ba.bar(x - w_bar/2, acc_per_state_all,    w_bar,
              color="steelblue", alpha=0.7, label="all")
    ax_ba.bar(x + w_bar/2, acc_per_state_stable, w_bar,
              color="navy",      alpha=0.85, label="stable")
    ax_ba.axhline(THRESH_ACC_STABLE, color="red", lw=1.2,
                  ls="--", label=f"thresh={THRESH_ACC_STABLE}")
    ax_ba.set_xticks(x)
    ax_ba.set_xticklabels([f"state {m}" for m in range(M)])
    ax_ba.set_ylim(0, 1.05)
    ax_ba.set_ylabel("Classification accuracy")
    ax_ba.set_title("Per-state accuracy\nall vs stable bins")
    ax_ba.legend(fontsize=8)

    # ---- (0,2) C_true vs C_hat scatter ----
    ax_sc = fig.add_subplot(gs[0, 2])
    colors = plt.cm.tab10(np.linspace(0, 0.5, M))
    for m in range(M):
        idx = np.arange(m, len(c_true_flat), M)
        ax_sc.scatter(c_true_flat[idx], c_hat_flat[idx],
                      s=4, alpha=0.3, color=colors[m],
                      label=f"m={m}")
    ax_sc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_sc.set_xlabel("C_true"); ax_sc.set_ylabel("C_hat")
    ax_sc.set_title(f"Coefficient scatter\n"
                    f"Pearson r={r_coeff:.4f}  (all states pooled)")
    ax_sc.legend(fontsize=7, markerscale=3)

    # ---- (1,0) Loss over time ----
    ax_ls = fig.add_subplot(gs[1, 0])
    ax_ls.plot(loss_t,   lw=0.5, color="royalblue",
               alpha=0.7, label="NLL")
    ax_ls.plot(smooth_t, lw=0.5, color="darkorange",
               alpha=0.7, label="smooth penalty")
    # shade transitions
    in_t, t0 = False, 0
    for t in range(T):
        if trans_mask[t] and not in_t:
            t0, in_t = t, True
        elif not trans_mask[t] and in_t:
            ax_ls.axvspan(t0, t, color="gray", alpha=0.15, lw=0)
            in_t = False
    ax_ls.set_xlabel("Time step t")
    ax_ls.set_ylabel("Loss value")
    ax_ls.set_title(f"NLL + smoothness per step\n"
                    f"mean NLL={loss_t.mean():.3f}  "
                    f"mean smooth={smooth_t.mean():.4f}")
    ax_ls.legend(fontsize=8)

    # ---- (1,1) Rolling accuracy ----
    ax_ra = fig.add_subplot(gs[1, 1])
    ax_ra.plot(rol, lw=1.0, color="steelblue",
               label=f"rolling acc (w={w})")
    ax_ra.axhline(acc_stable, color="navy", lw=1.2, ls="-.",
                  label=f"stable mean={acc_stable:.3f}")
    ax_ra.axhline(THRESH_ACC_STABLE, color="red", lw=1.0,
                  ls="--", label=f"thresh={THRESH_ACC_STABLE}")
    # shade transitions
    in_t, t0 = False, 0
    for t in range(T):
        if trans_mask[t] and not in_t:
            t0, in_t = t, True
        elif not trans_mask[t] and in_t:
            ax_ra.axvspan(t0, t, color="gray", alpha=0.15, lw=0)
            in_t = False
    ax_ra.set_ylim(0, 1.05)
    ax_ra.set_xlabel("Time step t")
    ax_ra.set_ylabel("Accuracy")
    ax_ra.set_title("Rolling classification accuracy\n(gray = transitioning)")
    ax_ra.legend(fontsize=7)
    lbl, col = pass_label(acc_stable, THRESH_ACC_STABLE)
    ax_ra.text(0.05, 0.10, f"{lbl} (stable)", transform=ax_ra.transAxes,
               color=col, fontweight="bold", fontsize=10)

    # ---- (1,2) Pass/fail summary ----
    ax_pf = fig.add_subplot(gs[1, 2])
    ax_pf.axis("off")
    rows = [
        ("PRISM Stage 2 Summary",           "black", 11, True),
        ("",                                 "black",  9, False),
        (f"T={T}  N={fitMD['num_neurons']}  "
         f"M={M}",                           "black",  9, False),
        (f"λ₂={fitMD['lam2']}  "
         f"lr={fitMD['lr']}  "
         f"n_inner={fitMD['n_inner']}",
                                             "black",  9, False),
        (f"stable thresh = {stable_thresh}", "black",  9, False),
        ("",                                 "black",  9, False),
        (f"Stable bins : {T_stable}/{T} "
         f"({100*T_stable/T:.1f}%)",         "black",  9, False),
        ("",                                 "black",  9, False),
        (f"Acc all   : {acc_all:.4f}",       "gray",   9, False),
        (f"Acc stable: {acc_stable:.4f} "
         f"(≥{THRESH_ACC_STABLE})",
         pass_label(acc_stable, THRESH_ACC_STABLE)[1], 9, True),
        ("",                                 "black",  9, False),
        (f"C corr (pooled): {r_coeff:.4f}", "black",   9, False),
        ("",                                 "black",  9, False),
        ("Per-state acc (stable):",          "black",  9, False),
    ]
    for m in range(M):
        v = acc_per_state_stable[m]
        lbl2, col2 = pass_label(v, THRESH_ACC_STABLE)
        rows.append((f"  state {m}: {v:.4f}",
                     col2, 9, True))

    y = 0.97
    for txt, color, fs, bold in rows:
        ax_pf.text(0.05, y, txt,
                   transform=ax_pf.transAxes,
                   fontsize=fs, color=color,
                   fontweight="bold" if bold else "normal",
                   verticalalignment="top",
                   family="monospace")
        y -= 0.072

    return fig, acc_stable, acc_all, r_coeff


# ================================================================
#  Main
# ================================================================
def main():
    args = get_parser()
    np.set_printoptions(precision=3, suppress=True)

    # ---- load files ----
    fitFF = os.path.join(args.fitPath,
                         f"{args.dataName}.stage2.npz")
    fitD, fitMD = read_data_npz(fitFF, verb=args.verb > 0)
    if args.verb > 1:
        print("\nstage2 metadata:"); pprint(fitMD)

    inpSpikesF=fitMD['input_spikes_name']
    prismTruthFF = os.path.join(args.spikesPath,
                                 f"{inpSpikesF}.prismTruth.npz")
    prismTruthD, _ = read_data_npz(prismTruthFF, verb=args.verb > 0)

    C_true    = prismTruthD["C_true"].astype(np.float32)   # (T_full, M)
    S_true    = prismTruthD["S_true"].astype(np.int32)     # (T_full,)
    C_hat     = fitD["C_hat"].astype(np.float32)           # (T, M)
    state_hat = fitD["state_hat"].astype(np.int32)         # (T,)
    loss_t    = fitD["loss_t"].astype(np.float32)          # (T,)
    smooth_t  = fitD["smooth_t"].astype(np.float32)        # (T,)

    tsr = fitMD["time_steps_range"]
    if tsr is not None:
        t_lo, t_hi = int(tsr[0]), int(tsr[1])
        C_true = C_true[t_lo:t_hi]
        S_true = S_true[t_lo:t_hi]
        print(f"Sliced truth data to time bins [{t_lo}, {t_hi})")

    T, M = C_true.shape
    stable_mask, _ = compute_stable_mask(C_true, args.stable_thresh)
    T_stable = int(stable_mask.sum())

    acc_all    = float((state_hat == S_true).mean())
    acc_stable = float((state_hat[stable_mask] == S_true[stable_mask]).mean())

    if args.verb > 0:
        print(f"\nT={T}  M={M}")
        print(f"Stable bins    : {T_stable}/{T} = {100*T_stable/T:.1f}%")
        print(f"Accuracy all   : {acc_all:.4f}")
        print(f"Accuracy stable: {acc_stable:.4f}")
        print(f"Mean NLL       : {fitMD['mean_nll']:.4f}")

    # ---- canvas 1 ----
    fig1 = make_canvas1(fitMD, C_true, C_hat, S_true, state_hat,
                        stable_mask, args.T_show)
    if args.plotFmt == "b":
        out1 = os.path.join(args.outPlots,
                             f"{args.dataName}_s2_canvas1.png")
        fig1.savefig(out1, bbox_inches="tight")
        print(f"   display  {out1}")
    else:
        plt.show(block=not args.noBlock)
    plt.close(fig1)

    # ---- canvas 2 ----
    fig2, acc_st, acc_al, r_c = make_canvas2(
        fitMD, C_true, C_hat, S_true, state_hat,
        loss_t, smooth_t, stable_mask, args.stable_thresh)
    if args.plotFmt == "b":
        out2 = os.path.join(args.outPlots,
                             f"{args.dataName}_s2_canvas2.png")
        fig2.savefig(out2, bbox_inches="tight")
        print(f"  display {out2}")
    else:
        plt.show(block=not args.noBlock)
    plt.close(fig2)

    # ---- terminal summary ----
    print(f"\n{'='*54}")
    print(f" PRISM Stage 2 Pass/Fail  —  {args.dataName}")
    print(f" Judged on stable bins  (thresh={args.stable_thresh})")
    print(f"{'='*54}")
    for label, val, thresh, higher in [
        ("Acc stable ", acc_st, THRESH_ACC_STABLE, True),
        ("Acc all    ", acc_al, THRESH_ACC_STABLE, True),
        ("C corr     ", r_c,   0.90,               True),
    ]:
        ok  = (val >= thresh) if higher else (val <= thresh)
        sym = "✓" if ok else "✗"
        print(f"  {sym} {label}: {val:.5f}  "
              f"({'≥' if higher else '≤'}{thresh})")
    print(f"{'='*54}")

    # add after stable accuracy print
    linf_stable = np.abs(C_true[stable_mask] - 
                         C_hat[stable_mask]).max(axis=1).mean()
    linf_trans  = np.abs(C_true[~stable_mask] - 
                         C_hat[~stable_mask]).max(axis=1).mean()
    print(f"L∞ mean stable : {linf_stable:.4f}")
    print(f"L∞ mean trans  : {linf_trans:.4f}")

if __name__ == "__main__":
    main()
 
