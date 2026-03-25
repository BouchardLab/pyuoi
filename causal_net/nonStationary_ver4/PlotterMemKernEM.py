#!/usr/bin/env python3
"""
Plotting utilities for prism EM evaluation.
"""

from toolbox.PlotterBackbone import PlotterBackbone
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors


class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self, args)


    def summary_memKerEM(self, fitD, md, figId=1):
        """EM convergence overview: 2 rows × 3 columns.

        Row 1: E-step NLL vs EM iter | M-step NLL+L1 vs M-epoch | spectral radius vs M-epoch
        Row 2: nz off-diag edges vs M-epoch | mean occupancy | A off-diag weight histogram
        """
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(12, 6))
        fig.subplots_adjust(hspace=0.45, wspace=0.35)

        trainMD = md["train"]
        short_name = md["short_name"]

        e_nll = fitD.get("e_nll_em", [])
        m_nll = fitD.get("m_nll_epoch", [])
        m_l1 = fitD.get("m_l1_epoch", [])
        m_loss = fitD.get("m_loss_epoch", [])
        rho = fitD.get("rho_epoch", [])
        nz = fitD.get("nz_edges_epoch", [])
        lr = fitD.get("learning_rates", [])

        n_em = trainMD.get("num_em_iters", 1)
        m_per_em = trainMD.get("m_epochs", 1)
        n_m_total = len(m_nll)
        m_epochs = np.arange(1, n_m_total + 1)
        em_iters = np.arange(1, len(e_nll) + 1)

        prune_em = int(trainMD.get("delay_em_iter_4_Aprune", 0))
        rho_start_em = int(trainMD.get("delay_em_iter_4_ArhoMax", 0))
        lr_drop_em = int(trainMD.get("delay_em_iter_4_lrDecay", int(n_em * 0.7)))
        prune_m_epoch = prune_em * m_per_em
        rho_start_m_epoch = rho_start_em * m_per_em
        lr_drop_m_epoch = lr_drop_em * m_per_em

        def draw_threshold_marker(ax, x_pos, x_max, txt, color,yFac=0.02):
            if not (0 < x_pos <= x_max):
                return
            ax.axvline(x_pos, color=color, ls='--', lw=0.9, alpha=0.95)
            y0, y1 = ax.get_ylim()
            x0, x1 = ax.get_xlim()
            x_off = 0.01 * max(1e-9, x1 - x0)
            y_txt = y1 - yFac * (y1 - y0)
            ax.text(
                x_pos + x_off, y_txt, txt, rotation=90, color=color, fontsize=7,
                ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.55, edgecolor='none', pad=0.2)
            )

        jSkipEM = 0
        jSkipM = int(jSkipEM * m_per_em)
        # ── Row 1, Col 1: E-step NLL vs EM iteration ────────────────
        ax = self.plt.subplot(2, 3, 1)
        ax.plot(em_iters[jSkipEM:], e_nll[jSkipEM:], 'o-', color='tab:blue', markersize=3,
                linewidth=1.2)
        ax.set(title="E-step weighted NLL", xlabel="EM iteration",
               ylabel="NLL / bin")
        ax.grid(True, alpha=0.3)
        draw_threshold_marker(ax, prune_em, len(e_nll), "start Aprune", "k")
        draw_threshold_marker(ax, rho_start_em, len(e_nll), "start rhoMax", "tab:brown")
        draw_threshold_marker(ax, lr_drop_em, len(e_nll), "start_lrDrop", "tab:gray", yFac=0.6)
        if "pgd_iter" in trainMD or "block1_iter" in trainMD:
            pgd_it = trainMD.get("pgd_iter", trainMD.get("block1_iter", "N/A"))
            lr_e = trainMD.get("lr_estep", trainMD.get("lr_kappa", "N/A"))
            txt = (f"block1_iter={pgd_it}\n"
                   f"lr_E={lr_e}\n"
                   f"λ₂={trainMD.get('lambda2', 'N/A')}")
            ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                    va="top", ha="right", fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # ── Row 1, Col 2: M-step NLL + L1 vs M-epoch ────────────────
        ax = self.plt.subplot(2, 3, 2)
        ax.plot(m_epochs[jSkipM:], m_nll[jSkipM:], color='tab:blue', linewidth=1, label='NLL')
        ax.set_ylabel('NLL', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax.set(title="M-step loss", xlabel="M-epoch (global)")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(m_epochs[jSkipM:], m_l1[jSkipM:] , color='tab:red', linewidth=1,
                 linestyle='--', label='L1')
        ax2.set_ylabel('L1', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc='upper right')
        draw_threshold_marker(ax, prune_m_epoch, n_m_total, "start Aprune", "k")
        draw_threshold_marker(ax, rho_start_m_epoch, n_m_total, "start rhoMax", "tab:brown")
        draw_threshold_marker(ax, lr_drop_m_epoch, n_m_total, "start_lrDrop", "tab:gray", yFac=0.4)

        lr_m = trainMD.get("lr_mstep", trainMD.get("lr_net", "N/A"))
        txt = (f"lr_M={lr_m}\n"
               f"L1 λ3={trainMD.get('lambda3', 'N/A')}\n"
               f"batch={trainMD.get('batch_size', 'N/A')}")
        ax.text(0.03, 0.03, txt, transform=ax.transAxes,
                va="bottom", ha="left", fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # ── Row 1, Col 3: spectral radius vs M-epoch ────────────────
        ax = self.plt.subplot(2, 3, 3)
        rho_max = trainMD.get("rho_max", 0.95)
        ax.plot(m_epochs[jSkipM:], rho[jSkipM:] , color='tab:green', linewidth=1,
                label='ρ(A)')
        ax.axhline(rho_max, color='red', ls='--', lw=1,
                   label=f'ρ_max={rho_max}')
        ax.set(title="Spectral radius ρ(A)", xlabel="M-epoch (global)",
               ylabel="ρ")
        ax.grid(True, alpha=0.3)
        draw_threshold_marker(ax, prune_m_epoch, n_m_total, "start Aprune", "k")
        draw_threshold_marker(ax, rho_start_m_epoch, n_m_total, "start rhoMax", "tab:brown")
        draw_threshold_marker(ax, lr_drop_m_epoch, n_m_total, "start_lrDrop", "tab:gray", yFac=0.4)
        ax.legend(fontsize=8)

        # ── Row 2, Col 1: non-zero off-diag edges vs M-epoch ────────
        ax = self.plt.subplot(2, 3, 4)
        ax.plot(m_epochs[jSkipM:], nz[jSkipM:], color='tab:purple', linewidth=1)
        ax.set(title=f"Non-zero off-diag edges (|A|>{trainMD['minW']})",
               xlabel="M-epoch (global)", ylabel="count")
        ax.grid(True, alpha=0.3)
        draw_threshold_marker(ax, prune_m_epoch, n_m_total, "start Aprune", "k")
        draw_threshold_marker(ax, rho_start_m_epoch, n_m_total, "start rhoMax", "tab:brown")
        draw_threshold_marker(ax, lr_drop_m_epoch, n_m_total, "start_lrDrop", "tab:gray", yFac=0.4)

        # learning rate on twin axis
        ax2 = ax.twinx()
        ax2.plot(m_epochs[jSkipM:], lr[jSkipM:], color='tab:orange', linewidth=0.8,
                 linestyle='--', alpha=0.6)
        ax2.set_ylabel('learning rate', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # ── Row 2, Col 2: temporal kernel kappa_hat ──────────────────
        ax = self.plt.subplot(2, 3, 5)
        kappa_hat = fitD.get("kappa_hat")
        if kappa_hat is not None:
            kappa_hat = np.asarray(kappa_hat)
            lags = np.arange(1, len(kappa_hat) + 1)
            ax.step(lags, kappa_hat, where='post', color='tab:blue', label='fit')
            ax.plot(lags, kappa_hat, 'o', markersize=3, alpha=0.5, color='tab:blue')
            
            # Check for truth kernel
            kappa_true = md.get("offdiag_kernel")
            if kappa_true is not None:
                kappa_true = np.asarray(kappa_true)
                lags_true = np.arange(1, len(kappa_true) + 1)
                ax.step(lags_true, kappa_true, where='post', color='tab:red', linestyle='--', alpha=0.7, label='truth')

            # Check for initial kernel
            kappa_init = fitD.get("kappa_init")
            if kappa_init is not None:
                kappa_init = np.asarray(kappa_init)
                ax.step(lags, kappa_init, where='post', color='black', linestyle='--', alpha=0.6, label='init')
            
            ax.legend(fontsize=8)
            ax.set(title="Temporal kernel $\kappa(\ell)$", xlabel="lag $\ell$", ylabel="weight")
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='k', lw=0.8, alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No kappa data", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

        # ── Row 2, Col 3: A off-diagonal weight histogram ────────────
        ax = self.plt.subplot(2, 3, 6)
        A_fit = fitD.get("A_hat", fitD.get("A_off_hat"))
        if A_fit is not None:
            A_fit = np.asarray(A_fit)
            Nn = A_fit.shape[0]
            diag_mask = np.eye(Nn, dtype=bool)
            if A_fit.ndim == 2:
                A_off = A_fit[~diag_mask]
            else:
                 A_off = A_fit # already 1D?
            valid = np.abs(A_off) > 1e-10
            A_off_nz = A_off[valid]
            n_edges = int(A_off_nz.size)
            ax.hist(A_off_nz, bins=100, color='g', alpha=0.8)
            ax.set_yscale('log')
            ax.set(title=f"A off-diagonal, {n_edges} edges",
                   xlabel="edge value", ylabel="edges")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No A_fit data", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

        fig.suptitle(f"MemKern EM: {short_name},  "
                     f"K_EM={n_em}×K_M={m_per_em}",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    def correl_fit_truth(self, fitD, md, figId=2):
        """Correlation plots for A_off, A_diag, and B compared to truth."""
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(13.5, 4.5))
        fig.subplots_adjust(wspace=0.35, bottom=0.15, top=0.85)

        short_name = md["short_name"]
        A_off_true = md.get("A_off_true")
        A_diag_true = md.get("A_diag_true")
        
        if A_off_true is None:
             print("correl_fit_truth: missing truth A_off_true, skipping")
             return

        # If it was saved as (N,N), we still need to mask it
        if A_off_true.ndim == 2:
            N = A_off_true.shape[0]
            diag_mask = np.eye(N, dtype=bool)
            A_off_true = A_off_true[~diag_mask]
        
        A_off_hat_full = np.asarray(fitD.get("A_off_hat"))
        N = A_off_hat_full.shape[0]
        diag_mask = np.eye(N, dtype=bool)
        A_off_hat = A_off_hat_full[~diag_mask]
        A_diag_hat = np.asarray(fitD.get("A_diag_hat"))
        B_true = md.get("B_true")
        B_hat = np.asarray(fitD.get("B_hat"))

        minW = md["train"].get("minW", 0.01)

        def get_corr_stats(x, y):
            if len(x) < 2: return 0.0, len(x)
            r = np.corrcoef(x, y)[0, 1]
            return float(r), len(x)

        def add_corr_ax(ax, x, y, labels, title, x_split=0, min_w=None, split_color='red', dot_color='blue', show_split=True):
            ax.scatter(x, y, s=10, alpha=0.3, color=dot_color, edgecolors='none')
            ax.axhline(0, color='k', lw=0.6, alpha=0.3)
            if show_split:
                ax.axvline(x_split, color=split_color, lw=1.0, ls='--')
            
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            low = min(x_min, y_min)
            high = max(x_max, y_max)
            ax.plot([low, high], [low, high], 'k--', lw=0.8, alpha=0.7)
            
            maskL = (x < x_split)
            maskR = (x > x_split)
            rL, nL = get_corr_stats(x[maskL], y[maskL])
            rR, nR = get_corr_stats(x[maskR], y[maskR])
            
            # Left/Right stats
            txtL = "rL=%.3f\nnL=%d" % (rL, nL)
            ax.text(0.05, 0.5, txtL, transform=ax.transAxes, va='center', ha='left', fontsize=9)
            txtR = "rR=%.3f\nnR=%d" % (rR, nR)
            ax.text(0.95, 0.4, txtR, transform=ax.transAxes, va='center', ha='right', fontsize=9)
            
            ax.set(title=title, xlabel=labels[0], ylabel=labels[1])
            ax.grid(True, alpha=0.3)
            
            if min_w is not None:
                ax.axvline(-min_w, color='red', ls='--', lw=0.8, alpha=0.6)
                ax.axvline(min_w, color='red', ls='--', lw=0.8, alpha=0.6)
                mask0 = np.abs(x) <= min_w
                n0 = np.sum(mask0)
                # Plot n0 vertically shifted to left of red lines
                ax.text(-4*min_w, 0.1, f"n0={n0}", color='red', transform=ax.get_xaxis_transform(),
                        rotation='vertical', va='bottom', ha='center', fontsize=9)

        # ── Panel 1: A_off correlation ────────────
        ax = fig.add_subplot(1, 3, 1)
        valid = np.abs(A_off_hat) >= minW
        add_corr_ax(ax, A_off_true[valid], A_off_hat[valid],
                    ["A_off_true", "A_off_hat"], f"$A_{{off}}$ fit, minW={minW:.2f}", 
                    min_w=minW, show_split=False, dot_color='green')

        # ── Panel 2: A_diag correlation ───────────
        ax = fig.add_subplot(1, 3, 2)
        ax.scatter(A_diag_true, A_diag_hat, s=12, alpha=0.6, color='salmon', edgecolors='none')
        r, n = get_corr_stats(A_diag_true, A_diag_hat)
        l, h = A_diag_true.min(), A_diag_true.max()
        ax.plot([l, h], [l, h], 'k--', lw=0.8, alpha=0.7)
        ax.set(title=f"A_diag fit, r={r:.3f}, n={n}", xlabel="A_diag_true", ylabel="A_diag_hat")
        ax.grid(True, alpha=0.3)

        # ── Panel 3: B correlation ─────────────────
        ax = fig.add_subplot(1, 3, 3)
        add_corr_ax(ax, B_true, B_hat, ["B_true", "B_hat"], "B fit", x_split=2.0, split_color='red', dot_color='blue')

        fig.suptitle(f"Prism EM: {short_name}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    def state_init_prismEM(self, fitD, md, figId=2, time_reb=20):
        """Two-panel plot: initialization vs truth state trajectories."""
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(13.5, 8.0))
        gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.85], hspace=0.75, wspace=0.35)

        trainMD = md["train"]
        t0_bin, t1_bin = trainMD["time_range_bins"]
        dt = float(trainMD["time_step_sec"])

        S_init = np.asarray(fitD["S_init"])
        C_init = np.asarray(fitD["c_init"])
        S_true = np.asarray(md["S_true"])[t0_bin : t1_bin + 1]
        C_true = np.asarray(md["C_true"])[t0_bin : t1_bin + 1]

        assert S_init.shape[0] == S_true.shape[0] == C_init.shape[0] == C_true.shape[0], \
            "S_init/S_true/C_init/C_true must have matching lengths"
        n_cmp = S_init.shape[0]

        # Use original EM time bins (dt), not coarse rate-bin or plotting rebinning.
        S_init_cl = 1.0 - np.max(C_init, axis=1)
        t = (t0_bin + np.arange(n_cmp)) * dt
        n_states = C_init.shape[1]

        ax = fig.add_subplot(gs[0, :])
        ax.step(t, S_init, where="post", color="k", linewidth=1.0, label="S_init")
        ax.plot(t, S_init, linestyle="none", marker=".", markersize=1.5, color="k", alpha=0.6)
        lo = np.clip(S_init - S_init_cl, -1.0, float(n_states - 1))
        hi = np.clip(S_init + S_init_cl, -1.0, float(n_states - 1))
        ax.fill_between(t, lo, hi, color="gray", alpha=0.3, label="S_init_CL")
        ax.set(title="Initialization", xlabel="time (s)", ylabel="state")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        for m in range(C_init.shape[1]):
            ax2.step(t, C_init[:, m], where="post", linewidth=0.8, alpha=0.85, label=f"C_init[{m}]")
            ax2.plot(t, C_init[:, m], linestyle="none", marker=".", markersize=1.4, alpha=0.55)
        ax2.set_ylabel("C_init")
        ax2.set_ylim(0.0, 1.0)

        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=8)
        ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=4, fontsize=8)

        ax = fig.add_subplot(gs[1, :])
        ax.step(t, S_true, where="post", color="k", linewidth=1.0, label="S_true")
        ax.plot(t, S_true, linestyle="none", marker=".", markersize=1.5, color="k", alpha=0.6)
        ax.set(title="Truth", xlabel="time (s)", ylabel="state")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        for m in range(C_true.shape[1]):
            ax2.step(t, C_true[:, m], where="post", linewidth=0.8, alpha=0.85, label=f"C_true[{m}]")
            ax2.plot(t, C_true[:, m], linestyle="none", marker=".", markersize=1.4, alpha=0.55)
        ax2.set_ylabel("C_true")

        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=8)
        ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=4, fontsize=8)

        # Bottom row, left: histogram of classification frequency with thresholds.
        ax = fig.add_subplot(gs[2, 0])
        init_state_md = md["init_state"]

        freq_h1d = np.asarray(fitD["freq_h1d"], dtype=np.float64).ravel()
        bins = min(40, max(10, int(np.sqrt(max(1, freq_h1d.size)))))
        ax.hist(freq_h1d, bins=bins, color='tab:blue', alpha=0.75)
        rate_bin_sec = float(init_state_md["rate_bin_sec"])
        ax.set(title=f"Rate (time bin {rate_bin_sec:g} s)", xlabel="rate", ylabel="count")
        ax.grid(True, alpha=0.3)

        thres = [float(x) for x in init_state_md["rate_thres"]]
        for thr in thres:
            ax.axvline(float(thr), color='k', linestyle='--', linewidth=1.0, alpha=0.9)

        # Label state regions 0,1,... inside histogram intervals.
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        edges = [xlo] + thres + [xhi]
        for i in range(max(0, len(edges) - 1)):
            xc = 0.5 * (edges[i] + edges[i + 1])
            ax.text(
                xc, ylo + 0.88 * (yhi - ylo), f"{i}",
                ha="center", va="center", fontsize=10, color="k",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            )

        # Bottom row, middle: B_init correlation vs log(single_rates).
        ax_mid = fig.add_subplot(gs[2, 1])
        b_init = np.asarray(fitD["B_init"], dtype=np.float64).ravel()
        single_rates = np.asarray(fitD["single_rates"], dtype=np.float64).ravel()
        assert b_init.ndim == 1 and b_init.size > 0, "Expected B_init to be 1D and non-empty"
        assert b_init.size == single_rates.size, "B_init and single_rates must have matching lengths"

        n = b_init.size
        assert n > 1, "Need at least 2 points for correlation plot"
        x = np.log(np.clip(single_rates, 1e-12, None))
        y = b_init
        ax_mid.scatter(x, y, s=10, alpha=0.75, color='tab:green', edgecolors='none')
        corr = float(np.corrcoef(x, y)[0, 1])
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        ax_mid.plot([lo, hi], [lo, hi], color='k', linestyle='--', linewidth=0.8, alpha=0.6)
        ax_mid.set(title=f"B_init r={corr:.3f}",
                   xlabel="log(single_rates)", ylabel="B_init")
        ax_mid.set_aspect("equal", adjustable="box")
        ax_mid.grid(True, alpha=0.3)

        # Bottom row, right: reserved cell for future diagnostics.
        ax_right = fig.add_subplot(gs[2, 2])
        ax_right.axis("off")

        t0s, t1s = trainMD["time_range_sec"]
        fig.suptitle(
            f"Prism EM Init: {md['short_name']}  T=[{t0s:.1f}, {t1s:.1f}] s",
            fontsize=12,
        )

   
