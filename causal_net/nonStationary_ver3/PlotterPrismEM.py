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

    def _rebin_1d(self, x, merge):
        merge = int(max(1, merge))
        if merge <= 1:
            return np.asarray(x)
        x = np.asarray(x)
        n = x.shape[0] // merge
        if n <= 0:
            return x
        return x[: n * merge].reshape(n, merge).mean(axis=1)

    def _rebin_2d(self, x, merge):
        merge = int(max(1, merge))
        if merge <= 1:
            return np.asarray(x)
        x = np.asarray(x)
        n = x.shape[0] // merge
        if n <= 0:
            return x
        return x[: n * merge].reshape(n, merge, x.shape[1]).mean(axis=1)

    def _rebin_mode_1d(self, x, merge):
        merge = int(max(1, merge))
        x = np.asarray(x)
        if merge <= 1:
            return x.astype(np.int64, copy=False)
        n = x.shape[0] // merge
        if n <= 0:
            return x.astype(np.int64, copy=False)
        x = x[: n * merge].reshape(n, merge)
        out = np.zeros((n,), dtype=np.int64)
        for i in range(n):
            vals = x[i].astype(np.int64, copy=False)
            vals = vals[vals >= 0]
            if vals.size == 0:
                out[i] = -1
            else:
                out[i] = np.bincount(vals).argmax()
        return out

    def summary_prismEM(self, fitD, md, figId=1):
        """EM convergence overview: 2 rows × 3 columns.

        Row 1: E-step NLL vs EM iter | M-step NLL+L1 vs M-epoch | spectral radius vs M-epoch
        Row 2: nz off-diag edges vs M-epoch | mean occupancy | A off-diag weight histogram
        """
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(12, 6))
        fig.subplots_adjust(hspace=0.45, wspace=0.35)

        trainMD = md["train"]
        short_name = md["short_name"]

        e_nll = np.asarray(fitD["e_nll_em"])
        m_nll = np.asarray(fitD["m_nll_epoch"])
        m_l1 = np.asarray(fitD["m_l1_epoch"])
        m_loss = np.asarray(fitD["m_loss_epoch"])
        rho = np.asarray(fitD["rho_epoch"])
        nz = np.asarray(fitD["nz_edges_epoch"])
        lr = np.asarray(fitD["learning_rates"])

        n_em = trainMD["num_em_iters"]
        m_per_em = trainMD["m_epochs"]
        n_m_total = len(m_nll)
        m_epochs = np.arange(1, n_m_total + 1)
        em_iters = np.arange(1, len(e_nll) + 1)

        prune_em = int(trainMD.get("delay_em_iter_4_Aprune", 0))
        rho_start_em = int(trainMD.get("delay_em_iter_4_ArhoMax", 0))
        lr_drop_em = int(trainMD.get("delay_em_iter_4_lrDecay", int(n_em * 0.7)))
        dyn_weight_em = int(trainMD.get("delay_em_iter_4_dynWeight", int(n_em * 0.8)))
        show_dyn_weight_marker = not bool(trainMD.get("noFreqWeight", False))
        prune_m_epoch = prune_em * m_per_em
        rho_start_m_epoch = rho_start_em * m_per_em
        lr_drop_m_epoch = lr_drop_em * m_per_em
        dyn_weight_m_epoch = dyn_weight_em * m_per_em

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
        if show_dyn_weight_marker:
            draw_threshold_marker(ax, dyn_weight_em, len(e_nll), "start dynWeight", "tab:pink")
        txt = (f"pgd_iter={trainMD['pgd_iter']}\n"
               f"lr_E={trainMD['lr_estep']}\n"
               f"λ₂={trainMD['lambda2']}")
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
        if show_dyn_weight_marker:
            draw_threshold_marker(ax, dyn_weight_m_epoch, n_m_total, "start dynWeight", "tab:pink")

        txt = (f"lr_M={trainMD['lr_mstep']}\n"
               f"L1 λ3={trainMD['lambda3']}\n"
               f"batch={trainMD['batch_size']}")
        ax.text(0.03, 0.03, txt, transform=ax.transAxes,
                va="bottom", ha="left", fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # ── Row 1, Col 3: spectral radius vs M-epoch ────────────────
        ax = self.plt.subplot(2, 3, 3)
        rho_max = trainMD["rho_max"]
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
        if show_dyn_weight_marker:
            draw_threshold_marker(ax, dyn_weight_m_epoch, n_m_total, "start dynWeight", "tab:pink")
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
        if show_dyn_weight_marker:
            draw_threshold_marker(ax, dyn_weight_m_epoch, n_m_total, "start dynWeight", "tab:pink")

        # learning rate on twin axis
        ax2 = ax.twinx()
        ax2.plot(m_epochs[jSkipM:], lr[jSkipM:], color='tab:orange', linewidth=0.8,
                 linestyle='--', alpha=0.6)
        ax2.set_ylabel('learning rate', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # ── Row 2, Col 2: mean occupancy ─────────────────────────────
        ax = self.plt.subplot(2, 3, 5)
        c_hat = np.asarray(fitD["c_hat"])
        occ = c_hat.mean(axis=0)
        M = len(occ)
        ax.bar(np.arange(M), occ, color='tab:blue', alpha=0.7)
        ax.set(title="Mean occupancy", xlabel="state", ylabel="mean c")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(np.arange(M))
        ax.grid(True, alpha=0.3, axis='y')

        t0s, t1s = trainMD["time_range_sec"]
        txt = (f"N={trainMD['num_neurons']}  M={M}\n"
               f"T=[{t0s:.0f},{t1s:.0f}]s\n"
               f"bins={trainMD['num_time_bins']}")
        ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                va="top", ha="right", fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # ── Row 2, Col 3: A off-diagonal weight histogram ────────────
        ax = self.plt.subplot(2, 3, 6)
        A_fit = np.asarray(fitD["A_hat"])
        Nn = A_fit.shape[0]
        diag_mask = np.eye(Nn, dtype=bool)
        A_off = A_fit[~diag_mask]
        valid = np.abs(A_off) > 1e-10
        A_off_nz = A_off[valid]
        n_edges = int(A_off_nz.size)
        ax.hist(A_off_nz, bins=100, color='g', alpha=0.8)
        ax.set_yscale('log')
        ax.set(title=f"A off-diagonal, {n_edges} edges",
               xlabel="edge value", ylabel="edges")
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"Prism EM: {short_name},  "
                     f"K_EM={n_em}×K_M={m_per_em}",
                     fontsize=12)
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

    def _draw_A_matrix(self, ax, A, title, num_exc=None, norm_map=None):
        A = np.asarray(A)
        if norm_map is None:
            vmin = float(np.min(A))
            vmax = float(np.max(A))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-9
            norm_map = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        im = ax.imshow(A, aspect=1.0, origin='lower', cmap='bwr',
                       norm=norm_map, interpolation='nearest')
        N = A.shape[0]
        if num_exc is not None:
            ax.axhline(num_exc - 0.5, color='k', ls='--', lw=0.8)
            ax.axvline(num_exc - 0.5, color='k', ls='--', lw=0.8)
        ax.plot([0, N], [0, N], '--', lw=0.8, color='magenta')
        ax.set_xlim(-0.5, N + 0.5)
        ax.set_ylim(-0.5, N + 0.5)
        ax.grid(True, alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel('presyn. neuron index (output)')
        ax.set_ylabel('postsyn. neuron index (input)')
        self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _get_A_true_info(self, md):
        A_true = np.asarray(md["A_true"])
        if A_true.ndim == 3:
            A_true = A_true[0]
        assert A_true.ndim == 2 and A_true.shape[0] == A_true.shape[1], "A_true must be square 2D"
        N = A_true.shape[0]
        num_exc = md["dale_conf"]["num_excite"]
        off_mask = ~np.eye(N, dtype=bool)
        n_diag = int(np.eye(N, dtype=bool).sum())
        n_off = int(np.count_nonzero(A_true[off_mask]))
        vmin = float(np.min(A_true))
        vmax = float(np.max(A_true))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-9
        shared_norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        return A_true, N, num_exc, n_diag, n_off, shared_norm

    def _scatter_true_vs_cmp(self, ax, x_true, y_cmp, title, color, split_by_true_sign=False):
        x_true = np.asarray(x_true, dtype=np.float64).ravel()
        y_cmp = np.asarray(y_cmp, dtype=np.float64).ravel()
        n = min(x_true.size, y_cmp.size)
        x = x_true[:n]
        y = y_cmp[:n]
        ax.scatter(x, y, alpha=0.7, color=color, marker='.', s=9)

        if n > 0:
            lo = float(min(np.min(x), np.min(y)))
            hi = float(max(np.max(x), np.max(y)))
            pad = 0.05 * max(1e-9, hi - lo)
            lo -= pad
            hi += pad
            ax.plot([lo, hi], [lo, hi], color='k', linestyle='--', linewidth=1.0, alpha=0.6)
            ax.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.axvline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

            if split_by_true_sign:
                m_neg = x < 0.0
                m_pos = x > 0.0
                if np.any(m_neg):
                    xm = float(np.mean(x[m_neg]))
                    ym = float(np.mean(y[m_neg]))
                    ax.plot([xm], [ym], marker='+', markersize=14, markeredgewidth=2.5, color='k')
                if np.any(m_pos):
                    xm = float(np.mean(x[m_pos]))
                    ym = float(np.mean(y[m_pos]))
                    ax.plot([xm], [ym], marker='+', markersize=14, markeredgewidth=2.5, color='k')
            else:
                xm = float(np.mean(x))
                ym = float(np.mean(y))
                ax.plot([xm], [ym], marker='+', markersize=14, markeredgewidth=2.5, color='k')

        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel('true weight')
        ax.set_ylabel('fitted')

    def compare_A_vs_truth(self, A_cmp, md, cmp_label='A_init', figId=3):
        """Reusable canvas to compare an A-matrix candidate against A_true."""
        A_true, N, num_exc, n_diag, n_off, shared_norm = self._get_A_true_info(md)
        A_cmp = np.asarray(A_cmp)
        assert A_cmp.ndim == 2 and A_cmp.shape == A_true.shape, "A_compare must match A_true shape"

        off_mask = ~np.eye(N, dtype=bool)
        neg_mask = off_mask & (A_true < 0)
        pos_mask = off_mask & (A_true > 0)
        diag_mask = np.eye(N, dtype=bool)

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(14, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.45)

        ax = fig.add_subplot(gs[0, 0])
        self._draw_A_matrix(
            ax, A_true, f"True Dale, N{N}, nEdges={n_diag}+{n_off}",
            num_exc=num_exc, norm_map=shared_norm
        )

        ax = fig.add_subplot(gs[0, 1])
        self._draw_A_matrix(
            ax, A_cmp, f"{cmp_label}, nEdges={n_diag}+{n_off}",
            num_exc=num_exc, norm_map=shared_norm
        )

        ax = fig.add_subplot(gs[0, 2])
        A_off = A_cmp[off_mask]
        A_off_nz = A_off[np.abs(A_off) > 1e-12]
        ax.hist(A_off_nz, bins=120, color='saddlebrown', alpha=0.85)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.35)
        ax.set_title(f"{cmp_label} off-diagonal")
        ax.set_xlabel("edge value")
        ax.set_ylabel("edges")

        initA = md["init_A"]
        txt = (
            "A-init diagnostics:\n"
            f"  cond(YpYp) = {initA['cond_YpYp']:.2e}\n"
            f"  rho(A_ols) = {initA['rho_A_init']:.3f}\n"
            f"  R2         = {initA['R2_1step']:.3f}\n"
            f"  ||A||_F    = {initA['fro_A_init']:.3f}\n"
            f"  bins_used  = {initA['num_bins_used']}/{initA['num_bins_total']}"
        )
        ax.text(
            0.05, 0.70, txt, transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            family="monospace",
        )

        ax = fig.add_subplot(gs[1, 0])
        self._scatter_true_vs_cmp(
            ax,
            A_true[neg_mask],
            A_cmp[neg_mask],
            f"{cmp_label} :neg TP",
            color='tab:green',
        )

        ax = fig.add_subplot(gs[1, 1])
        self._scatter_true_vs_cmp(
            ax,
            A_true[pos_mask],
            A_cmp[pos_mask],
            f"{cmp_label} :pos TP",
            color='tab:green',
        )

        ax = fig.add_subplot(gs[1, 2])
        self._scatter_true_vs_cmp(
            ax,
            A_true[diag_mask],
            A_cmp[diag_mask],
            f"{cmp_label} :diag",
            color='salmon',
            split_by_true_sign=True,
        )

        fig.suptitle(f"A_true  vs. {cmp_label}  comparison: {md['short_name']}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    def matrix_init_prismEM(self, fitD, md, figId=3, est_key="A_init", est_label="A_init"):
        """Wrapper for current EM use case; reusable for A_hat later."""
        self.compare_A_vs_truth(fitD[est_key], md, cmp_label=est_label, figId=figId)

    def edge_recovery_prismEM(self, fitD, md, minW=0.02, figId=4, est_key="A_hat", est_label="A_hat"):
        """One-row, four-panel edge recovery: A_true | A_est edges | confusion | stats."""
        A_true, N, num_exc, n_diag, n_off, shared_norm = self._get_A_true_info(md)
        A_est = np.asarray(fitD[est_key])
        assert A_est.ndim == 2 and A_est.shape == A_true.shape, "A_est must match A_true shape"

        E_true = np.asarray(md["E_true"]).astype(bool)
        if E_true.ndim == 3:
            E_true = E_true[0]
        assert E_true.shape == A_true.shape, "E_true shape must match A_true"

        off_diag = ~np.eye(N, dtype=bool)
        E_t = E_true & off_diag
        E_hat = (np.abs(A_est) > float(minW)) & off_diag

        TP = E_t & E_hat
        FP = (~E_t) & E_hat
        FN = E_t & (~E_hat)
        TN = (~E_t) & (~E_hat)

        tp = int(TP.sum()); fp = int(FP.sum()); fn = int(FN.sum()); tn = int(TN.sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        acc = (tp + tn) / max(1, tp + tn + fp + fn)

        conf_map = np.zeros((N, N), dtype=np.int8)
        conf_map[FN] = 1
        conf_map[FP] = 2
        conf_map[TP] = 3

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(12, 3.8))
        kw = dict(origin='lower', interpolation='nearest')

        ax = self.plt.subplot(1, 4, 1)
        self._draw_A_matrix(
            ax, A_true, f"True Dale, N{N}, nEdges={n_diag}+{n_off}",
            num_exc=num_exc, norm_map=shared_norm
        )

        ax = self.plt.subplot(1, 4, 2)
        ax.imshow(E_hat.astype(float), cmap='Greys', vmin=0, vmax=1, **kw)
        ax.set(title=f"{est_label} edges  (n={int(E_hat.sum())},  minW={float(minW):g})",
               xlabel='presyn. neuron (output)', ylabel='postsyn. neuron (input)')
        ax.plot([0, N-1], [0, N-1], '--', lw=0.8, color='magenta')

        cmap_conf = colors.ListedColormap(['white', 'magenta', 'red', 'green'])
        norm_conf = colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_conf.N)
        ax = self.plt.subplot(1, 4, 3)
        im = ax.imshow(conf_map, cmap=cmap_conf, norm=norm_conf, **kw)
        cbar = self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(['TN', 'FN', 'FP', 'TP'])
        ax.set(title='Confusion map',
               xlabel='presyn. neuron', ylabel='postsyn. neuron')
        ax.plot([0, N-1], [0, N-1], '--', lw=0.8, color='k')

        ax = self.plt.subplot(1, 4, 4)
        vals = [tp, fp, fn]
        ax.bar(['TP', 'FP', 'FN'], vals, color=['green', 'red', 'magenta'])
        ax.set(title='stats', ylabel='count')
        ax.grid(axis='y', alpha=0.4)
        y_pos = 0.25 * max(1, max(vals))
        for name, val in zip(['TP', 'FP', 'FN'], vals):
            ax.text(name, y_pos, str(val), ha='center', va='center', fontsize=10)
        txt = (f"precision={precision:.3f}\nrecall={recall:.3f}\n"
               f"f1={f1:.3f}\nacc={acc:.3f}")
        ax.text(0.55, 0.60, txt, transform=ax.transAxes, fontsize=9)

        fig.suptitle(
            f"A-matrix edge recovery, minW={float(minW):g} (off-diag only): {md['short_name']}",
            fontsize=12)
        fig.tight_layout()

    def _scatter_tp_true_vs_est(self, ax, a_true_tp, a_est_tp, title, color):
        x_true = np.asarray(a_true_tp, dtype=np.float64).ravel()
        y_init = np.asarray(a_est_tp, dtype=np.float64).ravel()
        n = min(x_true.size, y_init.size)
        if n <= 0:
            ax.set(title=f"{title}\n(no TP points)", xlabel="A_init", ylabel="A_true")
            ax.grid(True, alpha=0.35)
            return
        x_true = x_true[:n]
        y_init = y_init[:n]
        ax.scatter(y_init, x_true, s=10, marker='.', alpha=0.70, color=color, zorder=4)
        self._add_x45_lins(ax, only45=True)
        ax.axvline(0.0, linestyle='--', color='0.35', linewidth=0.9, alpha=0.7, zorder=1)
        ax.axhline(0.0, linestyle='--', color='0.35', linewidth=0.9, alpha=0.7, zorder=1)

        lo = float(min(np.min(x_true), np.min(y_init)))
        hi = float(max(np.max(x_true), np.max(y_init)))
        span = max(1e-6, hi - lo)
        pad = 0.05 * span
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.35)
        ax.set(title=f"{title} (n={n})", xlabel="A_init", ylabel="A_true")

    def _hist_tp_residuals(self, ax, a_true_tp, a_est_tp, title, color):
        x = np.asarray(a_true_tp, dtype=np.float64).ravel()
        y = np.asarray(a_est_tp, dtype=np.float64).ravel()
        n = min(x.size, y.size)
        if n <= 0:
            ax.set(title=f"{title}\n(no TP points)", xlabel="A_init - A_true", ylabel="count")
            ax.grid(True, alpha=0.35)
            return
        resid = y[:n] - x[:n]
        bins = min(80, max(15, int(np.sqrt(n))))
        ax.hist(resid, bins=bins, color=color, alpha=0.8)
        ax.axvline(0.0, color='k', linestyle='--', linewidth=1.0, alpha=0.8)
        ax.grid(True, alpha=0.35)
        ax.set(title=f"{title} (n={n})", xlabel="A_init - A_true", ylabel="count")

        mae = float(np.mean(np.abs(resid)))
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        ax.text(
            0.04, 0.96,
            f"MAE={mae:.3f}\nRMSE={rmse:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )

    def _hist_all_values(self, ax, vals, title, color, tp_vals=None):
        v = np.asarray(vals, dtype=np.float64).ravel()
        if v.size == 0:
            ax.set(title=f"{title}\n(no values)", xlabel="A_init", ylabel="count")
            ax.grid(True, alpha=0.35)
            return
        bins = min(120, max(25, int(np.sqrt(v.size))))
        counts, edges, _ = ax.hist(v, bins=bins, color=color, alpha=0.8, label="all")
        if tp_vals is not None:
            tp = np.asarray(tp_vals, dtype=np.float64).ravel()
            if tp.size > 0:
                ax.hist(
                    tp, bins=edges, histtype='step', color='k', linewidth=1.8,
                    label=f"TP (n={tp.size})"
                )
        ax.axvline(0.0, color='k', linestyle='--', linewidth=1.0, alpha=0.9)
        ax.grid(True, alpha=0.35)
        ax.set(title=f"{title} (n={v.size})", xlabel="A_init", ylabel="count")
        ax.text(
            0.04, 0.96,
            f"mean={float(np.mean(v)):.3f}\nstd={float(np.std(v)):.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )
        if tp_vals is not None:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    def _hist2d_values_vs_row(self, ax, A, row_lo, row_hi, title, cmap="bwr"):
        A = np.asarray(A, dtype=np.float64)
        n_rows = int(A.shape[0])
        n_cols = int(A.shape[1])
        r0 = max(0, int(row_lo))
        r1 = min(n_rows, int(row_hi))
        if r1 <= r0:
            ax.set(title=f"{title}\n(empty row range)", xlabel="A_init", ylabel="row neuron index")
            ax.grid(True, alpha=0.35)
            return

        sub = A[r0:r1, :]
        xx = sub.ravel()
        yy = np.repeat(np.arange(r0, r1, dtype=np.float64), n_cols)
        xbins = min(140, max(40, int(np.sqrt(xx.size))))
        ybins = max(12, min(80, r1 - r0))

        x_lo = float(np.min(xx))
        x_hi = float(np.max(xx))
        if np.isclose(x_lo, x_hi):
            x_hi = x_lo + 1e-9
        x_edges = np.linspace(x_lo, x_hi, xbins + 1)
        y_edges = np.linspace(r0 - 0.5, r1 - 0.5, ybins + 1)

        H, _, _ = np.histogram2d(xx, yy, bins=[x_edges, y_edges])
        x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
        sign_x = np.sign(x_cent)
        # Normalize by neuron count (N columns) per user request.
        H_signed = (H * sign_x[:, None]) / float(n_cols)

        vmin = float(np.min(H_signed))
        vmax = float(np.max(H_signed))
        if vmin >= 0.0:
            vmin = -1e-9
        if vmax <= 0.0:
            vmax = 1e-9
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        im = ax.pcolormesh(x_edges, y_edges, H_signed.T, shading='auto', cmap=cmap, norm=norm)
        ax.axvline(0.0, color='k', linestyle='--', linewidth=1.0, alpha=0.8)
        ax.set(title=title, xlabel="A_init", ylabel="row neuron index")
        ax.set_ylim(r0 - 0.5, r1 - 0.5)
        ax.grid(False)
        self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _hist2d_values_vs_row_mask(self, ax, A, mask, title, cmap="bwr"):
        A = np.asarray(A, dtype=np.float64)
        M = np.asarray(mask, dtype=bool)
        assert A.shape == M.shape, "A/mask shape mismatch"
        n_rows = int(A.shape[0])
        n_cols = int(A.shape[1])

        rr, cc = np.nonzero(M)
        if rr.size <= 0:
            ax.set(title=f"{title}\n(no values)", xlabel="A_init", ylabel="row neuron index")
            ax.grid(True, alpha=0.35)
            return

        xx = A[rr, cc]
        yy = rr.astype(np.float64, copy=False)
        xbins = min(140, max(40, int(np.sqrt(xx.size))))
        ybins = max(12, min(80, n_rows))

        x_lo = float(np.min(xx))
        x_hi = float(np.max(xx))
        if np.isclose(x_lo, x_hi):
            x_hi = x_lo + 1e-9
        x_edges = np.linspace(x_lo, x_hi, xbins + 1)
        y_edges = np.linspace(-0.5, n_rows - 0.5, ybins + 1)

        H, _, _ = np.histogram2d(xx, yy, bins=[x_edges, y_edges])
        x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
        sign_x = np.sign(x_cent)
        H_signed = (H * sign_x[:, None]) / float(max(1, n_cols))

        vmin = float(np.min(H_signed))
        vmax = float(np.max(H_signed))
        if vmin >= 0.0:
            vmin = -1e-9
        if vmax <= 0.0:
            vmax = 1e-9
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        im = ax.pcolormesh(x_edges, y_edges, H_signed.T, shading='auto', cmap=cmap, norm=norm)
        ax.axvline(0.0, color='k', linestyle='--', linewidth=1.0, alpha=0.8)
        ax.set(title=f"{title} (n={xx.size})", xlabel="A_init", ylabel="row neuron index")
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.grid(False)
        self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def initA_quality_prismEM(self, fitD, md, minW=0.02, figId=7):
        """Three-row diagnostics for A_init vs A_true with diag/exc/inh categories."""
        A_true, N, num_exc, n_diag, n_off, shared_norm = self._get_A_true_info(md)
        A_init = np.asarray(fitD["A_init"], dtype=np.float64)
        assert A_init.ndim == 2 and A_init.shape == A_true.shape, "A_init must match A_true"

        E_true = np.asarray(md["E_true"]).astype(bool)
        if E_true.ndim == 3:
            E_true = E_true[0]
        assert E_true.shape == A_true.shape, "E_true shape must match A_true"

        diag_mask = np.eye(N, dtype=bool)
        off_diag = ~diag_mask
        E_t = E_true & off_diag
        E_hat = (np.abs(A_init) > float(minW)) & off_diag
        TP = E_t & E_hat
        FP = (~E_t) & E_hat
        FN = E_t & (~E_hat)

        # Exclusive categories: diagonal | excitatory off-diagonal rows | inhibitory off-diagonal rows.
        presyn_is_exc = (np.arange(N, dtype=np.int64)[:, None] < int(num_exc))
        cat_exc = off_diag & presyn_is_exc
        cat_inh = off_diag & (~presyn_is_exc)
        cat_diag = diag_mask
        tp_exc = TP & cat_exc
        tp_inh = TP & cat_inh

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(14.0, 8.0))
        gs = fig.add_gridspec(3, 4, hspace=0.42, wspace=0.38)

        # Top-left: A_true matrix in the same style as -p c.
        ax = fig.add_subplot(gs[0, 0])
        self._draw_A_matrix(
            ax, A_true, f"True Dale, N{N}, nEdges={n_diag}+{n_off}",
            num_exc=num_exc, norm_map=shared_norm
        )

        ax = fig.add_subplot(gs[0, 1])
        self._scatter_tp_true_vs_est(
            ax, A_true[tp_exc], A_init[tp_exc],
            "excitatory TP scatter", color="red"
        )
        x0, x1 = ax.get_xlim()
        ax.set_xlim(min(x0, 0.0), max(x1, 0.0))

        ax = fig.add_subplot(gs[0, 2])
        self._scatter_tp_true_vs_est(
            ax, A_true[tp_inh], A_init[tp_inh],
            "inhibitory TP scatter", color="tab:blue"
        )
        x0, x1 = ax.get_xlim()
        ax.set_xlim(min(x0, 0.0), max(x1, 0.0))

        ax = fig.add_subplot(gs[0, 3])
        self._scatter_tp_true_vs_est(
            ax, A_true[cat_diag], A_init[cat_diag],
            "diagonal scatter", color="magenta"
        )
        x0, x1 = ax.get_xlim()
        ax.set_xlim(min(x0, 0.0), max(x1, 0.0))

        # Middle-left: A_init matrix in the same style as A_true.
        ax = fig.add_subplot(gs[1, 0])
        self._draw_A_matrix(
            ax, A_init, "A_init", num_exc=num_exc, norm_map=shared_norm
        )

        ax = fig.add_subplot(gs[1, 1])
        self._hist_all_values(
            ax, A_init[cat_exc],
            "A_init: excit off-diag", color="red",
            tp_vals=A_init[tp_exc]
        )

        ax = fig.add_subplot(gs[1, 2])
        self._hist_all_values(
            ax, A_init[cat_inh],
            "A_init: inhib off-diag", color="tab:blue",
            tp_vals=A_init[tp_inh]
        )

        ax = fig.add_subplot(gs[1, 3])
        self._hist_all_values(
            ax, A_init[cat_diag],
            "A_init: all diag", color="magenta",
            tp_vals=None
        )

        # Third row: A_init value vs presyn neuron index.
        ax = fig.add_subplot(gs[2, 0])
        self._hist2d_values_vs_row(
            ax, A_init, 0, N, "A_init value vs row idx (all)", cmap="bwr"
        )

        ax = fig.add_subplot(gs[2, 1])
        self._hist2d_values_vs_row_mask(
            ax, A_init, cat_exc, "excit off-diag", cmap="bwr"
        )

        ax = fig.add_subplot(gs[2, 2])
        self._hist2d_values_vs_row_mask(
            ax, A_init, cat_inh, "inhib off-diag", cmap="bwr"
        )

        ax = fig.add_subplot(gs[2, 3])
        self._hist2d_values_vs_row_mask(
            ax, A_init, cat_diag, "diagonal", cmap="bwr"
        )

        tp = int(TP.sum())
        fp = int(FP.sum())
        fn = int(FN.sum())
        tp_e = int(tp_exc.sum())
        tp_i = int(tp_inh.sum())
        n_d = int(cat_diag.sum())
        fig.suptitle(
            f"A_init TP quality vs A_true, minW={float(minW):g}: {md['short_name']}\n"
            f"off-diag TP={tp} (exc={tp_e}, inh={tp_i}), FP={fp}, FN={fn}; diag n={n_d}",
            fontsize=12,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])

    def _add_x45_lins(self, ax, only45=False):
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, '--', color='k', linewidth=0.8)
        if only45:
            return
        ax.axvline(0, linestyle='--', color='k', linewidth=1)
        ax.axhline(0, linestyle='--', color='k', linewidth=1)

    def _corrcoef_safe(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size == 0 or y.size == 0:
            return np.nan
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    def _find_bimodal_divider(self, xV):
        x = np.asarray(xV, dtype=float).ravel()
        assert x.size >= 2, f"Need >=2 samples for bimodal split, got {x.size}"
        assert np.isfinite(x).all(), "xV contains non-finite values"
        assert np.std(x) > 0.0, "xV must have non-zero variance for bimodal split"

        q1, q3 = np.quantile(x, [0.25, 0.75])
        c1, c2 = float(q1), float(q3)
        if c1 == c2:
            c1 = float(np.min(x))
            c2 = float(np.max(x))
        assert c1 != c2, "Failed to initialize two distinct mode centers"

        for _ in range(32):
            d1 = np.abs(x - c1)
            d2 = np.abs(x - c2)
            left = d1 <= d2
            n_left = int(left.sum())
            n_right = int((~left).sum())
            assert n_left > 0 and n_right > 0, "Bimodal split produced empty cluster"
            c1_new = float(np.mean(x[left]))
            c2_new = float(np.mean(x[~left]))
            if abs(c1_new - c1) < 1e-10 and abs(c2_new - c2) < 1e-10:
                c1, c2 = c1_new, c2_new
                break
            c1, c2 = c1_new, c2_new

        if c1 > c2:
            c1, c2 = c2, c1
        divide = 0.5 * (c1 + c2)
        assert np.isfinite(divide), "Computed non-finite bimodal divider"
        return float(divide)

    def _plot_corr_A_regions(self, ax, xV, yV, minW, title, xlab, ylab, s=6, alpha=0.5, color=None):
        x = np.asarray(xV)
        y = np.asarray(yV)
        ax.scatter(x, y, s=s, alpha=alpha, color=color)
        self._add_x45_lins(ax, only45=True)
        ax.axvline(-minW, color="red", linestyle="--", linewidth=1)
        ax.axvline(minW, color="red", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel=xlab, ylabel=ylab)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.4)

        mask_left = x < -minW
        mask_mid = (x >= -minW) & (x <= minW)
        mask_right = x > minW
        r_left = self._corrcoef_safe(x[mask_left], y[mask_left])
        r_mid = self._corrcoef_safe(x[mask_mid], y[mask_mid])
        r_right = self._corrcoef_safe(x[mask_right], y[mask_right])
        n_left = int(mask_left.sum())
        n_mid = int(mask_mid.sum())
        n_right = int(mask_right.sum())

        ax.text(0.20, 0.05, f"rL={r_left:.3f}\nnL={n_left}", transform=ax.transAxes)
        ax.text(0.50, 0.40, f"rM={r_mid:.3f}\nnM={n_mid}", transform=ax.transAxes)
        ax.text(0.65, 0.60, f"rR={r_right:.3f}\nnR={n_right}", transform=ax.transAxes)

    def _plot_corr_B_divisor(self, ax, xV, yV, title, xlab, ylab, s=6, alpha=0.5, color=None):
        x = np.asarray(xV)
        y = np.asarray(yV)
        assert x.shape == y.shape, f"x and y shape mismatch: {x.shape} vs {y.shape}"
        divide = self._find_bimodal_divider(x)

        ax.scatter(x, y, s=s, alpha=alpha, color=color)
        self._add_x45_lins(ax, only45=True)
        ax.axvline(divide, color="red", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel=xlab, ylabel=ylab)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.4)

        mask_left = x < divide
        mask_right = x >= divide
        r_left = self._corrcoef_safe(x[mask_left], y[mask_left])
        r_right = self._corrcoef_safe(x[mask_right], y[mask_right])
        n_left = int(mask_left.sum())
        n_right = int(mask_right.sum())
        ax.text(0.05, 0.45, f"rL={r_left:.3f}\nnL={n_left}", transform=ax.transAxes, va="top")
        ax.text(0.65, 0.55, f"rR={r_right:.3f}\nnR={n_right}", transform=ax.transAxes, va="top")

    def eval_ABcorr_prismEM(self, fitD, md, figId=6):
        A_hat = np.asarray(fitD["A_hat"])
        B_hat = np.asarray(fitD["B_hat"])
        A_true = np.asarray(md["A_true"])
        B_true = np.asarray(md["B_true"])
        minW = float(self.args.minW)

        assert A_true.ndim == 2, f"A_true must be 2D, got shape={A_true.shape}"
        assert A_hat.ndim == 2, f"A_hat must be 2D, got shape={A_hat.shape}"

        if B_hat.ndim == 1:
            B_hat = B_hat[None, :]
        if B_true.ndim == 1:
            B_true = B_true[None, :]
        n_states = min(B_hat.shape[0], B_true.shape[0])
        assert n_states >= 1, "Need at least one B-state for correlation plot"

        figId = self.smart_append(figId)
        ncol = 1 + n_states
        fig_w = max(10.0, 3.0 * ncol)
        fig = self.plt.figure(figId, facecolor='white', figsize=(fig_w, 3.8))

        ax = self.plt.subplot(1, ncol, 1)
        self._plot_corr_A_regions(
            ax, A_true.ravel(), A_hat.ravel(), minW,
            "A fit, non-zero ,", "A_true", "A_hat", s=6, alpha=0.4, color="green"
        )

        for m in range(n_states):
            ax = self.plt.subplot(1, ncol, 2 + m)
            self._plot_corr_B_divisor(
                ax, B_true[m], B_hat[m],
                f"B fit, state {m}", "B_true", "B_hat",
                s=8, alpha=0.5, color="blue"
            )

        fig.suptitle(f"2D correlations, minW={minW:g}: {md['short_name']}", fontsize=12)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    def state_seq_prismEM(self, fitD, md, figId=5, time_range_sec=None):
        """4-row state-sequence canvas in the style of prism_Estep_eval -p b."""
        trainMD = md["train"]
        dt = float(trainMD["time_step_sec"])
        t0_bin, t1_bin = [int(x) for x in trainMD["time_range_bins"]]

        t0_req, t1_req = [float(x) for x in time_range_sec]
        if t1_req < t0_req:
            t0_req, t1_req = t1_req, t0_req
        b0 = max(t0_bin, int(np.floor(t0_req / dt)))
        b1 = min(t1_bin, int(np.floor(t1_req / dt)))
        if b1 <= b0:
            raise ValueError("Requested --time_range_sec leaves no bins in fitted window")

        i0 = b0 - t0_bin
        i1 = b1 - t0_bin
        p0 = i0
        p1 = i1

        S_hat = np.asarray(fitD["S_hat"])[i0:i1 + 1]
        C_hat = np.asarray(fitD["c_hat"])[i0:i1 + 1]
        S_hat_CL = np.asarray(fitD["S_hat_CL"])[i0:i1 + 1]

        S_true = np.asarray(md["S_true"])[b0:b1 + 1]
        C_true = np.asarray(md["C_true"])[b0:b1 + 1]

        nll_t = np.asarray(md["eval_f"]["loss_nll_time"])[p0:p1]
        l2_t = np.asarray(md["eval_f"]["loss_l2_time"])[p0:p1]
        t_bins = np.arange(b0, b1 + 1, dtype=np.float64) * dt
        t_pairs = np.arange(b0, b1, dtype=np.float64) * dt
        x0, x1 = float(t_bins[0]), float(t_bins[-1])

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(12, 9))
        # Add an explicit spacer row between diagnostics (top row) and Fit row
        # so top-row x labels do not collide with Fit legends.
        gs = fig.add_gridspec(
            5, 3,
            height_ratios=[0.95, 0.18, 1.0, 1.0, 0.55],
            hspace=0.75,
            wspace=0.35,
        )

        # Row 2: Fit
        ax = fig.add_subplot(gs[2, :])
        ax.plot(t_bins, S_hat, color="k", linewidth=2.5, label="S_hat")
        ax.plot(t_bins, S_true, color="lime", linestyle="--", linewidth=1.5, label="S_true")
        lo = np.clip(S_hat - S_hat_CL, 0.0, float(C_hat.shape[1] - 1))
        hi = np.clip(S_hat + S_hat_CL, 0.0, float(C_hat.shape[1] - 1))
        ax.fill_between(t_bins, lo, hi, color="gray", alpha=0.3, label="S_hat_CL")
        ax.set(title=f"Fit (acc={float(md['eval_f']['acc']):.3f})", xlabel="time (s)", ylabel="state")
        ax.set_xlim(x0, x1)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        for m in range(C_hat.shape[1]):
            ax2.plot(t_bins, C_hat[:, m], linewidth=0.8, alpha=0.85, label=f"C_hat[{m}]")
        ax2.set_ylabel("C_hat")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=8)
        ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=4, fontsize=8)

        # Row 3: Truth
        ax = fig.add_subplot(gs[3, :])
        ax.plot(t_bins, S_true, color="lime",  linestyle="--",linewidth=1.5, label="S_true")
        ax.set(title="Truth", xlabel="time (s)", ylabel="state")
        ax.set_xlim(x0, x1)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        for m in range(C_true.shape[1]):
            ax2.plot(t_bins, C_true[:, m], linewidth=0.8, alpha=0.85, label=f"C_true[{m}]")
        ax2.set_ylabel("C_true")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=8)
        ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=4, fontsize=8)

        # Row 1: diagnostics panels (occupancy, state accuracy, confidence)
        srec = md["states_recovery_eval"]
        acc_avg = float(srec["avg_acc"])
        state_acc_cl = np.asarray(srec["state_acc_cl"], dtype=np.float64)
        acc_ps = state_acc_cl[:, 0]
        cl_ps = state_acc_cl[:, 1]

        ax = fig.add_subplot(gs[0, 0])
        occ = C_hat.mean(axis=0)
        x = np.arange(C_hat.shape[1], dtype=np.int64)
        ax.bar(x, occ, color="tab:blue", alpha=0.65)
        ax.set(title="Mean occupancy", xlabel="state", ylabel="mean c")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
        ax.set_box_aspect(0.9)

        ax = fig.add_subplot(gs[0, 1])
        x = np.arange(acc_ps.size, dtype=np.int64)
        ax.bar(x, acc_ps, color="tab:green", alpha=0.7)
        ax.errorbar(x, acc_ps, yerr=cl_ps, fmt='o', color='k', capsize=5, markersize=4)
        ax.axhline(1.0, color="green", linestyle="--", linewidth=1)
        ymin = min(0.5, float(np.nanmin(acc_ps - cl_ps)) - 0.05)
        ax.set_ylim(ymin, None)
        ax.set(title=f"State accuracy, avr={acc_avg:.3f}", xlabel="state", ylabel="accuracy")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3, axis="y")
        ylo, yhi = ax.get_ylim()
        y_txt = ylo + 0.80 * (yhi - ylo)
        for i, v in enumerate(acc_ps):
            ax.text(i + 0.3, y_txt, f"{v:.2f}", ha="center", va="center", fontsize=9)
        ax.set_box_aspect(0.9)

        ax = fig.add_subplot(gs[0, 2])
        cl = np.asarray(S_hat_CL, dtype=np.float64)
        ax.hist(cl, bins=30, color="tab:purple", alpha=0.7)
        ax.axvline(np.mean(cl), color="k", linestyle="--", linewidth=1.0)
        ax.set(title=f"Confidence (acc={acc_avg:.3f})", xlabel="S_hat_CL", ylabel="count")
        ax.grid(True, alpha=0.3)
        ax.set_box_aspect(0.9)

        # Row 4: Loss(time)
        ax = fig.add_subplot(gs[4, :])
        ax.plot(t_pairs, nll_t, color="tab:blue", linewidth=0.9, label="nll")
        ax.plot(t_pairs, l2_t, color="tab:orange", linewidth=0.9, label="l2")
        ax.set_yscale("log")
        ax.set(title="Loss (time)", xlabel="time (s)", ylabel="value")
        ax.set_xlim(x0, x1)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(t_bins, S_true, color="k", linewidth=0.8, alpha=0.6, label="S_true")
        ax2.set_ylabel("state")
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), fontsize=8)

        fig.suptitle(
            f"Dataset: {md['short_name']} | λ₂={trainMD['lambda2']}, lr={trainMD['lr_estep']}, "
            f"pgd_iter={trainMD['pgd_iter']}, decode_dwell={srec['decode_dwell_sec']}s",
            fontsize=12,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
