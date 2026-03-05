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
        fig = self.plt.figure(figId, facecolor='white', figsize=(14, 7.5))
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

        prune_em = trainMD["L1_prune_em_iter"]
        prune_m_epoch = prune_em * m_per_em

        # ── Row 1, Col 1: E-step NLL vs EM iteration ────────────────
        ax = self.plt.subplot(2, 3, 1)
        ax.plot(em_iters, e_nll, 'o-', color='tab:blue', markersize=3,
                linewidth=1.2)
        ax.set(title="E-step weighted NLL", xlabel="EM iteration",
               ylabel="NLL / bin")
        ax.grid(True, alpha=0.3)
        txt = (f"pgd_iter={trainMD['pgd_iter']}\n"
               f"lr_E={trainMD['lr_estep']}\n"
               f"λ₂={trainMD['lambda2']}")
        ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                va="top", ha="right", fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # ── Row 1, Col 2: M-step NLL + L1 vs M-epoch ────────────────
        ax = self.plt.subplot(2, 3, 2)
        ax.plot(m_epochs, m_nll, color='tab:blue', linewidth=1, label='NLL')
        ax.set_ylabel('NLL', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax.set(title="M-step loss", xlabel="M-epoch (global)")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(m_epochs, m_l1, color='tab:red', linewidth=1,
                 linestyle='--', label='L1')
        ax2.set_ylabel('L1', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        if prune_m_epoch > 0 and prune_m_epoch < n_m_total:
            ax.axvline(prune_m_epoch, color='k', ls=':', lw=0.8,
                       label=f'prune@{prune_em}')

        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc='upper right')

        txt = (f"lr_M={trainMD['lr_mstep']}\n"
               f"L1α={trainMD['L1_alpha']}\n"
               f"batch={trainMD['batch_size']}")
        ax.text(0.03, 0.03, txt, transform=ax.transAxes,
                va="bottom", ha="left", fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # ── Row 1, Col 3: spectral radius vs M-epoch ────────────────
        ax = self.plt.subplot(2, 3, 3)
        rho_max = trainMD["rho_max"]
        ax.plot(m_epochs, rho, color='tab:green', linewidth=1,
                label='ρ(A)')
        ax.axhline(rho_max, color='red', ls='--', lw=1,
                   label=f'ρ_max={rho_max}')
        ax.set(title="Spectral radius ρ(A)", xlabel="M-epoch (global)",
               ylabel="ρ")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # ── Row 2, Col 1: non-zero off-diag edges vs M-epoch ────────
        ax = self.plt.subplot(2, 3, 4)
        ax.plot(m_epochs, nz, color='tab:purple', linewidth=1)
        if prune_m_epoch > 0 and prune_m_epoch < n_m_total:
            ax.axvline(prune_m_epoch, color='k', ls=':', lw=0.8)
        ax.set(title=f"Non-zero off-diag edges (|A|>{trainMD['minW']})",
               xlabel="M-epoch (global)", ylabel="count")
        ax.grid(True, alpha=0.3)

        # learning rate on twin axis
        ax2 = ax.twinx()
        ax2.plot(m_epochs, lr, color='tab:orange', linewidth=0.8,
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

    def _scatter_true_vs_cmp(self, ax, x_true, y_cmp, title, color):
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
        ll_gap = np.asarray(md["eval_f"]["ll_gap"])[p0:p1]

        t_bins = np.arange(b0, b1 + 1, dtype=np.float64) * dt
        t_pairs = np.arange(b0, b1, dtype=np.float64) * dt
        x0, x1 = float(t_bins[0]), float(t_bins[-1])

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(12, 9))
        gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 0.55, 0.55], hspace=0.85)

        # Row 1: Fit
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(t_bins, S_hat, color="k", linewidth=1.0, label="S_hat")
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

        # Row 2: Truth
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(t_bins, S_true, color="k", linewidth=1.0, label="S_true")
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

        # Row 3: Loss(time)
        ax = fig.add_subplot(gs[2, 0])
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

        # Row 4: LL gap
        ax = fig.add_subplot(gs[3, 0])
        ax.plot(t_pairs, ll_gap, color="tab:blue", linewidth=0.9, label="LL gap")
        ax.set(title="LL gap (best - 2nd)", xlabel="time (s)", ylabel="value")
        ax.set_xlim(x0, x1)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=8)

        srec = md["states_recovery_eval"]
        fig.suptitle(
            f"Dataset: {md['short_name']} | λ₂={trainMD['lambda2']}, lr={trainMD['lr_estep']}, "
            f"pgd_iter={trainMD['pgd_iter']}, decode_dwell={srec['decode_dwell_sec']}s",
            fontsize=12,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
