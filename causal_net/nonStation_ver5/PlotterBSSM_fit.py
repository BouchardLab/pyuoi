#!/usr/bin/env python3
"""
Plotting utilities for BSSM-STD block-coordinate fit evaluation.
"""

import os
import warnings
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

from toolbox.PlotterBackbone import PlotterBackbone


def _scalar(arr):
    return float(np.asarray(arr).reshape(-1)[0])


def _offdiag(W):
    W = np.asarray(W)
    assert W.ndim == 2 and W.shape[0] == W.shape[1], "W must be square"
    return W[~np.eye(W.shape[0], dtype=bool)]


def _corr(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    assert x.shape == y.shape and x.size >= 2, "correlation arrays must match and be nontrivial"
    if np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _sym_limit(*arrs):
    vmax = 0.0
    for arr in arrs:
        if arr is None:
            continue
        val = float(np.nanmax(np.abs(np.asarray(arr))))
        vmax = max(vmax, val)
    return max(vmax, 1e-12)


class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self, args)

    def _heat(self, ax, mat, title, cmap="coolwarm", vlim=None):
        mat = np.asarray(mat)
        assert mat.ndim == 2, "heatmap input must be 2D"
        if vlim is None:
            im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
        else:
            im = ax.imshow(
                mat,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim),
            )
        ax.set(title=title, xlabel="pre neuron j", ylabel="post neuron i")
        self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return im

    def two_block_summary(self, fitD, diag, figId=1):
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(14, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.34)

        outer = np.asarray(fitD["outer_idx"], dtype=np.int32) + 1
        U = np.asarray(fitD["U_outer"], dtype=np.float64)
        tau_rec = np.asarray(fitD["tau_rec_outer"], dtype=np.float64)
        nll_A = np.asarray(fitD["nll_after_blockA"], dtype=np.float64)
        nll_B = np.asarray(fitD["nll_after_std_grid"], dtype=np.float64)
        nll = np.asarray(fitD["nll_outer"], dtype=np.float64)
        blockB_executed = np.asarray(fitD["blockB_executed"], dtype=np.int32).astype(bool)
        rho = np.asarray(fitD["spectral_radius_outer"], dtype=np.float64)
        nz = np.asarray(fitD["nz_weight_outer"], dtype=np.int64)
        block_outer = np.asarray(fitD["blockA_outer"], dtype=np.int32) + 1
        block_lr = np.asarray(fitD["blockA_lr"], dtype=np.float64)

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(outer, nll_A, "o-", label="after Block A", color="tab:blue")
        ax.plot(outer, nll, "s-", label="outer result", color="tab:green")
        ax.plot(outer[blockB_executed], nll_B[blockB_executed], "^-", label="after Block B", color="tab:orange")
        ax.set(title="Outer mean Bernoulli NLL", xlabel="outer iteration", ylabel="mean NLL")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        lr_outer = np.empty_like(outer, dtype=np.float64)
        for i, o in enumerate(outer):
            m = block_outer == o
            assert np.any(m), "missing Block A LR values for outer %d" % int(o)
            lr_outer[i] = block_lr[np.where(m)[0][-1]]
        ax_lr = ax.twinx()
        ax_lr.plot(outer, lr_outer, "d--", color="tab:red", lw=1.0, ms=4, label="Block A LR")
        ax_lr.set_ylabel("Block A LR", color="tab:red")
        ax_lr.tick_params(axis="y", labelcolor="tab:red")
        lines, labels = ax.get_legend_handles_labels()
        lines_lr, labels_lr = ax_lr.get_legend_handles_labels()
        ax.legend(lines + lines_lr, labels + labels_lr, fontsize=8)

        ax = fig.add_subplot(gs[0, 1])
        ax.plot(outer, U, "o-", color="tab:purple", label="U")
        ax.set(title=r"Block B STD parameter path", xlabel="outer iteration", ylabel=r"$U$")
        ax.tick_params(axis="y", labelcolor="tab:purple")
        ax.yaxis.label.set_color("tab:purple")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2 = ax.twinx()
        ax2.plot(outer, tau_rec, "s-", color="tab:brown", label=r"$\tau_{rec}$")
        ax2.set_ylabel(r"$\tau_{rec}$ (s)", color="tab:brown")
        ax2.tick_params(axis="y", labelcolor="tab:brown")

        ax = fig.add_subplot(gs[0, 2])
        ep = np.arange(1, block_outer.size + 1)
        bce = np.asarray(fitD["blockA_bce_loss"], dtype=np.float64)
        l1 = np.asarray(fitD["blockA_l1_loss"], dtype=np.float64)
        total = np.asarray(fitD["blockA_loss"], dtype=np.float64)
        ax.plot(ep, bce, color="tab:blue", lw=1.2, label="BCE")
        ax.plot(ep, total, color="k", lw=1.0, alpha=0.75, label="BCE+L1")
        ax.set(title="Block A objective", xlabel="global Block A epoch", ylabel=r"$\mathcal{L}_A$")
        ax.grid(True, alpha=0.3)
        ax_l1 = ax.twinx()
        ax_l1.plot(ep, l1, color="tab:red", lw=1.0, ls="--", label="L1")
        ax_l1.set_ylabel("L1", color="tab:red")
        ax_l1.tick_params(axis="y", labelcolor="tab:red")
        for o in np.unique(block_outer)[1:]:
            x = int(np.where(block_outer == o)[0][0]) + 1
            ax.axvline(x, color="gray", lw=0.6, alpha=0.35)
        ax.legend(fontsize=8, loc="upper right")

        ax = fig.add_subplot(gs[1, 0])
        rho_max = _scalar(fitD["rho_max"])
        ax.plot(outer, rho, "o-", color="tab:green")
        ax.axhline(rho_max, color="tab:red", lw=1.0, ls="--", label=r"$\rho_{max}$")
        ax.set(title="Spectral radius", xlabel="outer iteration", ylabel=r"$\rho(W)$")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[1, 1])
        ax.plot(outer, nz, "o-", color="tab:cyan")
        ax.set(
            title="Reported nonzero W count",
            xlabel="outer iteration",
            ylabel="count",
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 2])
        ax.axis("off")
        text = (
            f"N={int(_scalar(fitD['num_neurons']))}\n"
            f"eval bins={int(_scalar(fitD['num_eval_bins']))}\n"
            f"dt={_scalar(fitD['time_step_sec']):.4g} s\n"
            f"burn bins={int(_scalar(fitD['burn_bins']))}\n"
            f"init bins={int(_scalar(fitD['num_init_bins']))}\n"
            f"mean |W_init|={_scalar(fitD['W_init_abs_mean']):.3g}\n"
            f"M={len(fitD['kappa_fit'])}, alpha={_scalar(fitD['alpha_fit']):.5f}\n"
            f"lambda_l1={_scalar(fitD['lambda_l1']):.3g}\n"
            f"eta_clip={_scalar(fitD['eta_clip']):.3g}\n"
            f"final U={U[-1]:.5f}\n"
            f"final tau_rec={tau_rec[-1]:.5f} s\n"
            f"final NLL={nll[-1]:.6f}"
        )
        ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left", fontsize=10)
        fig.suptitle(f"BSSM-STD Two-Block Fit Summary: {diag['short_name']}", fontsize=14)

    def parameter_init_vs_fit(self, fitD, diag, figId=2):
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(15, 8.5))
        gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

        W0 = np.asarray(fitD["W_init"], dtype=np.float64)
        W = np.asarray(fitD["W_fit"], dtype=np.float64)
        B0 = np.asarray(fitD["B_init"], dtype=np.float64)
        B = np.asarray(fitD["B_fit"], dtype=np.float64)
        assert W0.shape == W.shape, "W_init/W_fit shape mismatch"
        assert B0.shape == B.shape == (W.shape[0],), "B_init/B_fit shape mismatch"

        vlim = _sym_limit(W0, W)
        self._heat(fig.add_subplot(gs[0, 0]), W0, "W_init", vlim=vlim)
        self._heat(fig.add_subplot(gs[0, 1]), W, "W_fit", vlim=vlim)
        self._heat(fig.add_subplot(gs[0, 2]), W - W0, "W_fit - W_init", vlim=_sym_limit(W - W0))

        ax = fig.add_subplot(gs[1, 0])
        rB = _corr(B0, B)
        ax.scatter(B0, B, s=14, alpha=0.75, color="tab:blue", edgecolors="none")
        lo = min(float(B0.min()), float(B.min()))
        hi = max(float(B0.max()), float(B.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set(title=f"$B_i$ init vs fit, r={rB:.3f}", xlabel="B_init", ylabel="B_fit")
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 1])
        w0 = _offdiag(W0)
        wf = _offdiag(W)
        rW = _corr(w0, wf)
        ax.scatter(w0, wf, s=5, alpha=0.25, color="tab:green", edgecolors="none")
        xspan = max(float(np.max(np.abs(w0))), 1e-12)
        yspan = max(float(np.max(np.abs(wf))), 1e-12)
        ax.set_xlim(float(w0.min()) - 0.05 * xspan, float(w0.max()) + 0.05 * xspan)
        ax.set_ylim(float(wf.min()) - 0.05 * yspan, float(wf.max()) + 0.05 * yspan)
        ax.axvline(0.0, color="k", lw=0.7, alpha=0.45)
        ax.axhline(0.0, color="k", lw=0.7, alpha=0.45)
        ax.set(
            title=f"Eq.17 offdiag $W_{{ij}}^{{(0)}}$ vs fit, r={rW:.3f}",
            xlabel="raw lag-1 covariance W_init",
            ylabel="W_fit",
        )
        ax.text(
            0.03,
            0.97,
            "mean |W_init|=%.3g\nmean |W_fit|=%.3g" % (np.mean(np.abs(w0)), np.mean(np.abs(wf))),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 2])
        ax.hist(w0, bins=80, histtype="step", lw=1.3, color="gray", label="W_init")
        ax.hist(wf, bins=80, histtype="step", lw=1.3, color="tab:green", label="W_fit")
        ax.set_yscale("log")
        ax.set(title=r"Off-diagonal $W_{ij}$ distribution", xlabel="weight", ylabel="count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.suptitle(f"BSSM-STD $B_i,W_{{ij}}$ Initialization and Fit: {diag['short_name']}", fontsize=14)

    def blockB_joint_grid(self, fitD, diag, figId=3):
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(15, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

        outer = np.asarray(fitD["joint_search_outer"], dtype=np.int32) + 1
        refine = np.asarray(fitD["joint_search_refine"], dtype=np.int32)
        eval_id = np.asarray(fitD["joint_search_eval"], dtype=np.int32)
        eval_total = np.asarray(fitD["joint_search_eval_total"], dtype=np.int32)
        U = np.asarray(fitD["joint_search_U"], dtype=np.float64)
        tau_rec = np.asarray(fitD["joint_search_tau_rec"], dtype=np.float64)
        nll = np.asarray(fitD["joint_search_nll"], dtype=np.float64)
        accepted_outer = np.asarray(fitD["outer_idx"], dtype=np.int32) + 1
        accepted_U = np.asarray(fitD["U_outer"], dtype=np.float64)
        accepted_tau = np.asarray(fitD["tau_rec_outer"], dtype=np.float64)
        assert outer.size > 0, "joint search arrays are empty"
        assert outer.shape == refine.shape == eval_id.shape == eval_total.shape == U.shape == tau_rec.shape == nll.shape
        assert accepted_outer.shape == accepted_U.shape == accepted_tau.shape, "outer STD trajectory shape mismatch"

        rows = []
        for o in np.unique(outer):
            for r in np.unique(refine[outer == o]):
                m = (outer == o) & (refine == r)
                idx = np.where(m)[0][np.argmin(nll[m])]
                rows.append((o, r, U[idx], tau_rec[idx], nll[idx], int(np.sum(m))))
        rows = np.asarray(rows, dtype=np.float64)

        ax = fig.add_subplot(gs[0, 0])
        for r in np.unique(rows[:, 1].astype(np.int32)):
            m = rows[:, 1] == r
            ax.plot(rows[m, 0], rows[m, 2], "o-", label=f"refine {r}")
        ax.set(title="Block B best U per grid round", xlabel="outer", ylabel=r"$U$")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[0, 1])
        for r in np.unique(rows[:, 1].astype(np.int32)):
            m = rows[:, 1] == r
            ax.plot(rows[m, 0], rows[m, 3], "o-", label=f"refine {r}")
        ax.set(title=r"Block B best $\tau_{rec}$ per grid round", xlabel="outer", ylabel=r"$\tau_{rec}$ (s)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[0, 2])
        for r in np.unique(rows[:, 1].astype(np.int32)):
            m = rows[:, 1] == r
            ax.plot(rows[m, 0], rows[m, 4], "o-", label=f"refine {r}")
        ax.set(title="Best Block B grid NLL", xlabel="outer", ylabel="mean NLL")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        last_outer = int(np.max(outer))
        last_refine = int(np.max(refine[outer == last_outer]))
        m = (outer == last_outer) & (refine == last_refine)
        ax = fig.add_subplot(gs[1, 0])
        sc = ax.scatter(U[m], tau_rec[m], c=nll[m], s=70, cmap="viridis", edgecolors="k", linewidths=0.2)
        ax.plot(
            accepted_U,
            accepted_tau,
            "o-",
            color="black",
            lw=1.2,
            ms=4,
            label="outer path",
            zorder=5,
        )
        for oo, uu, tt in zip(accepted_outer, accepted_U, accepted_tau):
            ax.text(uu, tt, str(int(oo)), fontsize=7, color="black", ha="left", va="bottom")
        best = np.where(m)[0][np.argmin(nll[m])]
        ax.plot(U[best], tau_rec[best], "r*", ms=16, label="best")
        ax.set(
            title=f"Block B candidates outer {last_outer}, refine {last_refine}",
            xlabel=r"$U$",
            ylabel=r"$\tau_{rec}$ (s)",
        )
        self.plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="mean NLL")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 1])
        u_vals = np.unique(U[m])
        tau_vals = np.unique(tau_rec[m])
        Z = np.full((tau_vals.size, u_vals.size), np.nan, dtype=np.float64)
        for uu, tt, nn in zip(U[m], tau_rec[m], nll[m]):
            iu = int(np.where(u_vals == uu)[0][0])
            it = int(np.where(tau_vals == tt)[0][0])
            if not np.isfinite(Z[it, iu]) or nn < Z[it, iu]:
                Z[it, iu] = nn
        im = ax.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=[u_vals.min(), u_vals.max(), tau_vals.min(), tau_vals.max()],
            cmap="viridis",
        )
        ax.set(title="Final Block B refine NLL surface", xlabel=r"$U$", ylabel=r"$\tau_{rec}$ (s)")
        self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = fig.add_subplot(gs[1, 2])
        ax.hist(nll - np.min(nll), bins=60, color="tab:purple", alpha=0.75)
        ax.set(title="All Block B candidates above best", xlabel=r"$\Delta$NLL", ylabel="count")
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"BSSM-STD Block B Joint $U,\\tau_{{rec}}$ Grid: {diag['short_name']}", fontsize=14)

    def bernoulli_observation_diagnostics(self, fitD, diag, figId=4):
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(15, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

        obs = np.asarray(diag["obs_rate_hz"], dtype=np.float64)
        model = np.asarray(diag["model_rate_hz"], dtype=np.float64)
        nll = np.asarray(diag["nll_neuron"], dtype=np.float64)
        single_rates = np.asarray(diag["single_rates_hz"], dtype=np.float64)
        resid = model - obs

        ax = fig.add_subplot(gs[0, 0])
        r = _corr(obs, model)
        ax.scatter(obs, model, s=16, alpha=0.75, color="tab:blue", edgecolors="none")
        lo = min(float(obs.min()), float(model.min()))
        hi = max(float(obs.max()), float(model.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set(title=f"Model vs observed rates, r={r:.3f}", xlabel="observed rate (Hz)", ylabel="model rate (Hz)")
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[0, 1])
        ax.hist(resid, bins=40, color="tab:orange", alpha=0.65, label="model - fit window obs")
        ax.hist(obs - single_rates, bins=40, histtype="step", color="tab:gray", lw=1.3, label="window obs - full data")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set(title="Rate residuals", xlabel="rate difference (Hz)", ylabel="neurons")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[0, 2])
        ax.plot(np.arange(nll.size), nll, color="tab:green")
        ax.set(title="Per-neuron mean Bernoulli NLL", xlabel="neuron index", ylabel="NLL")
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 0])
        calib_p = np.asarray(diag["calib_p"], dtype=np.float64)
        calib_obs = np.asarray(diag["calib_obs"], dtype=np.float64)
        calib_count = np.asarray(diag["calib_count"], dtype=np.int64)
        valid = calib_count > 0
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.scatter(calib_p[valid], calib_obs[valid], s=np.clip(calib_count[valid] / calib_count[valid].max() * 120, 20, 120),
                   color="tab:red", alpha=0.75, edgecolors="none")
        ax.set(title="Bernoulli probability calibration", xlabel=r"mean model $\tilde{p}_{i,t}$", ylabel="observed spike fraction")
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 1])
        edges = np.asarray(diag["p_hist_edges"], dtype=np.float64)
        cnt = np.asarray(diag["p_hist_count"], dtype=np.int64)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, cnt, width=np.diff(edges), align="center", color="tab:blue", alpha=0.75)
        ax.set_yscale("log")
        ax.set(title=r"Model probability distribution", xlabel=r"$\tilde{p}_{i,t}$", ylabel="count")
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 2])
        B = np.asarray(fitD["B_fit"], dtype=np.float64)
        x = np.log(np.clip(obs * _scalar(fitD["time_step_sec"]), 1e-9, 1.0 - 1e-9))
        ax.scatter(x, B, s=16, alpha=0.75, color="tab:purple", edgecolors="none")
        rr = _corr(x, B)
        ax.set(title=f"B_fit vs log observed spike prob, r={rr:.3f}", xlabel="log(obs p/bin)", ylabel="B_fit")
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"BSSM-STD Bernoulli Observation Diagnostics: {diag['short_name']}  recomputed NLL={diag['nll_eval']:.6f}",
            fontsize=14,
        )
