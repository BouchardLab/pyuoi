#!/usr/bin/env python3
"""Plotting utilities for prism EM 3c evaluation."""

import numpy as np

from toolbox.PlotterBackbone import PlotterBackbone
from UtilBioExp import clip_rebD_time


def real_fit_metadata(md):
    """Return the real-fit metadata block for ordinary or FDR-aggregated fits."""
    if is_stage_c_metadata(md):
        return md
    if "bagsFDR_stageA" in md and "real_fit" in md["bagsFDR_stageA"]:
        return md["bagsFDR_stageA"]["real_fit"]
    return md


def is_stage_c_metadata(md):
    return md.get("fit_type") == "prismEM_deBias_stageC" or "deBias_stageC" in md


def fit_stage_label(md):
    if is_stage_c_metadata(md):
        mode = real_fit_metadata(md).get("train", {}).get("state_mode", "?")
        return f"Stage (c) deBias, state_mode={mode}"
    if "bagsFDR_stageB" in md:
        return "Stage (b) FDR aggregate"
    if md.get("fit_type", "").startswith("prismEM_FDRbag") or "bagsFDR_stageA" in md:
        return "Stage (a) FDR bag"
    return "PRISM-EM"


def neuron_type_marker(neuron_type):
    """Marker convention for excitatory, inhibitory, and undetermined units."""
    return {1: "o", -1: "^", 0: "s"}[int(neuron_type)]


def neuron_type_marker_size(neuron_type, base_size):
    """Make inhibitory triangles 30 percent larger than other markers."""
    return 1.3 * base_size if int(neuron_type) == -1 else base_size


class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self, args)

    def figId2name(self, fid):
        if isinstance(fid, str):
            return f"{self.jobName}_{fid}"
        return f"{self.jobName}_f{fid}"

    def smart_append(self, fig_id):
        if fig_id in self.figL:
            raise ValueError(f"Figure id {fig_id!r} is already in use")
        self.figL.append(fig_id)
        return fig_id

    def canvas_title_prefix(self, md):
        data_label = "exper" if md.get("data_type") == "bioExp" else "simu"
        return f"{data_label} {md['short_name']}, {fit_stage_label(md)}"

    def _display_A_label(self, md, prune=False):
        if is_stage_c_metadata(md):
            return "A_debias"
        return "A_prune" if prune else "A_hat"

    def _display_B_label(self, md):
        return "B_debias" if is_stage_c_metadata(md) else "B_hat"

    def summary(self, fitD, md, figId="a"):
        """EM convergence overview."""
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(12, 6))
        fig.subplots_adjust(hspace=0.45, wspace=0.35)

        trainMD = real_fit_metadata(md)["train"]
        title_prefix = self.canvas_title_prefix(md)
        is_stage_c = is_stage_c_metadata(md)

        e_nll = np.asarray(fitD["e_nll_em"])
        m_nll = np.asarray(fitD["m_nll_epoch"])
        m_l1 = np.asarray(fitD["m_l1_epoch"])
        rho = np.asarray(fitD["rho_epoch"])
        nz = np.asarray(fitD["nz_edges_epoch"])
        lr = np.asarray(fitD["learning_rates"])

        n_em = int(trainMD.get("num_em_iters", 0))
        m_per_em = int(trainMD["m_epochs"])
        n_m_total = len(m_nll)
        m_epochs = np.arange(1, n_m_total + 1)
        em_iters = np.arange(1, len(e_nll) + 1)
        is_locked_fdr = n_em < 1 or md.get("fit_type", "").startswith("prismEM_FDRbags")

        prune_em = int(trainMD["delay_em_iter_4_Aprune"])
        rho_start_em = int(trainMD["delay_em_iter_4_ArhoMax"])
        lr_drop_em = int(trainMD["delay_em_iter_4_lrDecay"])
        prune_m_epoch = prune_em * m_per_em
        rho_start_m_epoch = rho_start_em * m_per_em
        lr_drop_m_epoch = lr_drop_em * m_per_em

        def draw_threshold_marker(ax, x_pos, x_max, txt, color, yFac=0.02):
            if not (0 < x_pos <= x_max):
                return
            ax.axvline(x_pos, color=color, ls="--", lw=0.9, alpha=0.95)
            y0, y1 = ax.get_ylim()
            x0, x1 = ax.get_xlim()
            x_off = 0.01 * max(1e-9, x1 - x0)
            y_txt = y1 - yFac * (y1 - y0)
            ax.text(
                x_pos + x_off, y_txt, txt, rotation=90, color=color, fontsize=7,
                ha="left", va="top",
                bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=0.2),
            )

        ax = self.plt.subplot(2, 3, 1)
        if len(e_nll) > 0:
            ax.plot(em_iters, e_nll, "o-", color="tab:blue", markersize=3, linewidth=1.2)
            ax.set(title="E-step NLL", xlabel="EM iteration", ylabel="NLL / bin")
        else:
            if is_stage_c:
                txt = (
                    "Stage (c) de-biased fit\n"
                    f"state_mode={trainMD.get('state_mode', '?')}\n"
                    "display: A_debias, B_debias"
                )
            else:
                txt = "FDR locked M-step fit\nstate labels from reference EM\nno bag E-step"
            if "bagsFDR_stageB" in md:
                stg = md["bagsFDR_stageB"]
                txt += f"\nbags={stg.get('num_bags', '?')}  stab_sel_thresh={stg.get('stab_sel_thresh', '?')}"
            ax.text(0.5, 0.55, txt, transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, bbox=dict(facecolor="white", alpha=0.75, edgecolor="0.8"))
            ax.set(title="Fit mode", xlabel="", ylabel="")
            ax.set_xticks([])
            ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        draw_threshold_marker(ax, prune_em, len(e_nll), "start Aprune", "k")
        draw_threshold_marker(ax, rho_start_em, len(e_nll), "start rhoMax", "tab:brown")
        draw_threshold_marker(ax, lr_drop_em, len(e_nll), "start_lrDrop", "tab:gray", yFac=0.6)
        txt = (f"pgd_iter={trainMD['pgd_iter']}\n"
               f"lr_E={trainMD['lr_estep']}\n"
               f"lambda2={trainMD['lambda2']}")
        ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                va="top", ha="right", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        ax = self.plt.subplot(2, 3, 2)
        ax.plot(m_epochs, m_nll, color="tab:blue", linewidth=1, label="NLL")
        ax.set_ylabel("NLL", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.set(title="M-step loss", xlabel="M-epoch (global)")
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(m_epochs, m_l1, color="tab:red", linewidth=1, linestyle="--", label="L1")
        ax2.set_ylabel("L1", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc="upper right")
        draw_threshold_marker(ax, prune_m_epoch, n_m_total, "start Aprune", "k")
        draw_threshold_marker(ax, rho_start_m_epoch, n_m_total, "start rhoMax", "tab:brown")
        draw_threshold_marker(ax, lr_drop_m_epoch, n_m_total, "start_lrDrop", "tab:gray", yFac=0.4)
        txt = (f"lr_M={trainMD['lr_mstep']}\n"
               f"L1 lambda3={trainMD['lambda3']}\n"
               f"batch={trainMD['batch_size']}")
        ax.text(0.03, 0.03, txt, transform=ax.transAxes,
                va="bottom", ha="left", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        ax = self.plt.subplot(2, 3, 3)
        rho_max = float(trainMD["rho_max"])
        ax.plot(m_epochs, rho, color="tab:green", linewidth=1, label="rho(A)")
        ax.axhline(rho_max, color="red", ls="--", lw=1, label=f"rho_max={rho_max:g}")
        ax.set(title="Spectral radius rho(A)", xlabel="M-epoch (global)", ylabel="rho")
        ax.grid(True, alpha=0.3)
        draw_threshold_marker(ax, prune_m_epoch, n_m_total, "start Aprune", "k")
        draw_threshold_marker(ax, rho_start_m_epoch, n_m_total, "start rhoMax", "tab:brown")
        draw_threshold_marker(ax, lr_drop_m_epoch, n_m_total, "start_lrDrop", "tab:gray", yFac=0.4)
        ax.legend(fontsize=8)

        ax = self.plt.subplot(2, 3, 4)
        ax.plot(m_epochs, nz, color="tab:purple", linewidth=1)
        ax.set(title="Non-zero off-diag edges", xlabel="M-epoch (global)", ylabel="count")
        ax.grid(True, alpha=0.3)
        draw_threshold_marker(ax, prune_m_epoch, n_m_total, "start Aprune", "k")
        draw_threshold_marker(ax, rho_start_m_epoch, n_m_total, "start rhoMax", "tab:brown")
        draw_threshold_marker(ax, lr_drop_m_epoch, n_m_total, "start_lrDrop", "tab:gray", yFac=0.4)
        ax2 = ax.twinx()
        ax2.plot(m_epochs, lr, color="tab:orange", linewidth=0.8, linestyle="--", alpha=0.6)
        ax2.set_ylabel("learning rate", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        ax = self.plt.subplot(2, 3, 5)
        c_hat = np.asarray(fitD["c_hat"])
        occ = c_hat.mean(axis=0)
        n_state = len(occ)
        ax.bar(np.arange(n_state), occ, color="tab:blue", alpha=0.7)
        ax.set(title="Mean occupancy", xlabel="state", ylabel="mean c")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(np.arange(n_state))
        ax.grid(True, alpha=0.3, axis="y")
        t0s, t1s = trainMD["time_range_sec"]
        txt = (f"N={trainMD['num_neurons']}  M={n_state}\n"
               f"T=[{t0s:.0f},{t1s:.0f}]s\n"
               f"bins={trainMD['num_time_bins']}")
        ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                va="top", ha="right", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        ax = self.plt.subplot(2, 3, 6)
        A_fit = np.asarray(fitD["A_hat"])
        diag_mask = np.eye(A_fit.shape[0], dtype=bool)
        A_off = A_fit[~diag_mask]
        A_off_nz = A_off[np.abs(A_off) > 1e-10]
        ax.hist(A_off_nz, bins=100, color="g", alpha=0.8)
        ax.set_yscale("log")
        ax.set(title=f"A off-diagonal, {A_off_nz.size} edges",
               xlabel="edge value", ylabel="edges")
        ax.grid(True, alpha=0.3)

        if is_stage_c:
            title_tail = (
                f"Stage (c) deBias: state_mode={trainMD.get('state_mode', '?')}, "
                f"K_epoch={n_m_total}"
            )
        elif is_locked_fdr:
            title_tail = f"FDR locked M-step: K_epoch={n_m_total}"
        else:
            title_tail = f"Prism EM: K_EM={n_em} x K_M={m_per_em}"
        fig.suptitle(f"{title_prefix}, {title_tail}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    def _draw_A_matrix(self, ax, A, title, num_exc=None, norm_map=None):
        import matplotlib.colors as colors

        A = np.asarray(A)
        if norm_map is None:
            vmin = float(np.min(A))
            vmax = float(np.max(A))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-9
            norm_map = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        im = ax.imshow(A, aspect=1.0, origin="lower", cmap="bwr",
                       norm=norm_map, interpolation="nearest")
        n_neuron = A.shape[0]
        if num_exc is not None:
            n_exc = int(num_exc)
            n_inh = n_neuron - n_exc
            ax.axhline(n_exc - 0.5, color="k", ls="--", lw=0.8)
            ax.axvline(n_exc - 0.5, color="k", ls="--", lw=0.8)
            ax.text(
                0.03, 0.97, f"exc={n_exc}\ninh={n_inh}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.3),
            )
        ax.plot([0, n_neuron], [0, n_neuron], "--", lw=0.8, color="magenta")
        ax.set_xlim(-0.5, n_neuron + 0.5)
        ax.set_ylim(-0.5, n_neuron + 0.5)
        ax.grid(True, alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("source / presynaptic neuron index (column)")
        ax.set_ylabel("target / postsynaptic neuron index (row)")
        self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _count_A_edges(self, A):
        A = np.asarray(A)
        n_neuron = A.shape[0]
        off_mask = ~np.eye(n_neuron, dtype=bool)
        n_diag = int(np.eye(n_neuron, dtype=bool).sum())
        n_off = int(np.count_nonzero(A[off_mask]))
        return n_neuron, n_diag, n_off

    def _A_prune_from_fitD(self, fitD):
        return np.asarray(fitD["A_prune"], dtype=np.float64)

    def _source_nz_edge_stats(self, A):
        A = np.asarray(A, dtype=np.float64)
        n_neuron = A.shape[0]
        strong = ~np.eye(n_neuron, dtype=bool) & (A != 0)
        cnt = strong.sum(axis=0).astype(np.int64)
        sum_src = np.where(strong, A, 0.0).sum(axis=0)
        return cnt, sum_src

    def _node_outgoing_edge_stats(self, A):
        return self._source_nz_edge_stats(A)

    def _node_incoming_edge_stats(self, A):
        A = np.asarray(A, dtype=np.float64)
        n_neuron = A.shape[0]
        strong = ~np.eye(n_neuron, dtype=bool) & (A != 0)
        cnt = strong.sum(axis=1).astype(np.int64)
        sum_target = np.where(strong, A, 0.0).sum(axis=1)
        return cnt, sum_target

    def _corrcoef_safe(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size == 0 or y.size == 0:
            return np.nan
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    def _add_x45_line(self, ax):
        lo = float(min(ax.get_xlim()[0], ax.get_ylim()[0]))
        hi = float(max(ax.get_xlim()[1], ax.get_ylim()[1]))
        ax.plot([lo, hi], [lo, hi], "--", color="k", linewidth=0.8, alpha=0.75)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    def _find_bimodal_divider(self, x_vals):
        x = np.asarray(x_vals, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        assert x.size >= 2, "Need at least two finite values for B split"
        if np.std(x) == 0:
            return float(np.median(x))
        c1, c2 = [float(v) for v in np.quantile(x, [0.25, 0.75])]
        if c1 == c2:
            c1, c2 = float(np.min(x)), float(np.max(x))
        for _ in range(32):
            left = np.abs(x - c1) <= np.abs(x - c2)
            if int(left.sum()) == 0 or int((~left).sum()) == 0:
                return float(np.median(x))
            c1_new = float(np.mean(x[left]))
            c2_new = float(np.mean(x[~left]))
            if abs(c1_new - c1) < 1e-10 and abs(c2_new - c2) < 1e-10:
                c1, c2 = c1_new, c2_new
                break
            c1, c2 = c1_new, c2_new
        if c1 > c2:
            c1, c2 = c2, c1
        return float(0.5 * (c1 + c2))

    def _plot_corr_A_regions(self, ax, A_true, A_hat, y_label="A_hat"):
        x = np.asarray(A_true, dtype=np.float64).ravel()
        y = np.asarray(A_hat, dtype=np.float64).ravel()
        assert x.shape == y.shape, f"A_true/A_hat shape mismatch after ravel: {x.shape} vs {y.shape}"
        ax.scatter(x, y, s=6, alpha=0.35, color="green", edgecolors="none")
        self._add_x45_line(ax)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.7, alpha=0.55)
        ax.axvline(0.0, color="k", linestyle="--", linewidth=0.7, alpha=0.55)
        ax.set(title="A fit", xlabel="A_true", ylabel=y_label)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.35)

        for mask, tag, tx, ty in [
            (x < 0.0, "L", 0.08, 0.10),
            (x == 0.0, "M", 0.43, 0.42),
            (x > 0.0, "R", 0.67, 0.66),
        ]:
            r = self._corrcoef_safe(x[mask], y[mask])
            n = int(mask.sum())
            ax.text(tx, ty, f"r{tag}={r:.3f}\nn{tag}={n}", transform=ax.transAxes, fontsize=9)

    def _scatter_true_vs_fit(self, ax, x_true, y_fit, title, color, split_by_true_sign=False):
        x = np.asarray(x_true, dtype=np.float64).ravel()
        y = np.asarray(y_fit, dtype=np.float64).ravel()
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]
        ax.scatter(x, y, alpha=0.7, color=color, marker=".", s=9)
        if n > 0:
            lo = float(min(np.min(x), np.min(y)))
            hi = float(max(np.max(x), np.max(y)))
            pad = 0.05 * max(1e-9, hi - lo)
            lo -= pad
            hi += pad
            ax.plot([lo, hi], [lo, hi], color="k", linestyle="--", linewidth=1.0, alpha=0.6)
            ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            if split_by_true_sign:
                for mask in (x < 0.0, x > 0.0):
                    if np.any(mask):
                        ax.plot(
                            [float(np.mean(x[mask]))], [float(np.mean(y[mask]))],
                            marker="+", markersize=14, markeredgewidth=2.5, color="k",
                        )
            else:
                ax.plot(
                    [float(np.mean(x))], [float(np.mean(y))],
                    marker="+", markersize=14, markeredgewidth=2.5, color="k",
                )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel("true weight")
        ax.set_ylabel("fitted")

    def _plot_corr_B_divisor(self, ax, B_true, B_hat, state_idx, y_label="B_hat"):
        x = np.asarray(B_true, dtype=np.float64).ravel()
        y = np.asarray(B_hat, dtype=np.float64).ravel()
        assert x.shape == y.shape, f"B_true/B_hat state {state_idx} shape mismatch: {x.shape} vs {y.shape}"
        divider = self._find_bimodal_divider(x)
        ax.scatter(x, y, s=8, alpha=0.45, color="tab:blue", edgecolors="none")
        self._add_x45_line(ax)
        ax.axvline(divider, color="red", linestyle="--", linewidth=1.0)
        ax.set(title=f"B fit, state {state_idx}", xlabel="B_true", ylabel=y_label)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.35)

        left = x < divider
        right = ~left
        r_left = self._corrcoef_safe(x[left], y[left])
        r_right = self._corrcoef_safe(x[right], y[right])
        ax.text(0.05, 0.45, f"rL={r_left:.3f}\nnL={int(left.sum())}", transform=ax.transAxes, va="top", fontsize=9)
        ax.text(0.64, 0.58, f"rR={r_right:.3f}\nnR={int(right.sum())}", transform=ax.transAxes, va="top", fontsize=9)

    def _edge_recovery_stats_ax(self, ax, fitD, md):
        A_true = np.asarray(md["A_true"])
        if A_true.ndim == 3:
            A_true = A_true[0]
        A_hat = np.asarray(fitD["A_hat"])
        assert A_hat.shape == A_true.shape, f"A_hat shape {A_hat.shape} != A_true {A_true.shape}"

        E_true = np.asarray(md["E_true"]) != 0
        if E_true.ndim == 3:
            E_true = E_true[0] != 0
        assert E_true.shape == A_true.shape, f"E_true shape {E_true.shape} != A_true {A_true.shape}"

        n_neur = A_true.shape[0]
        off_diag = ~np.eye(n_neur, dtype=bool)
        E_t = E_true & off_diag
        E_hat = (A_hat != 0) & off_diag
        tp = int(np.sum(E_t & E_hat))
        fp = int(np.sum((~E_t) & E_hat))
        fn = int(np.sum(E_t & (~E_hat)))
        tn = int(np.sum((~E_t) & (~E_hat)))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        acc = (tp + tn) / max(1, tp + tn + fp + fn)

        vals = [tp, fp, fn]
        ax.bar(["TP", "FP", "FN"], vals, color=["green", "red", "magenta"])
        ax.set(title="edge stats", ylabel="count")
        ax.grid(axis="y", alpha=0.4)
        y_pos = 0.25 * max(1, max(vals))
        for name, val in zip(["TP", "FP", "FN"], vals):
            ax.text(name, y_pos, str(val), ha="center", va="center", fontsize=10)
        txt = f"precision={precision:.3f}\nrecall={recall:.3f}\nf1={f1:.3f}\nacc={acc:.3f}"
        ax.text(0.50, 0.62, txt, transform=ax.transAxes, fontsize=9)

    def edge_stats_and_ABcorr(self, fitD, md, figId="n"):
        """Edge-recovery stats plus A/B truth-vs-fit correlations for simulations."""
        A_true = np.asarray(md["A_true"])
        if A_true.ndim == 3:
            A_true = A_true[0]
        A_hat = np.asarray(fitD["A_hat"])
        B_true = np.asarray(md["B_true"])
        B_hat = np.asarray(fitD["B_hat"])
        A_label = self._display_A_label(md)
        B_label = self._display_B_label(md)
        if B_true.ndim == 1:
            B_true = B_true[None, :]
        if B_hat.ndim == 1:
            B_hat = B_hat[None, :]
        n_state = min(B_true.shape[0], B_hat.shape[0])
        assert n_state >= 1, "Need at least one B state for correlation plot"

        ncol = 2 + n_state
        fig_w = max(13.0, 3.1 * ncol)
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(fig_w, 3.9))
        gs = fig.add_gridspec(1, ncol, left=0.045, right=0.99, bottom=0.18, top=0.80, wspace=0.38)

        ax = fig.add_subplot(gs[0, 0])
        self._edge_recovery_stats_ax(ax, fitD, md)

        ax = fig.add_subplot(gs[0, 1])
        self._plot_corr_A_regions(ax, A_true, A_hat, y_label=A_label)

        for m in range(n_state):
            ax = fig.add_subplot(gs[0, 2 + m])
            self._plot_corr_B_divisor(ax, B_true[m], B_hat[m], m, y_label=B_label)

        title_prefix = self.canvas_title_prefix(md)
        fig.suptitle(f"{title_prefix}, edge recovery and A/B correlations", fontsize=12)

    def matrix_init(self, fitD, md, figId="o", est_key="A_init", est_label="A_init"):
        """Compare a fitted A-like matrix against stored simulation A_true."""
        import matplotlib.colors as colors

        A_true = np.asarray(md["A_true"], dtype=np.float64)
        if A_true.ndim == 3:
            A_true = A_true[0]
        A_cmp = np.asarray(fitD[est_key], dtype=np.float64)
        assert A_cmp.shape == A_true.shape, f"{est_key} shape {A_cmp.shape} != A_true {A_true.shape}"
        n_neur = A_true.shape[0]
        num_exc = md["dale_conf"]["num_excite"]
        off_mask = ~np.eye(n_neur, dtype=bool)
        neg_mask = off_mask & (A_true < 0.0)
        pos_mask = off_mask & (A_true > 0.0)
        diag_mask = np.eye(n_neur, dtype=bool)
        n_diag = int(np.sum(diag_mask))
        n_off = int(np.count_nonzero(A_true[off_mask]))

        vmin = float(np.min(A_true))
        vmax = float(np.max(A_true))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-9
        shared_norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(14, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.45)

        ax = fig.add_subplot(gs[0, 0])
        self._draw_A_matrix(
            ax, A_true, f"True Dale, N{n_neur}, nEdges={n_diag}+{n_off}",
            num_exc=num_exc, norm_map=shared_norm,
        )

        ax = fig.add_subplot(gs[0, 1])
        _, cmp_diag, cmp_off = self._count_A_edges(A_cmp)
        self._draw_A_matrix(
            ax, A_cmp, f"{est_label}, nEdges={cmp_diag}+{cmp_off}",
            num_exc=num_exc, norm_map=shared_norm,
        )

        ax = fig.add_subplot(gs[0, 2])
        self._draw_A_offdiag_hist(ax, A_cmp, f"{est_label} off-diagonal")
        if est_key == "A_init":
            ax.text(
                0.05, 0.70, self._A_init_diagnostics_txt(md), transform=ax.transAxes,
                va="top", ha="left", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                family="monospace",
            )

        ax = fig.add_subplot(gs[1, 0])
        self._scatter_true_vs_fit(
            ax, A_true[neg_mask], A_cmp[neg_mask],
            f"{est_label}: neg TP", color="tab:green",
        )

        ax = fig.add_subplot(gs[1, 1])
        self._scatter_true_vs_fit(
            ax, A_true[pos_mask], A_cmp[pos_mask],
            f"{est_label}: pos TP", color="tab:green",
        )

        ax = fig.add_subplot(gs[1, 2])
        self._scatter_true_vs_fit(
            ax, A_true[diag_mask], A_cmp[diag_mask],
            f"{est_label}: diag", color="salmon", split_by_true_sign=True,
        )

        title_prefix = self.canvas_title_prefix(md)
        fig.suptitle(f"{title_prefix}, A_true vs {est_label} comparison", fontsize=12)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

    def _overlay_single_rates(self, ax, single_rates, x_src):
        rate = np.asarray(single_rates, dtype=np.float64).ravel()
        ax2 = ax.twinx()
        ax2.plot(x_src, rate, color="tab:orange", linewidth=0.9, alpha=0.75)
        ax2.set_ylabel("frequency (Hz)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        return ax2

    def _draw_A_offdiag_hist(
        self, ax, A, title, diagnostics_txt=None, weight_lines=None,
        neuron_type=None,
    ):
        A = np.asarray(A)
        n_neuron = A.shape[0]
        off_mask = ~np.eye(n_neuron, dtype=bool)
        A_off = A[off_mask]
        A_off_nz = A_off[np.abs(A_off) > 1e-12]
        if neuron_type is None:
            ax.hist(A_off_nz, bins=120, color="saddlebrown", alpha=0.85)
        else:
            neuron_type = np.asarray(neuron_type, dtype=np.int8)
            assert neuron_type.shape == (n_neuron,), (
                "neuron_type length must match A columns"
            )
            bin_edges = np.histogram_bin_edges(A_off_nz, bins=120)
            for cls, color, label in [
                (1, "magenta", f"exc={int(np.sum(neuron_type > 0))}"),
                (-1, "forestgreen", f"inh={int(np.sum(neuron_type < 0))}"),
                (0, "salmon", f"und={int(np.sum(neuron_type == 0))}"),
            ]:
                class_mask = off_mask & (neuron_type == cls)[None, :]
                values = A[class_mask]
                values = values[np.abs(values) > 1e-12]
                ax.hist(
                    values,
                    bins=bin_edges,
                    color=color,
                    alpha=0.20,
                    label=label,
                )
            ax.legend(loc="best", fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel("edge value")
        ax.set_ylabel("edges")
        if weight_lines is not None:
            for y_pos, w in zip([0.88, 0.78], weight_lines):
                if np.isfinite(w):
                    ax.axvline(w, color="tab:blue", ls="--", lw=0.8, alpha=0.9)
                    ax.text(
                        w, y_pos, f"{w:.3f}", transform=ax.get_xaxis_transform(),
                        va="top", ha="center", fontsize=8, color="tab:blue",
                        rotation=90,
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1),
                    )
        if diagnostics_txt is not None:
            ax.text(
                0.05, 0.70, diagnostics_txt, transform=ax.transAxes,
                va="top", ha="left", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                family="monospace",
            )

    def _draw_A_matrix_summary_3cols(self, fig, gs, row, A, label, single_rates, num_exc=None, weight_lines=None):
        A = np.asarray(A, dtype=np.float64)
        n_neuron = A.shape[0]
        x_src = np.arange(n_neuron, dtype=np.int64)
        single_rates = np.asarray(single_rates, dtype=np.float64).ravel()
        assert single_rates.shape[0] == n_neuron, (
            f"single_rates length {single_rates.shape[0]} != N={n_neuron}"
        )

        _, n_diag, n_off = self._count_A_edges(A)
        cnt_src, _ = self._source_nz_edge_stats(A)
        sum_nedge = int(np.sum(cnt_src))

        ax = fig.add_subplot(gs[row, 0])
        self._draw_A_offdiag_hist(ax, A, f"{label} off-diagonal", weight_lines=weight_lines)
        ax.text(
            0.98, 0.97, f"sum Nedge={sum_nedge:d}", transform=ax.transAxes,
            va="top", ha="right", fontsize=9, color="k",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
        )

        ax = fig.add_subplot(gs[row, 1])
        self._draw_A_matrix(ax, A, f"{label}, nEdges={n_diag}+{n_off}", num_exc=num_exc)

        ax = fig.add_subplot(gs[row, 2])
        ax.plot(x_src, cnt_src, color="tab:blue", linewidth=1.0)
        med_cnt = int(np.median(cnt_src))
        ax.axhline(med_cnt, color="green", ls="--", lw=1.0, alpha=0.9)
        ax.text(
            0.02, 0.95, f"median={med_cnt:d}", transform=ax.transAxes,
            va="top", ha="left", fontsize=9, color="green",
        )
        ax.set_title(f"{label}: # non-zero edges")
        ax.set_xlabel("source neuron index (column)")
        ax.set_ylabel("# outgoing non-zero edges")
        ax.grid(True, alpha=0.35)
        self._overlay_single_rates(ax, single_rates, x_src)

    def node_outgoing_edge_stats(
        self, fitD, md, single_rates, neuron_type,
        figId="e", est_key="A_hat", est_label="A_hat",
    ):
        A_est = np.asarray(fitD[est_key], dtype=np.float64)
        A_prune = self._A_prune_from_fitD(fitD)
        prune_label = self._display_A_label(md, prune=True)
        assert A_est.ndim == 2 and A_est.shape[0] == A_est.shape[1], "A_est must be square"
        assert A_prune.shape == A_est.shape, "A_prune shape must match A_hat"
        n_neuron = A_est.shape[0]

        nedge, sedge = self._node_outgoing_edge_stats(A_est)
        stage_c = md.get("fit_type") == "prismEM_deBias_stageC" or "deBias_stageC" in md
        if "neuron_Nedge" in fitD and not stage_c:
            stored_nedge = np.asarray(fitD["neuron_Nedge"], dtype=np.int64)
            assert np.array_equal(stored_nedge, nedge), (
                "stored neuron_Nedge does not match nonzero A_hat edges"
            )
        nedge_in, _ = self._node_incoming_edge_stats(A_est)
        n_bins = max(10, min(40, int(np.sqrt(n_neuron)) * 2))
        stage_b = md["bagsFDR_stageB"]
        min_posW = float(stage_b["min_posW"])
        max_negW = float(stage_b["max_negW"])
        nedge_scaled_rule = "Nedge" in stage_b.get("source_type_rule", "")
        neuron_type = np.asarray(neuron_type, dtype=np.int8)
        assert neuron_type.shape[0] == n_neuron, "neuron_type length must match A_hat columns"
        n_exc = int(np.sum(neuron_type > 0))
        n_inh = int(np.sum(neuron_type < 0))
        n_und = int(np.sum(neuron_type == 0))

        def add_hist_percentile_marker(ax, vals, fmt=".3f"):
            vals = np.asarray(vals, dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            p16, p50, p84 = np.percentile(vals, [16, 50, 84])
            ylo, yhi = ax.get_ylim()
            y_mark = 0.5 * (ylo + yhi)
            ax.errorbar(
                p50, y_mark,
                xerr=[[p50 - p16], [p84 - p50]],
                fmt="o", color="k", markersize=8,
                capsize=4, capthick=1.2, elinewidth=1.2, zorder=5,
            )
            txt = f"p16={format(p16, fmt)}\nmed={format(p50, fmt)}\np84={format(p84, fmt)}"
            ax.text(
                0.98, 0.97, txt, transform=ax.transAxes,
                va="top", ha="right", fontsize=9, color="k",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
            )

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(18, 8))
        gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15], hspace=0.45, wspace=0.35)

        cnt_src_prune, _ = self._source_nz_edge_stats(A_prune)
        sum_nedge_prune = int(np.sum(cnt_src_prune))
        x_src = np.arange(n_neuron, dtype=np.int64)

        ax = fig.add_subplot(gs[0, 0])
        for cls, color, label in [
            (1, "magenta", f"exc={n_exc}"),
            (-1, "forestgreen", f"inh={n_inh}"),
            (0, "salmon", f"und={n_und}"),
        ]:
            mask = neuron_type == cls
            ax.scatter(
                nedge[mask], sedge[mask],
                s=neuron_type_marker_size(cls, 18),
                marker=neuron_type_marker(cls), alpha=0.80, color=color,
                edgecolors="k" if cls == 0 else "none",
                linewidths=0.35 if cls == 0 else 0.0, label=label,
            )
        if nedge_scaled_rule:
            boundary_nedge = np.asarray([0, max(1, int(np.max(nedge)))])
            ax.plot(
                boundary_nedge, min_posW * boundary_nedge,
                color="lightskyblue", ls="--", lw=1.2, alpha=0.95,
            )
            ax.plot(
                boundary_nedge, max_negW * boundary_nedge,
                color="lightskyblue", ls="--", lw=1.2, alpha=0.95,
            )
        else:
            ax.axhline(min_posW, color="tab:blue", ls="--", lw=0.8, alpha=0.9)
            ax.axhline(max_negW, color="tab:blue", ls="--", lw=0.8, alpha=0.9)
        ax.set(
            title="Reco Nedge vs Sedge",
            xlabel="Nedge (# outgoing edges per source column)",
            ylabel="Sedge",
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.35)

        ax = fig.add_subplot(gs[0, 1])
        self._draw_A_offdiag_hist(
            ax,
            A_prune,
            f"Reco {prune_label} off-diagonal",
            weight_lines=(max_negW, min_posW),
            neuron_type=neuron_type,
        )
        ax.text(
            0.98, 0.97, f"sum Nedge={sum_nedge_prune:d}", transform=ax.transAxes,
            va="top", ha="right", fontsize=9, color="k",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
        )

        ax = fig.add_subplot(gs[0, 2])
        blk = 10
        n_blk = n_neuron // blk
        blk_edges = np.arange(n_blk + 1) * blk
        blk_cnt = cnt_src_prune[:n_blk * blk].reshape(n_blk, blk).mean(axis=1)
        ax.stairs(blk_cnt, blk_edges, color="tab:blue", linewidth=1.2)
        med_cnt = float(np.median(blk_cnt))
        ax.axhline(med_cnt, color="green", ls="--", lw=1.0, alpha=0.9)
        ax.text(
            0.02, 0.95, f"median={med_cnt:.1f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=9, color="green",
        )
        ax.set_title(f"Reco {prune_label}: median non-zero edges={med_cnt:.1f}")
        ax.set_xlabel(f"source neuron index (column), step={blk}")
        ax.set_ylabel("# outgoing non-zero edges")
        ax.grid(True, alpha=0.35)
        rate = np.asarray(single_rates, dtype=np.float64).ravel()
        blk_rate = rate[:n_blk * blk].reshape(n_blk, blk).mean(axis=1)
        ax2 = ax.twinx()
        ax2.stairs(blk_rate, blk_edges, color="tab:orange", linewidth=1.2, alpha=0.85)
        ax2.set_ylabel("frequency (Hz)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        # Keep the top-right slot intentionally empty.
        fig.add_subplot(gs[0, 3]).axis("off")

        single_rates = np.asarray(single_rates, dtype=np.float64).ravel()
        off_mask = ~np.eye(n_neuron, dtype=bool)
        med_weight_out = np.zeros((n_neuron,), dtype=np.float64)
        med_weight_in = np.zeros((n_neuron,), dtype=np.float64)
        for j in range(n_neuron):
            vals = A_est[off_mask[:, j], j]
            vals = vals[vals != 0.0]
            if vals.size:
                med_weight_out[j] = float(np.median(vals))
            vals = A_est[j, off_mask[j, :]]
            vals = vals[vals != 0.0]
            if vals.size:
                med_weight_in[j] = float(np.median(vals))

        def draw_edge_metric_vs_frequency(
            ax, x_values, title, xlabel, mark_zero=False, legend_title=None,
        ):
            for cls, color, label in [
                (1, "magenta", f"exc={n_exc}"),
                (-1, "forestgreen", f"inh={n_inh}"),
                (0, "salmon", f"und={n_und}"),
            ]:
                mask = neuron_type == cls
                ax.scatter(
                    x_values[mask], single_rates[mask],
                    s=neuron_type_marker_size(cls, 18),
                    marker=neuron_type_marker(cls), alpha=0.80, color=color,
                    edgecolors="k" if cls == 0 else "none",
                    linewidths=0.35 if cls == 0 else 0.0, label=label,
                )
            if mark_zero:
                ax.axvline(0.0, color="tab:blue", ls="--", lw=0.8, alpha=0.9)
            ax.set(title=title, xlabel=xlabel, ylabel="frequency (Hz)")
            ax.legend(loc="best", fontsize=9, title=legend_title)
            ax.grid(True, alpha=0.35)

        draw_edge_metric_vs_frequency(
            fig.add_subplot(gs[1, 0]), nedge,
            "Reco outgoing num edges",
            "Nedge (# outgoing edges per source column)",
        )
        draw_edge_metric_vs_frequency(
            fig.add_subplot(gs[1, 1]), nedge_in,
            "Input to neur.: num edges",
            "Nedge (# incoming edges per target row)",
            legend_title="target type",
        )
        draw_edge_metric_vs_frequency(
            fig.add_subplot(gs[1, 2]), med_weight_out,
            "Reco median outgoing weight",
            "median outgoing edge weight", mark_zero=True,
        )
        draw_edge_metric_vs_frequency(
            fig.add_subplot(gs[1, 3]), med_weight_in,
            "Input to neur: incoming weight",
            "median incoming edge weight", mark_zero=True,
            legend_title="target type",
        )

        title_prefix = self.canvas_title_prefix(md)
        fig.suptitle(
            f"{title_prefix}, {est_label} outgoing (column=source) and incoming (row=target) edges",
            fontsize=12,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

    def final_weight_distributions(self, fitD, md, figId="h"):
        """Final non-truth weight summaries for FDR aggregate outputs."""
        required = ("A_prune", "B_hat", "neuron_type", "A_diag_mean", "A_diag_stderr")
        missing = [key for key in required if key not in fitD]
        assert not missing, f"Plot h requires regenerated aggregate arrays; missing {missing}"

        A_prune = np.asarray(fitD["A_prune"], dtype=np.float64)
        B_hat = np.asarray(fitD["B_hat"], dtype=np.float64)
        neuron_type = np.asarray(fitD["neuron_type"], dtype=np.int8)
        diag_mean = np.asarray(fitD["A_diag_mean"], dtype=np.float64).ravel()
        diag_stderr = np.asarray(fitD["A_diag_stderr"], dtype=np.float64).ravel()
        A_label = self._display_A_label(md, prune=True)
        B_label = self._display_B_label(md)
        assert A_prune.ndim == 2 and A_prune.shape[0] == A_prune.shape[1], "A_prune must be square"
        n_neuron = A_prune.shape[0]
        assert neuron_type.shape[0] == n_neuron, "neuron_type length must match A_prune"
        assert diag_mean.shape[0] == n_neuron and diag_stderr.shape[0] == n_neuron, (
            "A_diag_mean/A_diag_stderr length must match A_prune"
        )
        if B_hat.ndim == 1:
            B_hat = B_hat[None, :]
        assert B_hat.ndim == 2 and B_hat.shape[1] == n_neuron, "B_hat must have shape (M,N)"

        off_mask = ~np.eye(n_neuron, dtype=bool)

        def outgoing_values_for_type(type_value):
            src_mask = neuron_type == int(type_value)
            vals = A_prune[:, src_mask][off_mask[:, src_mask]]
            return vals[np.abs(vals) > 1e-12]

        exc_w = outgoing_values_for_type(1)
        inh_w = outgoing_values_for_type(-1)
        diag_vals = np.diag(A_prune)
        n_exc = int(np.sum(neuron_type > 0))
        n_inh = int(np.sum(neuron_type < 0))
        n_und = int(np.sum(neuron_type == 0))

        def mark_edge(ax, val):
            if np.isfinite(val):
                ax.axvline(val, color="tab:blue", ls="--", lw=0.8, alpha=0.9)
                ax.text(
                    val, 0.92, f"{val:.3f}", transform=ax.get_xaxis_transform(),
                    va="top", ha="center", fontsize=8, color="tab:blue",
                    rotation=90,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1),
                )

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(14, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

        ax = fig.add_subplot(gs[0, 0])
        ax.hist(exc_w, bins=80, color="magenta", alpha=0.75)
        ax.axvline(0.0, color="k", ls="--", lw=0.8)
        mark_edge(ax, float(np.min(exc_w[exc_w > 0.0])) if np.any(exc_w > 0.0) else float("nan"))
        ax.set(title=f"Reco exc outgoing weights, neurons={n_exc}", xlabel=f"{A_label} weight", ylabel="edges")
        ax.grid(True, alpha=0.35)

        ax = fig.add_subplot(gs[0, 1])
        ax.hist(inh_w, bins=80, color="forestgreen", alpha=0.75)
        ax.axvline(0.0, color="k", ls="--", lw=0.8)
        mark_edge(ax, float(np.max(inh_w[inh_w < 0.0])) if np.any(inh_w < 0.0) else float("nan"))
        ax.set(title=f"Reco inh outgoing weights, neurons={n_inh}", xlabel=f"{A_label} weight", ylabel="edges")
        ax.grid(True, alpha=0.35)

        ax = fig.add_subplot(gs[0, 2])
        bins = np.linspace(diag_vals.min(), diag_vals.max(), 81)
        nonneg_counts = []
        for cls, color, label in [
            (1,  "magenta", f"exc={n_exc}"),
            (-1, "forestgreen", f"inh={n_inh}"),
            (0,  "salmon",  f"und={n_und}"),
        ]:
            mask = neuron_type == cls
            name = label.split("=")[0]
            nonneg_counts.append((name, int(np.sum(diag_vals[mask] >= 0.0))))
            if np.any(mask):
                ec = "k" if cls == 0 else "none"
                ax.hist(diag_vals[mask], bins=bins, color=color, alpha=0.75, label=label, edgecolor=ec, linewidth=0.5)
        ax.axvline(0.0, color="k", ls="--", lw=0.8)
        ax.legend(loc="best", fontsize=9)
        tbl = ax.table(
            cellText=[[name, f"{cnt:d}"] for name, cnt in nonneg_counts],
            colLabels=["type", ">=0"],
            cellLoc="center",
            colLoc="center",
            bbox=[0.70, 0.46, 0.27, 0.28],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for cell in tbl.get_celld().values():
            cell.set_edgecolor("0.65")
            cell.set_linewidth(0.5)
            cell.set_facecolor((1.0, 1.0, 1.0, 0.78))
        ax.set(title="Reco A diagonal", xlabel=f"{A_label} diagonal value", ylabel="neurons")
        ax.grid(True, alpha=0.35)

        ax = fig.add_subplot(gs[1, 0])
        ax.hist(B_hat[0], bins=80, color="tab:orange", alpha=0.8)
        ax.axvline(0.0, color="k", ls="--", lw=0.8)
        ax.set(title=f"{B_label} mode 0", xlabel="B value", ylabel="neurons")
        ax.grid(True, alpha=0.35)

        ax = fig.add_subplot(gs[1, 1])
        if B_hat.shape[0] >= 2:
            ax.hist(B_hat[1], bins=80, color="tab:purple", alpha=0.8)
            ax.axvline(0.0, color="k", ls="--", lw=0.8)
            ax.set(title=f"{B_label} mode 1", xlabel="B value", ylabel="neurons")
        else:
            ax.set_axis_off()
        ax.grid(True, alpha=0.35)

        ax = fig.add_subplot(gs[1, 2])
        for cls, color, label in [
            (1,  "magenta", f"exc={n_exc}"),
            (-1, "forestgreen", f"inh={n_inh}"),
            (0,  "salmon",  f"und={n_und}"),
        ]:
            mask = neuron_type == cls
            if not np.any(mask):
                continue
            ax.scatter(
                diag_mean[mask], diag_stderr[mask],
                s=neuron_type_marker_size(cls, 18),
                marker=neuron_type_marker(cls), alpha=0.80, color=color,
                edgecolors="k" if cls == 0 else "none",
                linewidths=0.5 if cls == 0 else 0.0, label=label,
            )
        ax.axvline(0.0, color="k", ls="--", lw=0.8)
        ax.legend(loc="best", fontsize=9)
        ax.set(title="A-diag summary", xlabel="mean diagonal A", ylabel="stderr diagonal A")
        ax.grid(True, alpha=0.35)

        title_prefix = self.canvas_title_prefix(md)
        fig.suptitle(
            f"{title_prefix}, final weight distributions ({A_label}, {B_label}): "
            f"exc={n_exc} inh={n_inh} und={n_und}",
            fontsize=12,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

    def offdiag_weight_investigation(self, fitD, md, figId="i"):
        """Off-diagonal A_prune weight investigation: histogram and weight-vs-frequency heatmap."""
        A_prune = np.asarray(fitD["A_prune"], dtype=np.float64)
        neuron_type = np.asarray(fitD["neuron_type"], dtype=np.int8)
        single_rates = np.asarray(fitD["single_rates"], dtype=np.float64).ravel()
        A_label = self._display_A_label(md, prune=True)
        n_neuron = A_prune.shape[0]
        off_mask = ~np.eye(n_neuron, dtype=bool)
        n_exc = int(np.sum(neuron_type > 0))
        n_inh = int(np.sum(neuron_type < 0))
        n_und = int(np.sum(neuron_type == 0))

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(18, 4))
        gs = fig.add_gridspec(1, 4, wspace=0.38, left=0.05, right=0.97)

        # --- plot 1: off-diagonal weight histogram by type ---
        ax = fig.add_subplot(gs[0, 0])
        all_off = A_prune[off_mask]
        bins = np.linspace(all_off.min(), all_off.max(), 81)
        edge_counts = {}
        for cls, color, label in [
            (1,  "magenta", f"exc={n_exc}"),
            (-1, "forestgreen", f"inh={n_inh}"),
            (0,  "salmon",  f"und={n_und}"),
        ]:
            src_mask = neuron_type == cls
            if not np.any(src_mask):
                continue
            vals = A_prune[:, src_mask][off_mask[:, src_mask]]
            vals = vals[np.abs(vals) > 1e-12]
            edge_counts[label.split("=")[0]] = len(vals)
            ec = "k" if cls == 0 else "none"
            ax.hist(vals, bins=bins, color=color, alpha=0.75, label=label, edgecolor=ec, linewidth=0.5)
        ax.axvline(0.0, color="k", ls="--", lw=0.8)
        leg = ax.legend(loc="upper left", fontsize=9, title="neurons")
        leg.get_title().set_fontsize(9)
        edge_txt = "\n".join(f"{k}: {v}" for k, v in edge_counts.items())
        ax.text(0.98, 0.97, f"edges:\n{edge_txt}", transform=ax.transAxes,
                va="top", ha="right", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
        ax.set(title="Off-diag weights by type", xlabel=f"{A_label} weight", ylabel="edges")
        ax.grid(True, alpha=0.35)

        # --- plot 2: weight vs spike frequency heatmap (column = source neuron) ---
        import matplotlib.colors as mcolors
        ax = fig.add_subplot(gs[0, 1])
        col_idx = np.where(off_mask)[1]   # source column for each off-diag entry
        w_vals  = A_prune[off_mask]
        nz = np.abs(w_vals) > 1e-12
        freq_rep = single_rates[col_idx[nz]]
        w_nz     = w_vals[nz]
        h, xedges, yedges = np.histogram2d(w_nz, freq_rep, bins=[80, 40])
        pcm = ax.pcolormesh(xedges, yedges, h.T, cmap="Blues", shading="flat",
                            norm=mcolors.LogNorm(vmin=1, vmax=h.max()))
        fig.colorbar(pcm, ax=ax, label="edges (log)")
        ax.axvline(0.0, color="k", ls="--", lw=0.8)
        ax.set(title="Weight vs source frequency", xlabel=f"{A_label} weight", ylabel="spike frequency (Hz)")
        ax.grid(True, alpha=0.2)

        # --- plot 3: diagonal weight histogram by type ---
        diag_vals = np.diag(A_prune)
        ax = fig.add_subplot(gs[0, 2])
        bins_diag = np.linspace(diag_vals.min(), diag_vals.max(), 81)
        for cls, color, label in [
            (1,  "magenta", f"exc={n_exc}"),
            (-1, "forestgreen", f"inh={n_inh}"),
            (0,  "salmon",  f"und={n_und}"),
        ]:
            mask = neuron_type == cls
            if not np.any(mask):
                continue
            vals = diag_vals[mask]
            vals = vals[np.abs(vals) > 1e-12]
            ec = "k" if cls == 0 else "none"
            ax.hist(vals, bins=bins_diag, color=color, alpha=0.75, label=label, edgecolor=ec, linewidth=0.5)
        ax.axvline(0.0, color="k", ls="--", lw=0.8)
        leg = ax.legend(loc="upper center", fontsize=9, title="neurons")
        leg.get_title().set_fontsize(9)
        ax.set(title="Diagonal weights by type", xlabel=f"{A_label} diagonal", ylabel="neurons")
        ax.grid(True, alpha=0.35)

        # --- plot 4: diagonal weight vs spike frequency heatmap ---
        ax = fig.add_subplot(gs[0, 3])
        diag_nz_mask = np.abs(diag_vals) > 1e-12
        h2, xedges2, yedges2 = np.histogram2d(
            diag_vals[diag_nz_mask], single_rates[diag_nz_mask], bins=[80, 40])
        pcm2 = ax.pcolormesh(xedges2, yedges2, h2.T, cmap="Blues", shading="flat",
                             norm=mcolors.LogNorm(vmin=1, vmax=h2.max()))
        fig.colorbar(pcm2, ax=ax, label="neurons (log)")
        ax.axvline(0.0, color="k", ls="--", lw=0.8)
        ax.set(title="Diagonal vs source frequency", xlabel=f"{A_label} diagonal", ylabel="spike frequency (Hz)")
        ax.grid(True, alpha=0.2)

        title_prefix = self.canvas_title_prefix(md)
        fig.suptitle(f"{title_prefix}, {A_label} off-diagonal weight investigation", fontsize=12)
        fig.subplots_adjust(top=0.88)

    def fdr_selection_summary(self, fitD, md, figId="g"):
        """Summarize Stage (b) FDR bag filtering and cross-bag stability."""
        import matplotlib.colors as colors

        stage_b = md.get("bagsFDR_stageB")
        assert stage_b is not None, "Plot g requires a Stage (b) aggregate with bagsFDR_stageB metadata"
        required = (
            "A_bag", "selected_in_bag", "selection_frequency", "selected_mask",
            "src_null_tau_bag", "selected_edges_per_bag", "selected_edges_per_bag_src",
            "A_hat", "A_mean_selected", "single_rates",
        )
        missing = [key for key in required if key not in fitD]
        assert not missing, f"Plot g requires Stage (b) diagnostic arrays; missing {missing}"

        A_bag = np.asarray(fitD["A_bag"], dtype=np.float64)
        selected_in_bag = np.asarray(fitD["selected_in_bag"], dtype=bool)
        sel_freq = np.asarray(fitD["selection_frequency"], dtype=np.float64)
        final_sel = np.asarray(fitD["selected_mask"], dtype=bool)
        tau = np.asarray(fitD["src_null_tau_bag"], dtype=np.float64)
        selected_edges_per_bag = np.asarray(fitD["selected_edges_per_bag"], dtype=np.int64)
        A_hat = np.asarray(fitD["A_hat"], dtype=np.float64)

        assert A_bag.ndim == 3 and A_bag.shape[1] == A_bag.shape[2], "A_bag must have shape (bag,N,N)"
        assert selected_in_bag.shape == A_bag.shape, "selected_in_bag shape must match A_bag"
        n_bag, n_neur, _ = A_bag.shape
        assert tau.shape == (n_bag, n_neur), f"src_null_tau_bag shape {tau.shape} != {(n_bag, n_neur)}"

        off_mask = ~np.eye(n_neur, dtype=bool)
        off3 = np.broadcast_to(off_mask, A_bag.shape)
        p_cand = int(np.sum(off_mask))
        per_bag_quantile = float(stage_b["per_bag_quantile"])
        stab_sel_thresh = float(stage_b["stab_sel_thresh"])
        final_count = int(np.sum(final_sel & off_mask))
        ever_count = int(np.sum((sel_freq > 0.0) & off_mask))
        mean_per_bag = float(np.mean(selected_edges_per_bag))
        median_per_bag = float(np.median(selected_edges_per_bag))

        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.abs(A_bag) / tau[:, None, :]
        ratio_all = ratio[off3]
        ratio_sel = ratio[selected_in_bag & off3]
        ratio_all = ratio_all[np.isfinite(ratio_all)]
        ratio_sel = ratio_sel[np.isfinite(ratio_sel)]
        log_ratio_all = np.log10(np.clip(ratio_all, 1e-8, 1e8))
        log_ratio_sel = np.log10(np.clip(ratio_sel, 1e-8, 1e8))

        freq_off = sel_freq[off_mask]
        freq_nonzero = freq_off[freq_off > 0.0]
        thresholds = np.arange(1, n_bag + 1, dtype=np.float64) / float(n_bag)
        survivor_counts = np.array([np.sum(freq_off >= th) for th in thresholds], dtype=np.int64)

        A_mean_selected = np.asarray(fitD["A_mean_selected"], dtype=np.float64)
        amp = np.abs(A_mean_selected[off_mask])
        final_for_scatter = (final_sel & off_mask)[off_mask]
        finite_amp = np.isfinite(amp)

        src_tau_mean = np.mean(tau, axis=0)
        src_tau_sd = np.std(tau, axis=0, ddof=1) if n_bag > 1 else np.zeros_like(src_tau_mean)
        src_selected_mean = np.mean(np.asarray(fitD["selected_edges_per_bag_src"]), axis=0)
        single_rates = np.asarray(fitD["single_rates"], dtype=np.float64).ravel()
        assert single_rates.shape[0] == n_neur, "single_rates length must match A_bag columns"
        src_order = np.argsort(single_rates)

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(17, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.48, wspace=0.36)

        ax = fig.add_subplot(gs[0, 0])
        funnel_labels = ["all\npairs", "mean\nbag pass", "ever\npass", "stable\nfinal"]
        funnel_counts = [p_cand, mean_per_bag, ever_count, final_count]
        bar_colors = ["0.72", "tab:blue", "tab:orange", "tab:green"]
        ax.bar(np.arange(4), funnel_counts, color=bar_colors, alpha=0.85)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(funnel_labels)
        ax.set_ylabel("off-diagonal edges")
        ax.set_title("FDR reduction funnel")
        ax.grid(True, axis="y", alpha=0.35)
        for i, val in enumerate(funnel_counts):
            ax.text(i, max(val, 1.0), f"{val:.0f}", ha="center", va="bottom", fontsize=9)

        ax = fig.add_subplot(gs[0, 1])
        xbag = np.arange(n_bag)
        ax.bar(xbag, selected_edges_per_bag, color="tab:blue", alpha=0.8)
        ax.axhline(mean_per_bag, color="k", ls="--", lw=1, label=f"mean={mean_per_bag:.1f}")
        ax.axhline(median_per_bag, color="tab:orange", ls=":", lw=1.4, label=f"median={median_per_bag:.1f}")
        ax.set(title="Per-bag edges passing source null threshold", xlabel="bag index", ylabel="edges")
        ax.grid(True, axis="y", alpha=0.35)
        ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[0, 2])
        ax.step(thresholds, survivor_counts, where="post", color="tab:green", linewidth=2)
        ax.axvline(stab_sel_thresh, color="red", ls="--", lw=1.2, label=f"stab_sel_thresh={stab_sel_thresh:g}")
        ax.axhline(final_count, color="k", ls=":", lw=1.0, label=f"final={final_count}")
        ax.set(title="Stable edge count", xlabel="stability sel. thres.", ylabel="stable edges")
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(0, max(1, int(np.max(survivor_counts)) + 1))
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[1, 0])
        bins = np.linspace(0.05, 1.05, 11)
        if freq_nonzero.size:
            ax.hist(freq_nonzero, bins=bins, color="tab:purple", alpha=0.75)
        ax.axvline(stab_sel_thresh, color="red", ls="--", lw=1.2)
        ax.set(title=f"Ever-pass edges, n={freq_nonzero.size}",
               xlabel="stability sel. thres.", ylabel="edges")
        ax.set_xlim(0.05, 1.05)
        ax.set_xticks(np.arange(0.1, 1.01, 0.1))
        ax.grid(True, axis="y", alpha=0.35)

        ax = fig.add_subplot(gs[1, 1])
        im = ax.imshow(sel_freq, origin="lower", aspect="equal", interpolation="nearest",
                       cmap="viridis", vmin=0.0, vmax=1.0)
        if final_count > 0:
            ax.contour(final_sel.astype(float), levels=[0.5], colors="red", linewidths=0.6)
        ax.plot([0, n_neur], [0, n_neur], "--", lw=0.7, color="white", alpha=0.9)
        ax.set(title="Selection frequency matrix", xlabel="source neuron", ylabel="target neuron")
        self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="selected fraction")

        ax = fig.add_subplot(gs[1, 2])
        vmax = float(np.nanmax(np.abs(A_hat))) if np.any(np.isfinite(A_hat)) else 1.0
        if vmax <= 0:
            vmax = 1.0
        norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        self._draw_A_matrix(ax, A_hat, f"Final A_hat, stable offdiag={final_count}", norm_map=norm)

        ax = fig.add_subplot(gs[2, 0])
        bins = np.linspace(-3.0, 3.0, 100)
        if log_ratio_all.size:
            ax.hist(log_ratio_all, bins=bins, color="0.65", alpha=0.75, label="all bag-edge tests")
        if log_ratio_sel.size:
            ax.hist(log_ratio_sel, bins=bins, color="tab:blue", alpha=0.65, label="per-bag pass")
        ax.axvline(0.0, color="red", ls="--", lw=1.2, label="|A| = tau")
        ax.set_yscale("log")
        ax.set(title=f"Per-bag threshold pressure, q={per_bag_quantile:g}",
               xlabel="log10(|A_bag| / source tau)", ylabel="bag-edge tests")
        ax.grid(True, axis="y", alpha=0.35)
        ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[2, 1])
        x = np.arange(n_neur)
        ax.plot(x, src_tau_mean[src_order], color="tab:red", linewidth=1.0, label="mean source tau")
        ax.fill_between(
            x,
            np.maximum(0.0, src_tau_mean[src_order] - src_tau_sd[src_order]),
            src_tau_mean[src_order] + src_tau_sd[src_order],
            color="tab:red", alpha=0.18, linewidth=0,
        )
        ax.set(title="Source null threshold and outgoing pass count",
               xlabel="source neuron sorted by firing rate", ylabel="source tau")
        ax.grid(True, alpha=0.35)
        ax2 = ax.twinx()
        ax2.plot(x, src_selected_mean[src_order], color="tab:blue", linewidth=0.9, alpha=0.9, label="mean outgoing pass count")
        ax2.set_ylabel("mean per-bag outgoing pass count")
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper right")
        rate_min = float(single_rates[src_order[0]])
        rate_max = float(single_rates[src_order[-1]])
        ax.text(
            0.02, 0.96, f"rate: {rate_min:.2g} to {rate_max:.2g} Hz",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
        )

        ax = fig.add_subplot(gs[2, 2])
        positive_amp = finite_amp & (amp > 0.0)
        nonfinal = positive_amp & ~final_for_scatter & (freq_off > 0.0)
        final = positive_amp & final_for_scatter
        ax.scatter(freq_off[nonfinal], amp[nonfinal], s=12, color="0.55", alpha=0.45, label="rejected after bag pass")
        ax.scatter(freq_off[final], amp[final], s=18, color="tab:green", alpha=0.8, label="final stable")
        ax.axvline(stab_sel_thresh, color="red", ls="--", lw=1.2)
        ax.set_yscale("log")
        ax.set(title="Amplitude vs stability", xlabel="stability sel. thres.", ylabel="|A_mean_selected|")
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)

        false_bound = float(stage_b.get("stability_false_edge_bound", np.nan))
        title_prefix = self.canvas_title_prefix(md)
        fig.suptitle(
            f"{title_prefix}, FDR bag selection: q={per_bag_quantile:g}, "
            f"stab_sel_thresh={stab_sel_thresh:g}, bags={n_bag}, final={final_count}, "
            f"bound={false_bound:.3g}",
            fontsize=13,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.955])

    def fdr_acceptance_truth(self, accD, md, figId="o"):
        """Display simulation-truth FDR/bag acceptance diagnostics precomputed by eval."""
        weight = accD["weight"]
        rate = accD["rate"]
        summary = accD["summary"]

        wc = np.asarray(weight["center"], dtype=np.float64)
        wprob = np.asarray(weight["prob"], dtype=np.float64)
        wtotal = np.asarray(weight["total"], dtype=np.int64)
        wpass = np.asarray(weight["passed"], dtype=np.int64)
        wbin = float(weight["bin_width"])

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(18, 4.2))
        gs = fig.add_gridspec(1, 4, left=0.055, right=0.99, bottom=0.18, top=0.78, wspace=0.34)

        ax = fig.add_subplot(gs[0, 0])
        m = np.isfinite(wprob)
        ax.bar(wc[m], wprob[m], width=0.92 * wbin, color="tab:purple", alpha=0.85, align="center")
        ax.set(
            title="Acceptance vs true edge value",
            xlabel="A_true edge value",
            ylabel="P(A_prune edge accepted)",
            ylim=(-0.03, 1.03),
        )
        ax.grid(True, alpha=0.35)

        ax = fig.add_subplot(gs[0, 1])
        for label, color in (("exc", "red"), ("inh", "blue")):
            d = rate[label]
            x = np.asarray(d["center"], dtype=np.float64)
            y = np.asarray(d["prob"], dtype=np.float64)
            mm = np.isfinite(y)
            ax.plot(x[mm], y[mm], "o-", color=color, lw=1.2, ms=4, label=label)
        if rate["scale"] == "log":
            ax.set_xscale("log")
        ax.set(
            title="Acceptance vs source frequency",
            xlabel="source neuron frequency (Hz)",
            ylabel="P(A_prune edge accepted)",
            ylim=(-0.03, 1.03),
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.35)

        ax = fig.add_subplot(gs[0, 2])
        mt = wtotal > 0
        mp = wpass > 0
        ax.bar(wc[mt], wtotal[mt], width=0.92 * wbin, color="0.75", alpha=0.8, label="true edges")
        ax.bar(wc[mp], wpass[mp], width=0.55 * wbin, color="tab:green", alpha=0.85, label="accepted")
        ax.set_yscale("log")
        ax.set(title="True-edge counts vs value", xlabel="A_true edge value", ylabel="edges")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.35)

        ax = fig.add_subplot(gs[0, 3])
        for label, color, ls in (("exc", "red", "-"), ("inh", "blue", "--")):
            d = rate[label]
            x = np.asarray(d["center"], dtype=np.float64)
            total = np.asarray(d["total"], dtype=np.int64)
            passed = np.asarray(d["passed"], dtype=np.int64)
            mt = total > 0
            mp = passed > 0
            ax.plot(x[mt], total[mt], ls=ls, color=color, lw=1.0, alpha=0.45, label=f"{label} true edges")
            ax.plot(x[mp], passed[mp], "o-", color=color, lw=1.2, ms=4, label=f"{label} accepted")
        if rate["scale"] == "log":
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(title="True-edge counts vs source frequency", xlabel="source neuron frequency (Hz)", ylabel="edges")
        ax.legend(fontsize=8, ncol=1)
        ax.grid(True, alpha=0.35)

        title_prefix = self.canvas_title_prefix(md)
        fig.suptitle(
            f"{title_prefix}, FDR/bag A_prune acceptance vs truth: "
            f"{summary['num_accepted']}/{summary['num_candidates']} true edges accepted, "
            f"truth exc={summary['num_exc']} inh={summary['num_inh']}",
            fontsize=12,
        )

    def _A_init_diagnostics_txt(self, md):
        initA = real_fit_metadata(md).get("init_A", {})
        if initA is None:
            initA = {}
        method = initA.get("method", "unknown")
        lines = ["A-init diagnostics:", f"  method = {method}"]
        if "source_key" in initA:
            lines.append(f"  source = {initA['source_key']}")
        if "cond_YpYp" in initA:
            lines.extend([
                f"  cond(YpYp) = {initA['cond_YpYp']:.2e}",
                f"  rho(A_ols) = {initA['rho_A_init']:.3f}",
                f"  R2         = {initA['R2_1step']:.3f}",
                f"  ||A||_F    = {initA['fro_A_init']:.3f}",
                f"  bins_used  = {initA['num_bins_used']}/{initA['num_bins_total']}",
            ])
        if "reference_file" in initA:
            lines.append(f"  ref = {initA['reference_file']}")
        return "\n".join(lines)

    def _debias_init_diagnostics_txt(self, md):
        stg = md.get("deBias_stageC", {})
        lines = [
            "Stage (c) init:",
            "  A = Stage (b) A_hat",
            "  B = Stage (b) B_hat",
            "  support = selected_mask + diag",
        ]
        if "num_dale_active" in stg:
            lines.append(f"  Dale edges = {stg['num_dale_active']}")
        if "num_free_active" in stg:
            lines.append(f"  free A params = {stg['num_free_active']}")
        return "\n".join(lines)

    def A_fitted(self, fitD, md, single_rates, figId="b"):
        """Fitted A summary using only arrays stored in the prismEM file."""
        A_init = np.asarray(fitD["A_init"])
        A_hat = np.asarray(fitD["A_hat"])
        assert A_init.shape == A_hat.shape and A_init.ndim == 2, "A_init and A_hat must match"
        n_neuron = A_init.shape[0]
        x_src = np.arange(n_neuron, dtype=np.int64)
        single_rates = np.asarray(single_rates, dtype=np.float64).ravel()
        assert single_rates.shape[0] == n_neuron, (
            f"single_rates length {single_rates.shape[0]} != N={n_neuron}"
        )

        num_exc = None
        dale_conf = md.get("dale_conf")
        if dale_conf is not None:
            num_exc = dale_conf.get("num_excite")

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(18, 8))
        gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.45)

        if is_stage_c_metadata(md):
            init_label = "A_debias_init"
            fit_label = "A_debias"
            init_txt = self._debias_init_diagnostics_txt(md)
        else:
            init_label = "A_init"
            fit_label = "A_hat"
            init_txt = self._A_init_diagnostics_txt(md)
        rows = [
            (A_init, init_label, init_txt),
            (A_hat, fit_label, None),
        ]
        for row, (A, label, diag_txt) in enumerate(rows):
            _, n_diag, n_off = self._count_A_edges(A)
            cnt_src, sum_src = self._source_nz_edge_stats(A)

            ax = fig.add_subplot(gs[row, 0])
            self._draw_A_matrix(ax, A, f"{label}, nEdges={n_diag}+{n_off}", num_exc=num_exc)

            ax = fig.add_subplot(gs[row, 1])
            self._draw_A_offdiag_hist(ax, A, f"{label} off-diagonal", diagnostics_txt=diag_txt)

            ax = fig.add_subplot(gs[row, 2])
            ax.plot(x_src, cnt_src, color="tab:blue", linewidth=1.0)
            med_cnt = int(np.median(cnt_src))
            ax.axhline(med_cnt, color="green", ls="--", lw=1.0, alpha=0.9)
            ax.text(
                0.02, 0.95, f"median={med_cnt:d}", transform=ax.transAxes,
                va="top", ha="left", fontsize=9, color="green",
            )
            ax.set_title(f"{label}: # non-zero edges")
            ax.set_xlabel("source neuron index (column)")
            ax.set_ylabel("# outgoing non-zero edges")
            ax.grid(True, alpha=0.35)
            if row == 1:
                self._overlay_single_rates(ax, single_rates, x_src)

            ax = fig.add_subplot(gs[row, 3])
            ax.plot(x_src, sum_src, color="tab:green", linewidth=1.0)
            ax.axhline(0.0, color="k", ls="--", lw=1.0, alpha=0.9)
            ax.set_title(f"{label}: sum non-zero edges")
            ax.set_xlabel("source neuron index (column)")
            ax.set_ylabel("sum outgoing edge value")
            ax.grid(True, alpha=0.35)
            if row == 1:
                self._overlay_single_rates(ax, single_rates, x_src)

        title_prefix = self.canvas_title_prefix(md)
        fig.suptitle(f"{title_prefix}, {fit_label} fitted summary: neur. freq. sorted", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    def _state_seq_window(self, trainMD, time_range_sec):
        dt = float(trainMD["time_step_sec"])
        t0_bin, t1_bin = [int(x) for x in trainMD["time_range_bins"]]

        assert time_range_sec is not None, "state sequence plots require time_range_sec"
        t0_req, t1_req = [float(x) for x in time_range_sec]
        if t1_req < t0_req:
            t0_req, t1_req = t1_req, t0_req
        b0 = max(t0_bin, int(np.floor(t0_req / dt)))
        b1 = min(t1_bin, int(np.floor(t1_req / dt)))
        if b1 <= b0:
            raise ValueError("Requested --time_range_sec leaves no bins in fitted window")

        i0 = b0 - t0_bin
        i1 = b1 - t0_bin
        t_bins = np.arange(b0, b1 + 1, dtype=np.float64) * dt
        t_pairs = np.arange(b0, b1, dtype=np.float64) * dt
        return i0, i1, t_bins, t_pairs

    def _fit_loss_time(self, fitD, md, p0, p1):
        if "loss_nll_time" in fitD and "loss_l2_time" in fitD:
            return (
                np.asarray(fitD["loss_nll_time"])[p0:p1],
                np.asarray(fitD["loss_l2_time"])[p0:p1],
            )
        eval_f = md.get("eval_f")
        if eval_f is not None and "loss_nll_time" in eval_f and "loss_l2_time" in eval_f:
            return (
                np.asarray(eval_f["loss_nll_time"])[p0:p1],
                np.asarray(eval_f["loss_l2_time"])[p0:p1],
            )
        raise AssertionError(
            "state_seq_fit requires precomputed loss_nll_time/loss_l2_time in the input npz; "
            "move time-loss computation upstream"
        )

    def state_seq_fit(self, fitD, md, figId="c", time_range_sec=None):
        from matplotlib.ticker import MaxNLocator

        trainMD = real_fit_metadata(md)["train"]
        i0, i1, t_bins, t_pairs = self._state_seq_window(trainMD, time_range_sec)
        p0, p1 = i0, i1

        S_hat = np.asarray(fitD["S_hat"])[i0:i1 + 1]
        C_hat = np.asarray(fitD["c_hat"])[i0:i1 + 1]
        S_hat_CL = np.asarray(fitD["S_hat_CL"])[i0:i1 + 1]
        nll_t, l2_t = self._fit_loss_time(fitD, md, p0, p1)

        eval_f = md.get("eval_f", {})
        acc_txt = f"acc={float(eval_f['acc']):.3f}" if "acc" in eval_f else "fit only"
        x0, x1 = float(t_bins[0]), float(t_bins[-1])

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(12, 7.2))
        gs = fig.add_gridspec(
            4, 3,
            height_ratios=[0.95, 0.22, 1.15, 0.70],
            hspace=0.78,
            wspace=0.38,
        )

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
        ax.axis("off")

        ax = fig.add_subplot(gs[0, 2])
        cl = np.asarray(S_hat_CL, dtype=np.float64)
        ax.hist(cl, bins=30, color="tab:purple", alpha=0.7)
        ax.axvline(np.mean(cl), color="k", linestyle="--", linewidth=1.0)
        ax.set(title=f"Confidence ({acc_txt})", xlabel="S_hat_CL", ylabel="count")
        ax.grid(True, alpha=0.3)
        ax.set_box_aspect(0.9)

        ax = fig.add_subplot(gs[2, :])
        line_s, = ax.plot(t_bins, S_hat, color="k", linewidth=2.0, label="S_hat", zorder=4)
        lo = np.clip(S_hat - S_hat_CL, 0.0, float(C_hat.shape[1] - 1))
        hi = np.clip(S_hat + S_hat_CL, 0.0, float(C_hat.shape[1] - 1))
        fill = ax.fill_between(t_bins, lo, hi, color="gray", alpha=0.25, label="S_hat_CL", zorder=3)
        ax.set(title=f"Fit ({acc_txt})", xlabel="time (s)", ylabel="state")
        ax.set_xlim(x0, x1)
        ax.set_ylim(-0.03, float(C_hat.shape[1] - 1) + 0.03)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        c_lines = []
        for m in range(C_hat.shape[1]):
            line, = ax2.plot(t_bins, C_hat[:, m], linewidth=0.8, alpha=0.85, label=f"C_hat[{m}]")
            c_lines.append(line)
        ax2.set_ylabel("C_hat")
        ax2.set_ylim(ax.get_ylim())
        handles = [line_s, fill] + c_lines
        labels = [h.get_label() for h in handles]
        ax.legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, 1.18),
            ncol=len(handles), fontsize=8, frameon=True,
        )

        ax = fig.add_subplot(gs[3, :])
        ax.plot(t_pairs, nll_t, color="tab:blue", linewidth=0.9, label="nll")
        ax.plot(t_pairs, l2_t, color="tab:orange", linewidth=0.9, label="l2")
        ax.set_yscale("log")
        ax.set(title="Loss (time)", xlabel="time (s)", ylabel="value")
        ax.set_xlim(x0, x1)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=8)

        ax2 = ax.twinx()
        ax2.plot(t_bins, S_hat, color="k", linewidth=0.8, alpha=0.65, label="S_hat")
        ax2.set_ylabel("state")
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), fontsize=8)

        title_prefix = self.canvas_title_prefix(md)
        fig.suptitle(
            f"{title_prefix}, fitted state sequence: lambda2={trainMD['lambda2']}, "
            f"lr={trainMD['lr_estep']}, pgd_iter={trainMD['pgd_iter']}",
            fontsize=12,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

    def spike_bursts(self, rebD, md, figId="d", time_range_sec=None):
        from PlotterBioExp import Plotter as PlotterBioExp

        if time_range_sec is None:
            time_range_sec = md["plot"]["time_rangeLR"]
        clip = clip_rebD_time(rebD, time_range_sec)
        rate_thr = float(rebD["rate_thres2"])
        high_chan = np.asarray(rebD["highChanCnt"])[clip["itL"]:clip["itR"]]
        timeV = clip["timeV"]
        Tbin = clip["Tbin"]
        tL, tR = clip["tL"], clip["tR"]
        nchan = clip["nchan"]

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(16, 11))
        gs = fig.add_gridspec(4, 1, height_ratios=[0.15, 0.15, 0.54, 0.01])

        ax = fig.add_subplot(gs[0, 0])
        ax.bar(timeV + Tbin * 0.5, high_chan, width=Tbin,
               color="forestgreen", align="center", alpha=0.7)
        ax.set(
            ylabel="num neurons",
            title=f"num neurons with instantaneous rate > thres={rate_thr:.0f} Hz, nchan={nchan}",
        )
        ax.set_xlim(tL, tR)
        ax.grid(True, alpha=0.35)

        ax = fig.add_subplot(gs[1, 0])
        PlotterBioExp.plot_bioexp_sum_rate(self, ax, rebD, clip)

        ax = fig.add_subplot(gs[2, 0])
        PlotterBioExp.plot_bioexp_heatmap(self, fig, ax, rebD, md["short_name"], clip)

        title_prefix = self.canvas_title_prefix(md)
        fig.suptitle(
            f"{title_prefix}, spike bursts: T=[{float(tL):g}, {float(tR):g}] s",
            fontsize=12,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

    def _node_Ahat_topology_ax(
        self, ax, loc_x, loc_y, A_hat, sign=None, neuron_type=None,
        node_size=16, source_select=None, und_color="salmon",
    ):
        ax.set_facecolor("white")
        if neuron_type is None:
            ax.scatter(
                loc_x, loc_y, s=node_size, marker="o", facecolors="white",
                edgecolors="k", linewidths=0.4, alpha=0.95, zorder=3,
            )
        else:
            neuron_type = np.asarray(neuron_type)
            for cls, color in ((1, "magenta"), (-1, "forestgreen"), (0, und_color)):
                mask = neuron_type == cls
                if np.any(mask):
                    ax.scatter(
                        loc_x[mask], loc_y[mask],
                        s=neuron_type_marker_size(cls, node_size),
                        marker=neuron_type_marker(cls), c=color,
                        edgecolors="k", linewidths=0.4, alpha=0.95, zorder=3,
                    )

        edge_mask = A_hat != 0
        np.fill_diagonal(edge_mask, False)
        edge_color = None
        if sign == "pos":
            edge_mask &= A_hat > 0
            edge_color = "red"
        elif sign == "neg":
            edge_mask &= A_hat < 0
            edge_color = "blue"

        post_idx, pre_idx = np.where(edge_mask)
        n_total = int(post_idx.size)
        if source_select is not None:
            source_select = np.asarray(source_select, dtype=bool)
            draw_mask = source_select[pre_idx]
            post_draw = post_idx[draw_mask]
            pre_draw = pre_idx[draw_mask]
        else:
            post_draw = post_idx
            pre_draw = pre_idx

        for i_post, j_pre in zip(post_draw, pre_draw):
            color = edge_color
            if color is None:
                color = "red" if A_hat[i_post, j_pre] > 0 else "blue"
            ax.plot(
                [loc_x[j_pre], loc_x[i_post]],
                [loc_y[j_pre], loc_y[i_post]],
                color=color, alpha=0.7, linewidth=0.7, zorder=2,
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("node x", fontsize=11)
        ax.set_ylabel("node y", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.35)
        return n_total, int(post_draw.size)

    def neuron_waveforms_by_nedge(
        self, fitD, nodeD, nodeMD, neuron_type, type_value, figId="j",
    ):
        """Plot high, median, and lower-quartile waveforms for one node type."""
        assert int(type_value) in (-1, 1), "type_value must be 1 (exc) or -1 (inh)"
        assert int(nodeMD.get("bioexp_schema_version", -1)) == 3, (
            "Waveform plots require bioexp_schema_version=3"
        )
        assert bool(nodeMD.get("waveforms_available", False)), (
            "Waveform plots require waveforms_available=true in the bioExp metadata"
        )
        required = (
            "raw_mean_templates",
            "waveform_num_samples",
            "waveform_unit_ids",
            "waveform_channel_ids",
            "waveform_ms_before",
            "waveform_ms_after",
            "waveform_grid_distance",
            "waveform_is_multichannel",
        )
        missing = [key for key in required if key not in nodeD]
        assert not missing, f"Waveform plots require node-data arrays: {missing}"

        waveforms = np.asarray(nodeD["raw_mean_templates"], dtype=np.float64)
        num_samples = np.asarray(nodeD["waveform_num_samples"], dtype=np.int64)
        unit_ids = np.asarray(nodeD["waveform_unit_ids"]).reshape(-1)
        channel_ids = np.asarray(nodeD["waveform_channel_ids"]).reshape(-1)
        ms_before = np.asarray(nodeD["waveform_ms_before"], dtype=np.float64)
        ms_after = np.asarray(nodeD["waveform_ms_after"], dtype=np.float64)
        grid_distance = np.asarray(
            nodeD["waveform_grid_distance"], dtype=np.float64
        )
        is_multichannel = np.asarray(nodeD["waveform_is_multichannel"])
        neuron_type = np.asarray(neuron_type, dtype=np.int8).reshape(-1)
        A_est = np.asarray(fitD["A_hat"], dtype=np.float64)

        assert waveforms.ndim == 2, "raw_mean_templates must have shape (N, samples)"
        n_neuron = waveforms.shape[0]
        assert A_est.shape == (n_neuron, n_neuron), (
            f"A_hat shape {A_est.shape} does not match {n_neuron} waveforms"
        )
        for name, values in (
            ("waveform_num_samples", num_samples),
            ("waveform_unit_ids", unit_ids),
            ("waveform_channel_ids", channel_ids),
            ("waveform_ms_before", ms_before),
            ("waveform_ms_after", ms_after),
            ("waveform_grid_distance", grid_distance),
            ("waveform_is_multichannel", is_multichannel),
            ("neuron_type", neuron_type),
        ):
            assert values.shape == (n_neuron,), (
                f"{name} shape {values.shape} does not match N={n_neuron}"
            )
        assert is_multichannel.dtype == np.bool_, (
            "waveform_is_multichannel must have Boolean dtype"
        )
        if "MEA_idx" in nodeD:
            assert np.array_equal(unit_ids, np.asarray(nodeD["MEA_idx"]).reshape(-1)), (
                "waveform_unit_ids must align with MEA_idx and the fitted neuron axis"
            )
        assert np.all((num_samples > 0) & (num_samples <= waveforms.shape[1])), (
            "waveform_num_samples must be within the raw_mean_templates width"
        )

        nedge, _ = self._node_outgoing_edge_stats(A_est)
        class_indices = np.flatnonzero(neuron_type == int(type_value))
        type_name = "excitatory" if int(type_value) == 1 else "inhibitory"
        assert class_indices.size >= 9, (
            f"Plot requires at least 9 {type_name} nodes, found {class_indices.size}"
        )
        order = class_indices[
            np.lexsort((class_indices, nedge[class_indices]))
        ]
        median_start = (order.size - 3) // 2
        highest_indices = order[-3:][::-1]
        median_indices = order[median_start:median_start + 3]
        lower_quartile_nedge = float(
            np.percentile(nedge[class_indices], 25.0)
        )
        reserved = set(highest_indices) | set(median_indices)
        quartile_candidates = np.asarray(
            [index for index in class_indices if index not in reserved],
            dtype=np.int64,
        )
        quartile_order = np.lexsort(
            (
                quartile_candidates,
                nedge[quartile_candidates],
                np.abs(nedge[quartile_candidates] - lower_quartile_nedge),
            )
        )
        lower_quartile_indices = quartile_candidates[quartile_order[:3]]
        groups = (
            ("highest", highest_indices),
            ("median", median_indices),
            ("25th percentile", lower_quartile_indices),
        )

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(10.5, 7.0))
        gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.30)
        waveform_color = (
            "magenta" if int(type_value) == 1 else "forestgreen"
        )
        for row, (rank_label, indices) in enumerate(groups):
            for col, index in enumerate(indices):
                axis = fig.add_subplot(gs[row, col])
                count = int(num_samples[index])
                waveform = waveforms[index, :count]
                assert np.all(np.isfinite(waveform)), (
                    f"Unit {unit_ids[index]} waveform contains non-finite samples"
                )
                time_ms = np.linspace(
                    -float(ms_before[index]),
                    float(ms_after[index]),
                    count,
                    endpoint=False,
                )
                if count < 2 or time_ms[-1] == time_ms[0]:
                    average_level = float(np.mean(waveform))
                else:
                    interval_integrals = (
                        0.5
                        * (waveform[:-1] + waveform[1:])
                        * np.diff(time_ms)
                    )
                    average_level = float(
                        np.sum(interval_integrals)
                        / (time_ms[-1] - time_ms[0])
                    )
                axis.plot(
                    time_ms, waveform, color=waveform_color, linewidth=1.7
                )
                axis.axhline(
                    average_level,
                    color="0.2",
                    linestyle="--",
                    linewidth=1.0,
                )
                axis.fill_between(
                    time_ms,
                    waveform,
                    average_level,
                    where=waveform >= average_level,
                    interpolate=True,
                    color="tab:orange",
                    alpha=0.30,
                )
                axis.fill_between(
                    time_ms,
                    waveform,
                    average_level,
                    where=waveform < average_level,
                    interpolate=True,
                    color="tab:blue",
                    alpha=0.25,
                )
                axis.axvline(0.0, color="0.45", ls="--", lw=0.8)
                axis.set_title(
                    f"{rank_label}\n"
                    f"unit {unit_ids[index]}, channel {channel_ids[index]}\n"
                    f"Nedge={nedge[index]}",
                    fontsize=8,
                )
                if row == 2:
                    axis.set_xlabel("Time (ms)", fontsize=9)
                axis.set_ylabel("Raw signal", fontsize=9)
                axis.tick_params(labelsize=8)
                axis.grid(True, alpha=0.30)

        fig.suptitle(
            f"{nodeMD['short_name']}: identified {type_name} node waveforms "
            "by outgoing Nedge",
            fontsize=10,
        )
        fig.subplots_adjust(
            left=0.07, right=0.98, bottom=0.07, top=0.88,
            hspace=0.65, wspace=0.30,
        )

    def neuron_spatial_edges_split(
        self, fitD, nodeD, nodeMD, neuron_type, neuron_Sedge,
        maxNeurons=24, figId="f",
    ):
        node_pos = np.asarray(nodeD["node_positions"], dtype=np.float64)
        assert node_pos.ndim == 2 and node_pos.shape[1] == 2, (
            f"node_positions must have shape (N, 2), got {node_pos.shape}"
        )
        loc_x = node_pos[:, 0]
        loc_y = node_pos[:, 1]
        A_prune = self._A_prune_from_fitD(fitD)
        n_neur = loc_x.shape[0]
        assert A_prune.shape == (n_neur, n_neur), (
            f"A_prune shape {A_prune.shape} does not match {n_neur} neuron locations"
        )
        neuron_type = np.asarray(neuron_type, dtype=np.int8)
        assert neuron_type.shape[0] == n_neur, "neuron_type length must match neuron locations"
        neuron_Sedge = np.asarray(neuron_Sedge, dtype=np.float64)
        assert neuron_Sedge.shape[0] == n_neur, "neuron_Sedge length must match neuron locations"
        n_exc = int(np.sum(neuron_type > 0))
        n_inh = int(np.sum(neuron_type < 0))
        n_und = int(np.sum(neuron_type == 0))

        def ranked_source_select(type_value):
            idx = np.where(neuron_type == int(type_value))[0]
            idx = idx[np.argsort(neuron_Sedge[idx])]
            n_side = max(1, int(maxNeurons) // 2)
            if idx.size <= int(maxNeurons):
                sel = idx
            else:
                sel = np.unique(np.concatenate([idx[:n_side], idx[-n_side:]]))
            source_select = np.zeros((n_neur,), dtype=bool)
            source_select[sel] = True
            return source_select, int(sel.size)

        exc_sources, n_exc_show = ranked_source_select(1)
        inh_sources, n_inh_show = ranked_source_select(-1)

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(14, 4.8))
        gs = fig.add_gridspec(
            1, 2, left=0.06, right=0.99, bottom=0.13, top=0.84, wspace=0.18
        )

        ax = fig.add_subplot(gs[0, 0])
        n_pos, n_pos_show = self._node_Ahat_topology_ax(
            ax, loc_x, loc_y, A_prune, sign="pos",
            neuron_type=neuron_type, node_size=32, source_select=exc_sources,
            und_color="yellow",
        )
        ax.set_title(
            f"A_prune>0, shown edges {n_pos_show}/{n_pos}, neurons {n_exc_show}/{n_exc}",
            fontsize=11, pad=8,
        )
        ax.scatter([], [], s=neuron_type_marker_size(1, 32), marker=neuron_type_marker(1), c="magenta", edgecolors="k", label=f"exc={n_exc}")
        ax.scatter([], [], s=neuron_type_marker_size(-1, 32), marker=neuron_type_marker(-1), c="forestgreen", edgecolors="k", label=f"inh={n_inh}")
        ax.scatter([], [], s=neuron_type_marker_size(0, 32), marker=neuron_type_marker(0), c="yellow", edgecolors="k", label=f"und={n_und}")
        ax.legend(loc="best", fontsize=9)

        ax = fig.add_subplot(gs[0, 1])
        n_neg, n_neg_show = self._node_Ahat_topology_ax(
            ax, loc_x, loc_y, A_prune, sign="neg",
            neuron_type=neuron_type, node_size=32, source_select=inh_sources,
            und_color="yellow",
        )
        ax.set_title(
            f"A_prune<0, shown edges {n_neg_show}/{n_neg}, neurons {n_inh_show}/{n_inh}",
            fontsize=11, pad=8,
        )
        ax.scatter([], [], s=neuron_type_marker_size(1, 32), marker=neuron_type_marker(1), c="magenta", edgecolors="k", label=f"exc={n_exc}")
        ax.scatter([], [], s=neuron_type_marker_size(-1, 32), marker=neuron_type_marker(-1), c="forestgreen", edgecolors="k", label=f"inh={n_inh}")
        ax.scatter([], [], s=neuron_type_marker_size(0, 32), marker=neuron_type_marker(0), c="yellow", edgecolors="k", label=f"und={n_und}")
        ax.legend(loc="best", fontsize=9)

        title_prefix = self.canvas_title_prefix(nodeMD)
        fig.suptitle(
            f"{title_prefix}, neuron spatial A_prune topology: max neurons per panel={int(maxNeurons)}",
            fontsize=12, y=0.97,
        )

    def state_seq_simu(self, fitD, md, figId="m", time_range_sec=None):
        from matplotlib.ticker import MaxNLocator

        trainMD = real_fit_metadata(md)["train"]
        srec = real_fit_metadata(md)["states_recovery_eval"]
        assert "eval_f" in md, (
            "state_seq_simu requires precomputed eval_f in the input metadata; "
            "move the old evaluator's eval_em_metrics_time computation upstream"
        )
        assert "avg_acc" in srec and "state_acc_cl" in srec, (
            "state_seq_simu requires precomputed states_recovery_eval avg_acc/state_acc_cl; "
            "move the old evaluator's eval_true_state_recovery computation upstream"
        )
        dt = float(trainMD["time_step_sec"])
        t0_bin = int(trainMD["time_range_bins"][0])
        i0, i1, t_bins, t_pairs = self._state_seq_window(trainMD, time_range_sec)
        b0 = t0_bin + i0
        b1 = t0_bin + i1
        p0 = i0
        p1 = i1

        S_hat = np.asarray(fitD["S_hat"])[i0:i1 + 1]
        C_hat = np.asarray(fitD["c_hat"])[i0:i1 + 1]
        S_hat_CL = np.asarray(fitD["S_hat_CL"])[i0:i1 + 1]

        S_true = np.asarray(md["S_true"])[b0:b1 + 1]
        C_true = np.asarray(md["C_true"])[b0:b1 + 1]

        eval_f = md["eval_f"]
        nll_t, l2_t = self._fit_loss_time(fitD, md, p0, p1)
        acc = float(eval_f["acc"])

        x0, x1 = float(t_bins[0]), float(t_bins[-1])

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(12, 9))
        gs = fig.add_gridspec(
            5, 3,
            height_ratios=[0.95, 0.18, 1.0, 1.0, 0.55],
            hspace=0.75,
            wspace=0.35,
        )

        ax = fig.add_subplot(gs[2, :])
        ax.plot(t_bins, S_hat, color="k", linewidth=2.5, label="S_hat")
        ax.plot(t_bins, S_true, color="lime", linestyle="--", linewidth=1.5, label="S_true")
        lo = np.clip(S_hat - S_hat_CL, 0.0, float(C_hat.shape[1] - 1))
        hi = np.clip(S_hat + S_hat_CL, 0.0, float(C_hat.shape[1] - 1))
        ax.fill_between(t_bins, lo, hi, color="gray", alpha=0.3, label="S_hat_CL")
        ax.set(title=f"Fit (acc={acc:.3f})", xlabel="time (s)", ylabel="state")
        ax.set_xlim(x0, x1)
        ax.set_ylim(-0.03, float(C_hat.shape[1] - 1) + 0.03)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        for m in range(C_hat.shape[1]):
            ax2.plot(t_bins, C_hat[:, m], linewidth=0.8, alpha=0.85, label=f"C_hat[{m}]")
        ax2.set_ylabel("C_hat")
        ax2.set_ylim(ax.get_ylim())
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=8)
        ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=4, fontsize=8)

        ax = fig.add_subplot(gs[3, :])
        ax.plot(t_bins, S_true, color="lime", linestyle="--", linewidth=1.5, label="S_true")
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
        ax.errorbar(x, acc_ps, yerr=cl_ps, fmt="o", color="k", capsize=5, markersize=4)
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

        title_prefix = self.canvas_title_prefix(md)
        fig.suptitle(
            f"{title_prefix}, state sequence: lambda2={trainMD['lambda2']}, "
            f"lr={trainMD['lr_estep']}, pgd_iter={trainMD['pgd_iter']}, "
            f"decode_dwell={srec['decode_dwell_sec']}s",
            fontsize=12,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
