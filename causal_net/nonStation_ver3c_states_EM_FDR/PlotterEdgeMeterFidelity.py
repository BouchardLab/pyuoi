#!/usr/bin/env python3
"""Plotting utilities for PRISM-FDR graph stability metrics (edgeMaterFidelity3c)."""

import numpy as np

from toolbox.PlotterBackbone import PlotterBackbone

SCOPES = ["all", "exc", "inh"]
SCOPE_TITLES = {"all": "all edges", "exc": "excitatory-source edges", "inh": "inhibitory-source edges"}
SCOPE_COLORS = {"all": "tab:blue", "exc": "tab:red", "inh": "tab:green"}


class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self, args)

    def figId2name(self, fid):
        if isinstance(fid, str):
            return f"{self.jobName}_{fid}"
        return f"{self.jobName}_f{fid}"

    def smart_append(self, fig_id):
        if fig_id in self.figL:
            raise ValueError(f"Figure id {fig_id!r} already in use")
        self.figL.append(fig_id)
        return fig_id

    # ------------------------------------------------------------------
    # shared helpers
    # ------------------------------------------------------------------

    def _x_and_label(self, out_d):
        x = np.asarray(out_d["x"], dtype=np.float64)
        mode = self._mode_str(out_d)
        if np.all(np.isfinite(x)):
            xlabel = "duration (min)" if mode == "last" else "duration midpoint (min)"
            return x, xlabel
        return np.arange(len(x), dtype=np.float64), "comparison index"

    def _subset_x_and_label(self, out_d):
        x = np.asarray(out_d["subset_x"], dtype=np.float64)
        if np.all(np.isfinite(x)):
            return x, "duration (min)"
        return np.arange(len(x), dtype=np.float64), "data subset index"

    def _xtick_labels(self, out_d):
        return [str(v) for v in np.asarray(out_d["labels"], dtype=object)]

    def _mode_str(self, out_d):
        return str(np.asarray(out_d["compare_mode"]).ravel()[0])

    def _suptitle(self, out_md, out_d, metric_name):
        mode = out_md.get("compare_mode", "?")
        trunk = out_md.get("fitNameTrunk", "")
        if mode == "last":
            ref_tag = str(np.asarray(out_d["subset_tags"]).ravel()[-1])
            mode_str = f"last={ref_tag}"
        else:
            mode_str = mode
        return f"{trunk} — {metric_name}  [{mode_str}]"

    def _apply_int_xticks(self, ax, x):
        """Format x-axis ticks as integers (no decimal point)."""
        ax.xaxis.set_major_formatter(self.plt.FuncFormatter(lambda v, _: f"{int(round(v))}"))
        ax.set_xticks(x)

    def _bin_edges_from_centers(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.size == 1:
            return np.array([x[0] - 0.5, x[0] + 0.5], dtype=np.float64)
        edges = np.empty(x.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (x[:-1] + x[1:])
        edges[0] = x[0] - 0.5 * (x[1] - x[0])
        edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
        return edges

    def _plot_scope_rows(self, fig, axs, x, xlabel, key, ylabel, ylim, out_d, out_md):
        """Fill a 3-row axes array, one row per scope, for a single scalar metric key."""
        for row, scope in enumerate(SCOPES):
            ax = axs[row]
            y = np.asarray(out_d[f"{scope}_{key}"], dtype=np.float64)
            color = SCOPE_COLORS[scope]
            ax.plot(x, y, "o-", lw=1.6, ms=5, color=color)
            ax.set(title=SCOPE_TITLES[scope], xlabel=xlabel, ylabel=ylabel)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # plot group a — compact topology/sign/Spearman summary  (3 rows x 2 cols)
    # ------------------------------------------------------------------

    def plot_jaccard_smr(self, out_d, out_md, figId="a"):
        figId = self.smart_append(figId)
        fig, axs = self.plt.subplots(3, 2, num=figId, facecolor="white", figsize=(7.2, 6.6))
        fig.subplots_adjust(hspace=0.45, wspace=0.28)
        x, xlabel = self._x_and_label(out_d)

        ax = axs[0, 0]
        for key, label, color in (
            ("all_J_pos", "J_exc", "tab:red"),
            ("all_J_neg", "J_inh", "tab:blue"),
        ):
            y = np.asarray(out_d[key], dtype=np.float64)
            ax.plot(x, y, "o-", lw=1.6, ms=5, color=color, label=label)
        ax.set(title="Jaccard - all edges", xlabel=xlabel, ylabel="Jaccard index")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        self._apply_int_xticks(ax, x)

        ax = axs[0, 1]
        y = np.asarray(out_d["all_SMR"], dtype=np.float64)
        ax.plot(x, y, "o-", lw=1.6, ms=5, color=SCOPE_COLORS["all"])
        ax.set(title="Signal Match Rate - all edges", xlabel=xlabel, ylabel="SMR")
        ax.grid(True, alpha=0.3)
        self._apply_int_xticks(ax, x)

        ax = axs[1, 1]
        y = np.asarray(out_d["diag_mae"], dtype=np.float64)
        ax.plot(x, y, "o-", lw=1.6, ms=5, color="tab:purple")
        ax.set(title="Mean |Delta A_ii|", xlabel=xlabel, ylabel="mean abs diagonal diff")
        ax.grid(True, alpha=0.3)
        self._apply_int_xticks(ax, x)

        axs[2, 1].set_axis_off()

        for row, key, title, ylabel in (
            (1, "r_spearman", "Spearman r - source types", "Spearman r"),
            (2, "r_spearman_w", "Weighted Spearman r - source types", "weighted Spearman r"),
        ):
            ax = axs[row, 0]
            for scope, label, color in (("exc", "exc", "tab:red"), ("inh", "inh", "tab:blue")):
                y = np.asarray(out_d[f"{scope}_{key}"], dtype=np.float64)
                ax.plot(x, y, "o-", lw=1.6, ms=5, color=color, label=label)
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            self._apply_int_xticks(ax, x)

        fig.suptitle(self._suptitle(out_md, out_d, "Edge Fidelity Summary"), fontsize=13)

    # ------------------------------------------------------------------
    # plot group b — weight thresholds min_posW / max_negW vs duration
    # ------------------------------------------------------------------

    def plot_weight_thresholds(self, out_d, out_md, figId="b"):
        figId = self.smart_append(figId)
        fig, axs = self.plt.subplots(1, 4, num=figId, facecolor="white", figsize=(16, 4))
        fig.subplots_adjust(wspace=0.38, left=0.06, right=0.97, top=0.88)

        sx, sxlabel = self._subset_x_and_label(out_d)
        min_posW = np.asarray(out_d["min_posW"], dtype=np.float64)
        max_negW = np.asarray(out_d["max_negW"], dtype=np.float64)

        ax = axs[0]
        ax.plot(sx, min_posW, "o-", lw=1.6, ms=5, color="tab:red",   label="min_posW")
        ax.plot(sx, max_negW, "s-", lw=1.6, ms=5, color="tab:green", label="max_negW")
        ax.axhline(0.0, color="k", ls="--", lw=0.8)
        ax.set(title="Weight thresholds vs duration", xlabel=sxlabel, ylabel="weight")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        self._apply_int_xticks(ax, sx)

        ax = axs[1]
        for key, color, label in [
            ("exc", "magenta", "exc"),
            ("inh", "#39FF14", "inh"),
        ]:
            lo = np.asarray(out_d[f"w_min_{key}"], dtype=np.float64)
            hi = np.asarray(out_d[f"w_max_{key}"], dtype=np.float64)
            ax.fill_between(sx, lo, hi, alpha=0.35, color=color, label=label)
            ax.plot(sx, lo, "-", lw=0.8, color=color)
            ax.plot(sx, hi, "-", lw=0.8, color=color)
            if key in ("exc", "inh"):
                med = np.asarray(out_d[f"w_med_{key}"], dtype=np.float64)
                ax.plot(sx, med, "--", lw=1.2, color="tab:blue")
                q4_idx = len(sx) * 3 // 4
                tx = sx[q4_idx]
                ty = med[q4_idx]
                va = "top" if key == "inh" else "bottom"
                ax.text(tx, ty, f"med {label}", fontsize=8, color="tab:blue",
                        va=va, ha="center")
        ax.axhline(0.0, color="k", ls="--", lw=0.8)
        ax.set(title="Off-diag weight range by type", xlabel=sxlabel, ylabel="weight")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        self._apply_int_xticks(ax, sx)

        ax = axs[2]
        for key, color, label in [
            ("exc", "magenta", "exc"),
            ("inh", "#39FF14", "inh"),
            ("und", "salmon",  "und"),
        ]:
            n = np.asarray(out_d[f"n_edges_{key}"], dtype=np.float64)
            ax.plot(sx, n, "o-", lw=1.6, ms=5, color=color, label=label)
        ax.set(title="# off-diag edges by type", xlabel=sxlabel, ylabel="# edges")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        self._apply_int_xticks(ax, sx)

        ax = axs[3]
        rate_edges = np.asarray(out_d["rate_bin_edges"], dtype=np.float64)
        edge_rate_hist = np.asarray(out_d["edge_count_rate_hist"], dtype=np.float64)
        x_edges = self._bin_edges_from_centers(sx)
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "YlOrBl", ["#fff7bc", "#fec44f", "#d95f0e", "#2b8cbe", "#08519c"]
        )
        cmap.set_bad("white")
        plot_hist = np.ma.masked_where(edge_rate_hist <= 0.0, edge_rate_hist)
        pcm = ax.pcolormesh(x_edges, rate_edges, plot_hist, shading="auto", cmap=cmap)
        fig.colorbar(pcm, ax=ax, label="num edges")
        ax.set(title="recovered exc+inh off-diag edges", xlabel=sxlabel, ylabel="single rate (Hz)")
        self._apply_int_xticks(ax, sx)

        fig.suptitle(self._suptitle(out_md, out_d, "Weight Stats"), fontsize=13)
