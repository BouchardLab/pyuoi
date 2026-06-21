#!/usr/bin/env python3
"""Plotting utilities for absolute graph-recovery metrics."""

import numpy as np

from toolbox.PlotterBackbone import PlotterBackbone


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

    def _x(self, metricD):
        x = np.asarray(metricD["duration_min"], dtype=np.float64)
        return x, "duration (min)"

    def _run_labels(self, metricD):
        tags = np.asarray(metricD["fit_tag"], dtype=object)
        return [str(x) for x in tags]

    def _plot_open_first(self, ax, x, y, marker, linestyle, color, label, lw=1.4, ms=4):
        y = np.asarray(y)
        ax.plot(x, y, marker + linestyle, lw=lw, ms=ms, label=label, color=color)
        if len(x) > 0:
            ax.plot(
                [x[0]], [y[0]], marker=marker, linestyle="None",
                ms=ms, mfc="white", mec=color, mew=1.5, color=color,
            )

    def recovery_summary(self, metricD, md, figId="a"):
        """Main recovery curves and source-type diagnostics."""
        figId = self.smart_append(figId)
        fig, axs = self.plt.subplots(2, 3, num=figId, facecolor="white", figsize=(15, 8))
        fig.subplots_adjust(hspace=0.38, wspace=0.32)
        x, xlabel = self._x(metricD)

        ax = axs[0, 0]
        for key, label, color in (
            ("precision", "precision", "tab:blue"),
            ("recall", "recall", "tab:orange"),
            ("f1", "F1", "tab:green"),
            ("mcc", "MCC", "tab:red"),
        ):
            lw = 2.5 if key == "mcc" else 1.4
            ms = 6 if key == "mcc" else 4
            self._plot_open_first(ax, x, metricD[key], "o", "-", color, label, lw=lw, ms=ms)
        ax.set(title="edge is present (metrics)", xlabel=xlabel, ylabel="metric")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = axs[0, 1]
        for key, label, color in (
            ("signed_precision", "signed precision", "tab:blue"),
            ("signed_recall", "signed recall", "tab:orange"),
            ("signed_f1", "signed F1", "tab:green"),
            ("signed_mcc", "signed MCC", "tab:red"),
            ("sign_flip_rate", "sign flip rate", "tab:purple"),
        ):
            lw = 2.5 if key == "signed_mcc" else 1.4
            ms = 6 if key == "signed_mcc" else 4
            self._plot_open_first(ax, x, metricD[key], "o", "-", color, label, lw=lw, ms=ms)
        ax.set(title="edge sign is correct (metrics)", xlabel=xlabel, ylabel="metric")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = axs[0, 2]
        labels = self._run_labels(metricD)
        xpos = np.arange(len(labels))
        bot = np.zeros(len(labels), dtype=np.float64)
        for key, label, color in (
            ("num_reco_exc_neurons", "exc", "magenta"),
            ("num_reco_inh_neurons", "inh", "limegreen"),
            ("num_reco_und_neurons", "und", "salmon"),
        ):
            vals = np.asarray(metricD[key], dtype=np.float64)
            ax.bar(xpos, vals, bottom=bot, color=color, alpha=0.75, label=label)
            bot += vals
        ax.set(title="Recovered source type", xlabel="fit tag", ylabel="neurons")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

        ax = axs[1, 0]
        self._plot_open_first(
            ax, x, metricD["fdp"], "o", "-", "tab:red",
            "realized FDP", lw=2.5, ms=6,
        )
        self._plot_open_first(
            ax, x, metricD["stability_bound_fdp"], "s", "--", "tab:purple",
            "stability bound / selected", lw=1.1, ms=4,
        )
        ax.set(title="False discoveries", xlabel=xlabel, ylabel="fraction")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = axs[1, 1]
        self._plot_open_first(ax, x, metricD["density_true"], "o", "-", "black", "true", lw=1.4, ms=4)
        self._plot_open_first(ax, x, metricD["density_reco"], "o", "-", "tab:cyan", "reco", lw=1.4, ms=4)
        ax.set(title="Graph density", xlabel=xlabel, ylabel="edges / candidates")
        ax.set_ylim(bottom=-0.002)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = axs[1, 2]
        ax.plot(x, metricD["frac_reco_und_neurons"], "o-", lw=1.4, ms=4,
                label="und neurons / N", color="tab:gray")
        ax.plot(x, metricD["frac_reco_edges_from_reco_und_src"], "o-", lw=1.4, ms=4,
                label="reco edges from und src", color="tab:purple")
        ax.plot(x, metricD["frac_true_edges_from_reco_und_src"], "o-", lw=1.4, ms=4,
                label="true edges from und src", color="tab:brown")
        ax.set(title="Undecided-source diagnostics", xlabel=xlabel, ylabel="fraction")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        fig.suptitle(
            f"{md['fitNameTrunk']}, absolute edge-meter diagnostics for A_prune",
            fontsize=13,
        )

    def _annotate_matrix(self, ax, mat):
        title_fs = self.plt.rcParams.get("axes.titlesize", 12)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(
                    j, i, f"{int(mat[i, j])}",
                    ha="center", va="center", fontsize=title_fs,
                    color="magenta",
                )

    def count_summary(self, metricD, md, figId="b"):
        """Raw presence and signed count bars."""
        figId = self.smart_append(figId)
        fig, axs = self.plt.subplots(1, 2, num=figId, facecolor="white", figsize=(13, 5))
        fig.subplots_adjust(wspace=0.28)
        labels = self._run_labels(metricD)
        xpos = np.arange(len(labels))

        ax = axs[0]
        bot = np.zeros(len(labels), dtype=np.float64)
        for key, label, color in (
            ("TP", "TP", "tab:green"),
            ("FN", "FN", "tab:orange"),
            ("FP", "FP", "tab:red"),
        ):
            vals = np.asarray(metricD[key], dtype=np.float64)
            ax.bar(xpos, vals, bottom=bot, label=label, color=color, alpha=0.75)
            bot += vals
        ax.set(title="edge is present (counts)", xlabel="fit tag", ylabel="off-diagonal pairs")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

        ax = axs[1]
        bot = np.zeros(len(labels), dtype=np.float64)
        for key, label, color in (
            ("signed_TP", "signed TP", "tab:green"),
            ("signed_FN", "signed FN", "tab:orange"),
            ("signed_FP", "signed FP", "tab:red"),
        ):
            vals = np.asarray(metricD[key], dtype=np.float64)
            ax.bar(xpos, vals, bottom=bot, label=label, color=color, alpha=0.75)
            bot += vals
        ax.set(title="edge sign is correct (counts)", xlabel="fit tag", ylabel="edges")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

        fig.suptitle(
            f"{md['fitNameTrunk']}, absolute edge-meter raw counts",
            fontsize=13,
        )

    def _resolve_tag_indices(self, metricD, tagIdxL):
        n_tag = len(metricD["fit_tag"])
        idx_l = []
        for idx in tagIdxL:
            idx = int(idx)
            if idx < 0:
                idx = n_tag + idx
            if idx < 0 or idx >= n_tag:
                raise IndexError(f"tagIdx {idx} outside 0..{n_tag - 1}")
            idx_l.append(idx)
        return idx_l

    def confusion_summary(self, metricD, md, tagIdxL=None, figId="c"):
        """Confusion matrices for selected fit tags."""
        if tagIdxL is None:
            tagIdxL = [-1]
        idx_l = self._resolve_tag_indices(metricD, tagIdxL)
        labels = self._run_labels(metricD)

        figId = self.smart_append(figId)
        fig, axs = self.plt.subplots(
            len(idx_l), 2, num=figId, facecolor="white",
            figsize=(12, 4.5 * len(idx_l)), squeeze=False,
        )
        fig.subplots_adjust(hspace=0.42, wspace=0.38)

        for row, tag_idx in enumerate(idx_l):
            tag = labels[tag_idx]

            ax = axs[row, 0]
            cm = np.asarray(metricD["confusion3"][tag_idx], dtype=np.int64)
            im = ax.imshow(cm, cmap="Blues")
            self._annotate_matrix(ax, cm)
            ax.set(title=f"Edge-label confusion, tag={tag}", xlabel="reco label", ylabel="true label")
            ax.set_xticks(np.arange(3))
            ax.set_yticks(np.arange(3))
            ax.set_xticklabels(["-", "0", "+"])
            ax.set_yticklabels(["-", "0", "+"])
            fig.colorbar(im, ax=ax, shrink=0.82)

            ax = axs[row, 1]
            scm = np.asarray(metricD["source_type_confusion"][tag_idx], dtype=np.int64)
            im = ax.imshow(scm, cmap="Greens")
            self._annotate_matrix(ax, scm)
            ax.set(title=f"Neuron type confusion, tag={tag}", xlabel="reco source", ylabel="true source")
            ax.set_xticks(np.arange(3))
            ax.set_yticks(np.arange(2))
            ax.set_xticklabels(["exc", "inh", "und"])
            ax.set_yticklabels(["exc", "inh"])
            fig.colorbar(im, ax=ax, shrink=0.82)

        fig.suptitle(
            f"{md['fitNameTrunk']}, selected confusion matrices",
            fontsize=13,
        )
