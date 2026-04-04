#!/usr/bin/env python3
"""
Plotting utilities for topology-correlation inspection.
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from matplotlib.lines import Line2D
import numpy as np

from toolbox.PlotterBackbone import PlotterBackbone


METRIC_PLOT_ORDER = (
    ("assortativity", "r_assort", "Directed assortativity"),
    ("jaccard_index", "jaccard", "Mean Jaccard"),
    ("cycle_decay", "cycle_decay", "Cycle decay"),
    ("homology_k1", "hom_k1", "Homology k1"),
)


class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self, args)

    def metrics_inspector(self, harvestD, nameTemplate, figId=1):
        figId = self.smart_append(figId)
        fig, axL = self.plt.subplots(1, 4, num=figId, facecolor="white", figsize=(18, 4.6))

        colorL = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray")
        markerL = ("o", "s", "^", "D", "v", "P", "X", "<")

        placement_LL = sorted({int(placement_L) for per_metric in harvestD.values() for placement_L in per_metric.keys()})
        styleD = {}
        for idx, placement_L in enumerate(placement_LL):
            styleD[placement_L] = {
                "color": colorL[idx % len(colorL)],
                "marker": markerL[idx % len(markerL)],
            }

        for ax, (metric_name, y_label, title_txt) in zip(axL, METRIC_PLOT_ORDER):
            metricD = harvestD.get(metric_name, {})
            for placement_L in sorted(metricD.keys()):
                recL = metricD[placement_L]
                if not recL:
                    continue
                xV = np.asarray([rec[0] for rec in recL], dtype=float)
                yV = np.asarray([rec[1] for rec in recL], dtype=float)
                sty = styleD[placement_L]
                ax.plot(
                    xV,
                    yV,
                    linestyle="none",
                    marker=sty["marker"],
                    color=sty["color"],
                    markersize=6.5,
                    markeredgewidth=0.7,
                    markeredgecolor="black",
                    alpha=0.9,
                )

            ax.set_title(title_txt)
            ax.set_xlabel("placement_ker_delta")
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.35)

            if metricD:
                finite_y = []
                for recL in metricD.values():
                    for _, y_val, _ in recL:
                        if np.isfinite(y_val):
                            finite_y.append(y_val)
                if finite_y:
                    y_min = min(finite_y)
                    y_max = max(finite_y)
                    if y_min == y_max:
                        pad = 0.05 if y_min == 0 else 0.08 * abs(y_min)
                        ax.set_ylim(y_min - pad, y_max + pad)

        leg_handles = [
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker=styleD[placement_L]["marker"],
                color=styleD[placement_L]["color"],
                markeredgecolor="black",
                markeredgewidth=0.7,
                markersize=7,
                label="L=%d" % placement_L,
            )
            for placement_L in placement_LL
        ]
        if leg_handles:
            fig.legend(handles=leg_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=min(6, len(leg_handles)))

        fig.text(0.5, 0.995, "nameTemplate: %s" % nameTemplate, ha="center", va="top", fontsize=16)
        fig.subplots_adjust(top=0.78, wspace=0.32)
