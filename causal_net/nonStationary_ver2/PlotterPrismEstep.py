#!/usr/bin/env python3
"""
Plotting utilities for prism E-step evaluation.
"""

from toolbox.PlotterBackbone import PlotterBackbone
import numpy as np


class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self, args)

    def summary_prismEstep(self, fitD, md, figId=1, time_bin_merge=20):
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(12, 6.5))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.25)

        loss_epoch = np.asarray(fitD.get("loss_epoch", []))
        loss_nll_epoch = np.asarray(fitD.get("loss_nll_epoch", []))
        loss_l2_epoch = np.asarray(fitD.get("loss_l2_epoch", []))
        loss_time = np.asarray(fitD.get("loss_time", []))
        c_hat = fitD.get("c_hat")
        S_hat = fitD.get("S_hat")
        S_hat_CL = fitD.get("S_hat_CL")

        epochs = np.arange(1, len(loss_epoch) + 1)

        # ---- Loss vs epoch
        ax = fig.add_subplot(gs[0, 0])
        if len(loss_nll_epoch) > 0 and len(loss_l2_epoch) > 0:
            ax.plot(epochs, loss_nll_epoch, label="nll")
            ax.plot(epochs, loss_l2_epoch, label="l2")
            ax.set(title="Loss (epoch)", xlabel="epoch", ylabel="value")
            ax.grid(True, alpha=0.4)
            ax.legend()
            ax.set_yscale('log')
        else:
            ax.axis("off")
            ax.text(0.05, 0.6, "missing loss_nll_epoch/loss_l2_epoch", fontsize=11)

        trainMD = md.get("train", {})
        if trainMD:
            txt = (
                f"lr={trainMD.get('lr')}\n"
                f"lambda2={trainMD.get('lambda2')}\n"
                f"pgd_iter={trainMD.get('pgd_iter')}\n"
                f"chunk={trainMD.get('chunk_size')}"
            )
            ax.text(0.02, 0.95, txt, transform=ax.transAxes, va="top", fontsize=9)

        # ---- Loss vs time
        ax = fig.add_subplot(gs[1, :])
        if len(loss_time) > 0:
            merge = int(max(1, time_bin_merge))
            if merge > 1 and len(loss_time) >= merge:
                n = len(loss_time) // merge
                loss_time = loss_time[: n * merge].reshape(n, merge).mean(axis=1)
            dt = trainMD.get("time_step_sec", None)
            t0_bin = None
            if isinstance(trainMD.get("time_range_bins", None), (list, tuple)):
                t0_bin = trainMD["time_range_bins"][0]
            if dt is not None and t0_bin is not None:
                t = (t0_bin + np.arange(len(loss_time)) * merge) * float(dt)
                ax.plot(t, loss_time, color="tab:blue", linewidth=0.7)
                ax.set(xlabel="time (s)")
            else:
                ax.plot(loss_time, color="tab:blue", linewidth=0.7)
                ax.set(xlabel="time bin")
            ax.set(title="Loss (time)", ylabel="value")
            ax.grid(True, alpha=0.4)
        else:
            ax.axis("off")
            ax.text(0.05, 0.6, "missing loss_time", fontsize=11)

        # ---- State occupancy (mean c_hat)
        ax = fig.add_subplot(gs[0, 1])
        if c_hat is not None:
            c_hat = np.asarray(c_hat)
            occ = c_hat.mean(axis=0)
            ax.bar(np.arange(len(occ)), occ, color="tab:green", alpha=0.7)
            ax.set(title="Mean occupancy", xlabel="state", ylabel="mean c")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis("off")
            ax.text(0.05, 0.6, "missing c_hat", fontsize=11)

        # ---- Confidence histogram
        ax = fig.add_subplot(gs[0, 2])
        if S_hat_CL is not None:
            cl = np.asarray(S_hat_CL)
            ax.hist(cl, bins=30, color="tab:purple", alpha=0.7)
            ax.set(title="Confidence", xlabel="S_hat_CL", ylabel="count")
            ax.grid(True, alpha=0.3)
            ax.axvline(np.mean(cl), color="k", linestyle="--", linewidth=1)
        else:
            ax.axis("off")
            ax.text(0.05, 0.6, "missing S_hat_CL", fontsize=11)

        fig.suptitle(f"Prism E-step: {md.get('short_name','')}", fontsize=12)
