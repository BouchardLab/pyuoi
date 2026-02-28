#!/usr/bin/env python3
"""
Plotting utilities for prism E-step evaluation.
"""

from toolbox.PlotterBackbone import PlotterBackbone
import numpy as np
from matplotlib.ticker import MaxNLocator


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
            out[i] = np.bincount(vals).argmax()
        return out

    def plot_loss_time(self, ax, fitD, md, trainMD, time_bin_merge=20):
        loss_nll_time = np.asarray(fitD["loss_nll_time"])
        loss_l2_time = np.asarray(fitD["loss_l2_time"])

        merge = int(max(1, time_bin_merge))
        if merge > 1 and len(loss_nll_time) >= merge:
            n = len(loss_nll_time) // merge
            loss_nll_time = loss_nll_time[: n * merge].reshape(n, merge).mean(axis=1)
            loss_l2_time = loss_l2_time[: n * merge].reshape(n, merge).mean(axis=1)

        dt = trainMD["time_step_sec"]
        time_range_bins = trainMD["time_range_bins"]
        t0_bin = time_range_bins[0]
        t = (t0_bin + np.arange(len(loss_nll_time)) * merge) * float(dt)

        ax.plot(t, loss_nll_time, color="tab:blue", linewidth=0.8, label="nll")
        ax.plot(t, loss_l2_time, color="tab:orange", linewidth=0.8, label="l2")
        ax.set(title="Loss (time)", xlabel="time (s)", ylabel="value")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=8)
        ax.set_yscale('log')

        s_true = np.asarray(md["S_true"])
        s_true = s_true[t0_bin + 1 : t0_bin + 1 + len(fitD["loss_nll_time"])]
        s_true_rb = self._rebin_mode_1d(s_true, merge)

        ax2 = ax.twinx()
        ax2.plot(t, s_true_rb, color="k", linewidth=0.8, alpha=0.6, label="S_true")
        ax2.set_ylabel("state")
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), fontsize=8)

    def plot_ll_gap(self, ax, md, trainMD, time_bin_merge=20):
        ll_gap = np.asarray(md["eval_Estep"]["ll_gap"])
        merge = int(max(1, time_bin_merge))
        if merge > 1 and len(ll_gap) >= merge:
            n = len(ll_gap) // merge
            ll_gap = ll_gap[: n * merge].reshape(n, merge).mean(axis=1)

        dt = trainMD["time_step_sec"]
        t0_bin = trainMD["time_range_bins"][0]
        t = (t0_bin + np.arange(len(ll_gap)) * merge) * float(dt)

        ax.plot(t, ll_gap, color="tab:blue", linewidth=0.8, label="LL gap")
        ax.set(title="LL gap (best - 2nd)", xlabel="time (s)", ylabel="value")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=8)

    def summary_prismEstep(self, fitD, md, figId=1, time_bin_merge=20):
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(10, 5.5))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.6, wspace=0.25)

        loss_epoch = np.asarray(fitD["loss_epoch"])
        loss_nll_epoch = np.asarray(fitD["loss_nll_epoch"])
        loss_l2_epoch = np.asarray(fitD["loss_l2_epoch"])
        loss_time = np.asarray(fitD["loss_time"])
        c_hat = fitD["c_hat"]
        S_hat = fitD["S_hat"]
        S_hat_CL = fitD["S_hat_CL"]

        epochs = np.arange(1, len(loss_epoch) + 1)

        # ---- Loss vs epoch
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(epochs, loss_nll_epoch, label="nll")
        ax.plot(epochs, loss_l2_epoch, label="l2")
        ax.set(title="Loss (epoch)", xlabel="epoch", ylabel="value")
        ax.grid(True, alpha=0.4)
        ax.legend()
        ax.set_yscale('log')

        trainMD = md["train"]
        acc = md["eval_Estep"]["acc"]
        txt = (
            f"lr={trainMD['lr']}\n"
            f"lambda2={trainMD['lambda2']}\n"
            f"pgd_iter={trainMD['pgd_iter']}\n"
            f"chunk={trainMD['chunk_size']}"
        )
        ax.text(0.02, 0.95, txt, transform=ax.transAxes, va="top", fontsize=9)

        # ---- Loss vs time
        ax = fig.add_subplot(gs[1, :])
        self.plot_loss_time(ax, fitD, md, trainMD, time_bin_merge=time_bin_merge)

        # ---- State occupancy (mean c_hat)
        ax = fig.add_subplot(gs[0, 1])
        c_hat = np.asarray(c_hat)
        occ = c_hat.mean(axis=0)
        ax.bar(np.arange(len(occ)), occ, color="tab:green", alpha=0.7)
        ax.set(title="Mean occupancy", xlabel="state", ylabel="mean c")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

        # ---- Confidence histogram
        ax = fig.add_subplot(gs[0, 2])
        cl = np.asarray(S_hat_CL)
        ax.hist(cl, bins=30, color="tab:purple", alpha=0.7)
        ax.set(title=f"Confidence (acc={acc:.3f})", xlabel="S_hat_CL", ylabel="count")
        ax.grid(True, alpha=0.3)
        ax.axvline(np.mean(cl), color="k", linestyle="--", linewidth=1)

        fig.suptitle(f"Prism E-step: {md['short_name']}", fontsize=12)

    def state_seq_prismEstep(self, fitD, md, figId=2, time_reb=20):
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(12, 8.5))
        gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 0.5, 0.5], hspace=0.9)

        S_hat = fitD["S_hat"]
        S_hat_CL = fitD["S_hat_CL"]
        C_hat = fitD["c_hat"]
        S_true = md["S_true"]
        C_true = md["C_true"]

        trainMD = md["train"]
        time_range_bins = trainMD["time_range_bins"]
        t0_bin, t1_bin = time_range_bins

        S_true = np.asarray(S_true)[t0_bin : t1_bin + 1]
        C_true = np.asarray(C_true)[t0_bin : t1_bin + 1]

        S_hat = np.asarray(S_hat)
        S_hat_CL = np.asarray(S_hat_CL)
        C_hat = np.asarray(C_hat)

        S_hat_rb = self._rebin_mode_1d(S_hat, time_reb)
        C_hat_rb = self._rebin_2d(C_hat, time_reb)
        S_true_rb = self._rebin_mode_1d(S_true, time_reb)
        C_true_rb = self._rebin_2d(C_true, time_reb)
        CL_hat_rb = self._rebin_1d(S_hat_CL, time_reb)

        dt = trainMD["time_step_sec"]
        t = (t0_bin + np.arange(len(S_hat_rb)) * max(1, int(time_reb))) * float(dt)

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(t, S_hat_rb, color="k", linewidth=1.0, label="S_hat")
        n_states = C_hat_rb.shape[1]
        lo = np.clip(S_hat_rb - CL_hat_rb, 0.0, float(n_states - 1))
        hi = np.clip(S_hat_rb + CL_hat_rb, 0.0, float(n_states - 1))
        ax.fill_between(t, lo, hi, color="gray", alpha=0.3, label="S_hat_CL")
        acc = md["eval_Estep"]["acc"]
        ax.set(title=f"Fit (acc={acc:.3f})", xlabel="time (s)", ylabel="state")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
       
        ax2 = ax.twinx()
        for m in range(C_hat_rb.shape[1]):
            ax2.plot(t, C_hat_rb[:, m], linewidth=0.8, alpha=0.8, label=f"C_hat[{m}]")
        ax2.set_ylabel("C_hat")

        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=8)
        ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=4, fontsize=8)

        ax = fig.add_subplot(gs[1, 0])
        ax.plot(t, S_true_rb, color="k", linewidth=1.0, label="S_true")
        ax.set(title="Truth           ", xlabel="time (s)", ylabel="state")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        for m in range(C_true_rb.shape[1]):
            ax2.plot(t, C_true_rb[:, m], linewidth=0.8, alpha=0.8, label=f"C_true[{m}]")
        ax2.set_ylabel("C_true")
        
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), fontsize=8)
        ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=4, fontsize=8)

        ax = fig.add_subplot(gs[2, 0])
        self.plot_loss_time(ax, fitD, md, trainMD, time_bin_merge=time_reb)

        ax = fig.add_subplot(gs[3, 0])
        self.plot_ll_gap(ax, md, trainMD, time_bin_merge=time_reb)

        fig.suptitle(
            f"Dataset: {md['short_name']} | $\\lambda_2$={trainMD['lambda2']}, "
            f"lr={trainMD['lr']}, pgd_iter={trainMD['pgd_iter']}, decode_dwell={trainMD['decode_dwell_sec']}s",
            fontsize=12,
        )
