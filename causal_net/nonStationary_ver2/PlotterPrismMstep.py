#!/usr/bin/env python3
"""
Plotting utilities for prism M-step evaluation.
"""

from toolbox.PlotterBackbone import PlotterBackbone
import numpy as np
import matplotlib.colors as colors


class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self, args)

    def summary_prismMstep(self, fitD, md, figId=1):
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(12, 3.5))

        required = ["loss_epochs", "loss_dev", "loss_group", "loss_total", "rho_max"]

        epochs = fitD["loss_epochs"]
        dev = fitD["loss_dev"]
        group = fitD["loss_group"]
        total = fitD["loss_total"]
        rho = fitD["rho_max"]

        # ---- Loss curves
        ax = self.plt.subplot(1, 4, 1)
        ax.plot(epochs, dev, label="deviance")
        ax.plot(epochs, total, label="total")
        ax.set(title="Loss", xlabel="epoch", ylabel="value")
        ax.grid(True, alpha=0.4)
        ax.legend()

        trainMD = md.get("train", {})
        if trainMD:
            txt = (
                f"lr={trainMD.get('lr')}\n"
                f"lambda3={trainMD.get('lambda3')}\n"
                f"chunk={trainMD.get('chunk_size')}\n"
                f"eta_clip={trainMD.get('eta_clip')}"
            )
            ax.text(0.02, 0.95, txt, transform=ax.transAxes, va="top", fontsize=9)

        # ---- Group lasso
        ax = self.plt.subplot(1, 4, 2)
        ax.plot(epochs, group, color="tab:orange")
        ax.set(title="Group Lasso", xlabel="epoch", ylabel="value")
        ax.grid(True, alpha=0.4)

        # ---- Spectral radius
        ax = self.plt.subplot(1, 4, 3)
        ax.plot(epochs, rho, color="tab:green")
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
        ax.set(title="Spectral Radius", xlabel="epoch", ylabel="rho_max")
        ax.grid(True, alpha=0.4)

        # ---- Edges above minW
        ax = self.plt.subplot(1, 4, 4)
        minW = float(getattr(self.args, "minW", md.get("train", {}).get("minW", 0.0)))
        state_counts = np.asarray(fitD["nz_edges_minW_state"])
        edges_mean = state_counts.mean(axis=1)
        max_diff = state_counts.max(axis=1) - state_counts.min(axis=1)
        lo = edges_mean - max_diff
        hi = edges_mean + max_diff
        ax.fill_between(epochs, lo, hi, color="gold", alpha=0.4)
        ax.plot(epochs, edges_mean, color="tab:purple", linewidth=1)
        ax.set(title=f"Edges/State |A|>{minW:g}", xlabel="epoch", ylabel="count")
        ax.grid(True, alpha=0.4)

        minW = float(getattr(self.args, "minW", md.get("train", {}).get("minW", 0.0)))
        fig.suptitle(f"Prism M-step, minW={minW:g}: {md.get('short_name','')}", fontsize=12)

    def state_corr_prismMstep(self, fitD, md, figId=2):
        self.state_corr_truth_prismMstep(fitD, md, type="fit", figId=figId)

    def state_ABcorr_prismMstep(self, fitD, md, type="truth", figId=5):
        figId = self.smart_append(figId)
        minW = float(getattr(self.args, "minW", 0.15))
        divideB = float(getattr(self.args, "divideB", 2.5))

        if type == "fit":
            A = fitD.get("fA") if fitD is not None else None
            B = fitD.get("fB") if fitD is not None else None
            title = "State correlations"
            a_lab = "A_hat"
            b_lab = "B_hat"
            a_color = "b"
        elif type == "truth":
            A = md.get("A_true")
            B = md.get("B_true")
            title = "Truth state correlations"
            a_lab = "A_true"
            b_lab = "B_true"
            a_color = "g"
        else:
            fig = self.plt.figure(figId, facecolor='white', figsize=(10, 3))
            ax = self.plt.subplot(1, 1, 1)
            ax.axis("off")
            ax.text(0.05, 0.6, f"Unknown state_corr type: {type}", fontsize=12)
            fig.suptitle(f"State correlations: {md.get('short_name','')}", fontsize=12)
            return

        A = np.asarray(A)
        B = np.asarray(B)
        if A.ndim == 2:
            A = A[None, :, :]
        if B.ndim == 1:
            B = B[None, :]

        n_states = A.shape[0]
        pairs = [(0, 1), (1, 2), (0, 2)]

        fig = self.plt.figure(figId, facecolor='white', figsize=(10, 6))

        for col, (i, j) in enumerate(pairs):
            ax = self.plt.subplot(2, 3, 1 + col)
            if i >= n_states or j >= n_states:
                ax.axis("off")
                ax.text(0.05, 0.6, f"missing states {i}-{j}", fontsize=11)
                continue

            x = A[i].ravel()
            y = A[j].ravel()
            plot_corr_A_regions(ax, x, y, minW, f"{a_lab} s{i}-s{j}", f"s{i}", f"s{j}", s=2, alpha=0.3, color=a_color)

            ax = self.plt.subplot(2, 3, 4 + col)
            x = B[i]
            y = B[j]
            plot_corr_B_divisor(ax, x, y, divideB, f"{b_lab} s{i}-s{j}", f"s{i}", f"s{j}", s=8, alpha=0.6, color=a_color)

        fig.suptitle(f"{title}: {md.get('short_name','')}", fontsize=12)

    def eval_ABcorr_prismMstep(self, fitD, md, figId=2):
        A_hat = fitD.get("fA")
        B_hat = fitD.get("fB")
        A_true = md.get("A_true")
        B_true = md.get("B_true")
        figId = self.smart_append(figId)
        minW = float(getattr(self.args, "minW", 0.15))
        divideB = float(getattr(self.args, "divideB", 2.5))

        if A_hat is None or B_hat is None or A_true is None or B_true is None:
            fig = self.plt.figure(figId, facecolor='white', figsize=(10, 3))
            ax = self.plt.subplot(1, 1, 1)
            ax.axis("off")
            ax.text(
                0.05,
                0.6,
                "Missing A_true/B_true or fA/fB for 2D correlations.\n"
                "Check provenance and truthDale inputs.",
                fontsize=12,
            )
            fig.suptitle(f"2D correlations: {md.get('short_name','')}", fontsize=12)
            return

        A_hat = np.asarray(A_hat)
        B_hat = np.asarray(B_hat)
        A_true = np.asarray(A_true)
        B_true = np.asarray(B_true)

        if A_true.ndim == 2:
            A_true = np.repeat(A_true[None, :, :], A_hat.shape[0], axis=0)
        if B_true.ndim == 1:
            B_true = np.repeat(B_true[None, :], B_hat.shape[0], axis=0)

        n_states = min(A_hat.shape[0], A_true.shape[0], B_hat.shape[0], B_true.shape[0])
        ncol = max(1, n_states)
        fig_w = max(9.0, 3.0 * ncol)
        fig = self.plt.figure(figId, facecolor='white', figsize=(fig_w, 6))

        for m in range(n_states):
            ax = self.plt.subplot(2, ncol, m + 1)
            x = A_true[m].ravel()
            y = A_hat[m].ravel()
            plot_corr_A_regions(ax, x, y, minW, f"A corr, state {m}", "A_true", "A_hat", s=6, alpha=0.4)

            ax = self.plt.subplot(2, ncol, ncol + m + 1)
            plot_corr_B_divisor(ax, B_true[m], B_hat[m], divideB, f"B corr, state {m}", "B_true", "B_hat", s=8, alpha=0.5)

        minW = float(getattr(self.args, "minW", md.get("train", {}).get("minW", 0.0)))
        fig.suptitle(f"2D correlations, minW={minW:g}: {md.get('short_name','')}", fontsize=12)

    def edge_state_prismMstep(self, fitD, md, figId=6):
        A_hat = fitD.get("fA")
        E_true = md.get("E_true")
        figId = self.smart_append(figId)
        minW = float(getattr(self.args, "minW", 0.15))

        A_hat = np.asarray(A_hat)
        E_true = np.asarray(E_true)
        if E_true.ndim == 3:
            E_true = E_true[0]

        n_states = A_hat.shape[0]
        N = A_hat.shape[1]
        off_diag_mask = ~np.eye(N, dtype=bool)
        fig_h = max(2.8, 2.8 * n_states)
        fig = self.plt.figure(figId, facecolor='white', figsize=(16, fig_h))

        for m in range(n_states):
            E_hat = np.abs(A_hat[m]) > minW
            E_t = E_true.astype(bool)
            E_hat = E_hat & off_diag_mask
            E_t = E_t & off_diag_mask

            TP = E_t & E_hat
            FP = (~E_t) & E_hat
            FN = E_t & (~E_hat)
            TN = (~E_t) & (~E_hat)

            tp = int(TP.sum())
            fp = int(FP.sum())
            fn = int(FN.sum())
            tn = int(TN.sum())
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            acc = (tp + tn) / max(1, tp + tn + fp + fn)

            ax = self.plt.subplot(n_states, 4, m * 4 + 1)
            im = ax.imshow(E_t, cmap='Greys', vmin=0, vmax=1, origin='lower')
            ax.set(title=f"E_true, state {m} (n={int(E_t.sum())})")
            self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.plot([0, N], [0, N], '--', lw=0.8, color='magenta')
            ax.set_aspect(1.0)
            ax.grid(True, alpha=0.4)
            ax.set_xlim(-0.5, N + 0.5)
            ax.set_ylim(-0.5, N + 0.5)
            ax.set(xlabel='presyn. neuron index (output)', ylabel='postsyn. neuron index (input)')

            ax = self.plt.subplot(n_states, 4, m * 4 + 2)
            im = ax.imshow(E_hat, cmap='Greys', vmin=0, vmax=1, origin='lower')
            ax.set(title=f"A_hat edges, state {m} (n={int(E_hat.sum())})")
            self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.plot([0, N], [0, N], '--', lw=0.8, color='magenta')
            ax.set_aspect(1.0)
            ax.grid(True, alpha=0.4)
            ax.set_xlim(-0.5, N + 0.5)
            ax.set_ylim(-0.5, N + 0.5)
            ax.set(xlabel='presyn. neuron index (output)', ylabel='postsyn. neuron index (input)')

            conf_map = np.zeros_like(E_t, dtype=int)
            conf_map[FN] = 1
            conf_map[FP] = 2
            conf_map[TP] = 3
            cmap_conf = colors.ListedColormap(['white', 'magenta', 'red', 'green'])
            bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
            norm = colors.BoundaryNorm(bounds, cmap_conf.N)
            ax = self.plt.subplot(n_states, 4, m * 4 + 3)
            im = ax.imshow(conf_map, cmap=cmap_conf, norm=norm, origin='lower')
            ax.set(title='Confusion map')
            cbar = self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks([0, 1, 2, 3])
            cbar.set_ticklabels(['TN', 'FN', 'FP', 'TP'])
            ax.plot([0, N], [0, N], '--', lw=0.8, color='k')
            ax.set_aspect(1.0)
            ax.grid(True, alpha=0.4)
            ax.set_xlim(-0.5, N + 0.5)
            ax.set_ylim(-0.5, N + 0.5)
            ax.set(xlabel='presyn. neuron index', ylabel='postsyn. neuron index')

            ax = self.plt.subplot(n_states, 4, m * 4 + 4)
            bar_names = ['TP', 'FP', 'FN']
            vals = [tp, fp, fn]
            ax.bar(bar_names, vals, color=['green', 'red', 'magenta'])
            ax.set_ylabel('count')
            y_pos = 0.25 * max(1, max(vals))
            for name, val in zip(bar_names, vals):
                ax.text(name, y_pos, f"{val}", ha='center', va='center', fontsize=10)
            txt = f'precision={precision:.3f}\nrecall={recall:.3f}\nf1={f1:.3f}\nacc={acc:.3f}'
            ax.text(0.6, 0.60, txt, transform=ax.transAxes)
            ax.set_title('stats')
            ax.grid(axis='y', alpha=0.4)

        fig.subplots_adjust(bottom=0.12)
        fig.suptitle(
            f"Edge detection vs truth, minW={minW} (off diagonal only): {md.get('short_name','')}",
            fontsize=14,
        )


def add_x45_lins(ax, only45=False):
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, '--', color='k', linewidth=0.8)
    if only45:
        return
    ax.axvline(0, linestyle='--', color='k', linewidth=1)
    ax.axhline(0, linestyle='--', color='k', linewidth=1)


def corrcoef_safe(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def plot_corr_A_regions(ax, xV, yV, minW, title, xlab, ylab, s=6, alpha=0.5, color=None):
    x = np.asarray(xV)
    y = np.asarray(yV)
    ax.scatter(x, y, s=s, alpha=alpha, color=color)
    add_x45_lins(ax, only45=True)
    ax.axvline(-minW, color="red", linestyle="--", linewidth=1)
    ax.axvline(minW, color="red", linestyle="--", linewidth=1)
    ax.set(title=title, xlabel=xlab, ylabel=ylab)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.4)

    mask_left = x < -minW
    mask_mid = (x >= -minW) & (x <= minW)
    mask_right = x > minW
    r_left = corrcoef_safe(x[mask_left], y[mask_left])
    r_mid = corrcoef_safe(x[mask_mid], y[mask_mid])
    r_right = corrcoef_safe(x[mask_right], y[mask_right])

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])
    posx = [0.2, 0.50, 0.65]
    posy = [0.05, 0.40, 0.60]
    ax.text(posx[0], posy[0], f"rL={r_left:.3f}", transform=ax.transAxes)
    ax.text(posx[1], posy[1], f"rM={r_mid:.3f}", transform=ax.transAxes)
    ax.text(posx[2], posy[2], f"rR={r_right:.3f}", transform=ax.transAxes)
    

def plot_corr_B_divisor(ax, xV, yV, divideB, title, xlab, ylab, s=6, alpha=0.5, color=None):
    x = np.asarray(xV)
    y = np.asarray(yV)
    ax.scatter(x, y, s=s, alpha=alpha, color=color)
    add_x45_lins(ax, only45=True)
    ax.axvline(divideB, color="red", linestyle="--", linewidth=1)
    ax.set(title=title, xlabel=xlab, ylabel=ylab)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.4)

    mask_left = x < divideB
    mask_right = x >= divideB
    r_left = corrcoef_safe(x[mask_left], y[mask_left])
    r_right = corrcoef_safe(x[mask_right], y[mask_right])
    txt = f"rL={r_left:.3f}"
    ax.text(0.05, 0.45, txt, transform=ax.transAxes, va="top")
    txt = f"rR={r_right:.3f}"
    ax.text(0.65, 0.55, txt, transform=ax.transAxes, va="top")
