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

    def summary_prismMstep(self, fitD, md, figId=1):
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(12, 5.))

        trainMD = md["train"]
        short_name = md["short_name"]
        total = np.asarray(fitD["loss_epoch"])
        l1 = np.asarray(fitD["loss_l1_epoch"])
        sparsity = np.asarray(fitD["sparsity_epoch"])
        rho = np.asarray(fitD["rho_epoch"])
        A_fit = np.asarray(fitD["A_hat"])
        Freq = np.asarray(fitD["single_rates"])
        loss_state = np.asarray(fitD["loss_state_total_epoch"])

        assert A_fit.ndim == 2, f"A_hat must be 2D, got shape={A_fit.shape}"
        assert Freq.ndim == 1, f"single_rates must be 1D, got shape={Freq.shape}"
        assert loss_state.ndim == 2, f"loss_state_total_epoch must be 2D, got shape={loss_state.shape}"
        assert loss_state.shape[0] == total.shape[0], "loss_state_total_epoch first dim must equal num epochs"

        n_epochs = len(total)
        epoch0 = 10 if n_epochs > 10 else 0
        epochs = np.arange(1 + epoch0, n_epochs + 1, dtype=np.int32)

        # ---- Row1 Col1: Loss + L1 (dual y-axis)
        ax = self.plt.subplot(2, 4, 1)
        ax.plot(epochs, total[epoch0:], label='total', color='blue', linestyle='-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.grid(True, alpha=0.3)
        add_delay_markers(ax, trainMD)

        ax2 = ax.twinx()
        ax2.plot(epochs, l1[epoch0:], label='L1 loss', color='red', linestyle='--')
        ax2.set_ylabel('L1 loss', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        n_samp = int(trainMD["num_samples_used"])
        ax.legend(lines1 + lines2, labels1 + labels2, title=f"{n_samp//1000}k samp")
        ax.set_title(f"Loss: {short_name}")

        # ---- Row1 Col2: A off-diagonal histogram
        ax = self.plt.subplot(2, 4, 2)
        Nn = A_fit.shape[0]
        diag_mask = np.eye(Nn, dtype=bool)
        A_off = A_fit[~diag_mask]
        valid_mask = np.abs(A_off) > 1e-10
        A_off_clean = A_off[valid_mask]
        n_edges = int(A_off_clean.size)
        ax.hist(A_off_clean, bins=100, color='g', alpha=0.8)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.4)
        ax.set(ylabel='edges', xlabel='edge value', title='A off-diagonal')
        acc_frac = n_edges / float(Nn * (Nn - 1))
        txt = f"edge sel meth: None\nacc frac={acc_frac:.2f}"
        ax.text(0.05, 0.75, txt, transform=ax.transAxes, fontsize=10)

        # ---- Row1 Col3: A sparsity vs epoch
        ax = self.plt.subplot(2, 4, 3)
        ax.plot(np.arange(1, len(sparsity) + 1), sparsity, color='tab:purple', linewidth=1.5)
        add_delay_markers(ax, trainMD)
        ax.set_ylim(0.0, 1.02)
        ax.set(ylabel='fraction', xlabel='epoch', title='A sparsity (off-diag)')
        ax.grid(True, alpha=0.4)

        # ---- Row1 Col4: Spectral radius vs epoch
        ax = self.plt.subplot(2, 4, 4)
        ax.plot(np.arange(1, len(rho) + 1), rho, color='tab:red', linewidth=1.5)
        add_delay_markers(ax, trainMD)
        ax.axhline(float(trainMD["rho_max"]), color='k', linestyle='--', linewidth=1.0)
        ax.set(ylabel='radius', xlabel='epoch', title='Spectral radius(A)')
        ax.grid(True, alpha=0.4)

        # ---- Row2 Col3: per-state total loss vs epoch
        ax = self.plt.subplot(2, 4, 5)
        epoch_all = np.arange(1, loss_state.shape[0] + 1, dtype=np.int32)
        for m in range(loss_state.shape[1]):
            ax.plot(epoch_all, loss_state[:, m], linewidth=1.2, label=f'state {m}')
        add_delay_markers(ax, trainMD)
        ax.set(ylabel='loss', xlabel='epoch', title='Per-state total loss')
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8, ncol=2)

 
        # ---- Row2 Col1: histogram of rates
        ax = self.plt.subplot(2, 4, 6)
        ax.hist(Freq, bins=20)
        ax.set(ylabel='num neurons', xlabel='Firing rate (Hz)', title=f'Single rates, {Nn} neurons')
        ax.grid(True, alpha=0.4)
        ax.set_xlim(0,)
        median_val = float(np.median(Freq))
        ax.text(median_val, ax.get_ylim()[1] * 0.8, f'median rate: {median_val:.2f} Hz',
                color='red', ha='left', va='bottom', fontsize=10)
        ax.axvline(median_val, color='red', linestyle='--', linewidth=1)

        # ---- Row2 Col2: 2D histogram edge value vs log10(rate)
        ax = self.plt.subplot(2, 4, 7)
        i_indices, j_indices = np.where(~diag_mask)
        Freq_expanded = Freq[i_indices]
        Freq_expanded_clean = Freq_expanded[valid_mask]
        h = ax.hist2d(A_off_clean, np.log10(Freq_expanded_clean), bins=30, cmap='viridis', norm=colors.LogNorm())
        fig.colorbar(h[3], ax=ax)
        ax.set(ylabel='log10( Firing rate/ Hz )', xlabel='edge value',
               title=f'accept {n_edges} of {Nn*(Nn-1)} edges')
        ax.grid(True, alpha=0.4)

          # ---- Row2 Col4: intentionally empty
        ax = self.plt.subplot(2, 4, 8)
        ax.axis("off")

    def eval_Amatrix_prismMstep(self, fitD, md, figId=2):
        """One-row, four-panel A-matrix evaluation: E_true | A_hat edges | confusion map | stats bar."""
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(12, 3.8))

        A_hat  = np.asarray(fitD["A_hat"])          # (N, N)
        if A_hat.ndim == 3:
            A_hat = A_hat[0]
        N = A_hat.shape[0]

        edge = md.get("edge_eval_main", None)
        if edge is None:
            raise ValueError("Missing md['edge_eval_main']; compute edge metrics in prism_Mstep_eval.py")

        minW = float(edge["minW"])
        E_hat = np.asarray(edge["E_hat"]).astype(bool)
        E_t = np.asarray(edge["E_true_offdiag"]).astype(bool)
        conf_map = np.asarray(edge["conf_map"])

        tp = int(edge["tp"]);  fp = int(edge["fp"])
        fn = int(edge["fn"]);  tn = int(edge["tn"])
        precision = float(edge["precision"])
        recall = float(edge["recall"])
        f1 = float(edge["f1"])
        acc = float(edge["acc"])

        kw = dict(origin='lower', interpolation='nearest')

        # ---- A_true (signed weights, bwr, E/I boundary)
        A_true  = np.asarray(md["A_true"])
        dmd     = md.get("dale_conf", {})
        numExc  = int(dmd.get("num_excite", N))
        vmin_t, vmax_t = A_true.min(), A_true.max()
        norm_t  = colors.TwoSlopeNorm(vmin=vmin_t, vcenter=0.0, vmax=vmax_t)
        n_exc_edges = int(E_t[:, :numExc].sum())
        n_inh_edges = int(E_t[:, numExc:].sum())

        ax = self.plt.subplot(1, 4, 1)
        im = ax.imshow(A_true, cmap='bwr', norm=norm_t, aspect=1., **kw)
        self.plt.colorbar(im, ax=ax, extend='both', shrink=0.7)
        ax.axhline(numExc - 0.5, color='k', ls='--')
        ax.axvline(numExc - 0.5, color='k', ls='--')
        ax.plot([0, N], [0, N], '--', lw=0.8, color='magenta')
        ax.set(title=f"True Dale, N{N}, nEdges={n_exc_edges}+{n_inh_edges}",
               xlabel='presyn. neuron index (output)',
               ylabel='postsyn. neuron index (input)')

        # ---- A_hat edges
        ax = self.plt.subplot(1, 4, 2)
        ax.imshow(E_hat.astype(float), cmap='Greys', vmin=0, vmax=1, **kw)
        ax.set(title=f"A_hat edges  (n={int(E_hat.sum())},  minW={minW:g})",
               xlabel='presyn. neuron (output)', ylabel='postsyn. neuron (input)')
        ax.plot([0, N-1], [0, N-1], '--', lw=0.8, color='magenta')

        # ---- Confusion map
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

        # ---- Stats bar
        ax = self.plt.subplot(1, 4, 4)
        vals = [tp, fp, fn]
        bars = ax.bar(['TP', 'FP', 'FN'], vals, color=['green', 'red', 'magenta'])
        ax.set(title='stats', ylabel='count')
        ax.grid(axis='y', alpha=0.4)
        y_pos = 0.25 * max(1, max(vals))
        for name, val in zip(['TP', 'FP', 'FN'], vals):
            ax.text(name, y_pos, str(val), ha='center', va='center', fontsize=10)
        txt = (f"precision={precision:.3f}\nrecall={recall:.3f}\n"
               f"f1={f1:.3f}\nacc={acc:.3f}")
        ax.text(0.55, 0.60, txt, transform=ax.transAxes, fontsize=9)

        fig.suptitle(
            f"A-matrix edge recovery, minW={minW:g} (off-diag only): {md['short_name']}",
            fontsize=12)
        fig.tight_layout()

    def state_seq_prismMstep(self, estepD, md, figId=2, time_reb=20):
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(12, 6))

        assert estepD is not None, "Missing E-step fit (S_hat/C_hat) for state sequence plot."

        S_hat = estepD["S_hat"]
        S_hat_CL = estepD["S_hat_CL"]
        C_hat = estepD["c_hat"]
        S_true = md["S_true"]
        C_true = md["C_true"]

        trainMD = md["estep_train"]
        time_range_bins = trainMD["time_range_bins"]
        t0_bin, t1_bin = time_range_bins

        S_true = np.asarray(S_true)
        C_true = np.asarray(C_true)
        S_true = S_true[t0_bin : t1_bin + 1]
        C_true = C_true[t0_bin : t1_bin + 1]

        S_hat = np.asarray(S_hat)
        S_hat_CL = np.asarray(S_hat_CL)
        C_hat = np.asarray(C_hat)

        # Rebin
        S_hat_rb = self._rebin_1d(S_hat, time_reb)
        C_hat_rb = self._rebin_2d(C_hat, time_reb)
        S_true_rb = self._rebin_1d(S_true, time_reb)
        C_true_rb = self._rebin_2d(C_true, time_reb)
        CL_hat_rb = self._rebin_1d(S_hat_CL, time_reb)

        dt = trainMD["time_step_sec"]
        t = (t0_bin + np.arange(len(S_hat_rb)) * max(1, int(time_reb))) * float(dt)

        ax = self.plt.subplot(2, 1, 1)
        ax.plot(t, S_hat_rb, color="k", linewidth=1.0, label="S_hat")
        ax.fill_between(t, S_hat_rb - CL_hat_rb, S_hat_rb + CL_hat_rb, color="gray", alpha=0.3, label="S_hat_CL")
        for m in range(C_hat_rb.shape[1]):
            ax.plot(t, C_hat_rb[:, m], linewidth=0.8, alpha=0.8, label=f"C_hat[{m}]")
        ax.set(title="Fit: S_hat and C_hat", xlabel="time (s)", ylabel="state")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=4, fontsize=8)

        ax = self.plt.subplot(2, 1, 2)
        ax.plot(t, S_true_rb, color="k", linewidth=1.0, label="S_true")
        for m in range(C_true_rb.shape[1]):
            ax.plot(t, C_true_rb[:, m], linewidth=0.8, alpha=0.8, label=f"C_true[{m}]")
        ax.set(title="Truth: S_true and C_true", xlabel="time (s)", ylabel="state")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=4, fontsize=8)

        fig.suptitle(f"State sequence: {md['short_name']}", fontsize=12)

    def state_corr_prismMstep(self, fitD, md, figId=2):
        self.state_corr_truth_prismMstep(fitD, md, type="fit", figId=figId)

    def state_Bcorr_prismMstep(self, fitD, md, figId=5):
        figId = self.smart_append(figId)

        B_hat = np.asarray(fitD["B_hat"])
        B_true = np.asarray(md["B_true"])
        if B_hat.ndim == 1:
            B_hat = B_hat[None, :]
        if B_true.ndim == 1:
            B_true = B_true[None, :]

        n_states = min(B_hat.shape[0], B_true.shape[0])
        if n_states < 2:
            fig = self.plt.figure(figId, facecolor='white', figsize=(8, 3))
            ax = self.plt.subplot(1, 1, 1)
            ax.axis("off")
            ax.text(0.05, 0.6, "Need at least 2 states for pairwise B-correlation", fontsize=11)
            fig.suptitle(f"State B correlations: {md['short_name']}", fontsize=12)
            return

        pairs = [(i, j) for i in range(n_states) for j in range(i + 1, n_states)]
        ncol = len(pairs)
        fig_w = max(10.0, 3.2 * ncol)
        fig = self.plt.figure(figId, facecolor='white', figsize=(fig_w, 6.2))

        for col, (i, j) in enumerate(pairs):
            ax = self.plt.subplot(2, ncol, 1 + col)
            plot_corr_B_divisor(
                ax, B_hat[i], B_hat[j],
                f"B_hat corr, s{i}-s{j}", f"B_hat s{i}", f"B_hat s{j}",
                s=8, alpha=0.6, color="orange"
            )

            ax = self.plt.subplot(2, ncol, ncol + 1 + col)
            plot_corr_B_divisor(
                ax, B_true[i], B_true[j],
                f"B_true corr, s{i}-s{j}", f"B_true s{i}", f"B_true s{j}",
                s=8, alpha=0.6, color="seagreen"
            )

        fig.suptitle(f"State B correlations: {md['short_name']}", fontsize=12)

    def eval_ABcorr_prismMstep(self, fitD, md, figId=2):
        A_hat = fitD["A_hat"]
        B_hat = fitD["B_hat"]
        A_true = md["A_true"]
        B_true = md["B_true"]
        figId = self.smart_append(figId)
        minW = float(self.args.minW)

        A_hat = np.asarray(A_hat)
        B_hat = np.asarray(B_hat)
        A_true = np.asarray(A_true)
        B_true = np.asarray(B_true)

        assert A_true.ndim == 2, f"A_true must be 2D, got shape={A_true.shape}"
        assert A_hat.ndim == 2, f"A_hat must be 2D, got shape={A_hat.shape}"

        if B_hat.ndim == 1:
            B_hat = B_hat[None, :]
        if B_true.ndim == 1:
            B_true = B_true[None, :]
        n_states = min(B_hat.shape[0], B_true.shape[0])
        assert n_states >= 1, "Need at least one B-state for correlation plot"

        ncol = 1 + n_states
        fig_w = max(10.0, 3.0 * ncol)
        fig = self.plt.figure(figId, facecolor='white', figsize=(fig_w, 3.8))

        ax = self.plt.subplot(1, ncol, 1)
        plot_corr_A_regions(ax, A_true.ravel(), A_hat.ravel(), minW, "A corr,", "A_true", "A_hat", s=6, alpha=0.4, color="green")

        for m in range(n_states):
            ax = self.plt.subplot(1, ncol, 2 + m)
            plot_corr_B_divisor(ax, B_true[m], B_hat[m], f"B corr, state {m}", "B_true", "B_hat", s=8, alpha=0.5, color="blue")

        fig.suptitle(f"2D correlations, minW={minW:g}: {md['short_name']}", fontsize=12)

    def edge_state_prismMstep(self, fitD, md, figId=6):
        A_hat = fitD["A_hat"]
        figId = self.smart_append(figId)
        edge_states = md.get("edge_eval_states", None)
        if edge_states is None:
            raise ValueError("Missing md['edge_eval_states']; compute edge metrics in prism_Mstep_eval.py")
        minW = float(edge_states[0]["minW"])

        A_hat = np.asarray(A_hat)
        if A_hat.ndim == 2:
            A_hat = A_hat[None, :, :]
        n_states = min(A_hat.shape[0], len(edge_states))
        N = A_hat.shape[1]
        fig_h = max(2.8, 2.8 * n_states)
        fig = self.plt.figure(figId, facecolor='white', figsize=(16, fig_h))

        for m in range(n_states):
            edge = edge_states[m]
            E_hat = np.asarray(edge["E_hat"]).astype(bool)
            E_t = np.asarray(edge["E_true_offdiag"]).astype(bool)
            conf_map = np.asarray(edge["conf_map"])

            tp = int(edge["tp"])
            fp = int(edge["fp"])
            fn = int(edge["fn"])
            precision = float(edge["precision"])
            recall = float(edge["recall"])
            f1 = float(edge["f1"])
            acc = float(edge["acc"])

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

            cmap_conf = colors.ListedColormap(['white', 'magenta', 'red', 'green'])
            norm_conf = colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_conf.N)
            ax = self.plt.subplot(n_states, 4, m * 4 + 3)
            im = ax.imshow(conf_map, cmap=cmap_conf, norm=norm_conf, origin='lower')
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
            vals = [tp, fp, fn]
            ax.bar(['TP', 'FP', 'FN'], vals, color=['green', 'red', 'magenta'])
            ax.set_ylabel('count')
            y_pos = 0.25 * max(1, max(vals))
            for name, val in zip(['TP', 'FP', 'FN'], vals):
                ax.text(name, y_pos, f"{val}", ha='center', va='center', fontsize=10)
            txt = f'precision={precision:.3f}\nrecall={recall:.3f}\nf1={f1:.3f}\nacc={acc:.3f}'
            ax.text(0.6, 0.60, txt, transform=ax.transAxes)
            ax.set_title('stats')
            ax.grid(axis='y', alpha=0.4)

        fig.subplots_adjust(bottom=0.12)
        fig.suptitle(
            f"Edge detection vs truth, minW={minW} (off diagonal only): {md['short_name']}",
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


def add_delay_markers(ax, train_md):
    delay = train_md.get("L1_prune_epoch", None)
    if delay is None:
        return
    delay = int(delay)
    if delay < 0:
        return
    # Training applies pruning at 0-based epoch>=delay; displayed epoch is delay+1.
    ax.axvline(delay + 1, color='k', linestyle='--', linewidth=1.0)


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
    n_left = int(mask_left.sum())
    n_mid = int(mask_mid.sum())
    n_right = int(mask_right.sum())

    posx = [0.2, 0.50, 0.65]
    posy = [0.05, 0.40, 0.60]
    ax.text(posx[0], posy[0], f"rL={r_left:.3f}\nnL={n_left}", transform=ax.transAxes)
    ax.text(posx[1], posy[1], f"rM={r_mid:.3f}\nnM={n_mid}", transform=ax.transAxes)
    ax.text(posx[2], posy[2], f"rR={r_right:.3f}\nnR={n_right}", transform=ax.transAxes)


def find_bimodal_divider(xV):
    """Return divider between two 1D modes using strict 2-cluster k-means."""
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


def plot_corr_B_divisor(ax, xV, yV, title, xlab, ylab, s=6, alpha=0.5, color=None):
    x = np.asarray(xV)
    y = np.asarray(yV)
    assert x.shape == y.shape, f"x and y shape mismatch: {x.shape} vs {y.shape}"
    divide = find_bimodal_divider(x)

    ax.scatter(x, y, s=s, alpha=alpha, color=color)
    add_x45_lins(ax, only45=True)
    ax.axvline(divide, color="red", linestyle="--", linewidth=1)
    ax.set(title=title, xlabel=xlab, ylabel=ylab)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.4)

    mask_left = x < divide
    mask_right = x >= divide
    r_left = corrcoef_safe(x[mask_left], y[mask_left])
    r_right = corrcoef_safe(x[mask_right], y[mask_right])
    n_left = int(mask_left.sum())
    n_right = int(mask_right.sum())
    ax.text(0.05, 0.45, f"rL={r_left:.3f}\nnL={n_left}", transform=ax.transAxes, va="top")
    ax.text(0.65, 0.55, f"rR={r_right:.3f}\nnR={n_right}", transform=ax.transAxes, va="top")
