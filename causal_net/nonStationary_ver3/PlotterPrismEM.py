#!/usr/bin/env python3
"""
Plotting utilities for prism EM evaluation.
"""

from toolbox.PlotterBackbone import PlotterBackbone
import numpy as np


class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self, args)

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

        prune_em = trainMD.get("L1_prune_em_iter", 0)
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
