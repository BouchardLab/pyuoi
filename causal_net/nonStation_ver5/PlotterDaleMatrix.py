#!/usr/bin/env python3
"""Plotting utilities for the BSSM-STD Bernoulli network generator."""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from toolbox.PlotterBackbone import PlotterBackbone
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from Util_pseudospectra import compute_pseudospectrum


def _placement_ker_delta(dmd):
    return float(dmd["placement_ker_delta"])


def _format_ker_delta_latex(delta):
    """Matplotlib title fragment: δ_ker value (integers without .0)."""
    x = float(delta)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return "%.4g" % x


def _kernel_figure_canvas_title(md):
    """One-line caption for the kernel / latent summary canvas."""
    dmd = md["dale_conf"]
    parts = []
    parts.append("dataset: %s" % md["short_name"])
    parts.append("N=%d" % int(dmd["num_neurons"]))
    parts.append(r"$\tau_s$=%.4g s" % float(dmd["synaptic_tau"]))
    parts.append(r"$\tau_{rec}$=%.4g s" % float(dmd["std_tau_rec"]))
    parts.append("U=%.3g" % float(dmd["std_u"]))
    parts.append("M=%d" % int(dmd["mem_lag_steps"]))
    parts.append("placement HxL=(%gx%g)" % (float(dmd["placement_H"]), float(dmd["placement_L"])))
    parts.append("δ_ker=%s" % _format_ker_delta_latex(_placement_ker_delta(dmd)))
    return "  ".join(parts)


def _dale_overview_title(md):
    dmd = md["dale_conf"]
    return r"BSSM-STD network, $N=%d$, $\delta_{\mathrm{ker}}=%s$, %s" % (
        int(dmd["num_neurons"]),
        _format_ker_delta_latex(_placement_ker_delta(dmd)),
        md["short_name"],
    )


def _baseline_rate_hz(trueD, md):
    b_true = np.asarray(trueD["b_true"], dtype=np.float64).reshape(-1)
    dt = float(md["evol_conf"]["step_size"])
    p0 = 1.0 / (1.0 + np.exp(-b_true))
    return p0 / dt


def _realized_edge_lengths(trueD):
    D = np.asarray(trueD["node_distance_matrix"], dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("node_distance_matrix must be square (N, N)")
    N = D.shape[0]
    edge_mask = np.asarray(trueD["sign_true"]) != 0
    if edge_mask.shape != (N, N):
        raise ValueError("edge mask shape mismatch with node_distance_matrix")
    edge_mask = edge_mask.copy()
    np.fill_diagonal(edge_mask, False)
    d_off = D[edge_mask]
    return d_off[d_off > 0]


#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)

    def _plot_signed_connectivity_matrix(self, ax, trueD, md, title_txt=None):
        dmd = md["dale_conf"]
        numNeur = int(dmd["num_neurons"])
        placement_ker_delta = _placement_ker_delta(dmd)

        A_sign = np.asarray(trueD["sign_true"], dtype=np.int8).copy()
        if A_sign.shape != (numNeur, numNeur):
            raise ValueError("signed connectivity shape mismatch")
        np.fill_diagonal(A_sign, 0)

        sign_cmap = colors.ListedColormap(["blue", "white", "red"])
        sign_norm = colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], sign_cmap.N)
        ax.imshow(
            A_sign.T,
            aspect=1.0,
            origin="lower",
            cmap=sign_cmap,
            norm=sign_norm,
            interpolation="nearest",
        )
        ax.set(
            xlabel="postsynaptic neuron index $i$",
            ylabel="presynaptic neuron index $j$",
        )
        ax.set_aspect(1.0)
        ax.grid(True, alpha=0.45)
        if title_txt is None:
            title_txt = _dale_overview_title(md)
        ax.set_title(title_txt)
        ax.plot([0, numNeur], [0, numNeur], "--", lw=0.8, color="magenta")

#...!...!..................
    def Dale_matrix_and_eigen(self,A,md,trueD,figId=3):
        
        figId=self.smart_append(figId)        
        nrow,ncol=1,2
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,5.))

        dmd=md['dale_conf']
        numNeur=dmd['num_neurons']
        placement_ker_delta = _placement_ker_delta(dmd)
        R_sel = md['sel_spect_radius']
        R_tag = ', R=%.3f' % R_sel
        
        vmin = A.min()
        vmax = A.max()
        normMap = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        #.... Position 1: Dale matrix (transpose: x = postsynaptic i, y = presynaptic j) ......
        ax = self.plt.subplot(nrow,ncol,1)
        im = ax.imshow(
            A.T, aspect=1., origin='lower', cmap='bwr', norm=normMap, interpolation='nearest'
        )
        ax.set(
            xlabel='postsynaptic neuron index $i$',
            ylabel='presynaptic neuron index $j$',
        )
        ax.set_aspect(1.0)
        ax.grid()
        cbar = fig.colorbar(im, ax=ax, extend="both", shrink=0.7)
        cbar.set_label('weight $W_{ij}$')
        
        tit_left = r"BSSM-STD weights $W$, $N=%d$, $\delta_{\mathrm{ker}}=%s$, %s" % (
            A.shape[0],
            _format_ker_delta_latex(placement_ker_delta),
            md["short_name"],
        )
        ax.set(title=tit_left)
        ax.plot([0,numNeur],[0,numNeur],'--',lw=0.8,color='magenta')
        
        #..... Position 2: Eigenvalues......
        ax = self.plt.subplot(nrow,ncol,2)
        Eigen=np.linalg.eigvals(A)
        real_parts = np.real(Eigen)
        imag_parts = np.imag(Eigen)
        ax.scatter(real_parts, imag_parts, color='blue', marker='o')
        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")
        tit3='Eigenvalues of recurrent $W$, N%d%s, %s'%(A.shape[0], R_tag, md['short_name'])
        ax.set_title(tit3)
        ax.axhline(0, color='black', lw=0.5)
        ax.axvline(0, color='black', lw=0.5)
        ax.axvline(0,color='red', linestyle='--')
        ax.set_aspect('equal')
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(R_sel*np.cos(theta), R_sel*np.sin(theta), color='magenta', linestyle='--', lw=1.5, label='R=%.3f'%R_sel)
        ax.legend()
        ax.grid(True)
      
    def Dale_matrix_pseudospectra(self, A, md, trueD, figId=5):  # p=e
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(8, 6))
        
        npts = 80
        minY = -0.2
        epsMin = 0.03

        X, Y, sigma_grid, eigs = compute_pseudospectrum(A, npts, minY)
        
        ax = self.plt.subplot(1, 1, 1)
        R_sel = md['sel_spect_radius']
        R_tag = ', R=%.3f' % R_sel
        title = 'Pseudospectra of recurrent $W$, N%d%s, %s' % (A.shape[0], R_tag, md['short_name'])
        
        levels = np.logspace(-2.5, -0.5, 10)
        contour = ax.contour(X, Y, sigma_grid, levels=levels, cmap='viridis', linewidths=0.8)
        ax.clabel(contour, inline=True, fontsize=8, fmt='ε=%.3f')
        ax.text(0.05, 0.85,   'green: ε<%.3f'%epsMin, transform=ax.transAxes) 
        mask = sigma_grid > epsMin
        sigma_grid_masked = np.ma.array(sigma_grid, mask=mask)

        contour_fill = ax.contourf(X, Y, sigma_grid_masked, levels=np.linspace(0, epsMin, 10), 
                                    colors=['lightgreen'], alpha=0.5)
        
        ax.scatter(np.real(eigs), np.imag(eigs), color='red', s=15, zorder=3, label='Eigenvalues')
        
        ax.axvline(0, color='black', linestyle='--', lw=1.5)
        ax.axhline(0, color='black', linestyle='--', lw=1.5)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        ax.set_ylim(minY,1.1)
        ax.set_xlim(-1.1,1.1)
        ax.set_aspect(1.)
        ax.axhline(1,color='m',linestyle='--')
        ax.axvline(1,color='m',linestyle='--')
        ax.axvline(-1,color='m',linestyle='--')

#...!...!..................
    def histo_weights_rates(self,trueD,spikeD,md,figId=3):        
        figId=self.smart_append(figId)        
        nrow,ncol=1,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(15,3.5))        
        dmd=md['dale_conf']
        numNeur=dmd['num_neurons']
        R_sel = md['sel_spect_radius']
        R_tag = ', R=%.2f' % R_sel

        W = np.asarray(trueD["W_true"], dtype=np.float64)
        off_mask = ~np.eye(numNeur, dtype=bool)
        pos_weights = W[(W > 0) & off_mask]
        neg_weights = W[(W < 0) & off_mask]
        single_rates = np.asarray(spikeD['single_rates']).reshape(-1)
        tau = np.asarray(trueD["node_is_inhibitory"]).reshape(-1)
        if tau.shape[0] != numNeur:
            raise ValueError("node_is_inhibitory length must match num_neurons")
        if tau.dtype.kind not in "iu":
            tau = tau.astype(np.int64)
        if np.any((tau != 0) & (tau != 1)):
            raise ValueError("node_is_inhibitory must be 0 (exc) or 1 (inh)")
        exc_mask = tau == 0
        inh_mask = tau == 1
        baseline_rates = _baseline_rate_hz(trueD, md)
        edge_mask = np.asarray(trueD["sign_true"]) != 0
        in_degree = np.sum(edge_mask, axis=1)
        out_degree = np.sum(edge_mask, axis=0)
        x_vals = np.arange(numNeur)
        
        #....   weights histogram 
        ax = self.plt.subplot(nrow,ncol,1)
        weight_all = np.concatenate([neg_weights, pos_weights]) if (neg_weights.size + pos_weights.size) > 0 else np.array([-1.0, 1.0])
        binX= np.linspace(float(np.min(weight_all)), float(np.max(weight_all)), 80)
        if neg_weights.size > 0:
            ax.hist(neg_weights, bins=binX, color='blue', alpha=0.7, edgecolor=None, label='inhib:%d'%neg_weights.size)
        if pos_weights.size > 0:
            ax.hist(pos_weights, bins=binX, color='red', alpha=0.7, edgecolor=None, label='excit:%d'%pos_weights.size)
        ax.legend(loc='upper left')
        tit='Recurrent weights $W_{ij}$, N%d%s, %s'%(W.shape[0], R_tag, md['short_name'])
        ax.set(title=tit, xlabel='off-diagonal weight value',ylabel='num edges')
        ax.axvline(0,color='k',ls='--')
        ax.grid(True, alpha=0.3)

        #....  in/out degree
        ax = self.plt.subplot(nrow,ncol,2)
        ax.plot(x_vals, out_degree, color='darkred', lw=1.2, alpha=0.85, label='out-degree')
        ax.plot(x_vals, in_degree, color='darkblue', lw=1.2, alpha=0.85, label='in-degree')
        ax.set_xlabel('neuron index')
        ax.set_ylabel('num edges')
        ax.set_ylim(0,)
        ax.grid(True, alpha=0.3)
        probLo, probHi = dmd['edge_prob']
        rho_title = f'realized degree counts, prob=[{probLo:.2f}, {probHi:.2f}]'
        ax.set_title(rho_title)
        ax.legend()

        #....  firing-rate histogram
        ax = self.plt.subplot(nrow,ncol,3)
        common_bins = np.linspace(0, max(20.0, float(np.max(single_rates)) * 1.05), 24)
        ax.hist(single_rates[exc_mask], bins=common_bins, color='red', alpha=0.6, edgecolor=None, label='exc')
        ax.hist(single_rates[inh_mask], bins=common_bins, color='blue', alpha=0.6, edgecolor=None, label='inh')
        ax.axvspan(5.0, 20.0, color='limegreen', alpha=0.12, label='target 5-20 Hz')
        ax.set_xlabel('firing rate (Hz)')
        ax.set_ylabel('num neurons')
        ax.grid(True, alpha=0.3)
        ax.set_title('Empirical rate distribution')
        ax.legend()

        #....  baseline-vs-empirical rates
        ax = self.plt.subplot(nrow,ncol,4)
        ax.scatter(baseline_rates[exc_mask], single_rates[exc_mask], s=18, alpha=0.65,
                   marker='^', facecolors='none', edgecolors='red', label='Excitatory')
        ax.scatter(baseline_rates[inh_mask], single_rates[inh_mask], s=18, alpha=0.65,
                   marker='s', facecolors='none', edgecolors='blue', label='Inhibitory')
        lim_hi = max(float(np.max(baseline_rates)), float(np.max(single_rates)), 1.0)
        ax.plot([0, lim_hi], [0, lim_hi], '--', color='black', lw=0.8, label='identity')
        ax.set_xlabel('baseline Bernoulli rate from $b_i$ (Hz)')
        ax.set_ylabel('empirical firing rate (Hz)')
        ax.grid(True, alpha=0.3)
        ax.set_title('Baseline rate vs empirical rate')
        ax.legend()
 

#............................
#............................
#............................
    def rates_study(self,trueD,spikeD,md,figId=4):
        figId=self.smart_append(figId)
        nrow,ncol=1,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(15,4))
        dmd=md['dale_conf']
        numNeur=dmd['num_neurons']
        R_sel = md['sel_spect_radius']
        R_tag = ', R=%.3f' % R_sel
        
        single_rates = np.asarray(spikeD["single_rates"]).reshape(-1)
        x_true = np.asarray(trueD["x_true"], dtype=np.float64)
        u_true = np.asarray(trueD["u_true"], dtype=np.float64)
        h_true = np.asarray(trueD["h_true"], dtype=np.float64)
        p_true = np.asarray(trueD["p_true"], dtype=np.float64)

        tau = np.asarray(trueD["node_is_inhibitory"]).reshape(-1)
        if tau.shape[0] != numNeur:
            raise ValueError("node_is_inhibitory length must match num_neurons")
        if tau.dtype.kind not in "iu":
            tau = tau.astype(np.int64)
        if np.any((tau != 0) & (tau != 1)):
            raise ValueError("node_is_inhibitory must be 0 (exc) or 1 (inh)")
        exc_mask = tau == 0
        inh_mask = tau == 1
        dt = float(md["evol_conf"]["step_size"])
        T = x_true.shape[0]
        stride = max(1, T // 2000)
        t_sec = dt * np.arange(0, T, stride, dtype=np.float64)
        mean_x_t = np.mean(x_true[::stride], axis=1)
        p10_x_t = np.percentile(x_true[::stride], 10, axis=1)
        p90_x_t = np.percentile(x_true[::stride], 90, axis=1)
        mean_u_t = np.mean(u_true[::stride], axis=1)
        mean_h_t = np.mean(h_true[::stride], axis=1)
        mean_p_rate = np.mean(p_true, axis=0) / dt
        mean_x_per_neuron = np.mean(x_true, axis=0)

        # 1) population-average STD resource trace
        ax = self.plt.subplot(nrow,ncol,1)
        ax.fill_between(t_sec, p10_x_t, p90_x_t, color='lightsteelblue', alpha=0.45, label='10-90 pct')
        ax.plot(t_sec, mean_x_t, color='navy', lw=1.4, label=r'mean $x_t$')
        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'STD resource $x$')
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        tit='BSSM-STD latent state, N%d%s, %s'%(numNeur, R_tag, md['short_name'])
        ax.set_title(tit)
        ax.legend()

        # 2) population-average release and filtered drive
        ax2 = self.plt.subplot(nrow,ncol,2)
        ax2.plot(t_sec, mean_u_t, color='darkorange', lw=1.2, label=r'mean $u_t$')
        ax2.plot(t_sec, mean_h_t, color='darkgreen', lw=1.2, label=r'mean $h_t$')
        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('population mean')
        ax2.grid(True, alpha=0.3)
        ax2.set_title(r'STD release $u_t$ and filtered drive $h_t$')
        ax2.legend()
        
        # 3) mean Bernoulli probability vs empirical rate
        ax3 = self.plt.subplot(nrow,ncol,3)
        ax3.scatter(mean_p_rate[exc_mask], single_rates[exc_mask], s=18, alpha=0.65,
                    marker='^', facecolors='none', edgecolors='red', label='Excitatory')
        ax3.scatter(mean_p_rate[inh_mask], single_rates[inh_mask], s=18, alpha=0.65,
                    marker='s', facecolors='none', edgecolors='blue', label='Inhibitory')
        lim_hi = max(float(np.max(mean_p_rate)), float(np.max(single_rates)), 1.0)
        ax3.plot([0, lim_hi], [0, lim_hi], '--', color='black', lw=0.8)
        ax3.set_xlabel('model-based mean rate from $p_{i,t}$ (Hz)')
        ax3.set_ylabel('observed spike rate from $s_{i,t}$ (Hz)')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Model-based vs observed firing rate')
        ax3.text(
            0.03, 0.55, r"$\hat r_i^{\mathrm{obs}}=\frac{1}{T\Delta t}\sum_{t=0}^{T-1}s_{i,t}$",
            transform=ax3.transAxes, ha="left", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.6", alpha=0.92),
        )
        ax3.text(
            0.97, 0.03, r"$\bar r_i^{\mathrm{model}}=\frac{1}{T\Delta t}\sum_{t=0}^{T-1}p_{i,t}$",
            transform=ax3.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.6", alpha=0.92),
        )
        ax3.legend()
        
        # 4) mean STD resource vs empirical rate
        ax4 = self.plt.subplot(nrow,ncol,4)
        ax4.scatter(mean_x_per_neuron[exc_mask], single_rates[exc_mask], s=18, alpha=0.65,
                    marker='^', facecolors='none', edgecolors='red', label='Excitatory')
        ax4.scatter(mean_x_per_neuron[inh_mask], single_rates[inh_mask], s=18, alpha=0.65,
                    marker='s', facecolors='none', edgecolors='blue', label='Inhibitory')
        ax4.set_xlabel(r'time-avg resource $\bar{x}_i$')
        ax4.set_ylabel('empirical firing rate (Hz)')
        ax4.grid(True, alpha=0.3)
        ax4.set_title(r'Resource depletion vs firing rate')
        ax4.legend()

    def _plot_placement_topology_ax(self, ax, trueD, md, title_txt=None):
        dmd = md["dale_conf"]
        L = float(dmd["placement_L"])
        H = float(dmd["placement_H"])
        if not (L > 0 and H > 0):
            raise ValueError("placement_L and placement_H must be positive")
        placement_ker_delta = _placement_ker_delta(dmd)

        P = np.asarray(trueD["node_positions"], dtype=float)
        tau = np.asarray(trueD["node_is_inhibitory"]).reshape(-1)
        if tau.dtype.kind not in "iu":
            tau = tau.astype(np.int64)
        N = P.shape[0]
        if tau.shape[0] != N:
            raise ValueError("node_positions and node_is_inhibitory length mismatch")
        exc_m = tau == 0
        inh_m = tau == 1
        if not np.any(exc_m) or not np.any(inh_m):
            raise ValueError("node_is_inhibitory must contain both excitatory (0) and inhibitory (1) labels")

        M = np.asarray(trueD["sign_true"])
        mask = M != 0

        red_segs = []
        blue_segs = []
        for j in range(N):
            for i in range(N):
                if i == j or not mask[i, j]:
                    continue
                seg = np.array([P[j], P[i]])
                if exc_m[j]:
                    red_segs.append(seg)
                elif inh_m[j]:
                    blue_segs.append(seg)

        if red_segs:
            ax.add_collection(LineCollection(red_segs, colors="red", linewidths=0.9, alpha=0.5, zorder=1))
        if blue_segs:
            ax.add_collection(LineCollection(blue_segs, colors="blue", linewidths=0.9, alpha=0.5, zorder=1))

        ax.scatter(
            P[exc_m, 0],
            P[exc_m, 1],
            s=65,
            marker="^",
            facecolors="none",
            edgecolors="darkred",
            linewidths=1.2,
            zorder=3,
        )
        ax.scatter(
            P[inh_m, 0],
            P[inh_m, 1],
            s=55,
            marker="s",
            facecolors="none",
            edgecolors="darkblue",
            linewidths=1.2,
            zorder=3,
        )

        label_color = "black"
        for k in range(0, N, 10):
            ax.text(
                P[k, 0],
                P[k, 1],
                str(k),
                color=label_color,
                fontsize=8,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.9),
                zorder=4,
            )

        leg_handles = []
        if red_segs:
            leg_handles.append(Line2D([0], [0], color="red", alpha=0.6, lw=2, label="Exc outgoing"))
        if blue_segs:
            leg_handles.append(Line2D([0], [0], color="blue", alpha=0.6, lw=2, label="Inh outgoing"))
        leg_handles.append(
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="none",
                markeredgecolor="darkred",
                markersize=9,
                label="Excitatory",
            )
        )
        leg_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="none",
                markeredgecolor="darkblue",
                markersize=8,
                label="Inhibitory",
            )
        )
        if title_txt is None:
            title_txt = r"Spatial connectivity, $N=%d$, $\delta_{\mathrm{ker}}=%s$, %s" % (
                N,
                _format_ker_delta_latex(placement_ker_delta),
                md["short_name"],
            )
        ax.set_title(title_txt, pad=36)
        ax.set_xlabel("x (placement)")
        ax.set_ylabel("y (placement)")
        ax.grid(True, alpha=0.3)
        ax.legend(
            handles=leg_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(leg_handles),
            fontsize=9,
            frameon=True,
        )

        mx, my = 0.2, 0.05
        ax.set_xlim(0.0 - mx, L + mx)
        ax.set_ylim(0.0 - my, H + my)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: "%d" % int(round(x))))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: "%d" % int(round(x))))

    def plot_placement_topology(self, trueD, md, figId=6):
        """
        2D placement: triangles = excitatory, squares = inhibitory;
        outgoing edges from j→i: red if j is excitatory, blue if inhibitory (presynaptic type).
        """
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        self._plot_placement_topology_ax(ax, trueD, md)

    def _plot_connection_length_hist_ax(self, ax, trueD, md, title_txt=None):
        d_off = _realized_edge_lengths(trueD)
        dmd = md["dale_conf"]
        d_ker = _format_ker_delta_latex(_placement_ker_delta(dmd))
        pctD = None
        if d_off.size > 0:
            ax.hist(
                d_off,
                bins=min(60, max(10, d_off.size // 5)),
                color="steelblue",
                edgecolor="white",
                alpha=0.9,
            )
            p16, p50, p84 = np.percentile(d_off, [16, 50, 84])
            pctD = {"p16": float(p16), "p50": float(p50), "p84": float(p84)}
            ax.axvline(p16, color="darkorange", linestyle="--", linewidth=1.3, zorder=4)
            ax.axvline(p50, color="darkgreen", linestyle="--", linewidth=1.3, zorder=4)
            ax.axvline(p84, color="darkviolet", linestyle="--", linewidth=1.3, zorder=4)
            pct_txt = "p16 = %.5g\np50 = %.5g\np84 = %.5g" % (p16, p50, p84)
            ax.text(
                0.97,
                0.97,
                pct_txt,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.6", alpha=0.92),
                zorder=5,
            )
        if title_txt is None:
            title_txt = r"Realized edge lengths ($\delta_{\mathrm{ker}}=%s$)" % d_ker
        ax.set_title(title_txt)
        ax.set_xlabel("distance")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3)
        return pctD

    def connection_length_kernel_hist(self, trueD, md, figId=6):
        """
        One row, three panels: edge lengths, lag-M synaptic kernel, STD recovery envelope.
        """
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(14, 4))
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_connection_length_hist_ax(ax1, trueD, md)

        ax2 = fig.add_subplot(gs[0, 1])
        kappa = np.asarray(trueD["kappa_true"], dtype=np.float64).ravel()
        kappa_plot = np.maximum(kappa, np.finfo(np.float64).tiny)
        dt = float(md["evol_conf"]["step_size"])
        tau_s = float(md["dale_conf"]["synaptic_tau"])
        lags = dt * np.arange(1, kappa.size + 1, dtype=np.float64)
        kernel_xmax_sec = 0.015
        lags_ms = lags * 1000.0
        kernel_xmax_ms = kernel_xmax_sec * 1000.0
        kernel_visible = lags <= kernel_xmax_sec
        if not np.any(kernel_visible):
            kernel_visible = lags <= lags[0]
        kappa_sum_in_window = float(np.sum(kappa[kernel_visible]))
        ax2.plot(lags_ms, kappa_plot, color="darkorange", marker="o", markersize=3.5, linewidth=1.0)
        ax2.set_title(r"Fast synaptic kernel $\kappa_\ell$ ($\tau_s$)")
        ax2.set_xlabel("lag time (ms)")
        ax2.set_ylabel(r"kernel entry $\kappa_\ell$")
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: "%.3f" % v))
        ax2.set_xlim(0.0, kernel_xmax_ms)
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: "%d" % int(round(v))))
        ymax = float(np.max(kappa_plot[kernel_visible]) * 1.05)
        ax2.set_ylim(0.0, ymax)
        ax2.grid(True, alpha=0.3)
        txt = "true $\\tau_s$=%.3f s\n$\\sum \\kappa$=%.4f" % (tau_s, kappa_sum_in_window)
        ax2.text(
            0.97, 0.97, txt, transform=ax2.transAxes, ha="right", va="top",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.6", alpha=0.92),
        )

        ax3 = fig.add_subplot(gs[0, 2])
        tau_rec = float(md["dale_conf"]["std_tau_rec"])
        recovery_xmax = 1.0
        t_recovery = np.linspace(0.0, recovery_xmax, 400)
        recovery_curve = 1.0 - np.exp(-t_recovery / tau_rec)
        ax3.plot(t_recovery, recovery_curve, color="darkgreen", linewidth=1.3)
        ax3.axhline(1.0 - np.exp(-1.0), color="black", linestyle="--", lw=0.8)
        ax3.axvline(tau_rec, color="magenta", linestyle="--", lw=1.0)
        ax3.set_ylim(-0.02, 1.02)
        ax3.set_title(r"STD recovery set by $\tau_{rec}$")
        ax3.set_xlabel("lag time (s)")
        ax3.set_ylabel("recovered fraction")
        ax3.set_xlim(0.0, recovery_xmax)
        ax3.grid(True, alpha=0.3)
        txt = r"$M\Delta t$=%.4g s" "\n" r"$\tau_{rec}$=%.4g s" "\n" r"$x(\tau_{rec})$=%.3f" % (
            float(lags[-1]), tau_rec, float(1.0 - np.exp(-1.0))
        )
        ax3.text(
            0.97, 0.08, txt, transform=ax3.transAxes, ha="right", va="bottom",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.6", alpha=0.92),
        )

        title_txt = _kernel_figure_canvas_title(md)
        fig.suptitle(title_txt, fontsize=10, y=0.93)
        fig.subplots_adjust(top=0.74)

    def topo_overview(self, trueD, md, figId=7):
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(13, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.35], hspace=0.42, wspace=0.28)

        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_signed_connectivity_matrix(ax1, trueD, md, title_txt="Signed connectivity")

        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_connection_length_hist_ax(ax2, trueD, md, title_txt="Realized connection lengths")

        ax3 = fig.add_subplot(gs[1, :])
        self._plot_placement_topology_ax(ax3, trueD, md, title_txt="Spatial connectivity")
        fig.suptitle(_dale_overview_title(md), fontsize=16, y=0.985)
        fig.subplots_adjust(top=0.90)
