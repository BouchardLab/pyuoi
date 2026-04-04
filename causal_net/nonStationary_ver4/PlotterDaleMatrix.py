#!/usr/bin/env python3
"""
Plotting utilities for simulated Poisson process visualization.

This module provides specialized plotting capabilities for analyzing
simulated Dale's principle neural networks. The Plotter class extends
PlotterBackbone to create visualizations including:
- Dale connectivity matrix plots with excitatory/inhibitory separation
- Eigenvalue analysis and network stability visualization
- Poisson process statistics and firing rate distributions
- Network dynamics and temporal evolution plots

Designed specifically for validating and analyzing simulated neural
networks that follow Dale's principle with Poisson spiking dynamics.
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from toolbox.PlotterBackbone import PlotterBackbone
from matplotlib import cm as cmap
import matplotlib.ticker as ticker
from pprint import pprint
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from Util_pseudospectra import compute_pseudospectrum
from UtilDalePoisson4 import get_offdiag_triplets


def _placement_ker_delta(dmd):
    return float(dmd["placement_ker_delta"])


def _format_ker_delta_latex(delta):
    """Matplotlib title fragment: δ_ker value (integers without .0)."""
    x = float(delta)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return "%.4g" % x


def _kernel_figure_canvas_title(md):
    """One-line caption: dataset, N, and (model B) Q, tau, placement H×L."""
    dmd = md["dale_conf"]
    parts = []
    sn = md.get("short_name", "")
    if sn:
        parts.append("dataset: %s" % sn)
    n = dmd.get("num_neurons")
    if n is not None:
        parts.append("N=%d" % int(n))
    if dmd.get("spike_model") == "B":
        if "mem_Q" in dmd:
            parts.append("Q=%.4g" % float(dmd["mem_Q"]))
        if "mem_tau" in dmd:
            parts.append("tau/sec=%.4g" % float(dmd["mem_tau"]))
    ph = dmd.get("placement_H")
    pl = dmd.get("placement_L")
    if ph is not None and pl is not None:
        parts.append("placement HxL=(%gx%g)" % (float(ph), float(pl)))
    parts.append("δ_ker=%s" % _format_ker_delta_latex(_placement_ker_delta(dmd)))
    return "  ".join(parts)


def _dale_overview_title(md):
    dmd = md["dale_conf"]
    return r"True Dale, $N=%d$, $\delta_{\mathrm{ker}}=%s$, %s" % (
        int(dmd["num_neurons"]),
        _format_ker_delta_latex(_placement_ker_delta(dmd)),
        md["short_name"],
    )


def _realized_edge_lengths(trueD):
    D = np.asarray(trueD["node_distance_matrix"], dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("node_distance_matrix must be square (N, N)")
    N = D.shape[0]
    edge_mask = np.asarray(trueD["E_true"]) != 0
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

    def _plot_signed_offdiag_matrix(self, ax, trueD, md, title_txt=None):
        dmd = md["dale_conf"]
        numNeur = int(dmd["num_neurons"])
        placement_ker_delta = _placement_ker_delta(dmd)

        if "E_true" in trueD:
            A_sign = np.asarray(trueD["E_true"], dtype=np.int8).copy()
        else:
            A_sign = np.sign(np.asarray(trueD["A_off_true"], dtype=np.float64)).astype(np.int8, copy=False)
        if A_sign.shape != (numNeur, numNeur):
            raise ValueError("signed off-diagonal topology shape mismatch")
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
            ylabel="neuron index (postsynaptic)",
            xlabel="presyn. neuron index (target)",
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
        numExc=dmd['num_excite']
        numNeur=dmd['num_neurons']
        placement_ker_delta = _placement_ker_delta(dmd)
        R_sel = md['sel_spect_radius']
        R_tag = ', R=%.3f' % R_sel if R_sel is not None else ''
        
        vmin = A.min()
        vmax = A.max()
        normMap = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        #.... Position 1: Dale matrix (transpose: x = postsynaptic i, y = presynaptic j) ......
        ax = self.plt.subplot(nrow,ncol,1)
        im = ax.imshow(
            A.T, aspect=1., origin='lower', cmap='bwr', norm=normMap, interpolation='nearest'
        )
        ax.set(
            ylabel=' neuron index (postsynaptic)',
            xlabel='presyn. neuron index (target)',
        )
        ax.set_aspect(1.0)
        ax.grid()
        cbar = fig.colorbar(im, ax=ax, extend="both", shrink=0.7)
        
        tit_left = r"True Dale, $N=%d$, $\delta_{\mathrm{ker}}=%s$, %s" % (
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
        tit3='Eigenvalues, N%d%s, %s'%(A.shape[0], R_tag, md['short_name'])
        ax.set_title(tit3)
        ax.axhline(0, color='black', lw=0.5)
        ax.axvline(0, color='black', lw=0.5)
        ax.axvline(0,color='red', linestyle='--')
        ax.set_aspect('equal')
        if R_sel is not None:
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
        R_tag = ', R=%.3f' % R_sel if R_sel is not None else ''
        title = 'Pseudospectra, true N%d%s, %s' % (A.shape[0], R_tag, md['short_name'])
        
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
        data_name=md['short_name']
        dmd=md['dale_conf']
        numNeur=dmd['num_neurons']
        R_sel = md['sel_spect_radius']
        R_tag = ', R=%.2f' % R_sel 
                
        A=trueD['A_true']
        single_rates = spikeD['single_rates']
  
        # output:  np.column_stack([i_indices, j_indices, values])
        EposT=get_offdiag_triplets(A,isPos=True)
        EnegT=get_offdiag_triplets(A,isPos=False)
        print('True0 num edges  pos=%d  neg=%d'%(EposT.shape[0],EnegT.shape[0]))

        A_diag = np.diag(A)
        wMin = min(np.min(EnegT[:,2]), np.min(A_diag))
        wMax = max(np.max(EposT[:,2]), np.max(A_diag))
                
        def count_elements(E, Nn=numNeur):
            i_indices = E[:, 0].astype(int)
            counts = np.bincount(i_indices, minlength=Nn)
            return counts
        
        edgeCount=count_elements(EnegT)  + count_elements(EposT)
        
        #....   weights histogram 
        ax = self.plt.subplot(nrow,ncol,1)
        binX= np.linspace(wMin, wMax, 100)
        ax.hist(EnegT[:,2], bins=binX, color='blue', alpha=0.7, edgecolor=None,label='neg:%d'%EnegT.shape[0])
        ax.hist(EposT[:,2], bins=binX, color='red', alpha=0.7, edgecolor=None,label='pos:%d'%EposT.shape[0])
        ax.hist(A_diag,   bins=binX, color='cyan', alpha=0.65, edgecolor=None,
                label='diag:%d'%A_diag.shape[0])

        ax.legend(loc='upper left')
        tit='True Dale, N%d%s, %s'%(A.shape[0], R_tag, md['short_name'])
        ax.set(title=tit, xlabel='True weight value',ylabel='num edges')
        ax.axvline(0,color='k',ls='--')
        ax.grid(True, alpha=0.3)

        #.... : histogram of rates
        ax = self.plt.subplot(nrow,ncol,3)
        ax.hist(single_rates, bins=20)
        ax.set_xlabel('Firing rate (Hz)')
        ax.set_ylabel('num neurons')
        ax.grid(True, alpha=0.3)
        ax.set_title('Single rates spectrum')
        ax.set_xlim(0,)
        median_val = np.median(single_rates)
        ax.axvline(median_val, color='r', linestyle='--', linewidth=1.5)
        y_max = ax.get_ylim()[1]
        median_text = f"median: {median_val:.2f} (Hz)\n N={single_rates.shape[0]}"
        ax.text( x=median_val * 1.1,  y=y_max * 0.7, s=median_text,  color='red')

        tau = np.asarray(trueD["node_is_inhibitory"]).reshape(-1)
        if tau.shape[0] != numNeur:
            raise ValueError("node_is_inhibitory length must match num_neurons")
        if tau.dtype.kind not in "iu":
            tau = tau.astype(np.int64)
        if np.any((tau != 0) & (tau != 1)):
            raise ValueError("node_is_inhibitory must be 0 (exc) or 1 (inh)")
        exc_mask = tau == 0
        inh_mask = tau == 1
        neurXlab = 'neuron index'
        
        x_vals = np.arange(numNeur)         
        #....  edge count
        ax = self.plt.subplot(nrow,ncol,2)
        ax.fill_between(x_vals, edgeCount, step='mid', color='salmon', alpha=0.7)
        ax.set_xlabel(neurXlab)
        ax.set_ylabel('num true edges')
        ax.set_ylim(0,)
        ax.grid(True, alpha=0.3)
        probLo, probHi = dmd['edge_prob']
        rho_title = f'outgoing edges, true, prob=[{probLo:.2f}, {probHi:.2f}]'
        ax.set_title(rho_title)
        
            
        #....  firing rates ..... 
        ax = self.plt.subplot(nrow,ncol,4)
        chanW=0.9        
               
        ax.bar(x_vals[exc_mask], single_rates[exc_mask], width=chanW, color='red', align='center', alpha=0.7, label='Excitatory')
        ax.bar(x_vals[inh_mask], single_rates[inh_mask], width=chanW, color='blue', align='center', alpha=0.7, label='Inhibitory')

        ax.set_xlabel(neurXlab)
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_ylim(0,)
        ax.grid(True, alpha=0.3)
        state_tag = md.get('sel_state', None)
        if state_tag is None:
            ax.set_title('Single Neurons')
        else:
            ax.set_title(f'Single Neurons, state={state_tag}')
        ax.legend()
 

#............................
#............................
#............................
    def rates_study(self,trueD,spikeD,md,figId=4):
        figId=self.smart_append(figId)
        nrow,ncol=1,4
        fig=self.plt.figure(figId,facecolor='white', figsize=(15,4))
        data_name=md['short_name']
        dmd=md['dale_conf']
        numNeur=dmd['num_neurons']
        R_sel = md['sel_spect_radius']
        R_tag = ', R=%.3f' % R_sel if R_sel is not None else ''
        
        B_idle = np.asarray(trueD["B_true"]).reshape(-1)
        single_rates = np.asarray(spikeD["single_rates"]).reshape(-1)
        if B_idle.shape[0] != numNeur:
            raise ValueError("B_true length must match dale_conf num_neurons")
        if single_rates.shape[0] != numNeur:
            raise ValueError("single_rates length must match num_neurons (same order as B_true, node_is_inhibitory)")

        tau = np.asarray(trueD["node_is_inhibitory"]).reshape(-1)
        if tau.shape[0] != numNeur:
            raise ValueError("node_is_inhibitory length must match num_neurons")
        if tau.dtype.kind not in "iu":
            tau = tau.astype(np.int64)
        if np.any((tau != 0) & (tau != 1)):
            raise ValueError("node_is_inhibitory must be 0 (exc) or 1 (inh)")
        exc_mask = tau == 0
        inh_mask = tau == 1
        n_exc = int(np.sum(exc_mask))
        n_inh = int(np.sum(inh_mask))

        # 1) Scatter: x=B_idle, y=log(single_rates)
        ax = self.plt.subplot(nrow,ncol,1)
        ax.scatter(B_idle[exc_mask], single_rates[exc_mask], s=14, alpha=0.6, marker='^', facecolors='none', edgecolors='red', label='Excitatory')
        ax.scatter(B_idle[inh_mask], single_rates[inh_mask], s=14, alpha=0.6, marker='^', facecolors='none', edgecolors='blue', label='Inhibitory')
        ax.set_xlabel('true B_idle ')
        ax.set_ylabel('single_rates (Hz)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        tit='True Dale, N%d%s, %s'%(numNeur, R_tag, data_name)
        ax.set_title(tit)
       
        # reference line: y = exp(B), clipped to central 80% of B range
        bmin = float(np.min(B_idle))
        bmax = float(np.max(B_idle))
        bLo = bmin + 0.1 * (bmax - bmin)
        bHi = bmax - 0.1 * (bmax - bmin)
        bx = np.linspace(bLo, bHi, 100)
        by = np.exp(bx)
        ax.plot(bx, by, linestyle='--', color='black', linewidth=0.8, label='y=exp(x)')
        ax.legend()

        # 2) Histogram: true B_idle (all neurons)
        ax2 = self.plt.subplot(nrow,ncol,2)
        b_bins = min(60, max(20, int(np.sqrt(numNeur) * 3)))
        ax2.hist(B_idle, bins=b_bins, color='dimgray', alpha=0.8, edgecolor=None)
        ax2.set_xlabel('true B_idle')
        ax2.set_ylabel('num neurons')
        ax2.grid(True, alpha=0.3)
        state_tag = md.get('sel_state', None)
        if state_tag is None:
            ax2.set_title('true B_idle')
        else:
            ax2.set_title(f'true B_idle, state={state_tag}')
        
        # 3) Histogram: single_rates for excitatory
        ax3 = self.plt.subplot(nrow,ncol,3)
        exc_vals = single_rates[exc_mask]
        inh_vals = single_rates[inh_mask]
        # compute common bins and range (start at 0)
        exc_max = np.max(exc_vals) if exc_vals.size > 0 else 0.0
        inh_max = np.max(inh_vals) if inh_vals.size > 0 else 0.0
        x_max = max(exc_max, inh_max)
        if x_max <= 0:  x_max = 1.0
        #common_bins = np.linspace(0, x_max, 21)
        common_bins = np.linspace(0, x_max, int(2*x_max))
        ax3.hist(exc_vals, bins=common_bins, color='red', alpha=0.7, edgecolor=None)
        ax3.set_xlabel('single_rates (Hz)')
        ax3.set_ylabel('num neurons')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Rates: Excitatory (N=%d)' % (n_exc))
        
        # 4) Histogram: single_rates for inhibitory
        ax4 = self.plt.subplot(nrow,ncol,4)
        ax4.hist(inh_vals, bins=common_bins, color='blue', alpha=0.7, edgecolor=None)
        ax4.set_xlabel('single_rates (Hz)')
        ax4.set_ylabel('num neurons')
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Rates: Inhibitory (N=%d)' % (n_inh))
        # unify x-range starting at 0 for both histograms
        ax3.set_xlim(0, x_max)
        ax4.set_xlim(0, x_max)

    def _plot_placement_topology_ax(self, ax, trueD, md, title_txt=None, arch_radius=None):
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

        M = np.asarray(trueD["E_true"])
        mask = M != 0
        if M.shape != (N, N):
            raise ValueError("E_true shape mismatch")

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
        leg_handles.append(Line2D([0], [0], color="lime", alpha=0.95, lw=2, label="1,2,3 hops"))

        if title_txt is None:
            title_txt = r"Neuron placement, $N=%d$, $\delta_{\mathrm{ker}}=%s$, %s" % (
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

        if arch_radius is not None and np.isfinite(arch_radius) and arch_radius > 0:
            thetaV = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 300)
            for x_center in (0.0, arch_radius, 2.0 * arch_radius):
                xV = x_center + arch_radius * np.cos(thetaV)
                yV = 0.5 + arch_radius * np.sin(thetaV)
                keep = np.abs(yV) <= 1.0
                if not np.any(keep):
                    continue
                xP = np.where(keep, xV, np.nan)
                yP = np.where(keep, yV, np.nan)
                ax.plot(xP, yP, color="lime", linewidth=2.0, alpha=0.95, zorder=2.2, clip_on=True)

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
        d_off = _realized_edge_lengths(trueD)
        arch_radius = float(np.percentile(d_off, 50)) if d_off.size > 0 else None
        self._plot_placement_topology_ax(ax, trueD, md, arch_radius=arch_radius)

    def _plot_offdiag_distance_hist_ax(self, ax, trueD, md, title_txt=None):
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

    def offdiag_distance_kernel_hist(self, trueD, md, figId=6):
        """
        One row, three panels: (1) histogram of off-diagonal distances from
        node_distance_matrix; (2) κ vs lag (model B); (3) empty.
        """
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(14, 4))
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_offdiag_distance_hist_ax(ax1, trueD, md)

        ax2 = fig.add_subplot(gs[0, 1])
        kappa = np.asarray(trueD["offdiag_kernel"], dtype=np.float64).ravel()
        if kappa.size > 0:
            lags = np.arange(1, kappa.size + 1, dtype=np.float64)
            ax2.plot(lags, kappa, color="darkorange", marker="o", markersize=4, linewidth=1.0)
            ax2.axhline(0.0, color="k", lw=0.5)
            ax2.set_title(r"offdiag_kernel $\kappa_\ell$ vs lag $\ell$")
            ax2.set_xlabel(r"lag $\ell$")
            ax2.set_ylabel(r"$\kappa$")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(
                0.5,
                0.5,
                r"model A: lag-1 (empty $\kappa$, $M=0$)",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_axis_off()

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_axis_off()

        title_txt = _kernel_figure_canvas_title(md)
        fig.suptitle(title_txt, fontsize=10, y=0.93)
        fig.subplots_adjust(top=0.74)

    def topo_overview(self, trueD, md, figId=7):
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(13, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.35], hspace=0.42, wspace=0.28)

        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_signed_offdiag_matrix(ax1, trueD, md, title_txt="Signed off-diagonal topology")

        ax2 = fig.add_subplot(gs[0, 1])
        pctD = self._plot_offdiag_distance_hist_ax(ax2, trueD, md, title_txt="Realized edge lengths")

        ax3 = fig.add_subplot(gs[1, :])
        arch_radius = pctD["p50"]
        self._plot_placement_topology_ax(ax3, trueD, md, title_txt="Neuron placement", arch_radius=arch_radius)
        fig.suptitle(_dale_overview_title(md), fontsize=16, y=0.985)
        fig.subplots_adjust(top=0.90)
