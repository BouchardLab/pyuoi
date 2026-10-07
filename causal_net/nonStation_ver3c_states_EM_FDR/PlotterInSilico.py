#!/usr/bin/env python3
"""
Plotting utilities for in-silico (NetPyNE synthetic network) data visualization.

The Plotter class extends PlotterBackboneV2 to create visualizations for
synthetic neuronal-network simulations, mirroring PlotterBioExp.py's plots but
sourcing node geometry/type from the prep_inSilico3c.py ground-truth schema
(node_positions / node_is_inhibitory) instead of a metrics_curated sheet.
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from toolbox.PlotterBackboneV2 import PlotterBackboneV2
from UtilBioExp import clip_rebD_time
import matplotlib.gridspec as gridspec
import numpy as np


#...!...!....................
def summary_column(md):
    rs = md['rate_summary']
    ds = md['data_selector']
    txt = 'dataset: ' + md['short_name']
    txt += '\nnum acc neurons: %d' % md['num_neurons']
    txt += '\ndrop neur: %d <%.1f Hz,  %d >%.1f Hz' % (
        ds['num_drop_neur_lo_hi_freq'][0], ds['freq_range'][0],
        ds['num_drop_neur_lo_hi_freq'][1], ds['freq_range'][1])
    txt += '\ndata type: %s\ntime_step=%.2f sec' % (md['data_type'], md['time_step_sec'])
    txt += '\nMedian rate: %.1f Hz' % rs['median_spike_rate']
    txt += '\nrate range [%.1f, %.1f Hz]' % (rs['min_spike_rate'], rs['max_spike_rate'])
    txt += '\nAvg Rate: %.1f±%.1f Hz' % (rs['avg_spike_rate'], rs['std_spike_rate'])
    txt += '\nAvg Fano: %.2f±%.2f' % (rs['avg_fano_factor'], rs['std_fano_factor'])
    return txt


#............................
#............................
#............................
class Plotter(PlotterBackboneV2):
    def __init__(self, args):
        PlotterBackboneV2.__init__(
            self,
            prjName=args.prjName,
            outPath=args.outPath,
            noXterm=args.noXterm,
            plotFormat=args.plotFormat,
        )

    #...!...!..................
    def freq_histo(self, spikeD, truthD, md, figId=1):
        tit = 'dataset ' + md['short_name']

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(16, 4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[7, 3])

        dataRates = spikeD['single_rates']
        is_inhib = np.asarray(truthD['node_is_inhibitory']).astype(bool)
        median_val = np.median(dataRates)
        txtM = f'median : {median_val:.2f} Hz'

        #.... freq per channel
        ax = self.plt.subplot(gs[0, 0])
        chanV = np.arange(dataRates.shape[0])
        ax.bar(chanV, dataRates, width=0.8, color='g', align='center', alpha=0.7)
        ax.axhline(median_val, color='red', linestyle='--', linewidth=1)
        ax.grid()
        ax.set(xlabel='input neuron index', ylabel='avr frequency (Hz)', title=tit)
        ax.text(ax.get_xlim()[1] * 0.1, median_val, txtM,
                color='red', ha='center', va='bottom', fontsize=10)

        txt = summary_column(md)
        ax.text(0.05, 0.3, txt, transform=ax.transAxes, fontsize=12, color='b')

        #........ freq histo .....
        ax = self.plt.subplot(gs[0, 1])
        bins = np.histogram_bin_edges(dataRates, bins=40)
        n_exc, n_inh = int(np.sum(~is_inhib)), int(np.sum(is_inhib))
        median_exc = np.median(dataRates[~is_inhib])
        median_inh = np.median(dataRates[is_inhib])

        ax.hist(dataRates, bins=bins, histtype='step', color='k', linewidth=1.3,
                label=f'all (N={dataRates.size})')
        ax.hist(dataRates[~is_inhib], bins=bins, color='red', alpha=0.5,
                label=f'excitatory (N={n_exc})')
        ax.hist(dataRates[is_inhib], bins=bins, fill=False, edgecolor='blue', hatch='//',
                label=f'inhibitory (N={n_inh})')
        ax.set(ylabel='num channels', xlabel='avr frequency (Hz)', title=tit)

        ymax = ax.get_ylim()[1]
        xoffset = (bins[-1] - bins[0]) * 0.02
        for val, color, yfrac, lab in [(median_val, 'k', 0.9, 'all'), (median_exc, 'red', 0.75, 'exc'),
                                        (median_inh, 'blue', 0.6, 'inh')]:
            ax.axvline(val, color=color, linestyle='--', linewidth=1)
            ax.text(val + xoffset, ymax * yfrac, f'{lab}: {val:.2f} Hz', color=color, ha='left', va='bottom', fontsize=9)

        ax.legend(loc='upper right', fontsize=9)
        ax.grid()

    #...!...!..................
    def plot_insilico_sum_rate(self, ax, rebD, clip):
        timeV = clip["timeV"]
        rate1D = clip["rate1D"]
        Tbin = clip["Tbin"]
        tL, tR = clip["tL"], clip["tR"]
        nchan = clip["nchan"]
        ax.bar(timeV + Tbin * 0.5, rate1D, width=Tbin, color='orange', align='center', alpha=0.7)
        ax.set(ylabel='sum rate (Hz)', title=f'sum rate from all {nchan} neurons')
        ax.set_xlim(tL, tR)
        ax.grid()

    #...!...!..................
    def plot_insilico_heatmap(self, fig, ax, rebD, dataset_name, clip):
        rate2D = clip["rate2D"]
        tL, tR = clip["tL"], clip["tR"]
        nchan = clip["nchan"]
        time_step = clip["time_step"]
        tit0 = f'dataset: {dataset_name}    nchan={nchan}  Tbin={time_step:.2f} sec'

        original_cmap = self.plt.get_cmap('tab20c')
        custom_cmap = original_cmap.copy()
        custom_cmap.set_under('white')

        cax = ax.imshow(
            rate2D.T, aspect='auto', cmap=custom_cmap, origin='lower',
            extent=[tL, tR, 0, nchan], interpolation='nearest',
            vmin=1, vmax=rate2D.max(),
        )
        ax.set_ylabel('Neuron index')
        ax.set_xlabel('Time (sec)')
        ax.set_title(tit0)
        cbar = fig.colorbar(
            cax, ax=ax, orientation='horizontal', pad=0.17,
            label=f'Instantanous freq  (Hz) per neuron over Tbin={time_step:.1f} sec',
            shrink=0.5,
        )
        cbar.ax.tick_params(labelsize=10)
        ax.grid()

    #...!...!..................
    def freq_vs_time(self, rebD, md, figId=2):
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(16, 11))

        rateThr2 = rebD['rate_thres2']
        clip = clip_rebD_time(rebD, md['plot']['time_rangeLR'])
        print('iTL,R', clip['itL'], clip['itR'])

        highChanCnt = rebD['highChanCnt'][clip['itL']:clip['itR']]
        timeV = clip['timeV']
        Tbin = clip['Tbin']
        tL, tR = clip['tL'], clip['tR']
        nchan = clip['nchan']

        gs = fig.add_gridspec(4, 1, height_ratios=[0.15, 0.15, 0.54, 0.01])

        ax = fig.add_subplot(gs[0, 0])
        ax.bar(timeV + Tbin * .5, highChanCnt, width=Tbin, color='forestgreen', align='center', alpha=0.7)
        tit = (f'num neurons with instantaneous rate > thres={rateThr2:.0f} (Hz), '
               f'nchan={nchan}')
        ax.set(ylabel='num neurons', title=tit)
        ax.set_xlim(tL, tR)
        ax.grid()

        ax = fig.add_subplot(gs[1, 0])
        self.plot_insilico_sum_rate(ax, rebD, clip)

        ax = fig.add_subplot(gs[2, 0])
        self.plot_insilico_heatmap(fig, ax, rebD, md['short_name'], clip)

    #...!...!..................
    def neuron_spatial(self, truthD, spikeD, md, figId=3):
        """Node layout: node_positions scatter colored by single_rates, E/I marked by shape."""
        pos = np.asarray(truthD['node_positions'], dtype=np.float64)
        loc_x, loc_y = pos[:, 0], pos[:, 1]
        rate = np.asarray(spikeD['single_rates'], dtype=np.float64)
        is_inhib = np.asarray(truthD['node_is_inhibitory']).astype(bool)

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(8, 9))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.06], hspace=0.12)

        ax = fig.add_subplot(gs[0, 0])
        ax.set_facecolor("white")
        vmax = float(np.max(rate))
        sc = ax.scatter(
            loc_x[~is_inhib], loc_y[~is_inhib], c=rate[~is_inhib], s=14, marker="o",
            cmap="Blues", vmin=0.0, vmax=vmax,
            linewidths=0.15, edgecolors="k", alpha=0.95, label="excitatory",
        )
        ax.scatter(
            loc_x[is_inhib], loc_y[is_inhib], c=rate[is_inhib], s=14, marker="^",
            cmap="Blues", vmin=0.0, vmax=vmax,
            linewidths=0.15, edgecolors="k", alpha=0.95, label="inhibitory",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("loc_x (um)", fontsize=11)
        ax.set_ylabel("loc_y (um)", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper right", fontsize=9)
        tit = f"dataset: {md['short_name']}   n={loc_x.shape[0]} neurons"
        ax.set_title(tit, fontsize=11, pad=8)

        cax = fig.add_subplot(gs[1, 0])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_label("Firing Rate [Hz]", fontsize=11)
        cb.ax.tick_params(labelsize=9)

        fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.10)
