#!/usr/bin/env python3
"""
Plotting utilities for biological experiment data visualization.

This module provides specialized plotting capabilities for experimental
neural data analysis. The Plotter class extends PlotterBackboneV2 to create
visualizations tailored for biological neural recordings including:
- Time series plots of neural activity patterns
- Statistical summaries and distribution analysis  
- Data quality assessment plots
- Comparative analysis between experimental conditions

Designed specifically for processing and visualizing data from biological
neural experiments, with automatic adaptation to experimental metadata.
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

from toolbox.PlotterBackboneV2 import PlotterBackboneV2
from UtilBioExp import clip_rebD_time
from matplotlib import cm as cmap
import matplotlib.ticker as ticker
from pprint import pprint
import numpy as np
import matplotlib.gridspec as gridspec

from matplotlib.colors import LinearSegmentedColormap

METRICS_HIST_COLUMNS = [
    "num_spikes",
    "firing_rate",
    "presence_ratio",
    "snr",
    "isi_violations_ratio",
    "isi_violations_count",
    "rp_contamination",
    "rp_violations",
    "sliding_rp_violation",
    "amplitude_cutoff",
    "amplitude_median",
    "amplitude_cv_median",
    "amplitude_cv_range",
    "sync_spike_2",
    "sync_spike_4",
    "sync_spike_8",
    "firing_range",
    "sd_ratio",
    "noise_cutoff",
    "noise_ratio",
]

METRICS_HIST_TITLE_UNITS = {
    "firing_rate": "Hz",
    "amplitude_median": "uV",
}

METRICS_HIST_XMIN_ZERO = {
    "snr",
    "firing_rate",
    "sync_spike_2",
    "sync_spike_4",
    "sync_spike_8",
    "firing_range",
    "sd_ratio",
}

METRICS_HIST_LOG_Y = {
    "presence_ratio",
    "isi_violations_ratio",
    "isi_violations_count",
    "rp_violations",
    "sliding_rp_violation",
    "amplitude_cutoff",
}

#...!...!....................
def summary_column(md):
    #print(sorted(md))
    rs=md['rate_summary']
    ds=md['data_selector']
    #pprint(md)
    txt='dataset: '+md['short_name']
    txt += '\nnum acc neurons: %d' % md['num_neurons']
    txt += '\ndrop neur: %d <%.1f Hz,  %d >%.1f Hz' % (ds['num_drop_neur_lo_hi_freq'][0],ds['freq_range'][0],ds['num_drop_neur_lo_hi_freq'][1],ds['freq_range'][1])
    txt+='\ndata type: %s\ntime_step=%.2f sec'%(md['data_type'],md['time_step_sec']) 
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
    def freq_histo(self,spikeD,md,figId=1):
        
        tit='dataset '+md['short_name']
        
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,4))
        
        # Create gridspec with 7:3 width ratio
        gs = gridspec.GridSpec(1, 2, width_ratios=[7, 3])

        dataYield, dataRates = spikeD['spikes'], spikeD['single_rates']

        # Compute median
        median_val = np.median(dataRates)
        txtM= f'median : {median_val:.2f} Hz'
        
        #.... freq per channel
        ax = self.plt.subplot(gs[0, 0])
        chanV=np.arange(dataRates.shape[0])
        ax.bar(chanV, dataRates , width=0.8, color='g', align='center', alpha=0.7)
         
        ax.axhline(median_val, color='red', linestyle='--', linewidth=1)
        if md['data_type']=='simDale':
            ax.set_yscale('log')
        ax.grid()
        ax.set(xlabel='input neuron index', ylabel='avr frequency (Hz)',title=tit)
        ax.text( ax.get_xlim()[1]*0.1,median_val,txtM, 
                 color='red', ha='center', va='bottom', fontsize=10)

        txt=summary_column(md)
        ax.text(0.05, 0.3,   txt, transform=ax.transAxes, fontsize=12,color='b')
        
        #........ freq histo .....
        ax = self.plt.subplot(gs[0, 1])
        ax.hist(dataRates,bins=20)
        #ax.set_yscale('log')
        ax.set(ylabel='num channels', xlabel='avr frequency (Hz)',title=tit)
        ax.axvline(median_val, color='red', linestyle='--', linewidth=1)
        # Add median text annotation
        ax.text(median_val+1, ax.get_ylim()[1]*0.5,txtM, 
                color='red', ha='left', va='bottom', fontsize=10,rotation=90)
        ax.text(0.25, 0.3,   txt, transform=ax.transAxes, fontsize=11,color='m')
      
        ax.grid()
      
    def plot_bioexp_sum_rate(self, ax, rebD, clip):
        timeV = clip["timeV"]
        rate1D = clip["rate1D"]
        Tbin = clip["Tbin"]
        tL, tR = clip["tL"], clip["tR"]
        nchan = clip["nchan"]
        ax.bar(timeV + Tbin * 0.5, rate1D, width=Tbin, color='orange', align='center', alpha=0.7)
        ax.set(ylabel='sum rate (Hz)', title=f'sum rate from all {nchan} neurons')
        ax.set_xlim(tL, tR)
        ax.grid()

    def plot_bioexp_heatmap(self, fig, ax, rebD, dataset_name, clip):
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

    def freq_vs_time(self,rebD,md,figId=2):
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(16,11))

        rateThr2=rebD['rate_thres2']
        clip = clip_rebD_time(rebD, md['plot']['time_rangeLR'])
        print('iTL,R', clip['itL'], clip['itR'])

        highChanCnt = rebD['highChanCnt'][clip['itL']:clip['itR']]
        timeV = clip['timeV']
        Tbin = clip['Tbin']
        tL, tR = clip['tL'], clip['tR']
        nchan = clip['nchan']

        gs = fig.add_gridspec(4, 1, height_ratios=[0.15, 0.15, 0.54, 0.01])

        ax = fig.add_subplot(gs[0, 0])
        ax.bar(timeV+Tbin*.5, highChanCnt, width=Tbin, color='forestgreen', align='center', alpha=0.7)
        tit = (f'num neurons with instantaneous rate > thres={rateThr2:.0f} (Hz), '
               f'nchan={nchan}')
        ax.set(ylabel='num neurons', title=tit)
        ax.set_xlim(tL, tR)
        ax.grid()

        ax = fig.add_subplot(gs[1, 0])
        self.plot_bioexp_sum_rate(ax, rebD, clip)

        ax = fig.add_subplot(gs[2, 0])
        self.plot_bioexp_heatmap(fig, ax, rebD, md['short_name'], clip)

    def _metrics_column(self, bioD, md, col_name):
        cols = md["metrics_curated_columns"]
        metrics = np.asarray(bioD["metrics_curated"])
        col_map = {str(c): i for i, c in enumerate(cols)}
        assert col_name in col_map, (
            f"metrics_curated missing column {col_name!r}; have {cols}"
        )
        return metrics[:, col_map[col_name]].astype(np.float64)

    def neuron_spatial(self, bioD, md, figId=3):
        """MEA layout: loc_x/loc_y scatter colored by firing_rate (metrics_curated)."""
        loc_x = self._metrics_column(bioD, md, "loc_x")
        loc_y = self._metrics_column(bioD, md, "loc_y")
        rate = self._metrics_column(bioD, md, "firing_rate")

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(8, 9))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.06], hspace=0.12)

        ax = fig.add_subplot(gs[0, 0])
        ax.set_facecolor("white")
        sc = ax.scatter(
            loc_x, loc_y, c=rate, s=12, marker="s",
            cmap="Blues", vmin=0.0, vmax=float(np.max(rate)),
            linewidths=0.15, edgecolors="k", alpha=0.95,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("loc_x (um)", fontsize=11)
        ax.set_ylabel("loc_y (um)", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.35)
        tit = f"dataset: {md['short_name']}   n={loc_x.shape[0]} neurons"
        ax.set_title(tit, fontsize=11, pad=8)

        cax = fig.add_subplot(gs[1, 0])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_label("Firing Rate [Hz]", fontsize=11)
        cb.ax.tick_params(labelsize=9)

        fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.10)

    def _hist_panel_title(self, col_name):
        unit = METRICS_HIST_TITLE_UNITS[col_name]
        return f"{col_name} [{unit}]"

    def _fmt_metric_val(self, val):
        av = abs(float(val))
        if av >= 100:
            return f"{val:.0f}"
        if av >= 1:
            return f"{val:.3f}"
        if av >= 0.01:
            return f"{val:.4f}"
        return f"{val:.2e}"

    def metrics_histograms(self, bioD, md, figId=4):
        """1D histograms for curated unit-quality metrics (metrics_curated columns)."""
        nrow, ncol = 3, 7
        n_panels = nrow * ncol
        assert len(METRICS_HIST_COLUMNS) == n_panels - 1, (
            f"expected {n_panels - 1} metric columns, got {len(METRICS_HIST_COLUMNS)}"
        )

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(18, 8))
        n_neur = np.asarray(bioD["metrics_curated"]).shape[0]
        fr_lo, fr_hi = md["data_selector"]["freq_range"]

        for i, col_name in enumerate(METRICS_HIST_COLUMNS):
            ax = fig.add_subplot(nrow, ncol, i + 1)
            vals = self._metrics_column(bioD, md, col_name)
            vals = vals[np.isfinite(vals)]
            assert vals.size > 0, f"no finite values for metrics column {col_name!r}"
            n_bins = min(40, max(10, int(np.sqrt(vals.size))))
            ax.hist(vals, bins=n_bins, color="steelblue", alpha=0.85, edgecolor="white")
            if col_name in METRICS_HIST_LOG_Y:
                ax.set_yscale("log")

            if col_name in METRICS_HIST_XMIN_ZERO:
                ax.set_xlim(left=0.0)

            p16, p50, p84 = np.percentile(vals, [16, 50, 84])
            ylo, yhi = ax.get_ylim()
            y_mark = 0.5 * (ylo + yhi)
            ax.errorbar(
                p50, y_mark,
                xerr=[[p50 - p16], [p84 - p50]],
                fmt="o", color="k", markersize=9,
                capsize=4, capthick=1.2, elinewidth=1.2, zorder=5,
            )
            txt = (
                f"p16={self._fmt_metric_val(p16)}\n"
                f"med={self._fmt_metric_val(p50)}\n"
                f"p84={self._fmt_metric_val(p84)}"
            )
            ax.text(
                0.98, 0.97, txt, transform=ax.transAxes,
                va="top", ha="right", fontsize=11, color="k",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
            )

            title = (
                self._hist_panel_title(col_name)
                if col_name in METRICS_HIST_TITLE_UNITS else col_name
            )
            ax.set_title(title, fontsize=14)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.35)

        fig.add_subplot(nrow, ncol, n_panels).axis("off")

        fig.suptitle(
            f"curated metrics, {md['short_name']},  N={n_neur} neurons, "
            f"{fr_lo:g}< freq <{fr_hi:g} Hz",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=1.62, w_pad=1.62)

    def metrics_correlations(self, bioD, md, figId=5):
        """3x2 panel grid; panel 0: firing_rate vs single_rates scatter."""
        rate_sheet = self._metrics_column(bioD, md, "firing_rate")
        rate_spikes = np.asarray(bioD["single_rates"], dtype=np.float64).ravel()
        assert rate_sheet.shape[0] == rate_spikes.shape[0], (
            "firing_rate and single_rates length mismatch"
        )

        n = rate_sheet.shape[0]
        r = float(np.corrcoef(rate_sheet, rate_spikes)[0, 1])

        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor="white", figsize=(12, 10))
        fr_lo, fr_hi = md["data_selector"]["freq_range"]

        for k in range(6):
            ax = fig.add_subplot(3, 2, k + 1)
            if k == 0:
                ax.scatter(
                    rate_sheet, rate_spikes, s=14, alpha=0.65,
                    color="steelblue", edgecolors="k", linewidths=0.2,
                )
                lo = float(min(rate_sheet.min(), rate_spikes.min()))
                hi = float(max(rate_sheet.max(), rate_spikes.max()))
                pad = 0.05 * max(1e-9, hi - lo)
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                        "k--", lw=0.9, alpha=0.5)
                ax.set_xlabel("firing_rate (metrics_curated)", fontsize=10)
                ax.set_ylabel("single_rates (bioExp)", fontsize=10)
                ax.set_title(f"r={r:.3f},  n={n}", fontsize=11)
                ax.grid(True, alpha=0.35)
            else:
                ax.axis("off")

        fig.suptitle(
            f"metrics correlations, {md['short_name']},  N={n} neurons, "
            f"{fr_lo:g}< freq <{fr_hi:g} Hz",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
