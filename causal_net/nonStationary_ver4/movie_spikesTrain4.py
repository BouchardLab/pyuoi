#!/usr/bin/env python3
"""
Spike train movie generator — uses the same ``--basePath`` / ``--dataName`` data layout as view_spikesTrain4.py.

Loads:
  <basePath>/truthDale/<dataName>.simTruth.npz  → node_positions, node_is_inhibitory
  <basePath>/truthDale/<dataName>.spikes.npz    → spikes (T, N) uint8

via toolbox.Util_NumpyIO.read_data_npz (same as view_spikesTrain4.py).

MP4 requires ``imageio-ffmpeg`` (see ``container/ubu24-python.dockerfile``).
Also writes ``<mp4_basename>_frame0.png`` (first movie frame, same DPI as MP4).
With ``--stillFrames N`` (default 20), also writes ``<mp4_basename>_still.html``:
first N frames as embedded PNGs; **click the image** to advance (no autoplay).
"""

__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import argparse
import base64
import io
import json
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
import imageio_ffmpeg


def get_parser():
    parser = argparse.ArgumentParser(description="Generate spike movie from Dale simTruth + spikes (same paths as view_spikesTrain4)")
    parser.add_argument("-v", "--verbosity", type=int, help="increase output verbosity", default=1, dest="verb")
    parser.add_argument("--basePath", default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="head dir for input data")
    parser.add_argument("--dataName", default=None, help="simulated Dale network base name")
    parser.add_argument("-T", "--time_range_sec", default=[0.0, 10.0], nargs=2, type=float, help="time range (seconds) to render")
    parser.add_argument("-r", "--time_rebin2", default=1, type=int, help="rebin factor on time axis (same meaning as view_spikesTrain4)")
    # Movie-only
    parser.add_argument("--output", type=str, default=None, help="Output path; must end with .mp4 (default: <outPath>/<dataName>_spike_movie.mp4)")
    parser.add_argument("--tau", type=int, default=20, help="Frames for spike glow decay (default: 20)")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second (default: 60)")
    parser.add_argument("--flushSize", type=float, default=0.05, help="Max spike-flash circle radius at g=1 (plot units); 0=auto 0.45×min inter-neuron distance (default: 0)")
    parser.add_argument("--marker_size", type=float, default=0.02, help="Neuron marker size scale; 0=auto (default: 0)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for output (default: 150)")
    parser.add_argument("--stillFrames", type=int, default=40, help="Also write <mp4_basename>_still.html: first N frames, click image to advance (0=skip)")

    args = parser.parse_args()
    args.inpPath = os.path.join(args.basePath, "truthDale")
    args.outPath = os.path.join(args.basePath, "plots")

    print("myArg-program:", parser.prog)
    for arg in vars(args):
        print("myArg:", arg, getattr(args, arg))

    if args.time_range_sec is not None:
        assert args.time_range_sec[0] < args.time_range_sec[1]
    assert os.path.exists(args.basePath), args.basePath
    assert os.path.exists(args.inpPath), f"missing inpPath: {args.inpPath}"
    if args.dataName is None:
        print("ERROR: --dataName is required", file=sys.stderr)
        sys.exit(1)
    if args.flushSize < 0:
        print("ERROR: --flushSize must be >= 0 (0 = auto radius)", file=sys.stderr)
        sys.exit(1)
    if args.stillFrames < 0:
        print("ERROR: --stillFrames must be >= 0", file=sys.stderr)
        sys.exit(1)
    _out = args.output if args.output else os.path.join(
        args.outPath, f"{args.dataName}_spike_movie.mp4"
    )
    if not _out.endswith(".mp4"):
        print("ERROR: --output must be a .mp4 path", file=sys.stderr)
        sys.exit(1)
    args.movie_output_path = _out
    return args


def rebin_spike_rates(spikeYield, md, tReb2):
    """Rebin 2D spike train (time x neurons); copied from view_spikesTrain4.py."""
    assert spikeYield.ndim == 2, f"Expected 2D spike train, got shape={spikeYield.shape}"
    assert tReb2 < 101

    ntime, nchan = spikeYield.shape
    if ntime % tReb2 != 0:
        ntime_c = ntime - (ntime % tReb2)
        spikeYield = spikeYield[:ntime_c]
    spikeYieldR = np.sum(spikeYield.reshape(-1, tReb2, nchan), axis=1)
    time_step = md["time_step_sec"]
    time_step2 = time_step * tReb2
    rate2D = spikeYieldR / time_step2
    pop_spike_count = np.sum(spikeYieldR, axis=1)
    pop_rate_hz = pop_spike_count / time_step2
    rebD = {"time_step2": time_step2}
    rebD["rate2D"] = rate2D
    rebD["pop_spike_count"] = pop_spike_count
    rebD["pop_rate_hz"] = pop_rate_hz
    ntime //= tReb2
    rebD["timeV"] = np.linspace(0, (ntime - 1) * time_step2, ntime)
    rebD["spike_counts"] = spikeYieldR
    print("rebinned: nchan=%d  dt=%.3f sec  rebin=%d" % (nchan, time_step2, tReb2))
    return rebD


def movie_title_header(spikeMD, trueMD, nchan, tbin_sec):
    """Metadata line matching PlotterSpikesTrain.freq_vs_time (dataset, R, nchan, Tbin, …)."""
    dale = trueMD["dale_conf"]
    R_sel = float(dale["spectral_radius"])
    sn = spikeMD["short_name"]
    tit0 = "dataset: %s  R=%.3f    nchan=%d  Tbin=%.2f sec" % (sn, R_sel, nchan, tbin_sec)
    if spikeMD["spike_model"] == "B":
        tit0 += "  Q=%.4g  tau/sec=%.4g" % (
            float(spikeMD["mem_Q"]),
            float(spikeMD["mem_tau"]),
        )
    tit0 += "  placement HxL=(%gx%g)" % (
        float(spikeMD["placement_H"]),
        float(spikeMD["placement_L"]),
    )
    return tit0


def write_still_click_html(fig, update, n_frames, dpi, out_path):
    """Write self-contained HTML: embedded PNGs, click image to cycle frames (no autoplay)."""
    chunks = []
    for i in range(n_frames):
        update(i)
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches=None)
        chunks.append(base64.standard_b64encode(buf.getvalue()).decode("ascii"))
    b64_json = json.dumps(chunks)
    html = (
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"/><title>still frames</title>\n"
        "<style>\n"
        "body { font-family: sans-serif; text-align: center; margin: 16px; }\n"
        "img { max-width: 100%; height: auto; cursor: pointer; border: 1px solid #ccc; }\n"
        "#cap { margin: 12px; color: #444; }\n"
        "</style></head><body>\n"
        "<p id=\"cap\"></p>\n"
        "<img id=\"img\" alt=\"frame\" onclick=\"next()\" />\n"
        "<script>\n"
        "var b64 = "
        + b64_json
        + ";\n"
        "var i = 0;\n"
        "var N = b64.length;\n"
        "function cap() {\n"
        "  document.getElementById('cap').textContent =\n"
        "    'Frame ' + (i + 1) + ' / ' + N + ' — click image to advance (wraps)';\n"
        "}\n"
        "function show() {\n"
        "  document.getElementById('img').src = 'data:image/png;base64,' + b64[i];\n"
        "  cap();\n"
        "}\n"
        "function next() { i = (i + 1) % N; show(); }\n"
        "show();\n"
        "</script>\n</body></html>\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    from toolbox.Util_NumpyIO import read_data_npz

    args = get_parser()
    np.set_printoptions(precision=3)

    spikesFF = os.path.join(args.inpPath, f"{args.dataName}.spikes.npz")
    truthFF = os.path.join(args.inpPath, f"{args.dataName}.simTruth.npz")
    assert os.path.isfile(spikesFF), f"missing {spikesFF}"
    assert os.path.isfile(truthFF), f"missing {truthFF}"

    spikeD, spikeMD = read_data_npz(spikesFF, verb=args.verb > 0)
    trueD, trueMD = read_data_npz(truthFF, verb=args.verb > 0)
    if args.verb > 1:
        from pprint import pprint

        print("\nspikeMD:"); pprint(spikeMD)
        print("\ntrueMD:"); pprint(trueMD)

    positions = np.asarray(trueD["node_positions"])
    is_inhib = np.asarray(trueD["node_is_inhibitory"])
    spikes = np.asarray(spikeD["spikes"])
    if spikes.ndim != 2:
        raise ValueError(
            "spikes must have shape (T, N); got %s" % (spikes.shape,)
        )

    dt = float(spikeMD["time_step_sec"])
    ntime, nchan = spikes.shape
    if positions.shape[0] != nchan or is_inhib.shape[0] != nchan:
        raise ValueError(
            "N mismatch: spikes N=%d vs positions %s vs inhib %s"
            % (nchan, positions.shape, is_inhib.shape)
        )

    t0s, t1s = float(args.time_range_sec[0]), float(args.time_range_sec[1])
    i0 = max(0, int(np.floor(t0s / dt)))
    i1 = min(ntime, int(np.ceil(t1s / dt)))
    if i0 >= i1:
        raise ValueError("empty time window: indices [%d,%d) from time_range_sec" % (i0, i1))
    spikes = spikes[i0:i1, :].astype(np.float64, copy=False)
    t_off_sec = float(i0) * dt

    t_reb = int(args.time_rebin2)
    if t_reb < 1:
        raise ValueError("time_rebin2 must be >= 1")
    if t_reb > 1:
        rebD = rebin_spike_rates(spikes, spikeMD, t_reb)
        spikes = rebD["spike_counts"].astype(np.float64, copy=False)
        frame_dt_sec = float(rebD["time_step2"])
    else:
        frame_dt_sec = dt

    title_header = movie_title_header(spikeMD, trueMD, nchan, frame_dt_sec)

    # ---- Load data (movie core) ----
    N = positions.shape[0]
    T_total = spikes.shape[0]

    t_start = 0
    t_end = T_total
    T_render = t_end - t_start

    tau = args.tau
    print(f"Neurons: {N}, frames: {T_total} (time slice + rebin), frame_dt={frame_dt_sec:.4g}s")
    print(f"Rendering frames {t_start} to {t_end-1} ({T_render} frames)")
    print(f"Tau (decay frames): {tau}, FPS: {args.fps}")

    x = positions[:, 0]
    y = positions[:, 1]

    if N > 1:
        from scipy.spatial import distance as sp_dist

        dists = sp_dist.pdist(positions)
        min_dist = float(np.min(dists))
    else:
        min_dist = 1.0

    if float(args.flushSize) > 0:
        max_circle_r = float(args.flushSize)
    else:
        max_circle_r = min_dist * 0.45

    # Axis span is L+2*flush and H+2*flush (one flush radius per side around [0,L]×[0,H]).
    pl = float(spikeMD["placement_L"])
    ph = float(spikeMD["placement_H"])
    pad = max_circle_r
    xlim_lo, xlim_hi = -pad, pl + pad
    ylim_lo, ylim_hi = -pad, ph + pad
    w_data = xlim_hi - xlim_lo
    h_data = ylim_hi - ylim_lo
    if args.verb > 0:
        print(
            "Axis limits: placement [−r,L+r]×[−r,H+r]  (window %.4g × %.4g, pad=flushSize=%.4g)"
            % (w_data, h_data, pad)
        )

    marker_size = args.marker_size if args.marker_size > 0 else min_dist * 0.18

    print(f"Min neuron distance: {min_dist:.4f}")
    print(
        f"Marker size: {marker_size:.4f}, max flash radius (flushSize): {max_circle_r:.4f}"
        + (
            f"  [explicit --flushSize {args.flushSize}]"
            if float(args.flushSize) > 0
            else "  [auto from min_dist]"
        )
    )

    print("Precomputing glow intensities...")
    glow = np.zeros((T_render, N), dtype=np.float32)
    time_since_spike = np.full(N, tau + 1, dtype=np.int32)

    pre_start = max(0, t_start - tau)
    for t in range(pre_start, t_start):
        firing = spikes[t] > 0
        time_since_spike[firing] = 0
        time_since_spike[~firing] += 1

    for frame in range(T_render):
        t = t_start + frame
        firing = spikes[t] > 0
        time_since_spike[firing] = 0
        active = time_since_spike <= tau
        glow[frame, active] = 1.0 - time_since_spike[active] / tau
        time_since_spike[~firing] += 1

    fig, ax = plt.subplots(1, 1, figsize=(16.0, 5.0))
    ax.set_xlim(xlim_lo, xlim_hi)
    ax.set_ylim(ylim_lo, ylim_hi)
    ax.set_facecolor("white")
    ax.set_xlabel("x", labelpad=12)
    ax.set_ylabel("y", labelpad=12)
    ax.tick_params(axis="x", which="major", pad=8)
    ax.tick_params(axis="y", which="major", pad=6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    title_text = ax.set_title(
        title_header + f"\nt = {t_off_sec:.3f}s  (frame 0)",
        fontsize=8,
        pad=14,
    )

    exc_mask = is_inhib == 0
    inh_mask = is_inhib == 1

    ax.scatter(
        x[exc_mask],
        y[exc_mask],
        marker="^",
        s=marker_size * 800,
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )
    ax.scatter(
        x[inh_mask],
        y[inh_mask],
        marker="s",
        s=marker_size * 600,
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )

    # Shrink default side margins so the axes use more of the frame (wide L×H plots were ~70% width).
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.13, top=0.82)

    # Ellipse in data coords with height/width = sx/sy so the patch is a circle in pixels (non-equal data aspect).
    fig.set_dpi(args.dpi)
    fig.canvas.draw()
    _bbox = ax.get_window_extent()
    _dx = xlim_hi - xlim_lo
    _dy = ylim_hi - ylim_lo
    flush_sx_over_sy = (_bbox.width / _dx) / (_bbox.height / _dy)

    circles = []
    for i in range(N):
        c = Ellipse(
            (x[i], y[i]),
            0.0,
            0.0,
            angle=0.0,
            facecolor="green",
            edgecolor="none",
            linewidth=0,
            alpha=0.0,
            zorder=5,
        )
        ax.add_patch(c)
        circles.append(c)

    if args.verb > 0:
        fw, fh = fig.get_size_inches()
        print(
            "figure: %.2f×%.2f in (data window %.4g×%.4g)"
            % (fw, fh, w_data, h_data)
        )

    _progress_seen = set()

    def update(frame):
        if frame % 200 == 0 and frame not in _progress_seen:
            _progress_seen.add(frame)
            print(f"  Rendering frame {frame}/{T_render}...")
        t_sec = t_off_sec + (t_start + frame) * frame_dt_sec
        title_text.set_text(title_header + f"\nt = {t_sec:.3f}s  (frame {frame})")
        for i in range(N):
            g = glow[frame, i]
            if g > 0:
                r = max_circle_r * g
                circles[i].set_width(2.0 * r)
                circles[i].set_height(2.0 * r * flush_sx_over_sy)
                circles[i].set_alpha(0.7 * g)
                circles[i].set_edgecolor("none")
                circles[i].set_linewidth(0)
                circles[i].set_visible(True)
            else:
                circles[i].set_visible(False)
        return circles

    output = args.movie_output_path
    _out_dir = os.path.dirname(os.path.abspath(output)) 
    assert os.path.isdir(_out_dir), "output directory must exist: %r" % _out_dir

    print(f"Creating animation with {T_render} frames -> {output}")
    # blit=True breaks growing flush Ellipse patches (bbox stays ~0); green flashes vanish.
    anim = animation.FuncAnimation(
        fig, update, frames=T_render, interval=1000 / args.fps, blit=False
    )

    png_frame0 = os.path.splitext(output)[0] + "_frame0.png"
    update(0)
    fig.canvas.draw()
    fig.savefig(png_frame0, format="png", dpi=args.dpi, bbox_inches=None)
    print(f"Saved first frame to {png_frame0}")

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    if not ff or not os.path.isfile(ff) or not os.access(ff, os.X_OK):
        raise RuntimeError(
            "imageio_ffmpeg.get_ffmpeg_exe() is missing or not executable: %r" % (ff,)
        )
    plt.rcParams["animation.ffmpeg_path"] = ff
    if args.verb > 0:
        print(f"Using ffmpeg: {ff}")

    writer = animation.FFMpegWriter(
        fps=args.fps, bitrate=2000, extra_args=["-pix_fmt", "yuv420p"]
    )
    anim.save(output, writer=writer, dpi=args.dpi)
    print(f"Saved movie to {output}")

    if args.stillFrames > 0:
        n_still = min(int(args.stillFrames), T_render)
        html_out = os.path.splitext(output)[0] + "_still.html"
        print(f"Creating still viewer ({n_still} frames, click to advance) -> {html_out}")
        write_still_click_html(fig, update, n_still, args.dpi, html_out)
        print(f"Saved still HTML to {html_out}")

    plt.close(fig)
    print("M:done")


if __name__ == "__main__":
    main()
