#!/usr/bin/env python3
"""Shared matplotlib canvas I/O: save PNG/PDF, optional on-screen display.

Always saves figures.  Displays them on screen unless ``noXterm``.

Constructor arguments:
  prjName     file-name stem; canvases are saved as ``<prjName>_a.png`` etc.
  outPath     directory for saved figures (created if missing), default ``out/``
  noXterm     True: save only; False: save and display (default True)
  plotFormat  ``png`` or ``pdf`` (default ``png``)

Typical caller (see ``main()`` below, and ``homing_stationary/eval_tracker.py``):

  1. Subclass ``PlotterBackboneV2``.
  2. Each canvas method takes ``fig_id`` (``"a"``, ``"b"``, ...) matching the
     letter used when that method is called from ``main()``.
  3. The canvas method calls ``smart_append(fig_id)`` then ``self.plt.figure``.
  4. After the selected canvases are built, call ``display_all()`` once.

Run this file as a script to write two demo figures::

    python toolbox/PlotterBackboneV2.py --outPath out/
    # writes  out/plotter_demo_a.png  and  plotter_demo_b.png
"""

import argparse
import math
import os


class PlotterBackboneV2(object):
    """Shared matplotlib canvas I/O."""

    def __init__(self, prjName, outPath="out/", noXterm=True, plotFormat="png"):
        import matplotlib as mpl
        if noXterm:
            mpl.use("Agg")
        else:
            mpl.use("TkAgg")
        import matplotlib.pyplot as plt

        plt.close("all")
        self.plt = plt
        self.figL = []
        self.jobName = prjName
        self.noXterm = noXterm
        self.plotFormat = plotFormat
        if self.plotFormat not in ("png", "pdf"):
            raise ValueError("plotFormat must be 'png' or 'pdf'")
        self.outPath = outPath.rstrip("/") + "/"
        os.makedirs(self.outPath, exist_ok=True)

    def figId2name(self, fig_id):
        return f"{self.jobName}_{fig_id}"

    def smart_append(self, fig_id):
        if fig_id in self.figL:
            raise ValueError(f"figure id {fig_id!r} is already in use")
        self.figL.append(fig_id)
        return fig_id

    def display_all(self):
        if not self.figL:
            print("display_all - nothing to plot, quit")
            return
        extension = "." + self.plotFormat
        for fig_id in self.figL:
            fig = self.plt.figure(fig_id)
            fig_name = self.outPath + self.figId2name(fig_id) + extension
            print("Graphics  display ", fig_name)
            fig.savefig(fig_name)
        if not self.noXterm:
            self.plt.show()


class _ExamplePlotter(PlotterBackboneV2):
    """Minimal subclass showing the canvas-letter convention used by callers."""

    def sine_wave(self, fig_id="a"):
        fig_id = self.smart_append(fig_id)
        fig = self.plt.figure(fig_id, facecolor="white", figsize=(6, 4))
        time_s = [0.1 * i for i in range(100)]
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(time_s, [math.sin(t) for t in time_s], color="tab:blue")
        ax.set(title="example a: sine wave", xlabel="t", ylabel="sin(t)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

    def histogram(self, fig_id="b"):
        fig_id = self.smart_append(fig_id)
        fig = self.plt.figure(fig_id, facecolor="white", figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(
            [math.sin(0.1 * i) for i in range(100)],
            bins=20,
            color="tab:orange",
            alpha=0.85,
        )
        ax.set(title="example b: sine histogram", xlabel="sin(t)", ylabel="count")
        ax.grid(True, alpha=0.25, axis="y")
        fig.tight_layout()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo for PlotterBackboneV2: save PNG/PDF, optional display.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prjName", default="plotter_demo", help="saved-file stem")
    parser.add_argument("--outPath", default="out/", help="directory for saved plots")
    parser.add_argument(
        "--plotFormat", choices=["png", "pdf"], default="png",
        help="saved figure format",
    )
    parser.add_argument(
        "-p", "--showPlots", type=str, nargs="+", default="ab",
        help="plot letters: a=sine wave, b=histogram",
    )
    parser.add_argument(
        "-X", "--noXterm", action="store_true", default=True,
        help="save plots without displaying them",
    )
    parser.add_argument(
        "--xterm", dest="noXterm", action="store_false",
        help="save and display plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.showPlots = "".join(args.showPlots)
    plotter = _ExamplePlotter(
        prjName=args.prjName,
        outPath=args.outPath,
        noXterm=args.noXterm,
        plotFormat=args.plotFormat,
    )
    # Letters passed as fig_id become the filename suffix: <prjName>_a.png, _b.png
    if "a" in args.showPlots:
        plotter.sine_wave(fig_id="a")
    if "b" in args.showPlots:
        plotter.histogram(fig_id="b")
    plotter.display_all()


if __name__ == "__main__":
    main()
