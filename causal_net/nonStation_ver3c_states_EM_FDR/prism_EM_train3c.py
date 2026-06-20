#!/usr/bin/env python3
"""Standalone PRISM-EM training CLI.

The reusable training implementation lives in PrismEM_Workhorse3c.py so the
ordinary trainer and the EM-FDR bagging driver run the same full-fit code.
"""

import argparse
import math
import os
import secrets
from pprint import pprint

import numpy as np

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
from PrismEM_Workhorse3c import (
    add_prism_em_args,
    barrier,
    broadcast_array,
    broadcast_object,
    cleanup_distributed,
    init_distributed,
    is_rank0,
    normalize_delay_args,
    run_full_fit,
    runtime_summary,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="EM training: non-stationary Poisson GLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_prism_em_args(parser, include_time_range=True)
    return parser.parse_args()


def slice_spikes_by_time(spikes, dt, time_range_sec):
    t0_sec, t1_sec = [float(x) for x in time_range_sec]
    if t1_sec < t0_sec:
        t0_sec, t1_sec = t1_sec, t0_sec
    T_raw = spikes.shape[0]
    start_bin = max(0, int(math.floor(t0_sec / dt)))
    end_bin = min(T_raw - 1, int(math.floor(t1_sec / dt)))
    if end_bin <= start_bin:
        raise ValueError("time_range_sec too small; need at least two bins")
    return spikes[start_bin:end_bin + 1], [t0_sec, t1_sec], [start_bin, end_bin]


def main():
    args = normalize_delay_args(parse_args())
    ctx = init_distributed()
    seed_everything(args.seed)

    try:
        inpPath = os.path.join(args.basePath, "spikesData")
        outPath = os.path.join(args.basePath, "prismFit")
        if is_rank0(ctx):
            os.makedirs(outPath, exist_ok=True)
            if args.verb > 0:
                print("Runtime:", runtime_summary(ctx))
            spikesFF = os.path.join(inpPath, f"{args.dataName}.spikes.npz")
            spikeD, spikeMD = read_data_npz(spikesFF, verb=args.verb > 0)
            spikes = np.asarray(spikeD["spikes"])
            single_rates = np.asarray(spikeD["single_rates"])
            if args.verb > 1:
                pprint(spikeMD)
            dt = float(spikeMD["time_step_sec"])
            spikes, time_range_sec, time_range_bins = slice_spikes_by_time(
                spikes, dt, args.time_range_sec
            )
            if args.verb > 0:
                print("\nEM-train args:", vars(args), "\n")
                print(
                    f"N={spikes.shape[1]} EM={args.num_em_iters} M={args.num_states} "
                    f"T={spikes.shape[0]} dt={dt} "
                    f"time=[{time_range_sec[0]:.1f}, {time_range_sec[1]:.1f}]s "
                    f"bins={time_range_bins}"
                )
        else:
            spikeMD = None
            spikes = None
            single_rates = None
            time_range_sec = None
            time_range_bins = None

        spikeMD = broadcast_object(spikeMD, ctx)
        spikes = broadcast_array(spikes, ctx)
        single_rates = broadcast_array(single_rates, ctx)
        time_range_sec = broadcast_object(time_range_sec, ctx)
        time_range_bins = broadcast_object(time_range_bins, ctx)

        if args.fitName is None:
            outF = f"{args.dataName}_{secrets.token_hex(2)}" if is_rank0(ctx) else None
            outF = broadcast_object(outF, ctx)
        else:
            outF = args.fitName

        fitD, fitMD = run_full_fit(
            spikes, spikeMD, single_rates, args, ctx,
            time_range_sec=time_range_sec,
            time_range_bins=time_range_bins,
            fit_name=outF,
            provenance_update={"dataName": args.dataName},
        )

        if is_rank0(ctx):
            outFF = os.path.join(outPath, f"{outF}.prismEM.npz")
            write_data_npz(fitD, outFF, metaD=fitMD, verb=args.verb > 1)
            print(f"\nSaved: {outFF}")
            print(f"  basePath={args.basePath}")
            print(
                f"  ./prism_EM_eval3c.py --basePath $basePath "
                f"--dataName {outF} -p a m n j i\n"
            )
        barrier(ctx)
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
