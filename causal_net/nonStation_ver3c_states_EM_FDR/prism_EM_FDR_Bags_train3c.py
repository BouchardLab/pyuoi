#!/usr/bin/env python3
"""Run one EM-FDR-Bagging Stage (a) job.

One invocation assembles one block-resampled bag, runs a real PRISM-EM full
fit, runs P circular-shift locked M-step null refits, and writes one NPZ file.
"""

import argparse
import hashlib
import math
import os
import re
import time
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
    init_null_params,
    is_rank0,
    normalize_delay_args,
    run_full_fit,
    run_locked_mstep,
    runtime_summary,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PRISM-EM FDR bagging Stage (a): one bag plus scrambles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_prism_em_args(parser, include_time_range=False)

    g = parser.add_argument_group("FDR bagging")
    g.add_argument("--bag_idx", type=int, required=True,
                   help="Bag index used in the output filename and RNG stream")
    g.add_argument("--bagsTag", default=None,
                   help="Tag appended to dataName in bag output names; default/None derives a 4-char hash")
    g.add_argument("--fdr_out_dir", type=str, default=None,
                   help="Output directory; default: <basePath>/prismFDR")
    g.add_argument("--time_range_sec", nargs=2, type=float, required=True,
                   help="Time range [start, end] in seconds for block starts")
    g.add_argument("--num_blocks", type=int, default=10,
                   help="Number of contiguous blocks in this bag")
    g.add_argument("--block_len_sec", type=float, default=60.0,
                   help="Contiguous block length in seconds")
    g.add_argument("--min_block_start_sep_sec", type=float, default=1.0,
                   help="Minimum separation between block starts")
    g.add_argument("--max_block_draw_trials", type=int, default=30,
                   help="Trials to satisfy block-start separation before accepting next draw")
    g.add_argument("--num_scrambles", type=int, default=4,
                   help="Number of circular-shift null refits")
    g.add_argument("--min_roll_shift_sec", type=float, default=2.0,
                   help="Two-sided exclusion-zone half-width for circular shifts")
    g.add_argument("--null_A_init", choices=["scrambled_data", "rand"],
                   default="scrambled_data",
                   help="Fresh A initialization mode for each null refit")
    g.add_argument("--null_B_init", choices=["scrambled_data", "rand"],
                   default="scrambled_data",
                   help="Fresh B initialization mode for each null refit")
    return parser.parse_args()


def resolve_bags_tag(args):
    """Return explicit bagsTag or a deterministic 4-char tag shared by all bags."""
    tag = getattr(args, "bagsTag", None)
    if tag is not None:
        tag = str(tag).strip()
    if tag and tag.lower() != "none":
        if not re.fullmatch(r"[A-Za-z0-9]+", tag):
            raise ValueError(f"--bagsTag must be alphanumeric, got {tag!r}")
        return tag

    tag_items = []
    for key in sorted(vars(args)):
        if key in {"bag_idx", "bagsTag", "basePath", "fdr_out_dir", "fitName", "verb"}:
            continue
        val = getattr(args, key)
        if isinstance(val, (list, tuple)):
            val = ",".join(str(x) for x in val)
        tag_items.append(f"{key}={val}")
    payload = "|".join(tag_items).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:4]


def time_range_bins(time_range_sec, dt, T_raw, block_len_bins):
    t0, t1 = [float(x) for x in time_range_sec]
    if t1 < t0:
        t0, t1 = t1, t0
    src0 = max(0, int(math.floor(t0 / dt)))
    src1_excl = min(int(T_raw), int(math.floor(t1 / dt)))
    max_start = src1_excl - int(block_len_bins)
    if max_start < src0:
        raise ValueError(
            "time_range_sec is too short for the requested block_len_sec: "
            f"window bins [{src0}, {src1_excl}), block_len_bins={block_len_bins}"
        )
    return src0, src1_excl, max_start


def draw_block_starts(rng, src0, max_start, num_blocks, min_sep_bins, max_trials):
    starts = []
    violations = []

    def draw_one():
        return int(rng.integers(src0, max_start + 1))

    for ib in range(int(num_blocks)):
        if not starts:
            starts.append(draw_one())
            continue

        accepted = None
        for _ in range(max(0, int(max_trials))):
            cand = draw_one()
            if all(abs(cand - old) >= int(min_sep_bins) for old in starts):
                accepted = cand
                break

        if accepted is None:
            accepted = draw_one()
            min_sep_seen = min(abs(accepted - old) for old in starts)
            if min_sep_seen < int(min_sep_bins):
                violations.append((ib, accepted, int(min_sep_seen)))
        starts.append(accepted)

    return np.asarray(starts, dtype=np.int64), np.asarray(violations, dtype=np.int64).reshape(-1, 3)


def assemble_bag(spikes, dt, args, rng):
    spikes = np.asarray(spikes)
    T_raw, _ = spikes.shape
    block_len_bins = int(round(float(args.block_len_sec) / float(dt)))
    if block_len_bins < 2:
        raise ValueError("block_len_sec must span at least two time bins")
    min_sep_bins = int(round(float(args.min_block_start_sep_sec) / float(dt)))
    src0, src1_excl, max_start = time_range_bins(
        args.time_range_sec, dt, T_raw, block_len_bins
    )
    if int(args.bag_idx) == 0:
        starts = src0 + np.arange(int(args.num_blocks), dtype=np.int64) * block_len_bins
        last_start = int(starts[-1])
        max_start_timeline = min(int(T_raw) - int(block_len_bins), int(src1_excl) - int(block_len_bins))
        if last_start > max_start_timeline:
            required_bins = int(args.num_blocks) * int(block_len_bins)
            available_bins = int(src1_excl) - int(src0)
            required_sec = required_bins * float(dt)
            available_sec = available_bins * float(dt)
            raise ValueError(
                "bag_idx=0 chronological blocks do not fit in the input timeline: "
                f"requires at least {required_sec:g} sec "
                f"({required_bins} bins = num_blocks {int(args.num_blocks)} "
                f"* block_len_sec {float(args.block_len_sec):g}), "
                f"but --time_range_sec provides {available_sec:g} sec "
                f"({available_bins} bins; bins [{src0}, {src1_excl})). "
                f"Increase --time_range_sec end or reduce --num_blocks/--block_len_sec."
            )
        violations = np.zeros((0, 3), dtype=np.int64)
    else:
        starts, violations = draw_block_starts(
            rng, src0, max_start, args.num_blocks, min_sep_bins,
            args.max_block_draw_trials
        )
    blocks = [spikes[s:s + block_len_bins] for s in starts]
    bag = np.concatenate(blocks, axis=0)
    boundary_pair_indices = (
        np.arange(1, int(args.num_blocks), dtype=np.int64) * block_len_bins - 1
    )
    bag_info = {
        "block_start_bins": starts,
        "block_start_sec": starts.astype(np.float64) * float(dt),
        "block_retry_violations": violations,
        "boundary_pair_indices": boundary_pair_indices,
        "time_range_bins": np.asarray([src0, src1_excl], dtype=np.int64),
        "block_len_bins": np.asarray([block_len_bins], dtype=np.int64),
        "min_block_start_sep_bins": np.asarray([min_sep_bins], dtype=np.int64),
    }
    return bag, bag_info


def circular_shift_by_neuron(spikes, shifts):
    spikes = np.asarray(spikes)
    shifts = np.asarray(shifts, dtype=np.int64)
    out = np.empty_like(spikes)
    for n, shift in enumerate(shifts):
        out[:, n] = np.roll(spikes[:, n], int(shift))
    return out


def draw_roll_shifts(rng, T, N, min_shift_bins):
    lo = int(min_shift_bins)
    hi = int(T) - int(min_shift_bins)
    if hi < lo:
        raise ValueError(
            f"Bag length {T} too short for min_roll_shift_bins={min_shift_bins}"
        )
    return rng.integers(lo, hi + 1, size=int(N), dtype=np.int64)


def stack_history(null_histories, key, dtype):
    vals = [np.asarray(h[key], dtype=dtype) for h in null_histories]
    return np.stack(vals, axis=0) if vals else np.zeros((0, 0), dtype=dtype)


def eval_loss_time_on_bag(fitD, fitMD, bag_spikes):
    """Precompute per-time fit-loss diagnostics on the assembled bag timeline."""
    trainMD = fitMD["train"]
    dt = float(trainMD["time_step_sec"])
    eta_clip = float(trainMD["eta_clip"])
    lambda2 = float(trainMD["lambda2"])

    spikes = np.asarray(bag_spikes, dtype=np.float64)
    c_hat = np.asarray(fitD["c_hat"], dtype=np.float64)
    assert c_hat.shape[0] == spikes.shape[0], (
        f"c_hat length {c_hat.shape[0]} does not match bag spikes length {spikes.shape[0]}"
    )

    yp = spikes[:-1]
    yc = spikes[1:]
    c_pairs = c_hat[1:]
    c_prev = c_hat[:-1]

    a_hat = np.asarray(fitD["A_hat"], dtype=np.float64)
    b_hat = np.asarray(fitD["B_hat"], dtype=np.float64)
    if b_hat.ndim == 1:
        b_hat = b_hat[None, :]

    rates = np.asarray(fitD["single_rates"], dtype=np.float64)
    w = 1.0 / np.maximum(rates, 0.1)
    w /= w.mean()

    eta = yp @ a_hat.T + c_pairs @ b_hat
    eta_c = np.minimum(eta, eta_clip)
    lam = np.exp(eta_c) * dt
    log_dt = np.log(dt)
    nll_t = np.sum(w[None, :] * (lam - yc * (eta + log_dt)), axis=1)

    dc = c_pairs - c_prev
    l2_t = lambda2 * np.sum(dc * dc, axis=1)
    return nll_t.astype(np.float32), l2_t.astype(np.float32)


def load_bag_truth(base_path, spikeMD, starts, block_len_bins, verb=1):
    """Load and assemble simulation truth on the same block timeline as the bag."""
    if spikeMD.get("data_type") == "bioExp":
        return None
    prov = spikeMD["provenance"]
    st_name = prov["state_transition_file"]
    truth_f = os.path.join(base_path, "spikesData", f"{st_name}.prismTruth.npz")
    truthD, _ = read_data_npz(truth_f, verb=verb > 1)
    starts = np.asarray(starts, dtype=np.int64)
    block_len_bins = int(block_len_bins)
    out = {}
    for key in ("S_true", "C_true", "S_oracle"):
        if key in truthD:
            arr = np.asarray(truthD[key])
            out[key] = np.concatenate(
                [arr[int(s):int(s) + block_len_bins] for s in starts],
                axis=0,
            )
    return out


def eval_true_state_recovery_on_bag(fitD, fitMD, S_true):
    """Precompute state-recovery table for one assembled simulation bag."""
    s_true = np.asarray(S_true, dtype=np.int64)
    s_hat = np.asarray(fitD["S_hat"], dtype=np.int64)
    s_hat_cl = np.asarray(fitD["S_hat_CL"], dtype=np.float64)
    assert s_true.shape[0] == s_hat.shape[0] == s_hat_cl.shape[0], (
        "S_true/S_hat/S_hat_CL length mismatch on bag timeline"
    )
    n_state = int(fitMD["train"]["num_states"])

    bins_hat = np.bincount(s_hat, minlength=n_state).astype(np.int64)
    enter_hat = np.zeros(n_state, dtype=np.int64)
    if s_hat.shape[0] > 1:
        enter_idx = np.where(s_hat[1:] != s_hat[:-1])[0] + 1
        if enter_idx.size > 0:
            enter_hat = np.bincount(s_hat[enter_idx], minlength=n_state).astype(np.int64)

    state_acc_cl = []
    for m in range(n_state):
        mask = s_hat == m
        if int(mask.sum()) > 0:
            acc_m = float((s_true[mask] == m).mean())
            cl_m = float(s_hat_cl[mask].mean())
        else:
            acc_m = float("nan")
            cl_m = float("nan")
        state_acc_cl.append([acc_m, cl_m, int(bins_hat[m]), int(enter_hat[m])])

    return {
        "avg_acc": float((s_true == s_hat).mean()),
        "state_acc_cl": state_acc_cl,
    }


def main():
    job_t0 = time.perf_counter()
    args = normalize_delay_args(parse_args())
    bags_tag = resolve_bags_tag(args)
    ctx = init_distributed()
    seed_base = int(args.seed) + int(args.bag_idx) * 1000003
    seed_everything(seed_base)

    try:
        out_dir = args.fdr_out_dir
        if out_dir is None:
            out_dir = os.path.join(args.basePath, "prismFDR")

        if is_rank0(ctx):
            os.makedirs(out_dir, exist_ok=True)
            if args.verb > 0:
                print("Runtime:", runtime_summary(ctx))
            inpF = os.path.join(args.basePath, "spikesData", f"{args.dataName}.spikes.npz")
            spikeD, spikeMD = read_data_npz(inpF, verb=args.verb > 0)
            source_spikes = np.asarray(spikeD["spikes"])
            if args.verb > 1:
                pprint(spikeMD)
            dt = float(spikeMD["time_step_sec"])
            rng = np.random.default_rng(seed_base)
            bag_spikes, bag_info = assemble_bag(source_spikes, dt, args, rng)
            single_rates = bag_spikes.mean(axis=0).astype(np.float64) / dt
            if args.verb > 0:
                print("\nFDR-bag args:", vars(args), "\n")
                print(
                    f"bagsTag={bags_tag} bag_idx={args.bag_idx} T_b={bag_spikes.shape[0]} "
                    f"N={bag_spikes.shape[1]} blocks={args.num_blocks} "
                    f"scrambles={args.num_scrambles}"
                )
        else:
            spikeMD = None
            bag_spikes = None
            single_rates = None
            bag_info = None

        spikeMD = broadcast_object(spikeMD, ctx)
        bag_spikes = broadcast_array(bag_spikes, ctx)
        single_rates = broadcast_array(single_rates, ctx)
        bag_info = broadcast_object(bag_info, ctx)

        dt = float(spikeMD["time_step_sec"])
        T_b, N = bag_spikes.shape
        if int(args.bag_idx) == 0:
            b0 = int(bag_info["block_start_bins"][0])
            b1 = b0 + int(T_b) - 1
            time_range_bins = [b0, b1]
            time_range_sec = [float(b0 * dt), float((b1 + 1) * dt)]
        else:
            time_range_sec = [0.0, float(T_b * dt)]
            time_range_bins = [0, T_b - 1]
        out_data_name = f"{args.dataName}_{bags_tag}"
        out_stem = f"{out_data_name}.bag{int(args.bag_idx):03d}"

        real_t0 = time.perf_counter()
        fitD, fitMD = run_full_fit(
            bag_spikes, spikeMD, single_rates, args, ctx,
            time_range_sec=time_range_sec,
            time_range_bins=time_range_bins,
            fit_name=out_stem,
            provenance_update={
                "dataName": args.dataName,
                "bagsFDR_stageA_dataName": out_data_name,
                "bagsFDR_stageA_file": out_stem,
            },
        )
        if is_rank0(ctx) and args.verb > 0:
            print(f"time-ordered fit finished: elapsed={time.perf_counter() - real_t0:.1f}s")

        if is_rank0(ctx):
            rng = np.random.default_rng(seed_base + 17)
            A_null = []
            B_null = []
            roll_shifts = []
            null_histories = []
            null_A_init_md = []
            null_B_init_md = []
        else:
            rng = None

        min_roll_shift_bins = int(round(float(args.min_roll_shift_sec) / dt))
        S_lock = np.asarray(fitD["S_hat"], dtype=np.int64)

        for p in range(int(args.num_scrambles)):
            null_t0 = time.perf_counter()
            seed_everything(seed_base + 1009 * (p + 1))
            if is_rank0(ctx):
                shifts = draw_roll_shifts(rng, T_b, N, min_roll_shift_bins)
            else:
                shifts = None
            shifts = broadcast_array(shifts, ctx)
            scrambled = circular_shift_by_neuron(bag_spikes, shifts)

            A_init, B_init, A_md, B_md = init_null_params(
                scrambled, spikeMD, args, ctx, args.null_A_init, args.null_B_init
            )
            A_p, B_p, hist_p = run_locked_mstep(
                scrambled[:-1].astype(np.float32),
                scrambled[1:].astype(np.float32),
                S_lock,
                A_init,
                B_init,
                spikeMD,
                args,
                ctx,
            )

            if is_rank0(ctx):
                A_null.append(A_p.astype(np.float32))
                B_null.append(B_p.astype(np.float32))
                roll_shifts.append(np.asarray(shifts, dtype=np.int64))
                null_histories.append(hist_p)
                null_A_init_md.append(A_md)
                null_B_init_md.append(B_md)
                if args.verb > 0:
                    nz = int(hist_p["nz_edges_epoch"][-1]) if hist_p["nz_edges_epoch"].size else -1
                    rho = float(hist_p["rho_epoch"][-1]) if hist_p["rho_epoch"].size else float("nan")
                    n_epoch = int(hist_p["m_loss_epoch"].size)
                    ela = time.perf_counter() - job_t0
                    print(
                        f"null {p + 1}/{args.num_scrambles} finished: "
                        f"elapsed={ela:.1f}s locked_m_epochs={n_epoch} "
                        f"rho={rho:.4f} nz={nz}"
                    )

        if is_rank0(ctx):
            outD = dict(fitD)
            loss_nll_time, loss_l2_time = eval_loss_time_on_bag(fitD, fitMD, bag_spikes)
            outD.update({
                "loss_nll_time": loss_nll_time,
                "loss_l2_time": loss_l2_time,
                "A_null": np.stack(A_null, axis=0).astype(np.float32),
                "B_null": np.stack(B_null, axis=0).astype(np.float32),
                "roll_shifts_bin": np.stack(roll_shifts, axis=0).astype(np.int64),
                "block_start_bins": bag_info["block_start_bins"],
                "block_start_sec": bag_info["block_start_sec"],
                "block_retry_violations": bag_info["block_retry_violations"],
                "boundary_pair_indices": bag_info["boundary_pair_indices"],
                "time_range_bins": bag_info["time_range_bins"],
                "block_len_bins": bag_info["block_len_bins"],
                "min_block_start_sep_bins": bag_info["min_block_start_sep_bins"],
                "null_m_loss_epoch": stack_history(null_histories, "m_loss_epoch", np.float64),
                "null_m_nll_epoch": stack_history(null_histories, "m_nll_epoch", np.float64),
                "null_m_l1_epoch": stack_history(null_histories, "m_l1_epoch", np.float64),
                "null_rho_epoch": stack_history(null_histories, "rho_epoch", np.float64),
                "null_nz_edges_epoch": stack_history(null_histories, "nz_edges_epoch", np.int64),
                "null_learning_rates": stack_history(null_histories, "learning_rates", np.float64),
            })

            bag_truth = load_bag_truth(
                args.basePath,
                spikeMD,
                bag_info["block_start_bins"],
                int(bag_info["block_len_bins"][0]),
                verb=args.verb,
            )
            if bag_truth is not None:
                for key, val in bag_truth.items():
                    outD[key] = val.astype(np.float32) if key == "C_true" else val.astype(np.int32)
                if "S_true" in bag_truth:
                    fitMD["states_recovery_eval"].update(
                        eval_true_state_recovery_on_bag(fitD, fitMD, bag_truth["S_true"])
                    )

            real_fit_md = {
                "train": fitMD["train"],
                "states_recovery_eval": fitMD["states_recovery_eval"],
                "init_A": fitMD["init_A"],
                "init_state": fitMD["init_state"],
                "init_B": fitMD["init_B"],
            }
            outMD = {
                k: v for k, v in fitMD.items()
                if k not in real_fit_md
            }
            outMD["fit_type"] = "prismEM_FDRbags_stageA"
            outMD["bagsFDR_stageA"] = {
                "program": "prism_EM_FDR_Bags_train3c.py",
                "dataName": args.dataName,
                "bagsTag": bags_tag,
                "output_dataName": out_data_name,
                "output_name": out_stem,
                "bag_idx": int(args.bag_idx),
                "time_range_sec": [float(x) for x in args.time_range_sec],
                "time_range_bins": [int(x) for x in bag_info["time_range_bins"]],
                "num_blocks": int(args.num_blocks),
                "bag0_mode": "chronological" if int(args.bag_idx) == 0 else "random_blocks",
                "block_len_sec": float(args.block_len_sec),
                "block_len_bins": int(bag_info["block_len_bins"][0]),
                "min_block_start_sep_sec": float(args.min_block_start_sep_sec),
                "min_block_start_sep_bins": int(bag_info["min_block_start_sep_bins"][0]),
                "max_block_draw_trials": int(args.max_block_draw_trials),
                "min_roll_shift_sec": float(args.min_roll_shift_sec),
                "min_roll_shift_bins": int(min_roll_shift_bins),
                "real_fit": real_fit_md,
                "null_refits": {
                    "num_scrambles": int(args.num_scrambles),
                    "A_init": args.null_A_init,
                    "B_init": args.null_B_init,
                    "A_init_md": null_A_init_md,
                    "B_init_md": null_B_init_md,
                },
                "saved_bag_spikes": False,
                "saved_bag_truth": bool(bag_truth is not None),
                "output_schema": "real_fit_fields_plus_A_null_B_null",
                "seed_base": int(seed_base),
            }
            outMD["eval_f"] = {
                "loss_nll_time_key": "loss_nll_time",
                "loss_l2_time_key": "loss_l2_time",
            }
            if bag_truth is not None and "S_true" in bag_truth:
                outMD["eval_f"]["acc"] = float(fitMD["states_recovery_eval"]["avg_acc"])

            outF = os.path.join(out_dir, f"{out_stem}.prismFDRbag.npz")
            write_data_npz(outD, outF, metaD=outMD)
            print(f"\nSaved FDR bag: {outF}")
        barrier(ctx)
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
