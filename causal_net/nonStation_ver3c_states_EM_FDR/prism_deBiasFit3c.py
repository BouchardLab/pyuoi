#!/usr/bin/env python3
"""Stage (c) de-biased magnitude refit after EM-FDR bag aggregation."""

import argparse
import copy
import math
import os
import re
import secrets
import sys
import time
from types import SimpleNamespace

import numpy as np

from toolbox.Util_NumpyIOv2 import json_safe_metadata, read_data_npz, write_data_npz
from PrismEM_Workhorse3c import (
    barrier,
    broadcast_array,
    broadcast_object,
    cleanup_distributed,
    init_distributed,
    is_rank0,
    runtime_summary,
    seed_everything,
)
from PrismDeBias_Workhorse3c import run_debias_fit


def print_argv_preflight():
    """Print raw invocation before argparse can exit on malformed CLI values."""
    argv = list(sys.argv)
    rank = os.environ.get("RANK", "0")
    local = os.environ.get("LOCAL_RANK", "0")
    world = os.environ.get("WORLD_SIZE", "1")
    if rank != "0":
        return
    missing_value_flags = []
    value_flags = {
        "--basePath",
        "--fdrFitName",
        "--outFitName",
        "--state_mode",
        "--num_debias_iters",
        "--warmup_locked_epochs",
        "--m_epochs",
        "--batch_size",
        "--lr_mstep",
        "--lr_end_factor",
        "--eps_u",
        "--pgd_iter",
        "--lr_estep",
        "--lambda2",
        "--decode_dwell_sec",
        "--rho_max",
        "--prescale_m_step_4_ArhoMax",
        "--delay_iter_4_ArhoMax",
        "--target_iter_4_ArhoMax",
        "--seed",
        "--progress_every_batches",
        "--progress_every_epochs",
        "--progress_every_pairs",
        "-v",
        "--verb",
    }
    for i, tok in enumerate(argv[1:], start=1):
        if tok in value_flags:
            if i + 1 >= len(argv) or argv[i + 1].startswith("-"):
                missing_value_flags.append(tok)
    print(
        "[prism_deBiasFit3c argv-preflight rank_env=%s local_env=%s/%s pid=%d] argv=%r"
        % (rank, local, world, os.getpid(), argv),
        flush=True,
    )
    print(
        "[prism_deBiasFit3c argv-preflight rank_env=%s] env basePath=%r BASEPATH=%r PWD=%r"
        % (
            rank,
            os.environ.get("basePath"),
            os.environ.get("BASEPATH"),
            os.environ.get("PWD"),
        ),
        flush=True,
    )
    if missing_value_flags:
        print(
            "[prism_deBiasFit3c argv-preflight rank_env=%s] ERROR likely missing value after option(s): %s. "
            "If you used --basePath $basePath, check that shell variable with: echo \"$basePath\""
            % (rank, ", ".join(missing_value_flags)),
            flush=True,
        )


def parse_args():
    print_argv_preflight()
    parser = argparse.ArgumentParser(
        description="Stage (c) de-biased PRISM-EM refit on FDR-selected support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--basePath", required=True,
                        help="Run directory containing spikesData/ and prismFit/")
    parser.add_argument("--fdrFitName", required=True,
                        help="Stage (b) aggregate fit stem in prismFit/")
    parser.add_argument("--outFitName", default=None,
                        help="Stage (c) output fit stem written to prismFit/")

    parser.add_argument("--state_mode", choices=["auto", "locked", "refit"], default="auto",
                        help="State treatment for Stage (c)")
    parser.add_argument("--num_debias_iters", type=int, default=None,
                        help="Outer E/M iterations for state_mode=refit")
    parser.add_argument("--warmup_locked_epochs", type=int, default=0,
                        help="Fixed-state M-step epochs before state refit")

    g = parser.add_argument_group("M-step")
    g.add_argument("--m_epochs", type=int, default=None,
                   help="M-step epochs per de-bias iteration, or total epochs in locked mode")
    g.add_argument("--batch_size", type=int, default=None)
    g.add_argument("--lr_mstep", type=float, default=None)
    g.add_argument("--lr_end_factor", type=float, default=None)
    g.add_argument("--eps_u", type=float, default=1e-4,
                   help="Positive floor for Dale squared-parameter initialization")

    g = parser.add_argument_group("E-step")
    g.add_argument("--pgd_iter", type=int, default=None)
    g.add_argument("--lr_estep", type=float, default=None)
    g.add_argument("--lambda2", type=float, default=None)
    g.add_argument("--decode_dwell_sec", type=float, default=None)

    g = parser.add_argument_group("spectral radius")
    g.add_argument("--rho_max", type=float, default=None)
    g.add_argument("--prescale_m_step_4_ArhoMax", type=int, default=None)
    g.add_argument("--delay_iter_4_ArhoMax", type=int, default=None)
    g.add_argument("--target_iter_4_ArhoMax", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress_every_batches", type=int, default=50,
                        help="Print M-step progress every this many minibatches; 0 disables batch progress")
    parser.add_argument("--progress_every_epochs", type=int, default=10,
                        help="Print M-step progress every this many M-epochs; 0 disables M-epoch progress")
    parser.add_argument("--progress_every_pairs", type=int, default=20000,
                        help="Print E-step progress every this many local lag pairs; 0 disables pair progress")
    parser.add_argument("-v", "--verb", type=int, default=1)
    return parser.parse_args()


def validate_fit_stem(name, arg_name):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(name)):
        raise ValueError(
            "%s must contain only letters, digits, underscores, dots, or hyphens: %r"
            % (arg_name, name)
        )
    if "/" in str(name):
        raise ValueError("%s must be a file stem, not a path: %r" % (arg_name, name))


def source_spike_name(md):
    prov = md["provenance"]
    if md.get("data_type") == "bioExp":
        return prov["experiment_name"]
    return prov["state_transition_file"]


def slice_spikes_by_bins(spikes, time_range_bins):
    b0, b1 = [int(x) for x in time_range_bins]
    if b1 <= b0:
        raise ValueError("Bad time_range_bins: %r" % (time_range_bins,))
    if b0 < 0 or b1 >= spikes.shape[0]:
        raise ValueError(
            "time_range_bins %r outside spike array length %d" % (time_range_bins, spikes.shape[0])
        )
    return np.asarray(spikes[b0:b1 + 1]), [b0, b1]


def as_2d_B(B):
    B = np.asarray(B, dtype=np.float32)
    if B.ndim == 1:
        B = B[None, :]
    return B


def resolve_state_mode(mode, num_states):
    if mode == "auto":
        return "refit" if int(num_states) > 1 else "locked"
    return str(mode)


def metadata_value(train_md, key, default):
    return train_md[key] if key in train_md else default


def effective_args(cli_args, stage_md, train_md):
    num_states = int(train_md["num_states"])
    state_mode = resolve_state_mode(cli_args.state_mode, num_states)
    num_debias_iters = (
        int(cli_args.num_debias_iters)
        if cli_args.num_debias_iters is not None
        else 4
    )
    if state_mode == "locked":
        num_debias_iters = 0
    if num_debias_iters < 0:
        raise ValueError("--num_debias_iters must be non-negative")

    m_epochs = int(
        cli_args.m_epochs
        if cli_args.m_epochs is not None
        else metadata_value(train_md, "m_epochs", 100)
    )
    if m_epochs < 1:
        raise ValueError("--m_epochs must be >= 1")
    warmup = int(cli_args.warmup_locked_epochs)
    if warmup < 0:
        raise ValueError("--warmup_locked_epochs must be >= 0")

    target_rho_iter = (
        int(cli_args.target_iter_4_ArhoMax)
        if cli_args.target_iter_4_ArhoMax is not None
        else max(1, num_debias_iters)
    )
    delay_rho_iter = (
        int(cli_args.delay_iter_4_ArhoMax)
        if cli_args.delay_iter_4_ArhoMax is not None
        else 0
    )
    if target_rho_iter <= delay_rho_iter:
        target_rho_iter = delay_rho_iter + 1

    decode_default = stage_md.get("states_recovery_eval", {}).get("decode_dwell_sec", 0.3)

    vals = {
        "state_mode": state_mode,
        "num_debias_iters": int(num_debias_iters),
        "warmup_locked_epochs": int(warmup),
        "m_epochs": int(m_epochs),
        "batch_size": int(
            cli_args.batch_size
            if cli_args.batch_size is not None
            else metadata_value(train_md, "batch_size", 4096)
        ),
        "lr_mstep": float(
            cli_args.lr_mstep
            if cli_args.lr_mstep is not None
            else metadata_value(train_md, "lr_mstep", 0.003)
        ),
        "lr_end_factor": float(
            cli_args.lr_end_factor
            if cli_args.lr_end_factor is not None
            else metadata_value(train_md, "lr_end_factor", 0.1)
        ),
        "pgd_iter": int(
            cli_args.pgd_iter
            if cli_args.pgd_iter is not None
            else metadata_value(train_md, "pgd_iter", 5)
        ),
        "lr_estep": float(
            cli_args.lr_estep
            if cli_args.lr_estep is not None
            else metadata_value(train_md, "lr_estep", 0.03)
        ),
        "lambda2": float(
            cli_args.lambda2
            if cli_args.lambda2 is not None
            else metadata_value(train_md, "lambda2", 2.0)
        ),
        "rho_max": float(
            cli_args.rho_max
            if cli_args.rho_max is not None
            else metadata_value(train_md, "rho_max", 0.95)
        ),
        "prescale_m_step_4_ArhoMax": int(
            cli_args.prescale_m_step_4_ArhoMax
            if cli_args.prescale_m_step_4_ArhoMax is not None
            else metadata_value(train_md, "prescale_m_step_4_ArhoMax", 120)
        ),
        "delay_iter_4_ArhoMax": int(delay_rho_iter),
        "target_iter_4_ArhoMax": int(target_rho_iter),
        "eps_u": float(cli_args.eps_u),
        "decode_dwell_sec": float(
            cli_args.decode_dwell_sec
            if cli_args.decode_dwell_sec is not None
            else decode_default
        ),
        "num_states": int(num_states),
        "seed": int(cli_args.seed),
        "progress_every_batches": int(cli_args.progress_every_batches),
        "progress_every_epochs": int(cli_args.progress_every_epochs),
        "progress_every_pairs": int(cli_args.progress_every_pairs),
        "verb": int(cli_args.verb),
    }
    return SimpleNamespace(**vals)


def compute_loss_time(spikes, A, B, c_hat, train_md):
    spikes = np.asarray(spikes, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    B = as_2d_B(B).astype(np.float64)
    c_hat = np.asarray(c_hat, dtype=np.float64)
    dt = float(train_md["time_step_sec"])
    eta_clip = float(train_md["eta_clip"])
    lambda2 = float(train_md["lambda2"])

    yp = spikes[:-1]
    yc = spikes[1:]
    c_pairs = c_hat[1:]
    c_prev = c_hat[:-1]
    eta = yp @ A.T + c_pairs @ B
    eta_c = np.minimum(eta, eta_clip)
    lam = np.exp(eta_c) * dt
    log_dt = math.log(dt)
    nll_t = np.sum(lam - yc * (eta + log_dt), axis=1)
    l2_t = lambda2 * np.sum((c_pairs - c_prev) ** 2, axis=1)
    return nll_t.astype(np.float32), l2_t.astype(np.float32)


def train_metadata(eff_args, source_train, spike_md, n_neuron, n_time, time_range_bins, time_range_sec):
    if str(eff_args.state_mode) == "locked":
        num_em_iters = 0
        total_m_epochs = int(eff_args.warmup_locked_epochs) + int(eff_args.m_epochs)
    else:
        num_em_iters = int(eff_args.num_debias_iters)
        total_m_epochs = int(eff_args.warmup_locked_epochs) + num_em_iters * int(eff_args.m_epochs)
    return {
        "fit_stage": "deBias_stageC",
        "num_em_iters": int(num_em_iters),
        "num_debias_iters": int(eff_args.num_debias_iters),
        "warmup_locked_epochs": int(eff_args.warmup_locked_epochs),
        "m_epochs": int(eff_args.m_epochs),
        "total_m_epochs": int(total_m_epochs),
        "pgd_iter": int(eff_args.pgd_iter),
        "lr_estep": float(eff_args.lr_estep),
        "lr_mstep": float(eff_args.lr_mstep),
        "lr_end_factor": float(eff_args.lr_end_factor),
        "lambda2": float(eff_args.lambda2),
        "lambda3": 0.0,
        "rho_max": float(eff_args.rho_max),
        "rho_projection_mode": "active_uv_scale",
        "prescale_m_step_4_ArhoMax": int(eff_args.prescale_m_step_4_ArhoMax),
        "delay_em_iter_4_ArhoMax": int(eff_args.delay_iter_4_ArhoMax),
        "target_em_iter_4_ArhoMax": int(eff_args.target_iter_4_ArhoMax),
        "delay_iter_4_ArhoMax": int(eff_args.delay_iter_4_ArhoMax),
        "target_iter_4_ArhoMax": int(eff_args.target_iter_4_ArhoMax),
        "rho_sync_mode": "DDP_identical_active_params",
        "delay_em_iter_4_lrDecay": 0,
        "target_em_iter_4_lrDecay": max(1, int(eff_args.num_debias_iters)),
        "delay_em_iter_4_Aprune": 0,
        "mstep_state_mode": "destination_bin_onehot",
        "state_mode": str(eff_args.state_mode),
        "batch_size": int(eff_args.batch_size),
        "num_states": int(eff_args.num_states),
        "num_neurons": int(n_neuron),
        "num_time_bins": int(n_time),
        "time_step_sec": float(spike_md["time_step_sec"]),
        "eta_clip": float(spike_md["poisson_eta_clip"]),
        "time_range_sec": [float(time_range_sec[0]), float(time_range_sec[1])],
        "time_range_bins": [int(time_range_bins[0]), int(time_range_bins[1])],
        "seed": int(eff_args.seed),
        "eps_u": float(eff_args.eps_u),
        "source_stageB_mstep_state_mode": source_train.get("mstep_state_mode"),
    }


def main():
    job_t0 = time.perf_counter()
    args = parse_args()
    env_rank = os.environ.get("RANK", "0")
    env_local = os.environ.get("LOCAL_RANK", "0")
    env_world = os.environ.get("WORLD_SIZE", "1")
    if args.verb > 0 and env_rank == "0":
        print(
            "[prism_deBiasFit3c pre-init rank_env=%s local_env=%s/%s] raw args=%s"
            % (env_rank, env_local, env_world, vars(args)),
            flush=True,
        )
    if not os.path.isdir(args.basePath):
        raise FileNotFoundError(
            "--basePath does not exist or is not a directory: %r. "
            "If you used --basePath $basePath, verify it with: echo \"$basePath\""
            % args.basePath
        )
    for subdir in ("spikesData", "prismFit"):
        sub_path = os.path.join(args.basePath, subdir)
        if not os.path.isdir(sub_path):
            raise FileNotFoundError(
                "--basePath %r is missing required subdirectory %r at %r"
                % (args.basePath, subdir, sub_path)
            )
    validate_fit_stem(args.fdrFitName, "--fdrFitName")
    if args.outFitName is None:
        args.outFitName = "%s_debias_%s" % (args.fdrFitName, secrets.token_hex(2))
    validate_fit_stem(args.outFitName, "--outFitName")

    ctx = init_distributed(verb=args.verb, program_name="prism_deBiasFit3c", rank0_only=True)
    seed_everything(args.seed)
    if args.verb > 0 and is_rank0(ctx):
        print(
            "[prism_deBiasFit3c rank=%d/%d] seed set to %d"
            % (ctx.rank, ctx.world_size, int(args.seed)),
            flush=True,
        )

    try:
        if is_rank0(ctx):
            if args.verb > 0:
                print("Runtime:", runtime_summary(ctx), flush=True)
                print("\nStage (c) deBias raw args:", vars(args), "\n", flush=True)
            fit_f = os.path.join(args.basePath, "prismFit", "%s.prismEM.npz" % args.fdrFitName)
            if args.verb > 0:
                print("[rank 0] loading Stage (b) aggregate: %s" % fit_f, flush=True)
            fdr_d, fdr_md = read_data_npz(fit_f, verb=args.verb > 1)
            if "bagsFDR_stageB" not in fdr_md:
                raise KeyError("Input must be a Stage (b) FDR aggregate with bagsFDR_stageB metadata")
            required = ("A_hat", "B_hat", "selected_mask", "neuron_type", "S_hat", "c_hat", "S_hat_CL")
            missing = [k for k in required if k not in fdr_d]
            if missing:
                raise KeyError("Stage (b) input missing required arrays: %s" % missing)

            train_b = dict(fdr_md["train"])
            eff_args = effective_args(args, fdr_md, train_b)
            if args.verb > 0:
                print("[rank 0] effective args:", vars(eff_args), flush=True)
            source_name = source_spike_name(fdr_md)
            spike_f = os.path.join(args.basePath, "spikesData", "%s.spikes.npz" % source_name)
            if args.verb > 0:
                print("[rank 0] loading source spikes: %s" % spike_f, flush=True)
            spike_d, spike_md = read_data_npz(spike_f, verb=args.verb > 1)
            source_spikes = np.asarray(spike_d["spikes"])
            spikes, time_bins = slice_spikes_by_bins(source_spikes, train_b["time_range_bins"])
            time_sec = list(train_b["time_range_sec"])

            A_stage = np.asarray(fdr_d["A_hat"], dtype=np.float32)
            B_stage = as_2d_B(fdr_d["B_hat"])
            selected_mask = np.asarray(fdr_d["selected_mask"], dtype=np.bool_)
            neuron_type = np.asarray(fdr_d["neuron_type"], dtype=np.int8)
            S_stageB = np.asarray(fdr_d["S_hat"], dtype=np.int64)
            c_stageB = np.asarray(fdr_d["c_hat"], dtype=np.float32)
            S_stageB_CL = np.asarray(fdr_d["S_hat_CL"], dtype=np.float32)

            if S_stageB.shape[0] != spikes.shape[0]:
                raise ValueError("Stage (b) S_hat length does not match sliced spikes")
            if c_stageB.shape[0] != spikes.shape[0]:
                raise ValueError("Stage (b) c_hat length does not match sliced spikes")
            if B_stage.shape[0] != int(eff_args.num_states):
                raise ValueError("B_hat row count does not match train.num_states")

            single_rates = (
                np.asarray(fdr_d["single_rates"], dtype=np.float32)
                if "single_rates" in fdr_d
                else (spikes.mean(axis=0) / float(spike_md["time_step_sec"])).astype(np.float32)
            )

            if args.verb > 0:
                print(
                    "Stage (c) input: %s N=%d M=%d T=%d state_mode=%s m_epochs=%d"
                    % (
                        fit_f,
                        A_stage.shape[0],
                        int(eff_args.num_states),
                        spikes.shape[0],
                        eff_args.state_mode,
                        int(eff_args.m_epochs),
                    ),
                    flush=True,
                )
        else:
            fdr_d = fdr_md = spike_md = eff_args = None
            spikes = A_stage = B_stage = selected_mask = neuron_type = None
            S_stageB = c_stageB = S_stageB_CL = single_rates = None
            source_name = time_bins = time_sec = None

        fdr_md = broadcast_object(fdr_md, ctx)
        if args.verb > 0 and is_rank0(ctx):
            print("[rank %d/%d] metadata broadcast complete" % (ctx.rank, ctx.world_size), flush=True)
        spike_md = broadcast_object(spike_md, ctx)
        eff_args = broadcast_object(eff_args, ctx)
        source_name = broadcast_object(source_name, ctx)
        time_bins = broadcast_object(time_bins, ctx)
        time_sec = broadcast_object(time_sec, ctx)
        spikes = broadcast_array(spikes, ctx)
        A_stage = broadcast_array(A_stage, ctx)
        B_stage = broadcast_array(B_stage, ctx)
        selected_mask = broadcast_array(selected_mask, ctx)
        neuron_type = broadcast_array(neuron_type, ctx)
        S_stageB = broadcast_array(S_stageB, ctx)
        c_stageB = broadcast_array(c_stageB, ctx)
        S_stageB_CL = broadcast_array(S_stageB_CL, ctx)
        single_rates = broadcast_array(single_rates, ctx)
        if args.verb > 0 and is_rank0(ctx):
            print(
                "[rank %d/%d] data broadcast complete spikes=%s A=%s B=%s selected=%s"
                % (
                    ctx.rank,
                    ctx.world_size,
                    getattr(spikes, "shape", None),
                    getattr(A_stage, "shape", None),
                    getattr(B_stage, "shape", None),
                    getattr(selected_mask, "shape", None),
                ),
                flush=True,
            )

        result = run_debias_fit(
            spikes,
            spike_md,
            A_stage,
            B_stage,
            selected_mask,
            neuron_type,
            c_stageB,
            S_stageB,
            eff_args,
            ctx,
        )
        if args.verb > 0 and is_rank0(ctx):
            print("[rank %d/%d] run_debias_fit complete" % (ctx.rank, ctx.world_size), flush=True)

        if is_rank0(ctx):
            out_d = dict(fdr_d)
            for key in ("m_loss_epoch", "m_nll_epoch", "m_l1_epoch", "rho_epoch",
                        "rho_correction_strength_epoch", "nz_edges_epoch",
                        "learning_rates", "e_nll_em"):
                if key in fdr_d:
                    out_d["stageB_" + key] = fdr_d[key]

            out_d["S_stageB"] = S_stageB.astype(np.int64)
            out_d["c_stageB"] = c_stageB.astype(np.float32)
            out_d["S_stageB_CL"] = S_stageB_CL.astype(np.float32)
            out_d["A_debias"] = result["A_debias"].astype(np.float32)
            out_d["B_debias"] = result["B_debias"].astype(np.float32)
            out_d["A_debias_init"] = result["A_debias_init"].astype(np.float32)
            out_d["B_debias_init"] = result["B_debias_init"].astype(np.float32)
            out_d["S_hat"] = result["S_hat"].astype(np.int64)
            out_d["c_hat"] = result["c_hat"].astype(np.float32)
            out_d["S_hat_CL"] = result["S_hat_CL"].astype(np.float32)
            out_d["single_rates"] = single_rates.astype(np.float32)

            hist = result["history"]
            out_d.update(hist)
            active = result["active"]
            out_d["debias_dale_i"] = active["dale_i"].astype(np.int64)
            out_d["debias_dale_j"] = active["dale_j"].astype(np.int64)
            out_d["debias_dale_sign"] = active["dale_sign"].astype(np.float32)
            out_d["debias_free_i"] = active["free_i"].astype(np.int64)
            out_d["debias_free_j"] = active["free_j"].astype(np.int64)
            out_d["A_diag_debias"] = np.diag(result["A_debias"]).astype(np.float32)
            out_d["neuron_Sedge_debias"] = (
                result["A_debias"] * (~np.eye(result["A_debias"].shape[0], dtype=bool))
            ).sum(axis=0).astype(np.float32)

            train_md = train_metadata(
                eff_args,
                fdr_md["train"],
                spike_md,
                result["A_debias"].shape[0],
                spikes.shape[0],
                time_bins,
                time_sec,
            )
            loss_nll_t, loss_l2_t = compute_loss_time(
                spikes, result["A_debias"], result["B_debias"], result["c_hat"], train_md
            )
            out_d["loss_nll_time"] = loss_nll_t
            out_d["loss_l2_time"] = loss_l2_t

            out_md = copy.deepcopy(fdr_md)
            out_md["fit_type"] = "prismEM_deBias_stageC"
            out_md["train"] = train_md
            out_md["eval_f"] = {
                "loss_nll_time_key": "loss_nll_time",
                "loss_l2_time_key": "loss_l2_time",
            }
            out_md["deBias_stageC"] = {
                "program": "prism_deBiasFit3c.py",
                "workhorse": "PrismDeBias_Workhorse3c.py",
                "input_fdrFitName": args.fdrFitName,
                "input_stageB_file": os.path.join(args.basePath, "prismFit", "%s.prismEM.npz" % args.fdrFitName),
                "outFitName": args.outFitName,
                "source_spike_name": source_name,
                "source_spike_file": os.path.join(args.basePath, "spikesData", "%s.spikes.npz" % source_name),
                "reference_emFitName": out_md.get("provenance", {}).get("emFitName"),
                "state_mode_requested": args.state_mode,
                "state_mode": eff_args.state_mode,
                "mstep_state_alignment": "destination_bin_S_t_for_pair_Y_tminus1_to_Y_t",
                "support_source": "stageB_selected_mask_offdiag_plus_all_diagonal",
                "dale_source": "stageB_neuron_type_source_columns",
                "A_initialization": "stageB_A_hat_for_all_active_entries",
                "B_initialization": "stageB_B_hat",
                "A_stageB_preserved_keys": ["A_hat", "A_prune"],
                "B_stageB_preserved_key": "B_hat",
                "A_output_key": "A_debias",
                "B_output_key": "B_debias",
                "no_second_pruning": True,
                "hyperparameters": vars(eff_args),
                "num_dale_active": int(active["num_dale"]),
                "num_free_active": int(active["num_free"]),
                "num_diag_free": int(active["num_diag"]),
                "num_selected_offdiag": int(active["num_selected_offdiag"]),
                "num_trainable_A": int(active["num_dale"] + active["num_free"]),
                "num_trainable_B": int(np.asarray(result["B_debias"]).size),
                "elapsed_train_sec": float(result["elapsed_train_sec"]),
                "elapsed_total_sec": float(time.perf_counter() - job_t0),
            }
            prov = dict(out_md.get("provenance", {}))
            prov.update({
                "deBias_stageC_input": args.fdrFitName,
                "deBias_stageC_file": args.outFitName,
                "deBias_stageC_outFitName": args.outFitName,
                "EMtrain_file": args.outFitName,
            })
            out_md["provenance"] = prov

            out_dir = os.path.join(args.basePath, "prismFit")
            os.makedirs(out_dir, exist_ok=True)
            out_f = os.path.join(out_dir, "%s.prismEM.npz" % args.outFitName)
            out_md = json_safe_metadata(out_md)
            write_data_npz(out_d, out_f, metaD=out_md, verb=args.verb > 1)
            if args.verb > 0:
                print("\nSaved Stage (c) deBias fit: %s" % out_f, flush=True)
                print(
                    "  A_debias: %s  B_debias: %s  state_mode=%s"
                    % (out_d["A_debias"].shape, out_d["B_debias"].shape, eff_args.state_mode),
                    flush=True,
                )
                print(
                    "  ./prism_EM_eval3c.py --basePath %s --dataName %s -p a b c e h i"
                    % (args.basePath, args.outFitName),
                    flush=True,
                )
        if args.verb > 0 and is_rank0(ctx):
            print("[rank %d/%d] entering final barrier" % (ctx.rank, ctx.world_size), flush=True)
        barrier(ctx)
        if args.verb > 0 and is_rank0(ctx):
            print("[rank %d/%d] finished total_elapsed=%.1fs" % (ctx.rank, ctx.world_size, time.perf_counter() - job_t0), flush=True)
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
