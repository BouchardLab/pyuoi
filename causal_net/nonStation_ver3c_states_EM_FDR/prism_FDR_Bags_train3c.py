#!/usr/bin/env python3
"""Run one PRISM-EM FDR bag using reference-locked state labels.

One invocation loads the reference full EM fit, samples a fraction of
reference-labeled lag pairs, runs one locked M-step for real A/B, runs
per-neuron time-shuffle locked-M-step null refits, and writes one bag file.
"""

import argparse
import os
import re
import secrets
import time
from types import SimpleNamespace

import numpy as np

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
from Util_PrismEM import init_A_from_spikes, init_B_from_spikes
from PrismEM_Workhorse3c import (
    barrier,
    broadcast_array,
    broadcast_object,
    broadcast_optional_array,
    cleanup_distributed,
    init_distributed,
    init_null_params,
    is_rank0,
    run_locked_mstep,
    runtime_summary,
    seed_everything,
    source_type_prune,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PRISM-EM FDR bagging Stage (a): one reference-locked bag",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--emFitName", required=True,
                        help="Input EM-train fit stem in prismFit/")
    parser.add_argument("--outFitName", default=None,
                        help="Output Stage (a) bag fit stem written to prismFDR/; default is emFitName plus random _hash4")
    parser.add_argument("--basePath", required=True,
                        help="Run directory containing spikesData/, prismFit/, prismFDR/")
    parser.add_argument("--bag_idx", type=int, required=True,
                        help="Bag index used in output filename and RNG stream")
    parser.add_argument("--fdr_out_dir", type=str, default=None,
                        help="Output directory; default: <basePath>/prismFDR")
    parser.add_argument("--bag_frac", type=float, default=0.8,
                        help="Fraction of reference lag pairs sampled without replacement")
    parser.add_argument("--num_scrambles", type=int, default=4,
                        help="Number of per-neuron time-shuffle null refits")
    parser.add_argument("--min_roll_shift_sec", type=float, default=2.0,
                        help="Two-sided exclusion-zone half-width for circular shifts")
    parser.add_argument("--init_A", choices=["ref", "data", "rand"], default="data",
                        help="A initialization for the real locked fit")
    parser.add_argument("--init_B", choices=["ref", "data", "rand"], default="data",
                        help="B initialization for the real locked fit")
    parser.add_argument("--null_A_init", choices=["scrambled_data", "rand"],
                        default="scrambled_data",
                        help="Fresh A initialization mode for each null refit")
    parser.add_argument("--null_B_init", choices=["scrambled_data", "rand"],
                        default="scrambled_data",
                        help="Fresh B initialization mode for each null refit")

    g = parser.add_argument_group("locked M-step overrides")
    g.add_argument("--epochs", type=int, default=None,
                   help="Locked M-step epochs; default from reference train.total_m_epochs")
    g.add_argument("--lr_mstep", type=float, default=None)
    g.add_argument("--lr_end_factor", type=float, default=None)
    g.add_argument("--lambda3", type=float, default=None)
    g.add_argument("--rho_max", type=float, default=None)
    g.add_argument("--prescale_m_step_4_ArhoMax", type=int, default=None)
    g.add_argument("--batch_size", type=int, default=None)
    g.add_argument("--init_A_Tmax", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verb", type=int, default=1)
    return parser.parse_args()


def validate_fit_stem(name, arg_name):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(name)):
        raise ValueError(f"{arg_name} must contain only letters, digits, underscores, dots, or hyphens: {name!r}")
    if "/" in str(name):
        raise ValueError(f"{arg_name} must be a file stem, not a path: {name!r}")


def ref_train_value(train_md, key, default=None):
    if key in train_md:
        return train_md[key]
    if key == "epochs":
        return int(train_md.get("total_m_epochs", int(train_md["num_em_iters"]) * int(train_md["m_epochs"])))
    return default


def source_spike_name(ref_md):
    prov = ref_md["provenance"]
    if ref_md.get("data_type") == "bioExp":
        return prov["experiment_name"]
    return prov["state_transition_file"]


def effective_locked_args(cli_args, ref_md, source_data_name):
    train_md = ref_md["train"]
    epochs = cli_args.epochs
    if epochs is None:
        epochs = ref_train_value(train_md, "epochs")
    if int(epochs) < 1:
        raise ValueError("--epochs must be >= 1")

    vals = {
        "dataName": source_data_name,
        "basePath": cli_args.basePath,
        "num_states": int(train_md["num_states"]),
        "num_em_iters": 1,
        "m_epochs": int(epochs),
        "epochs": int(epochs),
        "lr_mstep": float(cli_args.lr_mstep if cli_args.lr_mstep is not None else train_md["lr_mstep"]),
        "lr_end_factor": float(cli_args.lr_end_factor if cli_args.lr_end_factor is not None else train_md["lr_end_factor"]),
        "lambda3": float(cli_args.lambda3 if cli_args.lambda3 is not None else train_md["lambda3"]),
        "rho_max": float(cli_args.rho_max if cli_args.rho_max is not None else train_md["rho_max"]),
        "prescale_m_step_4_ArhoMax": int(
            cli_args.prescale_m_step_4_ArhoMax
            if cli_args.prescale_m_step_4_ArhoMax is not None
            else train_md["prescale_m_step_4_ArhoMax"]
        ),
        "batch_size": int(cli_args.batch_size if cli_args.batch_size is not None else train_md["batch_size"]),
        "init_A": cli_args.init_A,
        "init_B": cli_args.init_B,
        "init_A_Tmax": int(
            cli_args.init_A_Tmax
            if cli_args.init_A_Tmax is not None
            else train_md.get("init_A_Tmax", 50000)
        ),
        "seed": int(cli_args.seed),
        "verb": int(cli_args.verb),
    }
    return SimpleNamespace(**vals)


def locked_hyperparam_dict(eff_args):
    keys = (
        "epochs", "lr_mstep", "lr_end_factor", "lambda3", "rho_max",
        "prescale_m_step_4_ArhoMax", "batch_size", "init_A", "init_B",
        "init_A_Tmax", "num_states", "seed",
    )
    return {k: getattr(eff_args, k) for k in keys}


def slice_reference_spikes(spikes, ref_md):
    train_md = ref_md["train"]
    b0, b1 = [int(x) for x in train_md["time_range_bins"]]
    if b1 <= b0:
        raise ValueError(f"Bad reference time_range_bins: {[b0, b1]}")
    if b0 < 0 or b1 >= spikes.shape[0]:
        raise ValueError(
            f"Reference bins {[b0, b1]} outside spike array length {spikes.shape[0]}"
        )
    return np.asarray(spikes[b0:b1 + 1]), [b0, b1]


def draw_pair_indices(rng, n_pair, bag_frac):
    frac = float(bag_frac)
    if not (0.0 < frac <= 1.0):
        raise ValueError("--bag_frac must be in (0, 1]")
    k = int(np.floor(frac * int(n_pair)))
    k = max(1, min(int(n_pair), k))
    return np.sort(rng.choice(int(n_pair), size=k, replace=False)).astype(np.int64)


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
            f"Reference window length {T} too short for min_roll_shift_bins={min_shift_bins}"
        )
    return rng.integers(lo, hi + 1, size=int(N), dtype=np.int64)


def stack_history(histories, key, dtype):
    vals = [np.asarray(h[key], dtype=dtype) for h in histories]
    return np.stack(vals, axis=0) if vals else np.zeros((0, 0), dtype=dtype)


def real_init_params(mode_a, mode_b, ref_d, ref_spikes, spike_md, eff_args, ctx):
    if is_rank0(ctx):
        if mode_a == "ref":
            A_init = np.asarray(ref_d["A_hat"], dtype=np.float32)
            A_md = {"method": "reference_fit", "source_key": "A_hat"}
        elif mode_a == "data":
            a_args = SimpleNamespace(**vars(eff_args))
            a_args.init_A = "data"
            A_init, A_md = init_A_from_spikes(ref_spikes, a_args)
        elif mode_a == "rand":
            A_init, A_md = None, {"method": "rand"}
        else:
            raise ValueError(f"Bad init_A={mode_a}")

        if mode_b == "ref":
            B_init = np.asarray(ref_d["B_hat"], dtype=np.float32)
            B_md = {"method": "reference_fit", "source_key": "B_hat"}
        elif mode_b == "data":
            b_args = SimpleNamespace(**vars(eff_args))
            b_args.init_B = "data"
            B_init, B_md = init_B_from_spikes(ref_spikes, float(spike_md["time_step_sec"]), b_args)
        elif mode_b == "rand":
            B_init, B_md = None, {"method": "rand"}
        else:
            raise ValueError(f"Bad init_B={mode_b}")
    else:
        A_init = B_init = None
        A_md = B_md = None

    A_init = broadcast_optional_array(A_init, ctx)
    B_init = broadcast_optional_array(B_init, ctx)
    A_md = broadcast_object(A_md, ctx)
    B_md = broadcast_object(B_md, ctx)
    return A_init, B_init, A_md, B_md


def main():
    job_t0 = time.perf_counter()
    args = parse_args()
    if args.outFitName is None:
        args.outFitName = f"{args.emFitName}_{secrets.token_hex(2)}"
    validate_fit_stem(args.emFitName, "--emFitName")
    validate_fit_stem(args.outFitName, "--outFitName")
    ctx = init_distributed()
    seed_base = int(args.seed) + int(args.bag_idx) * 1000003
    seed_everything(seed_base)

    try:
        out_dir = args.fdr_out_dir or os.path.join(args.basePath, "prismFDR")
        em_fit_name = args.emFitName
        out_fit_name = args.outFitName

        if is_rank0(ctx):
            os.makedirs(out_dir, exist_ok=True)
            if args.verb > 0:
                print("Runtime:", runtime_summary(ctx))

            ref_f = os.path.join(args.basePath, "prismFit", f"{em_fit_name}.prismEM.npz")
            ref_d, ref_md = read_data_npz(ref_f, verb=args.verb > 1)
            source_data_name = source_spike_name(ref_md)
            spike_f = os.path.join(args.basePath, "spikesData", f"{source_data_name}.spikes.npz")
            spike_d, spike_md = read_data_npz(spike_f, verb=args.verb > 1)
            source_spikes = np.asarray(spike_d["spikes"])
            ref_spikes, ref_bins = slice_reference_spikes(source_spikes, ref_md)

            s_hat = np.asarray(ref_d["S_hat"], dtype=np.int64)
            if s_hat.shape[0] != ref_spikes.shape[0]:
                raise ValueError(
                    f"Reference S_hat length {s_hat.shape[0]} != reference spikes length {ref_spikes.shape[0]}"
                )
            if np.asarray(ref_d["B_hat"]).shape[0] != int(ref_md["train"]["num_states"]):
                raise ValueError("Reference B_hat row count does not match train.num_states")

            eff_args = effective_locked_args(args, ref_md, source_data_name)
            rng = np.random.default_rng(seed_base)
            n_pair = ref_spikes.shape[0] - 1
            pair_idx = draw_pair_indices(rng, n_pair, args.bag_frac)
            single_rates = ref_spikes.mean(axis=0).astype(np.float64) / float(spike_md["time_step_sec"])

            if args.verb > 0:
                print("\nFDR-bag args:", vars(args), "\n")
                print(
                    f"reference={ref_f}\n"
                    f"source_spikes={spike_f}\n"
                    f"outFitName={out_fit_name} bag_idx={args.bag_idx} "
                    f"N={ref_spikes.shape[1]} M={eff_args.num_states} "
                    f"T_ref={ref_spikes.shape[0]} pairs={pair_idx.size}/{n_pair} "
                    f"epochs={eff_args.epochs} scrambles={args.num_scrambles}"
                )
        else:
            ref_d = ref_md = spike_md = ref_spikes = pair_idx = single_rates = eff_args = ref_bins = None
            source_data_name = None

        ref_md = broadcast_object(ref_md, ctx)
        source_data_name = broadcast_object(source_data_name, ctx)
        em_fit_name = broadcast_object(em_fit_name, ctx)
        out_fit_name = broadcast_object(out_fit_name, ctx)
        spike_md = broadcast_object(spike_md, ctx)
        eff_args = broadcast_object(eff_args, ctx)
        ref_bins = broadcast_object(ref_bins, ctx)
        ref_spikes = broadcast_array(ref_spikes, ctx)
        pair_idx = broadcast_array(pair_idx, ctx)
        single_rates = broadcast_array(single_rates, ctx)

        if is_rank0(ctx):
            ref_A = np.asarray(ref_d["A_hat"], dtype=np.float32)
            ref_B = np.asarray(ref_d["B_hat"], dtype=np.float32)
            ref_S = np.asarray(ref_d["S_hat"], dtype=np.int64)
            ref_c = np.asarray(ref_d["c_hat"], dtype=np.float32)
            ref_cl = np.asarray(ref_d["S_hat_CL"], dtype=np.float32)
        else:
            ref_A = ref_B = ref_S = ref_c = ref_cl = None
        ref_A = broadcast_array(ref_A, ctx)
        ref_B = broadcast_array(ref_B, ctx)
        ref_S = broadcast_array(ref_S, ctx)
        ref_c = broadcast_array(ref_c, ctx)
        ref_cl = broadcast_array(ref_cl, ctx)
        ref_d_min = {"A_hat": ref_A, "B_hat": ref_B}

        yp = ref_spikes[:-1].astype(np.float32)
        yc = ref_spikes[1:].astype(np.float32)
        s_pair = ref_S[1:].astype(np.int64)
        yp_bag = yp[pair_idx]
        yc_bag = yc[pair_idx]
        s_bag = s_pair[pair_idx]

        A_init, B_init, A_init_md, B_init_md = real_init_params(
            eff_args.init_A, eff_args.init_B, ref_d_min, ref_spikes, spike_md, eff_args, ctx
        )

        real_t0 = time.perf_counter()
        A_hat, B_hat, hist = run_locked_mstep(
            yp_bag, yc_bag, s_bag, A_init, B_init, spike_md, eff_args, ctx
        )
        if is_rank0(ctx) and args.verb > 0:
            print(f"real locked fit finished: elapsed={time.perf_counter() - real_t0:.1f}s")

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

        dt = float(spike_md["time_step_sec"])
        min_roll_shift_bins = int(round(float(args.min_roll_shift_sec) / dt))
        T_ref, N = ref_spikes.shape
        null_eff_args = SimpleNamespace(**vars(eff_args))
        if int(args.verb) == 1:
            null_eff_args.verb = 0
        for p in range(int(args.num_scrambles)):
            seed_everything(seed_base + 1009 * (p + 1))
            if is_rank0(ctx):
                shifts = draw_roll_shifts(rng, T_ref, N, min_roll_shift_bins)
            else:
                shifts = None
            shifts = broadcast_array(shifts, ctx)
            scrambled = circular_shift_by_neuron(ref_spikes, shifts)

            A0, B0, A0_md, B0_md = init_null_params(
                scrambled, spike_md, null_eff_args, ctx, args.null_A_init, args.null_B_init
            )
            A_p, B_p, hist_p = run_locked_mstep(
                scrambled[:-1].astype(np.float32)[pair_idx],
                scrambled[1:].astype(np.float32)[pair_idx],
                s_bag,
                A0,
                B0,
                spike_md,
                eff_args,
                ctx,
            )

            if is_rank0(ctx):
                A_null.append(A_p.astype(np.float32))
                B_null.append(B_p.astype(np.float32))
                roll_shifts.append(np.asarray(shifts, dtype=np.int64))
                null_histories.append(hist_p)
                null_A_init_md.append(A0_md)
                null_B_init_md.append(B0_md)
                if args.verb > 0:
                    nz = int(hist_p["nz_edges_epoch"][-1]) if hist_p["nz_edges_epoch"].size else -1
                    rho = float(hist_p["rho_epoch"][-1]) if hist_p["rho_epoch"].size else float("nan")
                    print(
                        f"null {p + 1}/{args.num_scrambles} finished: "
                        f"elapsed={time.perf_counter() - job_t0:.1f}s "
                        f"rho={rho:.4f} nz={nz}"
                    )

        if is_rank0(ctx):
            A_prune, neuron_type, neuron_sedge = source_type_prune(A_hat)
            out_data_name = out_fit_name
            out_stem = f"{out_fit_name}.bag{int(args.bag_idx):03d}"
            ref_train = dict(ref_md["train"])
            train_md = dict(ref_train)
            train_md.update({
                "fit_stage": "FDR_locked_mstep_bag",
                "num_em_iters": 0,
                "m_epochs": int(eff_args.epochs),
                "total_m_epochs": int(eff_args.epochs),
                "lr_mstep": float(eff_args.lr_mstep),
                "lr_end_factor": float(eff_args.lr_end_factor),
                "lambda3": float(eff_args.lambda3),
                "rho_max": float(eff_args.rho_max),
                "prescale_m_step_4_ArhoMax": int(eff_args.prescale_m_step_4_ArhoMax),
                "batch_size": int(eff_args.batch_size),
                "mstep_state_mode": "reference_viterbi_onehot_currbin",
                "num_time_bins": int(T_ref),
                "time_range_bins": [int(ref_bins[0]), int(ref_bins[1])],
                "time_range_sec": list(ref_train["time_range_sec"]),
                "seed": int(seed_base),
            })

            outD = {
                "A_hat": A_hat.astype(np.float32),
                "A_prune": A_prune.astype(np.float32),
                "neuron_type": neuron_type.astype(np.int8),
                "neuron_Sedge": neuron_sedge.astype(np.float32),
                "B_hat": B_hat.astype(np.float32),
                "single_rates": np.asarray(single_rates, dtype=np.float32),
                "S_hat": ref_S.astype(np.int64),
                "c_hat": ref_c.astype(np.float32),
                "S_hat_CL": ref_cl.astype(np.float32),
                "m_loss_epoch": hist["m_loss_epoch"],
                "m_nll_epoch": hist["m_nll_epoch"],
                "m_l1_epoch": hist["m_l1_epoch"],
                "rho_epoch": hist["rho_epoch"],
                "rho_correction_strength_epoch": hist["rho_correction_strength_epoch"],
                "nz_edges_epoch": hist["nz_edges_epoch"],
                "learning_rates": hist["learning_rates"],
                "e_nll_em": np.zeros((0,), dtype=np.float64),
                "A_null": np.stack(A_null, axis=0).astype(np.float32),
                "B_null": np.stack(B_null, axis=0).astype(np.float32),
                "roll_shifts_bin": np.stack(roll_shifts, axis=0).astype(np.int64),
                "bag_pair_indices": pair_idx.astype(np.int64),
                "bag_pair_dest_bins": (int(ref_bins[0]) + pair_idx + 1).astype(np.int64),
                "bag_pair_state": s_bag.astype(np.int64),
                "null_m_loss_epoch": stack_history(null_histories, "m_loss_epoch", np.float64),
                "null_m_nll_epoch": stack_history(null_histories, "m_nll_epoch", np.float64),
                "null_m_l1_epoch": stack_history(null_histories, "m_l1_epoch", np.float64),
                "null_rho_epoch": stack_history(null_histories, "rho_epoch", np.float64),
                "null_nz_edges_epoch": stack_history(null_histories, "nz_edges_epoch", np.int64),
                "null_learning_rates": stack_history(null_histories, "learning_rates", np.float64),
            }
            if A_init is not None:
                outD["A_init"] = np.asarray(A_init, dtype=np.float32)
            if B_init is not None:
                outD["B_init"] = np.asarray(B_init, dtype=np.float32)

            outMD = dict(spike_md)
            outMD["fit_type"] = "prismEM_FDRbags_stageA"
            outMD["train"] = train_md
            outMD["states_recovery_eval"] = dict(ref_md.get("states_recovery_eval", {}))
            outMD["init_A"] = A_init_md
            outMD["init_B"] = B_init_md
            outMD["init_state"] = {
                "method": "reference_fit",
                "source_key": "S_hat",
                "em_fit_file": f"{em_fit_name}.prismEM.npz",
            }
            prov = dict(ref_md.get("provenance", spike_md.get("provenance", {})))
            prov.update({
                "dataName": source_data_name,
                "EMtrain_file": out_stem,
                "emFitName": em_fit_name,
                "outFitName": out_fit_name,
                "bagsFDR_stageA_dataName": out_data_name,
                "bagsFDR_stageA_file": out_stem,
            })
            outMD["provenance"] = prov
            outMD["bagsFDR_stageA"] = {
                "program": "prism_FDR_Bags_train3c.py",
                "method": "reference_locked_pair_subsample",
                "dataName": source_data_name,
                "source_spike_name": source_data_name,
                "emFitName": em_fit_name,
                "outFitName": out_fit_name,
                "output_dataName": out_data_name,
                "output_name": out_stem,
                "bag_idx": int(args.bag_idx),
                "bag_frac": float(args.bag_frac),
                "num_reference_pairs": int(T_ref - 1),
                "num_selected_pairs": int(pair_idx.size),
                "sample_without_replacement": True,
                "em_fit_file": os.path.join(args.basePath, "prismFit", f"{em_fit_name}.prismEM.npz"),
                "reference_time_range_bins": [int(ref_bins[0]), int(ref_bins[1])],
                "reference_time_range_sec": list(ref_train["time_range_sec"]),
                "locked_state_alignment": "S_hat[t] for pair (spikes[t-1], spikes[t])",
                "locked_mstep": locked_hyperparam_dict(eff_args),
                "real_init": {"A": A_init_md, "B": B_init_md},
                "null_refits": {
                    "num_scrambles": int(args.num_scrambles),
                    "A_init": args.null_A_init,
                    "B_init": args.null_B_init,
                    "A_init_md": null_A_init_md,
                    "B_init_md": null_B_init_md,
                    "min_roll_shift_sec": float(args.min_roll_shift_sec),
                    "min_roll_shift_bins": int(min_roll_shift_bins),
                },
                "seed_base": int(seed_base),
                "real_fit": {"train": train_md},
            }

            outF = os.path.join(out_dir, f"{out_stem}.prismFDRbag.npz")
            write_data_npz(outD, outF, metaD=outMD, verb=args.verb > 1)
            print(f"\nSaved FDR bag: {outF}")
        barrier(ctx)
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
