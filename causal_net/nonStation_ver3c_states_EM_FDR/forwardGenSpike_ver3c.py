#!/usr/bin/env python3
"""Generate spikes by replaying a fitted PRISM-EM latent-state timeline.

Accepted inputs are Stage (b) FDR-bag aggregates and Stage (c) de-biased
fits.  Stage (b) uses A_prune/B_hat; Stage (c) uses A_debias/B_debias.
The fitted hard state path starts at S_hat[-1] and wraps cyclically.
"""

import argparse
import copy
import math
import os
import re
import secrets
import time

import numpy as np

from toolbox.Util_NumpyIOv2 import (
    json_safe_metadata,
    read_data_npz,
    write_data_npz,
)


PROGRAM = "forwardGenSpike_ver3c.py"
STAGE_MODEL_KEYS = {
    "prismEM_FDRbags_stageB": {
        "stage": "stageB_aggregate",
        "A_key": "A_prune",
        "B_key": "B_hat",
    },
    "prismEM_deBias_stageC": {
        "stage": "stageC_debias",
        "A_key": "A_debias",
        "B_key": "B_debias",
    },
}
QUARTILE_PROBABILITIES = np.asarray([0.25, 0.50, 0.75, 1.00], dtype=np.float64)
QUARTILE_LABELS = np.asarray(["q25", "q50", "q75", "q100"], dtype=str)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a spike train from a Stage (b) aggregate or Stage (c) "
            "de-biased PRISM-EM model"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--basePath",
        required=True,
        help="Run directory containing prismFit/ and receiving spikesData/",
    )
    parser.add_argument(
        "--inputModel",
        required=True,
        help="Input fit stem in prismFit/, without .prismEM.npz",
    )
    parser.add_argument(
        "--time_range_sec",
        type=float,
        nargs=2,
        required=True,
        metavar=("START", "STOP"),
        help=(
            "Requested output time range in seconds; STOP-START sets the "
            "generated duration and must be an integer multiple of the "
            "fitted time step"
        ),
    )
    parser.add_argument(
        "--dataName",
        default=None,
        help="Output stem; default is forwardN<N>_<random 6-hex>",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for NumPy Poisson sampling",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=1,
        dest="verb",
        help="Diagnostic verbosity",
    )
    return parser.parse_args()


def validate_stem(value, option):
    value = str(value)
    if value.endswith(".prismEM.npz"):
        raise ValueError(
            f"{option} must be a file stem without the .prismEM.npz suffix: {value!r}"
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(
            f"{option} must contain only letters, digits, underscores, dots, "
            f"or hyphens: {value!r}"
        )
    return value


def _required_mapping(mapping, key, label):
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} must be a metadata dictionary")
    if key not in mapping:
        raise KeyError(f"{label} is missing required field {key!r}")
    return mapping[key]


def _as_finite_float(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return value


def time_range_to_num_steps(time_range_sec, time_step_sec):
    """Validate a physical time range and convert its duration to bins."""
    time_range = np.asarray(time_range_sec, dtype=np.float64)
    if time_range.shape != (2,) or not np.all(np.isfinite(time_range)):
        raise ValueError("--time_range_sec requires two finite values: START STOP")
    start_sec, stop_sec = (float(x) for x in time_range)
    if start_sec < 0.0:
        raise ValueError("--time_range_sec START must be non-negative")
    if stop_sec <= start_sec:
        raise ValueError("--time_range_sec requires STOP > START")

    duration_sec = stop_sec - start_sec
    exact_steps = duration_sec / float(time_step_sec)
    num_steps = int(round(exact_steps))
    tolerance = 1e-8 * max(1.0, abs(exact_steps))
    if abs(exact_steps - num_steps) > tolerance:
        raise ValueError(
            "--time_range_sec duration %.12g sec is not an integer multiple "
            "of time_step_sec=%.12g (%.12g bins)"
            % (duration_sec, time_step_sec, exact_steps)
        )
    if num_steps <= 0:
        raise ValueError("--time_range_sec selects no time bins")
    return np.asarray([start_sec, stop_sec], dtype=np.float64), num_steps


def load_forward_model(base_path, input_model, verb=1):
    """Load and validate one accepted fit, returning normalized model arrays."""
    model_file = os.path.join(
        base_path, "prismFit", f"{input_model}.prismEM.npz"
    )
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Input model does not exist: {model_file}")

    fit_d, fit_md = read_data_npz(model_file, verb=verb > 1)
    if not isinstance(fit_md, dict):
        raise ValueError(f"Input model has no schema-v2 metadata: {model_file}")

    fit_type = str(fit_md.get("fit_type", ""))
    if fit_type not in STAGE_MODEL_KEYS:
        raise ValueError(
            f"Unsupported fit_type {fit_type!r} in {model_file}; expected one of "
            f"{sorted(STAGE_MODEL_KEYS)}"
        )

    selection = STAGE_MODEL_KEYS[fit_type]
    a_key = selection["A_key"]
    b_key = selection["B_key"]
    missing = [key for key in (a_key, b_key, "S_hat") if key not in fit_d]
    if missing:
        raise KeyError(
            f"Input {fit_type} model is missing required arrays: {missing}"
        )

    train_md = _required_mapping(fit_md, "train", "fit metadata")
    if not isinstance(train_md, dict):
        raise ValueError("fit metadata field 'train' must be a dictionary")

    dt = _as_finite_float(
        _required_mapping(train_md, "time_step_sec", "train metadata"),
        "train.time_step_sec",
    )
    if dt <= 0.0:
        raise ValueError(f"train.time_step_sec must be positive, got {dt}")

    eta_clip = _as_finite_float(
        _required_mapping(train_md, "eta_clip", "train metadata"),
        "train.eta_clip",
    )
    if "poisson_eta_clip" in fit_md:
        top_eta_clip = _as_finite_float(
            fit_md["poisson_eta_clip"], "metadata.poisson_eta_clip"
        )
        if not math.isclose(
            eta_clip, top_eta_clip, rel_tol=1e-7, abs_tol=1e-12
        ):
            raise ValueError(
                "Clipping metadata disagree: "
                f"train.eta_clip={eta_clip} but "
                f"poisson_eta_clip={top_eta_clip}"
            )

    A = np.asarray(fit_d[a_key], dtype=np.float64)
    B = np.asarray(fit_d[b_key], dtype=np.float64)
    if B.ndim == 1:
        B = B[None, :]
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{a_key} must be square 2D, got shape {A.shape}")
    if B.ndim != 2:
        raise ValueError(f"{b_key} must be 1D or 2D, got shape {B.shape}")

    n_neurons = int(A.shape[0])
    n_states = int(B.shape[0])
    if B.shape[1] != n_neurons:
        raise ValueError(
            f"Neuron mismatch: {a_key} shape={A.shape}, {b_key} shape={B.shape}"
        )
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{a_key} contains non-finite values")
    if not np.all(np.isfinite(B)):
        raise ValueError(f"{b_key} contains non-finite values")

    md_neurons = int(
        _required_mapping(train_md, "num_neurons", "train metadata")
    )
    md_states = int(
        _required_mapping(train_md, "num_states", "train metadata")
    )
    if md_neurons != n_neurons:
        raise ValueError(
            f"train.num_neurons={md_neurons} but {a_key} implies N={n_neurons}"
        )
    if md_states != n_states:
        raise ValueError(
            f"train.num_states={md_states} but {b_key} implies M={n_states}"
        )

    states_raw = np.asarray(fit_d["S_hat"])
    if states_raw.ndim != 1 or states_raw.size == 0:
        raise ValueError(
            f"S_hat must be a non-empty 1D array, got shape {states_raw.shape}"
        )
    if np.issubdtype(states_raw.dtype, np.integer):
        states = states_raw.astype(np.int64, copy=False)
    else:
        states_float = np.asarray(states_raw, dtype=np.float64)
        if (
            not np.all(np.isfinite(states_float))
            or not np.all(states_float == np.floor(states_float))
        ):
            raise ValueError("S_hat must contain finite integer state indices")
        states = states_float.astype(np.int64)
    if int(states.min()) < 0 or int(states.max()) >= n_states:
        raise ValueError(
            f"S_hat state range [{states.min()}, {states.max()}] is outside "
            f"[0, {n_states - 1}]"
        )

    md_time_bins = int(
        _required_mapping(train_md, "num_time_bins", "train metadata")
    )
    if md_time_bins != states.size:
        raise ValueError(
            f"train.num_time_bins={md_time_bins} but len(S_hat)={states.size}"
        )

    return {
        "model_file": model_file,
        "fit_metadata": fit_md,
        "fit_type": fit_type,
        "stage": selection["stage"],
        "A_key": a_key,
        "B_key": b_key,
        "A": A,
        "B": B,
        "S_hat": states,
        "num_neurons": n_neurons,
        "num_states": n_states,
        "time_step_sec": dt,
        "eta_clip": eta_clip,
    }


def build_forward_states(states_fit, num_steps, num_states):
    """Begin at S_hat[-1] and cyclically wrap the fitted state timeline."""
    states_fit = np.asarray(states_fit, dtype=np.int64)
    if states_fit.ndim != 1 or states_fit.size == 0:
        raise ValueError("states_fit must be a non-empty 1D array")
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")

    indices = (
        np.arange(int(num_steps), dtype=np.int64) + states_fit.size - 1
    ) % states_fit.size
    states_forward = states_fit[indices].astype(np.int32, copy=False)
    coefficients = np.eye(int(num_states), dtype=np.float32)[states_forward]
    return states_forward, coefficients


def simulate_forward(A, B, states_forward, dt, eta_clip, seed, verb=1):
    """Sample a switching Poisson GLM from an all-zero lagged spike vector."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    states_forward = np.asarray(states_forward, dtype=np.int64)
    n_steps = int(states_forward.size)
    n_neurons = int(A.shape[0])

    spikes = np.empty((n_steps, n_neurons), dtype=np.int32)
    num_clipped = np.empty((n_steps,), dtype=np.int32)
    previous = np.zeros((n_neurons,), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    int32_max = np.iinfo(np.int32).max
    progress_every = max(1, n_steps // 10)

    start = time.perf_counter()
    for t, state in enumerate(states_forward):
        eta_raw = A @ previous + B[int(state)]
        num_clipped[t] = int(np.count_nonzero(eta_raw > eta_clip))
        eta = np.minimum(eta_raw, eta_clip)
        poisson_mean = np.exp(eta) * float(dt)
        if not np.all(np.isfinite(poisson_mean)):
            raise FloatingPointError(
                f"Non-finite Poisson mean at generated step {t}; "
                f"eta_clip={eta_clip}, dt={dt}"
            )
        try:
            sampled = rng.poisson(poisson_mean)
        except ValueError as exc:
            raise ValueError(
                f"Poisson sampling failed at generated step {t}: {exc}"
            ) from exc
        if sampled.size and int(sampled.max()) > int32_max:
            raise OverflowError(
                f"Generated spike count exceeds int32 at step {t}: "
                f"max={int(sampled.max())}"
            )
        spikes[t] = sampled.astype(np.int32)
        previous = sampled.astype(np.float64, copy=False)

        if verb > 1 and (
            t == 0 or (t + 1) % progress_every == 0 or t + 1 == n_steps
        ):
            print(
                f"forward step {t + 1}/{n_steps}: state={int(state)} "
                f"spikes={int(sampled.sum())} clipped={int(num_clipped[t])} "
                f"elapsed={time.perf_counter() - start:.1f}s"
            )

    return spikes, num_clipped


def summarize_clipping(num_clipped, num_neurons):
    """Return the exact clipping histogram and four requested quartiles."""
    num_clipped = np.asarray(num_clipped, dtype=np.int64)
    if num_clipped.ndim != 1 or num_clipped.size == 0:
        raise ValueError("num_clipped must be a non-empty 1D array")
    if int(num_clipped.min()) < 0 or int(num_clipped.max()) > int(num_neurons):
        raise ValueError("num_clipped values are outside [0, num_neurons]")

    hist_k = np.arange(int(num_neurons) + 1, dtype=np.int32)
    hist = np.bincount(
        num_clipped, minlength=int(num_neurons) + 1
    ).astype(np.int64)
    hist_fraction = hist.astype(np.float64) / float(num_clipped.size)
    quartiles = np.quantile(
        num_clipped.astype(np.float64), QUARTILE_PROBABILITIES
    ).astype(np.float64)
    num_any = int(np.count_nonzero(num_clipped))
    return {
        "hist_k": hist_k,
        "hist": hist,
        "hist_fraction": hist_fraction,
        "quartiles": quartiles,
        "num_bins_any": num_any,
        "fraction_bins_any": float(num_any / num_clipped.size),
    }


def choose_output_name(requested_name, num_neurons, out_dir):
    """Use the requested stem or create a collision-free forwardN<N> hash."""
    if requested_name is not None:
        return validate_stem(requested_name, "--dataName")

    for _ in range(100):
        candidate = f"forwardN{int(num_neurons)}_{secrets.token_hex(3)}"
        spikes_f = os.path.join(out_dir, f"{candidate}.spikes.npz")
        truth_f = os.path.join(out_dir, f"{candidate}.forwardTruth.npz")
        if not os.path.exists(spikes_f) and not os.path.exists(truth_f):
            return candidate
    raise RuntimeError("Could not generate an unused random output name")


def output_metadata(
    args, model, data_name, states_forward, clipping, elapsed_sec,
    time_range_sec
):
    """Construct JSON-safe metadata shared by the two forward outputs."""
    source_provenance = copy.deepcopy(
        model["fit_metadata"].get("provenance", {})
    )
    provenance = {
        "state_transition_file": data_name,
        "dataName": data_name,
        "forward_generator": PROGRAM,
        "forward_input_model": args.inputModel,
        "forward_input_file": model["model_file"],
        "source_model_provenance": source_provenance,
    }
    num_steps = int(states_forward.size)
    state_len = int(model["S_hat"].size)
    state_counts = np.bincount(
        np.asarray(states_forward, dtype=np.int64),
        minlength=model["num_states"],
    )
    config = {
        "program": PROGRAM,
        "input_model": args.inputModel,
        "input_model_file": model["model_file"],
        "input_fit_type": model["fit_type"],
        "input_stage": model["stage"],
        "A_input_key": model["A_key"],
        "B_input_key": model["B_key"],
        "state_input_key": "S_hat",
        "state_replay_rule": "last_state_then_cyclic_wrap",
        "initial_spike_state": "all_zero",
        "input_state_history_bins": state_len,
        "state_history_complete_wraps": int(num_steps // state_len),
        "state_history_remainder_bins": int(num_steps % state_len),
        "initial_state": int(model["S_hat"][-1]),
        "num_steps": num_steps,
        "time_range_sec": np.asarray(time_range_sec, dtype=float).tolist(),
        "duration_sec": float(num_steps * model["time_step_sec"]),
        "num_neurons": int(model["num_neurons"]),
        "num_states": int(model["num_states"]),
        "state_counts": state_counts.astype(np.int64).tolist(),
        "state_fractions": (
            state_counts.astype(np.float64) / float(num_steps)
        ).tolist(),
        "time_step_sec": float(model["time_step_sec"]),
        "eta_clip": float(model["eta_clip"]),
        "eta_clip_mode": "upper_only",
        "eta_clip_count_rule": "eta_raw_strictly_greater_than_eta_clip",
        "seed": int(args.seed),
        "poisson_rng": "numpy.random.Generator.poisson",
        "clipping_quartile_labels": QUARTILE_LABELS.tolist(),
        "clipping_quartiles": clipping["quartiles"].tolist(),
        "num_bins_any_clipping": int(clipping["num_bins_any"]),
        "fraction_bins_any_clipping": float(clipping["fraction_bins_any"]),
        "elapsed_sec": float(elapsed_sec),
    }
    return provenance, config


def print_summary(
    data_name, model, states_forward, spikes, clipping, spikes_f, truth_f
):
    state_counts = np.bincount(
        states_forward, minlength=model["num_states"]
    )
    state_parts = "  ".join(
        f"s{state}:{int(count)}({count / states_forward.size:.1%})"
        for state, count in enumerate(state_counts)
    )
    quartile_text = "  ".join(
        f"{label}={value:g}"
        for label, value in zip(QUARTILE_LABELS, clipping["quartiles"])
    )

    print(f"\nForward generation complete: {data_name}")
    print(
        f"  stage={model['stage']}  A={model['A_key']} "
        f"B={model['B_key']}  N={model['num_neurons']} "
        f"M={model['num_states']}"
    )
    print(
        f"  steps={spikes.shape[0]}  dt={model['time_step_sec']:g} sec  "
        f"duration={spikes.shape[0] * model['time_step_sec']:g} sec  "
        f"eta_clip={model['eta_clip']:g}"
    )
    print(f"  state occupancy: {state_parts}")
    print(
        "  clipped-neuron count quartiles: "
        f"{quartile_text}"
    )
    print(
        f"  bins with clipping: {clipping['num_bins_any']}/"
        f"{spikes.shape[0]} ({clipping['fraction_bins_any']:.3%})"
    )
    print("  clipped-neuron histogram (k: bins, fraction):")
    for k in np.flatnonzero(clipping["hist"]):
        print(
            f"    {int(k):4d}: {int(clipping['hist'][k]):10d}  "
            f"{clipping['hist_fraction'][k]:.6f}"
        )
    print(f"  spikes:       {spikes_f}")
    print(f"  forwardTruth: {truth_f}")
    print("\nView generated spikes:")
    print(
        "  ./view_spikesTrain3.py --basePath $basePath "
        f"--dataName {data_name} --idxState -1 --time_range_sec 0 15 -p b"
    )


def main():
    args = parse_args()
    if not os.path.isdir(args.basePath):
        raise FileNotFoundError(
            f"--basePath does not exist or is not a directory: {args.basePath}"
        )
    validate_stem(args.inputModel, "--inputModel")

    start = time.perf_counter()
    model = load_forward_model(
        args.basePath, args.inputModel, verb=args.verb
    )
    time_range_sec, num_steps = time_range_to_num_steps(
        args.time_range_sec, model["time_step_sec"]
    )
    out_dir = os.path.join(args.basePath, "spikesData")
    os.makedirs(out_dir, exist_ok=True)
    data_name = choose_output_name(
        args.dataName, model["num_neurons"], out_dir
    )
    spikes_f = os.path.join(out_dir, f"{data_name}.spikes.npz")
    truth_f = os.path.join(out_dir, f"{data_name}.forwardTruth.npz")
    existing = [path for path in (spikes_f, truth_f) if os.path.exists(path)]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing output file(s): {existing}"
        )

    if args.verb > 0:
        print(
            f"Input model: {model['model_file']}\n"
            f"  fit_type={model['fit_type']} A={model['A_key']} "
            f"B={model['B_key']}\n"
            f"  N={model['num_neurons']} M={model['num_states']} "
            f"state_bins={model['S_hat'].size} "
            f"dt={model['time_step_sec']:g} eta_clip={model['eta_clip']:g}\n"
            f"  time_range_sec={time_range_sec.tolist()} "
            f"num_steps={num_steps}\n"
            f"Output name: {data_name}  seed={args.seed}"
        )

    states_forward, coefficients_forward = build_forward_states(
        model["S_hat"], num_steps, model["num_states"]
    )
    spikes, num_clipped = simulate_forward(
        model["A"],
        model["B"],
        states_forward,
        model["time_step_sec"],
        model["eta_clip"],
        args.seed,
        verb=args.verb,
    )
    clipping = summarize_clipping(num_clipped, model["num_neurons"])
    single_rates = (
        spikes.mean(axis=0, dtype=np.float64) / model["time_step_sec"]
    ).astype(np.float32)

    elapsed_sec = float(time.perf_counter() - start)
    provenance, config = output_metadata(
        args,
        model,
        data_name,
        states_forward,
        clipping,
        elapsed_sec,
        time_range_sec,
    )
    spikes_md = {
        "short_name": data_name,
        "data_type": "forwardPrism",
        "time_step_sec": float(model["time_step_sec"]),
        "poisson_eta_clip": float(model["eta_clip"]),
        "num_neurons": int(model["num_neurons"]),
        "provenance": provenance,
        "forward_generation": config,
    }
    truth_md = {
        "short_name": data_name,
        "data_type": "forwardPrismTruth",
        "truth_type": "fixed_fitted_model_forward_realization",
        "provenance": provenance,
        "forward_generation": config,
    }

    spikes_d = {
        "spikes": spikes.astype(np.int32, copy=False),
        "single_rates": single_rates,
    }
    truth_d = {
        "A_forward": model["A"].astype(np.float32),
        "B_forward": model["B"].astype(np.float32),
        "S_forward": states_forward.astype(np.int32, copy=False),
        "C_forward": coefficients_forward.astype(np.float32, copy=False),
        "num_clipped_neurons_time": num_clipped.astype(np.int32, copy=False),
        "clipped_neuron_hist_k": clipping["hist_k"],
        "clipped_neuron_hist": clipping["hist"],
        "clipped_neuron_hist_fraction": clipping["hist_fraction"],
        "clipped_neuron_quartile_probabilities": QUARTILE_PROBABILITIES,
        "clipped_neuron_quartile_labels": QUARTILE_LABELS,
        "clipped_neuron_quartiles": clipping["quartiles"],
    }

    # Write truth first so a visible spikes file always has its companion.
    write_data_npz(
        truth_d,
        truth_f,
        metaD=json_safe_metadata(truth_md),
        verb=args.verb > 1,
    )
    write_data_npz(
        spikes_d,
        spikes_f,
        metaD=json_safe_metadata(spikes_md),
        verb=args.verb > 1,
    )

    if args.verb > 0:
        print_summary(
            data_name,
            model,
            states_forward,
            spikes,
            clipping,
            spikes_f,
            truth_f,
        )


if __name__ == "__main__":
    main()
