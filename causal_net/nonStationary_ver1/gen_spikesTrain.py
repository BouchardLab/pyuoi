#!/usr/bin/env python3
"""
Generate one spike train from A/B dictionary atoms in <truthName>.simTruth.npz.

Generation logic matches toy_Mudrik_spiker.py (smooth switching coefficients
and Poisson spikes). I/O pattern follows view_spikesTrain.py for reading,
and output writing follows gen_daleMatrices.py.

Output files:
  <basePath>/spikesData/<dataName>.spikes.npz
  <basePath>/spikesData/<dataName>.prismTruth.npz

Output arrays in .spikes.npz:
  spikes            (T, N) int32
  single_rates      (N,)   float

Output arrays in .prismTruth.npz:
  S_true            (T,)   int32
  C_true            (T, M) float32
  state_transition  (M, M) float32
  sigle_rates_var   (N,)   float
  single_fano_fact  (N,)   float

"""

import os
import hashlib
import argparse
from pprint import pprint
import numpy as np

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
# Reuse the same stats function import path used by gen_daleMatrices.py
from gen_daleMatrices import estimate_rates


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, default=1, dest="verb", help="Verbosity level.")
    parser.add_argument("--basePath", default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="head dir for input data")
    parser.add_argument("--truthName", default=None, help="input simTruth base name")
    parser.add_argument("--dataName", type=str, default=None, help="output spikes base name (default: <truthName>_<hash6>)")
   
    parser.add_argument("-t", "--num_steps", type=int, default=None, help="Number of time steps (default: from input evol_conf)")
    parser.add_argument("--max_delta_c", type=float, default=0.02, help="Max coefficient change per step.")
    parser.add_argument("--dwell_steps", type=int, default=500, help="Mean number of steps to stay in a target state.")
    parser.add_argument("--seed", type=int, default=42, help="Optional random seed.")

    args = parser.parse_args()
    args.inpPath = os.path.join(args.basePath, "truthDale")
    args.outPath = os.path.join(args.basePath, "spikesData")
    if args.dataName is None:
        args.dataName = f"{args.truthName}_{hashlib.md5(os.urandom(32)).hexdigest()[:6]}"

    print("myArg-program:", parser.prog)
    for arg in vars(args):
        print("myArg:", arg, getattr(args, arg))

    assert os.path.exists(args.basePath), f"missing basePath: {args.basePath}"
    assert os.path.exists(args.inpPath), f"missing inpPath: {args.inpPath}"
    os.makedirs(args.outPath, exist_ok=True)
    return args


def ensure_state_atoms(A_in, B_in):
    """Normalize A/B arrays to state-first tensors: A(M,N,N), B(M,N)."""
    if A_in.ndim == 2:
        A = A_in[None, ...]
    else:
        A = A_in
    if B_in.ndim == 1:
        B = B_in[None, ...]
    else:
        B = B_in

    assert A.ndim == 3, f"A_true must be 2D or 3D, got shape={A.shape}"
    assert B.ndim == 2, f"B_true must be 1D or 2D, got shape={B.shape}"
    assert A.shape[0] == B.shape[0], f"state mismatch: A states={A.shape[0]}, B states={B.shape[0]}"
    assert A.shape[1] == A.shape[2], f"A must be square per state, got {A.shape}"
    assert A.shape[1] == B.shape[1], f"neuron mismatch: A N={A.shape[1]}, B N={B.shape[1]}"
    return A.astype(float), B.astype(float)


def build_target_states(n_steps, n_states, dwell_steps, rng):
    """Markov target-state sequence with geometric dwell time."""
    if n_states == 1:
        return np.zeros(n_steps, dtype=np.int32), np.ones((1, 1), dtype=np.float32)

    p_exit = 1.0 / float(dwell_steps)
    p_stay = 1.0 - p_exit
    trans = np.full((n_states, n_states), p_exit / float(n_states - 1), dtype=float)
    np.fill_diagonal(trans, p_stay)

    S_true = np.zeros(n_steps, dtype=np.int32)
    state = 0
    for t in range(n_steps):
        state = rng.choice(n_states, p=trans[state])
        S_true[t] = state
    return S_true, trans.astype(np.float32)


def simulate_switching_poisson(n_steps, A_atoms, B_atoms, S_true, max_delta_c, dt, rng, verb=1):
    """Switching Poisson generator: smooth c_t toward one-hot target state S_true[t]."""
    n_states, n_neurons = B_atoms.shape
    spikes = np.zeros((n_steps, n_neurons), dtype=np.int32)
    C_true = np.zeros((n_steps, n_states), dtype=np.float32)

    c_curr = np.zeros(n_states, dtype=float)
    c_curr[S_true[0]] = 1.0

    if verb > 0:
        print(f"{'Step':<6} | {'Target':<6} | {'Coefficients (c_mt)':<40}")
        print("-" * 70)

    for t in range(n_steps):
        target = np.zeros(n_states, dtype=float)
        target[S_true[t]] = 1.0

        diff = target - c_curr
        c_curr += np.clip(diff, -max_delta_c, max_delta_c)
        c_sum = np.sum(c_curr)
        if c_sum <= 0:
            c_curr = target.copy()
            c_sum = 1.0
        c_curr /= c_sum
        C_true[t] = c_curr

        if verb > 1 and t < 12:
            c_str = ", ".join([f"{x:.2f}" for x in c_curr])
            print(f"{t:<6} | {int(S_true[t]):<6} | [{c_str}]")

        A_eff = np.einsum("m,mij->ij", c_curr, A_atoms)
        B_eff = np.einsum("m,mj->j", c_curr, B_atoms)

        prev_y = spikes[t - 1].astype(float) if t > 0 else np.zeros(n_neurons, dtype=float)
        eta_t = A_eff @ prev_y + B_eff
        lambda_t = np.exp(np.clip(eta_t, -5, 5))
        spikes[t] = rng.poisson(lambda_t * dt).astype(np.int32)

    return spikes, C_true


def main():
    args = get_parser()
    np.set_printoptions(precision=3, suppress=True)

    truthFF = os.path.join(args.inpPath, f"{args.truthName}.simTruth.npz")
    trueD, trueMD = read_data_npz(truthFF, verb=args.verb > 0)
    if args.verb > 1:
        print("\nInput simTruth metadata:")
        pprint(trueMD)

    assert isinstance(trueMD, dict), "Expected dictionary metadata in simTruth file"
    dale_conf_in = dict(trueMD["dale_conf"])
    evol_conf_in = dict(trueMD["evol_conf"])
    dale_stats0 = dict(trueMD["dale_simu_stats"][0])

    step_size = float(evol_conf_in["step_size"])
    var_time_window_sec = float(dale_stats0["var_time_window_sec"])
    max_samples = int(dale_stats0["max_samples"])
 
    if args.num_steps is None:
        args.num_steps = int(evol_conf_in["num_steps"])

    assert args.num_steps >= 100
    assert step_size > 0
    assert args.max_delta_c > 0
    assert args.dwell_steps >= 1
    assert max_samples >= 100
    assert var_time_window_sec > 0

    A_atoms, B_atoms = ensure_state_atoms(trueD["A_true"], trueD["B_true"])
    n_states, n_neurons = B_atoms.shape
    num_excite = int(dale_conf_in["num_excite"])
    num_excite = min(max(1, num_excite), n_neurons - 1)

    rng = np.random.default_rng(args.seed)
    S_true, transition_matrix = build_target_states(args.num_steps, n_states, args.dwell_steps, rng)

    spikes, C_true = simulate_switching_poisson(
        n_steps=args.num_steps,
        A_atoms=A_atoms,
        B_atoms=B_atoms,
        S_true=S_true,
        max_delta_c=args.max_delta_c,
        dt=step_size,
        rng=rng,
        verb=args.verb,
    )

    # Recompute stats using the same flow as gen_daleMatrices.py
    stats_dict, rates_dict, _ = estimate_rates(
        spikes,
        dt=step_size,
        num_excite=num_excite,
        max_samples=max_samples,
        varTwindow=var_time_window_sec,
        mxNn=5,
        verb=args.verb,
        spect_radius=None,
    )

    # Build new evol_conf for this generator
    evol_conf = {
        "num_steps": int(args.num_steps),
        "step_size": float(step_size),
        "evol_time": float(args.num_steps * step_size),
        "max_delta_c": float(args.max_delta_c),
        "dwell_steps": int(args.dwell_steps),
        "num_states": int(n_states),
        "seed": args.seed,
        "max_samples": int(max_samples),
        "var_time_window_sec": float(var_time_window_sec),
    }

    # Preserve input dale_conf when present; ensure core fields exist
    dale_conf = dict(dale_conf_in)
    dale_conf["num_neurons"] = int(dale_conf["num_neurons"])
    dale_conf["num_excite"] = int(dale_conf["num_excite"])

    # Keep dale_simu_stats compact in prismTruth metadata
    stats_meta = dict(stats_dict)
    for key in ("num_excitatory", "num_inhibitory", "num_neurons", "num_steps"):
        stats_meta.pop(key, None)

    # 1) Main spikes file: only spikes + single_rates with minimal metadata
    spikesD = {
        "spikes": spikes.astype(np.int32),  # requested format: (T, N) int32
        "single_rates": rates_dict["single_rates"],
    }
    spikesMD = {
        "data_type": "simPrism",
        "short_name": args.dataName,
        "time_step_sec": float(step_size),
    }

    # 2) Truth/aux file: all remaining arrays + remaining metadata
    prismTruthD = {
        "S_true": S_true.astype(np.int32),
        "C_true": C_true.astype(np.float32),
        "state_transition": transition_matrix.astype(np.float32),
        "sigle_rates_var": rates_dict["sigle_rates_var"],
        "single_fano_fact": rates_dict["single_fano_fact"],
    }
    prismTruthMD = {
        "short_name": args.dataName,
        "input_truth_name": args.truthName,
        "data_type": "simPrism",
        "var_time_window_sec": float(var_time_window_sec),
        "dale_conf": dale_conf,
        "evol_conf": evol_conf,
        "dale_simu_stats": [stats_meta],
    }

    outFs = os.path.join(args.outPath, args.dataName + ".spikes.npz")
    outFt = os.path.join(args.outPath, args.dataName + ".prismTruth.npz")
    write_data_npz(spikesD, outFs, metaD=spikesMD)
    write_data_npz(prismTruthD, outFt, metaD=prismTruthMD)

    if args.verb>1:
        print('\nspikes MD:'); pprint(spikesMD)
        print('\nprismTruth MD:'); pprint(prismTruthMD)

    print("\n  ./view_spikesTrain.py  --basePath $basePath   --dataName %s  --idxR -1  -p b     -X " % args.dataName)

if __name__ == "__main__":
    main()
 
