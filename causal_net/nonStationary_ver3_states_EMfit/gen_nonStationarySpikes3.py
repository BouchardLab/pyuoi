#!/usr/bin/env python3
"""
Generate one non-stationary spike train using the ground-truth dictionary
(A_true, B_true) produced by gen_daleMatrices3.py.

The connectivity matrix A is shared across all states.  Each state m has its
own bias vector B_m.  At every time bin the effective bias is the convex
combination  B_eff = sum_m c_mt * B_m,  where c_t tracks a slowly-moving
target that aims at the one-hot vector of the current target state S_true[t].

State-sequence schedules (--schedule):
  mc  — Markov chain with geometric dwell (mean = true_dwell_sec / dt bins),
        minimum dwell = ceil(0.3 * true_dwell_sec / dt).
  rr  — Round-robin: states cycle 0,1,...,M-1 with exactly
        true_dwell_sec / dt bins per visit.

Smooth coefficient update at each bin (state_change_speed = nu):
  c_t  <- Simplex_project( c_{t-1} + clip(e_{S_t} - c_{t-1}, -nu, nu) )

Spike generation (Poisson GLM):
  eta_t   = A @ Y_{t-1} + B_eff
  lambda_t = exp( clip(eta_t, max=eta_clip) ) * dt
  Y_t     ~ Poisson(lambda_t)

An oracle state sequence S_oracle is also computed: at each t the state with
the highest Poisson log-likelihood given (A, {B_m}) and the observed spikes.

Output files saved to <basePath>/spikesData/:
  <dataName>.spikes.npz     — spikes (T, N) int32, single_rates (N,)
  <dataName>.prismTruth.npz — S_true, C_true, S_oracle, state_transition,
                               A_true, B_true, rate variance, Fano factor

Output shapes (T = num_steps, N = num_neurons, M = num_states):
  spikes            (T, N)    int32   — non-stationary spike counts
  single_rates      (N,)      float   — mean firing rates (Hz)
  S_true            (T,)      int32   — target state index per bin
  C_true            (T, M)    float32 — smooth simplex coefficients
  S_oracle          (T,)      int32   — oracle (max-likelihood) state per bin
  state_transition  (M, M)    float32 — Markov transition matrix used/implied
  A_true            (N, N)    float   — shared connectivity matrix (copy)
  B_true            (M, N)    float   — per-state bias vectors (copy)
"""

import os
import hashlib
import argparse
from pprint import pprint
import numpy as np

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
from gen_daleMatrices3 import estimate_rates


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, default=1, dest="verb", help="Verbosity level.")
    parser.add_argument("--basePath", default="/pscratch/sd/b/balewski/2025_causalNet_tmp/", help="head dir for input data")
    parser.add_argument("--inputStates", default=None, help="input simTruth base name")
    parser.add_argument("--dataName", type=str, default=None, help="output spikes base name (default: <inputStates>_<hash6>)")

    parser.add_argument("-t", "--num_steps", type=int, default=None, help="Number of time steps (default: from input evol_conf)")
    parser.add_argument("--state_change_speed", type=float, default=0.33, help="Max coefficient change per step.")
    parser.add_argument("--true_dwell_sec", type=float, default=1.0,
                        help="Mean dwell time in seconds to stay in a target state.")
    parser.add_argument("--seed", type=int, default=42, help="Optional random seed.")
    parser.add_argument("--schedule", choices=["mc", "rr"], default="mc",
                        help="State schedule: 'mc' = random Markov chain (default), "
                             "'rr' = deterministic round-robin (equal state coverage).")
 
    args = parser.parse_args()
    args.inpPath = os.path.join(args.basePath, "truthDale")
    args.outPath = os.path.join(args.basePath, "spikesData")
    if args.dataName is None:
        args.dataName = f"{args.inputStates}_{hashlib.md5(os.urandom(32)).hexdigest()[:6]}"

    print("myArg-program:", parser.prog)
    for arg in vars(args):
        print("myArg:", arg, getattr(args, arg))

    assert os.path.exists(args.basePath), f"missing basePath: {args.basePath}"
    assert os.path.exists(args.inpPath), f"missing inpPath: {args.inpPath}"
    os.makedirs(args.outPath, exist_ok=True)
    return args


def ensure_state_atoms(A_in, B_in):
    """Normalize A/B arrays to A(N,N), B(M,N) with shared A across states."""
    A = A_in
    B = B_in
    assert A.ndim == 2, f"A_true must be 2D, got shape={A.shape}"
    if B.ndim == 1:
        B = B[None, ...]
    assert B.ndim == 2, f"B_true must be 1D or 2D, got shape={B.shape}"
    assert A.shape[0] == A.shape[1], f"A must be square, got {A.shape}"
    assert A.shape[0] == B.shape[1], f"neuron mismatch: A N={A.shape[0]}, B N={B.shape[1]}"
    return A.astype(float), B.astype(float)


def _print_state_stats(S_true, n_states, n_steps):
    """Print per-state counts and dwell statistics (shared by both schedules)."""
    counts = np.bincount(S_true, minlength=n_states)
    fracs = counts / n_steps
    parts = '  '.join(f's{m}:{counts[m]}({fracs[m]:.1%})' for m in range(n_states))
    print(f"Target state distribution (T={n_steps}): {parts}")

    dwell_lens = {m: [] for m in range(n_states)}
    run_state, run_len = S_true[0], 1
    for t in range(1, n_steps):
        if S_true[t] == run_state:
            run_len += 1
        else:
            dwell_lens[run_state].append(run_len)
            run_state, run_len = S_true[t], 1
    dwell_lens[run_state].append(run_len)
    for m in range(n_states):
        d = dwell_lens[m]
        print(f"  state {m}: {len(d)} episodes, dwell mean={np.mean(d):.1f} steps, "
              f"total={np.sum(d)} steps")


def build_target_states_mc(n_steps, n_states, dwell_steps, rng):
    """Random Markov chain target-state sequence with geometric dwell time.

    Each time step the chain either stays (prob = 1 - 1/dwell_steps) or
    transitions uniformly to one of the other states.  State coverage is
    uncontrolled — a single unlucky run can give very few bins to one state.
    """
    if n_states == 1:
        return np.zeros(n_steps, dtype=np.int32), np.ones((1, 1), dtype=np.float32)

    minDwellFrac = 0.3
    min_dwell_steps = max(1, int(np.ceil(minDwellFrac * float(dwell_steps))))
    p_exit = 1.0 / float(dwell_steps)
    p_stay = 1.0 - p_exit
    trans = np.full((n_states, n_states), p_exit / float(n_states - 1), dtype=float)
    np.fill_diagonal(trans, p_stay)

    S_true = np.zeros(n_steps, dtype=np.int32)
    state = 0
    run_len = 0
    for t in range(n_steps):
        if run_len < min_dwell_steps:
            next_state = state
        else:
            next_state = rng.choice(n_states, p=trans[state])
        S_true[t] = next_state
        if next_state == state:
            run_len += 1
        else:
            state = next_state
            run_len = 1

    _print_state_stats(S_true, n_states, n_steps)
    return S_true, trans.astype(np.float32)


def build_target_states_rr(n_steps, n_states, dwell_steps, rng):
    """Round-robin with fixed dwell_steps per visit.

    States cycle 0,1,2,0,1,2,...  Every visit lasts exactly dwell_steps
    steps (last visit of each state is trimmed to fit T).  Total steps per
    state is guaranteed to be T//M ± dwell_steps — no luck involved.
    """
    if n_states == 1:
        return np.zeros(n_steps, dtype=np.int32), np.ones((1, 1), dtype=np.float32)

    S_true = np.zeros(n_steps, dtype=np.int32)
    t = 0
    visit = 0
    while t < n_steps:
        state = visit % n_states                  # strict round-robin
        t_end = min(t + dwell_steps, n_steps)     # fixed dwell, trim at end
        S_true[t:t_end] = state
        t = t_end
        visit += 1

    # Transition matrix: same symmetric form as MC for metadata consistency
    p_exit = 1.0 / float(dwell_steps)
    p_stay = 1.0 - p_exit
    trans = np.full((n_states, n_states), p_exit / float(n_states - 1), dtype=float)
    np.fill_diagonal(trans, p_stay)

    _print_state_stats(S_true, n_states, n_steps)
    return S_true, trans.astype(np.float32)

def simulate_switching_poisson(n_steps, A, B_atoms, S_true, state_change_speed, dt, eta_clip, rng, verb=1):
    """Switching Poisson generator: smooth c_t toward one-hot target state S_true[t]."""
    n_states, n_neurons = B_atoms.shape
    spikes = np.zeros((n_steps, n_neurons), dtype=np.int32)
    C_true = np.zeros((n_steps, n_states), dtype=np.float32)

    c_curr = np.zeros(n_states, dtype=float)
    c_curr[S_true[0]] = 1.0

    for t in range(n_steps):
        target = np.zeros(n_states, dtype=float)
        target[S_true[t]] = 1.0

        diff = target - c_curr
        c_curr += np.clip(diff, -state_change_speed, state_change_speed)
        c_sum = np.sum(c_curr)
        if c_sum <= 0:
            c_curr = target.copy()
            c_sum = 1.0
        c_curr /= c_sum
        C_true[t] = c_curr

        if verb > 1 and t < 12:
            c_str = ", ".join([f"{x:.2f}" for x in c_curr])
            print(f"{t:<6} | {int(S_true[t]):<6} | [{c_str}]")

        A_eff = A
        B_eff = np.einsum("m,mj->j", c_curr, B_atoms)

        prev_y = spikes[t - 1].astype(float) if t > 0 else np.zeros(n_neurons, dtype=float)
        eta_t = A_eff @ prev_y + B_eff
        lambda_t = np.exp(np.clip(eta_t, max=eta_clip))
        spikes[t] = rng.poisson(lambda_t * dt).astype(np.int32)

    return spikes, C_true


def compute_oracle_states_comA(spikes, A, B_atoms, dt, eta_clip):
    """Oracle state by max Poisson log-likelihood using shared A and per-state B."""
    n_steps, n_neurons = spikes.shape
    n_states = B_atoms.shape[0]
    S_oracle = np.zeros((n_steps,), dtype=np.int32)
    prev_y = np.zeros(n_neurons, dtype=np.float64)
    for t in range(n_steps):
        y_curr = spikes[t].astype(np.float64)
        base = A @ prev_y
        eta = base[None, :] + B_atoms
        eta_c = np.clip(eta, max=eta_clip)
        lam = np.exp(eta_c) * dt
        scores = np.sum(y_curr * eta_c - lam, axis=1)
        S_oracle[t] = int(np.argmax(scores))
        prev_y = y_curr
    return S_oracle


def compute_oracle_score(S_true, S_oracle, n_states):
    """Return overall and per-state agreement between oracle and target states."""
    if S_true is None or S_oracle is None:
        return None, []
    nmin = min(S_true.shape[0], S_oracle.shape[0])
    if nmin <= 0:
        return None, []
    avr_score = float(np.mean(S_true[:nmin] == S_oracle[:nmin]))
    score_per_state = []
    for m in range(n_states):
        mask = S_true[:nmin] == m
        if mask.sum() > 0:
            score_per_state.append(float((S_oracle[:nmin][mask] == m).mean()))
        else:
            score_per_state.append(float("nan"))
    return avr_score, score_per_state


def main():
    args = get_parser()
    print('gen non-stationary spikes args:', vars(args), '\n')
    np.set_printoptions(precision=3, suppress=True)

    daleFF = os.path.join(args.inpPath, f"{args.inputStates}.simTruth.npz")
    daleD, daleMD = read_data_npz(daleFF, verb=args.verb > 0)
    if args.verb > 1:
        print("\nInput simTruth metadata:")
        pprint(daleMD)

    assert isinstance(daleMD, dict), "Expected dictionary metadata in simTruth file"
    dale_conf_in = daleMD["dale_conf"]
    evol_conf_in = daleMD["evol_conf"]
    proven_in=daleMD['provenance']
   
    step_size = evol_conf_in["step_size"]
    var_time_window_sec = 5 #(sec)
    max_samples = 100_000 # time steps

    if args.num_steps is None:
        args.num_steps = int(evol_conf_in["num_steps"])

    assert args.num_steps >= 100
    assert step_size > 0
    assert args.state_change_speed > 0
    assert args.true_dwell_sec > 0.0
    assert max_samples >= 100
    assert var_time_window_sec > 0

    true_dwell_steps = max(1, int(np.ceil(float(args.true_dwell_sec) / float(step_size))))

    A_atoms, B_atoms = ensure_state_atoms(daleD["A_true"], daleD["B_true"])
    n_states, n_neurons = B_atoms.shape
    num_excite = int(dale_conf_in["num_excite"])
    num_excite = min(max(1, num_excite), n_neurons - 1)

    rng = np.random.default_rng(args.seed)

    schedule_fn = {"mc": build_target_states_mc,
                   "rr": build_target_states_rr}[args.schedule]
    S_true, transition_matrix = schedule_fn(args.num_steps, n_states, true_dwell_steps, rng)

    spikes, C_true = simulate_switching_poisson(
        n_steps=args.num_steps,
        A=A_atoms,
        B_atoms=B_atoms,
        S_true=S_true,
        state_change_speed=args.state_change_speed,
        dt=step_size,
        eta_clip=evol_conf_in['poisson_eta_clip'],
        rng=rng,
        verb=args.verb,
    )
    S_oracle = compute_oracle_states_comA(
        spikes=spikes,
        A=A_atoms,
        B_atoms=B_atoms,
        dt=step_size,
        eta_clip=evol_conf_in['poisson_eta_clip'],
    )

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
    oracle_score, oracle_score_per_state = compute_oracle_score(S_true, S_oracle, n_states)
    true_dwell_sec_req = float(args.true_dwell_sec)
    true_dwell_sec_eff = float(true_dwell_steps * step_size)

    evol_conf = {
        "num_steps": int(args.num_steps),
        "step_size": float(step_size),
        "evol_time": float(args.num_steps * step_size),
        "state_change_speed": float(args.state_change_speed),
        "true_dwell_sec": true_dwell_sec_req,
        "true_dwell_sec_eff": true_dwell_sec_eff,
        "true_dwell_steps": int(true_dwell_steps),
        "true_dwell_time_sec": true_dwell_sec_eff,  # legacy alias
        "num_states": int(n_states),
        "seed": args.seed,
        "state_schedule": args.schedule,
        "num_states": int(n_states),
        "max_samples": int(max_samples),
        "truth_input_name" : args.inputStates,
    }

    dale_conf = dict(dale_conf_in)
    dale_conf["num_neurons"] = int(dale_conf["num_neurons"])
    dale_conf["num_excite"] = int(dale_conf["num_excite"])

    stats_meta = dict(stats_dict)
    for key in ("num_excitatory", "num_inhibitory", "num_neurons", "num_steps"):
        stats_meta.pop(key, None)

    spikesD = {
        "spikes": spikes.astype(np.int32),
        "single_rates": rates_dict["single_rates"],
    }
    spikesMD = {
        "data_type": "simPrism",
        "time_step_sec": evol_conf_in["step_size"],
        'poisson_eta_clip': evol_conf_in['poisson_eta_clip'],
        'provenance': proven_in
    }
    proven_in['state_transition_file']= args.dataName
  
    prismTruthD = {
        "A_true": daleD["A_true"],
        "B_true": daleD["B_true"],
        "S_true": S_true.astype(np.int32),
        "C_true": C_true.astype(np.float32),
        "S_oracle": S_oracle.astype(np.int32),
        "state_transition": transition_matrix.astype(np.float32),
        "sigle_rates_var": rates_dict["sigle_rates_var"],
        "single_fano_fact": rates_dict["single_fano_fact"],
     }

    
    prismTruthMD = {
        "short_name": args.dataName,
        "data_type": "simPrism",
        "var_time_window_sec": float(var_time_window_sec),
        "dale_conf": dale_conf,
        "evol_conf": evol_conf,
    }
    prismTruthMD['oracle_eval'] = {
        "avr_score": oracle_score,
        "score_per_state": oracle_score_per_state,
    }
    

    #  "dale_simu_stats": [stats_meta],
    outFs = os.path.join(args.outPath, args.dataName + ".spikes.npz")
    outFt = os.path.join(args.outPath, args.dataName + ".prismTruth.npz")
    write_data_npz(spikesD, outFs, metaD=spikesMD)
    write_data_npz(prismTruthD, outFt, metaD=prismTruthMD)

    if args.verb > 1:
        print('\nspikes MD:'); pprint(spikesMD)
        print('\nprismTruth MD:'); pprint(prismTruthMD)

    if oracle_score is not None:
        target_bins_per_state = np.bincount(S_true, minlength=n_states)
        switches_to_per_state = np.zeros(n_states, dtype=np.int64)
        if S_true.shape[0] > 1:
            switch_idx = np.where(S_true[1:] != S_true[:-1])[0] + 1
            if switch_idx.size > 0:
                switches_to_per_state = np.bincount(
                    S_true[switch_idx], minlength=n_states
                ).astype(np.int64)
        print(f"gen, oracle avr score {oracle_score:.3f}, {args.dataName}")
        print(f"  {'state':>5s}  {'score':>5s}  {'bins':>7s}  {'switches_to':>11s}")
        print(f"  {'-----':>5s}  {'-----':>5s}  {'-------':>7s}  {'-----------':>11s}")
        for m, sc in enumerate(oracle_score_per_state):
            print(
                f"  {m:5d}  {sc:5.3f}  {int(target_bins_per_state[m]):7d}  "
                f"{int(switches_to_per_state[m]):11d}"
            )

    print("\n  ./view_spikesTrain3.py  --basePath $basePath   --dataName %s  --idxState -1 --time_range_sec 0 15   -p b    " % args.dataName)
    print("\n  ./prism_Estep_train.py --basePath $basePath   --dataName %s      " % args.dataName)
    print("\n  ./prism_Mstep_train3.py --basePath $basePath   --dataName %s      " % args.dataName)

    print("  ./fit_lassoPoisson3.py  --basePath $basePath  --dataName   %s   --num_epochs  300  " % args.dataName)
   
    print("  ./fitPrismEM.sh   --basePath $basePath     --dataName %s  --num_states %d  --num_em_iters 4 --m_epochs 16  --time_range_sec 0 80 " % (args.dataName,n_states))
    print("  ./bigLassoBoots.sh    --basePath $basePath     --dataName %s  --num_epochs 100  --dropDataFrac 0.33  --num_bootstraps 2   --bootsTag b2   --desyncTime  " % args.dataName)


if __name__ == "__main__":
    main()
