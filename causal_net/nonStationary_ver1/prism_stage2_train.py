#!/usr/bin/env python3
"""
PRISM Stage 2 - Coefficient Inference (train).

Goal: Given ground-truth dictionaries (A_true, B_true), infer the mixing
coefficients C_hat(T, M) from spikes alone using sequential projected
gradient descent (PGD) on the probability simplex.

Optionally applies a forward-backward smoother (--fb_smooth):
  1. Forward pass  : PGD left-to-right, anchor = c_{t-1}
  2. Backward pass : PGD right-to-left, anchor = c_{t+1}
  3. Average       : C_hat = project_simplex( (C_fwd + C_bwd) / 2 )

Dictionaries are NOT updated. This isolates and validates the coefficient
inference step before any joint optimization. 

Reads:
  <basePath>/spikesData/<dataName>.spikes.npz      (spikes, -> truthName)
  <basePath>/truthDale/<truthName>.simTruth.npz    (A_true, B_true)
  <basePath>/spikesData/<dataName>.prismTruth.npz  (C_true, S_true - NOT
                                                    used in training,
                                                    loaded for shape check)
Writes:
  <basePath>/prismFit/<dataName>_<hash6>.stage2.npz

Output arrays:
  C_hat       (T, M)  float32   inferred mixing coefficients
  C_fwd       (T, M)  float32   forward-only pass (always saved)
  C_bwd       (T, M)  float32   backward pass (zeros if --fb_smooth not set)
  loss_t      (T,)    float32   Poisson NLL per time step (forward pass)
  smooth_t    (T,)    float32   smoothness penalty per time step (forward pass)
  state_hat   (T,)    int32     argmax(C_hat) at each step
"""

import os
import argparse
import time
import hashlib
from pprint import pprint
import numpy as np
import torch

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz

LOG_EPS = 1e-10


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        dest="verb", help="Verbosity level.")
    parser.add_argument("--basePath",
                        default="/dataVault2026/neurodata_tmp2",
                        help="Head dir for all data.")
    parser.add_argument("--dataName", default=None,
                        help="Spikes base name, e.g. daleN100_46f1c4_b90619")
    parser.add_argument("--device", default="cuda",
                        help="PyTorch device: cuda or cpu.")
    # hyperparameters
    parser.add_argument("--lam2", type=float, default=10.,
                        help="Temporal smoothness weight lambda_2.")
    parser.add_argument("--lr", type=float, default=0.02,
                        help="PGD step size (learning rate).")
    parser.add_argument("--n_inner", type=int, default=300,
                        help="PGD inner iterations per time step.")
    parser.add_argument("--time_steps_range", type=int, nargs=2, default=None,
                        help="Clip time bins to [start, end), "
                             "e.g. --time_steps_range 100 5000")
    # NEW: forward-backward smoother
    parser.add_argument("--fb_smooth", action="store_true", default=False,
                        help="Apply forward-backward smoothing pass. "
                             "Runs PGD in both directions then averages "
                             "on simplex. Roughly halves L-inf error "
                             "at 2x compute cost.")

    args = parser.parse_args()
    args.inpTruth  = os.path.join(args.basePath, "truthDale")
    args.inpSpikes = os.path.join(args.basePath, "spikesData")
    args.outPath   = os.path.join(args.basePath, "prismFit")

    print("myArg-program:", parser.prog)
    for arg in vars(args):
        print("myArg:", arg, getattr(args, arg))

    assert args.dataName is not None, "must provide --dataName"
    assert os.path.exists(args.basePath),  f"missing basePath: {args.basePath}"
    assert os.path.exists(args.inpTruth),  f"missing truthDale: {args.inpTruth}"
    assert os.path.exists(args.inpSpikes), f"missing spikesData: {args.inpSpikes}"
    os.makedirs(args.outPath, exist_ok=True)
    return args


def project_simplex(v):
    """
    Project vector v (M,) onto the probability simplex:
      { x : x_m >= 0, sum_m x_m = 1 }
    O(M log M) algorithm (Duchi et al. 2008).
    Works on CPU or GPU tensors.
    """
    M    = v.shape[0]
    u, _ = torch.sort(v, descending=True)
    idx  = torch.arange(1, M + 1, dtype=v.dtype, device=v.device)
    css  = torch.cumsum(u, dim=0)
    rho  = int(torch.nonzero(u * idx > (css - 1.0)).max())
    theta = (css[rho] - 1.0) / float(rho + 1)
    return torch.clamp(v - theta, min=0.0)


def project_simplex_rows(V):
    """
    Project each row of V (T, M) onto the simplex.
    Used for the FB averaging step.
    """
    T = V.shape[0]
    out = torch.zeros_like(V)
    for t in range(T):
        out[t] = project_simplex(V[t])
    return out


def precompute_features(A, B, Y_prev):
    """
    Compute per-state feature vectors f_m = A_m @ Y_prev + B_m for all m.

    Args:
        A      : (M, N, N) tensor
        B      : (M, N)    tensor
        Y_prev : (N,)      tensor
    Returns:
        F      : (M, N)    tensor   one row per state
    """
    return torch.einsum("mij,j->mi", A, Y_prev) + B   # (M, N)


def pgd_step(c, F, Y_t, dt, lam2, c_anchor, lr, n_inner, eta_clip):
    """
    Run n_inner steps of projected gradient descent on the simplex for c_t.

    Loss(c) = NLL(c) + lam2 * ||c - c_anchor||^2
    where  eta_i    = sum_m c_m * F_mi
           lambda_i = exp(clip(eta_i)) * dt

    c_anchor is c_{t-1} in the forward pass, c_{t+1} in the backward pass.

    Args:
        c        : (M,) initial coefficient vector (on simplex)
        F        : (M, N) feature matrix for this time step
        Y_t      : (N,)  observed spikes (float)
        dt       : float
        lam2     : float
        c_anchor : (M,) smoothness anchor (previous or next step)
        lr       : float step size
        n_inner  : int
        eta_clip : float

    Returns:
        c        : (M,) updated coefficient (on simplex)
        loss_nll : float  final Poisson NLL
        loss_sm  : float  final smoothness term
    """
    for _ in range(n_inner):
        eta   = torch.clamp(c @ F, min=-eta_clip, max=eta_clip)  # (N,)
        lam   = torch.exp(eta) * dt                               # (N,)
        resid = lam - Y_t                                         # (N,)
        g_nll = F @ resid                                         # (M,)
        g_sm  = 2.0 * lam2 * (c - c_anchor)                      # (M,)
        c     = project_simplex(c - lr * (g_nll + g_sm))

    with torch.no_grad():
        eta      = torch.clamp(c @ F, min=-eta_clip, max=eta_clip)
        lam      = torch.exp(eta) * dt
        loss_nll = (lam - Y_t * torch.log(lam + LOG_EPS)).sum().item()
        loss_sm  = float(lam2 * ((c - c_anchor) ** 2).sum().item())

    return c, loss_nll, loss_sm


# ------------------------------------------------------------------ #
#  Forward pass  (unchanged logic from original)                      #
# ------------------------------------------------------------------ #
def run_forward_pass(A, B, spikes, dt, eta_clip,
                     lam2, lr, n_inner, device, verb=1):
    """
    Sequential PGD left-to-right.  Anchor at t is C_fwd[t-1].

    Returns:
        C_fwd    : (T, M) cpu tensor
        loss_t   : (T,)   cpu tensor
        smooth_t : (T,)   cpu tensor
    """
    T, N = spikes.shape
    M    = A.shape[0]

    C_fwd    = torch.zeros(T, M, dtype=torch.float32)
    loss_t   = torch.zeros(T,    dtype=torch.float32)
    smooth_t = torch.zeros(T,    dtype=torch.float32)

    c = torch.full((M,), 1.0 / M, dtype=torch.float32, device=device)

    if verb > 0:
        print(f"\nForward pass: T={T}, N={N}, M={M}")
        print(f"  lam2={lam2}  lr={lr}  n_inner={n_inner}")

    log_interval = max(1, T // 10)
    t0_wall = time.time()

    for t in range(T):
        Y_prev = spikes[t-1].float() if t > 0 \
                 else torch.zeros(N, device=device)
        Y_t    = spikes[t].float()
        F      = precompute_features(A, B, Y_prev)

        c_anchor = C_fwd[t-1].to(device) if t > 0 else c.clone()
        c, l_nll, l_sm = pgd_step(
            c, F, Y_t, dt, lam2, c_anchor, lr, n_inner, eta_clip)

        C_fwd[t]    = c.cpu()
        loss_t[t]   = l_nll
        smooth_t[t] = l_sm

        if verb > 0 and (t % log_interval == 0 or t == T - 1):
            print(f"  fwd t={t:5d}/{T}  "
                  f"NLL={l_nll:.3f}  smooth={l_sm:.4f}  "
                  f"state={int(c.argmax())}  "
                  f"elaT={time.time()-t0_wall:.1f}s")

    return C_fwd, loss_t, smooth_t


# ------------------------------------------------------------------ #
#  NEW: Backward pass                                                 #
# ------------------------------------------------------------------ #
def run_backward_pass(A, B, spikes, dt, eta_clip,
                      lam2, lr, n_inner, device, verb=1):
    """
    Sequential PGD right-to-left.  Anchor at t is C_bwd[t+1].

    Note on features:
        At time step t the forward model uses Y_{t-1} to predict Y_t.
        In the backward pass we still want to evaluate how well c_t
        explains the spikes at t, so we still use F_t = A @ Y_{t-1} + B.
        The only thing that changes is the smoothness anchor direction:
        we anchor to the NEXT step instead of the previous one.

    Returns:
        C_bwd  : (T, M) cpu tensor
    """
    T, N = spikes.shape
    M    = A.shape[0]

    C_bwd = torch.zeros(T, M, dtype=torch.float32)

    # initialise from the last forward coefficient (warm start)
    c = torch.full((M,), 1.0 / M, dtype=torch.float32, device=device)

    if verb > 0:
        print(f"\nBackward pass: T={T}")

    log_interval = max(1, T // 10)
    t0_wall = time.time()

    for t in range(T - 1, -1, -1):           # T-1 down to 0
        Y_prev = spikes[t-1].float() if t > 0 \
                 else torch.zeros(N, device=device)
        Y_t    = spikes[t].float()
        F      = precompute_features(A, B, Y_prev)

        # anchor is the NEXT step (already filled because we go right-to-left)
        c_anchor = C_bwd[t+1].to(device) if t < T - 1 else c.clone()
        c, l_nll, _ = pgd_step(
            c, F, Y_t, dt, lam2, c_anchor, lr, n_inner, eta_clip)

        C_bwd[t] = c.cpu()

        if verb > 0 and (t % log_interval == 0 or t == 0):
            print(f"  bwd t={t:5d}/{T}  "
                  f"NLL={l_nll:.3f}  "
                  f"state={int(c.argmax())}  "
                  f"elaT={time.time()-t0_wall:.1f}s")

    return C_bwd


# ------------------------------------------------------------------ #
#  NEW: FB average on simplex                                         #
# ------------------------------------------------------------------ #
def fb_average(C_fwd, C_bwd, device):
    """
    Average forward and backward coefficient arrays and re-project onto
    the probability simplex row-by-row.

        C_hat[t] = project_simplex( (C_fwd[t] + C_bwd[t]) / 2 )

    The average of two simplex points is already on the simplex
    (convex set), so projection is a no-op in exact arithmetic.
    We project anyway to correct any floating-point drift.

    Args:
        C_fwd : (T, M) cpu float32 tensor
        C_bwd : (T, M) cpu float32 tensor
    Returns:
        C_hat : (T, M) cpu float32 tensor
    """
    avg   = (C_fwd + C_bwd) * 0.5          # already on simplex in theory
    C_hat = project_simplex_rows(avg)       # correct fp drift
    return C_hat


# ------------------------------------------------------------------ #
#  Main inference dispatcher                                          #
# ------------------------------------------------------------------ #
def run_coeff_inference(A, B, spikes, dt, eta_clip,
                        lam2, lr, n_inner, fb_smooth, device, verb=1):
    """
    Dispatch forward-only or forward-backward inference.

    Returns dict:
        C_hat    : (T, M)  final coefficients (averaged if fb_smooth)
        C_fwd    : (T, M)  forward pass only
        C_bwd    : (T, M)  backward pass (zeros if not fb_smooth)
        loss_t   : (T,)    NLL per step (forward pass)
        smooth_t : (T,)    smoothness per step (forward pass)
    """
    C_fwd, loss_t, smooth_t = run_forward_pass(
        A, B, spikes, dt, eta_clip,
        lam2, lr, n_inner, device, verb)

    if fb_smooth:
        C_bwd = run_backward_pass(
            A, B, spikes, dt, eta_clip,
            lam2, lr, n_inner, device, verb)
        C_hat = fb_average(C_fwd, C_bwd, device)
        if verb > 0:
            print("\nFB average applied.")
    else:
        C_bwd = torch.zeros_like(C_fwd)
        C_hat = C_fwd

    return {
        "C_hat":    C_hat,
        "C_fwd":    C_fwd,
        "C_bwd":    C_bwd,
        "loss_t":   loss_t,
        "smooth_t": smooth_t,
    }


def main():
    args = get_parser()
    np.set_printoptions(precision=3, suppress=True)

    # ---- load spikes ----
    spikesFF = os.path.join(args.inpSpikes, f"{args.dataName}.spikes.npz")
    spikesD, spikesMD = read_data_npz(spikesFF, verb=args.verb > 0)
    if args.verb > 1:
        print("\nspikes metadata:"); pprint(spikesMD)

    truthName = spikesMD["input_truth_name"]
    dt        = float(spikesMD["time_step_sec"])
    eta_clip  = float(spikesMD["poisson_eta_clip"])

    # ---- load simTruth ----
    truthFF = os.path.join(args.inpTruth, f"{truthName}.simTruth.npz")
    trueD, trueMD = read_data_npz(truthFF, verb=args.verb > 0)
    if args.verb > 1:
        print("\nsimTruth metadata:"); pprint(trueMD)

    # ---- load prismTruth for shape check only ----
    prismTruthFF = os.path.join(args.inpSpikes,
                                f"{args.dataName}.prismTruth.npz")
    prismTruthD, _ = read_data_npz(prismTruthFF, verb=args.verb > 0)

    # ---- extract tensors ----
    A_true = torch.tensor(trueD["A_true"], dtype=torch.float32)
    B_true = torch.tensor(trueD["B_true"], dtype=torch.float32)
    spikes = torch.tensor(spikesD["spikes"], dtype=torch.int32)

    if A_true.ndim == 2: A_true = A_true.unsqueeze(0)
    if B_true.ndim == 1: B_true = B_true.unsqueeze(0)

    M, N, _ = A_true.shape
    T       = spikes.shape[0]
    M_check = prismTruthD["C_true"].shape[1]
    assert M == M_check, f"M mismatch: simTruth={M}, prismTruth={M_check}"

    if args.time_steps_range is not None:
        t_lo, t_hi = args.time_steps_range
        t_lo = max(0, t_lo)
        t_hi = min(T, t_hi)
        spikes = spikes[t_lo:t_hi]
        T = spikes.shape[0]
        print(f"Clipped to time bins [{t_lo}, {t_hi}), T={T}")

    if args.verb > 0:
        print(f"\nShapes: A={tuple(A_true.shape)}, "
              f"B={tuple(B_true.shape)}, "
              f"spikes={tuple(spikes.shape)}, dt={dt}")
        print(f"fb_smooth = {args.fb_smooth}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.verb > 0:
        print(f"Using device: {device}")

    A_dev = A_true.to(device)
    B_dev = B_true.to(device)
    Y_dev = spikes.to(device)

    # ---- run inference ----
    t0 = time.time()
    results = run_coeff_inference(
        A_dev, B_dev, Y_dev, dt, eta_clip,
        lam2      = args.lam2,
        lr        = args.lr,
        n_inner   = args.n_inner,
        fb_smooth = args.fb_smooth,
        device    = device,
        verb      = args.verb,
    )
    elapsed_sec = time.time() - t0
    print(f"\nTotal elapsed time: {elapsed_sec:.1f} sec")

    C_hat     = results["C_hat"].numpy()
    state_hat = C_hat.argmax(axis=1).astype(np.int32)

    # ---- save ----
    outD = {
        "C_hat":     C_hat.astype(np.float32),
        "C_fwd":     results["C_fwd"].numpy().astype(np.float32),
        "C_bwd":     results["C_bwd"].numpy().astype(np.float32),
        "loss_t":    results["loss_t"].numpy().astype(np.float32),
        "smooth_t":  results["smooth_t"].numpy().astype(np.float32),
        "state_hat": state_hat,
    }
    hash6  = hashlib.md5(os.urandom(32)).hexdigest()[:6]
    outName = f"{args.dataName}_{hash6}"
    outMD = {
        "data_type":         "prismStage2",
        "short_name":        outName,
        "input_spikes_name": args.dataName,
        "input_truth_name":  truthName,
        "time_step_sec":     dt,
        "stage":             2,
        "eta_clip":          eta_clip,
        "num_neurons":       int(N),
        "num_steps":         int(T),
        "num_states":        int(M),
        "device":            str(device),
        "lam2":              args.lam2,
        "lr":                args.lr,
        "n_inner":           args.n_inner,
        "fb_smooth":         args.fb_smooth,
        "mean_nll":          float(results["loss_t"].mean().item()),
        "train_time_sec":    round(elapsed_sec, 1),
        "time_steps_range":  args.time_steps_range,
    }

    outFF = os.path.join(args.outPath, f"{outName}.stage2.npz")
    write_data_npz(outD, outFF, metaD=outMD)

    if args.verb > 1:
        print("\nstage2 metadata:"); pprint(outMD)

    print(f"\n  ./prism_stage2_eval.py  --basePath $basePath"
          f"  --dataName {outName}")


if __name__ == "__main__":
    main()
