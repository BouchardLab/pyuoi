#!/usr/bin/env python3
"""
PRISM Stage 1 - Forward Model Validation (train/compute).

Goal: Given ground-truth parameters (A_true, B_true, C_true), compute
the forward Poisson model and evaluate the deviance against observed spikes.
No optimization is performed. This validates the generative model,
numerical stability (eta clipping, log-floor), and loss implementation
before any fitting begins.

Reads:
  <basePath>/truthDale/<truthName>.simTruth.npz   (A_true, B_true)
  <basePath>/spikesData/<dataName>.spikes.npz     (spikes)
  <basePath>/spikesData/<dataName>.prismTruth.npz (C_true, S_true)

Writes:
  <basePath>/prismFit/<dataName>_s1.stage1.npz

Output arrays in .stage1.npz:
  eta_t          (T, N) float32   internal potentials
  lambda_t       (T, N) float32   predicted rates
  deviance_t     (T,)   float32   per-time-step Poisson deviance
  deviance_n     (N,)   float32   per-neuron Poisson deviance (summed over t)
  pred_rates     (N,)   float32   time-averaged predicted firing rate
  obs_rates      (N,)   float32   time-averaged observed firing rate
"""

import os
import argparse
from pprint import pprint
import numpy as np
import torch

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz


ETA_CLIP = 20.0      # exp(20) ~ 5e8, safe in float32
LOG_EPS  = 1e-10     # floor for log(lambda)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        dest="verb", help="Verbosity level.")
    parser.add_argument("--basePath",
                        default="/dataVault2026/neurodata_tmp2",
                        help="Head dir for all data.")
    parser.add_argument("--truthName", default=None,
                        help="simTruth base name, e.g. daleN100_46f1c4")
    parser.add_argument("--dataName", default=None,
                        help="Spikes base name, e.g. daleN100_46f1c4_b90619")
    parser.add_argument("--device", default="cuda",
                        help="PyTorch device: cuda or cpu.")

    args = parser.parse_args()
    args.inpTruth  = os.path.join(args.basePath, "truthDale")
    args.inpSpikes = os.path.join(args.basePath, "spikesData")
    args.outPath   = os.path.join(args.basePath, "prismFit")

    print("myArg-program:", parser.prog)
    for arg in vars(args):
        print("myArg:", arg, getattr(args, arg))

    assert args.truthName is not None, "must provide --truthName"
    assert args.dataName  is not None, "must provide --dataName"
    assert os.path.exists(args.basePath),   f"missing basePath: {args.basePath}"
    assert os.path.exists(args.inpTruth),   f"missing truthDale: {args.inpTruth}"
    assert os.path.exists(args.inpSpikes),  f"missing spikesData: {args.inpSpikes}"
    os.makedirs(args.outPath, exist_ok=True)
    return args


def poisson_deviance(Y, lambda_t, log_eps=LOG_EPS):
    """
    Poisson deviance (negative log-likelihood up to constant):
      D = sum_i [ lambda_i - Y_i * log(lambda_i) ]
    Returns per-sample scalar (summed over neurons).

    Args:
        Y        : (N,) int tensor   observed spike counts
        lambda_t : (N,) float tensor predicted rates
    Returns:
        dev      : scalar float tensor
    """
    dev = lambda_t.sum() - (Y.float() * torch.log(lambda_t + log_eps)).sum()
    return dev


def run_forward_model(A_true, B_true, C_true, spikes, dt, eta_clip, device, verb=1):
    """
    Compute forward Poisson model using ground-truth parameters.

    Args:
        A_true  : (M, N, N) float32 tensor
        B_true  : (M, N)    float32 tensor
        C_true  : (T, M)    float32 tensor
        spikes  : (T, N)    int32   tensor
        dt      : float
        device  : torch.device

    Returns dict of (T,N) or (N,) tensors:
        eta_t, lambda_t, deviance_t, deviance_n, pred_rates, obs_rates
    """
    T, N = spikes.shape
    M    = A_true.shape[0]

    # Move to device
    A  = A_true.to(device)   # (M, N, N)
    B  = B_true.to(device)   # (M, N)
    C  = C_true.to(device)   # (T, M)
    Y  = spikes.to(device)   # (T, N)

    eta_out  = torch.zeros(T, N, dtype=torch.float32, device=device)
    lam_out  = torch.zeros(T, N, dtype=torch.float32, device=device)
    dev_t    = torch.zeros(T,    dtype=torch.float32, device=device)

    if verb > 0:
        print(f"\nRunning forward model: T={T}, N={N}, M={M}, dt={dt}")

    for t in range(T):
        c_t  = C[t]                              # (M,)
        # Effective connectivity and bias
        A_eff = torch.einsum("m,mij->ij", c_t, A)   # (N, N)
        B_eff = torch.einsum("m,mj->j",  c_t, B)    # (N,)

        Y_prev = Y[t-1].float() if t > 0 else torch.zeros(N, device=device)

        eta_t = A_eff @ Y_prev + B_eff              # (N,)
        eta_t = torch.clamp(eta_t, min=-eta_clip, max=eta_clip)    # numerical safety

        lam_t = torch.exp(eta_t) * dt               # (N,)

        eta_out[t] = eta_t
        lam_out[t] = lam_t
        dev_t[t]   = poisson_deviance(Y[t], lam_t)

        if verb > 1 and t < 5:
            print(f"  t={t:4d} | eta min/max={eta_t.min():.3f}/{eta_t.max():.3f}"
                  f" | lam mean={lam_t.mean():.4f}"
                  f" | dev={dev_t[t]:.4f}")

    # Per-neuron deviance: sum over time, computed from lam and Y arrays
    dev_n = lam_out.sum(dim=0) \
            - (Y.float() * torch.log(lam_out + LOG_EPS)).sum(dim=0)

    pred_rates = lam_out.mean(dim=0)   # mean predicted rate per neuron
    obs_rates  = Y.float().mean(dim=0) # mean observed rate per neuron

    if verb > 0:
        total_dev = dev_t.sum().item()
        print(f"\nTotal Poisson deviance  : {total_dev:.4f}")
        print(f"Deviance per time step  : {total_dev/T:.4f}")
        print(f"Deviance per neuron*step: {total_dev/(T*N):.6f}")
        print(f"Pred rate mean/std      : {pred_rates.mean():.4f} / {pred_rates.std():.4f}")
        print(f"Obs  rate mean/std      : {obs_rates.mean():.4f}  / {obs_rates.std():.4f}")

    return {
        "eta_t":      eta_out.cpu(),
        "lambda_t":   lam_out.cpu(),
        "deviance_t": dev_t.cpu(),
        "deviance_n": dev_n.cpu(),
        "pred_rates": pred_rates.cpu(),
        "obs_rates":  obs_rates.cpu(),
    }


def main():
    args = get_parser()
    np.set_printoptions(precision=3, suppress=True)

    # ---- load simTruth (A_true, B_true) ----
    truthFF = os.path.join(args.inpTruth, f"{args.truthName}.simTruth.npz")
    trueD, trueMD = read_data_npz(truthFF, verb=args.verb > 0)
    if args.verb > 1:
        print("\nsimTruth metadata:"); pprint(trueMD)

    # ---- load spikes ----
    spikesFF = os.path.join(args.inpSpikes, f"{args.dataName}.spikes.npz")
    spikesD, spikesMD = read_data_npz(spikesFF, verb=args.verb > 0)
    if args.verb > 1:
        print("\nspikes metadata:"); pprint(spikesMD)

    # ---- load prismTruth (C_true, S_true) ----
    prismTruthFF = os.path.join(args.inpSpikes, f"{args.dataName}.prismTruth.npz")
    prismTruthD, prismTruthMD = read_data_npz(prismTruthFF, verb=args.verb > 0)
    if args.verb > 1:
        print("\nprismTruth metadata:"); pprint(prismTruthMD)

    # ---- extract arrays ----
    A_true = torch.tensor(trueD["A_true"],         dtype=torch.float32)
    B_true = torch.tensor(trueD["B_true"],         dtype=torch.float32)
    C_true = torch.tensor(prismTruthD["C_true"],   dtype=torch.float32)
    spikes = torch.tensor(spikesD["spikes"],       dtype=torch.int32)
    dt     = float(spikesMD["time_step_sec"])
    eta_clip     = float(spikesMD["poisson_eta_clip"])

    # Ensure A, B are state-first: (M,N,N), (M,N)
    if A_true.ndim == 2:
        A_true = A_true.unsqueeze(0)
    if B_true.ndim == 1:
        B_true = B_true.unsqueeze(0)

    M, N, _ = A_true.shape
    T       = spikes.shape[0]
    if args.verb > 0:
        print(f"\nShapes: A={tuple(A_true.shape)}, B={tuple(B_true.shape)}, "
              f"C={tuple(C_true.shape)}, spikes={tuple(spikes.shape)}, dt={dt}")

    # ---- device ----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.verb > 0:
        print(f"Using device: {device}")

    # ---- forward model ----
    results = run_forward_model(A_true, B_true, C_true, spikes, dt, eta_clip, device,  verb=args.verb)

    # ---- save ----
    outD = {k: v.numpy().astype(np.float32) for k, v in results.items()}

    outMD = {
        "data_type":        "prismStage1",
        "short_name":       args.dataName,
        "input_truth_name": args.truthName,
        "time_step_sec":    dt,
        "stage":            1,
        "eta_clip":         eta_clip,
        "log_eps":          LOG_EPS,
        "num_neurons":      int(N),
        "num_steps":        int(T),
        "num_states":       int(M),
        "device":           str(device),
        "total_deviance":   float(results["deviance_t"].sum().item()),
    }

    outFF = os.path.join(args.outPath, f"{args.dataName}_s1.stage1.npz")
    write_data_npz(outD, outFF, metaD=outMD)

    if args.verb > 1:
        print("\nstage1 metadata:"); pprint(outMD)

    print(f"\n  ./prism_stage1_eval.py  --basePath $basePath"
          f"  --dataName {args.dataName}  -p b  -X")


if __name__ == "__main__":
    main()
    
