#!/usr/bin/env python3
"""
E-step training for non-stationary Poisson dLDS with simplex-constrained coefficients.
Uses ground-truth dictionaries (A_true/B_true) to fit c_t sequentially.
"""

import os
import time
import math
import argparse
import secrets
from pprint import pprint

import numpy as np
import torch

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
from UtilTorch import check_gpu_availability


def parse_args():
    parser = argparse.ArgumentParser(description="E-step training for prism model (simplex PGD)")
    parser.add_argument("--dataName", type=str, required=True, help="Base name of spikes file in spikesData/")
    parser.add_argument("--basePath", type=str, default="/pscratch/sd/b/balewski/2026_causalNet_tmp2/", help="Head dir for input/output data")
    parser.add_argument("--lambda2", type=float, default=0.1, help="Temporal smoothness strength")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of full passes over time range")
    parser.add_argument("--pgd_iter", type=int, default=8, help="PGD iterations per time step")
    parser.add_argument("--lr", type=float, default=0.1, help="PGD step size")
    parser.add_argument("--decode_dwell_sec", type=float, default=0.2, help="Expected state dwell time in seconds for Viterbi decoding")
    parser.add_argument("--chunk_size", type=int, default=2 * 1024, help="Time chunk size for progress")
    parser.add_argument("-T", "--time_range_sec", default=[0.0, 50.0], nargs=2, type=float, help="display data time range in seconds")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("-v", "--verb", type=int, default=1)
    return parser.parse_args()


def make_prism_out_base(data_name: str) -> str:
    hash6 = secrets.token_hex(3)  # 6 hex digits
    return f"{data_name}-Estep-{hash6}"


def project_to_simplex(v: torch.Tensor) -> torch.Tensor:
    """Project 1D tensor v onto the probability simplex."""
    if v.numel() == 1:
        return torch.ones_like(v)
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    if torch.any(cond):
        rho = torch.nonzero(cond, as_tuple=False)[-1, 0]
        theta = cssv[rho] / (rho + 1.0)
    else:
        theta = cssv[-1] / v.numel()
    w = torch.clamp(v - theta, min=0.0)
    return w


def viterbi_decode(c_hat, p_stay):
    """Viterbi decode most likely state sequence from c_hat with stay/switch prior."""
    eps = 1e-12
    T, M = c_hat.shape
    if M == 1:
        return np.zeros((T,), dtype=np.int64), np.ones((T,), dtype=np.float32)

    p_switch = (1.0 - p_stay) / float(M - 1)
    trans = np.full((M, M), p_switch, dtype=np.float64)
    np.fill_diagonal(trans, p_stay)
    log_trans = np.log(np.clip(trans, eps, 1.0))

    log_emit = np.log(np.clip(c_hat, eps, 1.0))
    delta = np.zeros((T, M), dtype=np.float64)
    psi = np.zeros((T, M), dtype=np.int64)

    delta[0] = log_emit[0]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_trans
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores[psi[t], np.arange(M)] + log_emit[t]

    path = np.zeros((T,), dtype=np.int64)
    path[T - 1] = int(np.argmax(delta[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path


def main():
    args = parse_args()
    inpPath = os.path.join(args.basePath, "spikesData")
    truthPath = os.path.join(args.basePath, "truthDale")
    outPath = os.path.join(args.basePath, "prismFit")
    assert os.path.exists(outPath)

    print('E-train args:', vars(args), '\n')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = check_gpu_availability()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    spikesFF = os.path.join(inpPath, f"{args.dataName}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF, verb=args.verb > 0)
    spikes = spikeD["spikes"]
    single_rates = spikeD.get("single_rates")

    args.time_step_sec = float(spikeMD["time_step_sec"])
    args.eta_clip = float(spikeMD["poisson_eta_clip"])

    prov = spikeMD["provenance"]
    truthF = prov["state_model_file"]

    truthFF = os.path.join(truthPath, f"{truthF}.simTruth.npz")
    trueD, trueMD = read_data_npz(truthFF, verb=args.verb > 1)
    A_true = trueD["A_true"]
    B_true = trueD["B_true"]

    assert A_true.ndim == 2, "A_true must have shape (N,N)"
    assert B_true.ndim == 2, "B_true must have shape (M,N)"
    N, N2 = A_true.shape
    assert N == N2, "A_true must be square"
    M = B_true.shape[0]
    assert B_true.shape[1] == N, "B_true shape mismatch"

    T_full, N_spk = spikes.shape
    assert N_spk == N, "Spike data N does not match A_true/B_true"

    # Time range in bins
    t0_sec, t1_sec = float(args.time_range_sec[0]), float(args.time_range_sec[1])
    if t1_sec < t0_sec:
        t0_sec, t1_sec = t1_sec, t0_sec
    start_bin = max(0, int(math.floor(t0_sec / args.time_step_sec)))
    end_bin = min(T_full - 1, int(math.floor(t1_sec / args.time_step_sec)))
    if end_bin <= start_bin:
        raise ValueError("time_range_sec too small; need at least two bins")

    # Slice spikes to selected range (inclusive end_bin)
    spikes_sub = spikes[start_bin:end_bin + 1]
    T_eff = spikes_sub.shape[0]
    T_pairs = T_eff - 1
    if args.verb > 0:
        print(f"time bins: start_bin={start_bin}  num_bins={T_eff}  (pairs={T_pairs})")

    # Build per-time data (pairs)
    Y_prev = torch.tensor(spikes_sub[:-1], dtype=torch.float32, device=device)
    Y_curr = torch.tensor(spikes_sub[1:], dtype=torch.float32, device=device)

    # Move dictionaries to GPU
    A_t = torch.tensor(A_true, dtype=torch.float32, device=device)
    B_t = torch.tensor(B_true, dtype=torch.float32, device=device)

    # Initialize c_hat on GPU
    c_hat = torch.full((T_eff, M), 1.0 / float(M), dtype=torch.float32, device=device)

    loss_epoch = []
    loss_nll_epoch = []
    loss_l2_epoch = []
    loss_time = np.zeros((T_pairs,), dtype=np.float64)
    loss_nll_time = np.zeros((T_pairs,), dtype=np.float64)
    loss_l2_time = np.zeros((T_pairs,), dtype=np.float64)
    log_dt = math.log(args.time_step_sec)

    t_start = time.time()
    torch.set_grad_enabled(False)
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        prev_c_hat = c_hat.clone()
        c_prev = prev_c_hat[0]
        nll_sum = 0.0
        l2_sum = 0.0
        g_data_sum = 0.0
        g_smooth_sum = 0.0
        g_count = 0

        if epoch == args.num_epochs:
            loss_time.fill(0.0)
            loss_nll_time.fill(0.0)
            loss_l2_time.fill(0.0)

        for t0_idx in range(1, T_eff, args.chunk_size):
            t1_idx = min(T_eff, t0_idx + args.chunk_size)
            for t in range(t0_idx, t1_idx):
                y_prev = Y_prev[t - 1]
                y_curr = Y_curr[t - 1]

                # Predictor columns (M x N) for this time step
                base = torch.matmul(A_t, y_prev)
                z = base[None, :] + B_t

                c_t = prev_c_hat[t].clone()
                for _ in range(args.pgd_iter):
                    eta = torch.matmul(c_t, z)
                    eta_x = torch.clamp(eta, max=args.eta_clip)
                    lam = torch.exp(eta_x) * args.time_step_sec
                    resid = lam - y_curr
                    g_data = torch.matmul(z, resid)
                    g_smooth = 2.0 * args.lambda2 * (c_t - c_prev)
                    g = g_data + g_smooth
                    g_data_sum += float(torch.linalg.norm(g_data).item())
                    g_smooth_sum += float(torch.linalg.norm(g_smooth).item())
                    g_count += 1
                    c_t = c_t - args.lr * g
                    c_t = project_to_simplex(c_t)

                eta = torch.matmul(c_t, z)
                eta_y = torch.clamp(eta, max=args.eta_clip)
                lam = torch.exp(eta_y) * args.time_step_sec
                dev = lam - y_curr * (eta + log_dt)
                loss_nll = dev.sum()
                loss_l2 = args.lambda2 * torch.sum((c_t - c_prev) ** 2)
                loss_t = loss_nll + loss_l2

                nll_sum += float(loss_nll.item())
                l2_sum += float(loss_l2.item())
                if epoch == args.num_epochs:
                    loss_time[t - 1] = float(loss_t.item())
                    loss_nll_time[t - 1] = float(loss_nll.item())
                    loss_l2_time[t - 1] = float(loss_l2.item())

                c_hat[t] = c_t
                c_prev = c_t

        nll_mean = nll_sum / float(T_pairs)
        l2_mean = l2_sum / float(T_pairs)
        loss_mean = nll_mean + l2_mean
        loss_epoch.append(loss_mean)
        loss_nll_epoch.append(nll_mean)
        loss_l2_epoch.append(l2_mean)

        if args.verb > 0:
            elaT = time.time() - t_start
            g_data_mean = g_data_sum / float(max(1, g_count))
            g_smooth_mean = g_smooth_sum / float(max(1, g_count))
            print(
                f"epoch {epoch:3d}  nll={nll_mean:.4e}  l2={l2_mean:.4e}  "
                f"loss={loss_mean:.4e}  |g_data|={g_data_mean:.3e}  "
                f"|g_smooth|={g_smooth_mean:.3e}  elaT={elaT:.1f}s"
            )
    print(f"Total training time: {time.time() - t_start:.1f}s")
    
    # Decode most probable state (Viterbi) and confidence level
    p_stay = float(math.exp(-args.time_step_sec / float(args.decode_dwell_sec)))
    c_hat_np = c_hat.detach().cpu().numpy()
    S_hat_np = viterbi_decode(c_hat_np, p_stay)
    if M == 1:
        S_hat_CL = torch.zeros((T_eff,), dtype=torch.float32, device=c_hat.device)
    else:
        eps = 1e-12
        ent = -(c_hat * torch.log(c_hat + eps)).sum(dim=1)
        S_hat_CL = (ent / math.log(M)).to(dtype=torch.float32)
    S_hat = torch.from_numpy(S_hat_np).to(dtype=torch.int64, device=c_hat.device)

    # Save results
    out_base = make_prism_out_base(args.dataName)
    out_name = f"{out_base}.prismEstep.npz"  
    outFF = os.path.join(outPath, out_name)

    outD = {
        "c_hat": c_hat.detach().cpu().numpy(),
        "S_hat": S_hat.detach().cpu().numpy().astype(np.int64),
        "S_hat_CL": S_hat_CL.detach().cpu().numpy().astype(np.float32),
        "loss_time": np.asarray(loss_time, dtype=np.float64),
        "loss_nll_time": np.asarray(loss_nll_time, dtype=np.float64),
        "loss_l2_time": np.asarray(loss_l2_time, dtype=np.float64),
        "loss_epoch": np.asarray(loss_epoch, dtype=np.float64),
        "loss_nll_epoch": np.asarray(loss_nll_epoch, dtype=np.float64),
        "loss_l2_epoch": np.asarray(loss_l2_epoch, dtype=np.float64),
    }
    if single_rates is not None:
        outD["single_rates"] = single_rates

    outMD = dict(spikeMD)
    outMD["fit_type"] = "prismEstep"
    outMD["train"] = {
        "lambda2": float(args.lambda2),
        "num_epochs": int(args.num_epochs),
        "pgd_iter": int(args.pgd_iter),
        "lr": float(args.lr),
        "decode": "viterbi",
        "decode_dwell_sec": float(args.decode_dwell_sec),
        "chunk_size": int(args.chunk_size),
        "seed": int(args.seed),
        "time_step_sec": float(args.time_step_sec),
        "eta_clip": float(args.eta_clip),
        "num_states": int(M),
        "num_neurons": int(N),
        "num_steps": int(T_eff),
        "time_range_sec": [float(t0_sec), float(t1_sec)],
        "time_range_bins": [int(start_bin), int(end_bin)],
    }
    outMD["provenance"] = prov

    write_data_npz(outD, outFF, metaD=outMD)
    print(f"Saved prism E-step fit to: {outFF}")
    print('   basePath=' + args.basePath)
    print('   ./prism_Estep_eval.py  --basePath $basePath  --dataName %s   -p b a   \n ' % (out_base))

if __name__ == "__main__":
    main()
