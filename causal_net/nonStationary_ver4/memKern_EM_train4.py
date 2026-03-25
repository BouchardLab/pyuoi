#!/usr/bin/env python3
"""
memKern_EM_train.py — Two-block EM training for Poisson GLM
with off-diagonal memory kernel.

Multi-GPU version: uses all visible GPUs on the node via
torch.distributed (NCCL).  Each rank gets non-overlapping samples
(offset by rank * stride).  Only rank 0 reads/writes files and prints.

Jointly infers from observed spikes:
  - diagonal drive        A^diag  (N,)
  - off-diagonal matrix   A^off   (N, N), zeros on diagonal
  - bias vector           B       (N,)
  - shared lag kernel      kappa   (M_cut,), kappa[0]=1 fixed

Algorithm (from writeup):
  Block 1 (E-like): update kappa with lambda2 smoothness, A/B fixed
  Block 2 (M-like): update A^diag, A^off, B with lambda3 L1 sparsity, kappa fixed

Usage:
  torchrun --standalone --nproc_per_node=gpu memKern_EM_train4.py --dataName <name> --basePath <path>
  # Or single-GPU:
  python memKern_EM_train4.py --dataName <name> --basePath <path>
"""

import os
import time
import math
import secrets
import argparse
from pprint import pprint

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz


# ═══════════════════════════════════════════════════════════════════════
#  Distributed helpers
# ═══════════════════════════════════════════════════════════════════════

def setup_distributed():
    """Initialize distributed backend.  Returns (rank, world_size, device).
    Falls back to single-GPU if env vars are not set."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    return rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank):
    return rank == 0


def broadcast_array(arr, rank, device, dtype=torch.float32):
    """Broadcast a numpy array from rank 0 to all ranks.
    Returns a numpy array on CPU."""
    if rank == 0:
        t = torch.tensor(arr, device=device)
    else:
        t = torch.empty(0, device=device)

    # First broadcast shape
    if rank == 0:
        shape = torch.tensor(list(arr.shape), dtype=torch.long, device=device)
    else:
        shape = torch.empty(0, dtype=torch.long, device=device)

    ndim = torch.tensor(arr.ndim if rank == 0 else 0, dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.broadcast(ndim, src=0)
    nd = int(ndim.item())

    if rank != 0:
        shape = torch.empty(nd, dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.broadcast(shape, src=0)

    if rank != 0:
        t = torch.empty(*shape.tolist(), dtype=dtype, device=device)
    if dist.is_initialized():
        dist.broadcast(t, src=0)
    return t.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════
#  Generative model forward pass
# ═══════════════════════════════════════════════════════════════════════

def compute_S(Y_hist, kappa):
    """Compute off-diagonal history vector S_t for all samples.

    Args:
        Y_hist: (batch, M_cut, N) — spike history, Y_hist[:,0,:] = Y_{t-1}, etc.
        kappa:  (M_cut,) — kernel weights, kappa[0]=1

    Returns:
        S: (batch, N) — weighted sum of history
    """
    # S = sum_{ell=0}^{M-1} kappa[ell] * Y_hist[:, ell, :]
    # kappa: (M_cut,) -> (1, M_cut, 1) for broadcasting
    return (Y_hist * kappa[None, :, None]).sum(dim=1)  # (batch, N)


def compute_eta(Y_prev, S, A_diag, A_off, B, mu_Y=None):
    """Compute log-rate eta for all samples.

    Args:
        Y_prev: (batch, N) — most recent spike counts (= Y_hist[:,0,:])
        S:      (batch, N) — off-diagonal history
        A_diag: (N,)
        A_off:  (N, N) — zeros on diagonal
        B:      (N,)
        mu_Y:   (N,) optional — per-neuron mean for centering Y_prev

    Returns:
        eta: (batch, N)
    """
    # Center Y_prev if mu_Y provided (breaks A_diag <-> B coupling)
    Y_c = (Y_prev - mu_Y[None, :]) if mu_Y is not None else Y_prev
    # diagonal contribution: A_diag_i * (Y_{t-1,i} - mu_i)
    diag_term = Y_c * A_diag[None, :]  # (batch, N)
    # off-diagonal contribution: (A_off @ S^T)^T = S @ A_off^T
    off_term = S @ A_off.t()  # (batch, N)
    return diag_term + off_term + B[None, :]


def compute_mu(eta, eta_clip, dt):
    """Clipped intensity: mu = exp(clamp(eta)) * dt."""
    eta_c = torch.clamp(eta, max=eta_clip)
    return torch.exp(eta_c) * dt, eta_c


def poisson_nll(mu, eta_c, Y_curr, log_dt):
    """Poisson NLL: sum of [mu - Y * (eta_c + log_dt)]."""
    return (mu - Y_curr * (eta_c + log_dt)).sum()


# ═══════════════════════════════════════════════════════════════════════
#  Dataset: (Y_hist, Y_curr) pairs
# ═══════════════════════════════════════════════════════════════════════

def build_spike_history(spikes_np, M_cut, stride=1, offset=0):
    """Build (Y_hist, Y_curr) arrays from spike data — fully vectorized.

    Args:
        spikes_np: (T, N) int array of spike counts
        M_cut: history depth
        stride: sliding window step
        offset: starting offset for this rank's samples

    Returns:
        Y_hist: (N_samples, M_cut, N) float32
        Y_curr: (N_samples, N) float32
    """
    T, N = spikes_np.shape
    # indices of the 'current' time bin for each sample
    t_indices = np.arange(M_cut + offset, T, stride)
    n_samples = len(t_indices)

    # Vectorized: build lag indices (n_samples, M_cut)
    # lag_indices[i, ell] = t_indices[i] - 1 - ell
    lag_offsets = np.arange(M_cut)  # [0, 1, ..., M_cut-1]
    lag_indices = t_indices[:, None] - 1 - lag_offsets[None, :]  # (n_samples, M_cut)

    # Advanced indexing: gather all history windows at once
    Y_hist = spikes_np[lag_indices].astype(np.float32)  # (n_samples, M_cut, N)
    Y_curr = spikes_np[t_indices].astype(np.float32)    # (n_samples, N)
    return Y_hist, Y_curr


# ═══════════════════════════════════════════════════════════════════════
#  Block 1: kernel update (kappa)
# ═══════════════════════════════════════════════════════════════════════

def run_block1_kernel_update(
    Y_hist_gpu, Y_curr_gpu, A_diag, A_off, B,
    kappa, dt, eta_clip, lambda2, lr_kappa, n_iter, M_cut,
    rank, world_size, mu_Y=None
):
    """Update kappa[1:] by gradient descent on Poisson NLL + smoothness penalty.

    kappa[0] = 1 is fixed (anchor).
    The problem is convex in kappa since S_t is linear in kappa.
    Gradients are all-reduced across ranks.

    Args:
        Y_hist_gpu: (N_samples_local, M_cut, N) on GPU
        Y_curr_gpu: (N_samples_local, N) on GPU
        A_diag, A_off, B: fixed parameters on GPU
        kappa: (M_cut,) tensor on GPU, kappa[0]=1
        dt, eta_clip: scalars
        lambda2: smoothness penalty weight
        lr_kappa: learning rate for kappa update
        n_iter: number of gradient steps
        M_cut: kernel length
        rank, world_size: for gradient all-reduce

    Returns:
        kappa: updated (M_cut,) tensor
        nll_val: final NLL value
    """
    log_dt = math.log(dt)
    N_samples_local = Y_hist_gpu.shape[0]

    # Get total N_samples across all ranks for proper averaging
    n_total = torch.tensor(float(N_samples_local), device=kappa.device)
    if dist.is_initialized():
        dist.all_reduce(n_total, op=dist.ReduceOp.SUM)
    n_total_val = n_total.item()

    # We only optimize kappa[1:]
    kappa_free = kappa[1:].clone().detach().requires_grad_(True)

    for iteration in range(n_iter):
        # Reconstruct full kappa with anchor
        kappa_full = torch.cat([torch.ones(1, device=kappa.device), kappa_free])

        # Forward pass
        S = compute_S(Y_hist_gpu, kappa_full)
        Y_prev = Y_hist_gpu[:, 0, :]  # most recent frame
        eta = compute_eta(Y_prev, S, A_diag, A_off, B, mu_Y=mu_Y)
        mu, eta_c = compute_mu(eta, eta_clip, dt)

        # Poisson NLL (local, unnormalized)
        nll = poisson_nll(mu, eta_c, Y_curr_gpu, log_dt)

        # Smoothness penalty on kappa: sum_{ell=2}^{M-1} (kappa[ell] - kappa[ell-1])^2
        if M_cut >= 3:
            diffs = kappa_full[2:] - kappa_full[1:-1]
            smooth_pen = lambda2 * (diffs ** 2).sum()
        else:
            smooth_pen = torch.tensor(0.0, device=kappa.device)

        loss = nll / n_total_val + smooth_pen

        # Backward
        if kappa_free.grad is not None:
            kappa_free.grad.zero_()
        loss.backward()

        # All-reduce the gradient across ranks
        if dist.is_initialized():
            dist.all_reduce(kappa_free.grad, op=dist.ReduceOp.SUM)

        # Gradient step
        with torch.no_grad():
            kappa_free -= lr_kappa * kappa_free.grad

    # Reconstruct final kappa
    with torch.no_grad():
        kappa_out = torch.cat([torch.ones(1, device=kappa.device), kappa_free.detach()])

    # For reporting, get global NLL
    nll_report = nll.detach()
    if dist.is_initialized():
        dist.all_reduce(nll_report, op=dist.ReduceOp.SUM)

    return kappa_out, float((nll_report / n_total_val).item())


# ═══════════════════════════════════════════════════════════════════════
#  Block 2: network update (A_diag, A_off, B)
# ═══════════════════════════════════════════════════════════════════════

class NetworkModel(torch.nn.Module):
    """Wraps A_diag, A_off, B as learnable parameters for Adam.

    Stores mu_Y (per-neuron mean of Y_prev) as a buffer for centering,
    which breaks the A_diag <-> B coupling.
    """

    def __init__(self, N, A_diag_init, A_off_init, B_init, mu_Y):
        super().__init__()
        self.N = N
        self.A_diag = torch.nn.Parameter(A_diag_init.clone())
        self.A_off = torch.nn.Parameter(A_off_init.clone())
        self.B = torch.nn.Parameter(B_init.clone())
        self.register_buffer('mu_Y', mu_Y.clone())  # not trainable

    def forward(self, Y_prev, S, dt, eta_clip):
        """
        Args:
            Y_prev: (batch, N)
            S:      (batch, N) — precomputed with fixed kappa
            dt:     scalar
            eta_clip: scalar
        Returns:
            mu:   (batch, N)
            eta_c: (batch, N)
        """
        # Zero out diagonal of A_off during forward
        A_off_masked = self.A_off * (1.0 - torch.eye(self.N, device=self.A_off.device))
        eta = compute_eta(Y_prev, S, self.A_diag, A_off_masked, self.B,
                          mu_Y=self.mu_Y)
        mu, eta_c = compute_mu(eta, eta_clip, dt)
        return mu, eta_c


def offdiag_l1_mean(A_off, N):
    """Mean absolute value of off-diagonal entries: eq (2) from writeup."""
    mask = ~torch.eye(N, dtype=torch.bool, device=A_off.device)
    return A_off[mask].abs().mean()


def enforce_zero_diagonal_(A_off, N):
    """Zero out diagonal of A_off in-place."""
    with torch.no_grad():
        idx = torch.arange(N, device=A_off.device)
        A_off[idx, idx] = 0.0


def enforce_spectral_radius_(A_off, rho_max, N):
    """Hard spectral projection on A_off. Returns rho before projection."""
    with torch.no_grad():
        rho = torch.linalg.eigvals(A_off).abs().max().item()
        if rho > rho_max:
            A_off.mul_(rho_max / rho)
    return rho


def offdiag_soft_threshold_(A_off, lr, lam, N):
    """Proximal L1 on off-diagonal elements of A_off."""
    if lam <= 0:
        return
    with torch.no_grad():
        mask = ~torch.eye(N, dtype=torch.bool, device=A_off.device)
        thresh = lr * lam
        v = A_off[mask]
        A_off[mask] = v.sign() * (v.abs() - thresh).clamp(min=0.0)


# ═══════════════════════════════════════════════════════════════════════
#  Initialization helpers
# ═══════════════════════════════════════════════════════════════════════

def init_A_diag_from_spikes(N, dt):
    """Initialize A_diag with random values in range [-0.5, -0.15]."""
    return np.random.uniform(-0.5, -0.15, size=N).astype(np.float32)


def init_A_off_from_spikes(Y_hist_gpu, Y_curr_gpu):
    """Initialize A_off from lag-1 cross-correlations on GPU, zeros on diagonal."""
    # Y_hist_gpu[:, 0, :] is Y_{t-1}, Y_curr_gpu is Y_t
    Y0 = Y_hist_gpu[:, 0, :].double()  # (n_samples, N)
    Y1 = Y_curr_gpu.double()
    n = Y0.shape[0]
    m0 = Y0.mean(dim=0, keepdim=True)
    s0 = Y0.std(dim=0, keepdim=True) + 1e-8
    m1 = Y1.mean(dim=0, keepdim=True)
    s1 = Y1.std(dim=0, keepdim=True) + 1e-8
    Z0 = (Y0 - m0) / s0
    Z1 = (Y1 - m1) / s1
    C = (Z0.t() @ Z1) / n  # (N, N) on GPU
    A_off = (C * 0.1).float().cpu().numpy()
    np.fill_diagonal(A_off, 0.0)
    return A_off


def init_B_from_spikes(Y_curr_gpu, dt):
    """Initialize B from mean log firing rates using GPU data."""
    mean_rate = Y_curr_gpu.double().mean(dim=0) / dt
    mean_rate = mean_rate.clamp(min=1e-6)
    B = mean_rate.log().float().cpu().numpy()
    return B


def init_kappa_damped_oscillator(M_cut):
    """Initialize kappa as a damped oscillator, kappa[0]=1.
    Functional form: kappa(ell) = exp(-alpha*ell) * cos(omega*ell)
    where:
      omega = 1.5 * pi / M_cut  (1.5 pi phase over M_cut steps)
      alpha = omega / (2 * Q)
      Q = 1.5                   (quality factor)
    """
    Q = 1.5
    omega = 1.5 * np.pi / M_cut
    alpha = omega / (2.0 * Q)

    kappa = np.zeros(M_cut, dtype=np.float32)
    for ell in range(M_cut):
        kappa[ell] = np.exp(-alpha * ell) * np.cos(omega * ell)
    kappa[0] = 1.0  # enforce anchor
    return kappa


# ═══════════════════════════════════════════════════════════════════════
#  Args
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-block EM for Poisson GLM with memory kernel")
    p.add_argument("--dataName", required=True)
    p.add_argument("--basePath",
                   default="/pscratch/sd/b/balewski/2025_causalNet_tmp/")

    g = p.add_argument_group("model")
    g.add_argument("--M_cut", type=int, default=10,
                   help="History depth (kernel length)")
    g.add_argument("--eta_clip", type=float, default=5.0,
                   help="Log-rate clipping threshold")

    g = p.add_argument_group("EM structure")
    g.add_argument("--num_em_iters", type=int, default=20,
                   help="Outer EM iterations")
    g.add_argument("--block1_iter", type=int, default=50,
                   help="Gradient steps per Block 1 (kernel update)")
    g.add_argument("--block2_epochs", type=int, default=100,
                   help="Adam epochs per Block 2 (network update)")

    g = p.add_argument_group("Block 1 (kernel)")
    g.add_argument("--lr_kappa", type=float, default=0.01,
                   help="Learning rate for kappa update")
    g.add_argument("--lambda2", type=float, default=1.0,
                   help="Kernel smoothness penalty weight")

    g = p.add_argument_group("Block 2 (network)")
    g.add_argument("--lr_net", type=float, default=0.003,
                   help="Adam learning rate for A_diag, A_off, B")
    g.add_argument("--lr_end_factor", type=float, default=0.1,
                   help="LR decays to lr_net * lr_end_factor")
    g.add_argument("--lambda3", type=float, default=0.02,
                   help="L1 penalty on off-diagonal A_off")
    g.add_argument("--rho_max", type=float, default=0.95,
                   help="Hard spectral radius ceiling for A_off")
    g.add_argument("--rho_every", type=int, default=50,
                   help="Spectral projection frequency (batches)")
    g.add_argument("--delay_em_iter_4_ArhoMax", type=int, default=3,
                   help="EM iter to start rho_max enforcement")
    g.add_argument("--delay_em_iter_4_lrDecay", type=int, default=5,
                   help="EM iter to start LR decay")
    g.add_argument("--delay_em_iter_4_Aprune", type=int, default=3,
                   help="EM iter to start L1 pruning")
    g.add_argument("--batch_size", type=int, default=2048)
    g.add_argument("--minW", type=float, default=0.01,
                   help="Threshold for edge counting (reporting)")

    g = p.add_argument_group("data")
    g.add_argument("-T", "--time_range_sec", default=[0.0, 60.0],
                   nargs=2, type=float,
                   help="Time window [t0, t1] in seconds")
    g.add_argument("--sample_stride", type=int, default=1,
                   help="Stride for sliding window in data preparation")

    g = p.add_argument_group("init")

    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--fitName", type=str, default=None,
                   help="Output name; if omitted a random name is generated")
    g.add_argument("-v", "--verb", type=int, default=1)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    rank, world_size, device = setup_distributed()

    inpPath = os.path.join(args.basePath, "truthDale")
    outPath = os.path.join(args.basePath, "memKernFit")

    if is_rank0(rank):
        assert os.path.exists(inpPath), f"missing inpPath: {inpPath}"
        os.makedirs(outPath, exist_ok=True)
        print(f"\nmemKern_EM_train args (world_size={world_size}):")
        for arg in vars(args):
            print(f"  {arg}: {getattr(args, arg)}")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed)  # keep same on all ranks for init
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # ── Load data (rank 0 reads, then broadcasts) ────────────────
    if is_rank0(rank):
        spikesFF = os.path.join(inpPath, f"{args.dataName}.spikes.npz")
        spikeD, spikeMD = read_data_npz(spikesFF, verb=args.verb > 0)
        if args.verb > 1:
            pprint(spikeMD)
        spikes = np.asarray(spikeD["spikes"])
        assert spikes.ndim == 2, f"spikes must be (T, N); got {spikes.shape}"
        single_rates = np.asarray(spikeD["single_rates"])

        truthFF = os.path.join(inpPath, f"{args.dataName}.simTruth.npz")
        trueD, trueMD = read_data_npz(truthFF, verb=args.verb > 0)
    else:
        spikes = np.empty(0, dtype=np.uint8)
        single_rates = np.empty(0, dtype=np.float64)
        spikeMD = None

    # Broadcast spikes and single_rates to all ranks
    if world_size > 1:
        spikes = broadcast_array(spikes, rank, device, dtype=torch.uint8)
        single_rates = broadcast_array(single_rates, rank, device, dtype=torch.float64)
        # Broadcast spikeMD keys needed by all ranks
        dt_t = torch.tensor(float(spikeMD["time_step_sec"]) if rank == 0 else 0.0, device=device)
        if dist.is_initialized():
            dist.broadcast(dt_t, src=0)
        dt = float(dt_t.item())
    else:
        dt = float(spikeMD["time_step_sec"])

    T_raw, N = spikes.shape
    M_cut = args.M_cut
    eta_clip = args.eta_clip

    # ── Time range selection ─────────────────────────────────────
    t0_sec, t1_sec = float(args.time_range_sec[0]), float(args.time_range_sec[1])
    assert t0_sec < t1_sec, "time_range_sec[0] must be < time_range_sec[1]"
    start_bin = max(0, int(math.floor(t0_sec / dt)))
    end_bin = min(T_raw - 1, int(math.floor(t1_sec / dt)))
    assert end_bin - start_bin >= M_cut + 1, \
        f"Time range too short: need at least {M_cut + 2} bins, got {end_bin - start_bin + 1}"
    spikes = spikes[start_bin: end_bin + 1]
    T_full = spikes.shape[0]

    if is_rank0(rank):
        print(f"\nN={N}  M_cut={M_cut}  T={T_full}  "
              f"dt={dt}  eta_clip={eta_clip}")
        print(f"time=[{t0_sec:.1f}, {t1_sec:.1f}]s  bins=[{start_bin}, {end_bin}]")

    # ── Build dataset (vectorized, GPU-resident) ─────────────────
    # Each rank gets non-overlapping samples:
    # rank r sees samples at t_indices starting at M_cut + r*stride,
    # stepping by stride*world_size
    effective_stride = args.sample_stride * world_size
    rank_offset = rank * args.sample_stride
    t_build = time.time()
    Y_hist_np, Y_curr_np = build_spike_history(
        spikes, M_cut, stride=effective_stride, offset=rank_offset
    )
    N_samples_local = Y_hist_np.shape[0]

    # All ranks compute total
    n_total_t = torch.tensor(float(N_samples_local), device=device)
    if dist.is_initialized():
        dist.all_reduce(n_total_t, op=dist.ReduceOp.SUM)
    N_samples_total = int(n_total_t.item())

    if is_rank0(rank):
        print(f"N_samples_total={N_samples_total}  per_rank={N_samples_local}  "
              f"stride={args.sample_stride}  world_size={world_size}  "
              f"effective_stride={effective_stride}  build={time.time()-t_build:.1f}s")

    # Move entire dataset to GPU — no DataLoader needed
    Y_hist_gpu = torch.tensor(Y_hist_np, dtype=torch.float32, device=device)
    Y_curr_gpu = torch.tensor(Y_curr_np, dtype=torch.float32, device=device)
    del Y_hist_np, Y_curr_np  # free CPU memory

    # ── Initialize parameters (same on all ranks via same np seed) ─
    del spikes  # free CPU memory, no longer needed
    A_diag_init = init_A_diag_from_spikes(N, dt)
    A_off_init = init_A_off_from_spikes(Y_hist_gpu, Y_curr_gpu)
    B_init = init_B_from_spikes(Y_curr_gpu, dt)
    kappa_init = init_kappa_damped_oscillator(M_cut)

    if is_rank0(rank):
        print(f"\nInitialization:")
        print(f"  A_diag range: [{A_diag_init.min():.4f}, {A_diag_init.max():.4f}]")
        print(f"  A_off  range: [{A_off_init.min():.4f}, {A_off_init.max():.4f}]")
        print(f"  B      range: [{B_init.min():.4f}, {B_init.max():.4f}]")
        print(f"  kappa: {kappa_init}")

    # Move to GPU
    A_diag = torch.tensor(A_diag_init, dtype=torch.float32, device=device)
    A_off = torch.tensor(A_off_init, dtype=torch.float32, device=device)
    B = torch.tensor(B_init, dtype=torch.float32, device=device)
    kappa = torch.tensor(kappa_init, dtype=torch.float32, device=device)

    # Save copies of initial state
    A_diag_init_save = A_diag_init.copy()
    A_off_init_save = A_off_init.copy()
    B_init_save = B_init.copy()
    kappa_init_save = kappa_init.copy()

    # ── Build NetworkModel for Block 2 ───────────────────────────
    net_model = NetworkModel(N, A_diag, A_off, B).to(device)

    # Wrap in DDP for automatic gradient synchronization in Block 2
    if world_size > 1:
        ddp_model = DDP(net_model, device_ids=[rank])
    else:
        ddp_model = net_model
    # Access underlying model for parameter projections
    raw_model = ddp_model.module if world_size > 1 else ddp_model

    # Compile for kernel fusion (reduces launch overhead for small ops)
    ddp_model = torch.compile(ddp_model)

    # Optimizer and scheduler for Block 2
    total_b2_epochs = args.num_em_iters * args.block2_epochs
    decay_start = args.delay_em_iter_4_lrDecay * args.block2_epochs
    decay_epochs = max(1, total_b2_epochs - decay_start)
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.lr_net, fused=True)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=args.lr_end_factor,
        total_iters=decay_epochs, last_epoch=-1
    )
    scheduler._step_count = 1

    # ── History tracking ─────────────────────────────────────────
    h_b1_nll = []
    h_kappa = []

    # Monitoring histories (per global M-epoch)
    h_m_loss = []
    h_m_nll = []
    h_m_l1 = []
    h_rho = []
    h_nz = []
    h_lr = []

    h_kappa.append(kappa_init.copy())

    off_mask = ~torch.eye(N, dtype=torch.bool, device=device)

    t_start = time.time()
    b2_epoch_global = 0

    # ══════════════════════════════════════════════════════════════
    #  EM loop
    # ══════════════════════════════════════════════════════════════
    for em in range(1, args.num_em_iters + 1):
        apply_prune = em > args.delay_em_iter_4_Aprune
        apply_rho = em > args.delay_em_iter_4_ArhoMax

        # ── Block 1: kernel update ───────────────────────────────
        tb1_start = time.time()
        # Use current A_diag, A_off, B from net_model (detached)
        with torch.no_grad():
            A_diag_fixed = raw_model.A_diag.data.clone()
            A_off_fixed = raw_model.A_off.data.clone()
            # Zero diagonal of A_off
            idx = torch.arange(N, device=device)
            A_off_fixed[idx, idx] = 0.0
            B_fixed = raw_model.B.data.clone()

        kappa, b1_nll = run_block1_kernel_update(
            Y_hist_gpu, Y_curr_gpu,
            A_diag_fixed, A_off_fixed, B_fixed,
            kappa, dt, eta_clip, args.lambda2,
            args.lr_kappa, args.block1_iter, M_cut,
            rank, world_size
        )
        tb1 = time.time() - tb1_start
        h_b1_nll.append(float(b1_nll))
        h_kappa.append(kappa.cpu().numpy().copy())

        # ── Block 2: network update ──────────────────────────────
        tb2_start = time.time()

        # Fix kappa for this block — precompute S and Y_prev ONCE
        kappa_fixed = kappa.detach().clone()
        with torch.no_grad():
            S_full = compute_S(Y_hist_gpu, kappa_fixed)       # (N_local, N)
            Y_prev_full = Y_hist_gpu[:, 0, :].contiguous()    # (N_local, N)

        BS = args.batch_size
        n_batches = (N_samples_local + BS - 1) // BS
        log_dt = math.log(dt)

        for ep in range(args.block2_epochs):
            b2_epoch_global += 1
            # Accumulate losses on GPU to avoid CPU-GPU sync per batch
            acc_nll = torch.tensor(0.0, device=device)
            acc_l1 = torch.tensor(0.0, device=device)
            acc_loss = torch.tensor(0.0, device=device)

            # Shuffle indices on GPU
            perm = torch.randperm(N_samples_local, device=device)

            for bi in range(n_batches):
                i0 = bi * BS
                i1 = min(i0 + BS, N_samples_local)
                idx = perm[i0:i1]

                Y_prev_b = Y_prev_full[idx]
                S_b = S_full[idx]
                Y_curr_b = Y_curr_gpu[idx]

                optimizer.zero_grad(set_to_none=True)
                mu, eta_c = ddp_model(Y_prev_b, S_b, dt, eta_clip)
                nll = poisson_nll(mu, eta_c, Y_curr_b, log_dt) / Y_curr_b.shape[0]

                if args.lambda3 > 0:
                    l1_term = args.lambda3 * offdiag_l1_mean(raw_model.A_off, N)
                else:
                    l1_term = torch.tensor(0.0, device=device)

                loss = nll + l1_term
                loss.backward()
                optimizer.step()

                # Post-step projections (on raw model params)
                enforce_zero_diagonal_(raw_model.A_off.data, N)
                if apply_prune and args.lambda3 > 0:
                    offdiag_soft_threshold_(
                        raw_model.A_off.data,
                        optimizer.param_groups[0]["lr"],
                        args.lambda3, N
                    )
                if apply_rho and (bi % args.rho_every == 0):
                    enforce_spectral_radius_(raw_model.A_off.data, args.rho_max, N)

                # Accumulate on GPU — no .item() sync
                with torch.no_grad():
                    acc_nll += nll.detach()
                    acc_l1 += l1_term.detach()
                    acc_loss += loss.detach()

            if em > args.delay_em_iter_4_lrDecay:
                scheduler.step()

        # Record metrics ONCE per EM iteration (not per epoch) — single sync
        nb = max(1, n_batches)
        with torch.no_grad():
            rho = float(torch.linalg.eigvals(raw_model.A_off.data).abs().max().item())
            nz = int((raw_model.A_off.abs() > args.minW).sum().item())
            last_nll = float(acc_nll.item()) / nb
            last_l1 = float(acc_l1.item()) / nb
            last_loss = float(acc_loss.item()) / nb

        h_m_loss.append(last_loss)
        h_m_nll.append(last_nll)
        h_m_l1.append(last_l1)
        h_rho.append(rho)
        h_nz.append(nz)
        h_lr.append(float(optimizer.param_groups[0]["lr"]))

        tb2 = time.time() - tb2_start
        met = dict(nll=h_m_nll[-1], l1=h_m_l1[-1], rho=h_rho[-1], nz=h_nz[-1])

        # ── Re-synchronize parameters from rank 0 ────────────────
        # Prevents drift from non-gradient ops (pruning, spectral projection)
        if dist.is_initialized():
            for p in raw_model.parameters():
                dist.broadcast(p.data, src=0)

        if is_rank0(rank) and args.verb > 0:
            n_off = int(off_mask.sum().item())
            sp = 1.0 - met["nz"] / max(1, n_off)
            print(
                f"EM {em:3d}/{args.num_em_iters}  "
                f"B1_nll={b1_nll:.4e}({tb1:.1f}s)  "
                f"B2_nll={met['nll']:.4e} l1={met['l1']:.4e} "
                f"rho={met['rho']:.4f} sp={sp:.3f} nz={met['nz']} "
                f"lr={h_lr[-1]:.2e} ({tb2:.1f}s)  "
                f"kappa_norm={float(kappa.abs().sum()):.3f}  "
                f"tot={time.time() - t_start:.0f}s"
            )

    # ══════════════════════════════════════════════════════════════
    #  Save results (rank 0 only)
    # ══════════════════════════════════════════════════════════════
    if is_rank0(rank):
        print(f"\nTotal EM time: {time.time() - t_start:.1f}s")

        A_diag_hat = raw_model.A_diag.detach().cpu().numpy()
        A_off_hat = raw_model.A_off.detach().cpu().numpy()
        np.fill_diagonal(A_off_hat, 0.0)
        B_hat = raw_model.B.detach().cpu().numpy()
        kappa_hat = kappa.detach().cpu().numpy()

        if args.fitName is None:
            h6 = secrets.token_hex(3)
            outF = f"{args.dataName}-memKern-{h6}"
        else:
            outF = args.fitName
        outFF = os.path.join(outPath, f"{outF}.memKernEM.npz")

        outD = {
            "A_diag_init":    A_diag_init_save,
            "A_diag_hat":     A_diag_hat.astype(np.float32),
            "A_off_init":     A_off_init_save,
            "A_off_hat":      A_off_hat.astype(np.float32),
            "B_init":         B_init_save,
            "B_hat":          B_hat.astype(np.float32),
            "kappa_init":     kappa_init_save,
            "kappa_hat":      kappa_hat.astype(np.float32),
            "kappa_history":  np.array(h_kappa, dtype=np.float32),
            "single_rates":   single_rates.astype(np.float32),
            "e_nll_em":       np.array(h_b1_nll, dtype=np.float64),
            "m_loss_epoch":   np.array(h_m_loss, dtype=np.float64),
            "m_nll_epoch":    np.array(h_m_nll, dtype=np.float64),
            "m_l1_epoch":     np.array(h_m_l1, dtype=np.float64),
            "rho_epoch":      np.array(h_rho, dtype=np.float64),
            "nz_edges_epoch": np.array(h_nz, dtype=np.int64),
            "learning_rates": np.array(h_lr, dtype=np.float64),
        }

        spikeMD.pop('short_name')
        outMD = {'spike_gen':spikeMD}
        outMD["provenance"]={"memKernEM_file": outF, "spiksData_file": args.dataName}
        outMD['short_name']=outF

        outMD["fit_type"] = "memKernEM"
        outMD["train"] = {
            "num_em_iters":           args.num_em_iters,
            "sample_stride":          args.sample_stride,
            "m_epochs":               args.block2_epochs,
            "block1_iter":            args.block1_iter,
            "block2_epochs":          args.block2_epochs,
            "lr_kappa":               args.lr_kappa,
            "lr_net":                 args.lr_net,
            "lr_end_factor":          args.lr_end_factor,
            "lambda2":                args.lambda2,
            "lambda3":                args.lambda3,
            "rho_max":                args.rho_max,
            "rho_every":              args.rho_every,
            "delay_em_iter_4_ArhoMax":  args.delay_em_iter_4_ArhoMax,
            "delay_em_iter_4_lrDecay":  args.delay_em_iter_4_lrDecay,
            "delay_em_iter_4_Aprune":   args.delay_em_iter_4_Aprune,
            "batch_size":             args.batch_size,
            "minW":                   args.minW,
            "M_cut":                  M_cut,
            "num_neurons":            N,
            "num_time_bins":          T_full,
            "num_samples":            N_samples_total,
            "num_samples_per_rank":   N_samples_local,
            "world_size":             world_size,
            "time_step_sec":          dt,
            "eta_clip":               eta_clip,
            "time_range_sec":         [t0_sec, t1_sec],
            "time_range_bins":        [start_bin, end_bin],
            "seed":                   args.seed,
            "training_time_sec":      round(time.time() - t_start, 1),
        }

        if args.verb > 1: pprint(outMD)
        write_data_npz(outD, outFF, metaD=outMD)
        print(f"\nSaved: {outFF}")
        print(f"  basePath={args.basePath}")
        print(f"  ./memKern_EM_eval4.py --basePath {args.basePath} --dataName {outF} -p a b \n")

    cleanup_distributed()


if __name__ == "__main__":
    main()
