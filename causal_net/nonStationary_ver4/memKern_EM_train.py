#!/usr/bin/env python3
"""
memKern_EM_train.py — Two-block EM training for Poisson GLM
with off-diagonal memory kernel.

Jointly infers from observed spikes:
  - diagonal drive        A^diag  (N,)
  - off-diagonal matrix   A^off   (N, N), zeros on diagonal
  - bias vector           B       (N,)
  - shared lag kernel      kappa   (M_cut,), kappa[0]=1 fixed

Algorithm (from writeup):
  Block 1 (E-like): update kappa with lambda2 smoothness, A/B fixed
  Block 2 (M-like): update A^diag, A^off, B with lambda3 L1 sparsity, kappa fixed

Single A100 GPU.  No try/except.  No fallbacks.

Usage:
  python memKern_EM_train.py --dataName <name> --basePath <path>
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
from torch.utils.data import Dataset, DataLoader

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz


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


def compute_eta(Y_prev, S, A_diag, A_off, B):
    """Compute log-rate eta for all samples.

    Args:
        Y_prev: (batch, N) — most recent spike counts (= Y_hist[:,0,:])
        S:      (batch, N) — off-diagonal history
        A_diag: (N,)
        A_off:  (N, N) — zeros on diagonal
        B:      (N,)

    Returns:
        eta: (batch, N)
    """
    # diagonal contribution: A_diag_i * Y_{t-1,i}
    diag_term = Y_prev * A_diag[None, :]  # (batch, N)
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

class SpikeHistoryDataset(Dataset):
    """Each sample: M_cut history frames + 1 target frame.

    self.Y_hist: (N_samples, M_cut, N) — history window
    self.Y_curr: (N_samples, N) — target spike counts
    """

    def __init__(self, spikes_np, M_cut):
        """
        Args:
            spikes_np: (T, N) int array of spike counts
            M_cut: history depth
        """
        T, N = spikes_np.shape
        N_samples = T - M_cut
        # Build history windows: for t_out in [M_cut, T-1],
        # history is [Y_{t_out-1}, Y_{t_out-2}, ..., Y_{t_out-M_cut}]
        Y_hist = np.zeros((N_samples, M_cut, N), dtype=np.float32)
        for s in range(N_samples):
            t_out = s + M_cut
            for ell in range(M_cut):
                Y_hist[s, ell, :] = spikes_np[t_out - 1 - ell, :]
        self.Y_hist = Y_hist
        self.Y_curr = spikes_np[M_cut:].astype(np.float32)

    def __len__(self):
        return self.Y_hist.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.Y_hist[idx]),
            torch.from_numpy(self.Y_curr[idx]),
        )


# ═══════════════════════════════════════════════════════════════════════
#  Block 1: kernel update (kappa)
# ═══════════════════════════════════════════════════════════════════════

def run_block1_kernel_update(
    Y_hist_gpu, Y_curr_gpu, A_diag, A_off, B,
    kappa, dt, eta_clip, lambda2, lr_kappa, n_iter, M_cut
):
    """Update kappa[1:] by gradient descent on Poisson NLL + smoothness penalty.

    kappa[0] = 1 is fixed (anchor).
    The problem is convex in kappa since S_t is linear in kappa.

    Args:
        Y_hist_gpu: (N_samples, M_cut, N) on GPU
        Y_curr_gpu: (N_samples, N) on GPU
        A_diag, A_off, B: fixed parameters on GPU
        kappa: (M_cut,) tensor on GPU, kappa[0]=1
        dt, eta_clip: scalars
        lambda2: smoothness penalty weight
        lr_kappa: learning rate for kappa update
        n_iter: number of gradient steps
        M_cut: kernel length

    Returns:
        kappa: updated (M_cut,) tensor
        nll_val: final NLL value
    """
    log_dt = math.log(dt)
    N_samples = Y_hist_gpu.shape[0]

    # We only optimize kappa[1:]
    kappa_free = kappa[1:].clone().detach().requires_grad_(True)

    for iteration in range(n_iter):
        # Reconstruct full kappa with anchor
        kappa_full = torch.cat([torch.ones(1, device=kappa.device), kappa_free])

        # Forward pass
        S = compute_S(Y_hist_gpu, kappa_full)
        Y_prev = Y_hist_gpu[:, 0, :]  # most recent frame
        eta = compute_eta(Y_prev, S, A_diag, A_off, B)
        mu, eta_c = compute_mu(eta, eta_clip, dt)

        # Poisson NLL
        nll = poisson_nll(mu, eta_c, Y_curr_gpu, log_dt) / N_samples

        # Smoothness penalty on kappa: sum_{ell=2}^{M-1} (kappa[ell] - kappa[ell-1])^2
        # With 0-based indexing and kappa_full of length M_cut
        if M_cut >= 3:
            diffs = kappa_full[2:] - kappa_full[1:-1]
            smooth_pen = lambda2 * (diffs ** 2).sum()
        else:
            smooth_pen = torch.tensor(0.0, device=kappa.device)

        loss = nll + smooth_pen

        # Backward
        if kappa_free.grad is not None:
            kappa_free.grad.zero_()
        loss.backward()

        # Gradient step
        with torch.no_grad():
            kappa_free -= lr_kappa * kappa_free.grad

    # Reconstruct final kappa
    with torch.no_grad():
        kappa_out = torch.cat([torch.ones(1, device=kappa.device), kappa_free.detach()])

    return kappa_out, float(nll.item())


# ═══════════════════════════════════════════════════════════════════════
#  Block 2: network update (A_diag, A_off, B)
# ═══════════════════════════════════════════════════════════════════════

class NetworkModel(torch.nn.Module):
    """Wraps A_diag, A_off, B as learnable parameters for Adam."""

    def __init__(self, N, A_diag_init, A_off_init, B_init):
        super().__init__()
        self.N = N
        self.A_diag = torch.nn.Parameter(A_diag_init.clone())
        self.A_off = torch.nn.Parameter(A_off_init.clone())
        self.B = torch.nn.Parameter(B_init.clone())

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
        eta = compute_eta(Y_prev, S, self.A_diag, A_off_masked, self.B)
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


def run_block2_network_update(
    loader, kappa_fixed, net_model, optimizer, device,
    dt, eta_clip, lambda3, rho_max, N, n_epochs,
    apply_prune, apply_rho, rho_every, minW
):
    """Run n_epochs of Adam over the DataLoader, updating A_diag, A_off, B.

    Args:
        loader: DataLoader yielding (Y_hist, Y_curr) batches
        kappa_fixed: (M_cut,) tensor on GPU — fixed kernel
        net_model: NetworkModel on GPU
        optimizer: Adam optimizer
        device: torch device
        dt, eta_clip: scalars
        lambda3: L1 sparsity weight
        rho_max: spectral radius ceiling
        N: number of neurons
        n_epochs: Adam epochs
        apply_prune: bool, whether to apply L1 soft thresholding
        apply_rho: bool, whether to enforce spectral radius
        rho_every: apply rho projection every this many batches
        minW: threshold for edge counting

    Returns:
        metrics dict from last epoch
    """
    log_dt = math.log(dt)
    off_mask = ~torch.eye(N, dtype=torch.bool, device=device)
    metrics = {}

    for epoch in range(n_epochs):
        s_nll = 0.0
        s_l1 = 0.0
        s_loss = 0.0
        n_batches = 0

        for bi, (Y_hist_batch, Y_curr_batch) in enumerate(loader):
            Y_hist_batch = Y_hist_batch.to(device, non_blocking=True)
            Y_curr_batch = Y_curr_batch.to(device, non_blocking=True)

            # Precompute S with fixed kappa
            S = compute_S(Y_hist_batch, kappa_fixed)
            Y_prev = Y_hist_batch[:, 0, :]

            optimizer.zero_grad(set_to_none=True)
            mu, eta_c = net_model(Y_prev, S, dt, eta_clip)
            nll = poisson_nll(mu, eta_c, Y_curr_batch, log_dt) / Y_curr_batch.shape[0]

            if lambda3 > 0:
                l1_term = lambda3 * offdiag_l1_mean(net_model.A_off, N)
            else:
                l1_term = torch.tensor(0.0, device=device)

            loss = nll + l1_term
            loss.backward()
            optimizer.step()

            # Post-step projections
            enforce_zero_diagonal_(net_model.A_off.data, N)
            if apply_prune and lambda3 > 0:
                offdiag_soft_threshold_(
                    net_model.A_off.data,
                    optimizer.param_groups[0]["lr"],
                    lambda3, N
                )
            if apply_rho and (bi % rho_every == 0):
                enforce_spectral_radius_(net_model.A_off.data, rho_max, N)

            s_nll += nll.item()
            s_l1 += l1_term.item()
            s_loss += loss.item()
            n_batches += 1

        nb = max(1, n_batches)
        with torch.no_grad():
            nz = int((net_model.A_off[off_mask].abs() > minW).sum().item())
            rho = float(torch.linalg.eigvals(net_model.A_off.data).abs().max().item())

        metrics = dict(
            loss=s_loss / nb, nll=s_nll / nb, l1=s_l1 / nb,
            rho=rho, nz=nz
        )

    return metrics


# ═══════════════════════════════════════════════════════════════════════
#  Initialization helpers
# ═══════════════════════════════════════════════════════════════════════

def init_A_diag_from_spikes(spikes, dt):
    """Initialize A_diag from lag-1 autocorrelation of each neuron."""
    T, N = spikes.shape
    A_diag = np.zeros(N, dtype=np.float32)
    for i in range(N):
        y = spikes[:, i].astype(np.float64)
        if y.std() > 0:
            c1 = np.corrcoef(y[:-1], y[1:])[0, 1]
            A_diag[i] = float(np.clip(c1, -0.5, 0.5))
    return A_diag


def init_A_off_from_spikes(spikes, dt):
    """Initialize A_off from lag-1 cross-correlations, zeros on diagonal."""
    T, N = spikes.shape
    Y0 = spikes[:-1].astype(np.float64)
    Y1 = spikes[1:].astype(np.float64)
    # Standardize
    m0 = Y0.mean(axis=0, keepdims=True)
    s0 = Y0.std(axis=0, keepdims=True) + 1e-8
    m1 = Y1.mean(axis=0, keepdims=True)
    s1 = Y1.std(axis=0, keepdims=True) + 1e-8
    Z0 = (Y0 - m0) / s0
    Z1 = (Y1 - m1) / s1
    C = (Z0.T @ Z1) / (T - 1)  # (N, N)
    A_off = (C * 0.1).astype(np.float32)
    np.fill_diagonal(A_off, 0.0)
    return A_off


def init_B_from_spikes(spikes, dt):
    """Initialize B from mean log firing rates."""
    T, N = spikes.shape
    mean_rate = spikes.mean(axis=0).astype(np.float64) / dt
    mean_rate = np.clip(mean_rate, 1e-6, None)
    B = np.log(mean_rate).astype(np.float32)
    return B


def init_kappa_damped_oscillator(M_cut, decay=0.8, freq=0.0):
    """Initialize kappa as damped exponential, kappa[0]=1."""
    kappa = np.zeros(M_cut, dtype=np.float32)
    for ell in range(M_cut):
        kappa[ell] = (decay ** ell) * np.cos(2 * np.pi * freq * ell)
    kappa[0] = 1.0  # enforce anchor
    return kappa


# ═══════════════════════════════════════════════════════════════════════
#  Args
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-block EM: Poisson GLM with off-diagonal memory kernel")

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

    g = p.add_argument_group("init")
    g.add_argument("--kappa_decay", type=float, default=0.8,
                   help="Decay rate for initial damped-oscillator kernel")

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

    device = torch.device("cuda")
    torch.cuda.set_device(0)

    inpPath = os.path.join(args.basePath, "truthDale")
    outPath = os.path.join(args.basePath, "memKernFit")
    assert os.path.exists(inpPath), f"missing inpPath: {inpPath}"
    os.makedirs(outPath, exist_ok=True)

    print("\nmemKern_EM_train args:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # ── Load data (code1 pattern) ────────────────────────────────
    spikesFF = os.path.join(inpPath, f"{args.dataName}.spikes.npz")
    spikeD, spikeMD = read_data_npz(spikesFF, verb=args.verb > 0)
    if args.verb > 1:
        pprint(spikeMD)

    spikes = np.asarray(spikeD["spikes"])
    assert spikes.ndim == 2, f"spikes must be (T, N); got {spikes.shape}"

    single_rates = np.asarray(spikeD["single_rates"])
    assert single_rates.ndim == 1
    assert single_rates.shape[0] == spikes.shape[1]

    # Load truth for metadata (optional diagnostics)
    truthFF = os.path.join(inpPath, f"{args.dataName}.simTruth.npz")
    trueD, trueMD = read_data_npz(truthFF, verb=args.verb > 0)

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
    N_samples = T_full - M_cut

    print(f"\nN={N}  M_cut={M_cut}  T={T_full}  N_samples={N_samples}  "
          f"dt={dt}  eta_clip={eta_clip}")
    print(f"time=[{t0_sec:.1f}, {t1_sec:.1f}]s  bins=[{start_bin}, {end_bin}]")

    # ── Build dataset and DataLoader ─────────────────────────────
    dataset = SpikeHistoryDataset(spikes, M_cut)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=False, pin_memory=True, num_workers=0
    )

    # Preload full dataset to GPU for Block 1 (kernel update uses all data)
    Y_hist_full_gpu = torch.tensor(dataset.Y_hist, dtype=torch.float32, device=device)
    Y_curr_full_gpu = torch.tensor(dataset.Y_curr, dtype=torch.float32, device=device)

    # ── Initialize parameters ────────────────────────────────────
    A_diag_init = init_A_diag_from_spikes(spikes, dt)
    A_off_init = init_A_off_from_spikes(spikes, dt)
    B_init = init_B_from_spikes(spikes, dt)
    kappa_init = init_kappa_damped_oscillator(M_cut, decay=args.kappa_decay)

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

    # Optimizer and scheduler for Block 2
    total_b2_epochs = args.num_em_iters * args.block2_epochs
    decay_start = args.delay_em_iter_4_lrDecay * args.block2_epochs
    decay_epochs = max(1, total_b2_epochs - decay_start)
    optimizer = optim.Adam(net_model.parameters(), lr=args.lr_net, fused=True)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=args.lr_end_factor,
        total_iters=decay_epochs, last_epoch=-1
    )
    scheduler._step_count = 1

    # ── History tracking ─────────────────────────────────────────
    h_b1_nll = []
    h_b2_loss, h_b2_nll, h_b2_l1 = [], [], []
    h_rho, h_nz, h_lr = [], [], []
    h_kappa = [kappa_init.copy()]

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
            A_diag_fixed = net_model.A_diag.data.clone()
            A_off_fixed = net_model.A_off.data.clone()
            # Zero diagonal of A_off
            idx = torch.arange(N, device=device)
            A_off_fixed[idx, idx] = 0.0
            B_fixed = net_model.B.data.clone()

        kappa, b1_nll = run_block1_kernel_update(
            Y_hist_full_gpu, Y_curr_full_gpu,
            A_diag_fixed, A_off_fixed, B_fixed,
            kappa, dt, eta_clip, args.lambda2,
            args.lr_kappa, args.block1_iter, M_cut
        )
        tb1 = time.time() - tb1_start
        h_b1_nll.append(b1_nll)
        h_kappa.append(kappa.cpu().numpy().copy())

        # ── Block 2: network update ──────────────────────────────
        tb2_start = time.time()

        # Fix kappa for this block
        kappa_fixed = kappa.detach().clone()

        for ep in range(args.block2_epochs):
            b2_epoch_global += 1
            s_nll = 0.0
            s_l1 = 0.0
            s_loss = 0.0
            n_batches = 0
            log_dt = math.log(dt)

            for bi, (Y_hist_batch, Y_curr_batch) in enumerate(loader):
                Y_hist_batch = Y_hist_batch.to(device, non_blocking=True)
                Y_curr_batch = Y_curr_batch.to(device, non_blocking=True)

                S = compute_S(Y_hist_batch, kappa_fixed)
                Y_prev = Y_hist_batch[:, 0, :]

                optimizer.zero_grad(set_to_none=True)
                mu, eta_c = net_model(Y_prev, S, dt, eta_clip)
                nll = poisson_nll(mu, eta_c, Y_curr_batch, log_dt) / Y_curr_batch.shape[0]

                if args.lambda3 > 0:
                    l1_term = args.lambda3 * offdiag_l1_mean(net_model.A_off, N)
                else:
                    l1_term = torch.tensor(0.0, device=device)

                loss = nll + l1_term
                loss.backward()
                optimizer.step()

                # Post-step projections
                enforce_zero_diagonal_(net_model.A_off.data, N)
                if apply_prune and args.lambda3 > 0:
                    offdiag_soft_threshold_(
                        net_model.A_off.data,
                        optimizer.param_groups[0]["lr"],
                        args.lambda3, N
                    )
                if apply_rho and (bi % args.rho_every == 0):
                    enforce_spectral_radius_(net_model.A_off.data, args.rho_max, N)

                s_nll += nll.item()
                s_l1 += l1_term.item()
                s_loss += loss.item()
                n_batches += 1

            if em > args.delay_em_iter_4_lrDecay:
                scheduler.step()

        tb2 = time.time() - tb2_start

        # End-of-EM-iter metrics (from last epoch of Block 2)
        nb = max(1, n_batches)
        with torch.no_grad():
            nz = int((net_model.A_off[off_mask].abs() > args.minW).sum().item())
            rho = float(torch.linalg.eigvals(net_model.A_off.data).abs().max().item())

        met = dict(loss=s_loss / nb, nll=s_nll / nb, l1=s_l1 / nb, rho=rho, nz=nz)
        h_b2_loss.append(met["loss"])
        h_b2_nll.append(met["nll"])
        h_b2_l1.append(met["l1"])
        h_rho.append(met["rho"])
        h_nz.append(met["nz"])
        h_lr.append(float(optimizer.param_groups[0]["lr"]))

        if args.verb > 0:
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
    #  Save results
    # ══════════════════════════════════════════════════════════════
    print(f"\nTotal EM time: {time.time() - t_start:.1f}s")

    A_diag_hat = net_model.A_diag.detach().cpu().numpy()
    A_off_hat = net_model.A_off.detach().cpu().numpy()
    np.fill_diagonal(A_off_hat, 0.0)
    B_hat = net_model.B.detach().cpu().numpy()
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
        "b1_nll_em":      np.array(h_b1_nll, dtype=np.float64),
        "b2_loss_em":     np.array(h_b2_loss, dtype=np.float64),
        "b2_nll_em":      np.array(h_b2_nll, dtype=np.float64),
        "b2_l1_em":       np.array(h_b2_l1, dtype=np.float64),
        "rho_em":         np.array(h_rho, dtype=np.float64),
        "nz_edges_em":    np.array(h_nz, dtype=np.int64),
        "learning_rates": np.array(h_lr, dtype=np.float64),
    }

    outMD = dict(spikeMD)
    outMD["fit_type"] = "memKernEM"
    outMD["train"] = {
        "num_em_iters":           args.num_em_iters,
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
        "num_samples":            N_samples,
        "time_step_sec":          dt,
        "eta_clip":               eta_clip,
        "time_range_sec":         [t0_sec, t1_sec],
        "time_range_bins":        [start_bin, end_bin],
        "kappa_decay_init":       args.kappa_decay,
        "seed":                   args.seed,
    }
    outMD["provenance"] = dict(spikeMD.get("provenance", {}))
    outMD["provenance"]["memKernEM_file"] = outF

    write_data_npz(outD, outFF, metaD=outMD)
    print(f"\nSaved: {outFF}")
    print(f"  basePath={args.basePath}")


if __name__ == "__main__":
    main()
