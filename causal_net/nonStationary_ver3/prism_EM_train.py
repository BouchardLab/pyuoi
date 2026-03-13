#!/usr/bin/env python3
"""
prism_EM_train.py — EM training for non-stationary Poisson GLM.

Jointly infers from observed spikes only (no ground truth):
  - shared connectivity matrix  A    (N×N)
  - per-state bias vectors      B    (M×N)
  - time-varying coefficients   c_hat (T×M)  on the probability simplex

Algorithm: Block Coordinate Descent
  E-step: single forward sweep PGD over time bins → update c_hat
  M-step: m_epochs of Adam with DataLoader        → update (A, B)

Uses uniformly weighted Poisson NLL in both steps, L1 off-diagonal
soft-thresholding, and hard spectral radius projection on A.
Numerical tricks and defaults follow prism_Mstep_train3.py (M-step)
and prism_Estep_train.py (E-step).

Usage:
  ./prism_EM_train.py --dataName <name> --basePath $basePath -M 3
"""

import os
import time
import math
import secrets
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz
from Util_PrismEM import init_states_vs_time, init_B_from_spikes, init_A_from_spikes


# ═══════════════════════════════════════════════════════════════════════
#  Model  (from prism_Mstep_train3.py)
# ═══════════════════════════════════════════════════════════════════════

class SwitchingPoissonGLM(nn.Module):
    """Shared A (N×N), per-state B (M×N), mixed by coefficients c."""

    def __init__(self, N, M, eta_clip):
        super().__init__()
        self.N = N
        self.M = M
        self.eta_clip = eta_clip
        self.A = nn.Parameter(torch.randn(N, N) * 0.1)
        self.B = nn.Parameter(torch.randn(M, N) * 0.1)

    def forward(self, y_prev, c, dt):
        """y_prev (batch, N), c (batch, M) → predicted rate (batch, N)."""
        eta = y_prev @ self.A.t() + c @ self.B
        return torch.exp(torch.clamp(eta, max=self.eta_clip)) * dt


# ═══════════════════════════════════════════════════════════════════════
#  Dataset  (c array updated in-place after each E-step)
# ═══════════════════════════════════════════════════════════════════════

class PairDataset(Dataset):
    """(y_prev, y_curr, c) triplets.

    self.c is a mutable numpy array updated in-place after each E-step.
    """

    def __init__(self, y_prev, y_curr, c):
        self.y_prev = y_prev
        self.y_curr = y_curr
        self.c = c

    def __len__(self):
        return self.y_prev.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.y_prev[idx]).float(),
            torch.from_numpy(self.y_curr[idx]).float(),
            torch.from_numpy(self.c[idx]).float(),
        )


def init_distributed():
    """Initialize torch.distributed from torchrun environment."""
    is_dist = (int(os.environ.get("WORLD_SIZE", "1")) > 1) or ("RANK" in os.environ)
    if is_dist:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return is_dist, rank, world_size, local_rank, device


def cleanup_distributed(is_dist):
    if is_dist and dist.is_initialized():
        dist.destroy_process_group()


def pair_range_for_rank(t_pairs, world_size, rank):
    """Return inclusive pair-index range [p0, p1] assigned to given rank."""
    base = t_pairs // world_size
    rem = t_pairs % world_size
    n_loc = base + (1 if rank < rem else 0)
    p0 = rank * base + min(rank, rem)
    p1 = p0 + n_loc - 1
    return p0, p1, n_loc


# ═══════════════════════════════════════════════════════════════════════
#  E-step: simplex PGD  (from prism_Estep_train.py)
# ═══════════════════════════════════════════════════════════════════════

def project_to_simplex(v):
    """Project 1-D vector onto the probability simplex (Duchi et al.)."""
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
    return torch.clamp(v - theta, min=0.0)


def run_estep(Y_prev, Y_curr, A, B, c_hat,
              dt, eta_clip, lambda2, lr, pgd_iter):
    """Single forward sweep E-step.  Updates c_hat in-place on GPU.

    Args:
        Y_prev, Y_curr: (T_pairs, N) GPU tensors
        A: (N, N)  detached parameter tensor
        B: (M, N)  detached parameter tensor
        c_hat: (T_full, M) GPU tensor, modified in-place

    Returns:
        mean NLL over all time pairs
    """
    T_eff = c_hat.shape[0]
    log_dt = math.log(dt)
    c_prev = c_hat[0].clone()
    nll_sum = 0.0

    for t in range(1, T_eff):
        y_p = Y_prev[t - 1]
        y_c = Y_curr[t - 1]

        Z = (A @ y_p)[None, :] + B           # (M, N)

        c_t = c_hat[t].clone()
        for _ in range(pgd_iter):
            eta = c_t @ Z                     # (N,)
            eta_c = torch.clamp(eta, max=eta_clip)
            lam = torch.exp(eta_c) * dt
            g = Z @ (lam - y_c) + 2.0 * lambda2 * (c_t - c_prev)
            c_t = project_to_simplex(c_t - lr * g)

        eta = c_t @ Z
        eta_c = torch.clamp(eta, max=eta_clip)
        lam = torch.exp(eta_c) * dt
        nll_sum += float((lam - y_c * (eta_c + log_dt)).sum().item())

        c_hat[t] = c_t
        c_prev = c_t

    return nll_sum / max(1, T_eff - 1)


def run_estep_shard(Y_prev, Y_curr, A, B, c_hat,
                    dt, eta_clip, lambda2, lr, pgd_iter,
                    t0, t1):
    """E-step update on a shard of time bins t in [t0, t1], inclusive.

    Returns:
      nll_sum_local, n_pairs_local
    """
    if t1 < t0:
        return 0.0, 0
    log_dt = math.log(dt)
    c_prev = c_hat[t0 - 1].clone()
    nll_sum = 0.0
    n_pairs = 0

    for t in range(t0, t1 + 1):
        y_p = Y_prev[t - 1]
        y_c = Y_curr[t - 1]

        Z = (A @ y_p)[None, :] + B

        c_t = c_hat[t].clone()
        for _ in range(pgd_iter):
            eta = c_t @ Z
            eta_c = torch.clamp(eta, max=eta_clip)
            lam = torch.exp(eta_c) * dt
            g = Z @ (lam - y_c) + 2.0 * lambda2 * (c_t - c_prev)
            c_t = project_to_simplex(c_t - lr * g)

        eta = c_t @ Z
        eta_c = torch.clamp(eta, max=eta_clip)
        lam = torch.exp(eta_c) * dt
        nll_sum += float((lam - y_c * (eta_c + log_dt)).sum().item())

        c_hat[t] = c_t
        c_prev = c_t
        n_pairs += 1

    return nll_sum, n_pairs


# ═══════════════════════════════════════════════════════════════════════
#  M-step helpers  (from prism_Mstep_train3.py)
# ═══════════════════════════════════════════════════════════════════════

def offdiag_soft_threshold_(A, lr, lam):
    """In-place proximal L1 on off-diagonal A elements."""
    if lam <= 0:
        return
    with torch.no_grad():
        n = A.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=A.device)
        thresh = lr * lam
        v = A[mask]
        A[mask] = v.sign() * (v.abs() - thresh).clamp(min=0.0)


def enforce_spectral_radius_(A, rho_max):
    """Hard in-place spectral projection.  Returns rho before projection."""
    with torch.no_grad():
        rho = torch.linalg.eigvals(A).abs().max().item()
        if rho > rho_max:
            A.mul_(rho_max / rho)
    return rho


def sync_AB_via_rank0_avg_then_broadcast_(mdl, rho_max):
    """Rank-0 averages A/B, enforces rho on averaged A, then broadcasts A/B."""
    if not (dist.is_available() and dist.is_initialized()):
        enforce_spectral_radius_(mdl.A, rho_max)
        return

    rank = dist.get_rank()
    world = float(dist.get_world_size())
    with torch.no_grad():
        A_buf = mdl.A.data.clone()
        B_buf = mdl.B.data.clone()

        dist.reduce(A_buf, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(B_buf, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            A_buf.div_(world)
            B_buf.div_(world)
            enforce_spectral_radius_(A_buf, rho_max)

        dist.broadcast(A_buf, src=0)
        dist.broadcast(B_buf, src=0)
        mdl.A.data.copy_(A_buf)
        mdl.B.data.copy_(B_buf)


def run_mstep_epoch(model, loader, optimizer, device, dt,
                    l1_wt, lambda3, rho_max,
                    rho_every, apply_prune, apply_rho, off_mask, minW):
    """One DataLoader pass updating A and B.  Returns metrics dict.
    """
    model.train()
    mdl = model.module if hasattr(model, "module") else model
    s_tot = s_nll = s_l1 = 0.0
    eps = 1e-8

    for bi, (yp, yc, cc) in enumerate(loader):
        yp = yp.to(device, non_blocking=True)
        yc = yc.to(device, non_blocking=True)
        cc = cc.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(yp, cc, dt)
        nll = (-yc * torch.log(pred + eps) + pred).mean()
        if lambda3 > 0:
            n = mdl.A.shape[0]
            off_abs_mean = (mdl.A.abs() * l1_wt).sum() / float(n * (n - 1))
            l1 = lambda3 * off_abs_mean
        else:
            l1 = torch.tensor(0.0, device=device)
        loss = nll + l1
        loss.backward()
        optimizer.step()

        if apply_prune and lambda3 > 0:
            offdiag_soft_threshold_(mdl.A, optimizer.param_groups[0]["lr"], lambda3)
        if apply_rho and (bi % rho_every == 0):
            sync_AB_via_rank0_avg_then_broadcast_(mdl, rho_max)

        s_tot += loss.item()
        s_nll += nll.item()
        s_l1 += l1.item()

    nb = max(1, len(loader))
    if dist.is_available() and dist.is_initialized():
        v = torch.tensor([s_tot, s_nll, s_l1, float(nb)],
                         dtype=torch.float64, device=device)
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
        s_tot = float(v[0].item())
        s_nll = float(v[1].item())
        s_l1 = float(v[2].item())
        nb = int(v[3].item())

    with torch.no_grad():
        nz = int((mdl.A[off_mask].abs() > minW).sum().item())
        rho = float(torch.linalg.eigvals(mdl.A).abs().max().item())

    return dict(loss=s_tot / nb, nll=s_nll / nb, l1=s_l1 / nb, rho=rho, nz=nz)


# ═══════════════════════════════════════════════════════════════════════
#  Viterbi  (from prism_Estep_train.py)
# ═══════════════════════════════════════════════════════════════════════

def viterbi_decode(c_hat, p_stay):
    """Viterbi decode: find most likely state path from simplex coefficients."""
    eps = 1e-12
    T, M = c_hat.shape
    if M == 1:
        return np.zeros(T, dtype=np.int64)
    p_sw = (1.0 - p_stay) / (M - 1)
    trans = np.full((M, M), p_sw, dtype=np.float64)
    np.fill_diagonal(trans, p_stay)
    lt = np.log(np.clip(trans, eps, 1.0))
    le = np.log(np.clip(c_hat, eps, 1.0))

    delta = np.zeros((T, M), dtype=np.float64)
    psi = np.zeros((T, M), dtype=np.int64)
    delta[0] = le[0]
    for t in range(1, T):
        sc = delta[t - 1][:, None] + lt
        psi[t] = np.argmax(sc, axis=0)
        delta[t] = sc[psi[t], np.arange(M)] + le[t]

    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


# ═══════════════════════════════════════════════════════════════════════
#  Args
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="EM training: non-stationary Poisson GLM")

    p.add_argument("--dataName", required=True)
    p.add_argument("--basePath",
                   default="/pscratch/sd/b/balewski/2026_causalNet_tmp3/")
    p.add_argument("--num_states", "-M", type=int, required=True,
                   help="Number of latent states")

    g = p.add_argument_group("EM structure")
    g.add_argument("--num_em_iters", type=int, default=20,
                   help="Outer EM iterations")
    g.add_argument("--m_epochs", type=int, default=5,
                   help="M-step Adam epochs per EM iteration")

    g = p.add_argument_group("E-step")
    g.add_argument("--pgd_iter", type=int, default=5,
                   help="PGD iterations per time bin")
    g.add_argument("--lr_estep", type=float, default=0.03,
                   help="PGD step size")
    g.add_argument("--lambda2", type=float, default=2.0,
                   help="Temporal smoothness weight")

    g = p.add_argument_group("M-step")
    g.add_argument("--lr_mstep", type=float, default=0.003,
                   help="Adam learning rate (initial)")
    g.add_argument("--lr_end_factor", type=float, default=0.1,
                   help="LR decays to lr_mstep * lr_end_factor")
    g.add_argument("--lambda3", type=float, default=0.02,
                   help="L1 penalty on off-diagonal A")
    g.add_argument("--rho_max", type=float, default=0.95,
                   help="Hard spectral radius ceiling for A")
    g.add_argument("--prescale_m_step_4_ArhoMax", type=int, default=50,
                   help="Spectral projection frequency (batches)")
    g.add_argument("--delay_em_iter_4_ArhoMax", type=int, default=3,
                   help="EM iter to start rho_max enforcement "
                        "(default: int(num_em_iters * 0.6))")
    g.add_argument("--delay_em_iter_4_lrDecay", type=int, default=1,
                   help="EM iter to start LR decay "
                        "(default: int(num_em_iters * 0.7))")
    g.add_argument("--delay_em_iter_4_Aprune", type=int, default=1,
                   help="EM iter to start L1 pruning "
                        "(default: num_em_iters // 3)")
    g.add_argument("--batch_size", type=int, default=2048)
    g.add_argument("--minW", type=float, default=0.01,
                   help="Threshold for edge counting (reporting only)")
    g = p.add_argument_group("data")
    g.add_argument("-T", "--time_range_sec", default=[0.0, 60.0],
                   nargs=2, type=float,
                   help="Time window [t0, t1] in seconds")

    g = p.add_argument_group("decode")
    g.add_argument("--decode_dwell_sec", type=float, default=0.3,
                   help="Viterbi dwell prior τ_dwell")

    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--fitName", type=str, default=None,
                   help="Optional output fit name; if omitted a random name is generated")
    g.add_argument("--init_states", type=str, default="data",
                   choices=["data", "rand"],
                   help="Initial latent-state mode: 'data' or 'rand'")
    g.add_argument("--init_B", type=str, default="data",
                   choices=["data", "rand"],
                   help="Initial B mode: 'data' or 'rand'")
    g.add_argument("--init_A", type=str, default="data",
                   choices=["data", "rand"],
                   help="Initial A mode: 'data' or 'rand'")
    g.add_argument("-v", "--verb", type=int, default=1)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    is_dist, rank, world_size, local_rank, device = init_distributed()
    inpPath = os.path.join(args.basePath, "spikesData")
    outPath = os.path.join(args.basePath, "prismFit")
    out_ok = True
    out_err = ""
    if rank == 0:
        out_ok = os.path.exists(outPath)
        if not out_ok:
            out_err = f"Output dir missing: {outPath}"
    if is_dist:
        msg = [out_ok, out_err]
        dist.broadcast_object_list(msg, src=0)
        out_ok, out_err = msg
    if not out_ok:
        raise FileNotFoundError(out_err)

    if args.delay_em_iter_4_Aprune is None:
        args.delay_em_iter_4_Aprune = int(args.num_em_iters * 0.3)
    if args.delay_em_iter_4_ArhoMax is None:
        args.delay_em_iter_4_ArhoMax = int(args.num_em_iters * 0.6)
    if args.delay_em_iter_4_lrDecay is None:
        args.delay_em_iter_4_lrDecay = int(args.num_em_iters * 0.7)
    
    if rank == 0:
        print("\nEM-train args:", vars(args), "\n")
        print(f"world_size={world_size}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # ── load spikes (rank 0 only), then broadcast ───────────────
    if rank == 0:
        spikesFF = os.path.join(inpPath, f"{args.dataName}.spikes.npz")
        spikeD, spikeMD = read_data_npz(spikesFF, verb=args.verb > 0)
        spikes = spikeD["spikes"]
        single_rates = spikeD["single_rates"]
    else:
        spikeMD = None
        spikes = None
        single_rates = None

    if is_dist:
        obj = [spikeMD]
        dist.broadcast_object_list(obj, src=0)
        spikeMD = obj[0]

        if rank == 0:
            shp = torch.tensor(spikes.shape, dtype=torch.int64, device=device)
        else:
            shp = torch.zeros(2, dtype=torch.int64, device=device)
        dist.broadcast(shp, src=0)
        T_raw = int(shp[0].item())
        N = int(shp[1].item())

        if rank == 0:
            spikes_t = torch.as_tensor(spikes, dtype=torch.int32, device=device).contiguous()
        else:
            spikes_t = torch.empty((T_raw, N), dtype=torch.int32, device=device)
        dist.broadcast(spikes_t, src=0)
        spikes = spikes_t.cpu().numpy()
    else:
        T_raw, N = spikes.shape

    dt = float(spikeMD["time_step_sec"])
    eta_clip = float(spikeMD["poisson_eta_clip"])
    prov = dict(spikeMD["provenance"])
    M = args.num_states

    # ── time range selection ─────────────────────────────────────
    t0_sec, t1_sec = float(args.time_range_sec[0]), float(args.time_range_sec[1])
    if t1_sec < t0_sec:
        t0_sec, t1_sec = t1_sec, t0_sec
    start_bin = max(0, int(math.floor(t0_sec / dt)))
    end_bin = min(T_raw - 1, int(math.floor(t1_sec / dt)))
    if end_bin <= start_bin:
        raise ValueError("time_range_sec too small; need at least two bins")
    spikes = spikes[start_bin : end_bin + 1]
    T_full = spikes.shape[0]
    T_pairs = T_full - 1
    if rank == 0:
        print(f"N={N} EM={args.num_em_iters}  M={M}  T={T_full}  pairs={T_pairs}  "
              f"dt={dt}  eta_clip={eta_clip}  "
              f"time=[{t0_sec:.1f}, {t1_sec:.1f}]s  bins=[{start_bin}, {end_bin}]")

    # ── time pairs: GPU for E-step, CPU numpy for DataLoader ────
    yp_np = spikes[:-1].astype(np.float32)
    yc_np = spikes[1:].astype(np.float32)
    Yp_gpu = torch.tensor(yp_np, device=device)
    Yc_gpu = torch.tensor(yc_np, device=device)

    # ── initial states and parameter seeds (rank 0 computes, all ranks receive) ──
    if rank == 0:
        c_init_np, S_init, init_state_md, freq_h1d = init_states_vs_time(spikes, dt, args)
        A_seed_np, init_A_md = init_A_from_spikes(spikes, args)
        B_seed_np, init_B_md = init_B_from_spikes(spikes, dt, args)
    else:
        c_init_np = np.empty((T_full, M), dtype=np.float32)
        S_init = np.empty((T_full,), dtype=np.int64)
        freq_h1d = np.empty((0,), dtype=np.float32)
        A_seed_np = None
        B_seed_np = None
        init_state_md = None
        init_A_md = None
        init_B_md = None

    if is_dist:
        c_t = torch.as_tensor(c_init_np, dtype=torch.float32, device=device)
        s_t = torch.as_tensor(S_init, dtype=torch.int64, device=device)
        dist.broadcast(c_t, src=0)
        dist.broadcast(s_t, src=0)
        c_init_np = c_t.cpu().numpy()
        S_init = s_t.cpu().numpy()

        f_len_t = torch.tensor([int(freq_h1d.shape[0]) if rank == 0 else 0],
                               dtype=torch.int64, device=device)
        dist.broadcast(f_len_t, src=0)
        f_len = int(f_len_t.item())
        if rank != 0:
            freq_h1d = np.empty((f_len,), dtype=np.float32)
        f_t = (torch.as_tensor(freq_h1d, dtype=torch.float32, device=device)
               if rank == 0 else
               torch.empty((f_len,), dtype=torch.float32, device=device))
        if f_len > 0:
            dist.broadcast(f_t, src=0)
        freq_h1d = f_t.cpu().numpy()

        has_A_t = torch.tensor([1 if (rank == 0 and A_seed_np is not None) else 0],
                               dtype=torch.int64, device=device)
        dist.broadcast(has_A_t, src=0)
        if int(has_A_t.item()) == 1:
            A_t = (torch.as_tensor(A_seed_np, dtype=torch.float32, device=device)
                   if rank == 0 else
                   torch.empty((N, N), dtype=torch.float32, device=device))
            dist.broadcast(A_t, src=0)
            A_seed_np = A_t.cpu().numpy()
        else:
            A_seed_np = None

        has_B_t = torch.tensor([1 if (rank == 0 and B_seed_np is not None) else 0],
                               dtype=torch.int64, device=device)
        dist.broadcast(has_B_t, src=0)
        if int(has_B_t.item()) == 1:
            B_t = (torch.as_tensor(B_seed_np, dtype=torch.float32, device=device)
                   if rank == 0 else
                   torch.empty((M, N), dtype=torch.float32, device=device))
            dist.broadcast(B_t, src=0)
            B_seed_np = B_t.cpu().numpy()
        else:
            B_seed_np = None

    c_hat_gpu = torch.tensor(c_init_np, dtype=torch.float32, device=device)
    c_pairs_np = c_init_np[1:].copy()

    dataset = PairDataset(yp_np, yc_np, c_pairs_np)
    if is_dist:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler,
            drop_last=False, pin_memory=True, num_workers=0
        )
    else:
        sampler = None
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            drop_last=False, pin_memory=True, num_workers=0
        )

    # ── model + optimizer + scheduler ────────────────────────────
    model = SwitchingPoissonGLM(N, M, eta_clip).to(device)
    if A_seed_np is not None:
        with torch.no_grad():
            model.A.copy_(torch.tensor(A_seed_np, dtype=torch.float32, device=device))
    if B_seed_np is not None:
        with torch.no_grad():
            model.B.copy_(torch.tensor(B_seed_np, dtype=torch.float32, device=device))
    A_init_np = model.A.detach().cpu().numpy().copy()
    B_init_np = model.B.detach().cpu().numpy().copy()
    B_init_out_np = B_init_np[0] if B_init_np.ndim == 2 and B_init_np.shape[0] > 1 else B_init_np

    if is_dist:
        model = DDP(model, device_ids=[local_rank])

    total_m_epochs = args.num_em_iters * args.m_epochs
    decay_start_m_epoch = args.delay_em_iter_4_lrDecay * args.m_epochs
    decay_m_epochs = max(1, total_m_epochs - decay_start_m_epoch)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_mstep, fused=True)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=args.lr_end_factor,
        total_iters=decay_m_epochs, last_epoch=-1
    )
    scheduler._step_count = 1

    mdl = model.module if hasattr(model, "module") else model
    off_mask = ~torch.eye(N, dtype=torch.bool, device=device)
    l1_wt = torch.ones(N, N, device=device)
    l1_wt[torch.eye(N, dtype=torch.bool, device=device)] = 0.0

    # ── history ──────────────────────────────────────────────────
    h_e_nll = []
    h_m_loss, h_m_nll, h_m_l1 = [], [], []
    h_rho, h_nz, h_lr = [], [], []

    t_start = time.time()
    m_epoch_global = 0
    p_stay = math.exp(-dt / args.decode_dwell_sec)
    onehot_states = np.eye(M, dtype=np.float32)

    # ══════════════════════════════════════════════════════════════
    #  EM loop
    # ══════════════════════════════════════════════════════════════
    for em in range(1, args.num_em_iters + 1):
        if rank == 0:
            if em == args.delay_em_iter_4_ArhoMax + 1:
                print(f"threshold passed: 'delay_em_iter_4_ArhoMax': {args.delay_em_iter_4_ArhoMax} (em={em})")
            if em == args.delay_em_iter_4_lrDecay + 1:
                print(f"threshold passed: 'delay_em_iter_4_lrDecay': {args.delay_em_iter_4_lrDecay} (em={em})")
            if em == args.delay_em_iter_4_Aprune + 1:
                print(f"threshold passed: 'delay_em_iter_4_Aprune': {args.delay_em_iter_4_Aprune} (em={em})")
        apply_prune = em > args.delay_em_iter_4_Aprune
        apply_rho = em > args.delay_em_iter_4_ArhoMax

        # ── E-step ───────────────────────────────────────────────
        te0 = time.time()
        with torch.no_grad():
            if is_dist:
                p0, p1, n_loc = pair_range_for_rank(T_pairs, world_size, rank)
                t0 = p0 + 1
                t1 = p1 + 1
                nll_local, n_pair_local = run_estep_shard(
                    Yp_gpu, Yc_gpu, mdl.A.data, mdl.B.data, c_hat_gpu,
                    dt, eta_clip, args.lambda2, args.lr_estep, args.pgd_iter,
                    t0, t1
                )

                c_upd = torch.zeros_like(c_hat_gpu)
                c_msk = torch.zeros((T_full, 1), dtype=torch.float32, device=device)
                c_upd[0] = c_hat_gpu[0]
                c_msk[0] = 1.0
                if n_loc > 0:
                    c_upd[t0:t1 + 1] = c_hat_gpu[t0:t1 + 1]
                    c_msk[t0:t1 + 1] = 1.0

                dist.all_reduce(c_upd, op=dist.ReduceOp.SUM)
                dist.all_reduce(c_msk, op=dist.ReduceOp.SUM)
                c_avg = c_upd / torch.clamp(c_msk, min=1.0)
                c_hat_gpu.copy_(torch.where(c_msk > 0.0, c_avg, c_hat_gpu))

                ev = torch.tensor([nll_local, float(n_pair_local)],
                                  dtype=torch.float64, device=device)
                dist.all_reduce(ev, op=dist.ReduceOp.SUM)
                e_nll = float(ev[0].item() / max(1.0, ev[1].item()))
            else:
                e_nll = run_estep(
                    Yp_gpu, Yc_gpu, mdl.A.data, mdl.B.data,
                    c_hat_gpu, dt, eta_clip,
                    args.lambda2, args.lr_estep, args.pgd_iter
                )

        te = time.time() - te0
        h_e_nll.append(e_nll)

        # sync E-step states to CPU dataset (in-place update) for M-step:
        # Viterbi decode -> one-hot rows (classification-style M-step).
        c_hat_np_iter = c_hat_gpu.cpu().numpy()
        if is_dist:
            if rank == 0:
                s_iter_np = viterbi_decode(c_hat_np_iter, p_stay).astype(np.int64, copy=False)
                s_iter_t = torch.as_tensor(s_iter_np, dtype=torch.int64, device=device)
            else:
                s_iter_t = torch.empty((T_full,), dtype=torch.int64, device=device)
            dist.broadcast(s_iter_t, src=0)
            s_iter_np = s_iter_t.cpu().numpy()
        else:
            s_iter_np = viterbi_decode(c_hat_np_iter, p_stay).astype(np.int64, copy=False)
        np.copyto(c_pairs_np, onehot_states[s_iter_np[1:]])

        # ── M-step: m_epochs of Adam ─────────────────────────────
        tm0 = time.time()
        for _ in range(args.m_epochs):
            m_epoch_global += 1
            if sampler is not None:
                sampler.set_epoch(m_epoch_global)
            met = run_mstep_epoch(
                model, loader, optimizer, device, dt,
                l1_wt, args.lambda3, args.rho_max,
                args.prescale_m_step_4_ArhoMax, apply_prune, apply_rho,
                off_mask, args.minW
            )
            if em > args.delay_em_iter_4_lrDecay:
                scheduler.step()

            h_m_loss.append(met["loss"])
            h_m_nll.append(met["nll"])
            h_m_l1.append(met["l1"])
            h_rho.append(met["rho"])
            h_nz.append(met["nz"])
            h_lr.append(float(optimizer.param_groups[0]["lr"]))
        tm = time.time() - tm0

        if rank == 0 and args.verb > 0:
            n_off = int(off_mask.sum().item())
            sp = 1.0 - met["nz"] / max(1, n_off)
            print(
                f"EM {em:3d}/{args.num_em_iters}  "
                f"E_nll={e_nll:.4e}({te:.1f}s)  "
                f"M_nll={met['nll']:.4e} l1={met['l1']:.4e} "
                f"rho={met['rho']:.4f} sp={sp:.3f} nz={met['nz']} "
                f"lr={h_lr[-1]:.2e} ({tm:.1f}s)  "
                f"tot={time.time() - t_start:.0f}s"
            )

    if rank == 0:
        print(f"\nTotal EM time: {time.time() - t_start:.1f}s")

        # ── Final Viterbi decode ─────────────────────────────────
        c_hat_np = c_hat_gpu.cpu().numpy()
        S_hat = viterbi_decode(c_hat_np, p_stay)
        S_hat_CL = (1.0 - c_hat_np.max(axis=1)).astype(np.float32)

        mdl = model.module if hasattr(model, "module") else model
        A_hat = mdl.A.detach().cpu().numpy()
        B_hat = mdl.B.detach().cpu().numpy()

        if args.fitName is None:
            h6 = secrets.token_hex(3)
            outF = f"{args.dataName}-EM-{h6}"
        else:
            outF = args.fitName
        outFF = os.path.join(outPath, f"{outF}.prismEM.npz")
        prov["EMtrain_file"] = outF

        outD = {
            "A_init":         A_init_np.astype(np.float32),
            "A_hat":          A_hat.astype(np.float32),
            "B_init":         B_init_out_np.astype(np.float32),
            "B_hat":          B_hat.astype(np.float32),
            "freq_h1d":       np.asarray(freq_h1d, dtype=np.float32),
            "c_init":         c_init_np.astype(np.float32),
            "c_hat":          c_hat_np.astype(np.float32),
            "S_init":         S_init.astype(np.int64),
            "S_hat":          S_hat.astype(np.int64),
            "S_hat_CL":       S_hat_CL,
            "single_rates":   np.asarray(single_rates),
            "e_nll_em":       np.asarray(h_e_nll, dtype=np.float64),
            "m_loss_epoch":   np.asarray(h_m_loss, dtype=np.float64),
            "m_nll_epoch":    np.asarray(h_m_nll, dtype=np.float64),
            "m_l1_epoch":     np.asarray(h_m_l1, dtype=np.float64),
            "rho_epoch":      np.asarray(h_rho, dtype=np.float64),
            "nz_edges_epoch": np.asarray(h_nz, dtype=np.int64),
            "learning_rates": np.asarray(h_lr, dtype=np.float64),
        }

        outMD = dict(spikeMD)
        outMD["fit_type"] = "prismEM"
        outMD["train"] = {
            "num_em_iters":             args.num_em_iters,
            "m_epochs":                 args.m_epochs,
            "total_m_epochs":           total_m_epochs,
            "pgd_iter":                 args.pgd_iter,
            "lr_estep":                 args.lr_estep,
            "lr_mstep":                 args.lr_mstep,
            "lr_end_factor":            args.lr_end_factor,
            "lambda2":                  args.lambda2,
            "lambda3":                  args.lambda3,
            "rho_max":                  args.rho_max,
            "prescale_m_step_4_ArhoMax":  args.prescale_m_step_4_ArhoMax,
            "delay_em_iter_4_ArhoMax":    args.delay_em_iter_4_ArhoMax,
            "rho_sync_mode":            "broadcast",
            "delay_em_iter_4_lrDecay":    args.delay_em_iter_4_lrDecay,
            "delay_em_iter_4_Aprune":         args.delay_em_iter_4_Aprune,
            "mstep_state_mode":         "viterbi_onehot",
            "batch_size":               args.batch_size,
            "minW":                     args.minW,
            "num_states":               M,
            "num_neurons":              N,
            "num_time_bins":            T_full,
            "time_step_sec":            dt,
            "eta_clip":                 eta_clip,
            "time_range_sec":           [t0_sec, t1_sec],
            "time_range_bins":          [start_bin, end_bin],
            "seed":                     args.seed,
        }
        outMD["states_recovery_eval"] = {
            "decode":           "viterbi",
            "decode_dwell_sec": args.decode_dwell_sec,
        }
        outMD["init_A"] = init_A_md
        outMD["init_state"] = init_state_md
        outMD["init_B"] = init_B_md
        outMD["provenance"] = prov

        write_data_npz(outD, outFF, metaD=outMD)
        print(f"\nSaved: {outFF}")
        print(f"  basePath={args.basePath}")
        print(f"  ./prism_EM_eval.py --basePath $basePath "
              f"--dataName {outF} -p a e f g \n")
    cleanup_distributed(is_dist)


if __name__ == "__main__":
    main()
