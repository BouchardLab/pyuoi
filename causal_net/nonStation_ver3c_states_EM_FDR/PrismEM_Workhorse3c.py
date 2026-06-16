#!/usr/bin/env python3
"""Reusable PRISM-EM training workhorse for full EM and FDR bagging."""

import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from Util_PrismEM import init_states_vs_time, init_B_from_spikes, init_A_from_spikes


class DistContext:
    def __init__(self, is_dist, rank, world_size, local_rank, device):
        self.is_dist = is_dist
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = device


class SwitchingPoissonGLM(nn.Module):
    """Shared A (N x N), per-state B (M x N), mixed by state coefficients."""

    def __init__(self, N, M, eta_clip):
        super().__init__()
        self.N = N
        self.M = M
        self.eta_clip = eta_clip
        self.A = nn.Parameter(torch.randn(N, N) * 0.1)
        self.B = nn.Parameter(torch.randn(M, N) * 0.1)

    def forward(self, y_prev, c, dt):
        eta = y_prev @ self.A.t() + c @ self.B
        return torch.exp(torch.clamp(eta, max=self.eta_clip)) * dt


class PairDataset(Dataset):
    """(y_prev, y_curr, c) triplets; c may be updated in-place between epochs."""

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


def add_prism_em_args(parser, include_time_range=True):
    """Add ordinary PRISM-EM training arguments to an argparse parser."""
    parser.add_argument("--dataName", required=True)
    parser.add_argument("--basePath",
                        default="/pscratch/sd/b/balewski/2026_causalNet_tmp3/")
    parser.add_argument("--num_states", "-M", type=int, required=True,
                        help="Number of latent states")

    g = parser.add_argument_group("EM structure")
    g.add_argument("--num_em_iters", type=int, default=20,
                   help="Outer EM iterations")
    g.add_argument("--m_epochs", type=int, default=2,
                   help="M-step Adam epochs per EM iteration")

    g = parser.add_argument_group("E-step")
    g.add_argument("--pgd_iter", type=int, default=5,
                   help="PGD iterations per time bin")
    g.add_argument("--lr_estep", type=float, default=0.03,
                   help="PGD step size")
    g.add_argument("--lambda2", type=float, default=2.0,
                   help="Temporal smoothness weight")

    g = parser.add_argument_group("M-step")
    g.add_argument("--lr_mstep", type=float, default=0.003,
                   help="Adam learning rate (initial)")
    g.add_argument("--lr_end_factor", type=float, default=0.1,
                   help="LR decays to lr_mstep * lr_end_factor")
    g.add_argument("--lambda3", type=float, default=0.02,
                   help="L1 penalty on off-diagonal A")
    g.add_argument("--rho_max", type=float, default=0.95,
                   help="Spectral radius target for soft correction of A")
    g.add_argument("--prescale_m_step_4_ArhoMax", type=int, default=120,
                   help="Spectral projection frequency in M-step batches")
    g.add_argument("--delay_em_iter_4_ArhoMax", type=int, nargs="+", default=[3],
                   help="One or two EM iters: start rho_max enforcement; optional target iter for full correction")
    g.add_argument("--delay_em_iter_4_lrDecay", type=int, nargs="+", default=[1],
                   help="One or two EM iters: start LR decay; optional target iter for final LR")
    g.add_argument("--delay_em_iter_4_Aprune", type=int, default=1,
                   help="EM iter after which L1 proximal pruning starts")
    g.add_argument("--batch_size", type=int, default=2048)

    g = parser.add_argument_group("initialization")
    g.add_argument("--init_states", type=str, default="data",
                   choices=["data", "rand"],
                   help="Initial latent-state mode for the real fit")
    g.add_argument("--init_B", type=str, default="data",
                   choices=["data", "rand"],
                   help="Initial B mode for the real fit")
    g.add_argument("--init_A", type=str, default="data",
                   choices=["data", "rand"],
                   help="Initial A mode for the real fit")
    g.add_argument("--init_A_Tmax", type=int, default=50000,
                   help="Maximum bins used by OLS/covariance A initialization")

    if include_time_range:
        g = parser.add_argument_group("data")
        g.add_argument("-T", "--time_range_sec", default=[0.0, 60.0],
                       nargs=2, type=float,
                       help="Time window [t0, t1] in seconds")

    g = parser.add_argument_group("decode")
    g.add_argument("--decode_dwell_sec", type=float, default=0.3,
                   help="Viterbi dwell prior tau_dwell")

    g = parser.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--fitName", type=str, default=None,
                   help="Optional output fit name")
    g.add_argument("-v", "--verb", type=int, default=1)
    return parser


def normalize_delay_args(args):
    if args.delay_em_iter_4_Aprune is None:
        args.delay_em_iter_4_Aprune = int(args.num_em_iters * 0.3)
    if not hasattr(args, "target_em_iter_4_ArhoMax"):
        args.delay_em_iter_4_ArhoMax, args.target_em_iter_4_ArhoMax = normalize_start_target_arg(
            args.delay_em_iter_4_ArhoMax,
            int(args.num_em_iters * 0.6),
            int(args.num_em_iters),
            "--delay_em_iter_4_ArhoMax",
        )
    if not hasattr(args, "target_em_iter_4_lrDecay"):
        args.delay_em_iter_4_lrDecay, args.target_em_iter_4_lrDecay = normalize_start_target_arg(
            args.delay_em_iter_4_lrDecay,
            int(args.num_em_iters * 0.7),
            int(args.num_em_iters),
            "--delay_em_iter_4_lrDecay",
        )
    return args


def normalize_start_target_arg(value, default_start, default_target, name):
    if value is None:
        start = int(default_start)
        target = int(default_target)
    elif isinstance(value, (list, tuple)):
        if len(value) < 1 or len(value) > 2:
            raise ValueError(f"{name} expects 1 or 2 integers")
        start = int(value[0])
        target = int(value[1]) if len(value) == 2 else int(default_target)
    else:
        start = int(value)
        target = int(default_target)

    if start < 0:
        raise ValueError(f"{name} start must be non-negative")
    if target <= start:
        raise ValueError(f"{name} target must be greater than start")
    if target > int(default_target):
        raise ValueError(f"{name} target must be <= num_em_iters ({default_target})")
    return start, target


def init_distributed():
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
    return DistContext(is_dist, rank, world_size, local_rank, device)


def cleanup_distributed(ctx):
    if ctx.is_dist and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def is_rank0(ctx):
    return ctx.rank == 0


def barrier(ctx):
    if ctx.is_dist:
        if ctx.device.type == "cuda":
            dist.barrier(device_ids=[ctx.local_rank])
        else:
            dist.barrier()


def runtime_summary(ctx):
    if ctx.device.type == "cuda":
        dev_idx = ctx.local_rank if ctx.is_dist else torch.cuda.current_device()
        dev_name = torch.cuda.get_device_name(dev_idx)
    else:
        dev_name = "cpu"
    return (
        f"distributed={ctx.is_dist} rank={ctx.rank}/{ctx.world_size} "
        f"local_rank={ctx.local_rank} device={ctx.device} "
        f"cuda_available={torch.cuda.is_available()} device_name={dev_name}"
    )


def broadcast_object(obj, ctx, src=0):
    if not ctx.is_dist:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def _torch_dtype(np_dtype):
    dt = np.dtype(np_dtype)
    if dt == np.dtype("float32"):
        return torch.float32
    if dt == np.dtype("float64"):
        return torch.float64
    if dt == np.dtype("int64"):
        return torch.int64
    if dt == np.dtype("int32"):
        return torch.int32
    if dt == np.dtype("int16"):
        return torch.int16
    if dt == np.dtype("int8"):
        return torch.int8
    if dt == np.dtype("uint8"):
        return torch.uint8
    if dt == np.dtype("bool"):
        return torch.bool
    raise TypeError(f"Unsupported broadcast dtype: {dt}")


def broadcast_array(arr, ctx, src=0):
    if not ctx.is_dist:
        return arr
    if is_rank0(ctx):
        arr = np.asarray(arr)
        info = (tuple(arr.shape), str(arr.dtype))
    else:
        info = None
    shape, dtype_s = broadcast_object(info, ctx, src=src)
    dtype = np.dtype(dtype_s)
    torch_dtype = _torch_dtype(dtype)
    if is_rank0(ctx):
        tens = torch.as_tensor(arr, dtype=torch_dtype, device=ctx.device).contiguous()
    else:
        tens = torch.empty(shape, dtype=torch_dtype, device=ctx.device)
    dist.broadcast(tens, src=src)
    return tens.cpu().numpy()


def broadcast_optional_array(arr, ctx, src=0):
    if not ctx.is_dist:
        return arr
    has_arr = bool(arr is not None) if is_rank0(ctx) else False
    has_arr = broadcast_object(has_arr, ctx, src=src)
    if not has_arr:
        return None
    return broadcast_array(arr, ctx, src=src)


def pair_range_for_rank(t_pairs, world_size, rank):
    base = t_pairs // world_size
    rem = t_pairs % world_size
    n_loc = base + (1 if rank < rem else 0)
    p0 = rank * base + min(rank, rem)
    p1 = p0 + n_loc - 1
    return p0, p1, n_loc


def project_to_simplex(v):
    if v.numel() == 1:
        return torch.ones_like(v)
    if v.numel() == 2:
        x0 = torch.clamp(0.5 * (v[0] - v[1] + 1.0), min=0.0, max=1.0)
        return torch.stack((x0, 1.0 - x0))
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


def run_estep(Y_prev, Y_curr, A, B, c_hat, dt, eta_clip, lambda2, lr, pgd_iter):
    T_eff = c_hat.shape[0]
    log_dt = math.log(dt)
    c_prev = c_hat[0].clone()
    nll_sum = torch.zeros((), dtype=torch.float32, device=c_hat.device)

    for t in range(1, T_eff):
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
        nll_sum = nll_sum + (lam - y_c * (eta_c + log_dt)).sum()
        c_hat[t] = c_t
        c_prev = c_t

    return float((nll_sum / max(1, T_eff - 1)).item())


def run_estep_shard(Y_prev, Y_curr, A, B, c_hat,
                    dt, eta_clip, lambda2, lr, pgd_iter, t0, t1):
    if t1 < t0:
        return 0.0, 0
    log_dt = math.log(dt)
    c_prev = c_hat[t0 - 1].clone()
    nll_sum = torch.zeros((), dtype=torch.float32, device=c_hat.device)
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
        nll_sum = nll_sum + (lam - y_c * (eta_c + log_dt)).sum()
        c_hat[t] = c_t
        c_prev = c_t
        n_pairs += 1

    return float(nll_sum.item()), n_pairs


def offdiag_soft_threshold_(A, lr, lam):
    if lam <= 0:
        return
    with torch.no_grad():
        n = A.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=A.device)
        thresh = lr * lam
        v = A[mask]
        A[mask] = v.sign() * (v.abs() - thresh).clamp(min=0.0)


def enforce_spectral_radius_(A, rho_max, correction_strength=1.0):
    with torch.no_grad():
        rho = torch.linalg.eigvals(A).abs().max().item()
        alpha = min(1.0, max(0.0, float(correction_strength)))
        if rho > rho_max and alpha > 0.0:
            hard_scale = float(rho_max) / float(rho)
            scale = 1.0 - alpha * (1.0 - hard_scale)
            A.mul_(scale)
    return rho


def sync_AB_via_rank0_avg_then_broadcast_(mdl, rho_max, correction_strength=1.0):
    if not (dist.is_available() and dist.is_initialized()):
        enforce_spectral_radius_(mdl.A, rho_max, correction_strength)
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
            enforce_spectral_radius_(A_buf, rho_max, correction_strength)
        dist.broadcast(A_buf, src=0)
        dist.broadcast(B_buf, src=0)
        mdl.A.data.copy_(A_buf)
        mdl.B.data.copy_(B_buf)


def run_mstep_epoch(model, loader, optimizer, device, dt,
                    l1_wt, lambda3, rho_max, rho_every,
                    apply_prune, apply_rho, rho_correction_strength,
                    off_mask):
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
            sync_AB_via_rank0_avg_then_broadcast_(
                mdl, rho_max, rho_correction_strength
            )

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
        nz = int((mdl.A[off_mask] != 0).sum().item())
        rho = float(torch.linalg.eigvals(mdl.A).abs().max().item())
    return dict(loss=s_tot / nb, nll=s_nll / nb, l1=s_l1 / nb, rho=rho, nz=nz)


def viterbi_decode(c_hat, p_stay):
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


def onehot_from_prev_states(S, M):
    """Return one-hot rows aligned as S[t-1] for lag pair (Y[t-1], Y[t])."""
    S = np.asarray(S, dtype=np.int64)
    if S.ndim != 1 or S.shape[0] < 2:
        raise ValueError("S must be 1D with at least two time bins")
    eye = np.eye(M, dtype=np.float32)
    return eye[S[:-1]]


def make_pair_loader(yp_np, yc_np, c_pairs_np, args, ctx, shuffle=True):
    dataset = PairDataset(yp_np, yc_np, c_pairs_np)
    if ctx.is_dist:
        sampler = DistributedSampler(
            dataset, num_replicas=ctx.world_size, rank=ctx.rank,
            shuffle=shuffle, drop_last=False
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler,
            drop_last=False, pin_memory=True, num_workers=0
        )
    else:
        sampler = None
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=shuffle,
            drop_last=False, pin_memory=True, num_workers=0
        )
    return loader, sampler


def make_optimizer_and_scheduler(model, args, total_epochs, decay_from_epoch=0, decay_to_epoch=None):
    first_param = next(model.parameters())
    fused_ok = bool(first_param.is_cuda)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_mstep, fused=fused_ok)
    if decay_to_epoch is None:
        decay_to_epoch = int(total_epochs)
    decay_epochs = max(1, int(decay_to_epoch) - int(decay_from_epoch))
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=args.lr_end_factor,
        total_iters=decay_epochs, last_epoch=-1
    )
    scheduler._step_count = 1
    return optimizer, scheduler


def init_model(N, M, eta_clip, A_init, B_init, ctx):
    model = SwitchingPoissonGLM(N, M, eta_clip).to(ctx.device)
    if A_init is not None:
        with torch.no_grad():
            model.A.copy_(torch.tensor(A_init, dtype=torch.float32, device=ctx.device))
    if B_init is not None:
        with torch.no_grad():
            model.B.copy_(torch.tensor(B_init, dtype=torch.float32, device=ctx.device))
    if ctx.is_dist:
        model = DDP(model, device_ids=[ctx.local_rank])
    return model


def source_type_prune(A_hat):
    """Compute source-neuron signs and Dale-style pruned A using columns."""
    A_hat = np.asarray(A_hat, dtype=np.float32)
    N = A_hat.shape[0]
    A_thr = A_hat.copy()
    np.fill_diagonal(A_thr, 0.0)

    neuron_Sedge = A_thr.sum(axis=0)
    neuron_type = np.zeros((N,), dtype=np.int8)
    neuron_type[neuron_Sedge > 0.0] = 1
    neuron_type[neuron_Sedge < 0.0] = -1

    A_prune = A_hat.copy()
    diag_A = np.diag(A_hat).copy()
    exc_cols = neuron_type > 0
    inh_cols = neuron_type < 0
    A_prune[:, exc_cols] = np.where(A_prune[:, exc_cols] > 0, A_prune[:, exc_cols], 0.0)
    A_prune[:, inh_cols] = np.where(A_prune[:, inh_cols] < 0, A_prune[:, inh_cols], 0.0)
    np.fill_diagonal(A_prune, diag_A)
    return A_prune.astype(np.float32), neuron_type, neuron_Sedge.astype(np.float32)


def _history_dict():
    return dict(
        e_nll_em=[],
        m_loss_epoch=[],
        m_nll_epoch=[],
        m_l1_epoch=[],
        rho_epoch=[],
        rho_correction_strength_epoch=[],
        nz_edges_epoch=[],
        learning_rates=[],
    )


def _history_arrays(h):
    return dict(
        e_nll_em=np.asarray(h["e_nll_em"], dtype=np.float64),
        m_loss_epoch=np.asarray(h["m_loss_epoch"], dtype=np.float64),
        m_nll_epoch=np.asarray(h["m_nll_epoch"], dtype=np.float64),
        m_l1_epoch=np.asarray(h["m_l1_epoch"], dtype=np.float64),
        rho_epoch=np.asarray(h["rho_epoch"], dtype=np.float64),
        rho_correction_strength_epoch=np.asarray(h["rho_correction_strength_epoch"], dtype=np.float64),
        nz_edges_epoch=np.asarray(h["nz_edges_epoch"], dtype=np.int64),
        learning_rates=np.asarray(h["learning_rates"], dtype=np.float64),
    )


def run_full_fit(spikes, spikeMD, single_rates, args, ctx,
                 time_range_sec=None, time_range_bins=None,
                 fit_name=None, provenance_update=None):
    """Run full E/M training on an in-memory spike matrix on all ranks."""
    args = normalize_delay_args(args)
    spikes = np.asarray(spikes)
    if spikes.ndim != 2:
        raise ValueError("spikes must be shaped (T, N)")
    T_full, N = spikes.shape
    if T_full < 2:
        raise ValueError("Need at least two spike bins")
    M = int(args.num_states)
    T_pairs = T_full - 1
    dt = float(spikeMD["time_step_sec"])
    eta_clip = float(spikeMD["poisson_eta_clip"])
    p_stay = math.exp(-dt / float(args.decode_dwell_sec))

    yp_np = spikes[:-1].astype(np.float32)
    yc_np = spikes[1:].astype(np.float32)
    Yp_gpu = torch.tensor(yp_np, device=ctx.device)
    Yc_gpu = torch.tensor(yc_np, device=ctx.device)

    if is_rank0(ctx):
        c_init_np, S_init, init_state_md, freq_h1d = init_states_vs_time(spikes, dt, args)
        A_seed_np, init_A_md = init_A_from_spikes(spikes, args)
        B_seed_np, init_B_md = init_B_from_spikes(spikes, dt, args)
    else:
        c_init_np = S_init = freq_h1d = None
        A_seed_np = B_seed_np = None
        init_state_md = init_A_md = init_B_md = None

    c_init_np = broadcast_array(c_init_np, ctx)
    S_init = broadcast_array(S_init, ctx)
    freq_h1d = broadcast_array(freq_h1d, ctx)
    A_seed_np = broadcast_optional_array(A_seed_np, ctx)
    B_seed_np = broadcast_optional_array(B_seed_np, ctx)
    init_state_md = broadcast_object(init_state_md, ctx)
    init_A_md = broadcast_object(init_A_md, ctx)
    init_B_md = broadcast_object(init_B_md, ctx)

    c_hat_gpu = torch.tensor(c_init_np, dtype=torch.float32, device=ctx.device)
    c_pairs_np = onehot_from_prev_states(S_init, M)
    loader, sampler = make_pair_loader(yp_np, yc_np, c_pairs_np, args, ctx, shuffle=True)

    model = init_model(N, M, eta_clip, A_seed_np, B_seed_np, ctx)
    mdl = model.module if hasattr(model, "module") else model
    A_init_np = mdl.A.detach().cpu().numpy().copy()
    B_init_np = mdl.B.detach().cpu().numpy().copy()
    B_init_out_np = B_init_np[0] if B_init_np.ndim == 2 and B_init_np.shape[0] > 1 else B_init_np

    total_m_epochs = int(args.num_em_iters) * int(args.m_epochs)
    decay_start_m_epoch = int(args.delay_em_iter_4_lrDecay) * int(args.m_epochs)
    decay_target_m_epoch = int(args.target_em_iter_4_lrDecay) * int(args.m_epochs)
    optimizer, scheduler = make_optimizer_and_scheduler(
        model, args, total_m_epochs,
        decay_from_epoch=decay_start_m_epoch,
        decay_to_epoch=decay_target_m_epoch,
    )

    off_mask = ~torch.eye(N, dtype=torch.bool, device=ctx.device)
    l1_wt = torch.ones(N, N, device=ctx.device)
    l1_wt[torch.eye(N, dtype=torch.bool, device=ctx.device)] = 0.0
    h = _history_dict()
    m_epoch_global = 0
    t_start = time.time()
    reported_rho_activation = False
    reported_lr_activation = False

    for em in range(1, int(args.num_em_iters) + 1):
        apply_prune = em > int(args.delay_em_iter_4_Aprune)
        apply_rho = em > int(args.delay_em_iter_4_ArhoMax)
        em_iters_left = max(1, int(args.target_em_iter_4_ArhoMax) - em + 1)
        rho_correction_strength = 1.0 / float(em_iters_left) if apply_rho else 0.0
        apply_lr_decay = int(args.delay_em_iter_4_lrDecay) < em <= int(args.target_em_iter_4_lrDecay)

        if is_rank0(ctx) and args.verb > 0 and apply_rho and not reported_rho_activation:
            print(
                f"ArhoMax activated at EM iter {em} "
                f"(start_after={int(args.delay_em_iter_4_ArhoMax)}, "
                f"target_iter={int(args.target_em_iter_4_ArhoMax)}, "
                f"target_rho={float(args.rho_max):.6g})"
            )
            reported_rho_activation = True
        if is_rank0(ctx) and args.verb > 0 and apply_lr_decay and not reported_lr_activation:
            target_lr = float(args.lr_mstep) * float(args.lr_end_factor)
            print(
                f"lrDecay activated at EM iter {em} "
                f"(start_after={int(args.delay_em_iter_4_lrDecay)}, "
                f"target_iter={int(args.target_em_iter_4_lrDecay)}, "
                f"target_lr={target_lr:.6g})"
            )
            reported_lr_activation = True

        te0 = time.time()
        with torch.no_grad():
            if ctx.is_dist:
                p0, p1, n_loc = pair_range_for_rank(T_pairs, ctx.world_size, ctx.rank)
                t0 = p0 + 1
                t1 = p1 + 1
                nll_local, n_pair_local = run_estep_shard(
                    Yp_gpu, Yc_gpu, mdl.A.data, mdl.B.data, c_hat_gpu,
                    dt, eta_clip, args.lambda2, args.lr_estep, args.pgd_iter,
                    t0, t1
                )
                c_upd = torch.zeros_like(c_hat_gpu)
                c_msk = torch.zeros((T_full, 1), dtype=torch.float32, device=ctx.device)
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
                                  dtype=torch.float64, device=ctx.device)
                dist.all_reduce(ev, op=dist.ReduceOp.SUM)
                e_nll = float(ev[0].item() / max(1.0, ev[1].item()))
            else:
                e_nll = run_estep(
                    Yp_gpu, Yc_gpu, mdl.A.data, mdl.B.data, c_hat_gpu,
                    dt, eta_clip, args.lambda2, args.lr_estep, args.pgd_iter
                )
        te = time.time() - te0
        h["e_nll_em"].append(e_nll)

        c_hat_np_iter = c_hat_gpu.cpu().numpy()
        if ctx.is_dist:
            if is_rank0(ctx):
                s_iter_np = viterbi_decode(c_hat_np_iter, p_stay).astype(np.int64, copy=False)
                s_iter_t = torch.as_tensor(s_iter_np, dtype=torch.int64, device=ctx.device)
            else:
                s_iter_t = torch.empty((T_full,), dtype=torch.int64, device=ctx.device)
            dist.broadcast(s_iter_t, src=0)
            s_iter_np = s_iter_t.cpu().numpy()
        else:
            s_iter_np = viterbi_decode(c_hat_np_iter, p_stay).astype(np.int64, copy=False)
        np.copyto(c_pairs_np, onehot_from_prev_states(s_iter_np, M))

        tm0 = time.time()
        for _ in range(int(args.m_epochs)):
            m_epoch_global += 1
            if sampler is not None:
                sampler.set_epoch(m_epoch_global)
            met = run_mstep_epoch(
                model, loader, optimizer, ctx.device, dt,
                l1_wt, args.lambda3, args.rho_max,
                args.prescale_m_step_4_ArhoMax, apply_prune, apply_rho,
                rho_correction_strength, off_mask
            )
            if apply_lr_decay:
                scheduler.step()
            h["m_loss_epoch"].append(met["loss"])
            h["m_nll_epoch"].append(met["nll"])
            h["m_l1_epoch"].append(met["l1"])
            h["rho_epoch"].append(met["rho"])
            h["nz_edges_epoch"].append(met["nz"])
            h["learning_rates"].append(float(optimizer.param_groups[0]["lr"]))
            h["rho_correction_strength_epoch"].append(rho_correction_strength)
        tm = time.time() - tm0

        if is_rank0(ctx) and args.verb > 0:
            n_off = int(off_mask.sum().item())
            sp = 1.0 - met["nz"] / max(1, n_off)
            print(
                f"EM {em:3d}/{args.num_em_iters}  "
                f"E_nll={e_nll:.4e}({te:.1f}s)  "
                f"M_nll={met['nll']:.4e} l1={met['l1']:.4e} "
                f"rho={met['rho']:.4f} sp={sp:.3f} nz={met['nz']} "
                f"lr={h['learning_rates'][-1]:.2e} ({tm:.1f}s)  "
                f"tot={time.time() - t_start:.0f}s"
            )

    c_hat_np = c_hat_gpu.cpu().numpy()
    S_hat = viterbi_decode(c_hat_np, p_stay)
    S_hat_CL = (1.0 - c_hat_np.max(axis=1)).astype(np.float32)
    A_hat = mdl.A.detach().cpu().numpy().astype(np.float32)
    B_hat = mdl.B.detach().cpu().numpy().astype(np.float32)
    A_prune, neuron_type, neuron_Sedge = source_type_prune(A_hat)

    hist = _history_arrays(h)
    outD = {
        "A_init": A_init_np.astype(np.float32),
        "A_hat": A_hat,
        "A_prune": A_prune,
        "neuron_type": neuron_type,
        "neuron_Sedge": neuron_Sedge,
        "B_init": B_init_out_np.astype(np.float32),
        "B_hat": B_hat,
        "freq_h1d": np.asarray(freq_h1d, dtype=np.float32),
        "c_init": c_init_np.astype(np.float32),
        "c_hat": c_hat_np.astype(np.float32),
        "S_init": S_init.astype(np.int64),
        "S_hat": S_hat.astype(np.int64),
        "S_hat_CL": S_hat_CL,
        "single_rates": np.asarray(single_rates),
    }
    outD.update(hist)

    prov = dict(spikeMD.get("provenance", {}))
    if provenance_update:
        prov.update(provenance_update)
    if fit_name:
        prov["EMtrain_file"] = fit_name

    if time_range_sec is None:
        time_range_sec = [0.0, float((T_full - 1) * dt)]
    if time_range_bins is None:
        time_range_bins = [0, T_full - 1]

    outMD = dict(spikeMD)
    outMD["fit_type"] = "prismEM"
    outMD["train"] = {
        "num_em_iters": int(args.num_em_iters),
        "m_epochs": int(args.m_epochs),
        "total_m_epochs": int(total_m_epochs),
        "pgd_iter": int(args.pgd_iter),
        "lr_estep": float(args.lr_estep),
        "lr_mstep": float(args.lr_mstep),
        "lr_end_factor": float(args.lr_end_factor),
        "lambda2": float(args.lambda2),
        "lambda3": float(args.lambda3),
        "rho_max": float(args.rho_max),
        "rho_projection_mode": "soft_inverse_em_iters_left",
        "prescale_m_step_4_ArhoMax": int(args.prescale_m_step_4_ArhoMax),
        "delay_em_iter_4_ArhoMax": int(args.delay_em_iter_4_ArhoMax),
        "target_em_iter_4_ArhoMax": int(args.target_em_iter_4_ArhoMax),
        "rho_sync_mode": "broadcast",
        "delay_em_iter_4_lrDecay": int(args.delay_em_iter_4_lrDecay),
        "target_em_iter_4_lrDecay": int(args.target_em_iter_4_lrDecay),
        "delay_em_iter_4_Aprune": int(args.delay_em_iter_4_Aprune),
        "mstep_state_mode": "viterbi_onehot_prevbin",
        "batch_size": int(args.batch_size),
        "num_states": int(M),
        "num_neurons": int(N),
        "num_time_bins": int(T_full),
        "time_step_sec": float(dt),
        "eta_clip": float(eta_clip),
        "time_range_sec": [float(time_range_sec[0]), float(time_range_sec[1])],
        "time_range_bins": [int(time_range_bins[0]), int(time_range_bins[1])],
        "seed": int(args.seed),
        "init_A_Tmax": int(getattr(args, "init_A_Tmax", 50000)),
    }
    outMD["states_recovery_eval"] = {
        "decode": "viterbi",
        "decode_dwell_sec": float(args.decode_dwell_sec),
    }
    outMD["init_A"] = init_A_md
    outMD["init_state"] = init_state_md
    outMD["init_B"] = init_B_md
    outMD["provenance"] = prov
    return outD, outMD


def run_locked_mstep(Y_prev, Y_curr, S_lock, A_init, B_init, spikeMD, args, ctx):
    """Run locked M-step with state covariate S[t-1], no E-step/Viterbi."""
    Y_prev = np.asarray(Y_prev, dtype=np.float32)
    Y_curr = np.asarray(Y_curr, dtype=np.float32)
    S_lock = np.asarray(S_lock, dtype=np.int64)
    if Y_prev.shape != Y_curr.shape:
        raise ValueError("Y_prev and Y_curr must have matching shape")
    T_pairs, N = Y_prev.shape
    if S_lock.shape[0] != T_pairs + 1:
        raise ValueError("S_lock must have length T_pairs + 1")

    M = int(args.num_states)
    dt = float(spikeMD["time_step_sec"])
    eta_clip = float(spikeMD["poisson_eta_clip"])
    c_pairs_np = onehot_from_prev_states(S_lock, M)
    loader, sampler = make_pair_loader(Y_prev, Y_curr, c_pairs_np, args, ctx, shuffle=True)
    model = init_model(N, M, eta_clip, A_init, B_init, ctx)
    mdl = model.module if hasattr(model, "module") else model

    total_epochs = int(args.num_em_iters) * int(args.m_epochs)
    optimizer, scheduler = make_optimizer_and_scheduler(
        model, args, total_epochs, decay_from_epoch=0
    )
    off_mask = ~torch.eye(N, dtype=torch.bool, device=ctx.device)
    l1_wt = torch.ones(N, N, device=ctx.device)
    l1_wt[torch.eye(N, dtype=torch.bool, device=ctx.device)] = 0.0

    h = _history_dict()
    for epoch in range(1, total_epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        met = run_mstep_epoch(
            model, loader, optimizer, ctx.device, dt,
            l1_wt, args.lambda3, args.rho_max,
            args.prescale_m_step_4_ArhoMax,
            apply_prune=True, apply_rho=True, rho_correction_strength=1.0,
            off_mask=off_mask
        )
        scheduler.step()
        h["m_loss_epoch"].append(met["loss"])
        h["m_nll_epoch"].append(met["nll"])
        h["m_l1_epoch"].append(met["l1"])
        h["rho_epoch"].append(met["rho"])
        h["nz_edges_epoch"].append(met["nz"])
        h["learning_rates"].append(float(optimizer.param_groups[0]["lr"]))
        h["rho_correction_strength_epoch"].append(1.0)

    A_hat = mdl.A.detach().cpu().numpy().astype(np.float32)
    B_hat = mdl.B.detach().cpu().numpy().astype(np.float32)
    return A_hat, B_hat, _history_arrays(h)


def copy_args_with_init(args, init_A=None, init_B=None):
    d = vars(args).copy()
    if init_A is not None:
        d["init_A"] = init_A
    if init_B is not None:
        d["init_B"] = init_B
    return SimpleNamespace(**d)


def init_null_params(scrambled_spikes, spikeMD, args, ctx, null_A_init, null_B_init):
    """Fresh null A/B initialization on rank 0, broadcast to all ranks."""
    if is_rank0(ctx):
        if null_A_init == "scrambled_data":
            a_args = copy_args_with_init(args, init_A="data")
            A_init, A_md = init_A_from_spikes(scrambled_spikes, a_args)
        elif null_A_init == "rand":
            A_init, A_md = None, {"method": "rand"}
        else:
            raise ValueError(f"Unsupported null_A_init={null_A_init}")

        if null_B_init == "scrambled_data":
            b_args = copy_args_with_init(args, init_B="data")
            B_init, B_md = init_B_from_spikes(scrambled_spikes, float(spikeMD["time_step_sec"]), b_args)
        elif null_B_init == "rand":
            B_init, B_md = None, {"method": "rand"}
        else:
            raise ValueError(f"Unsupported null_B_init={null_B_init}")
    else:
        A_init = B_init = None
        A_md = B_md = None

    A_init = broadcast_optional_array(A_init, ctx)
    B_init = broadcast_optional_array(B_init, ctx)
    A_md = broadcast_object(A_md, ctx)
    B_md = broadcast_object(B_md, ctx)
    return A_init, B_init, A_md, B_md
