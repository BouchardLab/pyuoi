#!/usr/bin/env python3
"""
Block-coordinate fit for the BSSM-STD Bernoulli model.

The script fits

    p_t = sigmoid(b + W h_t)

from observed spikes.  The fast synaptic kernel is fixed from command-line
arguments, while the STD release trajectory h_t is recomputed from
observed spikes for each candidate (U, tau_rec).

Blocks:
  A. fit b and W with U, tau_rec fixed, using mini-batch Bernoulli NLL;
  B. jointly update U and tau_rec by bounded 2D grid-refinement search with b,W fixed.

Run with a PyTorch module on NERSC, for example:

  module load pytorch
  ./fit5_BSSM_STD_blocks.py --basePath $basePath --dataName daleN100_xxxxxx
"""

import argparse
import json
import math
import os
import secrets
import time
import zipfile
from pprint import pprint

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise SystemExit("PyTorch is required. On NERSC run: module load pytorch") from exc

from toolbox.Util_NumpyIO import write_data_npz


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit BSSM-STD Bernoulli model by block coordinate descent."
    )
    p.add_argument("--dataName", required=True)
    p.add_argument(
        "--basePath",
        default="/pscratch/sd/b/balewski/2026_causalNet_tmp/",
        help="Input root; reads <basePath>/truthDale/<dataName>.*.npz.",
    )
    p.add_argument(
        "--outPath",
        default=None,
        help="Output directory. Default: <basePath>/fitBssmStd.",
    )
    p.add_argument(
        "-T",
        "--time_range_sec",
        nargs=2,
        type=float,
        default=[0.0, 100.0],
        help="Requested post-burn fitting window [t0, t1] seconds. "
             "The input must contain this window plus burn-in.",
    )
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument(
        "--synaptic_tau",
        type=float,
        default=0.005,
        help="Fast synaptic tau_s in seconds for the fixed exponential kernel.",
    )
    p.add_argument(
        "--kernel_len_steps",
        type=int,
        default=25,
        help="Number of lag bins M retained in the fixed exponential kernel.",
    )
    p.add_argument(
        "--h_chunk_steps",
        type=int,
        default=2048,
        help="CUDA chunk length for the STD recurrence. Lower this only if an underflow abort occurs.",
    )
    p.add_argument("--u0", type=float, default=0.4, help="Initial STD utilization U.")
    p.add_argument("--tau_rec0", type=float, default=0.4, help="Initial tau_rec in seconds.")
    p.add_argument(
        "--init_samples",
        "--init_sample",
        dest="init_samples",
        type=int,
        required=True,
        help="Mandatory number of post-burn bins, >1000, used to initialize b from rates "
             "and W from lag-1 covariances.",
    )
    p.add_argument(
        "--u_bounds",
        type=float,
        nargs=2,
        required=True,
        help="Mandatory U scan range [U_lo U_hi] for the joint STD grid search.",
    )
    p.add_argument(
        "--tau_bounds",
        type=float,
        nargs=2,
        required=True,
        help="Mandatory tau_rec scan range [tau_lo tau_hi] in seconds for the joint STD grid search.",
    )
    p.add_argument(
        "--burn_sec",
        type=float,
        required=True,
        help="Mandatory positive burn-in seconds excluded from likelihood.",
    )
    p.add_argument("--num_outer", type=int, default=6)
    p.add_argument(
        "--freeze_std_outer",
        type=int,
        default=2,
        help="Number of initial outer iterations that update only b,W.",
    )
    p.add_argument("--blockA_epochs", type=int, default=40,
                   help="Block A epochs for frozen-STD outer iterations.")
    p.add_argument("--blockA_epochs_live", type=int, default=-1,
                   help="Block A epochs once STD params are live (outer >= freeze_std_outer). "
                        "If <0, reuse --blockA_epochs.")
    p.add_argument(
        "--progress_every",
        type=int,
        default=5,
        help="Print Block A progress every this many epochs when --verb > 0; use 0 to disable.",
    )
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--lr_w", type=float, default=1e-2)
    p.add_argument("--lambda_l1", type=float, default=1e-4)
    p.add_argument(
        "--rho_max",
        type=float,
        default=0.99,
        help="If >0, rescale W after each Block A epoch when spectral radius exceeds this value.",
    )
    p.add_argument(
        "--delay_epoch_4_rhoMax",
        type=int,
        default=0,
        help="Number of initial Block A epochs per outer iteration to skip before applying --rho_max.",
    )
    p.add_argument("--eta_clip", type=float, default=10.0)
    p.add_argument(
        "--zero_diag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force W diagonal to zero.",
    )
    p.add_argument("--u_grid_points", type=int, default=9)
    p.add_argument("--tau_grid_points", type=int, default=9)
    p.add_argument("--grid_refine", type=int, default=1)
    p.add_argument(
        "--grid_shrink",
        type=float,
        default=0.35,
        help="Refinement half-width as a fraction of the current U/tau_rec search intervals.",
    )
    p.add_argument(
        "--weight_threshold",
        type=float,
        default=0.03,
        help="Absolute off-diagonal weight threshold used only for reporting nz_weight_outer.",
    )
    p.add_argument("--fitName", type=str, default=None)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("-v", "--verb", type=int, default=1)
    return p.parse_args()


def require_keys(obj, keys, obj_name):
    if obj is None:
        abort_fit("missing required object %s" % obj_name)
    for key in keys:
        if key not in obj:
            abort_fit("missing required key %s[%r]" % (obj_name, key))


def _extract_json_object(text):
    start = text.find('{"')
    if start < 0:
        abort_fit("could not locate JSON object in meta.JSON.npy")
    depth = 0
    in_str = False
    escape = False
    for pos in range(start, len(text)):
        ch = text[pos]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    abort_fit("unterminated JSON object in meta.JSON.npy")


def read_meta_json(npz_path):
    with zipfile.ZipFile(npz_path, "r") as zf:
        if "meta.JSON.npy" not in zf.namelist():
            abort_fit("missing meta.JSON.npy in %s" % npz_path)
        raw = zf.read("meta.JSON.npy")
    text = raw.decode("utf-8", errors="ignore")
    return json.loads(_extract_json_object(text))


def read_npz_records(npz_path, keys, verb=0):
    """Read selected non-object arrays plus metadata without loading all records."""
    out = {}
    with np.load(npz_path, allow_pickle=False) as data:
        for key in keys:
            if key not in data.files:
                abort_fit("missing required key %s in %s" % (key, npz_path))
            out[key] = data[key]
    meta = read_meta_json(npz_path)
    if verb > 0:
        print("read selected records from:", npz_path)
        for key, val in out.items():
            print("  %s %s %s" % (key, val.shape, val.dtype))
    return out, meta


def choose_device():
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA GPU is required but torch.cuda.is_available() is false")
    if torch.cuda.device_count() < 1:
        raise SystemExit("ERROR: no CUDA GPU devices are visible")
    return torch.device("cuda:0")


def abort_fit(msg):
    raise SystemExit("ERROR: %s" % msg)


def time_window_with_burn(spikes, dt, time_range_sec, burn_bins):
    t0_sec, t1_sec = float(time_range_sec[0]), float(time_range_sec[1])
    if not t0_sec < t1_sec:
        abort_fit("time_range_sec[0] must be < time_range_sec[1]")
    if burn_bins < 0:
        abort_fit("burn_bins must be nonnegative")

    requested_start_bin = int(math.floor(t0_sec / dt))
    requested_end_bin = int(math.floor(t1_sec / dt))
    if requested_start_bin < 0:
        abort_fit("requested start_bin=%d is negative" % requested_start_bin)

    requested_bins = requested_end_bin - requested_start_bin + 1
    raw_start_bin = requested_start_bin
    raw_end_bin = requested_end_bin + int(burn_bins)
    input_bins = int(spikes.shape[0])
    if raw_end_bin >= input_bins:
        abort_fit(
            "not enough input time bins for requested window plus burn-in: "
            "requested_bins=%d burn_bins=%d requires raw bins [%d,%d], "
            "but input has bins [0,%d]"
            % (
                requested_bins,
                int(burn_bins),
                raw_start_bin,
                raw_end_bin,
                input_bins - 1,
            )
        )
    return (
        raw_start_bin,
        raw_end_bin,
        requested_start_bin,
        requested_end_bin,
        requested_bins,
        spikes[raw_start_bin : raw_end_bin + 1],
    )


def logit_np(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


def init_b_w_from_postburn_spikes(spikes_w, burn_bins, init_samples, zero_diag=True):
    init_samples = int(init_samples)
    if init_samples < 2:
        abort_fit("--init_samples must be at least 2")
    postburn = np.asarray(spikes_w[burn_bins:], dtype=np.float64)
    init_bins = int(postburn.shape[0])
    if init_bins < init_samples:
        abort_fit(
            "not enough post-burn bins for initialization: "
            "available=%d required --init_samples=%d"
            % (init_bins, init_samples)
        )

    init_spikes = postburn[:init_samples]
    p0 = np.mean(init_spikes, axis=0)
    b0 = logit_np(p0)

    pre = init_spikes[:-1]
    post = init_spikes[1:]
    pre_centered = pre - np.mean(pre, axis=0, keepdims=True)
    post_centered = post - np.mean(post, axis=0, keepdims=True)
    W0 = (post_centered.T @ pre_centered) / float(pre.shape[0])
    if zero_diag:
        np.fill_diagonal(W0, 0.0)
    return b0, W0, init_samples


def infer_alpha(kappa):
    kappa = np.asarray(kappa, dtype=np.float64).reshape(-1)
    if kappa.size < 1:
        abort_fit("kernel must contain at least one entry")
    if kappa.size == 1:
        return 0.0
    if not kappa[0] > 0:
        abort_fit("kernel first entry must be positive")
    alpha = float(kappa[1] / kappa[0])
    if not (0.0 <= alpha < 1.0):
        abort_fit("kernel ratio alpha must satisfy 0 <= alpha < 1")
    return alpha


def build_exponential_kernel(dt, synaptic_tau, mem_lag_steps):
    synaptic_tau = float(synaptic_tau)
    mem_lag_steps = int(mem_lag_steps)
    if not synaptic_tau > 0.0:
        abort_fit("--synaptic_tau must be positive")
    if mem_lag_steps < 1:
        abort_fit("--kernel_len_steps must be at least 1")
    alpha = float(math.exp(-float(dt) / synaptic_tau))
    ell = np.arange(mem_lag_steps, dtype=np.float64)
    return (1.0 - alpha) * np.power(alpha, ell)


def project_w_(W, zero_diag=True):
    with torch.no_grad():
        if zero_diag:
            W.diagonal().zero_()


def enforce_spectral_radius_(W, rho_max):
    if rho_max is None or float(rho_max) <= 0.0:
        return
    with torch.no_grad():
        rho = torch.linalg.eigvals(W).abs().max()
        if torch.isfinite(rho) and rho > float(rho_max):
            W.mul_(float(rho_max) / float(rho.item()))


@torch.no_grad()
def spectral_radius_torch(W):
    if W.device.type != "cuda":
        abort_fit("spectral radius diagnostic requires CUDA tensor")
    if W.numel() == 0:
        return float("nan")
    vals = torch.linalg.eigvals(W)
    return float(torch.max(torch.abs(vals)).item())


_H_FORWARD_SEC_PER_BIN = None


def _record_h_timing(elapsed, n_bins):
    global _H_FORWARD_SEC_PER_BIN
    sec_per_bin = float(elapsed) / float(max(1, int(n_bins)))
    if _H_FORWARD_SEC_PER_BIN is None:
        _H_FORWARD_SEC_PER_BIN = sec_per_bin
    else:
        _H_FORWARD_SEC_PER_BIN = 0.5 * float(_H_FORWARD_SEC_PER_BIN) + 0.5 * sec_per_bin


def std_h_forward_vectorized_cuda(
    spikes,
    U,
    tau_rec,
    dt,
    alpha,
    mem_lag_steps,
    burn_bins,
    label="H",
    verb=1,
    t0=None,
    chunk_steps=2048,
):
    """GPU-friendly H computation. Abort if the CUDA path cannot be used safely."""
    if spikes.device.type != "cuda":
        abort_fit("H forward requires CUDA tensors; no CPU fallback is available")

    T = int(spikes.shape[0])
    N = int(spikes.shape[1])
    dtype = spikes.dtype
    work_dtype = torch.float64 if dtype == torch.float32 else dtype
    device = spikes.device
    rec = float(math.exp(-float(dt) / float(tau_rec)))
    beta = 1.0 - rec
    tiny = 1e-250 if work_dtype == torch.float64 else 1e-35

    x0 = torch.ones((N,), dtype=work_dtype, device=device)
    u_all = torch.empty((T, N), dtype=dtype, device=device)
    last_print = time.time()
    t_start = time.time() if t0 is None else float(t0)

    for start in range(0, T, int(chunk_steps)):
        end = min(T, start + int(chunk_steps))
        s = spikes[start:end].to(dtype=work_dtype)
        a = rec * (1.0 - float(U) * s)
        p_after = torch.cumprod(a, dim=0)
        min_abs = float(torch.min(torch.abs(p_after)).item())
        if (not math.isfinite(min_abs)) or min_abs < tiny:
            abort_fit(
                "CUDA H recurrence underflow in %s at bins [%d,%d) with --h_chunk_steps=%d; "
                "rerun with a smaller --h_chunk_steps"
                % (label, start, end, int(chunk_steps))
            )

        inv_cumsum = torch.cumsum(torch.reciprocal(p_after), dim=0)
        p_before = torch.empty_like(p_after)
        sum_before = torch.empty_like(inv_cumsum)
        p_before[0].fill_(1.0)
        sum_before[0].zero_()
        if end - start > 1:
            p_before[1:] = p_after[:-1]
            sum_before[1:] = inv_cumsum[:-1]

        x_vals = p_before * (x0.unsqueeze(0) + beta * sum_before)
        x_vals = torch.clamp(x_vals, 0.0, 1.0)
        u_all[start:end] = (float(U) * x_vals * s).to(dtype=dtype)
        x0 = p_after[-1] * (x0 + beta * inv_cumsum[-1])
        x0 = torch.clamp(x0, 0.0, 1.0)

        now = time.time()
        if verb > 0 and now - last_print >= 5.0:
            done_bins = end
            frac = min(0.999, float(done_bins) / float(max(1, T)))
            elapsed = now - t_start
            eta = elapsed * (1.0 / max(1e-9, frac) - 1.0)
            print(
                "    %s release scan: %d/%d bins %.0f%% elapsed=%.0fs eta=%.0fs"
                % (label, done_bins, T, 100.0 * frac, elapsed, eta),
                flush=True,
            )
            last_print = now

    if verb > 0:
        print("    %s synaptic convolution on GPU..." % label, flush=True)
    ell = torch.arange(int(mem_lag_steps), dtype=dtype, device=device)
    kappa = (1.0 - float(alpha)) * torch.pow(
        torch.tensor(float(alpha), dtype=dtype, device=device),
        ell,
    )
    weight = torch.flip(kappa, dims=[0]).view(1, 1, int(mem_lag_steps)).repeat(N, 1, 1)
    u_ch = u_all.transpose(0, 1).unsqueeze(0).contiguous()
    u_pad = F.pad(u_ch, (int(mem_lag_steps), 0))
    h_ch = F.conv1d(u_pad, weight, groups=N)
    H = h_ch[0, :, :T].transpose(0, 1).contiguous()
    H = torch.clamp(H[int(burn_bins):], min=0.0)
    return H


def make_h(
    spikes_t,
    U,
    tau_rec,
    dt,
    alpha,
    mem_lag_steps,
    burn_bins,
    h_chunk_steps,
    label="H",
    verb=1,
):
    global _H_FORWARD_SEC_PER_BIN
    if not (0.0 < float(U) < 0.999):
        abort_fit("U must satisfy 0 < U < 0.999")
    if not tau_rec > 0.0:
        abort_fit("tau_rec must be positive")

    n_bins = int(spikes_t.shape[0])
    t0 = time.time()
    H = std_h_forward_vectorized_cuda(
        spikes_t,
        U,
        tau_rec,
        dt,
        alpha,
        mem_lag_steps,
        burn_bins,
        label=label,
        verb=verb,
        t0=t0,
        chunk_steps=int(h_chunk_steps),
    )
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    _record_h_timing(elapsed, n_bins)

    if verb > 0:
        print("    %s forward pass done in %.1fs  shape=%s mean=%.6g max=%.6g"
              % (label, elapsed, tuple(H.shape),
                 float(torch.mean(H).item()), float(torch.max(H).item())), flush=True)
    return H


@torch.no_grad()
def bernoulli_nll_mean(H, Y_eval, W, b, eta_clip, batch_size):
    n = H.shape[0]
    N = H.shape[1]
    total = 0.0
    for i0 in range(0, n, batch_size):
        i1 = min(n, i0 + batch_size)
        eta = H[i0:i1] @ W.t() + b
        eta = torch.clamp(eta, -float(eta_clip), float(eta_clip))
        loss = F.binary_cross_entropy_with_logits(
            eta, Y_eval[i0:i1].to(dtype=H.dtype), reduction="sum"
        )
        total += float(loss.item())
    return total / float(max(1, n * N))


def progress_line(args, t_start, msg):
    if args.verb > 0:
        print("  %s elapsed=%.1fs" % (msg, time.time() - t_start), flush=True)


def train_block_a(H, Y_eval, W, b, args, offdiag_mask, generator, outer, t_start, epochs=None):
    params = [W, b]
    optimizer = torch.optim.Adam(params, lr=float(args.lr_w))
    n = int(H.shape[0])
    N = int(H.shape[1])
    bce_losses = []
    l1_losses = []
    total_losses = []
    progress_every = int(args.progress_every)
    rho_delay = int(args.delay_epoch_4_rhoMax)
    if epochs is None:
        epochs = int(args.blockA_epochs)

    for epoch in range(epochs):
        perm = torch.randperm(n, device=H.device, generator=generator)
        epoch_bce_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_total_loss = 0.0
        epoch_count = 0
        for i0 in range(0, n, int(args.batch_size)):
            idx = perm[i0 : min(n, i0 + int(args.batch_size))]
            Hb = H.index_select(0, idx)
            Yb = Y_eval.index_select(0, idx).to(dtype=H.dtype)

            optimizer.zero_grad(set_to_none=True)
            eta = Hb @ W.t() + b
            eta = torch.clamp(eta, -float(args.eta_clip), float(args.eta_clip))
            base_loss = F.binary_cross_entropy_with_logits(eta, Yb, reduction="mean")
            l1_loss = torch.zeros((), dtype=base_loss.dtype, device=base_loss.device)
            if args.lambda_l1 > 0:
                l1_loss = float(args.lambda_l1) * torch.mean(torch.abs(W[offdiag_mask]))
            loss = base_loss + l1_loss
            loss.backward()
            optimizer.step()
            project_w_(W, zero_diag=bool(args.zero_diag))

            nb = int(idx.numel()) * N
            epoch_bce_loss += float(base_loss.item()) * nb
            epoch_l1_loss += float(l1_loss.item()) * nb
            epoch_total_loss += float(loss.item()) * nb
            epoch_count += nb

        epoch_bce = epoch_bce_loss / float(max(1, epoch_count))
        epoch_l1 = epoch_l1_loss / float(max(1, epoch_count))
        epoch_total = epoch_total_loss / float(max(1, epoch_count))
        bce_losses.append(epoch_bce)
        l1_losses.append(epoch_l1)
        total_losses.append(epoch_total)
        if epoch + 1 > rho_delay:
            enforce_spectral_radius_(W, float(args.rho_max))
        if (
            args.verb > 0
            and progress_every > 0
            and (
                epoch == 0
                or (epoch + 1) % progress_every == 0
                or epoch + 1 == epochs
            )
        ):
            print(
                "  progress outer=%d/%d epoch=%d/%d blockA_bce=%.6f blockA_l1=%.3g blockA_total=%.6f elapsed=%.1fs"
                % (
                    outer + 1,
                    int(args.num_outer),
                    epoch + 1,
                    epochs,
                    epoch_bce,
                    epoch_l1,
                    epoch_total,
                    time.time() - t_start,
                ),
                flush=True,
            )

    return (
        np.asarray(bce_losses, dtype=np.float64),
        np.asarray(l1_losses, dtype=np.float64),
        np.asarray(total_losses, dtype=np.float64),
    )


def joint_grid_search(
    current_u,
    current_tau,
    u_bounds,
    tau_bounds,
    u_grid_points,
    tau_grid_points,
    refine_steps,
    shrink,
    objective_fn,
    t_start=None,
    outer=None,
    num_outer=None,
    verb=1,
    eval_progress_every=5,
):
    u_lo = float(u_bounds[0])
    u_hi = float(u_bounds[1])
    tau_lo = float(tau_bounds[0])
    tau_hi = float(tau_bounds[1])
    if not u_lo < u_hi:
        abort_fit("joint grid U bounds must satisfy U_lo < U_hi")
    if not tau_lo < tau_hi:
        abort_fit("joint grid tau bounds must satisfy tau_lo < tau_hi")
    best_u = float(np.clip(current_u, u_lo, u_hi))
    best_tau = float(np.clip(current_tau, tau_lo, tau_hi))
    best_loss = float("inf")
    u_all = []
    tau_all = []
    losses_all = []

    cur_u_lo, cur_u_hi = u_lo, u_hi
    cur_tau_lo, cur_tau_hi = tau_lo, tau_hi
    for ref in range(int(refine_steps) + 1):
        u_grid = np.linspace(cur_u_lo, cur_u_hi, int(u_grid_points), dtype=np.float64)
        tau_grid = np.linspace(cur_tau_lo, cur_tau_hi, int(tau_grid_points), dtype=np.float64)
        if best_u > cur_u_lo and best_u < cur_u_hi:
            u_grid = np.unique(np.sort(np.append(u_grid, best_u)))
        if best_tau > cur_tau_lo and best_tau < cur_tau_hi:
            tau_grid = np.unique(np.sort(np.append(tau_grid, best_tau)))

        losses = []
        pairs = []
        n_eval = int(u_grid.size * tau_grid.size)
        i_eval = 0
        for u_val in u_grid:
            for tau_val in tau_grid:
                i_eval += 1
                loss = float(objective_fn(float(u_val), float(tau_val)))
                u_all.append(float(u_val))
                tau_all.append(float(tau_val))
                losses_all.append(loss)
                losses.append(loss)
                pairs.append((float(u_val), float(tau_val)))
                print_eval = (
                    int(eval_progress_every) > 0
                    and (i_eval % int(eval_progress_every) == 0 or i_eval == n_eval)
                )
                if verb > 0 and print_eval:
                    prefix = (
                        "    search U_tau refine=%d eval=%d/%d U=%.6g tau_rec=%.6g nll=%.6f"
                        % (ref, i_eval, n_eval, float(u_val), float(tau_val), loss)
                    )
                    if outer is not None and num_outer is not None:
                        prefix = "    outer=%d/%d %s" % (int(outer) + 1, int(num_outer), prefix.strip())
                    if t_start is not None:
                        prefix += " elapsed=%.1fs" % (time.time() - t_start)
                    print(prefix, flush=True)
        losses = np.asarray(losses, dtype=np.float64)
        idx = int(np.argmin(losses))
        best_u, best_tau = pairs[idx]
        best_loss = float(losses[idx])

        u_width = (cur_u_hi - cur_u_lo) * float(shrink)
        tau_width = (cur_tau_hi - cur_tau_lo) * float(shrink)
        cur_u_lo = max(u_lo, best_u - u_width)
        cur_u_hi = min(u_hi, best_u + u_width)
        cur_tau_lo = max(tau_lo, best_tau - tau_width)
        cur_tau_hi = min(tau_hi, best_tau + tau_width)
        if verb > 1:
            print(
                "    U_tau refine=%d best_U=%.6g best_tau=%.6g nll=%.6f"
                % (ref, best_u, best_tau, best_loss),
                flush=True,
            )

    return (
        best_u,
        best_tau,
        best_loss,
        np.asarray(u_all, dtype=np.float64),
        np.asarray(tau_all, dtype=np.float64),
        np.asarray(losses_all, dtype=np.float64),
    )


def main():
    args = parse_args()
    t_start = time.time()
    device = choose_device()
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    np.random.seed(int(args.seed))

    if not (0.0 < args.u0 < 0.999):
        abort_fit("--u0 must satisfy 0 < U < 0.999")
    if not args.tau_rec0 > 0.0:
        abort_fit("--tau_rec0 must be positive")
    if not args.synaptic_tau > 0.0:
        abort_fit("--synaptic_tau must be positive")
    if not args.kernel_len_steps >= 1:
        abort_fit("--kernel_len_steps must be at least 1")
    if args.h_chunk_steps < 1:
        abort_fit("--h_chunk_steps must be at least 1")
    if args.burn_sec <= 0.0:
        abort_fit("--burn_sec must be positive")
    if args.init_samples <= 1000:
        abort_fit("--init_samples must be above 1000")
    if not (args.u_bounds[0] > 0.0 and args.u_bounds[0] < args.u_bounds[1] < 0.999):
        abort_fit("--u_bounds must satisfy 0 < U_lo < U_hi < 0.999")
    if not (args.tau_bounds[0] > 0.0 and args.tau_bounds[0] < args.tau_bounds[1]):
        abort_fit("--tau_bounds must satisfy 0 < tau_lo < tau_hi")
    if args.num_outer < 1:
        abort_fit("--num_outer must be at least 1")
    if args.blockA_epochs < 1:
        abort_fit("--blockA_epochs must be at least 1")
    if args.progress_every < 0:
        abort_fit("--progress_every must be nonnegative")
    if args.delay_epoch_4_rhoMax < 0:
        abort_fit("--delay_epoch_4_rhoMax must be nonnegative")
    if args.batch_size < 1:
        abort_fit("--batch_size must be at least 1")
    if args.u_grid_points < 3 or args.tau_grid_points < 3:
        abort_fit("--u_grid_points and --tau_grid_points must each be at least 3")
    if args.grid_refine < 0:
        abort_fit("--grid_refine must be nonnegative")
    if not (0.0 < args.grid_shrink <= 1.0):
        abort_fit("--grid_shrink must satisfy 0 < grid_shrink <= 1")

    inp_path = os.path.join(args.basePath, "truthDale")
    out_path = args.outPath if args.outPath is not None else os.path.join(args.basePath, "fitBssmStd")
    if not os.path.exists(inp_path):
        abort_fit("missing input path: %s" % inp_path)

    spikes_ff = os.path.join(inp_path, args.dataName + ".spikes.npz")
    if not os.path.exists(spikes_ff):
        abort_fit("missing spikes file: %s" % spikes_ff)

    spike_d, spike_md = read_npz_records(spikes_ff, ["spikes"], verb=args.verb > 1)
    require_keys(spike_d, ["spikes"], "spike_d")
    require_keys(spike_md, ["time_step_sec"], "spike_md")

    spikes_np = np.asarray(spike_d["spikes"])
    if spikes_np.ndim != 2:
        abort_fit("spikes array must be 2D")
    if np.max(spikes_np) > 1:
        abort_fit("BSSM-STD fit expects binary Bernoulli spikes")
    dt = float(spike_md["time_step_sec"])
    burn_sec = float(args.burn_sec)
    burn_bins = int(math.ceil(burn_sec / dt))
    (
        start_bin,
        end_bin,
        requested_start_bin,
        requested_end_bin,
        requested_bins,
        spikes_w,
    ) = time_window_with_burn(spikes_np, dt, args.time_range_sec, burn_bins)
    T_w, N = spikes_w.shape
    if T_w != burn_bins + requested_bins:
        abort_fit(
            "internal time-window mismatch: loaded=%d burn_bins=%d requested_bins=%d"
            % (T_w, burn_bins, requested_bins)
        )
    if requested_bins < int(args.init_samples):
        abort_fit(
            "not enough requested post-burn bins for initialization: "
            "requested_bins=%d required --init_samples=%d"
            % (requested_bins, int(args.init_samples))
        )
    os.makedirs(out_path, exist_ok=True)

    kappa = build_exponential_kernel(
        dt=dt,
        synaptic_tau=float(args.synaptic_tau),
        mem_lag_steps=int(args.kernel_len_steps),
    )
    mem_lag_steps = int(kappa.size)
    alpha = infer_alpha(kappa)
    if mem_lag_steps < 1:
        abort_fit("kernel length must be at least 1")

    spikes_t = torch.as_tensor(spikes_w.astype(np.float32), dtype=dtype, device=device)
    Y_eval = spikes_t[burn_bins:]
    b0, W0, init_bins = init_b_w_from_postburn_spikes(
        spikes_w,
        burn_bins,
        args.init_samples,
        zero_diag=bool(args.zero_diag),
    )
    np_dtype = np.float32 if dtype == torch.float32 else np.float64
    b0 = b0.astype(np_dtype)
    W0 = W0.astype(np_dtype)
    offdiag_np = ~np.eye(N, dtype=bool)
    W0_abs_mean = float(np.mean(np.abs(W0[offdiag_np]))) if np.any(offdiag_np) else 0.0

    W = torch.tensor(W0, dtype=dtype, device=device, requires_grad=True)
    b = torch.tensor(b0, dtype=dtype, device=device, requires_grad=True)

    offdiag_mask = ~torch.eye(N, dtype=torch.bool, device=device)
    project_w_(W, zero_diag=bool(args.zero_diag))

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))

    U = float(args.u0)
    tau_rec = float(args.tau_rec0)

    outer_rows = []
    blockA_outer = []
    blockA_epoch = []
    blockA_bce_loss = []
    blockA_l1_loss = []
    blockA_loss = []
    joint_search_outer = []
    joint_search_U = []
    joint_search_tau_rec = []
    joint_search_loss = []

    if args.verb > 0:
        print("\nfit5_BSSM_STD_blocks.py", flush=True)
        print("  dataName=%s" % args.dataName, flush=True)
        print("  device=%s dtype=%s torch=%s" % (device, args.dtype, torch.__version__), flush=True)
        print("  gpu=%s" % torch.cuda.get_device_name(device), flush=True)
        print(
            "  bins raw=[%d,%d] requested=[%d,%d] T_raw=%d N=%d dt=%.6g burn_bins=%d eval_bins=%d"
            % (
                start_bin,
                end_bin,
                requested_start_bin,
                requested_end_bin,
                T_w,
                N,
                dt,
                burn_bins,
                int(Y_eval.shape[0]),
            ),
            flush=True,
        )
        print(
            "  init b,W from first %d post-burn bins: b_mean=%.6g W_lag1_cov_abs_mean=%.6g"
            % (init_bins, float(np.mean(b0)), W0_abs_mean),
            flush=True,
        )
        print(
            "  kernel: tau_s=%.6g M=%d alpha=%.6g  init U=%.6g tau_rec=%.6g"
            % (float(args.synaptic_tau), mem_lag_steps, alpha, U, tau_rec),
            flush=True,
        )
        print("  CUDA H recurrence chunk_steps=%d" % int(args.h_chunk_steps), flush=True)

    for outer in range(int(args.num_outer)):
        if args.verb > 0:
            print(
                "\nOuter %d/%d  U=%.6f tau_rec=%.6f"
                % (outer + 1, args.num_outer, U, tau_rec),
                flush=True,
            )

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        progress_line(args, t_start, "start computing H for outer=%d/%d" % (outer + 1, int(args.num_outer)))
        H = make_h(
            spikes_t,
            U,
            tau_rec,
            dt,
            alpha,
            mem_lag_steps,
            burn_bins,
            args.h_chunk_steps,
            label="H outer=%d/%d" % (outer + 1, int(args.num_outer)),
            verb=args.verb,
        )

        live_std = outer >= int(args.freeze_std_outer)
        epochs_this_outer = (
            int(args.blockA_epochs_live) if (live_std and args.blockA_epochs_live >= 0)
            else int(args.blockA_epochs)
        )
        progress_line(
            args,
            t_start,
            "start Block A outer=%d/%d epochs=%d batch_size=%d"
            % (outer + 1, int(args.num_outer), epochs_this_outer, int(args.batch_size)),
        )
        bce_losses, l1_losses, losses = train_block_a(
            H, Y_eval, W, b, args, offdiag_mask, gen, outer, t_start,
            epochs=epochs_this_outer,
        )
        progress_line(args, t_start, "finished Block A outer=%d/%d" % (outer + 1, int(args.num_outer)))
        for ep, val in enumerate(losses):
            blockA_outer.append(outer)
            blockA_epoch.append(ep)
            blockA_bce_loss.append(float(bce_losses[ep]))
            blockA_l1_loss.append(float(l1_losses[ep]))
            blockA_loss.append(float(val))

        progress_line(args, t_start, "start evaluating nll_after_blockA outer=%d/%d" % (outer + 1, int(args.num_outer)))
        nll_after_A = bernoulli_nll_mean(H, Y_eval, W, b, args.eta_clip, args.batch_size)
        progress_line(
            args,
            t_start,
            "finished nll_after_blockA outer=%d/%d nll_A=%.9f"
            % (outer + 1, int(args.num_outer), nll_after_A),
        )

        if outer >= int(args.freeze_std_outer):
            progress_line(
                args,
                t_start,
                "start joint U/tau_rec grid search outer=%d/%d current_U=%.6f current_tau=%.6f"
                % (outer + 1, int(args.num_outer), U, tau_rec),
            )

            def obj_std(u_val, tau_val):
                H_std = make_h(
                    spikes_t,
                    float(u_val),
                    float(tau_val),
                    dt,
                    alpha,
                    mem_lag_steps,
                    burn_bins,
                    args.h_chunk_steps,
                    verb=0,
                )
                loss_std = bernoulli_nll_mean(H_std, Y_eval, W, b, args.eta_clip, args.batch_size)
                del H_std
                return loss_std

            U, tau_rec, nll_std, vals_u, vals_tau, losses_std = joint_grid_search(
                U,
                tau_rec,
                args.u_bounds,
                args.tau_bounds,
                args.u_grid_points,
                args.tau_grid_points,
                args.grid_refine,
                args.grid_shrink,
                obj_std,
                t_start=t_start,
                outer=outer,
                num_outer=args.num_outer,
                verb=args.verb,
            )
            progress_line(
                args,
                t_start,
                "finished joint U/tau_rec grid search outer=%d/%d best_U=%.6f best_tau=%.6f nll=%.9f"
                % (outer + 1, int(args.num_outer), U, tau_rec, nll_std),
            )
            joint_search_outer.extend([outer] * vals_u.size)
            joint_search_U.extend(vals_u.tolist())
            joint_search_tau_rec.extend(vals_tau.tolist())
            joint_search_loss.extend(losses_std.tolist())
            nll_outer = float(nll_std)
        else:
            progress_line(
                args,
                t_start,
                "skip joint U/tau_rec search outer=%d/%d because freeze_std_outer=%d"
                % (outer + 1, int(args.num_outer), int(args.freeze_std_outer)),
            )
            nll_std = np.nan
            nll_outer = float(nll_after_A)

        rho_w = spectral_radius_torch(W.detach())
        nz = int(torch.sum(torch.abs(W.detach()[offdiag_mask]) >= float(args.weight_threshold)).item())
        outer_rows.append(
            [
                outer,
                U,
                tau_rec,
                nll_after_A,
                nll_std,
                nll_outer,
                rho_w,
                nz,
            ]
        )

        if args.verb > 0:
            print(
                "  nll_A=%.9f nll=%.9f rhoW=%.4f nz=%d elapsed=%.1fs"
                % (nll_after_A, nll_outer, rho_w, nz, time.time() - t_start),
                flush=True,
            )
            print("  now U=%.6f   tau_rec=%.6f" % (U, tau_rec), flush=True)

        del H
        torch.cuda.empty_cache()

    outer_arr = np.asarray(outer_rows, dtype=np.float64)
    elapsed = time.time() - t_start

    if args.fitName is None:
        out_stem = "%s-bssmStdFit-%s" % (args.dataName, secrets.token_hex(3))
    else:
        out_stem = args.fitName
    out_ff = os.path.join(out_path, out_stem + ".fitBSSMSTD.npz")

    W_fit = W.detach().cpu().numpy().astype(np.float32)
    b_fit = b.detach().cpu().numpy().astype(np.float32)
    out_d = {
        "W_fit": W_fit,
        "b_fit": b_fit,
        "outer_idx": outer_arr[:, 0].astype(np.int32),
        "U_outer": outer_arr[:, 1],
        "tau_rec_outer": outer_arr[:, 2],
        "nll_after_blockA": outer_arr[:, 3],
        "nll_after_std_grid": outer_arr[:, 4],
        "nll_outer": outer_arr[:, 5],
        "spectral_radius_outer": outer_arr[:, 6],
        "nz_weight_outer": outer_arr[:, 7].astype(np.int32),
        "blockA_outer": np.asarray(blockA_outer, dtype=np.int32),
        "blockA_epoch": np.asarray(blockA_epoch, dtype=np.int32),
        "blockA_bce_loss": np.asarray(blockA_bce_loss, dtype=np.float64),
        "blockA_l1_loss": np.asarray(blockA_l1_loss, dtype=np.float64),
        "blockA_loss": np.asarray(blockA_loss, dtype=np.float64),
        "joint_search_outer": np.asarray(joint_search_outer, dtype=np.int32),
        "joint_search_U": np.asarray(joint_search_U, dtype=np.float64),
        "joint_search_tau_rec": np.asarray(joint_search_tau_rec, dtype=np.float64),
        "joint_search_loss": np.asarray(joint_search_loss, dtype=np.float64),
        "kappa_fit": kappa.astype(np.float32),
        "alpha_fit": np.asarray([alpha], dtype=np.float64),
        "burn_bins": np.asarray([burn_bins], dtype=np.int32),
        "time_range_bins": np.asarray([start_bin, end_bin], dtype=np.int64),
        "requested_time_range_bins": np.asarray([requested_start_bin, requested_end_bin], dtype=np.int64),
        "fit_time_range_bins": np.asarray([start_bin + burn_bins, end_bin], dtype=np.int64),
        "time_step_sec": np.asarray([dt], dtype=np.float64),
    }

    out_md = {
        "short_name": out_stem,
        "fit_type": "BSSM_STD_blockFit",
        "provenance": {
            "script": os.path.basename(__file__),
            "spikesData_file": args.dataName,
        },
        "config": vars(args),
        "model": {
            "num_neurons": int(N),
            "num_input_bins": int(T_w),
            "num_eval_bins": int(Y_eval.shape[0]),
            "num_requested_bins": int(requested_bins),
            "num_init_bins": int(init_bins),
            "init_start_bin": int(start_bin + burn_bins),
            "init_end_bin": int(start_bin + burn_bins + init_bins - 1),
            "time_step_sec": float(dt),
            "kernel_synaptic_tau": float(args.synaptic_tau),
            "kernel_len_steps": int(mem_lag_steps),
            "kernel_alpha": float(alpha),
            "zero_diag": bool(args.zero_diag),
        },
        "result": {
            "U_final": float(out_d["U_outer"][-1]),
            "tau_rec_final": float(out_d["tau_rec_outer"][-1]),
            "nll_final": float(out_d["nll_outer"][-1]),
            "elapsed_sec": round(elapsed, 3),
        },
    }

    if args.verb > 1:
        pprint(out_md)
    write_data_npz(out_d, out_ff, metaD=out_md, verb=max(1, int(args.verb)))

    if args.verb > 0:
        print("\nSaved: %s" % out_ff)
        print(
            "  final_U=%.6f  final_tau=%.6f  elapsed=%.1fs"
            % (out_d["U_outer"][-1], out_d["tau_rec_outer"][-1], elapsed)
        )


if __name__ == "__main__":
    main()
