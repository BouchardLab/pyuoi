#!/usr/bin/env python3
"""Stage (c) de-biased active-set fitting for PRISM-EM 3c."""

import math
import os
import time

os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from PrismEM_Workhorse3c import (
    _history_dict,
    _history_arrays,
    is_rank0,
    make_optimizer_and_scheduler,
    make_pair_loader,
    onehot_from_curr_states,
    pair_range_for_rank,
    run_estep,
    run_estep_shard,
    viterbi_decode,
)


class DebiasActiveSetGLM(nn.Module):
    """Poisson GLM with active-vector parameterization for constrained A."""

    def __init__(
        self,
        n_neuron,
        n_state,
        eta_clip,
        dale_i,
        dale_j,
        dale_sign,
        free_i,
        free_j,
        u_init,
        v_init,
        b_init,
    ):
        super().__init__()
        self.N = int(n_neuron)
        self.M = int(n_state)
        self.eta_clip = float(eta_clip)

        self.register_buffer("dale_i", torch.as_tensor(dale_i, dtype=torch.long))
        self.register_buffer("dale_j", torch.as_tensor(dale_j, dtype=torch.long))
        self.register_buffer("dale_sign", torch.as_tensor(dale_sign, dtype=torch.float32))
        self.register_buffer("free_i", torch.as_tensor(free_i, dtype=torch.long))
        self.register_buffer("free_j", torch.as_tensor(free_j, dtype=torch.long))

        self.u = nn.Parameter(torch.as_tensor(u_init, dtype=torch.float32).clone())
        self.v = nn.Parameter(torch.as_tensor(v_init, dtype=torch.float32).clone())
        self.B = nn.Parameter(torch.as_tensor(b_init, dtype=torch.float32).clone())

    def effective_A(self):
        A = torch.zeros((self.N, self.N), dtype=self.B.dtype, device=self.B.device)
        if self.u.numel() > 0:
            A[self.dale_i, self.dale_j] = self.dale_sign * self.u.pow(2)
        if self.v.numel() > 0:
            A[self.free_i, self.free_j] = self.v
        return A

    def forward(self, y_prev, c, dt):
        A = self.effective_A()
        eta = y_prev @ A.t() + c @ self.B
        return torch.exp(torch.clamp(eta, max=self.eta_clip)) * float(dt)


def build_debias_active_set(A_stage, selected_mask, neuron_type, eps_u):
    """Build active index arrays and initial u/v from Stage (b) arrays."""
    A_stage = np.asarray(A_stage, dtype=np.float32)
    selected_mask = np.asarray(selected_mask, dtype=bool)
    neuron_type = np.asarray(neuron_type, dtype=np.int8)
    if A_stage.ndim != 2 or A_stage.shape[0] != A_stage.shape[1]:
        raise ValueError("A_stage must be square")
    if selected_mask.shape != A_stage.shape:
        raise ValueError("selected_mask must have same shape as A_stage")
    n_neuron = A_stage.shape[0]
    if neuron_type.shape[0] != n_neuron:
        raise ValueError("neuron_type length must match A_stage columns")

    dale_i = []
    dale_j = []
    dale_sign = []
    free_i = []
    free_j = []
    u_init = []
    v_init = []
    eps_u = float(eps_u)

    for i in range(n_neuron):
        for j in range(n_neuron):
            if i == j:
                free_i.append(i)
                free_j.append(j)
                v_init.append(float(A_stage[i, j]))
                continue
            if not bool(selected_mask[i, j]):
                continue
            src_type = int(neuron_type[j])
            if src_type > 0:
                dale_i.append(i)
                dale_j.append(j)
                dale_sign.append(1.0)
                u_init.append(max(math.sqrt(abs(float(A_stage[i, j]))), eps_u))
            elif src_type < 0:
                dale_i.append(i)
                dale_j.append(j)
                dale_sign.append(-1.0)
                u_init.append(max(math.sqrt(abs(float(A_stage[i, j]))), eps_u))
            else:
                free_i.append(i)
                free_j.append(j)
                v_init.append(float(A_stage[i, j]))

    return {
        "dale_i": np.asarray(dale_i, dtype=np.int64),
        "dale_j": np.asarray(dale_j, dtype=np.int64),
        "dale_sign": np.asarray(dale_sign, dtype=np.float32),
        "free_i": np.asarray(free_i, dtype=np.int64),
        "free_j": np.asarray(free_j, dtype=np.int64),
        "u_init": np.asarray(u_init, dtype=np.float32),
        "v_init": np.asarray(v_init, dtype=np.float32),
        "num_dale": int(len(dale_i)),
        "num_free": int(len(free_i)),
        "num_diag": int(n_neuron),
        "num_selected_offdiag": int(np.sum(selected_mask & ~np.eye(n_neuron, dtype=bool))),
    }


def init_debias_model(active, B_init, eta_clip, ctx):
    B_init = np.asarray(B_init, dtype=np.float32)
    if B_init.ndim == 1:
        B_init = B_init[None, :]
    n_state, n_neuron = B_init.shape
    model = DebiasActiveSetGLM(
        n_neuron,
        n_state,
        eta_clip,
        active["dale_i"],
        active["dale_j"],
        active["dale_sign"],
        active["free_i"],
        active["free_j"],
        active["u_init"],
        active["v_init"],
        B_init,
    ).to(ctx.device)
    if ctx.is_dist:
        model = DDP(model, device_ids=[ctx.local_rank])
    return model


def project_active_spectral_(mdl, rho_max, correction_strength=1.0):
    """Scale u/v so the reconstructed A obeys the spectral-radius limit."""
    with torch.no_grad():
        A = mdl.effective_A()
        rho = float(torch.linalg.eigvals(A).abs().max().item())
        alpha = min(1.0, max(0.0, float(correction_strength)))
        if rho > float(rho_max) and alpha > 0.0:
            hard_scale = float(rho_max) / float(rho)
            scale = 1.0 - alpha * (1.0 - hard_scale)
            if mdl.u.numel() > 0:
                mdl.u.mul_(math.sqrt(scale))
            if mdl.v.numel() > 0:
                mdl.v.mul_(scale)
    return rho


def run_debias_mstep_epoch(
    model,
    loader,
    optimizer,
    device,
    dt,
    rho_max,
    rho_every,
    apply_rho,
    rho_correction_strength,
    off_mask,
    progress_every_batches=0,
    progress_prefix="M-step",
):
    model.train()
    mdl = model.module if hasattr(model, "module") else model
    s_tot = 0.0
    s_nll = 0.0
    eps = 1e-8
    rho_every = max(1, int(rho_every))
    progress_every_batches = int(progress_every_batches)
    n_batch = len(loader)
    if progress_every_batches > 0:
        print("%s start batches=%d" % (progress_prefix, n_batch), flush=True)

    for bi, (yp, yc, cc) in enumerate(loader):
        yp = yp.to(device, non_blocking=True)
        yc = yc.to(device, non_blocking=True)
        cc = cc.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(yp, cc, dt)
        nll = (-yc * torch.log(pred + eps) + pred).mean()
        nll.backward()
        optimizer.step()

        if apply_rho and (bi % rho_every == 0):
            project_active_spectral_(mdl, rho_max, rho_correction_strength)

        s_tot += float(nll.item())
        s_nll += float(nll.item())
        batch_done = bi + 1
        if progress_every_batches > 0 and (
            batch_done % progress_every_batches == 0 or batch_done == n_batch
        ):
            print(
                "%s batch %d/%d nll=%.4e lr=%.2e"
                % (progress_prefix, batch_done, n_batch, float(nll.item()), optimizer.param_groups[0]["lr"]),
                flush=True,
            )

    nb = max(1, len(loader))
    if dist.is_available() and dist.is_initialized():
        v = torch.tensor([s_tot, s_nll, float(nb)], dtype=torch.float64, device=device)
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
        s_tot = float(v[0].item())
        s_nll = float(v[1].item())
        nb = int(v[2].item())

    with torch.no_grad():
        A = mdl.effective_A()
        nz = int((A[off_mask] != 0).sum().item())
        rho = float(torch.linalg.eigvals(A).abs().max().item())
    return dict(loss=s_tot / nb, nll=s_nll / nb, l1=0.0, rho=rho, nz=nz)


def _decode_and_broadcast(c_hat_np, p_stay, ctx):
    if ctx.is_dist:
        if is_rank0(ctx):
            s_np = viterbi_decode(c_hat_np, p_stay).astype(np.int64, copy=False)
            s_t = torch.as_tensor(s_np, dtype=torch.int64, device=ctx.device)
        else:
            s_t = torch.empty((c_hat_np.shape[0],), dtype=torch.int64, device=ctx.device)
        dist.broadcast(s_t, src=0)
        return s_t.cpu().numpy()
    return viterbi_decode(c_hat_np, p_stay).astype(np.int64, copy=False)


def _run_one_estep(Yp_gpu, Yc_gpu, model, c_hat_gpu, dt, eta_clip, args, ctx):
    mdl = model.module if hasattr(model, "module") else model
    progress_every = getattr(args, "progress_every_pairs", 0) if is_rank0(ctx) else 0
    with torch.no_grad():
        A_cur = mdl.effective_A()
        B_cur = mdl.B.data
        if ctx.is_dist:
            t_pairs = Yp_gpu.shape[0]
            p0, p1, n_loc = pair_range_for_rank(t_pairs, ctx.world_size, ctx.rank)
            t0 = p0 + 1
            t1 = p1 + 1
            nll_local, n_pair_local = run_estep_shard(
                Yp_gpu,
                Yc_gpu,
                A_cur,
                B_cur,
                c_hat_gpu,
                dt,
                eta_clip,
                float(args.lambda2),
                float(args.lr_estep),
                int(args.pgd_iter),
                t0,
                t1,
                progress_every=progress_every,
                progress_prefix="E-step",
            )
            c_upd = torch.zeros_like(c_hat_gpu)
            c_msk = torch.zeros((c_hat_gpu.shape[0], 1), dtype=torch.float32, device=ctx.device)
            c_upd[0] = c_hat_gpu[0]
            c_msk[0] = 1.0
            if n_loc > 0:
                c_upd[t0:t1 + 1] = c_hat_gpu[t0:t1 + 1]
                c_msk[t0:t1 + 1] = 1.0
            dist.all_reduce(c_upd, op=dist.ReduceOp.SUM)
            dist.all_reduce(c_msk, op=dist.ReduceOp.SUM)
            c_avg = c_upd / torch.clamp(c_msk, min=1.0)
            c_hat_gpu.copy_(torch.where(c_msk > 0.0, c_avg, c_hat_gpu))
            ev = torch.tensor([nll_local, float(n_pair_local)], dtype=torch.float64, device=ctx.device)
            dist.all_reduce(ev, op=dist.ReduceOp.SUM)
            return float(ev[0].item() / max(1.0, ev[1].item()))
        return run_estep(
            Yp_gpu,
            Yc_gpu,
            A_cur,
            B_cur,
            c_hat_gpu,
            dt,
            eta_clip,
            float(args.lambda2),
            float(args.lr_estep),
            int(args.pgd_iter),
            progress_every=progress_every,
            progress_prefix="E-step",
        )


def _rho_correction_for_iter(iter_idx, args):
    delay = int(args.delay_iter_4_ArhoMax)
    target = int(args.target_iter_4_ArhoMax)
    apply_rho = int(iter_idx) > delay
    iters_left = max(1, target - int(iter_idx) + 1)
    strength = 1.0 / float(iters_left) if apply_rho else 0.0
    return apply_rho, strength


def _should_log_m_epoch(epoch_idx, total_epochs, every):
    every = int(every)
    if every <= 0:
        return False
    epoch_idx = int(epoch_idx)
    total_epochs = int(total_epochs)
    return epoch_idx == 1 or epoch_idx == total_epochs or (epoch_idx % every) == 0


def _append_m_history(h, met, optimizer, rho_strength):
    h["m_loss_epoch"].append(met["loss"])
    h["m_nll_epoch"].append(met["nll"])
    h["m_l1_epoch"].append(0.0)
    h["rho_epoch"].append(met["rho"])
    h["nz_edges_epoch"].append(met["nz"])
    h["learning_rates"].append(float(optimizer.param_groups[0]["lr"]))
    h["rho_correction_strength_epoch"].append(float(rho_strength))


def run_debias_fit(spikes, spikeMD, A_stage, B_stage, selected_mask,
                   neuron_type, c_stageB, S_stageB, args, ctx):
    """Run Stage (c) active-set de-biased fitting on all distributed ranks."""
    spikes = np.asarray(spikes)
    if spikes.ndim != 2 or spikes.shape[0] < 2:
        raise ValueError("spikes must have shape (T,N) with at least two bins")
    T_full, N = spikes.shape
    M = int(args.num_states)
    dt = float(spikeMD["time_step_sec"])
    eta_clip = float(spikeMD["poisson_eta_clip"])
    p_stay = math.exp(-dt / float(args.decode_dwell_sec))
    do_log = is_rank0(ctx) and int(args.verb) > 0
    progress_batches = getattr(args, "progress_every_batches", 0) if is_rank0(ctx) else 0
    progress_epochs = getattr(args, "progress_every_epochs", 10)
    if do_log:
        print(
            "deBias start T=%d N=%d M=%d dt=%.6g device=%s "
            "state_mode=%s m_epochs=%d num_debias_iters=%d batch=%d"
            % (
                T_full,
                N,
                M,
                dt,
                ctx.device,
                args.state_mode,
                int(args.m_epochs),
                int(args.num_debias_iters),
                int(args.batch_size),
            ),
            flush=True,
        )

    c_stageB = np.asarray(c_stageB, dtype=np.float32)
    S_stageB = np.asarray(S_stageB, dtype=np.int64)
    if c_stageB.shape != (T_full, M):
        raise ValueError("c_stageB shape must match spike window and num_states")
    if S_stageB.shape[0] != T_full:
        raise ValueError("S_stageB length must match spike window")

    active = build_debias_active_set(A_stage, selected_mask, neuron_type, args.eps_u)
    if do_log:
        print(
            "deBias active set: dale=%d free=%d diag=%d selected_offdiag=%d"
            % (
                active["num_dale"],
                active["num_free"],
                active["num_diag"],
                active["num_selected_offdiag"],
            ),
            flush=True,
        )
    model = init_debias_model(active, B_stage, eta_clip, ctx)
    mdl = model.module if hasattr(model, "module") else model
    A_init = mdl.effective_A().detach().cpu().numpy().astype(np.float32)
    if do_log:
        print(
            "deBias model initialized A_init_shape=%s B_shape=%s"
            % (A_init.shape, tuple(mdl.B.shape)),
            flush=True,
        )

    yp_np = spikes[:-1].astype(np.float32)
    yc_np = spikes[1:].astype(np.float32)
    Yp_gpu = torch.tensor(yp_np, dtype=torch.float32, device=ctx.device)
    Yc_gpu = torch.tensor(yc_np, dtype=torch.float32, device=ctx.device)
    c_hat_gpu = torch.tensor(c_stageB, dtype=torch.float32, device=ctx.device)

    c_pairs_np = onehot_from_curr_states(S_stageB, M)
    loader, sampler = make_pair_loader(yp_np, yc_np, c_pairs_np, args, ctx, shuffle=True)
    if do_log:
        print(
            "deBias tensors ready Yp=%s Yc=%s c_hat=%s loader_batches=%d"
            % (
                tuple(Yp_gpu.shape),
                tuple(Yc_gpu.shape),
                tuple(c_hat_gpu.shape),
                len(loader),
            ),
            flush=True,
        )

    if str(args.state_mode) == "locked":
        total_m_epochs = int(args.warmup_locked_epochs) + int(args.m_epochs)
    else:
        total_m_epochs = int(args.warmup_locked_epochs) + int(args.num_debias_iters) * int(args.m_epochs)
    total_m_epochs = max(1, total_m_epochs)
    optimizer, scheduler = make_optimizer_and_scheduler(
        model, args, total_m_epochs, decay_from_epoch=0
    )

    off_mask = ~torch.eye(N, dtype=torch.bool, device=ctx.device)
    h = _history_dict()
    m_epoch_global = 0
    t_start = time.time()
    if do_log:
        print(
            "deBias optimizer ready total_m_epochs=%d lr=%.3e progress_every_epochs=%d"
            % (total_m_epochs, float(args.lr_mstep), int(progress_epochs)),
            flush=True,
        )

    for _ in range(int(args.warmup_locked_epochs)):
        m_epoch_global += 1
        epoch_t0 = time.time()
        if sampler is not None:
            sampler.set_epoch(m_epoch_global)
        apply_rho, rho_strength = _rho_correction_for_iter(1, args)
        log_m_epoch = do_log and _should_log_m_epoch(m_epoch_global, total_m_epochs, progress_epochs)
        met = run_debias_mstep_epoch(
            model, loader, optimizer, ctx.device, dt, args.rho_max,
            args.prescale_m_step_4_ArhoMax, apply_rho, rho_strength, off_mask,
            progress_every_batches=progress_batches if log_m_epoch else 0,
            progress_prefix=(
                "warmup M-epoch %d/%d"
                % (m_epoch_global, total_m_epochs)
            ),
        )
        scheduler.step()
        _append_m_history(h, met, optimizer, rho_strength)
        if log_m_epoch:
            print(
                "warmup M-epoch %d/%d done nll=%.4e rho=%.4f nz=%d lr=%.2e ela=%.1fs"
                % (
                    m_epoch_global,
                    total_m_epochs,
                    met["nll"],
                    met["rho"],
                    met["nz"],
                    h["learning_rates"][-1],
                    time.time() - epoch_t0,
                ),
                flush=True,
            )

    if str(args.state_mode) == "locked":
        for _ in range(int(args.m_epochs)):
            m_epoch_global += 1
            epoch_t0 = time.time()
            if sampler is not None:
                sampler.set_epoch(m_epoch_global)
            apply_rho, rho_strength = _rho_correction_for_iter(1, args)
            log_m_epoch = do_log and _should_log_m_epoch(m_epoch_global, total_m_epochs, progress_epochs)
            met = run_debias_mstep_epoch(
                model, loader, optimizer, ctx.device, dt, args.rho_max,
                args.prescale_m_step_4_ArhoMax, apply_rho, rho_strength, off_mask,
                progress_every_batches=progress_batches if log_m_epoch else 0,
                progress_prefix=(
                    "locked M-epoch %d/%d"
                    % (m_epoch_global, total_m_epochs)
                ),
            )
            scheduler.step()
            _append_m_history(h, met, optimizer, rho_strength)
            if log_m_epoch:
                print(
                    "locked M-epoch %d/%d done nll=%.4e rho=%.4f nz=%d lr=%.2e ela=%.1fs"
                    % (
                        m_epoch_global,
                        total_m_epochs,
                        met["nll"],
                        met["rho"],
                        met["nz"],
                        h["learning_rates"][-1],
                        time.time() - epoch_t0,
                    ),
                    flush=True,
                )
        S_hat = S_stageB.astype(np.int64, copy=True)
    else:
        S_hat = S_stageB.astype(np.int64, copy=True)
        for deb_iter in range(1, int(args.num_debias_iters) + 1):
            te0 = time.time()
            if do_log:
                print(
                    "deBias iter %d/%d E-step start"
                    % (deb_iter, int(args.num_debias_iters)),
                    flush=True,
                )
            e_nll = _run_one_estep(
                Yp_gpu, Yc_gpu, model, c_hat_gpu, dt, eta_clip, args, ctx
            )
            if do_log:
                print(
                    "deBias iter %d/%d E-step done e_nll=%.4e ela=%.1fs"
                    % (
                        deb_iter,
                        int(args.num_debias_iters),
                        e_nll,
                        time.time() - te0,
                    ),
                    flush=True,
                )
            h["e_nll_em"].append(e_nll)
            c_np_iter = c_hat_gpu.detach().cpu().numpy()
            S_hat = _decode_and_broadcast(c_np_iter, p_stay, ctx)
            np.copyto(c_pairs_np, onehot_from_curr_states(S_hat, M))
            if do_log:
                occ = np.bincount(S_hat, minlength=M).astype(np.float64) / float(max(1, S_hat.size))
                print(
                    "deBias iter %d decoded state occupancy=%s"
                    % (deb_iter, np.array2string(occ, precision=4)),
                    flush=True,
                )

            for _ in range(int(args.m_epochs)):
                m_epoch_global += 1
                epoch_t0 = time.time()
                if sampler is not None:
                    sampler.set_epoch(m_epoch_global)
                apply_rho, rho_strength = _rho_correction_for_iter(deb_iter, args)
                log_m_epoch = do_log and _should_log_m_epoch(m_epoch_global, total_m_epochs, progress_epochs)
                met = run_debias_mstep_epoch(
                    model, loader, optimizer, ctx.device, dt, args.rho_max,
                    args.prescale_m_step_4_ArhoMax, apply_rho, rho_strength, off_mask,
                    progress_every_batches=progress_batches if log_m_epoch else 0,
                    progress_prefix=(
                        "iter %d/%d M-epoch %d/%d"
                        % (
                            deb_iter,
                            int(args.num_debias_iters),
                            m_epoch_global,
                            total_m_epochs,
                        )
                    ),
                )
                scheduler.step()
                _append_m_history(h, met, optimizer, rho_strength)
                if log_m_epoch:
                    print(
                        "iter %d M-epoch %d/%d done nll=%.4e rho=%.4f nz=%d lr=%.2e ela=%.1fs"
                        % (
                            deb_iter,
                            m_epoch_global,
                            total_m_epochs,
                            met["nll"],
                            met["rho"],
                            met["nz"],
                            h["learning_rates"][-1],
                            time.time() - epoch_t0,
                        ),
                        flush=True,
                    )

            if do_log:
                print(
                    "deBias iter %3d/%d E_nll=%.4e M_nll=%.4e rho=%.4f nz=%d lr=%.2e elaT=%.1fs"
                    % (
                        deb_iter,
                        int(args.num_debias_iters),
                        e_nll,
                        met["nll"],
                        met["rho"],
                        met["nz"],
                        h["learning_rates"][-1],
                        time.time() - te0,
                    ),
                    flush=True,
                )

    c_hat_np = c_hat_gpu.detach().cpu().numpy().astype(np.float32)
    if str(args.state_mode) == "locked":
        S_hat = S_stageB.astype(np.int64, copy=True)
    S_hat_CL = (1.0 - c_hat_np.max(axis=1)).astype(np.float32)
    A_debias = mdl.effective_A().detach().cpu().numpy().astype(np.float32)
    B_debias = mdl.B.detach().cpu().numpy().astype(np.float32)

    hist = _history_arrays(h)
    if do_log:
        print(
            "deBias final tensors ready A=%s B=%s train_elapsed=%.1fs"
            % (A_debias.shape, B_debias.shape, time.time() - t_start),
            flush=True,
        )
    return {
        "A_debias": A_debias,
        "B_debias": B_debias,
        "A_debias_init": A_init,
        "B_debias_init": np.asarray(B_stage, dtype=np.float32),
        "c_hat": c_hat_np,
        "S_hat": np.asarray(S_hat, dtype=np.int64),
        "S_hat_CL": S_hat_CL,
        "active": active,
        "history": hist,
        "elapsed_train_sec": float(time.time() - t_start),
    }
