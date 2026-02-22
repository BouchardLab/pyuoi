#!/usr/bin/env python3
"""
prism_stage4_train.py
Stage 4: Dictionary learning with Group LASSO (λ3) + E/I sign-consistency (λ4).
Optimizer: SGD + proximal gradient (group soft-thresholding after each gradient step).
Coefficients c_{m,t} are held fixed from Stage 2 output.
No E_true used during training.
Spectral stability enforced by hard projection only (no soft penalty).
"""

import numpy as np
import torch
import argparse, os, time, json
import hashlib

from toolbox.Util_NumpyIO import read_data_npz, write_data_npz

# ── helpers ────────────────────────────────────────────────────────────────────

def save_npz(path, data_dict, verb=1):
    if verb: print(f'saving data as npz: {path}')
    save_dict = {}
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        save_dict[k] = v
        if verb: print(f'  npz-write : {k} {np.array(v).shape} {np.array(v).dtype}')
    np.savez(path, **save_dict)
    sz = os.path.getsize(path+'.npz')/1e6 if os.path.exists(path+'.npz') else 0
    if verb: print(f'closed  npz: {path}.npz  size={sz:.2f} MB')

def short_hash(s, n=6):
    return hashlib.md5(s.encode()).hexdigest()[:n]

# ── spectral radius functions ──────────────────────────────────────────────────

def compute_rhos(fA):
    """Return list of spectral radii for each mode. No grad, no side effects."""
    rhos = []
    with torch.no_grad():
        for m in range(fA.shape[0]):
            try:
                rho = torch.linalg.eigvals(fA[m]).abs().max().item()
            except Exception:
                rho = 0.0
            rhos.append(rho)
    return rhos

def spectral_scale_to_(fA, rho_target):
    """
    In-place: scale each mode TO exactly rho_target.
    Used only at initialization.
    """
    with torch.no_grad():
        for m in range(fA.shape[0]):
            try:
                rho = torch.linalg.eigvals(fA[m]).abs().max().item()
                fA[m].mul_(rho_target / max(rho, 1e-8))
            except Exception:
                pass

def spectral_project_(fA, rho_max):
    """
    In-place: clip each mode DOWN to rho_max if exceeded.
    Used every training step after the SGD update.
    """
    with torch.no_grad():
        for m in range(fA.shape[0]):
            try:
                rho = torch.linalg.eigvals(fA[m]).abs().max().item()
                if rho > rho_max:
                    fA[m].mul_(rho_max / max(rho, 1e-8))
            except Exception:
                pass

# ── group LASSO proximal operator ──────────────────────────────────────────────

def group_soft_threshold_(fA, alpha_lam3):
    """
    In-place Group Lasso proximal step for each synapse (i,j):
      G_{i,j} = sqrt(sum_m fA[m,i,j]^2)
      fA[m,i,j] *= max(0, 1 - alpha*lam3 / G_{i,j})
    alpha_lam3 = learning_rate * lam_grp
    """
    with torch.no_grad():
        G     = fA.pow(2).sum(dim=0).sqrt()                              # (N, N)
        scale = (1.0 - alpha_lam3 / G.clamp(min=1e-12)).clamp(min=0.0)  # (N, N)
        fA.mul_(scale.unsqueeze(0))                                      # broadcast over M

# ── E/I observables and penalty ────────────────────────────────────────────────

def compute_PQ(fA):
    """
    fA: (M, N, N)  fA[m, i, j] = weight from j to i in mode m
    Returns P (N,), Q (N,) — excitatory/inhibitory weight norms per neuron j.
    """
    pos = fA.clamp(min=0.0)
    neg = fA.clamp(max=0.0)
    P   = pos.pow(2).sum(dim=0).sum(dim=0).sqrt()   # (N,)
    Q   = neg.pow(2).sum(dim=0).sum(dim=0).sqrt()   # (N,)
    return P, Q

def ei_penalty_grad(fA, lam_ei, P, Q, eps=1e-8):
    """
    Analytic gradient of E/I penalty w.r.t. fA[m,i,j].
    Returns grad tensor same shape as fA.
    """
    grad     = torch.zeros_like(fA)
    pos_mask = fA > 0
    neg_mask = fA < 0
    Q_over_P = (Q / (P + eps)).unsqueeze(0).unsqueeze(0)   # (1, 1, N)
    P_over_Q = (P / (Q + eps)).unsqueeze(0).unsqueeze(0)   # (1, 1, N)
    grad[pos_mask] = lam_ei * Q_over_P.expand_as(fA)[pos_mask] * fA[pos_mask]
    grad[neg_mask] = lam_ei * P_over_Q.expand_as(fA)[neg_mask] * fA[neg_mask]
    return grad

# ── NLL (Poisson deviance) ─────────────────────────────────────────────────────

def poisson_nll_batch(spikes_t, spikes_tm1, fA, fB, C_t, dt=0.01, eta_clip=20.0):
    """
    spikes_t:   (B, N)
    spikes_tm1: (B, N)
    fA: (M, N, N),  fB: (M, N),  C_t: (B, M)
    Returns scalar NLL per time-step.
    """
    Ay  = torch.einsum('mnj,bj->bmn', fA, spikes_tm1)
    eta = torch.einsum('bm,bmn->bn', C_t, Ay) + \
          torch.einsum('bm,mn->bn',  C_t, fB)
    eta = eta.clamp(max=eta_clip)
    lam = eta.exp() * dt
    nll = lam.sum() - (spikes_t * torch.log(lam + 1e-10)).sum()
    return nll / spikes_t.shape[0]

# ── correlation helper ─────────────────────────────────────────────────────────

def pearson_r(a, b):
    a = a.flatten();  b = b.flatten()
    a = a - a.mean();  b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return (a * b).sum() / denom

# ── main ───────────────────────────────────────────────────────────────────────

def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-v","--verbosity", type=int, default=1, dest="verb")
    p.add_argument("--basePath", default="/dataVault2026/neurodata_tmp2")
    p.add_argument("--dataName", default=None)
    p.add_argument("--device", default="cuda")
    # optimiser
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--end_lr_frac", type=float, default=0.2,
                   help="Final LR as fraction of initial (linear decay).")
    p.add_argument("--n_epoch",     type=int,   default=150)
    p.add_argument("--batch",       type=int,   default=500)
    # constraints / penalties
    p.add_argument("--rho_max",   type=float, default=0.97,
                   help="Spectral radius hard ceiling (projection).")
    p.add_argument("--lam_grp",   type=float, default=10.0,
                   help="Group LASSO weight λ3.")
    p.add_argument("--lam_ei",    type=float, default=0.0,
                   help="E/I sign-consistency weight λ4.")
    p.add_argument("--ei_warmup", type=int,   default=50,
                   help="Epoch at which λ4 reaches full strength (linear ramp).")
    p.add_argument("--init_rho",  type=float, default=0.40)
    p.add_argument("--stable_thresh", type=float, default=0.65)
    p.add_argument("--time_steps_range", type=int, nargs=2, default=None)
    p.add_argument("--grp_warmup", type=int, default=50,
               help="Epoch at which lam_grp reaches full strength (linear ramp).")
    return p

def main():
    parser = get_parser()
    args   = parser.parse_args()

    print(f'myArg-program: prism_stage4_train.py')
    for k, v in vars(args).items():
        print(f'  myArg: {k} {v}')

    base     = args.basePath
    inp_fit  = os.path.join(base, 'prismFit')
    inp_spk  = os.path.join(base, 'spikesData')
    inp_tru  = os.path.join(base, 'truthDale')
    out_path = os.path.join(base, 'prismFit')
    os.makedirs(out_path, exist_ok=True)

    # ── load stage 2 output ───────────────────────────────────────────────────
    s2_name  = args.dataName
    s2_file  = os.path.join(inp_fit, f'{s2_name}.stage2.npz')
    s2D, s2MD = read_data_npz(s2_file, verb=args.verb > 0)
    C_hat    = s2D['C_hat'].astype(np.float32)    # (T, M)
    M        = C_hat.shape[1]
    dt       = float(s2MD.get('time_step_sec', s2MD.get('dt', 0.01)))
    eta_clip = float(s2MD.get('eta_clip', 20.0))

    spikes_name = s2MD["input_spikes_name"]
    truth_name  = s2MD["input_truth_name"]

    # ── load spikes ───────────────────────────────────────────────────────────
    spk_file = os.path.join(inp_spk, f'{spikes_name}.spikes.npz')
    spk_data, _ = read_data_npz(spk_file, verb=args.verb > 0)
    spikes   = spk_data['spikes'].astype(np.float32)   # (T, N)
    N        = spikes.shape[1]

    if args.time_steps_range is not None:
        t0, t1 = args.time_steps_range
        spikes = spikes[t0:t1]
        C_hat  = C_hat[t0:t1]
    T = spikes.shape[0]
    print(f'\nspikes=({T},{N})  C_hat=({T},{M})  M={M}  dt={dt}  eta_clip={eta_clip}')

    # ── load truth (diagnostics only — not used in training) ──────────────────
    tru_file  = os.path.join(inp_tru, f'{truth_name}.simTruth.npz')
    has_truth = os.path.exists(tru_file)
    A_true = B_true = E_true = None
    if has_truth:
        truD, _ = read_data_npz(tru_file, verb=args.verb > 0)
        A_true  = torch.tensor(truD['A_true'], dtype=torch.float32)
        B_true  = torch.tensor(truD['B_true'], dtype=torch.float32)
        E_true  = truD['E_true']

    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── stable-bin mask per mode ──────────────────────────────────────────────
    stable_mask = C_hat >= args.stable_thresh   # (T, M) bool numpy
    for m in range(M):
        n_stable = stable_mask[:, m].sum()
        print(f'  state {m}: {n_stable} stable bins ({100*n_stable/T:.1f}% of T)')

    # ── tensors on device ─────────────────────────────────────────────────────
    spk_gpu = torch.tensor(spikes, dtype=torch.float32, device=device)
    C_gpu   = torch.tensor(C_hat,  dtype=torch.float32, device=device)

    # ── initialize dictionaries ───────────────────────────────────────────────
    torch.manual_seed(42)
    fA = torch.randn(M, N, N, dtype=torch.float32, device=device) * 0.01
    fB = torch.zeros(M, N,    dtype=torch.float32, device=device)

    mean_rate = spk_gpu.mean(dim=0).clamp(min=1e-6)
    for m in range(M):
        fB[m] = torch.log(mean_rate / dt)

    spectral_scale_to_(fA, args.init_rho)   # scale TO init_rho

    fA.requires_grad_(True)
    fB.requires_grad_(True)

    print('\nInitialized fA spectral radii: '
          + '  '.join([f'ρ{m}={r:.3f}' for m, r in enumerate(compute_rhos(fA))]))

    # ── training loop ─────────────────────────────────────────────────────────
    t_start = time.time()
    hist    = dict(nll=[], grp=[], ei=[], loss=[],
                   rhos=[[] for _ in range(M)])

    for epoch in range(1, args.n_epoch + 1):

        frac       = 1.0 - (1.0 - args.end_lr_frac) * (epoch - 1) / (args.n_epoch - 1)
        lr         = args.lr * frac
        ei_scale   = min(1.0, epoch / max(args.ei_warmup, 1)) \
                     if args.ei_warmup > 0 else 1.0
        lam_ei_eff = args.lam_ei * ei_scale

        idx       = torch.randperm(T - 1, device=device)
        n_batches = max(1, (T - 1) // args.batch)
        epoch_nll = epoch_grp = epoch_ei = 0.0
        n_seen    = 0

        for b in range(n_batches):
            batch_idx = idx[b * args.batch: (b + 1) * args.batch]
            if len(batch_idx) == 0:
                continue
            t_idx   = batch_idx + 1
            spk_t   = spk_gpu[t_idx]
            spk_tm1 = spk_gpu[t_idx - 1]
            C_t     = C_gpu[t_idx]

            # ── forward: NLL only (no eigvals in autograd graph) ─────────────
            nll  = poisson_nll_batch(spk_t, spk_tm1, fA, fB, C_t, dt, eta_clip)
            loss = nll

            # E/I observables (detached — analytic grad added below)
            with torch.no_grad():
                P, Q   = compute_PQ(fA)
                ei_val = lam_ei_eff * (P * Q).sum()

            # ── backward ─────────────────────────────────────────────────────
            if fA.grad is not None: fA.grad.zero_()
            if fB.grad is not None: fB.grad.zero_()
            loss.backward()

            if args.lam_ei > 0 and ei_scale > 0:
                with torch.no_grad():
                    fA.grad.add_(ei_penalty_grad(fA, lam_ei_eff, P, Q))

            # ── SGD step ─────────────────────────────────────────────────────
            with torch.no_grad():
                fA.data.sub_(lr * fA.grad)
                fB.data.sub_(lr * fB.grad)

            # ── proximal: group soft-threshold ────────────────────────────────
            grp_scale = min(1.0, epoch / max(args.grp_warmup, 1))
            group_soft_threshold_(fA, lr * args.lam_grp * grp_scale)
            
            # ── spectral projection: hard clip DOWN ───────────────────────────
            spectral_project_(fA, args.rho_max)

            epoch_nll += nll.item()
            epoch_grp += (fA.detach().pow(2).sum(dim=0).sqrt().sum().item()
                          * args.lam_grp)
            epoch_ei  += ei_val.item()
            n_seen    += 1

        epoch_nll /= max(n_seen, 1)

        # post-epoch diagnostics (all no_grad) ─────────────────────────────────
        rhos = compute_rhos(fA)

        mode_stats = []
        with torch.no_grad():
            for m in range(M):
                t_idx_m = torch.where(
                    torch.tensor(stable_mask[1:, m], device=device))[0] + 1
                if len(t_idx_m) == 0:
                    mode_stats.append(float('nan'))
                    continue
                nll_m = poisson_nll_batch(
                    spk_gpu[t_idx_m], spk_gpu[t_idx_m - 1],
                    fA, fB, C_gpu[t_idx_m], dt, eta_clip)
                mode_stats.append(nll_m.item())

        r_A_str = ''
        if has_truth:
            r_vals  = [pearson_r(fA[m].detach().cpu(), A_true[m]).item()
                       for m in range(M)]
            r_A_str = '  r_A=' + '/'.join([f'{r:.3f}' for r in r_vals])

        with torch.no_grad():
            G        = fA.detach().pow(2).sum(dim=0).sqrt()
            sparsity = float((G < 1e-6).sum()) / G.numel()

        hist['nll'].append(epoch_nll)
        hist['grp'].append(epoch_grp)
        hist['ei'].append(epoch_ei)
        hist['loss'].append(epoch_nll + epoch_grp + epoch_ei)
        for m in range(M):
            hist['rhos'][m].append(rhos[m])

        if epoch % 2 == 1 or epoch == args.n_epoch:
            rho_str = '  '.join([f'ρ{m}={rhos[m]:.3f}' for m in range(M)])
            print(f'  epoch {epoch:4d}/{args.n_epoch}  '
                  f'nll={epoch_nll:.4f}  '
                  f'grp={epoch_grp:.2e}  '
                  f'ei={epoch_ei:.2e}  '
                  f'lr={lr:.2e}  '
                  f'λ4_eff={lam_ei_eff:.2e}  '
                  f'sparsity={sparsity:.3f}  '
                  f'{rho_str}  '
                  f'elaT={time.time()-t_start:.1f}s')
            mode_str = '  '.join([f'[m{m}: nll={mode_stats[m]:.4f}]'
                                   for m in range(M)])
            print(f'    {mode_str}{r_A_str}')

    print(f'\nTotal elapsed: {time.time()-t_start:.1f} sec')

    # ── post-training E/I diagnostics ─────────────────────────────────────────
    fA_np = fA.detach().cpu().numpy()
    fB_np = fB.detach().cpu().numpy()

    P_fin, Q_fin = compute_PQ(fA.detach())
    P_np  = P_fin.cpu().numpy()
    Q_np  = Q_fin.cpu().numpy()
    sigma = np.sign(P_np**2 - Q_np**2)
    kappa = np.abs(P_np**2 - Q_np**2) / (P_np**2 + Q_np**2 + 1e-8)
    khi, klo = 0.7, 0.3
    tier1 = int(np.sum(kappa > khi))
    tier2 = int(np.sum((kappa >= klo) & (kappa <= khi)))
    tier3 = int(np.sum(kappa < klo))
    n_exc = int(np.sum((sigma > 0) & (kappa > khi)))
    n_inh = int(np.sum((sigma < 0) & (kappa > khi)))

    print('\nDale diagnostics — A_hat (emergent)')
    print(f'  E/I tier 1 (confident, κ>{khi}): {tier1}  '
          f'excitatory={n_exc}  inhibitory={n_inh}')
    print(f'  E/I tier 2 (tentative):           {tier2}')
    print(f'  E/I tier 3 (undecided, κ<{klo}):  {tier3}')

    if has_truth and E_true is not None:
        true_type = np.sign(
            np.sum([A_true[m].numpy() for m in range(M)], axis=0).sum(axis=0))
        conf_mask = kappa > khi
        if conf_mask.sum() > 0:
            agree = float(np.mean(sigma[conf_mask] == true_type[conf_mask]))
            print(f'  E/I agreement with truth (tier-1): {agree:.3f}')

    # ── save ──────────────────────────────────────────────────────────────────
    param_str = f'{s2_name}_lgrp{args.lam_grp:.0e}_lei{args.lam_ei:.0e}'
    out_hash  = short_hash(param_str + str(time.time()))
    out_name  = f'{s2_name}_{out_hash}'
    out_file  = os.path.join(out_path, f'{out_name}.stage4')

    meta = {
        'stage': 4, 'dataName': args.dataName,
        'N': N, 'M': M, 'T': T, 'dt': dt,
        'lr': args.lr, 'n_epoch': args.n_epoch,
        'rho_max': args.rho_max,
        'lam_grp': args.lam_grp, 'lam_ei': args.lam_ei,
        'ei_warmup': args.ei_warmup,
    }

    save_dict = {
        'fA_hat':     fA_np,
        'fB_hat':     fB_np,
        'P':          P_np,
        'Q':          Q_np,
        'sigma':      sigma.astype(np.float32),
        'kappa':      kappa.astype(np.float32),
        'nll_epoch':  np.array(hist['nll'],  dtype=np.float32),
        'grp_epoch':  np.array(hist['grp'],  dtype=np.float32),
        'ei_epoch':   np.array(hist['ei'],   dtype=np.float32),
        'loss_epoch': np.array(hist['loss'], dtype=np.float32),
        'rho_epoch':  np.array(hist['rhos'], dtype=np.float32),
        'meta.JSON':  np.array([json.dumps(meta)])
    }
    if has_truth:
        save_dict['A_true'] = A_true.numpy().astype(np.float32)
        save_dict['B_true'] = B_true.numpy().astype(np.float32)
        save_dict['E_true'] = E_true.astype(np.int32)

    save_npz(out_file, save_dict, args.verb)
    print(f'\n  ./prism_stage4_eval.py  --basePath $basePath  '
          f'--dataName {out_name}')

if __name__ == '__main__':
    main()
