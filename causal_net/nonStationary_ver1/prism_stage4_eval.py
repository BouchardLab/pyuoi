#!/usr/bin/env python3
"""
prism_stage4_eval.py
Evaluates Stage 4 output: A/B recovery, Group LASSO edge diagnostics,
E/I classification quality. Produces 3 canvas PNG files.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse, os, json
from matplotlib.gridspec import GridSpec

# ── helpers ────────────────────────────────────────────────────────────────────

def load_npz(path, verb=1):
    if verb: print(f'read data from npz: {path}')
    d = {}
    with np.load(path, allow_pickle=True) as f:
        for k in f.files:
            d[k] = f[k]
            if verb: print(f'  read obj: {k} {f[k].shape} {f[k].dtype}')
    if verb: print(f' done npz, num rec:{len(d)}')
    return d

def pearson_r(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    a -= a.mean(); b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + 1e-12))

def spectral_radius(A):
    return float(np.abs(np.linalg.eigvals(A)).max())

def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-v","--verbosity", type=int, default=1, dest="verb")
    p.add_argument("--basePath", default="/dataVault2026/neurodata_tmp2")
    p.add_argument("--dataName", default=None)
    p.add_argument("--plotFmt",  default="b")
    p.add_argument("--prune_thresh", type=float, default=0.04,
                   help="Group-norm threshold for edge pruning.")
    p.add_argument("--khi", type=float, default=0.7)
    p.add_argument("--klo", type=float, default=0.3)
    return p

def main():
    parser = get_parser()
    args   = parser.parse_args()

    print(f'myArg-program: prism_stage4_eval.py')
    for k, v in vars(args).items():
        print(f'  myArg: {k} {v}')

    base     = args.basePath
    inp_fit  = os.path.join(base, 'prismFit')
    out_dir  = os.path.join(base, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────────
    npz_file = os.path.join(inp_fit, f'{args.dataName}.stage4.npz')
    d = load_npz(npz_file, args.verb)

    fA_hat = d['fA_hat']        # (M, N, N)
    fB_hat = d['fB_hat']        # (M, N)
    P      = d['P']             # (N,)
    Q      = d['Q']             # (N,)
    sigma  = d['sigma']         # (N,)
    kappa  = d['kappa']         # (N,)

    nll_ep  = d['nll_epoch']
    pen_ep  = d.get('pen_epoch', np.zeros_like(nll_ep))   # optional (stage4 uses projection only)
    grp_ep  = d['grp_epoch']
    ei_ep   = d['ei_epoch']
    loss_ep = d['loss_epoch']
    rho_ep  = d['rho_epoch']    # (M, n_epoch)

    meta = json.loads(str(d['meta.JSON'][0]))
    M = fA_hat.shape[0]
    N = fA_hat.shape[1]

    has_truth = 'A_true' in d
    A_true = d['A_true'] if has_truth else None   # (M, N, N)
    B_true = d['B_true'] if has_truth else None
    E_true = d['E_true'] if 'E_true' in d else None

    # ── screen: per-mode recovery ──────────────────────────────────────────────
    print('\n' + '='*60)
    print('  STAGE 4 RECOVERY METRICS')
    print('='*60)

    rho_hat = [spectral_radius(fA_hat[m]) for m in range(M)]
    if has_truth:
        rho_true = [spectral_radius(A_true[m]) for m in range(M)]
        r_A = [pearson_r(fA_hat[m], A_true[m]) for m in range(M)]
        r_B = [pearson_r(fB_hat[m], B_true[m]) for m in range(M)]
        rmse_A = [float(np.sqrt(np.mean((fA_hat[m] - A_true[m])**2)))
                  for m in range(M)]
        print(f'\n{"mode":<6} {"r_A":>8} {"r_B":>8} {"RMSE_A":>10} '
              f'{"ρ_true":>8} {"ρ_hat":>8}')
        print('-'*56)
        for m in range(M):
            flag = 'OK' if r_A[m] > 0.9 else '!!'
            print(f'  {m:<4} {r_A[m]:>8.4f} {r_B[m]:>8.4f} {rmse_A[m]:>10.4f} '
                  f'{rho_true[m]:>8.3f} {rho_hat[m]:>8.3f}  {flag}')
    else:
        print(f'\n{"mode":<6} {"ρ_hat":>8}')
        for m in range(M): print(f'  {m:<4} {rho_hat[m]:>8.3f}')

    # ── screen: group LASSO edge diagnostics ──────────────────────────────────
    print('\n' + '='*60)
    print(f'  GROUP LASSO EDGE DIAGNOSTICS  '
          f'(λ3={meta["lam_grp"]:.1e}, prune_thresh={args.prune_thresh})')
    print('='*60)

    G = np.sqrt((fA_hat**2).sum(axis=0))   # (N, N)
    np.fill_diagonal(G, 0.0)

    survived_mask = G >= args.prune_thresh  # (N, N)
    n_survived    = survived_mask.sum()

    print(f'\n[1] Edge group norms  G_{{i,j}} = sqrt(sum_m fA_m,i,j^2)')
    print(f'    max={G.max():.4f}  mean(nonzero)={G[G>1e-6].mean():.4f}  '
          f'fraction_zero={float((G<1e-6).mean()):.3f}')

    if has_truth and E_true is not None:
        # Build true edge mask from A_true: edge exists if any mode nonzero
        A_true_2d = (A_true**2).sum(axis=0)  # (N,N)
        np.fill_diagonal(A_true_2d, 0.0)
        true_edge  = A_true_2d > 1e-6          # (N,N) bool
        true_zero  = ~true_edge
        n_true_edge = true_edge.sum()
        n_true_zero = true_zero.sum()

        TP = int((survived_mask &  true_edge).sum())
        FP = int((survived_mask & ~true_edge).sum())
        FN = int((~survived_mask & true_edge).sum())
        TN = int((~survived_mask & ~true_edge).sum())
        recall    = TP / max(n_true_edge, 1)
        precision = TP / max(TP + FP, 1)
        f1        = 2*recall*precision / max(recall+precision, 1e-8)

        print(f'\n[2] After pruning (G < {args.prune_thresh})')
        print(f'    True non-zero edges : {n_true_edge}')
        print(f'    True zero edges     : {n_true_zero}')
        print(f'    TP={TP}  FP={FP}  FN={FN}  TN={TN}')
        print(f'    Recall   : {recall:.4f}')
        print(f'    Precision: {precision:.4f}')
        print(f'    F1       : {f1:.4f}')

        # recall by |A_true| bin
        print(f'\n[3] Recall by |A_true| group-norm bin')
        A_true_G = np.sqrt((A_true**2).sum(axis=0))
        np.fill_diagonal(A_true_G, 0.0)
        true_vals = A_true_G[true_edge]
        bins = np.percentile(true_vals, [0, 20, 40, 60, 80, 100])
        for k in range(len(bins)-1):
            lo, hi = bins[k], bins[k+1]
            mask_bin = true_edge & (A_true_G >= lo) & (A_true_G <= hi)
            n_bin    = mask_bin.sum()
            n_surv   = (survived_mask & mask_bin).sum()
            rec_bin  = n_surv / max(n_bin, 1)
            print(f'    G_true in [{lo:.3f}, {hi:.3f}] : '
                  f'n={n_bin:5d}  survived={n_surv:5d}  recall={rec_bin:.3f}')

        # per-mode recall
        print(f'\n[4] Per-mode recall and r_A')
        for m in range(M):
            true_m  = np.abs(A_true[m]) > 1e-6
            np.fill_diagonal(true_m, False)
            hat_m   = np.abs(fA_hat[m]) > 1e-6
            rec_m   = float((hat_m & true_m).sum()) / max(true_m.sum(), 1)
            print(f'    Mode {m}: n_true={true_m.sum():6d}  '
                  f'recall={rec_m:.4f}  r_A={r_A[m]:.4f}')

    # ── screen: E/I classification ─────────────────────────────────────────────
    print('\n' + '='*60)
    print(f'  E/I CLASSIFICATION  (κhi={args.khi}, κlo={args.klo})')
    print('='*60)

    tier1_mask = kappa > args.khi
    tier2_mask = (kappa >= args.klo) & (kappa <= args.khi)
    tier3_mask = kappa < args.klo
    n1 = tier1_mask.sum()
    n2 = tier2_mask.sum()
    n3 = tier3_mask.sum()
    n_exc = int(((sigma > 0) & tier1_mask).sum())
    n_inh = int(((sigma < 0) & tier1_mask).sum())

    print(f'\n  Tier 1 (confident, κ>{args.khi}): {n1:3d}  '
          f'→  E={n_exc}  I={n_inh}')
    print(f'  Tier 2 (tentative):              {n2:3d}')
    print(f'  Tier 3 (undecided, κ<{args.klo}):  {n3:3d}')
    print(f'\n  κ stats: mean={kappa.mean():.3f}  '
          f'median={np.median(kappa):.3f}  '
          f'min={kappa.min():.3f}  max={kappa.max():.3f}')

    if has_truth and E_true is not None:
        # E_true as neuron type: column sums of A_true determine sign
        true_A_sum = sum(A_true[m] for m in range(M))  # (N,N)
        true_col   = true_A_sum.sum(axis=0)             # (N,) net outgoing
        true_sigma = np.sign(true_col)
        conf_mask  = tier1_mask
        if conf_mask.sum() > 0:
            agree = float(np.mean(sigma[conf_mask] == true_sigma[conf_mask]))
            print(f'\n  E/I agreement with truth (tier-1): {agree:.4f}  '
                  f'({conf_mask.sum()} neurons)')
        # Dale violation count in A_hat
        dale_viol = 0
        for j in range(N):
            col = fA_hat[:, :, j].flatten()
            if col.max() > 0 and col.min() < 0:
                dale_viol += 1
        print(f'  Dale violations in A_hat: {dale_viol}')

    print(f'\n  NLL: {nll_ep[0]:.4f} → {nll_ep[-1]:.4f}  '
          f'(Δ={100*(nll_ep[-1]-nll_ep[0])/abs(nll_ep[0]):.2f}%)')

    # ══════════════════════════════════════════════════════════════════════════
    # CANVAS 1: Training curves + A scatter + spectral radii
    # ══════════════════════════════════════════════════════════════════════════
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle(
        f'PRISM Stage 4 — Training & Recovery\n{args.dataName}\n'
        f'N={N} M={M} lr={meta["lr"]} n_epoch={meta["n_epoch"]} '
        f'λ3={meta["lam_grp"]:.1e} λ4={meta["lam_ei"]:.1e} ρmax={meta["rho_max"]}',
        fontsize=9)
    epochs = np.arange(1, len(nll_ep)+1)

    # [0,0] Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, loss_ep, 'b-',  lw=1.5, label='total loss')
    ax.plot(epochs, nll_ep,  'b--', lw=1.2, label='NLL')
    ax.plot(epochs, pen_ep,  'r:',  lw=1.0, label='ρ penalty')
    ax.plot(epochs, grp_ep,  'g:',  lw=1.0, label='group L1')
    if meta['lam_ei'] > 0:
        ax.plot(epochs, ei_ep, 'm:', lw=1.0, label='E/I pen')
    ax.set_title(f'Training curves\nNLL: {nll_ep[0]:.3f}→{nll_ep[-1]:.3f} '
                 f'(Δ={100*(nll_ep[-1]-nll_ep[0])/abs(nll_ep[0]):.2f}%)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # [0,1] Spectral radii over training
    ax = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 0.5, M))
    for m in range(M):
        ax.plot(epochs, rho_ep[m], color=colors[m], lw=1.5, label=f'm={m}')
    ax.axhline(meta['rho_max'], color='k', ls='--', lw=1, label=f'ρmax={meta["rho_max"]}')
    ax.set_title('Spectral radii over training')
    ax.set_xlabel('Epoch'); ax.set_ylabel('ρ')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # [0,2] A scatter if truth available
    ax = axes[0, 2]
    if has_truth:
        cmap = plt.cm.tab10(np.linspace(0, 0.5, M))
        for m in range(M):
            a_t = A_true[m].flatten()
            a_h = fA_hat[m].flatten()
            ax.scatter(a_t, a_h, s=1, alpha=0.15, color=cmap[m],
                       label=f'm={m} r={r_A[m]:.3f}')
        lim = max(np.abs(A_true).max(), np.abs(fA_hat).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title('A scatter (all modes pooled)')
        ax.set_xlabel('A_true'); ax.set_ylabel('A_hat')
        ax.legend(fontsize=7, markerscale=5)
    else:
        ax.text(0.5, 0.5, 'No truth available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('A scatter')

    # [1,0] r_A per mode bar
    ax = axes[1, 0]
    if has_truth:
        bars = ax.bar(range(M), r_A, color=[plt.cm.tab10(m/10) for m in range(M)])
        ax.axhline(0.9, color='r', ls='--', lw=1, label='thresh=0.9')
        ax.set_title('A correlation per mode')
        ax.set_xlabel('mode'); ax.set_ylabel('Pearson r')
        ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
        for i, v in enumerate(r_A):
            ax.text(i, v+0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No truth', ha='center', va='center',
                transform=ax.transAxes)

    # [1,1] RMSE per mode
    ax = axes[1, 1]
    if has_truth:
        ax.bar(range(M), rmse_A,
               color=[plt.cm.tab10(m/10) for m in range(M)])
        ax.set_title('A matrix RMSE per mode\n||A_hat - A_true||_F / N')
        ax.set_xlabel('mode'); ax.set_ylabel('RMSE')
        for i, v in enumerate(rmse_A):
            ax.text(i, v+1e-4, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No truth', ha='center', va='center',
                transform=ax.transAxes)

    # [1,2] Summary text
    ax = axes[1, 2]
    ax.axis('off')
    lines = [
        f'PRISM Stage 4 Summary',
        f'',
        f'N={N}  M={M}  n_epoch={meta["n_epoch"]}',
        f'lr={meta["lr"]}  end_lr_frac={meta.get("end_lr_frac",0.2)}',
        f'λ3(grp)={meta["lam_grp"]:.1e}  λ4(EI)={meta["lam_ei"]:.1e}',
        f'ρ_max={meta["rho_max"]}',
        f'',
        f'NLL init : {nll_ep[0]:.4f}',
        f'NLL final: {nll_ep[-1]:.4f}  '
        f'({100*(nll_ep[-1]-nll_ep[0])/abs(nll_ep[0]):.2f}%)',
        f'',
    ]
    if has_truth:
        for m in range(M):
            col = 'green' if r_A[m] > 0.9 else 'red'
            lines.append(f'mode {m}: r_A={r_A[m]:.4f}  '
                         f'ρ {rho_true[m]:.3f}→{rho_hat[m]:.3f}')
        lines += [
            f'',
            f'E/I tier1={n1} (E={n_exc} I={n_inh})',
            f'E/I tier2={n2}  tier3={n3}',
            f'mean κ = {kappa.mean():.3f}',
        ]
    txt = '\n'.join(lines)
    ax.text(0.05, 0.95, txt, transform=ax.transAxes,
            va='top', fontsize=9, family='monospace')

    fig1.tight_layout()
    p1 = os.path.join(out_dir, f'{args.dataName}_s4_canvas1.png')
    fig1.savefig(p1)
    print(f'\n  display {p1}')
    plt.close(fig1)

    # ══════════════════════════════════════════════════════════════════════════
    # CANVAS 2: Group LASSO edge diagnostics
    # ══════════════════════════════════════════════════════════════════════════
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle(
        f'PRISM Stage 4 — Group LASSO Edge Diagnostics\n{args.dataName}  '
        f'λ3={meta["lam_grp"]:.1e}  prune_thresh={args.prune_thresh}',
        fontsize=9)

    # [0,0] G distribution
    ax = axes[0, 0]
    G_flat = G.flatten()
    G_nz   = G_flat[G_flat > 1e-6]
    ax.hist(G_nz, bins=80, color='steelblue', alpha=0.8, log=True)
    ax.axvline(args.prune_thresh, color='r', ls='--', lw=1.5,
               label=f'thresh={args.prune_thresh}')
    ax.set_title(f'Group norm G_{{i,j}} distribution\n'
                 f'(non-zero edges, log-y)')
    ax.set_xlabel('G_{i,j}'); ax.set_ylabel('count (log)')
    ax.legend(fontsize=8)

    # [0,1] PR curve (if truth)
    ax = axes[0, 1]
    if has_truth and E_true is not None:
        thresholds = np.linspace(0, G.max()*0.8, 200)
        prec_list, rec_list = [], []
        for thr in thresholds:
            surv = G >= thr
            np.fill_diagonal(surv, False)
            tp = float((surv &  true_edge).sum())
            fp = float((surv & ~true_edge).sum())
            fn = float((~surv & true_edge).sum())
            prec_list.append(tp / max(tp+fp, 1))
            rec_list.append(tp  / max(tp+fn, 1))
        # find current operating point
        surv_op = G >= args.prune_thresh
        np.fill_diagonal(surv_op, False)
        tp_op = float((surv_op &  true_edge).sum())
        fp_op = float((surv_op & ~true_edge).sum())
        fn_op = float((~surv_op & true_edge).sum())
        rec_op  = tp_op / max(tp_op+fn_op, 1)
        prec_op = tp_op / max(tp_op+fp_op, 1)

        ax.plot(rec_list, prec_list, 'b-', lw=1.5)
        ax.scatter([rec_op], [prec_op], c='r', s=80, zorder=5,
                   label=f'thresh={args.prune_thresh}\n'
                         f'rec={rec_op:.3f} prec={prec_op:.3f}')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall curve\n(Group LASSO threshold sweep)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)

    # [0,2] G distribution: true edges vs true zeros
    ax = axes[0, 2]
    if has_truth and E_true is not None:
        g_te = G[true_edge]
        g_tz = G[true_zero & (G > 1e-6)]
        ax.hist(g_te, bins=60, color='green',  alpha=0.6,
                label=f'true edges n={len(g_te)}',   density=True)
        ax.hist(g_tz, bins=60, color='orange', alpha=0.6,
                label=f'true zeros n={len(g_tz)}',   density=True)
        ax.axvline(args.prune_thresh, color='r', ls='--', lw=1.5)
        ax.set_title('G distribution:\ntrue edges vs true zeros')
        ax.set_xlabel('G_{i,j}'); ax.set_ylabel('density')
        ax.legend(fontsize=8)

    # [1,0] Recall by G_true bin
    ax = axes[1, 0]
    if has_truth and E_true is not None:
        bin_labels, bin_recalls = [], []
        for k in range(len(bins)-1):
            lo, hi = bins[k], bins[k+1]
            mb = true_edge & (A_true_G >= lo) & (A_true_G <= hi)
            if mb.sum() == 0: continue
            rec_b = float((survived_mask & mb).sum()) / mb.sum()
            bin_labels.append(f'[{lo:.2f},{hi:.2f}]')
            bin_recalls.append(rec_b)
        ax.bar(range(len(bin_recalls)), bin_recalls, color='steelblue')
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=30, ha='right', fontsize=8)
        ax.set_title('True edge recall by G_true bin')
        ax.set_xlabel('|G_true| bin'); ax.set_ylabel('Recall')
        ax.set_ylim(0, 1.05); ax.axhline(1.0, color='k', ls='--', lw=0.8)

    # [1,1] Scatter: survived vs lost true edges
    ax = axes[1, 1]
    if has_truth and E_true is not None:
        # pool all modes, plot A_true vs A_hat for survived/lost
        A_t_flat = A_true.mean(axis=0).flatten()  # average over modes
        A_h_flat = fA_hat.mean(axis=0).flatten()
        te_flat  = true_edge.flatten()
        surv_flat= survived_mask.flatten()
        # true zeros
        ax.scatter(A_t_flat[~te_flat], A_h_flat[~te_flat],
                   s=0.3, alpha=0.1, color='gray', label='true zero')
        # survived true edges
        mask_surv = te_flat & surv_flat
        ax.scatter(A_t_flat[mask_surv], A_h_flat[mask_surv],
                   s=2, alpha=0.3, color='green', label='survived')
        # lost true edges
        mask_lost = te_flat & ~surv_flat
        ax.scatter(A_t_flat[mask_lost], A_h_flat[mask_lost],
                   s=10, alpha=0.8, color='red',
                   label=f'LOST ({mask_lost.sum()})')
        lim = max(np.abs(A_true).max(), np.abs(fA_hat).max()) * 1.05
        ax.plot([-lim,lim],[-lim,lim],'k--',lw=0.8)
        ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
        ax.set_title('Scatter: survived vs lost true edges\n(mean over modes)')
        ax.set_xlabel('A_true (mean)'); ax.set_ylabel('A_hat (mean)')
        ax.legend(fontsize=7, markerscale=3)

    # [1,2] Per-mode recall bar
 
    ax = axes[1, 2]
    if has_truth and E_true is not None:
        rec_by_mode = []
        for m in range(M):
            true_m = np.abs(A_true[m]) > 1e-6
            np.fill_diagonal(true_m, False)
            hat_m  = survived_mask  # use pruned A_hat
            rec_m  = float((hat_m & true_m).sum()) / max(true_m.sum(), 1)
            rec_by_mode.append(rec_m)
        ax.bar(range(M), rec_by_mode,
               color=[plt.cm.tab10(m/10) for m in range(M)])
        ax.set_title('True edge recall per mode\n(after group pruning)')
        ax.set_xlabel('mode'); ax.set_ylabel('Recall')
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color='k', ls='--', lw=0.8)
        for i, v in enumerate(rec_by_mode):
            ax.text(i, v+0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No truth', ha='center', va='center',
                transform=ax.transAxes)

    fig2.tight_layout()
    p2 = os.path.join(out_dir, f'{args.dataName}_s4_canvas2.png')
    fig2.savefig(p2)
    print(f'  display {p2}')
    plt.close(fig2)

    # ══════════════════════════════════════════════════════════════════════════
    # CANVAS 3: E/I classification
    # ══════════════════════════════════════════════════════════════════════════
    fig3, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig3.suptitle(
        f'PRISM Stage 4 — E/I Classification\n{args.dataName}  '
        f'λ4={meta["lam_ei"]:.1e}  κhi={args.khi}  κlo={args.klo}',
        fontsize=9)

    # [0,0] P vs Q scatter, colored by tier
    ax = axes[0, 0]
    tier_color = np.where(kappa > args.khi, 0,
                 np.where(kappa >= args.klo, 1, 2))
    colors_ei = ['steelblue', 'orange', 'gray']
    labels_ei = [f'Tier1 confident (n={n1})',
                 f'Tier2 tentative (n={n2})',
                 f'Tier3 undecided (n={n3})']
    for t, col, lab in zip([0,1,2], colors_ei, labels_ei):
        mask_t = tier_color == t
        if mask_t.sum() > 0:
            ax.scatter(P[mask_t], Q[mask_t], s=20, alpha=0.7,
                       color=col, label=lab)
    ax.set_xlabel('P_j  (excitatory norm)')
    ax.set_ylabel('Q_j  (inhibitory norm)')
    ax.set_title('P vs Q per neuron\n(colored by tier)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # [0,1] κ distribution
    ax = axes[0, 1]
    ax.hist(kappa, bins=40, color='steelblue', alpha=0.8, edgecolor='k', lw=0.3)
    ax.axvline(args.khi, color='g', ls='--', lw=1.5, label=f'κhi={args.khi}')
    ax.axvline(args.klo, color='r', ls='--', lw=1.5, label=f'κlo={args.klo}')
    ax.set_xlabel('κ  (confidence score)')
    ax.set_ylabel('count')
    ax.set_title(f'E/I confidence κ distribution\n'
                 f'mean={kappa.mean():.3f}  median={np.median(kappa):.3f}')
    ax.legend(fontsize=8)

    # [0,2] σ (E/I label) distribution
    ax = axes[0, 2]
    exc_conf = int(((sigma > 0) & (kappa > args.khi)).sum())
    inh_conf = int(((sigma < 0) & (kappa > args.khi)).sum())
    exc_tent = int(((sigma > 0) & (kappa >= args.klo) & (kappa <= args.khi)).sum())
    inh_tent = int(((sigma < 0) & (kappa >= args.klo) & (kappa <= args.khi)).sum())
    exc_und  = int(((sigma > 0) & (kappa < args.klo)).sum())
    inh_und  = int(((sigma < 0) & (kappa < args.klo)).sum())

    categories  = ['Exc\nTier1', 'Inh\nTier1',
                   'Exc\nTier2', 'Inh\nTier2',
                   'Exc\nTier3', 'Inh\nTier3']
    counts      = [exc_conf, inh_conf, exc_tent, inh_tent, exc_und, inh_und]
    bar_colors  = ['steelblue','coral','lightskyblue','lightsalmon',
                   'lightgray','lightgray']
    ax.bar(categories, counts, color=bar_colors, edgecolor='k', lw=0.5)
    ax.set_title('E/I assignment by tier')
    ax.set_ylabel('# neurons')
    for i, v in enumerate(counts):
        if v > 0:
            ax.text(i, v+0.3, str(v), ha='center', va='bottom', fontsize=9)

    # [1,0] P/Q ratio histogram (log scale)
    ax = axes[1, 0]
    ratio = P / (Q + 1e-8)
    ratio_log = np.log10(ratio + 1e-8)
    ax.hist(ratio_log, bins=50, color='steelblue', alpha=0.8, edgecolor='k', lw=0.3)
    ax.axvline(0, color='k', ls='--', lw=1, label='P=Q (ambiguous)')
    ax.set_xlabel('log10(P_j / Q_j)')
    ax.set_ylabel('count')
    ax.set_title('P/Q ratio distribution\n(log10 scale)')
    ax.legend(fontsize=8)

    # [1,1] E/I agreement with truth per tier (if available)
    ax = axes[1, 1]
    if has_truth and E_true is not None:
        true_A_sum = sum(A_true[m] for m in range(M))
        true_col   = true_A_sum.sum(axis=0)
        true_sigma = np.sign(true_col)
        tier_labels = ['Tier1\n(conf)', 'Tier2\n(tent)', 'Tier3\n(undet)']
        tier_masks  = [kappa > args.khi,
                       (kappa >= args.klo) & (kappa <= args.khi),
                       kappa < args.klo]
        agreements  = []
        ns          = []
        for tm in tier_masks:
            if tm.sum() > 0:
                agreements.append(
                    float(np.mean(sigma[tm] == true_sigma[tm])))
                ns.append(tm.sum())
            else:
                agreements.append(0.0)
                ns.append(0)
        bars = ax.bar(tier_labels, agreements,
                      color=['steelblue','orange','gray'],
                      edgecolor='k', lw=0.5)
        ax.axhline(0.5, color='r', ls='--', lw=1, label='chance=0.5')
        ax.set_title('E/I agreement with truth per tier')
        ax.set_ylabel('Fraction correct')
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
        for i, (v, n) in enumerate(zip(agreements, ns)):
            ax.text(i, v+0.02, f'{v:.3f}\n(n={n})',
                    ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No truth available',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('E/I agreement with truth')

    # [1,2] Summary text
    ax = axes[1, 2]
    ax.axis('off')
    lines = [
        'PRISM Stage 4 — E/I Summary',
        '',
        f'λ4 = {meta["lam_ei"]:.1e}   ei_warmup={meta["ei_warmup"]}',
        f'κhi={args.khi}   κlo={args.klo}',
        '',
        f'Tier 1 (confident): {n1:4d}',
        f'  Excitatory:       {n_exc:4d}',
        f'  Inhibitory:       {n_inh:4d}',
        f'Tier 2 (tentative): {n2:4d}',
        f'Tier 3 (undecided): {n3:4d}',
        '',
        f'κ mean   = {kappa.mean():.4f}',
        f'κ median = {np.median(kappa):.4f}',
        f'κ max    = {kappa.max():.4f}',
        '',
    ]
    if has_truth and E_true is not None:
        if tier_masks[0].sum() > 0:
            lines.append(
                f'Truth agreement (T1) = {agreements[0]:.4f}')
        dale_viol = sum(
            1 for j in range(N)
            if fA_hat[:, :, j].flatten().max() > 0
            and fA_hat[:, :, j].flatten().min() < 0)
        lines.append(f'Dale violations in A_hat: {dale_viol}')
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            va='top', fontsize=9, family='monospace')

    fig3.tight_layout()
    p3 = os.path.join(out_dir, f'{args.dataName}_s4_canvas3.png')
    fig3.savefig(p3)
    print(f'  display {p3}')
    plt.close(fig3)


if __name__ == '__main__':
    main()
