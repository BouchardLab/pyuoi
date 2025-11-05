#...!...!..................
import os
import numpy as np

if 0:  # pop-up canvas
    import matplotlib as mpl
    mpl.use('TkAgg')
    
#...!...!..................
def qa_Bfit(B_truth, B_fit):
    assert B_truth.shape == B_fit.shape
    bterm_res = B_fit - B_truth
    bterm_mean = np.mean(bterm_res)
    bterm_std = np.std(bterm_res)
    N = bterm_res.shape[0]
    bterm_se_s = 0.
    if N > 1:
        bterm_se_s = bterm_std / np.sqrt(2 * (N - 1))
    print('B term: mean=%.2f    std=%.3f +/- %.3f' % (bterm_mean, bterm_std, bterm_se_s))

    out = {
        'tval': B_truth,
        'fval': B_fit,
        'res_mean': bterm_mean,
        'res_std': bterm_std
    }
    return out


#...!...!..................
def qa_Afit(A_truth, A_fit):
    assert A_truth.shape == A_fit.shape
    assert A_truth.shape[0] == A_truth.shape[1]

    M = A_truth.shape[0]
    diag_m = np.eye(M, dtype=bool)
    out = {}

    # --- Diagonal elements
    diag_tval = A_truth[diag_m]
    diag_fval = A_fit[diag_m]
    res_diag = diag_fval - diag_tval
    diag_mean = np.mean(res_diag) if res_diag.size > 0 else 0
    diag_std = np.std(res_diag) if res_diag.size > 0 else 0
    out['diag'] = {
        'tval': diag_tval,
        'fval': diag_fval,
        'res_mean': diag_mean,
        'res_std': diag_std
    }

    # --- Off-diagonal elements (Excitatory and Inhibitory)
    for name in ['inh','exc']:
        if name == 'exc':
            true_m = (A_truth > 0) & (~diag_m)
            pred_m = (A_fit > 0) & (~diag_m)
        else:  # inh
            true_m = (A_truth < 0) & (~diag_m)
            pred_m = (A_fit < 0) & (~diag_m)

        tp_m = true_m & pred_m
        tval = A_truth[tp_m]
        fval = A_fit[tp_m]
        res = fval - tval
        res_mean = np.mean(res) if res.size > 0 else 0
        res_std = np.std(res) if res.size > 0 else 0

        TP = np.sum(tp_m)
        FP = np.sum((~true_m) & pred_m)
        FN = np.sum(true_m & (~pred_m))

        out[name] = {
            'tval': tval,
            'fval': fval,
            'res_mean': res_mean,
            'res_std': res_std,
            'TP': TP,
            'FP': FP,
            'FN': FN
        }

    # Keep printing for compatibility
    for name, stats in [('exc', out['exc']), ('inh', out['inh'])]:
        res_N = stats['tval'].shape[0]
        se_s = 0.
        if res_N > 1:
            se_s = stats['res_std'] / np.sqrt(2 * (res_N - 1))

        print('A type=%s  TP:%d  FP:%d FN:%d  TP: mean=%.2f   std=%.3f +/- %.3f'%(name, stats['TP'], stats['FP'], stats['FN'], stats['res_mean'], stats['res_std'], se_s))

    diag_stats = out['diag']
    diag_N = diag_stats['tval'].shape[0]
    diag_se_s = 0.
    if diag_N > 1:
        diag_se_s = diag_stats['res_std'] / np.sqrt(2 * (diag_N - 1))

    print('A diag: mean=%.2f    std=%.3f +/- %.3f'%(diag_stats['res_mean'], diag_stats['res_std'], diag_se_s))

    return out


#...!...!..................
def plot_edges_correl(args, qaD, outName):
    #import matplotlib as mpl
    #mpl.use('TkAgg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.))

    colors = {'exc': 'red', 'inh': 'blue', 'diag': 'brown', 'bterm': 'green'}

    for i, name in enumerate(['inh', 'exc', 'diag', 'bterm']):
        ax = axes[i]
        stats = qaD[name]
        color = colors[name]

        if stats['tval'].size > 0:
            ax.scatter(stats['tval'], stats['fval'], alpha=0.5, s=8, c=color)
            # Add y=x line
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
            ax.set_aspect('equal', 'box')
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            # Add center of gravity cross
            mean_t = np.mean(stats['tval'])
            mean_f = np.mean(stats['fval'])
            ax.plot(mean_t, mean_f, '+', c='black', markersize=20, markeredgewidth=3)

        ax.grid()
        ax.set_title(name.capitalize())
        ax.set_xlabel('True Value')
        if i == 0:
            ax.set_ylabel('Fitted Value')

        # Add mean and std text
        mean_val = stats['res_mean']
        std_val = stats['res_std']
        ax.text(0.95, 0.05, f'TP N={stats["tval"].shape[0]}  \nmean={mean_val:.3f}\nstd={std_val:.3f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

        if name in ['exc', 'inh']:
            tp_val = stats['TP']
            fp_val = stats['FP']
            fn_val = stats['FN']
            ax.text(0.05, 0.95, f'TP={tp_val}\nFP={fp_val}\nFN={fn_val}',
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

    fig.suptitle(f'Fitted  {outName}  %d samples'%(args.samples))
    fig.tight_layout()
    plotF = os.path.join(args.out, f'{outName}_corr.png')
    fig.savefig(plotF)
    print(f'Saved   display  {plotF}')
    plt.show()


#...!...!..................
def plot_eigenvalues(args, A_truth, A_fit, outName):
    import matplotlib.pyplot as plt
    eigT = np.linalg.eigvals(A_truth)
    eigF = np.linalg.eigvals(A_fit)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    reT = np.real(eigT)
    imT = np.imag(eigT)
    ax.scatter(reT, imT, color='blue', marker='o', label='True', s=10)

    reF = np.real(eigF)
    imF = np.imag(eigF)
    ax.scatter(reF, imF, color='red', marker='o', facecolors='none', label='fit', s=20)

    ax.set_ylim(-0.1,)
    #ax.set_xlim(right=1)
    ax.axhline(0, linestyle='--', color='k', linewidth=1)
    ax.axvline(0, linestyle='--', color='k', linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.legend()
    ax.set_title(f'Eigenvalues for {outName}  %d samples'%(args.samples))

    plotF = os.path.join(args.out, f'{outName}_eigen.png')
    fig.savefig(plotF)
    print(f'Saved   display  {plotF}')
    plt.show()


#...!...!..................
def plot_loss(args, A_truth, A_fit, outName, l1_loss_sel, loss_skip=0):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: L1 loss vs. iteration
    if l1_loss_sel.shape[1] > 0:
        mask = l1_loss_sel[0] >= loss_skip
        axes[0].plot(l1_loss_sel[0, mask], l1_loss_sel[1, mask], marker='o')
    #axes[0].set_yscale('log')
    axes[0].set_title('Selection L1 loss')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('L1 Loss')
    axes[0].grid(True)

    # Prepare masks
    M = A_truth.shape[0]
    diag_mask = np.eye(M, dtype=bool)
    exc_mask = (A_truth > 0) & (~diag_mask)
    inh_mask = (A_truth < 0) & (~diag_mask)

    # Panel B: Excitatory distribution
    for idx, name in enumerate(['exc', 'inh']):
        ax = axes[idx + 1]
        if name == 'exc':
            mask = exc_mask
            title = 'Excitatory edge weights'
        else:
            mask = inh_mask
            title = 'Inhibitory edge weights'

        truth_vals = A_truth[mask]
        fit_vals = A_fit[mask]
        if truth_vals.size > 0 or fit_vals.size > 0:
            ax.hist(truth_vals, bins=40, alpha=0.6, label='truth')
            ax.hist(fit_vals, bins=40, alpha=0.6, label='fit')
        ax.set_title(title)
        ax.set_xlabel('Weight')
        ax.set_ylabel('Count')
        ax.legend()

    fig.suptitle(f'Fitted  {outName}  %d samples'%(args.samples))
    fig.tight_layout()
    plotF = os.path.join(args.out, f'{outName}_loss_edges.png')
    fig.savefig(plotF)
    print('Saved   display  %s'%plotF)
    plt.show()


#...!...!..................
def extract_loss_traces(model, loss_stride):
    loss_iter = np.empty(0, dtype=np.int64)
    loss_l1 = np.empty(0, dtype=np.float64)

    if loss_stride is None:
        loss_stride = 10

    if model is not None and hasattr(model, 'loss'):
        loss = model.loss
        if loss is not None and 'l1' in loss:
            loss_l1 = np.asarray(loss['l1'], dtype=np.float64)
            loss_iter = np.arange(loss_l1.size, dtype=np.int64) * loss_stride

    return loss_iter, loss_l1

