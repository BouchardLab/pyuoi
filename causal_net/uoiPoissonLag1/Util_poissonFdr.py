#...!...!..................
import os
import numpy as np

    
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

