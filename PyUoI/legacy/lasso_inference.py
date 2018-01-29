# !/usr/bin/env python
import h5py
import numpy as np
import scipy.stats as stats
import sklearn.linear_model as lm
from scipy import optimize, linalg
from statsmodels.robust.scale import mad

"""
Authors : Alex Bujan (adapted from Sharmodeep Bhattacharyya)
Date    : 08/12/2015
"""


def SoftThreshold(x, lambd):
    '''
    # Standard soft thresholding
    '''
    if x > lambd:
        return x - lambd
    else:
        if x < -lambd:
            return x + lambd
        else:
            return 0


def InverseLinftyOneRow(sigma, i, mu, maxiter=50, threshold=1e-2):
    p = sigma.shape[0]
    rho = np.max(np.abs(np.delete(sigma[i], i))) * 1. / sigma[i, i]
    mu0 = rho * 1. / (1 + rho)
    beta = np.zeros(p)
    if mu >= mu0:
        beta[i] = (1 - mu0) * 1. / sigma[i, i]
        returndict = dict({"optsol": beta, "iter": 0})
        return returndict
    else:
        diff_norm2 = 1
        last_norm2 = 1
        count = 1
        count_old = 1
        beta[i] = (1 - mu0) * 1. / sigma[i, i]
        beta_old = np.copy(beta)
        sigma_tilde = np.copy(sigma)
        for i in range(sigma_tilde.shape[0]):
            sigma_tilde[i, i] = 0
        vs = np.dot(-sigma_tilde, beta)
        while count <= maxiter and diff_norm2 >= (threshold * last_norm2):
            for j in range(p):
                oldval = beta[j]
                v = vs[j]
                if j == i:
                    v += 1
                beta[j] = SoftThreshold(v, mu) * 1. / sigma[j, j]
                if oldval != beta[j]:
                    vs = vs + (oldval - beta[j]) * sigma_tilde[:, j]
            count += 1
            if count == (2 * count_old):
                d = beta - beta_old
                diff_norm2 = np.sqrt(np.sum(d * d))
                last_norm2 = np.sqrt(np.sum(beta * beta))
                count_old = count
                beta_old = np.copy(beta)
                if count > 10:
                    vs = np.dot(-sigma_tilde, beta)
        returndict = dict({"optsol": beta, "iter": count})
        return returndict


def InverseLinfty(sigma, n, resol=1.5, mu=None, maxiter=50, threshold=1e-2,
                  verbose=True):
    p = sigma.shape[0]
    M = np.zeros((p, p))
    beta = np.zeros(p)
    isgiven = True
    if mu == None:
        mu = (1. / np.sqrt(n)) * stats.norm.ppf(1 - (.1 / (p ** 2)))
        isgiven = False
    for i in range(p):
        mu_stop = False
        try_no = 1
        incr = 0
        while not mu_stop and try_no < 10:
            last_beta = np.copy(beta)
            output = InverseLinftyOneRow(sigma, i, mu, maxiter=maxiter,
                                         threshold=threshold)
            beta = output["optsol"]
            count = output["iter"]
            if isgiven:
                mu_stop = True
            else:
                if try_no == 1:
                    if count == (maxiter + 1):
                        incr = 1
                        mu = mu * resol
                    else:
                        incr = 0
                        mu = mu * 1. / resol
                else:
                    if incr == 1 and count == (maxiter + 1):
                        mu = mu * resol
                    elif incr == 1 and count < (maxiter + 1):
                        mu_stop = True
                    elif incr == 0 and count < (maxiter + 1):
                        mu = mu * 1. / resol
                    elif incr == 0 and count == (maxiter + 1):
                        mu = mu * resol
                        beta = np.copy(last_beta)
                        mu_stop = True
            try_no += 1
        M[i] = beta
    return M


def NoiseSd(yh, A, n):
    ynorm = np.sqrt(n) * (yh * 1. / np.sqrt(np.diag(A)))
    sd_hat0 = mad(ynorm)
    zeros = np.abs(ynorm) < 3 * sd_hat0
    y2norm = np.sum(yh[zeros] ** 2)
    Atrace = np.sum(np.diag(A)[zeros])
    sd_hat1 = np.sqrt(n * y2norm / Atrace)
    ratio = sd_hat0 * 1. / sd_hat1
    if np.max((ratio, 1. / ratio)) > 2:
        print("Warning: Noise estimate problematic")
    s0 = np.sum(zeros == False)
    return dict({"sd": sd_hat1, "nz": s0})


def Lasso(X, y, lambd=None, intercept=True):
    '''
    # Compute the Lasso estimator:
    # - If lambd is given, use standard Lasso
    # - If lambd is not given, use square root Lasso
    '''
    p = X.shape[1]
    n = X.shape[0]
    if lambd == None:
        lambd = np.sqrt(stats.norm.ppf(1 - (.1 / p)) * 1. / n)
        print('\nSQRT Lasso with Lambda: %.2f!' % lambd)

        def sqrtLasso(coef):
            L_1_penalty = np.sum(np.abs(coef[intercept * 1:]))
            if intercept:
                RS = y - coef[0] - np.dot(X, coef[1:])
            else:
                RS = y - np.dot(X, coef)
            RSS = np.dot(RS.T, RS)
            return np.sqrt(RSS * 1. / n) + lambd * L_1_penalty

        coef_init = np.random.uniform(-.5, .5, size=p + intercept * 1)
        outLas = optimize.fmin_bfgs(sqrtLasso, x0=coef_init)
        # Objective : sqrt(RSS/n) + sqrt(lambda/n)*penalty
        return outLas
    else:
        print('\nSciKit-Learn Lasso with Lambda: %.2f!' % lambd)
        outLas = lm.Lasso(alpha=lambd, fit_intercept=intercept)
        outLas.fit(X, y)
        # Objective :1/2 RSS/n + lambda*penalty
        if intercept:
            return np.hstack((outLas.intercept_, outLas.coef_))
        else:
            return outLas.coef_


def SSLasso(X, y, alpha=0.05, lambd=None, mu=None, \
            intercept=True, resol=1.3, maxiter=50, \
            threshold=1e-2, verbose=True):
    '''
    # Compute confidence intervals and p-values.
    #
    # Args:
    #   X     :  design matrix
    #   y     :  response
    #   alpha :  significance level
    #   lambda:  Lasso regularization parameter (if null, fixed by sqrt lasso)
    #   mu    :  Linfty constraint on M (if null, searches)
    #   resol :  step parameter for the function that computes M
    #   maxiter: iteration parameter for computing M
    #   threshold : tolerance criterion for computing M
    #   verbose : verbose?
    #
    # Returns:
    #   noise.sd: Estimate of the noise standard deviation
    #   norm0   : Estimate of the number of 'significant' coefficients
    #   coef    : Lasso estimated coefficients
    #   unb.coef: Unbiased coefficient estimates
    #   low.lim : Lower limits of confidence intervals
    #   up.lim  : upper limit of confidence intervals
    #   pvals   : p-values for the coefficients
    '''

    p = X.shape[1]
    n = X.shape[0]
    pp = p
    col_norm = 1. / np.sqrt(np.diag(np.dot(X.T, X)) * 1. / n)
    X = np.dot(X, np.diag(col_norm))
    htheta = Lasso(X, y, lambd=lambd, intercept=intercept)
    if intercept:
        Xb = np.hstack((np.ones(n)[:, np.newaxis], X))
        col_norm = np.hstack((1, col_norm))
        pp += 1
    else:
        Xb = X

    sigma_hat = np.dot(Xb.T, Xb) * 1. / n
    if n >= 2 * p:
        tmp = linalg.eig(sigma_hat)
        tmp = np.min(tmp[0]) * 1. / np.max(tmp[0])
    else:
        tmp = 0
    if n >= 2 * p and tmp >= 1e-4:
        M = linalg.inv(sigma_hat)
    else:
        M = InverseLinfty(sigma_hat, n, resol=resol, mu=mu, \
                          maxiter=maxiter, threshold=threshold, \
                          verbose=verbose);

    unbiased_Lasso = htheta + np.dot(np.dot(M, Xb.T), y - \
                                     np.dot(Xb, htheta)) * 1. / n

    A = np.dot(np.dot(M, sigma_hat), M.T)
    noise = NoiseSd(unbiased_Lasso, A, n)
    s_hat = noise["sd"]

    interval_sizes = stats.norm.ppf(1. - (alpha * 1. / 2)) * s_hat * np.sqrt(
        np.diag(A)) * 1. / np.sqrt(n)

    if lambd == None:
        lambd = s_hat * np.sqrt(stats.norm.ppf(1. - (.1 / p)) * 1. / n)
    addlength = np.zeros(pp)
    MM = np.dot(M, sigma_hat) - np.identity(pp)
    for i in range(pp):
        effectivemuvec = np.sort(np.abs(MM[i]))[::-1]
        effectivemuvec = effectivemuvec[:noise['nz'] - 1]
        addlength[i] = np.sqrt(np.sum(effectivemuvec * effectivemuvec)) * lambd

    htheta = htheta * col_norm
    unbiased_Lasso = unbiased_Lasso * col_norm
    interval_sizes = interval_sizes * col_norm
    addlength = addlength * col_norm
    ##
    col_norm1 = col_norm[intercept * 1:]

    htheta = htheta[intercept * 1:]
    unbiased_Lasso = unbiased_Lasso[intercept * 1:]
    interval_sizes = interval_sizes[intercept * 1:]
    addlength = addlength[intercept * 1:]
    p_vals = 2 * (1 - stats.norm.cdf(np.sqrt(n) * np.abs(unbiased_Lasso) \
                                     / (s_hat * col_norm[
                                                intercept * 1:] * np.sqrt(
        np.diag(A[intercept * 1:, intercept * 1:])))))

    ###New###
    beta0 = unbiased_Lasso[0]
    sel_coef = np.zeros(p)
    sel_coef[p_vals < alpha] = unbiased_Lasso[p_vals < alpha]
    est_beta = sel_coef
    ######

    returnDict = dict({"noise_sd": s_hat,
                       "norm0": noise['nz'],
                       "coef": htheta,
                       "unb_coef": unbiased_Lasso,
                       "low_lim": unbiased_Lasso - interval_sizes - addlength,
                       "up_lim": unbiased_Lasso + interval_sizes + addlength,
                       "pvals": p_vals,
                       "est_coef": sel_coef,
                       #                        "beta0"     : beta0,
                       "col_norm1": col_norm1
                       })
    return returnDict


def SSLasso_rep(inputFile, nrnd=100, alpha=0.05, lambd=None, mu=None, \
                intercept=True, resol=1.3, maxiter=50, \
                threshold=1e-2, verbose=True, n_jobs=1, rndfrct=.9):
    with h5py.File(inputFile, 'r') as f:
        X = np.copy(f['data/X'].value)
        y = np.copy(f['data/y'].value)

    n = X.shape[0]
    p = X.shape[1]

    n_frac = int(round(rndfrct * n))

    B = np.zeros((nrnd, p))
    RSD = np.zeros((nrnd, n - n_frac))
    R2 = np.zeros(nrnd)
    BIC = np.zeros(nrnd)

    '''
    Select lambda with CV
    '''
    if lambd == None:
        cvLasso = lm.LassoCV(cv=5, n_jobs=n_jobs)
        cvLasso.fit(X, y)
        lambd = cvLasso.alpha_

    for c in range(nrnd):
        rndsd = np.random.permutation(n)
        L = rndsd[:n_frac]
        T = np.setdiff1d(np.arange(n), L)

        # Learning
        deb_l = SSLasso(X[L, :], y[L], alpha, lambd, mu, intercept, resol, \
                        maxiter, threshold, verbose=False)

        est_beta = deb_l['est_coef']
        #        beta0       = deb_l['beta0']
        col_norm1 = deb_l['col_norm1']

        # Testing
        #        if intercept:
        #            yhat = np.dot(X[T,:],est_beta) + beta0
        #        else:
        #            yhat = np.dot(X[T,:],est_beta)
        yhat = np.dot(np.dot(X[T, :], np.diag(1. / col_norm1)), est_beta)

        r = np.corrcoef(yhat, y[T] - np.mean(y[T]))
        r2 = r[1, 0] ** 2
        rsd = (y[T] - np.mean(y[T])) - yhat
        bic = len(T) * np.log(np.dot(rsd, rsd) * 1. / len(T)) + np.log(
            len(T)) * p

        B[c] = est_beta
        RSD[c] = rsd
        R2[c] = r2
        BIC[c] = bic

    return dict({"B": B,
                 "rsd": RSD,
                 "R2": R2,
                 'bic': BIC})
