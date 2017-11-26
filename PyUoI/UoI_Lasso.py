#!/usr/bin/env python

import h5py
import os
import time
from optparse import OptionParser
import pdb

import numpy as np
import sklearn.linear_model as lm
from tqdm import trange

from PyUoI import lasso_admm as admm

__authors__ = "Alex Bujan, Kris Bouchard"

np.seterr(invalid='ignore')


def main():
    usage = '%prog [options]'
    parser = OptionParser(usage)

    parser.add_option("--bgdOpt", type="int", default=1,
                      help="bagging options")
    parser.add_option("--initrnd", type="int", default=10,
                      help="number of initial randomizations")
    parser.add_option("--nrnd", type="int", default=10,
                      help="number of randomizations for bagged estimates")
    parser.add_option("--nbootS", type="int", default=48,
                      help="number of bootstraps for model selection")
    parser.add_option("--nbootE", type="int", default=48,
                      help="number of bootstraps for model estimation")
    parser.add_option("--cvlfrct", type="float", default=.9,
                      help="data fraction for training during each bagged-OLS "
                           "randomization")
    parser.add_option("--rndfrct", type="float", default=.8,
                      help="data fraction used for linear regression fitting")
    parser.add_option("--rndfrctL", type="float", default=.8,
                      help="fraction of data used for Lasso fitting")
    parser.add_option("--nMP", type="int", default=48,
                      help="number of regularization parameters to evaluate")
    parser.add_option("-o", default=os.path.join(os.getcwd(), 'results.h5'),
                      type="string", help="hdf5 file to store the results")
    parser.add_option("-i", type="string", default='data.h5',
                      help="hdf5 file containing input data (X,y pairs)")
    parser.add_option("--verbose", action='store_true', dest='verbose',
                      help="print information in the terminal")
    parser.add_option("--seed", type="int", default=1234,
                      help="initial seed for pseudo-random number generator")
    parser.add_option("--with-admm", action='store_true', dest='with_admm',
                      help="use ADMM solver")
    parser.add_option("--dtype", type="string", default='f4',
                      help="float type to be used. Options: 'f4' and 'f8' bit")
    parser.add_option("--ncov", type="int", default=-1,
                help="number of covariates in the design matrix (optional)")
    parser.add_option("--nsamp", type="int", default=-1,
                      help="number of samples in the design matrix (optional)")
    parser.add_option("--n_minibatch", type="int", default=10,
                help="number of minibatches used with partial fit (optional)")

    (options, args) = parser.parse_args()

    if options.verbose:
        verbose = True
    else:
        verbose = False

    if options.with_admm:
        with_admm = True
    else:
        with_admm = False

    if options.nsamp == -1 or options.ncov == -1:
        nsamp = None
        ncov = None
    else:
        nsamp = options.nsamp
        ncov = options.ncov

    BoLASSO_BgdOLS(bgdOpt=options.bgdOpt, nrnd=options.nrnd,
                   initrnd=options.initrnd, nbootE=options.nbootE,
                   cvlfrct=options.cvlfrct, rndfrct=options.rndfrct,
                   nMP=options.nMP, inputFile=options.i, outputFile=options.o,
                   verbose=verbose, seed=options.seed, nbootS=options.nbootS,
                   rndfrctL=options.rndfrctL, with_admm=with_admm,
                   dtype=options.dtype, nsamp=nsamp, ncov=ncov,
                   n_minibatch=options.n_minibatch)


def lasso_sweep(X, y, lambdas, nboots, m_frac, with_admm, n_minibatch, desc=''):

    '''
    Create arrays to collect results
    '''
    B = np.zeros((nboots, X.shape[1], len(lambdas)), dtype=np.float32)
    R2m = np.zeros((nboots, len(lambdas)), dtype=np.float32)

    for c in trange(nboots, desc=desc):

        inds = np.random.permutation(len(X))
        train = inds[:m_frac]
        test = inds[m_frac:]

        for i, lamb in enumerate(lambdas):
            # train
            if not with_admm:
                try:
                    outLas = lm.Lasso(alpha=lamb)
                    outLas.fit(X[train], y[train]-y[train].mean())
                except:
                    outLas = lm.SGDRegressor(penalty='l1', alpha=lamb)
                    for j in range(n_minibatch):
                        minibatch = range(j, len(X), n_minibatch)
                        outLas.partial_fit(X[minibatch],
                                           y[minibatch]-y[minibatch].mean())
                B[c, :, i] = outLas.coef_
            else:
                B[c, :, i] = admm.lasso_admm(X[train],
                                (y[train]-y[train].mean())[..., np.newaxis],
                                alpha=lamb)[0]
            # test
            #pdb.set_trace()
            yhat = X[test].dot(B[c, :, i])
            r = np.corrcoef(yhat, y[test]-y[test].mean())
            R2m[c, i] = r[1, 0]**2

    return B, R2m


def BoLASSO_BgdOLS(inputFile, outputFile, bgdOpt=1, nrnd=10, initrnd=10,
                   nbootE=48, cvlfrct=.9, rndfrct=.8, nMP=48, verbose=True,
                   seed=1234, comm=None, nbootS=48, rndfrctL=.8,
                   with_admm=False, dtype='f4', nsamp=None, ncov=None,
                   n_minibatch=10):
    """Performs boostrap Lasso with bagged OLS (a.k.a. BoLBO)

    Parameters
    ----------

    inputFile     : hdf5 file with input data (X,y pairs)
    outputFile    : hdf5 file to store the results
    bgdOpt        : bagging options
    nrnd          : number of randomizations for bagging
    initrnd       : number of initial randomizations
    nbootE        : number of bootstrap samples for bagging (estimation)
    nbootS        : number of bootstrap samples for bolasso (selection)
    cvlfrct       : data fraction for training for CV of bagged-OLS
    rndfrct       : data fraction used for OLS fitting
    rndfrctL      : fraction of data used for Lasso fitting
    nMP           : number of regularization parameters to evaluate
    verbose       : print information in the terminal
    seed          : initial seed for pseudo-random number generator
    with_admm     : use ADMM solver for Lasso


    Notes
    -----

    The final parameters are estimated using linear regression with bagging.
    Two options are currently supported:

    1) for each bootstrap sample, choose lambda that optimized performance,
    store those weights, and take the mean of these model parameters across
    bootstrap samples as bagged estimate

    2) choose the single lamdba that gave the best average performance across
    bootstrap samples, and take the mean of model parameters associated with
    this lambda as bagged estimate

    The model estimation is repeated nrnd number of times

    """

    '''
    Data
    '''

    print("\nLoading the data with h5py ...")
    start_load_time = time.time()

    with h5py.File(inputFile, 'r') as f:
        X = f['data/X'].value.astype(dtype)
        y = f['data/y'].value.astype(dtype)
        m, n = f['data/X'].shape

    end_loadTime = time.time()-start_load_time
    print("\nData loaded in %.4f seconds" % end_loadTime)

    '''
    Timer and info
    '''
    maxBoot = np.maximum(nbootE, nbootS)
    print('\nBoLBO analysis initialized')
    print('----------------------------')
    print('\t*No lambda elements    : \t%i' % nMP)
    print('\t*Max. No iterations    : \t%i' % maxBoot)
    print('\t*No model dimensions   : \t%i' % n)
    print('\t*No model samples      : \t%i' % m)
    start_compTime = time.time()

    """
    ===============
    Model Selection
    ===============
    """

    '''
    Lambda vector for initial coarse sweep
    '''
    if nMP == 1:
        lamb0 = np.array([1.0])
    else:
        lamb0 = np.logspace(-3, 3, nMP, dtype=np.float64)

    '''
    Generate ids for separating training and testing blocks
    '''
    np.random.seed(seed)
    m_frac = int(round(rndfrctL*m))

    '''
    Lasso
    '''
    start_las1Time = time.time()

    B0, R2m0 = lasso_sweep(X, y, lamb0, initrnd, m_frac, with_admm, n_minibatch,
                           desc='initial fitting')

    end_las1Time = time.time()-start_las1Time

    '''
    Compute new Lambda vector for dense sweep
    '''
    Mt = np.fix(1e4*np.mean(R2m0, 0))
    Lids = np.where(Mt == np.max(np.ma.masked_invalid(Mt)))[0]
    v = lamb0[Lids[len(Lids)//2]]
    dv = 10**(np.floor(np.log10(v))-1)

    if nMP == 1:
        lambL = np.array([1.0])
    else:
        lambL = np.linspace(v-5*dv, v+5*dv, nMP)

    start_las2Time = time.time()

    B, R2m = lasso_sweep(X, y, lambL, nbootS, m_frac, with_admm, n_minibatch,
                         desc='boosting')

    end_las2Time = time.time()-start_las2Time

    '''
    Compute family of supports
    '''
    sprt = np.ones((nMP, n))*np.nan
    # for each regularization parameter
    for i in range(nMP):
        # for each bootstrap sample
        for r in range(nbootS):
            tmp_ids = np.where(B[r, :, i] != 0)[0]
            if r == 0:
                intv = tmp_ids
            # bolasso support is intersection of supports across
            # bootstrap samples
            intv = np.intersect1d(intv, tmp_ids).astype('int')
        ## BOLASSO support for regularization parameter
        sprt[i, :len(intv)] = intv

    """
    ================
    Model Estimation
    ================
    """

    m_frac = int(round(cvlfrct*m))
    L_frac = int(round(rndfrct*m_frac))

    '''
    Create arrays to collect final results
    '''
    Bgd = np.zeros((nrnd, n), dtype=np.float32)
    R2 = np.zeros(nrnd, dtype=np.float32)
    rsd = np.zeros((nrnd, m-m_frac), dtype=np.float32)
    bic = np.zeros(nrnd, dtype=np.float32)

    start_bolsTime = time.time()

    for cc in trange(nrnd, desc='Model Estimation', total=nrnd):
        '''
        Create arrays to collect the results
        '''
        Bgols_B = np.zeros((nbootE, n, nMP), dtype=np.float32)
        Bgols_R2m = np.zeros((nbootE, nMP), dtype=np.float32)

        '''
        Generate ids for separating training and testing blocks
        '''
        inds = np.random.permutation(m)
        L = inds[:m_frac]
        T = inds[m_frac:]

        # number of bootstrap samples for aggregation
        for c in trange(nbootE, desc='boostE'):
            inds = np.random.permutation(m_frac)
            train = inds[:L_frac]
            test = inds[L_frac:]

            # for each regularization parameter
            for i in range(nMP):
                """
                Select support
                """
                sprt_ids = sprt[i][~np.isnan(sprt[i])].astype('int')
                zdids = np.setdiff1d(np.arange(n), sprt_ids)
                
                if len(sprt_ids) > 0:
                    '''
                    #Linear regression
                    '''
                    # train
                    try:
                        outLR = lm.LinearRegression()
                        outLR.fit(X[L[train]][:, sprt_ids],
                                  y[L[train]]-y[L[train]].mean())
                    except:
                        outLR = lm.SGDRegressor(penalty='none')
                        for j in range(n_minibatch):
                            minibatch = train[j::n_minibatch]
                            # print '\nols - mini-batch %i/%i size: %i'%(j,n_minibatch,len(minibatch))
                            outLR.partial_fit(
                                X[L[minibatch]][:, sprt_ids],
                                y[L[minibatch]]-y[L[minibatch]].mean())
                    rgstrct = outLR.coef_
                    Bgols_B[c, sprt_ids, i] = rgstrct
                    Bgols_B[c, zdids, i] = 0

                    # test
                    yhat = X[L[test]].dot(Bgols_B[c, :, i])
                    r = np.corrcoef(yhat, y[L[test]]-y[L[test]].mean())
                    Bgols_R2m[c, i] = r[1, 0]**2
                else:
                    print('%i parameters were selected for lambda: %.4f' % \
                          (sprt_ids.sum(), lambL[i]))

        """
        Bagging
        """
        if bgdOpt == 1:
            # for each bootstrap sample, find regularization parameter
            # that gave best results
            v = np.max(Bgols_R2m, 1)
            ids = np.where(Bgols_R2m == v.reshape((nbootE, 1)))
            btmp = np.zeros((nbootE, n))
            # this loop can probably be removed
            for kk in range(nbootE):
                # some maxima are constant across a 
                # variable range of parameter values
                ids_kk = ids[1][np.where(ids[0] == kk)]
                # collect best results
                btmp[kk] = Bgols_B[kk, :, ids_kk[int(len(ids_kk)/2)]]

            # bagged estimates of model parameters
            Bgd[cc] = np.median(btmp, 0)
            del btmp
        else:
            # for the average across bootstrap samples,
            # find regularization parameter that gave best results
            mean_Bgols_R2m = np.mean(Bgols_R2m, 0)
            v = np.max(mean_Bgols_R2m)
            ids = np.where(mean_Bgols_R2m == v)[0]

            # bagged estimates of model parameters
            Bgd[cc] = np.median(Bgols_B[:, :, ids], 0)

        yhat = X[T].dot(Bgd[cc])
        r = np.corrcoef(yhat, y[T]-y[T].mean())
        R2[cc] = r[1, 0]**2
        rsd[cc] = (y[T]-y[T].mean())-yhat
        bic[cc] = (m-m_frac)*np.log(np.dot(rsd[cc], rsd[cc])*1./(m-m_frac))+\
                   np.log(m-m_frac)*n

    end_bolsTime = time.time()-start_bolsTime
    end_compTime = time.time()-start_compTime
    start_saveTime = time.time()

    """
    Store results
    """

    with h5py.File(outputFile, 'w') as f:

        f.attrs['bgdOpt'] = bgdOpt
        f.attrs['nrnd'] = nrnd
        f.attrs['cvlfrct'] = cvlfrct
        f.attrs['rndfrct'] = rndfrct
        f.attrs['rndfrctL'] = rndfrctL
        f.attrs['nbootE'] = nbootE
        f.attrs['nbootS'] = nbootS
        f.attrs['nMP'] = nMP
        f.attrs['initrnd'] = initrnd
        f.attrs['seed'] = seed
        f.attrs['with_admm'] = str(admm)

        f.attrs['n_samp'] = m
        f.attrs['n_cov'] = n

        f.attrs['loadTime'] = end_loadTime
        f.attrs['las1Time'] = end_las1Time
        f.attrs['las2Time'] = end_las2Time
        f.attrs['bolsTime'] = end_bolsTime
        f.attrs['compTime'] = end_compTime

        g1 = f.create_group('/lasso')
        g1.create_dataset(name='B0', data=B0, compression='gzip')
        g1.create_dataset(name='R2m0', data=R2m0, compression='gzip')
        g1.create_dataset(name='lamb0', data=lamb0, compression='gzip')
        g1.create_dataset(name='B', data=B, compression='gzip')
        g1.create_dataset(name='R2m', data=R2m, compression='gzip')
        g1.create_dataset(name='lambL', data=lambL, compression='gzip')

        g2 = f.create_group('/bolbo')
        g2.create_dataset(name='sprt', data=sprt, compression='gzip')
        g2.create_dataset(name='Bgd', data=Bgd, compression='gzip')
        g2.create_dataset(name='R2', data=R2, compression='gzip')
        g2.create_dataset(name='rsd', data=rsd, compression='gzip')
        g2.create_dataset(name='bic', data=bic, compression='gzip')

        end_saveTime = time.time() - start_saveTime
        f.attrs['saveTime'] = end_saveTime

    print('\nBoLBO analysis completed')
    print('------------------------')
    print('\t*Results stored in %s' % outputFile)
    print("\t*BoLBO times:")
    print("\t\t-load time: %.4f" % end_loadTime)
    print("\t\t-las1 time: %.4f" % end_las1Time)
    print("\t\t-las2 time: %.4f" % end_las2Time)
    print("\t\t-bols time: %.4f" % end_bolsTime)
    print("\t\t-comp time: %.4f" % end_compTime)
    print("\t\t-save time: %.4f" % end_saveTime)


if __name__ == '__main__':
    main()
