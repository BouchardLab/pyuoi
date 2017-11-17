#!/usr/bin/env python

import os
from optparse import OptionParser

import h5py
import numpy as np
from mpi4py import MPI

from PyUoI import lasso_admm_MPI as admm

np.seterr(invalid='ignore')

__authors__ = "Alex Bujan, Kris Bouchard"


def main():
    usage = '%prog [options]'
    parser = OptionParser(usage)

    parser.add_option("--bgdOpt", type="int", default=1,
                      help="bagging options")
    parser.add_option("--nrnd", type="int", default=10,
                      help="number of randomizations for bagged estimates")
    parser.add_option("--nbootS", type="int", default=48,
                      help="number of bootstraps for model selection")
    parser.add_option("--nbootE", type="int", default=48,
                      help="number of bootstraps for model estimation")
    parser.add_option("--cvlfrct", type="float", default=.9,
                      help="data fraction for training during each bagged-OLS randomization")
    parser.add_option("--rndfrct", type="float", default=.8,
                      help="data fraction used for linear regression fitting")
    parser.add_option("--rndfrctL", type="float", default=.8,
                      help="fraction of data used for Lasso fitting")
    parser.add_option("--nMP", type="int", default=48,
                      help="number of regularization parameters to evaluate")
    parser.add_option("-o", type="string", default=os.getcwd(),
                      help="hdf5 file to store the results")
    parser.add_option("-i", type="string", default='',
                      help="hdf5 file containing input data (X,y pairs)")
    parser.add_option("--verbose", action='store_true',
                      dest='verbose', help="print information in the terminal")
    parser.add_option("--seed", type="int", default=1234,
                      help="initial seed for pseudo-random number generator")
    parser.add_option("--dtype", type="string", default='f4',
                      help="float type to be used. Options: 'f4' and 'f8' bit")
    parser.add_option("--ncov", type="int", default=-1,
                      help="number of covariates in the design matrix (optional)")
    parser.add_option("--nsamp", type="int", default=-1,
                      help="number of samples in the design matrix (optional)")
    parser.add_option("--nreaders", type="int", default=10,
                      help="number of processes to load the data (optional)")

    (options, args) = parser.parse_args()

    if options.verbose:
        verbose = True
    else:
        verbose = False

    if options.nsamp == -1 or options.ncov == -1:
        nsamp = None
        ncov = None
    else:
        nsamp = options.nsamp
        ncov = options.ncov

    BoLASSO_BgdOLS(bgdOpt=options.bgdOpt, nrnd=options.nrnd,
                   nbootE=options.nbootE,
                   cvlfrct=options.cvlfrct, rndfrct=options.rndfrct,
                   nMP=options.nMP, inputFile=options.i, outputFile=options.o,
                   verbose=verbose, seed=options.seed, nbootS=options.nbootS,
                   rndfrctL=options.rndfrctL, dtype=options.dtype, nsamp=nsamp,
                   ncov=ncov, nreaders=options.nreaders)


def split_data(m, m_frac):
    """Generate ids for separating training and testing data

    Parameters
    ----------

    m : int
        number of samples in the data
    m_frac : float
        fraction of samples used for training 
    
    Returns
    -------
    
    test_ids : array
        sample ids for testing
    train_ids : array
        sample ids for training
    """
    ids = np.random.permutation(m)
    test_ids = ids[m_frac:]
    train_ids = ids[:m_frac]
    return train_ids, test_ids


def BoLASSO_BgdOLS(inputFile, outputFile, bgdOpt=1, nrnd=10,
                   nbootE=48, cvlfrct=.9, rndfrct=.8, nMP=48, verbose=True,
                   seed=1234, comm=None, nbootS=48, rndfrctL=.9,
                   dtype='f4', nsamp=None, ncov=None, nreaders=10):
    """Boostrap Lasso with Bagged OLS (BoLBO)

    Parameters
    ----------

    - inputFile     : hdf5 file with input data (X,y pairs)
    - outputFile    : hdf5 file to store the results
    - bgdOpt        : bagging options
    - nrnd          : number of randomizations for bagging
    - initrnd       : number of initial randomizations
    - nbootE        : number of bootstrap samples for bagging (estimation)
    - nbootS        : number of bootstrap samples for bolasso (selection)
    - cvlfrct       : data fraction for training for CV of bagged-OLS
    - rndfrct       : data fraction used for OLS fitting
    - rndfrctL      : fraction of data used for Lasso fitting
    - nMP           : number of regularization parameters to evaluate
    - verbose       : print information in the terminal
    - seed          : initial seed for pseudo-random number generator

    Notes
    -----

    Model parameters are selected using linear regression with bagging:

    Options:

    1) for each bootstrap sample, choose lambda that optimized performance,
    store those weights, and take the mean of these model parameters across 
    bootstrap samples as bagged estimate

    2) choose the single lamdba that gave the best average performance across
    bootstrap samples, and take the mean of model parameters associated with
    this lambda as bagged estimate

    """

    # bytes per element
    bpe = int(dtype[1])

    '''
    Global communicator
    '''
    comm_ALL = MPI.COMM_WORLD
    size = comm_ALL.Get_size()
    rank = comm_ALL.Get_rank()

    # check if correct number of processes - TODO: make it more flexible
    maxBoot = np.maximum(nbootE, nbootS)
    minSize = nMP * maxBoot
    assert size >= minSize, ValueError('The number of processes must be '
                                       'at least %i' % minSize)

    '''
    ADMM pool size/ Create comm_ADMM by dividing global comm
    '''
    # number of processes that will be used for each ADMM analysis
    admm_psize = size // minSize
    comm_admm_id = rank // admm_psize

    comm_ADMM = comm_ALL.Split(comm_admm_id)
    admm_rank = comm_ADMM.Get_rank()

    assert admm_psize == comm_ADMM.size, ValueError(
        'Requested ADMM pool size and '
        'ADMM comm size do not match!')

    '''
    Bootstrap and MP ids
    '''
    bt_id = rank // (nMP * admm_psize)
    mp_id = np.tile(np.repeat(np.arange(nMP), admm_psize), maxBoot)[rank]
    admm_ids = np.tile(np.tile(np.arange(admm_psize), nMP), maxBoot)

    """
    Create communicator with ADMM root ranks only 
    """

    admm_root_ids = np.where(admm_ids == 0)[0].tolist()
    admm_root_grp = comm_ALL.Get_group().Incl(admm_root_ids)
    comm_ROOT = comm_ALL.Create(admm_root_grp)

    '''
    Get size of the design matrix
    '''
    if rank == 0:
        start_sizeTime = MPI.Wtime()

    if ncov == None or nsamp == None:
        if rank == 0:
            with h5py.File(inputFile, 'r') as f:
                m, n = f['data/X'].shape
        else:
            n = None
            m = None

        comm_ALL.Barrier()
        n = comm_ALL.bcast(obj=n, root=0)
        m = comm_ALL.bcast(obj=m, root=0)

    else:
        m = nsamp
        n = ncov
        if rank == 0:
            t_start_with_io = MPI.Wtime()
            print("\nData size: (%i,%i)" % (m, n))

    if rank == 0:
        end_sizeTime = MPI.Wtime() - start_sizeTime
        print("\nData size: (%i,%i)" % (m, n))
        print("\nData size read/broadcasted in %.4f seconds" % end_sizeTime)

    """
    create readers group
    """

    if rank == 0:
        print("\nInitializing communicator for data loading ...")
        print("\nNumber of loading prodesses (readers): %i" % nreaders)
        start_commTime = MPI.Wtime()

    reader_ids = np.arange(nreaders)
    readers = comm_ALL.Get_group().Incl(reader_ids)
    comm_READ = comm_ALL.Create(readers)

    if rank == 0:
        end_commTime = MPI.Wtime() - start_commTime
        print("\nCommunicator created in %.4f seconds" % (end_commTime))

    """
    Get the data
    """
    n_samp = m // nreaders
    loc_id = np.arange(m) // n_samp
    loc_id[loc_id >= nreaders] = nreaders - 1

    if rank in reader_ids:

        sids = np.where(loc_id == rank)[0]

        if rank == 0:
            print("\nLoading the data with parallel-h5py ...")
            start_loadTime = MPI.Wtime()

        #        with h5py.File(inputFile,'r',driver='mpio',comm=comm_READ) as f:
        #            f.atomic = True
        with h5py.File(inputFile, 'r') as f:
            d1 = f['data/X'][sids[0]:sids[-1] + 1].astype(dtype)
            d2 = f['data/y'][sids[0]:sids[-1] + 1].astype(dtype)

        if rank == 0:
            end_loadTime = MPI.Wtime() - start_loadTime
            print("\nData loaded in %.4f seconds" % (end_loadTime))
        d = np.hstack([d1, d2[..., np.newaxis]])
    else:
        d = None
    """
    Create window in each process
    """
    if rank == 0:
        print("\nInitializing RMA window ...")
        start_winInitTime = MPI.Wtime()

    win = MPI.Win.Create(d, comm=comm_ALL)

    if rank == 0:
        print("\nWindow created in %.4f seconds" % (
        MPI.Wtime() - start_winInitTime))

    """
    Distribute the data
    """

    if rank == 0:
        print("\nStart data distribution ...")
        start_distTime = MPI.Wtime()

    '''
    Set seed according to the bootstrap id
    '''

    np.random.seed(seed)
    seeds = np.random.randint(9999, size=maxBoot)
    seed = seeds[bt_id]
    np.random.seed(seed)

    '''
    Select sample ids and create data container with the appropriate size
    and generate ids for separating training and testing blocks
    '''

    shuf_m = np.random.permutation(m)

    m_train = int(round(rndfrctL * m))

    train_ids = shuf_m[:m_train]

    # create data container of size my_m
    ids = train_ids[admm_rank::admm_psize]

    # add test samples to the root ADMM processes
    if admm_rank == 0:
        ids = shuf_m

    my_m = len(ids)

    X = np.zeros((my_m, n + 1)).astype(dtype)

    win.Fence()

    for i in range(len(ids)):

        srank = loc_id[ids[i]]

        if ids[i] != 0:
            row = np.bincount(loc_id[:ids[i]])[-1] - 1
        else:
            row = 0

        win.Get(origin=X[i], target_rank=srank, target=row * (n + 1) * bpe)

        if rank == 0:
            print('\nGot message -->\t%i/%i' % (i, len(ids)))

    win.Fence()

    win.Free()

    if rank == 0:
        end_distTime = MPI.Wtime() - start_distTime
        print("\nData distributed in %.4f seconds" % end_distTime)

    y = X[:, -1]
    X = X[:, :-1]

    if rank == 0 or rank == 1:
        print("rank: %i X shape: %s" % (rank, str(X.shape)))

    '''
    Timer and info
    '''
    if rank == 0:
        print('\nBoLBO analysis initialized')
        print('----------------------------')
        print('\t*No proceses           : \t%i' % size)
        print('\t*No lambda elements    : \t%i' % nMP)
        print('\t*Max. No iterations    : \t%i' % maxBoot)
        print('\t*No model dimensions   : \t%i' % n)
        print('\t*No model samples      : \t%i' % m)
        start_compTime = MPI.Wtime()

    """
    ===============
    Model Selection
    ===============
    """

    '''
    Lambda vector for initial coarse sweep
    '''
    lamb0 = np.logspace(-3, 3, nMP, dtype=np.float32)
    my_lamb0 = lamb0[mp_id]

    '''
    Create arrays to collect results
    '''
    my_B0 = np.zeros(n, dtype=np.float32)
    if admm_rank == 0:
        my_R2m0 = np.zeros(1, dtype=np.float32)

    if rank == 0:
        B0 = np.zeros((maxBoot * nMP, n), dtype=np.float32)
        R2m0 = np.zeros(maxBoot * nMP, dtype=np.float32)
    else:
        B0 = None
        R2m0 = None

    '''
    Lasso
    '''
    if rank == 0:
        start_las1Time = MPI.Wtime()

    if admm_rank == 0:
        X_train = X[:m_train][admm_rank::admm_psize]
        y_train = y[:m_train][admm_rank::admm_psize]
    else:
        X_train = X
        y_train = y

    # train
    my_B0 = admm.lasso_admm(X_train, \
                            (y_train - y_train.mean())[..., np.newaxis], \
                            alpha=my_lamb0, comm=comm_ADMM).ravel()

    if admm_rank == 0:
        # test
        X_test = X[m_train:]
        y_test = y[m_train:]
        yhat = X_test.dot(my_B0)
        r = np.corrcoef(yhat, y_test - y_test.mean())
        my_R2m0 = r[1, 0] ** 2

    if rank == 0:
        end_las1Time = MPI.Wtime() - start_las1Time
        start_bca1Time = MPI.Wtime()

    '''
    Gather results
    '''
    if rank in admm_root_ids:
        comm_ROOT.Barrier()
        comm_ROOT.Gather([my_B0.astype('float32'), MPI.FLOAT], [B0, MPI.FLOAT])
        comm_ROOT.Gather([my_R2m0.astype('float32'), MPI.FLOAT],
                         [R2m0, MPI.FLOAT])

    comm_ALL.Barrier()

    if rank == 0:
        B0 = B0.reshape((maxBoot, nMP, n))
        R2m0 = R2m0.reshape((maxBoot, nMP))
        end_bca1Time = MPI.Wtime() - start_bca1Time

        '''
        Compute new Lambda vector for dense sweep
        '''
        Mt = np.fix(1e4 * R2m0.mean(0))
        Lids = np.where(Mt == np.max(np.ma.masked_invalid(Mt)))[0]
        v = lamb0[Lids[len(Lids) // 2]]
        dv = 10 ** (np.floor(np.log10(v)) - 1)
        lambL = np.linspace(v - 5 * dv, v + 5 * dv, nMP)
    else:
        lambL = np.zeros(nMP)

    comm_ALL.Bcast([lambL, MPI.DOUBLE])
    my_lambL = lambL[mp_id]

    '''
    Create arrays to collect results
    '''
    my_B = np.zeros(n, dtype=np.float32)

    if admm_rank == 0:
        my_R2m = np.zeros(1, dtype=np.float32)

    if rank == 0:
        B = np.zeros((maxBoot * nMP, n), dtype=np.float32)
        R2m = np.zeros(maxBoot * nMP, dtype=np.float32)
    else:
        B = None
        R2m = None

    '''
    Lasso
    '''

    if rank == 0:
        start_las2Time = MPI.Wtime()

    # train
    my_B = admm.lasso_admm(X_train,
                           (y_train - y_train.mean())[..., np.newaxis],
                           alpha=my_lambL, comm=comm_ADMM).ravel()

    if admm_rank == 0:
        # test
        yhat = X_test.dot(my_B)
        r = np.corrcoef(yhat, y_test - y_test.mean())
        my_R2m = r[1, 0] ** 2

    if rank == 0:
        end_las2Time = MPI.Wtime() - start_las2Time
        start_bca2Time = MPI.Wtime()

    '''
    Gather results
    '''
    if rank in admm_root_ids:
        comm_ROOT.Barrier()
        comm_ROOT.Gather([my_B.astype('float32'), MPI.FLOAT], [B, MPI.FLOAT])
        comm_ROOT.Gather([my_R2m.astype('float32'), MPI.FLOAT],
                         [R2m, MPI.FLOAT])

    if rank == 0:
        B = B.reshape((maxBoot, nMP, n))
        R2m = R2m.reshape((maxBoot, nMP))
        end_bca2Time = MPI.Wtime() - start_bca2Time

        '''
        Compute family of supports
        '''
        sprt = np.ones((nMP, n)) * np.nan
        for i in range(nMP):
            for r in range(nbootS):
                tmp_ids = np.where(B[r, i] != 0)[0]
                if r == 0:
                    intv = tmp_ids
                intv = np.intersect1d(intv, tmp_ids).astype(np.int32)
            sprt[i, :len(intv)] = intv
    else:
        sprt = np.zeros((nMP, n))

    comm_ALL.Bcast([sprt, MPI.DOUBLE])

    """
    ================
    Model Estimation
    ================
    """

    m_frac = int(round(cvlfrct * m))
    L_frac = int(round(rndfrct * m_frac))

    '''
    Create arrays to collect final results (root only)
    '''

    if rank == 0:
        Bgd = np.zeros((nrnd, n), dtype=np.float32)
        R2 = np.zeros(nrnd, dtype=np.float32)
        rsd = np.zeros((nrnd, m - m_frac), dtype=np.float32)
        bic = np.zeros(nrnd, dtype=np.float32)

    for cc in range(nrnd):

        '''
        Create arrays to collect results
        '''
        my_Bgols_B = np.zeros(n, dtype=np.float32)
        my_Bgols_R2m = np.zeros(1, dtype=np.float32)

        if rank == 0:
            Bgols_B = np.zeros((maxBoot * nMP, n), dtype=np.float32)
            Bgols_R2m = np.zeros(maxBoot * nMP, dtype=np.float32)
        else:
            Bgols_B = None
            Bgols_R2m = None

        '''
        Generate ids for separating training and testing blocks
        '''

        inds = np.random.permutation(m)
        L = inds[:m_frac]
        # test set for final evaluation
        T = inds[m_frac:]

        # train set is divided into training and testing set
        inds = np.random.permutation(m_frac)
        train = inds[:L_frac]
        test = inds[L_frac:]

        if admm_rank != 0:
            # from the training samples the process already  has select the ones
            # that have been selected as training samples for this CV round of OLS
            train = np.array([i in L[train] for i in ids])
        else:
            # admm_rank 0 contain all the training samples therefore they can
            # complete the list of training samples for this CV round CV of OLS
            others_ids = np.setdiff1d(train_ids,
                                      train_ids[admm_rank::admm_psize])
            train = L[train][np.array([i not in others_ids for i in L[train]])]

        """
        Select support
        """
        sprt_ids = sprt[mp_id][~np.isnan(sprt[mp_id])].astype('int')
        zdids = np.setdiff1d(np.arange(n), sprt_ids)

        '''
        #Linear regression
        '''

        if rank == 0 and cc == 0:
            start_olsTime = MPI.Wtime()

        if len(sprt_ids) > 0:

            # We use Lasso without penalty which has good convergence
            rgstrct = admm.lasso_admm(X[train], \
                                      (y[train] - y[train].mean())[
                                          ..., np.newaxis], \
                                      alpha=0., comm=comm_ADMM).ravel()

            # apply support
            my_Bgols_B[sprt_ids] = rgstrct
            my_Bgols_B[zdids] = 0

            if admm_rank == 0:
                # test
                yhat = X[L[test]].dot(my_Bgols_B)
                r = np.corrcoef(yhat, y[L[test]] - y[L[test]].mean())
                my_Bgols_R2m = r[1, 0] ** 2
        else:
            print('%i parameters were selected for lambda: %.4f' % (
            np.sum(sprt_ids), my_lambL))

        if rank == 0 and cc == 0:
            end_olsTime = MPI.Wtime() - start_olsTime
            start_bca3Time = MPI.Wtime()

        '''
        Gather results
        '''
        if rank in admm_root_ids:
            comm_ROOT.Barrier()
            comm_ROOT.Gather([my_Bgols_B.astype('float32'), MPI.FLOAT],
                             [Bgols_B, MPI.FLOAT])
            comm_ROOT.Gather([my_Bgols_R2m.astype('float32'), MPI.FLOAT],
                             [Bgols_R2m, MPI.FLOAT])

        if rank == 0:
            Bgols_B = Bgols_B.reshape((maxBoot, nMP, n))
            Bgols_R2m = Bgols_R2m.reshape((maxBoot, nMP))
            if cc == 0:
                end_bca3Time = MPI.Wtime() - start_bca3Time

            """
            Bagging
            """
            if bgdOpt == 1:
                v = np.max(Bgols_R2m, 1)
                ids = np.where(Bgols_R2m == v[:, np.newaxis])
                btmp = np.zeros((nbootE, n))
                for kk in range(nbootE):
                    ids_kk = ids[1][np.where(ids[0] == kk)]
                    btmp[kk] = Bgols_B[kk, ids_kk[len(ids_kk) // 2]]
                Bgd[cc] = np.median(btmp, 0)
            else:
                mean_Bgols_R2m = np.mean(Bgols_R2m, 0)
                v = np.max(mean_Bgols_R2m)
                ids = np.where(mean_Bgols_R2m == v)[0]
                Bgd[cc] = np.median(Bgols_B[:, ids], 0)

            yhat = X[T].dot(Bgd[cc])
            r = np.corrcoef(yhat, y[T] - y[T].mean())

            R2[cc] = r[1, 0] ** 2
            rsd[cc] = (y[T] - y[T].mean()) - yhat
            bic[cc] = (m - m_frac) * np.log(
                np.dot(rsd[cc], rsd[cc]) / (m - m_frac)) + \
                      np.log(m - m_frac) * n

    comm_ALL.Barrier()

    """
    Store results
    """
    if rank == 0:
        end_compTime = MPI.Wtime() - start_compTime
        start_saveTime = MPI.Wtime()

        with h5py.File(outputFile, 'w') as f:
            f.attrs['bgdOpt'] = bgdOpt
            f.attrs['nrnd'] = nrnd
            f.attrs['cvlfrct'] = cvlfrct
            f.attrs['rndfrct'] = rndfrct
            f.attrs['rndfrctL'] = rndfrctL
            f.attrs['nbootE'] = nbootE
            f.attrs['nbootS'] = nbootS
            f.attrs['nMP'] = nMP
            f.attrs['seed'] = seed

            f.attrs['n_samp'] = m
            f.attrs['n_cov'] = n

            f.attrs['sizeTime'] = end_sizeTime
            f.attrs['commTime'] = end_commTime
            f.attrs['loadTime'] = end_loadTime
            f.attrs['distTime'] = end_distTime
            f.attrs['compTime'] = end_compTime

            f.attrs['las1Time'] = end_las1Time
            f.attrs['bca1Time'] = end_bca1Time
            f.attrs['las2Time'] = end_las2Time
            f.attrs['bca2Time'] = end_bca2Time
            f.attrs['olsTime'] = end_olsTime
            f.attrs['bca3Time'] = end_bca3Time

            f.create_dataset(data=seeds, name='seeds', compression='gzip')

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

            end_saveTime = MPI.Wtime() - start_saveTime
            f.attrs['saveTime'] = end_saveTime

        print('\nBoLBO analysis completed')
        print('------------------------')
        print('\t*Results stored in %s' % (outputFile))
        print("\t*BoLBO times:")
        print("\t\t-size time: %.4f" % end_sizeTime)
        print("\t\t-comm time: %.4f" % end_commTime)
        print("\t\t-load time: %.4f" % end_loadTime)
        print("\t\t-dist time: %.4f" % end_distTime)
        print("\t\t-comp time: %.4f" % end_compTime)
        print("\t\t\t-las1 time: %.4f" % end_las1Time)
        print("\t\t\t-bca1 time: %.4f" % end_bca1Time)
        print("\t\t\t-las2 time: %.4f" % end_las2Time)
        print("\t\t\t-bca2 time: %.4f" % end_bca2Time)
        print("\t\t\t-ols  time: %.4f" % end_olsTime)
        print("\t\t\t-bca3 time: %.4f" % end_bca3Time)
        print("\t\t-save time: %.4f" % end_saveTime)


if __name__ == '__main__':
    main()
