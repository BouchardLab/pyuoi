#!/usr/bin/env python

import pdb,os,h5py,time
import numpy as np
from mpi4py import MPI
import sklearn.linear_model as lm
from optparse import OptionParser

np.seterr(invalid='ignore')


__authors__ = "Alex Bujan, Kris Bouchard"


def main():
    usage = '%prog [options]'
    parser = OptionParser(usage)

    parser.add_option("--bgdOpt",type="int",default=1,\
        help="bagging options")
    parser.add_option("--nrnd",type="int",default=10,\
        help="number of randomizations for bagged estimates")
    parser.add_option("--nbootS",type="int",default=48,\
        help="number of bootstraps for model selection")
    parser.add_option("--nbootE",type="int",default=48,\
        help="number of bootstraps for model estimation")
    parser.add_option("--cvlfrct",type="float",default=.9,\
        help="data fraction for training during each bagged-OLS randomization")
    parser.add_option("--rndfrct",type="float",default=.8,\
        help="data fraction used for linear regression fitting")
    parser.add_option("--rndfrctL",type="float",default=.8,\
        help="fraction of data used for Lasso fitting")
    parser.add_option("--nMP",type="int",default=48,\
        help="number of regularization parameters to evaluate")
    parser.add_option("-o",type="string",default=os.getcwd(),\
        help="hdf5 file to store the results")
    parser.add_option("-i",type="string",default='',\
        help="hdf5 file containing input data (X,y pairs)")
    parser.add_option("--verbose",action='store_true',\
        dest='verbose',help="print information in the terminal")
    parser.add_option("--seed",type="int",default=1234,\
        help="initial seed for pseudo-random number generator")
    parser.add_option("--dtype",type="string",default='f4',\
        help="float type to be used. Options: 'f4' and 'f8' bit")
    parser.add_option("--ncov",type="int",default=-1,\
        help="number of covariates in the design matrix (optional)")
    parser.add_option("--nsamp",type="int",default=-1,\
        help="number of samples in the design matrix (optional)")
    parser.add_option("--nreaders",type="int",default=10,\
        help="number of processes to load the data (optional)")
    parser.add_option("--n_minibatch",type="int",default=10,\
        help="number of minibatches used with partial fit (optional)")

    (options, args) = parser.parse_args()

    if options.verbose:
        verbose=True
    else:
        verbose=False

    if options.nsamp==-1 or options.ncov==-1:
        nsamp = None
        ncov = None
    else:
        nsamp = options.nsamp
        ncov  = options.ncov

    BoLASSO_BgdOLS(bgdOpt=options.bgdOpt,nrnd=options.nrnd,nbootE=options.nbootE,\
                    cvlfrct=options.cvlfrct,rndfrct=options.rndfrct,\
                    nMP=options.nMP,inputFile=options.i,outputFile=options.o,\
                    verbose=verbose,seed=options.seed,nbootS=options.nbootS,\
                    rndfrctL=options.rndfrctL,dtype=options.dtype,nsamp=nsamp,\
                    ncov=ncov,nreaders=options.nreaders,n_minibatch=options.n_minibatch)


def split_data(m,m_frac):
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
    return train_ids,test_ids


def BoLASSO_BgdOLS(inputFile,outputFile,bgdOpt=1,nrnd=10,\
                    nbootE=48,cvlfrct=.9,rndfrct=.8,nMP=48,verbose=True,\
                    seed=1234,comm=None,nbootS=48,rndfrctL=.9,\
                    dtype='f4',nsamp=None,ncov=None,nreaders=10,n_minibatch=10):
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

    '''
    Global communicator
    '''
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    #check if correct number of processes - TODO: make it more flexible
    maxBoot = np.maximum(nbootE,nbootS)
    minSize = nMP*maxBoot
    assert size==minSize,ValueError('The number of processes must be at least %i'%minSize)

    '''
    Asignment of MP and bootstrap ids
    '''
    bt_id = rank//nMP
    mp_id = np.tile(np.arange(nMP),maxBoot)[rank]

    '''
    Get size of the design matrix
    '''
    if rank==0:
        start_sizeTime = MPI.Wtime()

    if ncov==None or nsamp==None:
        if rank==0:
            with h5py.File(inputFile,'r') as f:
                m,n = f['data/X'].shape
        else:
            n = None
            m = None

        comm.Barrier()
        n = comm.bcast(obj=n,root=0)
        m = comm.bcast(obj=m,root=0)

    else:
        m = nsamp
        n = ncov
        if rank==0:
            t_start_with_io = MPI.Wtime()
            print("\nData size: (%i,%i)"%(m,n))

    if rank==0:
        end_sizeTime = MPI.Wtime()-start_sizeTime
        print("\nData size: (%i,%i)"%(m,n))
        print("\nData size read/broadcasted in %.4f seconds"%end_sizeTime)

    """
    Create data container
    """

    X = np.zeros((m,n),dtype=dtype)
    y = np.zeros(m,dtype=dtype)

    """
    Create readers group
    """

    if rank==0:
        print("\nInitializing communicator for data loading ...")
        print("\nNumber of loading prodesses (readers): %i"%nreaders)
        start_commTime = MPI.Wtime()

    reader_ids = np.arange(nreaders)
    readers = comm.Get_group().Incl(reader_ids)
    new_comm = comm.Create(readers)

    if rank==0:
        end_commTime = MPI.Wtime()-start_commTime
        print("\nCommunicator created in %.4f seconds"%(end_commTime))

    """
    Get the data
    """
    n_samp = m//nreaders
    loc_id = np.arange(m)//n_samp
    loc_id[loc_id>=nreaders]=nreaders-1

    if rank in reader_ids:

        sids = np.where(loc_id==rank)[0]

        if rank==0:
            print("\nLoading the data with parallel-h5py ...")
            start_loadTime = MPI.Wtime()

        with h5py.File(inputFile,'r',driver='mpio',comm=new_comm) as f:
            f.atomic = True
            X[sids[0]:sids[-1]+1] = f['data/X'][sids[0]:sids[-1]+1].astype(dtype)
            y[sids[0]:sids[-1]+1] = f['data/y'][sids[0]:sids[-1]+1].astype(dtype)

        if rank==0:
            end_loadTime = MPI.Wtime()-start_loadTime
            print("\nData loaded in %.4f seconds"%(end_loadTime))

    """
    Select seed (according to bootstrap ids)
    """
    
    np.random.seed(seed)
    seeds = np.random.randint(9999,size=maxBoot)
    seed = seeds[bt_id]
    np.random.seed(seed)

    """
    Distribute the data
    """

    if rank==0:
        print("\nStart data distribution ...")
        start_distTime = MPI.Wtime()

    comm.Barrier()
    for i in range(nreaders):

        sids = np.where(loc_id==i)[0]

        if dtype=='f4':
            comm.Bcast([X[sids[0]:sids[-1]+1],MPI.FLOAT],root=i)
            comm.Bcast([y[sids[0]:sids[-1]+1],MPI.FLOAT],root=i)

        elif dtype=='f8':
            comm.Bcast([X[sids[0]:sids[-1]+1],MPI.DOUBLE],root=i)
            comm.Bcast([y[sids[0]:sids[-1]+1],MPI.DOUBLE],root=i)

    if rank==0:
        end_distTime = MPI.Wtime()-start_distTime
        print("\nData distributed in %.4f seconds"%end_distTime)

    '''
    Print info and start compute timer
    '''
    if rank==0:
        print('\nBoLBO analysis initialized')
        print('----------------------------')
        print('\t*No proceses           : \t%i'%size)
        print('\t*No lambda elements    : \t%i'%nMP)
        print('\t*Max. No iterations    : \t%i'%maxBoot)
        print('\t*No model dimensions   : \t%i'%n)
        print('\t*No model samples      : \t%i'%m)
        start_compTime = MPI.Wtime()

    """
    ===============
    Model Selection
    ===============
    """

    '''
    Lambda vector for initial coarse sweep
    '''
    lamb0    = np.logspace(-3,3,nMP,dtype=np.float32)
    my_lamb0 = lamb0[mp_id]

    '''
    Create arrays to collect results
    '''
    my_B0    = np.zeros(n,dtype=np.float32)
    my_R2m0  = np.zeros(1,dtype=np.float32)

    if rank==0:
        B0    = np.zeros((maxBoot*nMP,n),dtype=np.float32)
        R2m0  = np.zeros( maxBoot*nMP,dtype=np.float32)
    else:
        B0    = None
        R2m0  = None

    '''
    Generate ids for separating training and testing blocks
    '''
    m_frac = int(round(rndfrctL*m))
    inds = np.random.permutation(m)
    train = inds[:m_frac]
    test  = inds[m_frac:]

    '''
    Lasso
    '''
    if rank==0:
        start_las1Time = MPI.Wtime()

    #train
    try:
        outLas = lm.Lasso(alpha=my_lamb0)
        outLas.fit(X[train],\
                   y[train]-y[train].mean())
    except:
        outLas = lm.SGDRegressor(penalty='l1',alpha=my_lamb0)
        for j in range(n_minibatch):
            minibatch = train[j::n_minibatch]
            #print '\nlasso 1 - mini-batch %i/%i size: %i'%(j,n_minibatch,len(minibatch))
            outLas.partial_fit(X[minibatch],\
                               y[minibatch]-y[minibatch].mean())
    my_B0 = outLas.coef_

    #test
    yhat = X[test].dot(my_B0)
    r = np.corrcoef(yhat,y[test]-y[test].mean())
    my_R2m0 = r[1,0]**2

    if rank==0:
        end_las1Time = MPI.Wtime()-start_las1Time
        start_bca1Time = MPI.Wtime()

    '''
    Gather results
    '''
    comm.Barrier()
    comm.Gather([my_B0.astype('float32'),MPI.FLOAT],[B0,MPI.FLOAT])
    comm.Gather([my_R2m0.astype('float32'),MPI.FLOAT],[R2m0,MPI.FLOAT])

    if rank==0:
        B0    = B0.reshape((maxBoot,nMP,n))
        R2m0  = R2m0.reshape((maxBoot,nMP))
        end_bca1Time = MPI.Wtime()-start_bca1Time

        '''
        Compute new Lambda vector for dense sweep
        '''
        Mt      = np.fix(1e4*R2m0.mean(0))
        Lids    = np.where(Mt==np.max(np.ma.masked_invalid(Mt)))[0]
        v       = lamb0[Lids[len(Lids)//2]]
        dv      = 10**(np.floor(np.log10(v))-1)
        lambL   = np.linspace(v-5*dv,v+5*dv,nMP)
    else:
        lambL   = np.zeros(nMP)

    comm.Bcast([lambL,MPI.DOUBLE])
    my_lambL = lambL[mp_id]

    '''
    Create arrays to collect results
    '''
    my_B     = np.zeros(n,dtype=np.float32)
    my_R2m   = np.zeros(1,dtype=np.float32)

    if rank==0:
        B     = np.zeros((maxBoot*nMP,n),dtype=np.float32)
        R2m   = np.zeros( maxBoot*nMP,dtype=np.float32)
    else:
        B     = None
        R2m   = None

    '''
    Generate ids for separating training and testing blocks
    '''
    inds  = np.random.permutation(m)
    train = inds[:m_frac]
    test  = inds[m_frac:]

    '''
    Lasso
    '''

    if rank==0:
        start_las2Time = MPI.Wtime()

    #train
    try:
        outLas = lm.Lasso(alpha=my_lambL)
        outLas.fit(X[train],\
                   y[train]-y[train].mean())
    except:
        outLas = lm.SGDRegressor(penalty='l1',alpha=my_lambL)
        for j in range(n_minibatch):
            minibatch = train[j::n_minibatch]
            #print '\nlasso 1 - mini-batch %i/%i size: %i'%(j,n_minibatch,len(minibatch))
            outLas.partial_fit(X[minibatch],\
                               y[minibatch]-y[minibatch].mean())
    my_B = outLas.coef_

    #test
    yhat = X[test].dot(my_B)
    r = np.corrcoef(yhat,y[test]-y[test].mean())
    my_R2m = r[1,0]**2

    if rank==0:
        end_las2Time = MPI.Wtime()-start_las2Time
        start_bca2Time = MPI.Wtime()

    '''
    Gather results
    '''
    comm.Barrier()
    comm.Gather([my_B.astype('float32'),MPI.FLOAT],[B,MPI.FLOAT])
    comm.Gather([my_R2m.astype('float32'),MPI.FLOAT],[R2m,MPI.FLOAT])

    if rank==0:
        B    = B.reshape((maxBoot,nMP,n))
        R2m  = R2m.reshape((maxBoot,nMP))
        end_bca2Time = MPI.Wtime()-start_bca2Time

        '''
        Compute family of supports
        '''
        sprt = np.ones((nMP,n))*np.nan
        for i in range(nMP):
            for r in range(nbootS):
                tmp_ids = np.where(B[r,i]!=0)[0]
                if r==0:
                    intv = tmp_ids
                intv = np.intersect1d(intv,tmp_ids).astype(np.int32)
            sprt[i,:len(intv)] = intv
    else:
        sprt   = np.zeros((nMP,n))

    comm.Bcast([sprt,MPI.DOUBLE])

    """
    ================
    Model Estimation
    ================
    """

    m_frac = int(round(cvlfrct*m))
    L_frac = int(round(rndfrct*m_frac))

    '''
    Create arrays to collect final results (root only)
    '''

    if rank==0:
        Bgd     = np.zeros((nrnd,n),dtype=np.float32)
        R2      = np.zeros(nrnd,dtype=np.float32)
        rsd     = np.zeros((nrnd,m-m_frac),dtype=np.float32)
        bic     = np.zeros(nrnd,dtype=np.float32)


    for cc in range(nrnd):

        '''
        Create arrays to collect results
        '''
        my_Bgols_B      = np.zeros(n,dtype=np.float32)
        my_Bgols_R2m    = np.zeros(1,dtype=np.float32)

        if rank==0:
            Bgols_B     = np.zeros((maxBoot*nMP,n),dtype=np.float32)
            Bgols_R2m   = np.zeros(maxBoot*nMP,dtype=np.float32)
        else:
            Bgols_B     = None
            Bgols_R2m   = None

        '''
        Generate ids for separating training and testing blocks
        '''
        inds  = np.random.permutation(m)
        L     = inds[:m_frac]
        T     = inds[m_frac:]

        inds  = np.random.permutation(m_frac)
        train = inds[:L_frac]
        test  = inds[L_frac:]

        """
        Select support
        """
        sprt_ids = sprt[mp_id][~np.isnan(sprt[mp_id])].astype('int')
        zdids = np.setdiff1d(np.arange(n),sprt_ids)



        '''
        #Linear regression
        '''

        if rank==0 and cc==0:
            start_olsTime = MPI.Wtime()

        if len(sprt_ids)>0:

            #train
            try:
                outLR = lm.LinearRegression()
                outLR.fit(X[L[train]][:,sprt_ids],\
                          y[L[train]]-y[L[train]].mean())
            except:
                outLR = lm.SGDRegressor(penalty='none')
                for j in range(n_minibatch):
                    minibatch = train[j::n_minibatch]
                    #print '\nols - mini-batch %i/%i size: %i'%(j,n_minibatch,len(minibatch))
                    outLR.partial_fit(X[L[minibatch]][:,sprt_ids],\
                                      y[L[minibatch]]-y[L[minibatch]].mean())
            rgstrct = outLR.coef_

            #apply support
            my_Bgols_B[sprt_ids] = rgstrct
            my_Bgols_B[zdids] = 0

            #test
            yhat = X[L[test]].dot(my_Bgols_B)
            r = np.corrcoef(yhat,y[L[test]]-y[L[test]].mean())
            my_Bgols_R2m = r[1,0]**2

        else:
            print('%i parameters were selected for lambda: %.4f'%(sprt_ids.sum(),my_lambL))


        if rank==0 and cc==0:
            end_olsTime = MPI.Wtime()-start_olsTime
            start_bca3Time = MPI.Wtime()

        '''
        Gather results
        '''
        comm.Barrier()
        comm.Gather([my_Bgols_B.astype('float32'),MPI.FLOAT],[Bgols_B,MPI.FLOAT])
        comm.Gather([my_Bgols_R2m.astype('float32'),MPI.FLOAT],[Bgols_R2m,MPI.FLOAT])

        if rank==0:
            Bgols_B    = Bgols_B.reshape((maxBoot,nMP,n))
            Bgols_R2m  = Bgols_R2m.reshape((maxBoot,nMP))
            if cc==0:
                end_bca3Time = MPI.Wtime()-start_bca3Time

            """
            Bagging
            """
            if bgdOpt==1:
                v    = np.max(Bgols_R2m,1)
                ids  = np.where(Bgols_R2m==v[:,np.newaxis])
                btmp = np.zeros((nbootE,n))
                for kk in range(nbootE):
                    ids_kk = ids[1][np.where(ids[0]==kk)] 
                    btmp[kk] = Bgols_B[kk,ids_kk[len(ids_kk)//2]]
                Bgd[cc] = np.median(btmp,0)
            else:
                mean_Bgols_R2m = np.mean(Bgols_R2m,0)
                v    = np.max(mean_Bgols_R2m)
                ids  = np.where(mean_Bgols_R2m==v)[0]
                Bgd[cc] = np.median(Bgols_B[:,ids],0)

            yhat = X[T].dot(Bgd[cc])
            r = np.corrcoef(yhat,y[T]-y[T].mean())

            R2[cc]  = r[1,0]**2
            rsd[cc] = (y[T]-y[T].mean())-yhat
            bic[cc] = (m-m_frac)*np.log(np.dot(rsd[cc],rsd[cc])/(m-m_frac))+\
                        np.log(m-m_frac)*n

    """
    Store results
    """
    if rank==0:
        end_compTime = MPI.Wtime()-start_compTime
        start_saveTime = MPI.Wtime()

        with h5py.File(outputFile,'w') as f:

            f.attrs['bgdOpt']   = bgdOpt
            f.attrs['nrnd']     = nrnd
            f.attrs['cvlfrct']  = cvlfrct
            f.attrs['rndfrct']  = rndfrct
            f.attrs['rndfrctL'] = rndfrctL
            f.attrs['nbootE']   = nbootE
            f.attrs['nbootS']   = nbootS
            f.attrs['nMP']      = nMP
            f.attrs['seed']     = seed

            f.attrs['n_samp']   = m
            f.attrs['n_cov']    = n

            f.attrs['sizeTime'] = end_sizeTime
            f.attrs['commTime'] = end_commTime
            f.attrs['loadTime'] = end_loadTime
            f.attrs['distTime'] = end_distTime
            f.attrs['compTime'] = end_compTime

            f.attrs['las1Time'] = end_las1Time
            f.attrs['bca1Time'] = end_bca1Time
            f.attrs['las2Time'] = end_las2Time
            f.attrs['bca2Time'] = end_bca2Time
            f.attrs['olsTime']  = end_olsTime
            f.attrs['bca3Time'] = end_bca3Time

            f.create_dataset(data=seeds,name='seeds',compression='gzip')

            g1 = f.create_group('/lasso')
            g1.create_dataset(name='B0',data=B0,compression='gzip')
            g1.create_dataset(name='R2m0',data=R2m0,compression='gzip')
            g1.create_dataset(name='lamb0',data=lamb0,compression='gzip')
            g1.create_dataset(name='B',data=B,compression='gzip')
            g1.create_dataset(name='R2m',data=R2m,compression='gzip')
            g1.create_dataset(name='lambL',data=lambL,compression='gzip')

            g2 = f.create_group('/bolbo')
            g2.create_dataset(name='sprt',data=sprt,compression='gzip')
            g2.create_dataset(name='Bgd',data=Bgd,compression='gzip')
            g2.create_dataset(name='R2',data=R2,compression='gzip')
            g2.create_dataset(name='rsd',data=rsd,compression='gzip')
            g2.create_dataset(name='bic',data=bic,compression='gzip')

            end_saveTime = MPI.Wtime()-start_saveTime
            f.attrs['saveTime'] = end_saveTime

        print('\nBoLBO analysis completed')
        print('------------------------')
        print('\t*Results stored in %s'%(outputFile))
        print("\t*BoLBO times:")
        print("\t\t-size time: %.4f"%end_sizeTime)
        print("\t\t-comm time: %.4f"%end_commTime)
        print("\t\t-load time: %.4f"%end_loadTime)
        print("\t\t-dist time: %.4f"%end_distTime)
        print("\t\t-comp time: %.4f"%end_compTime)
        print("\t\t\t-las1 time: %.4f"%end_las1Time)
        print("\t\t\t-bca1 time: %.4f"%end_bca1Time)
        print("\t\t\t-las2 time: %.4f"%end_las2Time)
        print("\t\t\t-bca2 time: %.4f"%end_bca2Time)
        print("\t\t\t-ols  time: %.4f"%end_olsTime)
        print("\t\t\t-bca3 time: %.4f"%end_bca3Time)
        print("\t\t-save time: %.4f"%end_saveTime)

if __name__=='__main__':
    main()
