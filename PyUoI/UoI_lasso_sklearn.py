#!/usr/bin/env python

import numpy as np
import sklearn.linear_model as lm
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from tqdm import trange

from PyUoI import lasso_admm as admm


class UoI_lasso(lm):
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=1, init_rand=10, nbootS=48, nbootE=48, cvlfrct=0.9,
                 rndfrct=0.8, rndfrctL=0.8, nMP=48, n_minibatch=10,
                 with_admm=False, random_state=None):
        """

        Parameters
        ----------
        fit_intercept
        normalize
        copy_X
        n_jobs: int
        init_rand: int
            Number of initial randomizations (default=10)
        nbootS: int
            Number of boots for model selection (default=48)
        nbootE: int
            Number of boots for model estimation (default=48)
        cvlfrct: float
            Data fraction for training during each bagged-OLS randomization
            (default=0.9)
        rndfrct: float
            Data fraction used for linear regression fitting (default=0.8)
        rndfrctL: float
            Data fraction used for Lasso fitting (default=0.8)
        nMP: int
            Number of regularization parameters to evaluate (default=48)
        n_minibatch:
            Number of minibatches used with partial fit (optional, default=10)
        with_admm: bool
            use ADMM solver for Lasso
        random_state: int


        """

        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.init_rand = init_rand
        self.nbootS = nbootS
        self.nbootE = nbootE
        self.cvlfrct = cvlfrct
        self.rndfrct = rndfrct
        self.rndfrctL = rndfrctL
        self.nMP = nMP
        self.n_minibatch = n_minibatch
        self.with_admm = with_admm
        self.random_state = random_state

    def fit(self, X, y):

        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         y_numeric=True, multi_output=True)

        m, n = X.shape

        """
        ===============
        Model Selection
        ===============
        """

        '''
        Lambda vector for initial coarse sweep
        '''
        if self.nMP == 1:
            lamb0 = np.array([1.0])
        else:
            lamb0 = np.logspace(-3, 3, self.nMP, dtype=np.float64)

        '''
        Create arrays to collect results
        '''
        B0 = np.zeros((self.initrnd, n, self.nMP), dtype=np.float32)
        R2m0 = np.zeros((self.initrnd, self.nMP), dtype=np.float32)

        '''
        Generate ids for separating training and testing blocks
        '''
        if self.seed is not None:
            np.random.seed(self.seed)
        m_frac = int(round(self.rndfrctL * m))

        '''
        Lasso
        '''
        for c in trange(self.initrnd, total=self.initrnd, desc='fitting models'):

            inds = np.random.permutation(m)
            train = inds[:m_frac]
            test = inds[m_frac:]

            for i in range(self.nMP):

                # train
                if not self.with_admm:
                    try:
                        outLas = lm.Lasso(alpha=lamb0[i])
                        outLas.fit(X[train], y[train]-y[train].mean())
                    except:
                        outLas = lm.SGDRegressor(penalty='l1', alpha=lamb0[i])
                        for j in range(self.n_minibatch):
                            minibatch = train[j::self.n_minibatch]
                            # print '\nlasso 1 - mini-batch %i/%i size: %i'%(j,n_minibatch,len(minibatch))
                            outLas.partial_fit(X[minibatch],
                                               y[minibatch]-y[minibatch].mean())
                    B0[c, :, i] = outLas.coef_
                else:
                    B0[c, :, i] = admm.lasso_admm(
                        X[train], (y[train]-y[train].mean())[..., np.newaxis],
                        alpha=lamb0[i])[0]

                # test
                yhat = X[test].dot(B0[c, :, i])
                r = np.corrcoef(yhat, y[test]-y[test].mean())
                R2m0[c, i] = r[1, 0]**2
        '''
        Compute new Lambda vector for dense sweep
        '''
        Mt = np.fix(1e4*np.mean(R2m0, 0))
        Lids = np.where(Mt == np.max(np.ma.masked_invalid(Mt)))[0]
        v = lamb0[Lids[len(Lids)//2]]
        dv = 10**(np.floor(np.log10(v))-1)

        if self.nMP == 1:
            lambL = np.array([1.0])
        else:
            lambL = np.linspace(v-5*dv, v+5*dv, self.nMP)

        '''
        Create arrays to collect results
        '''
        B = np.zeros((self.nbootS, n, self.nMP), dtype=np.float32)
        R2m = np.zeros((self.nbootS, self.nMP), dtype=np.float32)

        for c in trange(self.nbootS, desc='boosting'):
            '''
            Generate ids for separating training and testing blocks
            '''
            inds = np.random.permutation(m)
            train = inds[:m_frac]
            test = inds[m_frac:]

            '''
            Lasso
            '''
            for i in range(self.nMP):

                # train
                if not self.with_admm:
                    try:
                        outLas = lm.Lasso(alpha=lambL[i])
                        outLas.fit(X[train], y[train]-y[train].mean())
                    except:
                        outLas = lm.SGDRegressor(penalty='l1', alpha=lambL[i])
                        for j in range(self.n_minibatch):
                            minibatch = train[j::self.n_minibatch]
                            #print '\nlasso 2 - mini-batch %i/%i size: %i'%(j,n_minibatch,len(minibatch))
                            outLas.partial_fit(X[minibatch],
                                               y[minibatch]-y[minibatch].mean())
                    B[c, :, i] = outLas.coef_
                else:
                    B[c, :, i] = admm.lasso_admm(
                        X[train], (y[train]-y[train].mean())[..., np.newaxis],
                        alpha=lambL[i])[0]

            # test
            yhat = X[test].dot(B[c, :, i])
            r = np.corrcoef(yhat, y[test]-y[test].mean())
            R2m[c, i] = r[1, 0]**2

        try:
            del outLas, r, yhat
        except:
            print('could not find all vars to delete')

        '''
        Compute family of supports
        '''
        sprt = np.ones((self.nMP, n))*np.nan
        # for each regularization parameter
        for i in range(self.nMP):
            # for each bootstrap sample
            for r in range(self.nbootS):
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

        m_frac = int(round(self.cvlfrct*m))
        L_frac = int(round(self.rndfrct*m_frac))

        '''
        Create arrays to collect final results
        '''
        Bgd = np.zeros((self.nrnd, n), dtype=np.float32)
        R2 = np.zeros(self.nrnd, dtype=np.float32)
        rsd = np.zeros((self.nrnd, m-m_frac), dtype=np.float32)
        bic = np.zeros(self.nrnd, dtype=np.float32)


        for cc in trange(self.nrnd, desc='Model Estimation', total=self.nrnd):
            '''
            Create arrays to collect the results
            '''
            Bgols_B = np.zeros((self.nbootE, n, self.nMP), dtype=np.float32)
            Bgols_R2m = np.zeros((self.nbootE, self.nMP), dtype=np.float32)

            '''
            Generate ids for separating training and testing blocks
            '''
            inds = np.random.permutation(m)
            L = inds[:m_frac]
            T = inds[m_frac:]

            # number of bootstrap samples for aggregation
            for c in trange(self.nbootE, desc='boostE'):
                inds = np.random.permutation(m_frac)
                train = inds[:L_frac]
                test = inds[L_frac:]

                # for each regularization parameter
                for i in range(self.nMP):
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
                            for j in range(self.n_minibatch):
                                minibatch = train[j::self.n_minibatch]
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
                        print('%i parameters were selected for lambda: %.4f'%\
                              (sprt_ids.sum(), lambL[i]))

            """
            Bagging
            """
            if bgdOpt == 1:
                # for each bootstrap sample, find regularization parameter
                # that gave best results
                v = np.max(Bgols_R2m, 1)
                ids = np.where(Bgols_R2m == v.reshape((self.nbootE, 1)))
                btmp = np.zeros((self.nbootE, n))
                # this loop can probably be removed
                for kk in range(self.nbootE):
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

        self.coef_ = Bgd[cc]

    def _decision_function(self, X):

        check_is_fitted(self, "coef_")

        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        return safe_sparse_dot(X, self.coef_.T,
                               dense_output=True) + self.intercept_

    def predict(self, X):

        return self._decision_function(X)

