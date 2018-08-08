from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import NMF
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

from sklearn.utils.validation import check_non_negative

import scipy.linalg as spla
import scipy.optimize as spo
import numpy as np

class UoINMF(BaseEstimator, TransformerMixin):

    def __init__(self, n_boostraps_u=10, n_bootstraps_i=10,
                 random_state=None,
                 ranks=None, nmf=None, dbscan=None, lasso=None):
        """
        Union of Intersections Nonnegative Matrix Factorization

        Args:
            n_bootstraps_i (int):   number of bootstraps to use for model selection
            n_bootstraps_u (int):   number of bootstraps to use for model estimation
            ranks (int, list):      the range of k to use. if *ranks* is an int,
                                    range(2, ranks+1) will be used. If not specified, will
                                    range(X.shape[1]) will be used.
            nmf:                    the NMF class to use. Note: this class must take
                                    the argument *n_components* as an argument
            dbscan:                 the DBSCAN object to use.  By default use sklearn.cluster.DBSCAN
                                    with MinPts=3 and epsilon=0.2
            lasso:                  the lasso object to use.
        """

        self.__initialize(
            n_boostraps_u = n_boostraps_u,
            n_bootstraps_i = n_bootstraps_i,
            ranks = ranks,
            nmf = nmf,
            dbscan = dbscan,
            random_state = random_state,
        )

    def set_params(self, **kwargs):
        self.__initialize(**kwargs)

    def __initialize(self, **kwargs):
        n_boostraps_u = kwargs['n_boostraps_u']
        n_bootstraps_i = kwargs['n_bootstraps_i']
        ranks = kwargs['ranks']
        nmf = kwargs['nmf']
        dbscan = kwargs['dbscan']
        random_state = kwargs['random_state']
        self.n_bootstraps_u = n_boostraps_u
        self.n_bootstraps_i = n_bootstraps_i
        self.components_ = None
        if ranks is not None:
            if isinstance(ranks, int):
                self.ranks = list(range(2,ranks+1)) if isinstance(ranks, int) else list(ranks)
            elif isinstance(ranks, (list, tuple, range, np.array)):
                self.ranks = tuple(ranks)
            else:
                raise ValueError('specify a max value or an array-like for k')
        if nmf is not None:
            if isinstance(nmf, type):
                raise ValueError('nmf must be an instance, not a class')
            self.nmf = nmf
        else:
            self.nmf = NMF(beta_loss='kullback-leibler', solver='mu', max_iter=400)
        if dbscan is not None:
            if isinstance(dbscan, type):
                raise ValueError('dbscan must be an instance, not a class')
            self.dbscan = dbscan
        else:
            self.dbscan = DBSCAN()

        if random_state is None:
            self._rand = np.random
        else:
            if isinstance(random_state, int):
                self._rand = np.random.RandomState(random_state)
            elif isinstance(random_state, np.random.RandomState):
                self._rand = random_state
            self.nmf.set_params(random_state=self._rand)
        self.cons_meth = np.mean              # the method for computing consensus H bases after clustering
        self.components_ = None
        self.bases_samples_ = None
        self.boostraps_ = None

    def fit(self, X, y=None):
        """
        Perform first phase of UoI NMF decomposition.

        Compute H matrix.

        Iterate across a range of k (as specified with the *ranks* argument).

        Args:
            X:  array of shape (n_samples, n_features)
            y:  ignored
        """
        check_non_negative(X, 'UoINMF')
        n, p = X.shape
        Wall = list()
        k_tot = sum(self.ranks)
        n_H_samples = k_tot*self.n_bootstraps_i
        H_samples = np.zeros((n_H_samples, p), dtype=np.float64)
        ridx = list()
        rep_idx = self._rand.randint(n, size=(self.n_bootstraps_i, n))
        for i in range(self.n_bootstraps_i):
            # compute NMF bases for k across bootstrap replicates
            H_i = i * k_tot
            sample = X[rep_idx[i]]
            for (k_idx, k) in enumerate(self.ranks):
                H_samples[H_i:H_i+k:,] = self.nmf.set_params(n_components=k).fit(sample).components_             # concatenate k by p
                H_i += k
        H_samples = H_samples[np.sum(H_samples, axis=1) != 0.0]      # remove zero bases
        H_samples = normalize(H_samples)                             # normalize by 2-norm   # TODO: double check normalizing across correct axis

        labels = self.dbscan.fit_predict(H_samples)                  # cluster all bases
        # compute consensus bases from clusters
        cluster_ids = np.unique([x for x in labels if x != -1])                              # TODO: check if we need to filter out -1
        nclusters = len(cluster_ids)
        H_cons = np.zeros((nclusters, p), dtype=np.float64)
        for i in cluster_ids:
            H_cons[i,:] = self.cons_meth(H_samples[labels==i], axis=0)
        # H_cons = H_cons[np.any(np.isnan(H_cons), axis=1),]        # remove nans            # TODO: check if we need to remove NaNs
        H_cons = normalize(H_cons)                                          # normalize by 2-norm
        self.components_ = H_cons
        self.bases_samples_ = H_samples
        self.boostraps_ = rep_idx
        self.reconstruction_err_ = None
        return self


    def transform(self, X, reconstruction_err=True):
        """
        Transform the data X according to the fitted UoI-NMF model

        Args:
            X                           : array-like; shape (n_samples, n_features)
            reconstruction_err          : bool, True to compute reconstruction error, False otherwise
                                          default True
        """
        if self.components_ is None:
            raise ValueError('UoINMF not fit')
        if X.shape[1] != self.components_[1]:
            raise ValueError('incompatible shape: cannot reconstruct with %s and %s' % (X.shape, self.components_.shape))
        H_t = self.components_.T
        ret = np.zeros(X.shape, dtype=X.dtype)
        for i in range(X.shape[0]):
            ret[i] = spo.nnls(H_t, X[i])
        if reconstruction_error:
            self.reconstruction_err_ = np.linalg.norm(self.inverse_transform(ret))
        return ret

    def fit_transform(self, X, y=None, reconstruction_err=True):
        """
        Transform the data X according to the fitted UoI-NMF model

        Args:
            X                           : array-like; shape (n_samples, n_features)
            y                           : ignored
            reconstruction_err          : bool, True to compute reconstruction error, False otherwise
                                          default True
        Returns:
            W                           : array-like; shape (n_samples, n_components)
                                          Transformed data
        """
        self.fit(X)
        return self.transform(X, reconstruction_err=reconstruction_err)

    def inverse_transform(self, W):
        """
        Transform data back to its original space.

        Args:
            W                           : array-like; shape (n_samples, n_components)
                                          Transformed data matrix
        Returns:
            X                           : array-like; shape (n_samples, n_features)
                                          Data matrix of original shape
        """
        if self.components_ is None:
            raise ValueError('UoINMF not fit')
        if W.shape[1] != self.components_[0]:
            raise ValueError('incompatible shape: cannot multiply %s with %s' % (W.shape, self.components_.shape))
        return np.matmul(W, self.components_)

