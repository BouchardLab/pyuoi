from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import NMF as skNMF
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

from sklearn.utils.validation import check_non_negative

import scipy.optimize as spo
import numpy as np


class NMF(BaseEstimator, TransformerMixin):
    def __init__(
        self, n_boots=10, ranks=None, nmf=None, dbscan=None, nnreg=None,
        cons_meth=None, random_state=None
    ):
        """Performs non-negative matrix factorization in the Union of
        Intersection framework.

        See Bouchard et al., NIPS, 2017, for more details on the Union of
        Intersections framework.

        Parameters
        ----------
        n_boots : int
            The number of bootstraps to use for model selection.

        ranks : int, list, or None, default None
            The range of k to use. If *ranks* is an int,
            range(2, ranks + 1) will be used. If not specified,
            range(X.shape[1]) will be used.

        nmf : NMF object
            The NMF object to use to perform fitting. Note: this class must
            take argument *n_components* as an argument.

        dbscan : DBSCAN object
            DBSCAN object to use. By default, use sklearn.cluster.DBSCAN
            with MinPts=3 and epsilon=0.2.

        nnreg : NNLS object
            Non-negative regressor to use. If None, defaults to
            scipy.optimize.nnls.

        cons_meth : function
            The method for computing consensus bases after clustering. If None,
            uses np.median.
        """
        self.__initialize(
            n_boots=n_boots,
            ranks=ranks,
            nmf=nmf,
            dbscan=dbscan,
            nnreg=nnreg,
            cons_meth=cons_meth,
            random_state=random_state
        )

    def set_params(self, **kwargs):
        self.__initialize(**kwargs)

    def __initialize(self, **kwargs):
        """Initializes the NMF class."""
        n_boots = kwargs['n_boots']
        ranks = kwargs['ranks']
        nmf = kwargs['nmf']
        dbscan = kwargs['dbscan']
        nnreg = kwargs['nnreg']
        cons_meth = kwargs['cons_meth']
        random_state = kwargs['random_state']

        self.n_boots = n_boots
        self.components_ = None

        # initialize NMF ranks to use
        if ranks is not None:
            if isinstance(ranks, int):
                self.ranks = list(range(2, ranks + 1)) \
                    if isinstance(ranks, int) else list(ranks)
            elif isinstance(ranks, (list, tuple, range, np.array)):
                self.ranks = tuple(ranks)
            else:
                raise ValueError('Specify a max value or an array-like for k.')

        # initialize NMF solver
        if nmf is not None:
            if isinstance(nmf, type):
                raise ValueError('nmf must be an instance, not a class.')
            self.nmf = nmf
        else:
            self.nmf = skNMF(beta_loss='kullback-leibler', solver='mu',
                             max_iter=400, init='random')

        # initialize DBSCAN solver
        if dbscan is not None:
            if isinstance(dbscan, type):
                raise ValueError('dbscan must be an instance, not a class.')
            self.dbscan = dbscan
        else:
            self.dbscan = DBSCAN(min_samples=self.n_boots / 2)

        # initialize non-negative regression solver
        if nnreg is None:
            self.nnreg = lambda A, b: spo.nnls(A, b)[0]
        else:
            if isinstance(nnreg, ):
                self.nnreg = lambda A, B: nnreg.fit(A, B).coef_
            else:
                raise ValueError("Unrecognized regressor.")

        # initialize method for computing consensus H bases after clustering
        if cons_meth is None:
            # default uses median
            self.cons_meth = np.median
        else:
            self.cons_meth = cons_meth

        # initialize random state
        if random_state is None:
            self._rand = np.random
        else:
            if isinstance(random_state, int):
                self._rand = np.random.RandomState(random_state)
            elif isinstance(random_state, np.random.RandomState):
                self._rand = random_state
            self.nmf.set_params(random_state=self._rand)

        self.components_ = None
        self.bases_samples_ = None
        self.bases_samples_labels_ = None
        self.boostraps_ = None

    def fit(self, X, y=None):
        """Perform first phase of UoI NMF decomposition.

        Compute H matrix.

        Iterate across a range of k (as specified with the *ranks* argument).

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data matrix to be decomposed.

        y :  Ignored
        """
        check_non_negative(X, 'UoI NMF')
        n, p = X.shape

        k_tot = sum(self.ranks)
        n_H_samples = k_tot * self.n_boots
        H_samples = np.zeros((n_H_samples, p), dtype=np.float64)

        rep_idx = self._rand.randint(n, size=(self.n_boots, n))
        for i in range(self.n_boots):
            # compute NMF bases for k across bootstrap replicates
            H_i = i * k_tot
            sample = X[rep_idx[i]]
            for (k_idx, k) in enumerate(self.ranks):
                # concatenate k by p
                H_samples[H_i:H_i + k:, ] = (self.nmf.set_params(n_components=k)
                                             .fit(sample).components_)
                H_i += k

        # remove zero bases
        H_samples = H_samples[np.sum(H_samples, axis=1) != 0.0]
        # normalize by 2-norm
        # TODO: double check normalizing across correct axis
        H_samples = normalize(H_samples)

        # cluster all bases
        labels = self.dbscan.fit_predict(H_samples)

        # compute consensus bases from clusters
        # TODO: check if we need to filter out -1
        cluster_ids = np.unique([x for x in labels if x != -1])
        nclusters = len(cluster_ids)
        H_cons = np.zeros((nclusters, p), dtype=np.float64)
        for i in cluster_ids:
            H_cons[i, :] = self.cons_meth(H_samples[labels == i], axis=0)
        # remove nans
        # TODO: check if we need to remove NaNs
        # H_cons = H_cons[np.any(np.isnan(H_cons), axis=1),]
        # normalize by 2-norm
        H_cons = normalize(H_cons)
        self.components_ = H_cons
        self.bases_samples_ = H_samples
        self.bases_samples_labels_ = labels
        self.boostraps_ = rep_idx
        self.reconstruction_err_ = None
        return self

    def transform(self, X, reconstruction_err=True):
        """Transform the data X according to the fitted UoI-NMF model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be decomposed.

        reconstruction_err : bool
            True to compute reconstruction error, False otherwise.
            default True.
        """
        if self.components_ is None:
            raise ValueError('UoI NMF not fit')
        if X.shape[1] != self.components_.shape[1]:
            raise ValueError(
                'incompatible shape: cannot reconstruct with %s and %s'
                % (X.shape, self.components_.shape))
        H_t = self.components_.T
        ret = np.zeros((X.shape[0], self.components_.shape[0]), dtype=X.dtype)
        for i in range(X.shape[0]):
            ret[i] = self.nnreg(H_t, X[i])
        if reconstruction_err:
            self.reconstruction_err_ = np.linalg.norm(
                X - self.inverse_transform(ret))
        return ret

    def fit_transform(self, X, y=None, reconstruction_err=True):
        """Transform the data X according to the fitted UoI-NMF model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be decomposed.

        y : Ignored

        reconstruction_err : bool, default True
            True to compute reconstruction error, False otherwise.

        Returns
        -------
        W : array-like, shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X)
        return self.transform(X, reconstruction_err=reconstruction_err)

    def inverse_transform(self, W):
        """Transform data back to its original space.

        Parameters
        ----------
        W : array-like, shape (n_samples, n_components)
            Transformed data matrix.

        Returns
        -------
        X : array-like, shape (n_samples, n_features)
            Data matrix of original shape.
        """
        if self.components_ is None:
            raise ValueError('NMF algorithm not fit.')

        if W.shape[1] != self.components_.shape[0]:
            raise ValueError(
                'Incompatible shape: cannot multiply %s with %s.'
                % (W.shape, self.components_.shape))

        return np.matmul(W, self.components_)
