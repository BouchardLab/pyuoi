import scipy.optimize as spo
import numpy as np
import logging
import sys

from .base import AbstractDecompositionModel

from sklearn.decomposition import NMF as skNMF
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_non_negative
from sklearn.utils.validation import check_is_fitted


class UoI_NMF_Base(AbstractDecompositionModel):
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
        The NMF object to use to perform fitting.
        Note: this class must take *n_components* as an argument.

    cluster : Clustering object.
        Clustering object to use. If None, defaults to DBSCAN.

    nnreg : NNLS object
        Non-negative regressor to use. If None, defaults to
        scipy.optimize.nnls.

    cons_meth : function
        The method for computing consensus bases after clustering. If None,
        uses np.median.
    """
    def __init__(
        self, n_boots=10, ranks=None, nmf=None, cluster=None, nnreg=None,
        cons_meth=None, random_state=None, logger=None
    ):
        self.__initialize(
            n_boots=n_boots,
            ranks=ranks,
            nmf=nmf,
            cluster=cluster,
            nnreg=nnreg,
            cons_meth=cons_meth,
            logger=logger,
            random_state=random_state
        )

    def set_params(self, **kwargs):
        self.__initialize(**kwargs)

    def __initialize(self, **kwargs):
        """Initializes the NMF class."""
        n_boots = kwargs['n_boots']
        ranks = kwargs['ranks']
        nmf = kwargs['nmf']
        cluster = kwargs['cluster']
        nnreg = kwargs['nnreg']
        cons_meth = kwargs['cons_meth']
        random_state = kwargs['random_state']

        self.n_boots = n_boots
        self.components_ = None
        logger = kwargs['logger']

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

        # initialize clusterer
        if cluster is not None:
            if isinstance(cluster, type):
                raise ValueError('dbscan must be an instance, not a class.')
            self.cluster = cluster
        else:
            self.cluster = DBSCAN(min_samples=self.n_boots / 2)

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
        self.bootstraps_ = None

        self.comm = None

        if logger is None:
            name = "uoi_linear_model"
            if self.comm is not None and self.comm.Get_size() > 1:
                r, s = self.comm.Get_rank(), self.comm.Get_size()
                name += " " + str(r).rjust(int(np.log10(s)) + 1)

            self._logger = logging.getLogger(name=name)
            handler = logging.StreamHandler(sys.stdout)

            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

            handler.setFormatter(logging.Formatter(fmt))
            self._logger.addHandler(handler)
        else:
            self._logger = logger

    def fit(self, X, y=None, verbose=False):
        """
        Perform first phase of UoI NMF decomposition.

        Compute H matrix.

        Iterate across a range of k (as specified with the *ranks* argument).

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data matrix to be decomposed.
        """
        if verbose:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.WARNING)

        check_non_negative(X, 'UoI NMF')
        n_samples, n_features = X.shape

        k_tot = sum(self.ranks)
        n_H_samples = k_tot * self.n_boots
        H_samples = np.zeros((n_H_samples, n_features))

        rep_idx = self._rand.randint(n_samples, size=(self.n_boots, n_samples))
        for i in range(self.n_boots):
            self._logger.info("bootstrap %d" % i)
            # compute NMF bases for k across bootstrap replicates
            H_i = i * k_tot
            sample = X[rep_idx[i]]
            for k_idx, k in enumerate(self.ranks):
                # concatenate k by p
                H_samples[H_i:H_i + k:, ] = (self.nmf.set_params(n_components=k)
                                             .fit(sample).components_)
                H_i += k

        # remove zero bases and normalize across features
        H_samples = H_samples[np.sum(H_samples, axis=1) != 0.0]
        H_samples = normalize(H_samples, norm='l2', axis=1)

        # cluster all bases
        self._logger.info("clustering bases samples")
        labels = self.cluster.fit_predict(H_samples)

        # compute consensus bases from clusters
        cluster_ids = np.unique(labels[labels != -1])
        n_clusters = cluster_ids.size
        self._logger.info("found %d bases, computing consensus bases" %
                          n_clusters)
        H_cons = np.zeros((n_clusters, n_features))

        for c_id in cluster_ids:
            H_cons[c_id, :] = self.cons_meth(H_samples[labels == c_id], axis=0)

        # re-normalize across features
        H_cons = normalize(H_cons, norm='l2', axis=1)

        self.components_ = H_cons
        self.n_components = self.components_.shape[0]
        self.bases_samples_ = H_samples
        self.bases_samples_labels_ = labels
        self.bootstraps_ = rep_idx
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

        Returns
        -------
        W : array-like, shape (n_samples, n_components)
            Transformed data (coefficients of bases).
        """
        check_is_fitted(self, ['components_'])
        n_samples, n_features = X.shape

        if n_features != self.components_.shape[1]:
            raise ValueError(
                'Incompatible shape: cannot reconstruct with %s and %s'
                % (X.shape, self.components_.shape))

        H_t = self.components_.T
        W = np.zeros((n_samples, self.n_components), dtype=X.dtype)

        # calculate coefficients for each sample
        for sample in range(n_samples):
            W[sample] = self.nnreg(H_t, X[sample])

        # calculate reconstruction error
        if reconstruction_err:
            self.reconstruction_err_ = np.linalg.norm(
                X - self.inverse_transform(W))

    def fit_transform(self, X, y=None, reconstruction_err=True, verbose=None):
        """
        Transform the data X according to the fitted UoI-NMF model

        Args:
            X : array-like; shape (n_samples, n_features)
            y
                ignored
            reconstruction_err : bool
                True to compute reconstruction error, False otherwise.
                default True.
        Returns:
            W : array-like; shape (n_samples, n_components)
                Transformed data.
        """
        self.fit(X, verbose=verbose)
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
        check_is_fitted(self, ['components_'])
        n_samples = W.shape[1]

        if n_samples != self.components_.shape[0]:
            raise ValueError(
                'Incompatible shape: cannot multiply %s with %s.'
                % (W.shape, self.components_.shape))

        return np.matmul(W, self.components_)


class UoI_NMF(UoI_NMF_Base):
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

    nmf_init :  "random" | "nndsvd" |  "nndsvda" | "nndsvdar" | "custom"
        Method used to initialize the NMF procedure.
        Default: "random".
        Valid options:

        * "random": non-negative random matrices, scaled with
            :code:`sqrt(X.mean() / n_components)`

        * "nndsvd": Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        * "nndsvda": NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        * "nndsvdar": NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        * "custom": use custom matrices W and H

    nmf_solver : 'cd' | 'mu'
        Default: 'mu'.
        Numerical solver to use for NMF:
        'cd' is a Coordinate Descent solver.
        'mu' is a Multiplicative Update solver.

    nmf_beta_loss : float or string, default 'kullback-leibler'
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver.

    nmf_tol : float, default: 1e-4
        Tolerance of the stopping condition for NMF algorithm.

    nmf_max_iter : integer, default: 200
        Maximum number of iterations before timing out in NMF.

    db_eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood in the DBSCAN algorithm.

    db_min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    db_metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square. X may be a sparse matrix, in which case only "nonzero"
        elements may be considered neighbors for DBSCAN.

    db_metric_params : dict, optional
        Additional keyword arguments for the metric function.

    db_algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.

    db_leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.
    """
    def __init__(
        self, n_boots, ranks=None,
        nmf_init='random', nmf_solver='mu', nmf_beta_loss='kullback-leibler',
        nmf_tol=0.0001, nmf_max_iter=400,
        db_eps=0.5, db_min_samples=None, db_metric='euclidean',
        db_metric_params=None, db_algorithm='auto', db_leaf_size=30,
        random_state=None
    ):
        # create NMF solver
        nmf = skNMF(init=nmf_init,
                    solver=nmf_solver,
                    beta_loss=nmf_beta_loss,
                    tol=nmf_tol,
                    max_iter=nmf_max_iter)

        # create DBSCAN solver
        if db_min_samples is None:
            db_min_samples = n_boots / 2
        dbscan = DBSCAN(eps=db_eps,
                        min_samples=db_min_samples,
                        metric=db_metric,
                        metric_params=db_metric_params,
                        algorithm=db_algorithm)

        super(UoI_NMF, self).__init__(
            n_boots=n_boots,
            ranks=ranks,
            nmf=nmf,
            cluster=dbscan,
            nnreg=None,
            cons_meth=np.median,
            random_state=random_state
        )
