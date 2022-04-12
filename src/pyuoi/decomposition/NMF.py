import scipy.optimize as spo
import numpy as np
import logging

from .base import AbstractDecompositionModel
from .utils import dissimilarity

from ..utils import check_logger

from itertools import combinations
from sklearn.base import BaseEstimator
from sklearn.decomposition import NMF as skNMF
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_non_negative
from sklearn.utils.validation import check_is_fitted


class UoI_NMF_Base(AbstractDecompositionModel):
    """Performs non-negative matrix factorization in the Union of Intersections
    framework.

    This base class accepts objects or functions that perform the NMF fitting,
    clustering, non-negative regression, and consensus grouping.

    Parameters
    ----------
    n_boots : int
        The number of bootstraps to use for model selection.
    ranks : int, list, or None
        The range of k to use. If *ranks* is an int, range(2, ranks + 1) will
        be used. If not specified, range(X.shape[1]) will be used.
    nmf : NMF object
        The NMF object to use to perform fitting.
        Note: this class must take *n_components* as an argument.
    cluster : Clustering object
        Clustering object to use. If None, defaults to DBSCAN.
    nnreg : NNLS object
        Non-negative regressor to use. If None, defaults to
        scipy.optimize.nnls.
    cons_meth : function
        The method for computing consensus bases after clustering. If None,
        uses np.mean.
    use_dissimilarity : bool
        Whether to use dissimilarity to choose the final rank. If False, all
        bases across ranks are concatenated and clustered. The final rank in
        this case is how many clusters are chosen.
    random_state : int, RandomState instance, or None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.
    logger : Logger
        The logger to use for messages when ``verbose=True`` in ``fit``.
        If *None* is passed, a logger that writes to ``sys.stdout`` will be
        used.
    """
    def __init__(
        self, n_boots=10, ranks=None, nmf=None, cluster=None, nnreg=None,
        cons_meth=None, use_dissimilarity=True, random_state=None, logger=None
    ):
        super(UoI_NMF_Base, self).__init__()
        self.__initialize(
            n_boots=n_boots,
            ranks=ranks,
            nmf=nmf,
            cluster=cluster,
            nnreg=nnreg,
            cons_meth=cons_meth,
            logger=logger,
            random_state=random_state)
        if self.n_boots == 1 and use_dissimilarity:
            raise ValueError("Cannot use dissimilarity to compute best rank "
                             "with a single bootstrap")
        self.use_dissimilarity = use_dissimilarity

    def set_params(self, **kwargs):
        """Set the parameters of this estimator."""
        super(UoI_NMF_Base, self).set_params(**kwargs)
        self.__initialize(**self.get_params())

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
            elif isinstance(ranks, (list, tuple, range, np.ndarray)):
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
            if isinstance(nnreg, BaseEstimator):
                self.nnreg = lambda A, B: nnreg.fit(A, B).coef_
            elif callable(nnreg):
                self.nnreg = nnreg
            else:
                raise ValueError("Unrecognized regressor.")

        # initialize method for computing consensus H bases after clustering
        if cons_meth is None:
            # default uses mean
            self.cons_meth = np.mean
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

        self._logger = check_logger(logger, 'uoi_decomposition', self.comm)

    def fit(self, X, verbose=False):
        r"""Compute the basis matrix on the provided data matrix using the
        UoI\ :sub:`NMF` algorithm.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data matrix to be decomposed.
        verbose : bool
            If True, outputs status updates.
        """
        if verbose:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.WARNING)

        check_non_negative(X, 'UoI NMF')
        n_samples, n_features = X.shape

        H_samples = {k: np.zeros((self.n_boots, k, n_features))
                     for k in self.ranks}

        rep_idx = self._rand.randint(n_samples, size=(self.n_boots, n_samples))
        for boot_idx in range(self.n_boots):
            self._logger.info("Bootstrap %d" % boot_idx)
            # compute NMF bases for k across bootstrap replicates
            sample = X[rep_idx[boot_idx]]
            for k_idx, k in enumerate(self.ranks):
                # concatenate k by p
                H_samples[k][boot_idx] = \
                    self.nmf.set_params(n_components=k).fit(sample).components_

        if self.use_dissimilarity:
            gamma = np.zeros(len(self.ranks))
            # iterate over each rank
            for k_idx, k in enumerate(self.ranks):
                # extract the bases for each rank
                H_k = H_samples[k]
                for boot1, boot2 in combinations(range(self.n_boots), 2):
                    gamma[k_idx] += dissimilarity(H_k[boot1], H_k[boot2])
            self.dissimilarity_ = (gamma * 2) /\
                                  (self.n_boots *
                                   (self.n_boots - 1))
            k_min = self.ranks[np.argmin(gamma)]
            H_pre_cluster = H_samples[k_min].reshape((self.n_boots * k_min,
                                                      n_features))
        else:
            H_pre_cluster = np.zeros((self.n_boots * np.sum(self.ranks),
                                      n_features))
            start_idx = 0
            for k in self.ranks:
                H_k = H_samples[k].reshape((self.n_boots * k, n_features))
                end_idx = start_idx + self.n_boots * k
                H_pre_cluster[start_idx:end_idx] = H_k
                start_idx = end_idx

        # remove zero bases and normalize across features
        H_pre_cluster = H_pre_cluster[np.sum(H_pre_cluster, axis=1) != 0.0]
        H_pre_cluster = normalize(H_pre_cluster, norm='l2', axis=1)

        # cluster all bases
        self._logger.info("clustering bases samples")
        labels = self.cluster.fit_predict(H_pre_cluster)

        # compute consensus bases from clusters
        cluster_ids = np.unique(labels[labels != -1])
        n_clusters = cluster_ids.size
        self._logger.info("found %d bases, computing consensus bases" %
                          n_clusters)
        H_cons = np.zeros((n_clusters, n_features))

        for c_id in cluster_ids:
            H_cons[c_id, :] = self.cons_meth(H_pre_cluster[labels == c_id],
                                             axis=0)

        # re-normalize across features
        H_cons = normalize(H_cons, norm='l2', axis=1)

        self.components_ = H_cons
        self.n_components = self.components_.shape[0]
        self.bases_samples_ = H_pre_cluster
        self.bases_samples_labels_ = labels
        self.bootstraps_ = rep_idx
        self.reconstruction_err_ = None
        return self

    def transform(self, X, reconstruction_err=True):
        r"""Transform the data according to the fitted UoI\ :sub:`NMF` model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be decomposed.
        reconstruction_err : bool
            If True, the reconstruction error is computed and stored as a class
            attribute.

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

        return W

    def fit_transform(self, X, reconstruction_err=True, verbose=None):
        r"""Fit and transform the data according to the fitted UoI\ :sub:`NMF`
        model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be decomposed.
        reconstruction_err : bool
            If True, the reconstruction error is computed and stored as a class
            attribute.
        verbose : bool
            If True, outputs status updates.

        Returns
        -------
        W : array-like, shape (n_samples, n_components)
            Transformed data (coefficients of bases).
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
        n_components = W.shape[1]

        if n_components != self.components_.shape[0]:
            raise ValueError(
                'Incompatible shape: cannot multiply %s with %s.'
                % (W.shape, self.components_.shape))

        return np.matmul(W, self.components_)


class UoI_NMF(UoI_NMF_Base):
    r"""Performs non-negative matrix factorization in the Union of
    Intersections framework.

    This derived class uses (and accepts the keyword arguments for)
    ``scikit-learn``’s NMF and DBSCAN objects, ``scipy``’s non-negative least
    squares function, and a mean function for consensus grouping.

    Parameters
    ----------
    n_boots : int
        The number of bootstraps to use for model selection.
    ranks : int, list, or None
        The range of k to use. If *ranks* is an int, range(2, ranks + 1) will
        be used. If not specified, range(X.shape[1]) will be used.
    nmf_init :  "random" | "nndsvd" |  "nndsvda" | "nndsvdar" | "custom"
        Method used to initialize the NMF procedure. Valid options:
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
    nmf_solver : 'cd' | 'mu', optional
        Numerical solver to use for NMF: 'cd' is a Coordinate Descent solver,
        while 'mu' is a Multiplicative Update solver.
    nmf_beta_loss : float or string, optional
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver.
    nmf_tol : float, optional
        Tolerance of the stopping condition for NMF algorithm.
    nmf_max_iter : integer, optional
        Maximum number of iterations before timing out in NMF.
    db_eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood in the DBSCAN algorithm.
    db_min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    db_metric : string, or callable, optional
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
    random_state : int, RandomState instance, or None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.
    logger : Logger
        The logger to use for messages when ``verbose=True`` in ``fit``.
        If *None* is passed, a logger that writes to ``sys.stdout`` will be
        used.
    """
    def __init__(
        self, n_boots, ranks=None,
        nmf_init='random', nmf_solver='mu', nmf_beta_loss='kullback-leibler',
        nmf_tol=0.0001, nmf_max_iter=400,
        db_eps=0.5, db_min_samples=None, db_metric='euclidean',
        db_metric_params=None, db_algorithm='auto', db_leaf_size=30,
        use_dissimilarity=True, random_state=None, logger=None,
        nmf=None, cluster=None, nnreg=None, cons_meth=None
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
            cons_meth=np.mean,
            random_state=random_state,
            use_dissimilarity=use_dissimilarity,
            logger=logger)
