from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import NMF
import scipy.linalg as spla
import scipy.optimize as spo

class UoINMF(BaseEstimator, TransformerMixin):

    def __init__(self, n_boostraps_u=10, n_bootstraps_i=10
                 ranks=None, nmf=None, dbscan=None, lasso=None)
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

        self.n_bootstraps_u = n_boostraps_u
        self.n_bootstraps_i = n_bootstraps_i
        self.fit_frac = fit_frac
        self.ranks = list(range(2,ranks+1)) if isinstance(ranks, int) else list(ranks)
        self.nmf = nmf isinstance(nmf, type) else NMF
        self.dbscan = dbscan

    def __normH(self, H):
        norms = spla.norm(x,axis=1)
        for i in range(H.shape[0]):
            H[i] = H[i]/norms[i]

    def fit(self, X, y=None):
        """
        Perform UoINMF decomposition.

        Args:
            X:  array of shape (n_samples, n_features)
            y:  ignored
        """
        n, p = X.shape
        Wall = list()
        Hall = list()
        ridx = list()
        for k in self.ranks:
            # compute NMF bases for k across bootstrap replicates
            nmf = self.nmf(n_components=k)
            H = np.zeros((self.n_bootstraps_i * p, k))
            rep_idx = np.zeros((n, n), dtype=int)
            W_i = 0
            H_i = 0
            for i in range(self.n_bootstraps_i):
                rep_idx[i] = np.randint(n, size=n)
                nmf.fit_transform(X[rep_idx[i]])                     # n by k
                H[H_i:H_i+p,:] = nmf.components_.T                   # transpose(k by p)
                W_i += n
                H_i += p
            H = H[np.sum(H, axis=1) != 0.0]                          # remove zero bases
            self.__normH(H)                                          # normalize by 2-norm
            labels = self.dbscan.fit_predict(H)                      # cluster all bases

            # compute consensus bases from clusters
            cluster_ids = np.unique([x for x in labels if x != -1])
            nclusters = len(cluster_ids)
            H_cons = np.zeros((nclusters, k))
            for i in cluster_ids:
                H_cons[i,:] = np.mean(H[labels==i],axis=0)
            H_cons = H_cons[np.any(np.isnan(H_cons), axis=1),]        # remove nans
            self.__normH(H_cons)                                      # normalize by 2-norm

            W_boot = list()
            for i in range(rep_idx.shape[0]):
                bs_sample = X[rep_idx[i]]
                W_boot.append(spo.nnls(H_cons.T, X[bs_sample,:].T).T)     # n_prime x k


            Hall.append(H)
            Wall.append(W)
            ridx.append(rep_idx)


