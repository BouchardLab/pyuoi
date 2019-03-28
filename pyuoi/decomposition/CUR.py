from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

import numpy as np


class CUR(BaseEstimator):
    def __init__(
        self, n_resamples, max_k, resample_frac, algorithm='randomized',
        n_iter=5, tol=0.0, random_state=None
    ):
        """Performs column subset selection (CUR decomposition) in the Union
        of Intersections framework.

        See Bouchard et al., NIPS, 2017, for more details on the Union of
        Intersections framework.

        Parameters
        ----------
        n_resamples : int
            Number of resamples (bootstraps).

        max_k : int
            The maximum rank of the singular value decomposition.

        resample_frac : float
            The fraction of data to use in the resample.

        algorithm : string, default = “randomized”
            SVD solver to use. Either “arpack” for the ARPACK wrapper in SciPy
            (scipy.sparse.linalg.svds), or “randomized” for the randomized
            algorithm due to Halko (2009).

        n_iter : int, optional (default 5)
            Number of iterations for randomized SVD solver. Not used by ARPACK.
            The default is larger than the default in randomized_svd to handle
            sparse matrices that may have large slowly decaying spectrum.

        random_state : int, RandomState instance or None, optional,
                       default = None
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by np.random.

        tol : float, optional
            Tolerance for ARPACK. 0 means machine precision. Ignored by
            randomized SVD solver.

        Attributes
        ----------
        columns_ : ndarray
            The indices of the columns selected by the algorithm.
        """
        self.n_resamples = n_resamples
        self.max_k = max_k
        self.resample_frac = resample_frac
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, c=None, stratify=None):
        """Performs column subset selection in the UoI framework on a provided
        matrix.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The data matrix.

        stratify : array-like or None, default None
            Ensures groups of samples are alloted to resamples proportionally.
            Labels for each group must be an int greater than zero. Must be of
            size equal to the number of samples, with further restrictions on
            the number of groups.

        Returns
        -------
        union : ndarray
            A numpy array containing the indices of the selected columns.
        """
        n_samples, n_features = X.shape
        # initialize list of lists to contain the extracted column indices
        indices = [[] for _ in range(self.max_k)]

        # truncated SVD fitter object
        tsvd = TruncatedSVD(n_components=self.max_k,
                            algorithm=self.algorithm,
                            n_iter=self.n_iter,
                            tol=self.tol,
                            random_state=self.random_state)

        # iterate over bootstraps
        for bootstrap in range(self.n_resamples):
            # extract resample
            X_train, _ = train_test_split(X, test_size=1 - self.resample_frac,
                                          stratify=stratify,
                                          random_state=self.random_state)

            # perform truncated SVD on resample
            tsvd.fit(X_train)
            # extract right singular vectors
            V = tsvd.components_.T

            # iterate over ranks
            for k_idx, k in enumerate(range(1, self.max_k + 1)):
                # perform column selection on the subset of singular vectors
                if c is None:
                    column_flags = self.column_select(V[:, :k], k + 20)
                else:
                    column_flags = self.column_select(V[:, :k], c)
                # convert column flags to set
                column_indices = set(np.argwhere(column_flags).ravel())
                indices[k_idx].append(column_indices)

        # calculate intersections
        intersection = [set.intersection(*indices[k_idx])
                        for k_idx in range(self.max_k)]
        # calculate union of intersections
        union = list(set.union(*intersection))

        self.columns_ = np.sort(np.array(union))
        return self

    @staticmethod
    def column_select(V, c):
        """Chooses column indices from a matrix given its SVD.

        Parameters
        ----------
        V : ndarray, shape (n_features, rank)
            The set of singular vectors.

        c : float
            The expected number of columns to select.

        Returns
        -------
        column_flags : ndarray of bools, shape (n_features)
            A boolean flag array indicating which columns are selected.
        """
        # extract number of samples and rank
        n_features, k = V.shape

        # calculate normalized leverage score
        pi = np.sum(V**2, axis=1) / k

        # iterate through columns
        column_flags = np.zeros(n_features, dtype=bool)
        for column in range(n_features):
            # Mahoney (2009), eqn 3
            p = min(1, c * pi[column])
            # selected column randomly
            column_flags[column] = p > np.random.rand()
