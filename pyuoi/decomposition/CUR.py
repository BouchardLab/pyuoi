import numpy as np

from .base import AbstractDecompositionModel

from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .utils import column_select


class UoI_CUR(BaseEstimator):
    def __init__(
        self, n_boots, max_k, boots_frac, algorithm='randomized',
        n_iter=5, tol=0.0, random_state=None
    ):
        """Performs column subset selection (CUR decomposition) in the Union
        of Intersections framework.

        See Bouchard et al., NIPS, 2017, for more details on the Union of
        Intersections framework.

        Parameters
        ----------
        n_boots : int
            Number of bootstraps.

        max_k : int
            The maximum rank of the singular value decomposition.

        boots_frac : float
            The fraction of data to use in the bootstrap.

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
        self.n_boots = n_boots
        self.max_k = max_k
        self.boots_frac = boots_frac
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

        c : float
            The expected number of columns to select. If None, c will vary with
            the rank k.

        stratify : array-like or None, default None
            Ensures groups of samples are alloted to bootstraps proportionally.
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
        for bootstrap in range(self.n_boots):
            # extract bootstrap
            X_train, _ = train_test_split(X, test_size=1 - self.boots_frac,
                                          stratify=stratify,
                                          random_state=self.random_state)

            # perform truncated SVD on bootstrap
            tsvd.fit(X_train)
            # extract right singular vectors
            V = tsvd.components_.T

            # iterate over ranks
            for k_idx, k in enumerate(range(1, self.max_k + 1)):
                # perform column selection on the subset of singular vectors
                if c is None:
                    column_indices = column_select(V[:, :k], k + 20)
                else:
                    column_indices = column_select(V[:, :k], c)
                # convert column flags to set
                column_indices = set(column_indices)
                indices[k_idx].append(column_indices)

        # calculate intersections
        intersection = [set.intersection(*indices[k_idx])
                        for k_idx in range(self.max_k)]
        # calculate union of intersections
        union = list(set.union(*intersection))

        self.columns_ = np.sort(np.array(union))
        return self


class CUR(AbstractDecompositionModel):
    """Performs ordinary column subset selection through a CUR decomposition.

    Parameters
    ----------
    n_boots : int
        Number of bootstraps.

    max_k : int
        The maximum rank of the singular value decomposition.

    boots_frac : float
        The fraction of data to use in the bootstrap.

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
    def __init__(
        self, max_k, algorithm='randomized', n_iter=5, tol=0.0,
        random_state=None
    ):
        self.max_k = max_k
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, c=None):
        """Performs column subset selection in the UoI framework on a provided
        matrix.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The data matrix.

        c : float
            The expected number of columns to select. If None, c will vary with
            the rank k.

        stratify : array-like or None, default None
            Ensures groups of samples are alloted to bootstraps proportionally.
            Labels for each group must be an int greater than zero. Must be of
            size equal to the number of samples, with further restrictions on
            the number of groups.

        Returns
        -------
        column_indices : ndarray
            A numpy array containing the indices of the selected columns.
        """
        n_samples, n_features = X.shape

        # truncated SVD fitter object
        tsvd = TruncatedSVD(n_components=self.max_k,
                            algorithm=self.algorithm,
                            n_iter=self.n_iter,
                            tol=self.tol,
                            random_state=self.random_state)

        # perform truncated SVD on bootstrap
        tsvd.fit(X)
        # extract right singular vectors
        V = tsvd.components_.T[:, :self.max_k]

        # perform column selection on the subset of singular vectors
        if c is None:
            c = self.max_k + 20

        column_indices = column_select(V=V,
                                       c=self.max_k + 20,
                                       leverage_sort=True)

        self.column_indices_ = column_indices
        self.components_ = X[:, self.column_indices_]
        return self

    def transform(self, X):
        """Transform the data X according to the fitted selected columns.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be decomposed.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self, ['components_'])
        X = check_array(X)

    def fit_transform(self, X, c=None):
        """Transform the data X according to the fitted decomposition.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be decomposed.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X, c=c)
        return self.transform(X)
