import numpy as np

from .base import AbstractDecompositionModel

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .utils import column_select, stability_selection_to_threshold


class UoI_CUR(AbstractDecompositionModel):
    """Performs column subset selection (CUR decomposition) in the Union of
    Intersections framework.

    Parameters
    ----------
    n_boots : int
        Number of bootstraps.
    max_k : int
        The maximum rank of the singular value decomposition.
    boots_frac : float
        The fraction of data to use in the bootstrap.
    algorithm : string, optional
        SVD solver to use. Either “arpack” for the ARPACK wrapper in SciPy
        (scipy.sparse.linalg.svds), or “randomized” for the randomized
        algorithm due to Halko (2009).
    n_iter : int, optional
        Number of iterations for randomized SVD solver. Not used by ARPACK.
        The default is larger than the default in randomized_svd to handle
        sparse matrices that may have large slowly decaying spectrum.
    random_state : int, RandomState instance, or None, optional
        If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by np.random.
    tol : float, optional
        Tolerance for ARPACK. 0 means machine precision. Ignored by
        randomized SVD solver.

    Attributes
    ----------
    components_ : ndarray, shape (n_samples, n_components)
        The selected columns of the design matrix.

    column_indices_ : ndarray, shape (n_components,)
        The indices of the columns selected by the algorithm.
    """
    def __init__(
        self, n_boots, max_k, boots_frac, stability_selection=1.,
        algorithm='randomized', n_iter=5, tol=0.0, random_state=None
    ):
        self.n_boots = n_boots
        self.max_k = max_k
        self.boots_frac = boots_frac
        self.stability_selection = stability_selection
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, ks=None, cs=None, stratify=None):
        """Performs column subset selection in the UoI framework on a provided
        matrix.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The data matrix.

        cs : int, float or None
            The expected number of columns to select. If None, c will vary with
            the rank k.

        ks : int, list, ndarray, or None
            The ranks to consider union over. If None, all ranks from
            (1, ..., max_k) will be used.

        stratify : array-like or None
            Ensures groups of samples are alloted to bootstraps proportionally.
            Labels for each group must be an int greater than zero. Must be of
            size equal to the number of samples, with further restrictions on
            the number of groups.

        Returns
        -------
        union : ndarray, shape (n_components,)
            A numpy array containing the indices of the selected columns.
        """
        ks, cs = self.check_ks_and_cs(ks=ks, cs=cs)
        n_ks = ks.size
        n_samples, n_features = X.shape
        selection_threshold = stability_selection_to_threshold(
            self.stability_selection,
            self.n_boots)

        # keep track of which columns are chosen
        columns = np.zeros((n_ks, self.n_boots, n_features), dtype='int')

        # truncated SVD fitter object
        tsvd = TruncatedSVD(n_components=self.max_k,
                            algorithm=self.algorithm,
                            n_iter=self.n_iter,
                            tol=self.tol,
                            random_state=self.random_state)

        # iterate over bootstraps
        for bootstrap in range(self.n_boots):
            # extract bootstrap
            if self.boots_frac == 1:
                X_train = X
            else:
                X_train, _ = train_test_split(X, test_size=1 - self.boots_frac,
                                              stratify=stratify,
                                              random_state=self.random_state)
            # perform truncated SVD on bootstrap
            tsvd.fit(X_train)
            # extract right singular vectors
            V = tsvd.components_.T

            # iterate over ranks
            for k_idx, k in enumerate(ks):
                c = cs[k_idx]

                column_indices = column_select(V[:, :k], c=c,
                                               random_state=self.random_state)
                # convert column flags to set
                columns[k_idx, bootstrap][column_indices] += 1

        intersection = np.sum(columns, axis=1) >= selection_threshold

        # calculate union of intersections
        union = np.sum(intersection, axis=0) > 0
        self.column_indices_ = np.argwhere(union).ravel()

        if self.column_indices_.size == 0:
            self.components_ = np.array([[]])
        else:
            self.components_ = X[:, self.column_indices_]
        return self

    def check_ks_and_cs(self, ks=None, cs=None):
        """Process the set of ranks to calculate leverage scores over, and the
        expected number of columns for each rank.

        Parameters
        ----------
        ks : ndarray
            The ranks to compute leverage scores over.
        cs : ndarray
            The expected number of columns to select for each rank.

        Returns
        -------
        ks : ndarray
            Processed and checked ranks.
        cs : ndarray
            Processed expected number of columns.
        """
        # convert ks to a numpy array
        if ks is None:
            ks = 1 + np.arange(self.max_k)
        elif isinstance(ks, int):
            ks = np.array([ks])
        elif isinstance(ks, list):
            ks = np.array(ks)
        elif not isinstance(ks, np.ndarray):
            raise ValueError('ks must be a valid int, list or numpy array.')

        # check that the numpy array contains valid values
        if (
            (not np.all(ks <= self.max_k))
            or (not np.all(ks > 0))
            or (not np.issubdtype(ks.dtype, np.integer))
        ):
            raise ValueError('Ranks must be positive, integers, and no more'
                             ' than the max rank.')

        # check expected column numbers
        n_ks = ks.size

        if cs is None:
            cs = ks + 20
        elif isinstance(cs, int) or isinstance(cs, float):
            cs = cs * np.ones(n_ks)
        elif isinstance(cs, list):
            cs = np.array(cs)
        elif not isinstance(cs, np.ndarray):
            raise ValueError('cs must be a valid int, float, list or numpy'
                             ' array.')

        # check that the numpy array contains valid values
        if (
            (not np.all(cs > 0))
            or (not np.issubdtype(cs.dtype, np.number))
            or (not cs.size == ks.size)
        ):
            raise ValueError('Expected number of columns must consist of'
                             ' positive numbers, with total size equal to ks.')
        return ks, cs

    def transform(self, X):
        """Transform the data by extracting the selected columns.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix from which to select columns.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Data matrix comprised of selected columns.
        """
        check_is_fitted(self, ['components_'])
        X = check_array(X)
        X_new = X[:, self.column_indices_]
        return X_new

    def fit_transform(self, X, ks=None, cs=None):
        """Fit and transform the data by choosing and extracting specific
        columns.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix from which to select columns.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Data matrix comprised of selected columns.
        """
        self.fit(X, ks=ks, cs=cs)
        return self.transform(X)


class CUR(AbstractDecompositionModel):
    """Performs ordinary column subset selection through a CUR decomposition.

    Parameters
    ----------
    max_k : int
        The maximum rank of the singular value decomposition.
    algorithm : string, optional
        SVD solver to use. Either “arpack” for the ARPACK wrapper in SciPy
        (scipy.sparse.linalg.svds), or “randomized” for the randomized
        algorithm due to Halko (2009).
    n_iter : int, optional
        Number of iterations for randomized SVD solver. Not used by ARPACK.
        The default is larger than the default in randomized_svd to handle
        sparse matrices that may have large slowly decaying spectrum.
    random_state : int, RandomState instance, or None, optional
        If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by np.random.
    tol : float, optional
        Tolerance for ARPACK. 0 means machine precision. Ignored by
        randomized SVD solver.

    Attributes
    ----------
    components_ : ndarray, shape (n_samples, n_components)
        The selected columns of the design matrix.
    column_indices_ : ndarray, shape (n_components,)
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

        column_indices = column_select(V=V, c=c,
                                       leverage_sort=True,
                                       random_state=self.random_state)

        self.column_indices_ = column_indices
        self.components_ = X[:, self.column_indices_]
        return self

    def transform(self, X):
        """Transform the data by extracting the selected columns.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix from which to select columns.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Data matrix comprised of selected columns.
        """
        check_is_fitted(self, ['column_indices_'])
        X = check_array(X)
        X_new = X[:, self.column_indices_]
        return X_new

    def fit_transform(self, X, c=None):
        """Fit and transform the data by choosing and extracting specific
        columns.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix from which to select columns.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Data matrix comprised of selected columns.
        """
        self.fit(X, c=c)
        return self.transform(X)
