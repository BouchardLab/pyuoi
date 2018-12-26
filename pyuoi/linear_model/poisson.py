import numpy as np

from sklearn.linear_model.base import LinearModel


class Poisson(LinearModel):
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 normalize=True, max_iter=1000, tol=1e-5, warm_start=True):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y, beta_init=None):
        """"""
        self.n_samples, self.n_features = X.shape

        # initialization
        if beta_init is not None:
            beta = beta_init
        else:
            beta = np.random.normal(self.n_features)

        active_idx = np.arange(self.n_features)

        for iteration in range(self.max_iter):
            w, z = self.adjusted_response(X, y, beta)
            beta = self.cd_sweep(
                beta=beta,
                X=X,
                w=w,
                z=z,
                active_idx=active_idx
            )
            active_idx = np.argwhere(beta != 0).ravel()

        return beta

    def cd_sweep(self, beta, X, w, z, active_idx):
        n_active = active_idx.size
        n_features = beta.size
        beta_update = np.zeros(beta.size)

        num_update = np.zeros(n_active)
        den_update = np.zeros(n_active)

        for idx, coordinate in enumerate(active_idx):
            mask = np.ones(n_features)
            mask[coordinate] = 0

            z_tilde = np.dot(X, beta * mask)
            num_update[idx] = np.dot(w, X[:, coordinate] * (z - z_tilde))
            den_update[idx] = np.dot(w, X[:, coordinate]**2)

        updates = self.soft_threshold(num_update, self.l1_ratio * self.alpha) \
            / (den_update + self.alpha * (1 - self.l1_ratio))

        beta_update[active_idx] = updates

        return beta_update

    @staticmethod
    def soft_threshold(X, threshold):
        """Performs the soft-thresholding necessary for coordinate descent
        lasso updates.

        Parameters
        ----------
        X : array-like, shape (n_features, n_samples)
            Matrix to be thresholded.

        threshold : float
            Soft threshold.

        Returns
        -------
        X_soft_threshold : array-like
            Soft thresholded X.
        """
        X_thresholded = (np.abs(X) > threshold).astype('int')
        X_soft_threshold = np.sign(X) * (np.abs(X) - threshold) * X_thresholded
        return X_soft_threshold

    @staticmethod
    def adjusted_response(X, y, beta):
        """Calculates the adjusted response when posing the fitting procedure
        as Iteratively Reweighted Least Squares (Newton update on log
        likelihood).

        Parameters
        ----------
        X : array-like, shape (n_features, n_samples)
            The design matrix.

        y : array-like, shape (n_samples)
            The response vector.

        beta : array-like, shape (n_features)
            Current estimate of the parameters.

        Returns
        -------
        w : array-like, shape (n_samples)
            Weights for samples. The log-likelihood for a GLM, when posed as
            a linear regression problem, requires reweighting the samples.

        z : array-like, shape (n_samples)
            Working response. The linearized response when rewriting coordinate
            descent for the Poisson likelihood as a iteratively reweighted
            least squares.
        """
        Xbeta = np.dot(X, beta)
        w = np.exp(Xbeta)
        z = Xbeta + (y / w) - 1
        return w, z
