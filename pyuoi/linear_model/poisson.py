import numpy as np

from sklearn.linear_model.base import LinearModel


class Poisson(LinearModel):
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 normalize=True, max_iter=1000, tol=1e-5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape

        # initialization
        beta0 = np.log(np.mean(y))
        beta = np.zeros(n_features)

        # initially set all parameters to be active
        active_idx = np.ones(n_features)

        for cd_iter in range(max_iter):
            pass


    def _cd_poisson(self, X, y, beta_old):
        n_active = np.count_nonzero(active_idx)

        w, z = adjusted_response(X, y, beta_old)

        # for cd_sweep in range(self.max_iter):

    def soft_threshold(X, threshold):
        X_thresholded = (np.abs(X) > threshold).astype('int')
        return np.sign(X) * (np.abs(X) - threshold) * X_thresholded

    def adjusted_response(X, y, beta):
        Xbeta = np.dot(X, beta)
        w = np.exp(Xbeta)
        z = Xbeta + (y / w) - 1
        return w, z
