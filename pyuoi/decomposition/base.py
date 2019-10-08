import abc as _abc

from sklearn.linear_model.base import BaseEstimator


class AbstractDecompositionModel(BaseEstimator, metaclass=_abc.ABCMeta):
    @_abc.abstractmethod
    def fit(X):
        """Placeholder for fit. Subclasses should implement this method.
        Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        pass

    @_abc.abstractmethod
    def transform(self, X):
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data matrix to be transformed.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            The transformed data matrix.
        """
        pass

    @_abc.abstractmethod
    def fit_transform(self, X):
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
        pass
