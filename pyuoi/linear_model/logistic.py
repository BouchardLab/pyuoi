import numbers, warnings

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.svm import l1_min_c
from sklearn.metrics import log_loss
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils import (check_X_y, check_random_state, compute_class_weight,
                           check_consistent_length, check_array)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.extmath import safe_sparse_dot, log_logistic, squared_norm
from sklearn.linear_model.logistic import (_check_multi_class,
                                           _intercept_dot)

from scipy.optimize import minimize
from scipy import optimize
from scipy.special import expit, logsumexp

import numpy as np

from .base import AbstractUoIGeneralizedLinearRegressor
from ..utils import sigmoid, softmax
from ..lbfgs import fmin_lbfgs, AllZeroLBFGSError


class UoI_L1Logistic(AbstractUoIGeneralizedLinearRegressor,
                     LogisticRegression):

    def __init__(self, n_boots_sel=48, n_boots_est=48, selection_frac=0.9,
                 estimation_frac=0.9, n_C=48, stability_selection=1.,
                 warm_start=False, estimation_score='acc', multi_class='auto',
                 copy_X=True, fit_intercept=True, standardize=True,
                 random_state=None, max_iter=10000, tol=1e-3,
                 shared_support=True, comm=None):
        super(UoI_L1Logistic, self).__init__(
            n_boots_sel=n_boots_sel,
            n_boots_est=n_boots_est,
            selection_frac=selection_frac,
            estimation_frac=estimation_frac,
            stability_selection=stability_selection,
            estimation_score=estimation_score,
            random_state=random_state,
            copy_X=copy_X,
            fit_intercept=fit_intercept,
            standardize=standardize,
            shared_support=shared_support,
            comm=comm)
        self.n_C = n_C
        self.Cs = None
        self.multi_class = multi_class
        self.tol = tol
        self.solver = 'lbfgs'
        self._selection_lm = MaskedCoefLogisticRegression(
            penalty='l1',
            max_iter=max_iter,
            warm_start=warm_start,
            random_state=random_state,
            multi_class=multi_class,
            fit_intercept=fit_intercept,
            tol=tol)

        self._estimation_lm = MaskedCoefLogisticRegression(
            C=np.inf,
            random_state=random_state,
            multi_class='auto',
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol)

    def get_reg_params(self, X, y):
        if self.Cs is None:
            self.Cs = l1_min_c(X, y, loss='log') * np.logspace(0, 7, self.n_C)
        ret = list()
        for c in self.Cs:
            ret.append(dict(C=c))
        return ret

    def _fit_intercept_no_features(self, y):
        """"Fit a model with only an intercept.

        This is used in cases where the model has no support selected.
        """
        return LogisticInterceptFitterNoFeatures(y, self.output_dim)

    def _fit_intercept(self, X, y):
        """"Fit a model with an intercept and fixed coefficients.

        This is used to re-fit the intercept after the coefficients are
        estimated.
        """
        if self.fit_intercept:
            self.intercept_ = fit_intercept_fixed_coef(X, self.coef_, y,
                                                       self.output_dim)
        else:
            self.intercept_ = np.zeros(self.output_dim)

    def _pre_fit(self, X, y):
        X, y = super()._pre_fit(X, y)
        le = LabelEncoder()
        y = le.fit_transform(y)
        self.classes_ = le.classes_
        if self.classes_.size > 2:
            self.output_dim = self.classes_.size
        elif self.multi_class == 'multinomial':
            self.output_dim = 2
        else:
            self.output_dim = 1
        return X, y


def fit_intercept_fixed_coef(X, coef_, y, output_dim):
    """Optimize the likelihood w.r.t. the intercept for a logistic
    model."""
    if output_dim == 1:
        def f_df(intercept):
            py = sigmoid(X.dot(coef_.T) + intercept)
            dfdb = py.mean() - y.mean()
            return log_loss(y, py), np.atleast_1d(dfdb)

        opt = minimize(f_df, np.atleast_1d(np.zeros(1)),
                       method='L-BFGS-B', jac=True)
        return opt.x
    else:
        def f_df(short_intercept):
            intercept = np.concatenate([np.atleast_1d(1.), short_intercept])
            py = softmax(X.dot(coef_.T) + intercept)

            def dlogpi_dintk(ii, pyi):
                if ii == 0:
                    return -pyi[1:]
                else:
                    rval = np.eye(output_dim - 1)[ii - 1]
                    rval -= pyi[1:]
                    return rval

            dfdb = np.zeros_like(short_intercept)
            for yi, pyi in zip(y, py):
                dfdb -= dlogpi_dintk(yi, pyi) / y.size
            return log_loss(y, py, labels=np.arange(output_dim)), dfdb
        opt = minimize(f_df, np.atleast_1d(np.zeros(output_dim - 1)),
                       method='L-BFGS-B', jac=True)
        intercept = np.concatenate([np.atleast_1d(1.), opt.x])
        return intercept - intercept.max()


class LogisticInterceptFitterNoFeatures(object):
    """Intercept-only bernoulli logistic regression.

    Parameters
    ----------
    y : ndarray
        Class labels.
    """
    def __init__(self, y, output_dim):
        self.output_dim = output_dim
        eps = 1e-10
        if output_dim == 1:
            p = y.mean(axis=0)
            p = np.minimum(np.maximum(p, eps), 1 - eps)
            self.intercept_ = np.log(p / (1. - p))
        else:
            py = np.equal(y[:, np.newaxis],
                          np.arange(self.output_dim)[np.newaxis]).mean(axis=0)
            n_included = np.count_nonzero(py)
            if n_included < self.output_dim:
                new_mass = eps * (self.output_dim - n_included)
                py *= (1. - new_mass)
                py[np.equal(py, 0.)] = eps
            intercept = np.log(py)
            self.intercept_ = intercept - intercept.max()

    def predict(self, X, mask=None):
        n_samples = X.shape[0]
        if self.output_dim == 1:
            return np.tile(int(self.intercept_ >= 0.), n_samples)
        else:
            return np.tile(int(np.argmax(self.intercept_)), n_samples)

    def predict_proba(self, X, mask=None):
        n_samples = X.shape[0]
        if self.output_dim == 1:
            return np.tile(sigmoid(self.intercept_), n_samples)
        else:
            return np.tile(softmax(self.intercept_)[np.newaxis], (n_samples, 1))


class MaskedCoefLogisticRegression(LogisticRegression):
    """Logistic regression with a binary mask on the coef.
    Parameters
    ----------
    penalty : str
        Type of regularization: 'l1' or 'l2'.
    tol : float, optional (default=1e-4)
        Tolerance for stopping criteria.
    C : float, optional (default=1.0)
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.
    fit_intercept : bool, optional (default=True)
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
    class_weight : dict or 'balanced', optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
        .. versionadded:: 0.17
           *class_weight='balanced'*
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag' or
        'liblinear'.
    max_iter : int, optional (default=100)
        Maximum number of iterations taken for the solvers to converge.
    multi_class : str, {'ovr', 'multinomial', 'auto'}, optional (default='ovr')
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.
        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.20
            Default will change from 'ovr' to 'auto' in 0.22.
    verbose : int, optional (default=0)
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver. See :term:`the Glossary <warm_start>`.
        .. versionadded:: 0.17
           *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.
    n_jobs : int or None, optional (default=None)
        Number of CPU cores used when parallelizing over classes if
        multi_class='ovr'". This parameter is ignored when the ``solver`` is
        set to 'liblinear' regardless of whether 'multi_class' is specified or
        not. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.
        See :term:`Glossary <n_jobs>` for more details.
    """
    def __init__(self, penalty='l2', tol=1e-3, C=1.,
                 fit_intercept=True, class_weight=None,
                 random_state=None, max_iter=10000,
                 multi_class='auto', verbose=0, warm_start=False,
                 n_jobs=None):
        super().__init__(penalty=penalty, tol=tol, C=C,
                         fit_intercept=fit_intercept,
                         class_weight=class_weight,
                         random_state=random_state, max_iter=max_iter,
                         multi_class=multi_class, verbose=verbose,
                         warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None, coef_mask=None):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.
        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
            .. versionadded:: 0.17
               *sample_weight* support to LogisticRegression.
        coef_mask : array-like, shape (n_features), (n_classes, n_features)
                    optional
            Masking array for coef.
        Returns
        -------
        self : object
        Notes
        -----
        """
        solver = 'lbfgs'

        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)"
                             % self.C)
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError("Maximum number of iteration must be positive;"
                             " got (max_iter=%r)" % self.max_iter)
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError("Tolerance for stopping criteria must be "
                             "positive; got (tol=%r)" % self.tol)

        _dtype = np.float64

        X, y = check_X_y(X, y, accept_sparse='csr', dtype=_dtype, order="C",
                         accept_large_sparse=True)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        multi_class = _check_multi_class(self.multi_class, solver,
                                         len(self.classes_))

        n_classes = len(self.classes_)
        if multi_class == 'multinomial' and coef_mask is not None:
            coef_mask = coef_mask.reshape(n_classes, -1)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % classes_[0])

        if len(self.classes_) == 2:
            n_classes = 1
            classes_ = classes_[1:]

        if self.warm_start:
            warm_start_coef = getattr(self, 'coef_', None)
        else:
            warm_start_coef = None
        if warm_start_coef is not None and self.fit_intercept:
            warm_start_coef = np.append(warm_start_coef,
                                        self.intercept_[:, np.newaxis],
                                        axis=1)

        self.coef_ = list()
        self.intercept_ = np.zeros(n_classes)

        # Hack so that we iterate only once for the multinomial case.
        if multi_class == 'multinomial':
            classes_ = [None]
            warm_start_coef = [warm_start_coef]
        if warm_start_coef is None:
            warm_start_coef = [None] * n_classes

        path_func = delayed(_logistic_regression_path)

        # The SAG solver releases the GIL so it's more efficient to use
        # threads for this solver.
        prefer = 'processes'
        fold_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                               **_joblib_parallel_args(prefer=prefer))(
            path_func(X, y, pos_class=class_, Cs=[self.C],
                      fit_intercept=self.fit_intercept,
                      tol=self.tol, verbose=self.verbose,
                      multi_class=multi_class, max_iter=self.max_iter,
                      class_weight=self.class_weight, penalty=self.penalty,
                      check_input=False,
                      random_state=self.random_state, coef=warm_start_coef_,
                      sample_weight=sample_weight, coef_mask=coef_mask)
            for class_, warm_start_coef_ in zip(classes_, warm_start_coef))

        fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
        self.n_iter_ = np.asarray(n_iter_, dtype=np.int32)[:, 0]

        if multi_class == 'multinomial':
            self.coef_ = fold_coefs_[0][0]
        else:
            self.coef_ = np.asarray(fold_coefs_)
            self.coef_ = self.coef_.reshape(n_classes, n_features +
                                            int(self.fit_intercept))

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]

        return self


def _logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                              max_iter=100, tol=1e-4, verbose=0, coef=None,
                              class_weight=None, penalty='l2',
                              multi_class='warn',
                              random_state=None, check_input=True,
                              sample_weight=None,
                              l1_ratio=None, coef_mask=None):
    """Compute a Logistic Regression model for a list of regularization
    parameters.
    This is an implementation that uses the result of the previous model
    to speed up computations along the set of solutions, making it faster
    than sequentially calling LogisticRegression for the different parameters.
    Note that there will be no speedup with liblinear solver, since it does
    not handle warm-starting.
    Read more in the :ref:`User Guide <logistic_regression>`.
    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Input data.
    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Input data, target values.
    pos_class : int, None
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.
    Cs : int | array-like, shape (n_cs,)
        List of values for the regularization parameter or integer specifying
        the number of regularization parameters that should be used. In this
        case, the parameters will be chosen in a logarithmic scale between
        1e-4 and 1e4.
    fit_intercept : bool
        Whether to fit an intercept for the model. In this case the shape of
        the returned array is (n_cs, n_features + 1).
    max_iter : int
        Maximum number of iterations for the solver.
    tol : float
        Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
        will stop when ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.
    verbose : int
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    coef : array-like, shape (n_features,), default None
        Initialization value for coefficients of logistic regression.
        Useless for liblinear solver.
    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
    multi_class : str, {'ovr', 'multinomial', 'auto'}, default: 'ovr'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.
        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.20
            Default will change from 'ovr' to 'auto' in 0.22.
    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag' or
        'liblinear'.
    check_input : bool, default True
        If False, the input arrays X and y will not be checked.
    sample_weight : array-like, shape(n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.
    coef_mask : array-like, shape (n_features), (n_classes, n_features) optional
        Masking array for coef.
    Returns
    -------
    coefs : ndarray, shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept. For
        ``multiclass='multinomial'``, the shape is (n_classes, n_cs,
        n_features) or (n_classes, n_cs, n_features + 1).
    Cs : ndarray
        Grid of Cs used for cross-validation.
    n_iter : array, shape (n_cs,)
        Actual number of iteration for each Cs.
    Notes
    -----
    You might get slightly different results with the solver liblinear than
    with the others since this uses LIBLINEAR which penalizes the intercept.
    .. versionchanged:: 0.19
        The "copy" parameter was removed.
    """
    solver = 'lbfgs'
    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, Cs)

    # Preprocessing.
    if check_input:
        X = check_array(X, accept_sparse='csr', dtype=np.float64,
                        accept_large_sparse=True)
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)
    _, n_features = X.shape

    classes = np.unique(y)
    random_state = check_random_state(random_state)

    multi_class = _check_multi_class(multi_class, solver, len(classes))
    if pos_class is None and multi_class != 'multinomial':
        if (classes.size > 2):
            raise ValueError('To fit OvR, use the pos_class argument')
        # np.unique(y) gives labels in sorted order.
        pos_class = classes[1]

    # If sample weights exist, convert them to array (support for lists)
    # and check length
    # Otherwise set them to 1 for all examples
    if sample_weight is not None:
        sample_weight = np.array(sample_weight, dtype=X.dtype, order='C')
        check_consistent_length(y, sample_weight)
    else:
        sample_weight = np.ones(X.shape[0], dtype=X.dtype)

    # If class_weights is a dict (provided by the user), the weights
    # are assigned to the original labels. If it is "balanced", then
    # the class_weights are assigned after masking the labels with a OvR.
    le = LabelEncoder()
    if isinstance(class_weight, dict) or multi_class == 'multinomial':
        class_weight_ = compute_class_weight(class_weight, classes, y)
        sample_weight *= class_weight_[le.fit_transform(y)]

    # For doing a ovr, we need to mask the labels first. for the
    # multinomial case this is not necessary.
    if multi_class == 'ovr':
        coef_size = n_features
        w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
        mask_classes = np.array([-1, 1])
        mask = (y == pos_class)
        y_bin = np.ones(y.shape, dtype=X.dtype)
        y_bin[~mask] = -1.
        # for compute_class_weight

        if class_weight == "balanced":
            class_weight_ = compute_class_weight(class_weight, mask_classes,
                                                 y_bin)
            sample_weight *= class_weight_[le.fit_transform(y_bin)]

    else:
        coef_size = classes.size * n_features
        lbin = LabelBinarizer()
        Y_multi = lbin.fit_transform(y)
        if Y_multi.shape[1] == 1:
            Y_multi = np.hstack([1 - Y_multi, Y_multi])
        w0 = np.zeros((classes.size, n_features + int(fit_intercept)),
                      dtype=X.dtype)
        w0[:, -1] = LogisticInterceptFitterNoFeatures(y,
                                                      classes.size).intercept_

    if coef is not None:
        # it must work both giving the bias term and not
        if multi_class == 'ovr':
            if coef.size not in (n_features, w0.size):
                raise ValueError(
                    'Initialization coef is of shape %d, expected shape '
                    '%d or %d' % (coef.size, n_features, w0.size))
            w0[:coef.size] = coef
        else:
            # For binary problems coef.shape[0] should be 1, otherwise it
            # should be classes.size.
            n_classes = classes.size
            if n_classes == 2:
                n_classes = 1

            if (coef.shape[0] != n_classes or
                    coef.shape[1] not in (n_features, n_features + 1)):
                raise ValueError(
                    'Initialization coef is of shape (%d, %d), expected '
                    'shape (%d, %d) or (%d, %d)' % (
                        coef.shape[0], coef.shape[1], classes.size,
                        n_features, classes.size, n_features + 1))

            if n_classes == 1:
                w0[0, :coef.shape[1]] = -coef
                w0[1, :coef.shape[1]] = coef
            else:
                w0[:, :coef.shape[1]] = coef

    # Mask initial array
    if coef_mask is not None:
        if multi_class == 'ovr':
            w0[:n_features] *= coef_mask
        else:
            w0[:, :n_features] *= coef_mask

    if multi_class == 'multinomial':
        # fmin_l_bfgs_b and newton-cg accepts only ravelled parameters.
        target = Y_multi
        if penalty == 'l2':
            w0 = w0.ravel()

            def func(x, *args):
                return _multinomial_loss_grad(x, *args)[0:2]
        else:
            w0 = w0.T.ravel().copy()

            def inner_func(x, *args):
                return _multinomial_loss_grad(x, *args)[0:2]

            def func(x, g, *args):
                x = x.reshape(-1, classes.size).T.ravel()
                loss, grad = inner_func(x, *args)
                grad = grad.reshape(classes.size, -1).T.ravel()
                g[:] = grad
                return loss
    else:
        target = y_bin
        if penalty == 'l2':
            func = _logistic_loss_and_grad
        else:
            def func(x, g, *args):
                loss, grad = _logistic_loss_and_grad(x, *args)
                g[:] = grad
                return loss

    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        iprint = [-1, 50, 1, 100, 101][
            np.searchsorted(np.array([0, 1, 2, 3]), verbose)]
        if penalty == 'l2':
            w0, loss, info = optimize.fmin_l_bfgs_b(
                func, w0, fprime=None,
                args=(X, target, 1. / C, coef_mask, sample_weight),
                iprint=iprint, pgtol=tol, maxiter=max_iter)
        else:
            zeros_seen = [0]

            def zero_coef(x, *args):
                if multi_class == 'multinomial':
                    x = x.reshape(-1, classes.size)[:-1]
                else:
                    x = x[:-1]
                now_zeros = np.array_equiv(x, 0.)
                if now_zeros:
                    zeros_seen[0] += 1
                else:
                    zeros_seen[0] = 0
                if zeros_seen[0] > 1:
                    return -2048
            try:
                w0 = fmin_lbfgs(func, w0, orthantwise_c=1. / C,
                                args=(X, target, 0., coef_mask, sample_weight),
                                max_iterations=max_iter,
                                epsilon=tol,
                                orthantwise_end=coef_size,
                                progress=zero_coef)
            except AllZeroLBFGSError:
                w0 *= 0.
            info = None
        if info is not None and info["warnflag"] == 1:
            warnings.warn("lbfgs failed to converge. Increase the number "
                          "of iterations.", ConvergenceWarning)
        # In scipy <= 1.0.0, nit may exceed maxiter.
        # See https://github.com/scipy/scipy/issues/7854.
        if info is None:
            n_iter_i = -1
        else:
            n_iter_i = min(info['nit'], max_iter)

        if multi_class == 'multinomial':
            n_classes = max(2, classes.size)
            if penalty == 'l2':
                multi_w0 = np.reshape(w0, (n_classes, -1))
            else:
                multi_w0 = np.reshape(w0, (-1, n_classes)).T
            if coef_mask is not None:
                multi_w0[:, :n_features] *= coef_mask
            if n_classes == 2:
                multi_w0 = multi_w0[1][np.newaxis, :]
            coefs.append(multi_w0.copy())
        else:
            if coef_mask is not None:
                w0[:n_features] *= coef_mask
            coefs.append(w0.copy())

        n_iter[i] = n_iter_i

    return np.array(coefs), np.array(Cs), n_iter


def _multinomial_loss(w, X, Y, alpha, sample_weight):
    """Computes multinomial loss and class probabilities.
    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,)
        Array of weights that are assigned to individual samples.
    Returns
    -------
    loss : float
        Multinomial loss.
    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities.
    w : ndarray, shape (n_classes, n_features)
        Reshaped param vector excluding intercept terms.
    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_samples, n_features = X.shape
    fit_intercept = w.size == (n_classes * (n_features + 1))
    w = w.reshape(n_classes, -1)
    sample_weight = sample_weight[:, np.newaxis]
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0
    p = safe_sparse_dot(X, w.T)
    p += intercept
    p -= logsumexp(p, axis=1)[:, np.newaxis]
    loss = -(sample_weight * Y * p).sum() / n_samples
    loss += 0.5 * alpha * squared_norm(w)
    p = np.exp(p, p)
    return loss, p, w


def _logistic_loss_and_grad(w, X, y, alpha, mask, sample_weight=None):
    """Computes the logistic loss and gradient.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    mask : array-like, shape (n_features), (n_classes, n_features) optional
        Masking array for coef.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.
    Returns
    -------
    out : float
        Logistic loss.
    grad : ndarray, shape (n_features,) or (n_features + 1,)
        Logistic gradient.
    """
    n_samples, n_features = X.shape
    if mask is not None:
        w[:n_features] *= mask
    grad = np.empty_like(w)

    w, c, yz = _intercept_dot(w, X, y)

    if sample_weight is None:
        sample_weight = np.ones(n_samples)

    # Logistic loss is the negative of the log of the logistic function.
    out = -np.sum(sample_weight * log_logistic(yz)) / n_samples
    out += .5 * alpha * np.dot(w, w)

    z = expit(yz)
    z0 = sample_weight * (z - 1) * y

    grad[:n_features] = (safe_sparse_dot(X.T, z0) / n_samples) + alpha * w
    if mask is not None:
        grad[:n_features] *= mask

    # Case where we fit the intercept.
    if grad.shape[0] > n_features:
        grad[-1] = z0.sum() / n_samples
    return out, grad


def _multinomial_loss_grad(w, X, Y, alpha, mask, sample_weight):
    """Computes the multinomial loss, gradient and class probabilities.
    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    mask : array-like, shape (n_features), (n_classes, n_features) optional
        Masking array for coef.
    sample_weight : array-like, shape (n_samples,)
        Array of weights that are assigned to individual samples.
    Returns
    -------
    loss : float
        Multinomial loss.
    grad : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Ravelled gradient of the multinomial loss.
    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities
    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_samples, n_features = X.shape
    fit_intercept = (w.size == n_classes * (n_features + 1))
    if mask is not None:
        w = w.reshape(n_classes, n_features + bool(fit_intercept))
        w[:, :n_features] *= mask
        w = w.ravel()
    grad = np.zeros((n_classes, n_features + bool(fit_intercept)),
                    dtype=X.dtype)
    loss, p, w = _multinomial_loss(w, X, Y, alpha, sample_weight)
    sample_weight = sample_weight[:, np.newaxis]
    diff = sample_weight * (p - Y)
    grad[:, :n_features] = safe_sparse_dot(diff.T, X) / n_samples
    grad[:, :n_features] += alpha * w
    if mask is not None:
        grad[:, :n_features] *= mask
    if fit_intercept:
        grad[:, -1] = diff.sum(axis=0) / n_samples
    return loss, grad.ravel(), p
