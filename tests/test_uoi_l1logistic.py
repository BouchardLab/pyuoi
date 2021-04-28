import pytest, numbers, warnings
import numpy as np

from numpy.testing import assert_array_equal, assert_allclose, assert_equal

from scipy.sparse import rand as sprand
from scipy import optimize

from pyuoi import UoI_L1Logistic
from pyuoi.linear_model.logistic import (fit_intercept_fixed_coef,
                                         MaskedCoefLogisticRegression,
                                         LogisticInterceptFitterNoFeatures,
                                         _logistic_regression_path,
                                         _multinomial_loss_grad,
                                         _logistic_loss_and_grad)
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import (compute_class_weight,
                           check_consistent_length, check_array)
from sklearn.exceptions import ConvergenceWarning

from pyuoi.datasets import make_classification
from pyuoi.lbfgs import fmin_lbfgs, AllZeroLBFGSError


def _logistic_regression_path_old(X, y, Cs=48, fit_intercept=True,
                                  max_iter=100, tol=1e-4, verbose=0, coef=None,
                                  class_weight=None, penalty='l2',
                                  multi_class='auto',
                                  check_input=True,
                                  sample_weight=None,
                                  l1_ratio=None, coef_mask=None):
    """Compute a Logistic Regression model for a list of regularization
    parameters.

    This is the original function used to check the new indexing-based
    version rather than the masking version implemented here.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Input data.
    y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Input data, target values.
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
    multi_class : str, {'multinomial', 'auto'}, default: 'auto'
        For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'auto' selects binary if the data is binary
        and otherwise selects 'multinomial'.
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
    """
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

    if multi_class == 'auto':
        if len(classes) > 2:
            multi_class = 'multinomial'
        else:
            multi_class = 'ovr'

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
        class_weight_ = compute_class_weight(class_weight, classes=classes, y=y)
        sample_weight *= class_weight_[le.fit_transform(y)]

    # For doing a ovr, we need to mask the labels first. for the
    # multinomial case this is not necessary.
    if multi_class == 'ovr':
        coef_size = n_features
        w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
        mask_classes = np.array([-1, 1])
        mask = (y == 1)
        y_bin = np.ones(y.shape, dtype=X.dtype)
        y_bin[~mask] = -1.
        # for compute_class_weight

        if class_weight == "balanced":
            class_weight_ = compute_class_weight(class_weight,
                                                 classes=mask_classes,
                                                 y=y_bin)
            sample_weight *= class_weight_[le.fit_transform(y_bin)]

    else:
        coef_size = classes.size * n_features
        lbin = OneHotEncoder(categories=[range(classes.size)], sparse=False)
        Y_multi = lbin.fit_transform(y[:, np.newaxis])
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
            coefs.append(multi_w0.copy())
        else:
            if coef_mask is not None:
                w0[:n_features] *= coef_mask
            coefs.append(w0.copy())

        n_iter[i] = n_iter_i

    return np.array(coefs), np.array(Cs), n_iter


def test_fit_intercept_fixed_coef():
    """Test that the intercept in fit correctly for fixed coefficients."""
    X = np.zeros((6, 5))
    coef = np.ones((1, 5))
    y = np.ones(6, dtype=int)
    y[:3] = 0
    b = fit_intercept_fixed_coef(X, coef, y, 2)
    assert_allclose(b, 0.)

    X = np.zeros((7, 5))
    y = np.ones(7, dtype=int)
    y[:3] = 0
    b = fit_intercept_fixed_coef(X, coef, y, 3)
    assert_allclose(b.argmax(), 1)
    assert_allclose(b.argmin(), 2)


def test_fit_intercept_no_features():
    """Test that the intercept in fit correctly for fixed coefficients."""
    X = np.zeros((5, 1))
    y = np.ones(6, dtype=int)
    y[:3] = 0
    LR = LogisticInterceptFitterNoFeatures(y, 1)
    b = LR.intercept_
    assert_allclose(b, 0.)

    y = np.ones(7, dtype=int)
    y[:3] = 0
    LR = LogisticInterceptFitterNoFeatures(y, 1)
    yhat = LR.predict(X)
    assert_allclose(yhat, 1)
    py = LR.predict_proba(X)
    assert np.all(py > .5)

    y = np.ones(7, dtype=int)
    y[:3] = 0
    LR = LogisticInterceptFitterNoFeatures(y, 3)

    yhat = LR.predict(X)
    assert_allclose(yhat, 1)

    py = LR.predict_proba(X)
    assert_allclose(py.argmax(axis=1), 1)
    assert_allclose(py.argmin(axis=1), 2)


def test_l1logistic_intercept():
    """Test that binary L1 Logistic fits an intercept when run."""
    for fi in [True, False]:
        X, y, w, b = make_classification(n_samples=100,
                                         random_state=11,
                                         n_features=4,
                                         w_scale=4.,
                                         include_intercept=fi)
        l1log = UoI_L1Logistic(fit_intercept=fi,
                               n_boots_sel=3,
                               n_boots_est=3).fit(X, y)
        if not fi:
            assert_array_equal(l1log.intercept_, 0.)
        else:
            l1log.intercept_


def test_l1logistic_binary():
    """Test that binary L1 Logistic runs in the UoI framework."""
    n_inf = 10
    X, y, w, b = make_classification(n_samples=200,
                                     random_state=6,
                                     n_informative=n_inf,
                                     n_features=20,
                                     w_scale=4.,
                                     include_intercept=True)

    l1log = UoI_L1Logistic(random_state=10).fit(X, y)
    l1log = UoI_L1Logistic(random_state=10, fit_intercept=False).fit(X, y)
    l1log.predict_proba(X)
    l1log.predict_log_proba(X)
    y_hat = l1log.predict(X)
    assert_equal(accuracy_score(y, y_hat), l1log.score(X, y))
    assert (np.sign(abs(w)) == np.sign(abs(l1log.coef_))).mean() >= .8


def test_l1logistic_binary_multinomial():
    """Test that binary L1 Logistic runs in the UoI framework
    using multi_class='multinomial'."""
    n_inf = 10
    X, y, w, b = make_classification(n_samples=200,
                                     random_state=6,
                                     n_informative=n_inf,
                                     n_features=20,
                                     w_scale=4.,
                                     include_intercept=True)

    UoI_L1Logistic(random_state=10, multi_class='multinomial').fit(X, y)
    UoI_L1Logistic(random_state=10, fit_intercept=False,
                   multi_class='multinomial').fit(X, y)


def test_l1logistic_no_ovr():
    """Test that binary L1 Logistic model raises an error for
    multiclass='ovr'."""
    with pytest.raises(ValueError):
        UoI_L1Logistic(multi_class='ovr')


def test_l1logistic_multiclass():
    """Test that multiclass L1 Logistic runs in the UoI framework when all
       classes share a support."""
    n_features = 20
    n_inf = 10
    X, y, w, b = make_classification(n_samples=200,
                                     random_state=10,
                                     n_classes=5,
                                     n_informative=n_inf,
                                     n_features=n_features,
                                     shared_support=True,
                                     w_scale=4.)
    l1log = UoI_L1Logistic().fit(X, y)
    l1log.predict_proba(X)
    l1log.predict_log_proba(X)
    y_hat = l1log.predict(X)
    assert_equal(accuracy_score(y, y_hat), l1log.score(X, y))
    assert (np.sign(abs(w)) == np.sign(abs(l1log.coef_))).mean() >= .8


def test_l1logistic_multiclass_not_shared():
    """Test that multiclass L1 Logistic runs in the UoI framework when all
       classes share a support."""
    n_features = 20
    n_inf = 10
    X, y, w, b = make_classification(n_samples=400,
                                     random_state=10,
                                     n_classes=5,
                                     n_informative=n_inf,
                                     n_features=n_features,
                                     shared_support=False,
                                     w_scale=4.)
    l1log = UoI_L1Logistic(shared_support=False).fit(X, y)
    l1log.predict_log_proba(X)
    y_hat = l1log.predict(X)
    assert_equal(accuracy_score(y, y_hat), l1log.score(X, y))
    assert (np.sign(abs(w)) == np.sign(abs(l1log.coef_))).mean() >= .7


def test_masked_logistic():
    """Test the masked logistic regression class."""
    n_features = 20
    n_inf = 10
    for shared_support in [True, False]:
        for n_classes in [2, 3]:
            for intercept in [True, False]:
                X, y, w, b = make_classification(n_samples=200,
                                                 random_state=10,
                                                 n_classes=n_classes,
                                                 n_informative=n_inf,
                                                 n_features=n_features,
                                                 shared_support=shared_support,
                                                 include_intercept=intercept,
                                                 w_scale=4.)
                mask = np.squeeze(np.logical_not(np.equal(w, 0)))
                for penalty in ['l1', 'l2']:
                    lr = MaskedCoefLogisticRegression(penalty=penalty, C=10.,
                                                      warm_start=True,
                                                      fit_intercept=intercept)
                    lr.fit(X, y, coef_mask=mask)
                    coef_idxs = np.flatnonzero(np.equal(lr.coef_, 0.))
                    coef_idxs = set(coef_idxs.tolist())
                    mask_idxs = np.flatnonzero(np.equal(mask, 0))
                    mask_idxs = set(mask_idxs.tolist())
                    assert mask_idxs.issubset(coef_idxs)
                    lr.fit(X, y, coef_mask=mask)


def test_masked_logistic_standardize():
    """Test the masked logistic regression class with `standardize=True`."""
    n_features = 20
    n_inf = 10
    for shared_support in [True, False]:
        for n_classes in [2, 3]:
            for intercept in [True, False]:
                X, y, w, b = make_classification(n_samples=200,
                                                 random_state=10,
                                                 n_classes=n_classes,
                                                 n_informative=n_inf,
                                                 n_features=n_features,
                                                 shared_support=shared_support,
                                                 include_intercept=intercept,
                                                 w_scale=4.)
                mask = np.squeeze(np.logical_not(np.equal(w, 0)))
                for penalty in ['l1', 'l2']:
                    lr = MaskedCoefLogisticRegression(penalty=penalty, C=10.,
                                                      warm_start=True,
                                                      fit_intercept=intercept,
                                                      standardize=True)
                    lr.fit(X, y, coef_mask=mask)
                    coef_idxs = np.flatnonzero(np.equal(lr.coef_, 0.))
                    coef_idxs = set(coef_idxs.tolist())
                    mask_idxs = np.flatnonzero(np.equal(mask, 0))
                    mask_idxs = set(mask_idxs.tolist())
                    assert mask_idxs.issubset(coef_idxs)
                    lr.fit(X, y, coef_mask=mask)


@pytest.mark.parametrize("n_classes,penalty,fit_intercept", [(3, "l2", True),
                                                             (3, "l2", False),
                                                             (3, "l1", True),
                                                             (3, "l1", False),
                                                             (2, "l2", True),
                                                             (2, "l2", False),
                                                             (2, "l1", True),
                                                             (2, "l1", False)])
def test_masking_with_indexing(n_classes, penalty, fit_intercept):
    """Check that indexing the masks gives the same results as masking with
    logistic regression.
    """
    X, y, w, intercept = make_classification(n_samples=1000,
                                             n_classes=n_classes,
                                             n_features=20,
                                             n_informative=10,
                                             random_state=0)
    mask = w != 0.
    if n_classes == 2:
        mask = mask.ravel()
    coefs, _, _ = _logistic_regression_path(X, y, [10.], coef_mask=mask,
                                            penalty=penalty,
                                            fit_intercept=fit_intercept)
    coefs_old, _, _ = _logistic_regression_path_old(X, y, [10.], coef_mask=mask,
                                                    penalty=penalty,
                                                    fit_intercept=fit_intercept)
    assert_allclose(coefs, coefs_old)
    coefs, _, _ = _logistic_regression_path(X, y, [10.],
                                            penalty=penalty,
                                            fit_intercept=fit_intercept)
    coefs_old, _, _ = _logistic_regression_path_old(X, y, [10.],
                                                    penalty=penalty,
                                                    fit_intercept=fit_intercept)
    assert_allclose(coefs, coefs_old)


@pytest.mark.parametrize("n_classes,penalty,fit_intercept", [(3, "l2", True),
                                                             (3, "l2", False),
                                                             (3, "l1", True),
                                                             (3, "l1", False),
                                                             (2, "l2", True),
                                                             (2, "l2", False),
                                                             (2, "l1", True),
                                                             (2, "l1", False)])
def test_all_masked_with_indexing(n_classes, penalty, fit_intercept):
    """Check masking all of the coef either works with intercept or raises an error.
    """
    X, y, w, intercept = make_classification(n_samples=1000,
                                             n_classes=n_classes,
                                             n_features=20,
                                             n_informative=10,
                                             random_state=0)
    mask = np.zeros_like(w)
    if n_classes == 2:
        mask = mask.ravel()
    coefs, _, _ = _logistic_regression_path(X, y, [10.], coef_mask=mask,
                                            fit_intercept=fit_intercept)
    if fit_intercept:
        if n_classes == 2:
            assert_equal(coefs[0][:-1], 0.)
        else:
            assert_equal(coefs[0][:, :-1], 0.)
    else:
        assert_equal(coefs[0], 0.)


def test_estimation_score_usage():
    """Test the ability to change the estimation score in UoI L1Logistic"""
    methods = ('acc', 'log', 'BIC', 'AIC', 'AICc')
    X, y, w, b = make_classification(n_samples=200,
                                     random_state=6,
                                     n_informative=5,
                                     n_features=10)
    scores = []
    for method in methods:
        l1log = UoI_L1Logistic(random_state=12, estimation_score=method,
                               tol=1e-2, n_boots_sel=24, n_boots_est=24)
        assert_equal(l1log.estimation_score, method)
        l1log.fit(X, y)
        scores.append(l1log.scores_)
    scores = np.stack(scores)
    assert_equal(len(np.unique(scores, axis=0)), len(methods))


def test_set_random_state():
    """Tests whether random states are handled correctly."""
    X, y, w, b = make_classification(n_samples=100,
                                     random_state=60,
                                     n_informative=4,
                                     n_features=5,
                                     w_scale=4.)
    # same state
    l1log_0 = UoI_L1Logistic(random_state=13)
    l1log_1 = UoI_L1Logistic(random_state=13)
    l1log_0.fit(X, y)
    l1log_1.fit(X, y)
    assert_array_equal(l1log_0.coef_, l1log_1.coef_)

    # different state
    l1log_1 = UoI_L1Logistic(random_state=14)
    l1log_1.fit(X, y)
    assert not np.array_equal(l1log_0.coef_, l1log_1.coef_)

    # different state, not set
    l1log_0 = UoI_L1Logistic()
    l1log_1 = UoI_L1Logistic()
    l1log_0.fit(X, y)
    l1log_1.fit(X, y)
    assert not np.array_equal(l1log_0.coef_, l1log_1.coef_)


def test_normalization_by_samples():
    """Test that coef_ does not depend directly on the number of samples."""
    n_features = 20
    for n_classes in [2, 3]:
        X, y, w, b = make_classification(n_samples=200,
                                         random_state=10,
                                         n_classes=n_classes,
                                         n_informative=n_features,
                                         n_features=n_features,
                                         w_scale=4.)
        for penalty in ['l1', 'l2']:
            lr1 = MaskedCoefLogisticRegression(penalty=penalty, C=1e2)
            lr1.fit(X, y)

            lr3 = MaskedCoefLogisticRegression(penalty=penalty, C=1e2)
            lr3.fit(np.tile(X, (3, 1)), np.tile(y, 3))
            assert_allclose(lr1.coef_, lr3.coef_)


def test_l1logistic_binary_strings():
    """Test that binary L1 Logistic runs in the UoI framework."""
    n_inf = 10
    X, y, w, b = make_classification(n_samples=200,
                                     random_state=6,
                                     n_informative=n_inf,
                                     n_features=20,
                                     w_scale=4.,
                                     include_intercept=True)

    classes = ['a', 'b']
    lb = LabelEncoder()
    lb.fit(classes)
    y = lb.inverse_transform(y)

    l1log = UoI_L1Logistic(random_state=10).fit(X, y)
    y_hat = l1log.predict(X)
    assert set(classes) >= set(y_hat)


def test_l1logistic_multiclass_strings():
    """Test that multiclass L1 Logistic runs in the UoI framework when all
       classes share a support."""
    n_features = 20
    n_inf = 10
    X, y, w, b = make_classification(n_samples=200,
                                     random_state=10,
                                     n_classes=5,
                                     n_informative=n_inf,
                                     n_features=n_features,
                                     shared_support=True,
                                     w_scale=4.)
    classes = ['a', 'b', 'c', 'd', 'e']
    lb = LabelEncoder()
    lb.fit(classes)
    y = lb.inverse_transform(y)

    l1log = UoI_L1Logistic(random_state=10).fit(X, y)
    y_hat = l1log.predict(X)
    assert set(classes) >= set(y_hat)


def test_l1logistic_sparse_input():
    """Test that multiclass L1 Logistic works when using sparse matrix
       inputs"""
    rs = np.random.RandomState(17)
    X = sprand(100, 100, random_state=rs)
    classes = ['abc', 'de', 'fgh']
    y = np.array(classes)[rs.randint(3, size=100)]

    kwargs = dict(
        fit_intercept=False,
        random_state=rs,
        n_boots_sel=4,
        n_boots_est=4,
        n_C=7,
    )
    l1log = UoI_L1Logistic(**kwargs).fit(X, y)

    y_hat = l1log.predict(X)
    assert set(classes) >= set(y_hat)


def test_l1logistic_sparse_input_no_center():
    """Test that multiclass L1 Logistic raises an error when asked to center
    sparse data.
    """
    rs = np.random.RandomState(17)
    X = sprand(10, 10, random_state=rs)
    classes = ['abc', 'de', 'fgh']
    y = np.array(classes)[rs.randint(3, size=10)]

    with pytest.raises(ValueError):
        UoI_L1Logistic(fit_intercept=True).fit(X, y)


def test_l1logistic_bad_est_score():
    """Test that multiclass L1 Logistic raises an error when given a bad
    estimation_score value.
    """
    X = np.random.randn(20, 5)
    y = np.ones(20)

    with pytest.raises(ValueError):
        UoI_L1Logistic(estimation_score='z',
                       n_boots_sel=10, n_boots_est=10).fit(X, y)


def test_reg_params():
    """Test whether the upper bound on the regularization parameters correctly
    zero out the coefficients."""
    n_features = 20
    n_inf = 10
    n_classes = 5
    X, y, w, b = make_classification(n_samples=200,
                                     random_state=101,
                                     n_classes=n_classes,
                                     n_informative=n_inf,
                                     n_features=n_features,
                                     shared_support=True)

    uoi_log = UoI_L1Logistic()
    uoi_log.output_dim = n_classes
    reg_params = uoi_log.get_reg_params(X, y)
    C = reg_params[0]['C']
    # check that coefficients get set to zero
    lr = MaskedCoefLogisticRegression(penalty='l1',
                                      C=0.99 * C,
                                      standardize=False,
                                      fit_intercept=True)
    lr.fit(X, y)
    assert_equal(lr.coef_, 0.)

    # check that coefficients above the bound are not set to zero
    lr = MaskedCoefLogisticRegression(penalty='l1',
                                      C=1.01 * C,
                                      standardize=False,
                                      fit_intercept=True)
    lr.fit(X, y)
    assert np.count_nonzero(lr.coef_) > 0


def test_fit_intercept():
    """Tests whether `include_intercept` in passed through to the linear models.
    """
    lr = UoI_L1Logistic(fit_intercept=True)
    assert lr._selection_lm.fit_intercept
    assert lr._estimation_lm.fit_intercept

    lr = UoI_L1Logistic(fit_intercept=False)
    assert not lr._selection_lm.fit_intercept
    assert not lr._estimation_lm.fit_intercept
