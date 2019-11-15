import pytest
import numpy as np

from numpy.testing import assert_array_equal, assert_allclose, assert_equal

from scipy.sparse import rand as sprand

from pyuoi import UoI_L1Logistic
from pyuoi.linear_model.logistic import (fit_intercept_fixed_coef,
                                         MaskedCoefLogisticRegression,
                                         LogisticInterceptFitterNoFeatures)
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from pyuoi.datasets import make_classification


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
