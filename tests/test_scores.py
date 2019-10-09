import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, log_loss

from pyuoi.utils import log_likelihood_glm
from pyuoi.utils import (AIC, BIC, AICc)

from pyuoi.linear_model import (UoI_Lasso, UoI_L1Logistic, UoI_Poisson,
                                UoI_ElasticNet)


def test_ll():
    """Tests that the log-likelihood for generalized linear models is correctly
    calculated."""

    # identity
    y_true = np.array([1, 2, 3])
    y_pred = np.array([np.e + 1, np.e + 2, np.e + 3])
    ll = log_likelihood_glm('normal', y_true, y_pred)
    assert_almost_equal(ll, -4.5)

    # poisson
    y_true = np.array([1 / np.log(2.), 1 / np.log(3.), 1 / np.log(4.)])
    y_pred = np.array([2., 3., 4.])
    ll = log_likelihood_glm('poisson', y_true, y_pred)
    assert_almost_equal(ll, -2)

    # poisson with all zeros
    y_true = np.zeros(3)
    y_pred = np.zeros(3)
    ll = log_likelihood_glm('poisson', y_true, y_pred)
    assert_equal(ll, 0.)

    # poisson with all zeros, but predicted is not all zeros
    y_pred = np.zeros(3)
    y_true = np.array([0., 0., 1.])
    ll = log_likelihood_glm('poisson', y_true, y_pred)
    assert_equal(ll, -np.inf)


def test_ll_error():
    """Tests that the log-likelihood function correctly raises an error when an
    incorrect string is passed as a parameter."""

    y_true = np.array([1., 2., 3.])
    y_pred = np.array([3., 4., 5.])

    assert_raises(ValueError,
                  log_likelihood_glm,
                  'error',
                  y_true,
                  y_pred)


def test_information_criteria():
    """Tests the information criteria (AIC, AICc, BIC) functions."""
    ll = -1.
    n_features = 5
    n_samples = 1000

    aic = AIC(ll, n_features)
    assert_equal(aic, 12.)

    aicc = AICc(ll, n_features, n_samples)
    assert_equal(aicc, 12. + 30. / 497.)

    # additional test: AICc should equal AIC if the number of samples is one
    # greater than the number of features
    aicc = AICc(ll, n_features, n_features + 1)
    assert_equal(aicc, aic)

    bic = BIC(ll, n_features, n_samples)
    assert_equal(bic, 5 * np.log(1000) + 2)


def test_LinearRegressor_scoring_defaults():
    """Tests that the correct default train/test data are being used
    for scoring estimates in UoIAbstractLinearRegressor. Further
    tests that the scoring itself is being done correctly."""
    seed = 5

    X, y = make_regression(n_samples=100, n_features=10, n_informative=10,
                           random_state=seed)

    train_idxs, test_idxs = train_test_split(np.arange(X.shape[0]),
                                             test_size=0.1,
                                             random_state=seed)
    X_train = X[train_idxs]
    y_train = y[train_idxs]

    X_test = X[test_idxs]
    y_test = y[test_idxs]

    fitter = LinearRegression().fit(X_train, y_train)
    support = np.ones(X.shape[1]).astype(bool)
    # r2 - must use test data
    uoi = UoI_Lasso(estimation_score='r2')
    assert(uoi._estimation_target == 1)

    score = uoi._score_predictions('r2', fitter, X, y, support,
                                   (train_idxs, test_idxs))
    assert_equal(r2_score(y_test, fitter.predict(X_test)), score)

    ll = log_likelihood_glm('normal', y_train,
                            fitter.predict(X_train[:, support]))
    # BIC - must use train data
    uoi = UoI_Lasso(estimation_score='BIC')
    assert(uoi._estimation_target == 0)
    score = -1 * uoi._score_predictions('BIC', fitter, X, y, support,
                                        (train_idxs, test_idxs))
    assert_equal(BIC(ll, *X_train.T.shape), score)

    # AIC - must use train data
    uoi = UoI_Lasso(estimation_score='AIC')
    assert(uoi._estimation_target == 0)

    score = -1 * uoi._score_predictions('AIC', fitter, X, y, support,
                                        (train_idxs, test_idxs))
    assert_equal(AIC(ll, X_train.shape[1]), score)

    # AICc - must use train data
    uoi = UoI_Lasso(estimation_score='AICc')
    assert(uoi._estimation_target == 0)

    score = -1 * uoi._score_predictions('AICc', fitter, X, y, support,
                                        (train_idxs, test_idxs))
    assert_equal(AICc(ll, *X_train.T.shape), score)


def test_GeneralizedLinearRegressor_scoring_defaults():
    """Tests that the correct default train/test data are being used
    for scoring estimates in UoIAbstractGeneralizedLinearRegressor. Further
    tests that the scoring itself is being done correctly."""
    seed = 5

    X, y = make_classification(n_samples=100, n_features=3, n_informative=3,
                               n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=2, random_state=seed)

    train_idxs, test_idxs = train_test_split(np.arange(X.shape[0]),
                                             test_size=0.1,
                                             random_state=seed)

    X_train = X[train_idxs]
    y_train = y[train_idxs]

    X_test = X[test_idxs]
    y_test = y[test_idxs]

    fitter = LogisticRegression().fit(X_train, y_train)
    support = np.ones(X.shape[1]).astype(bool)

    # acc - must use test data
    uoi = UoI_L1Logistic(estimation_score='acc')
    assert(uoi._estimation_target == 1)
    uoi.classes_ = np.unique(y)
    score = uoi._score_predictions('acc', fitter, X, y, support,
                                   (train_idxs, test_idxs))
    assert_equal(accuracy_score(y_test, fitter.predict(X_test)), score)

    # log - must use test data. Note the sign difference
    uoi = UoI_L1Logistic(estimation_score='log')
    assert(uoi._estimation_target == 1)
    uoi.classes_ = np.unique(y)
    score = uoi._score_predictions('log', fitter, X, y, support,
                                   (train_idxs, test_idxs))

    y_pred_test = fitter.predict_proba(X_test[:, support])
    assert_equal(log_loss(y_test, y_pred_test, labels=np.unique(y)),
                 -1 * score)

    ll = -log_loss(y_train, fitter.predict_proba(X_train[:, support]),
                   labels=np.unique(y))
    total_ll = ll * X_train.shape[0]
    # BIC - must use train data
    uoi = UoI_L1Logistic(estimation_score='BIC')
    assert(uoi._estimation_target == 0)
    uoi.classes_ = np.unique(y)
    score = -1 * uoi._score_predictions('BIC', fitter, X, y, support,
                                        (train_idxs, test_idxs))
    assert_equal(BIC(total_ll, *X_train.T.shape), score)

    # AIC
    uoi = UoI_L1Logistic(estimation_score='AIC')
    assert(uoi._estimation_target == 0)
    uoi.classes_ = np.unique(y)
    score = -1 * uoi._score_predictions('AIC', fitter, X, y, support,
                                        (train_idxs, test_idxs))
    assert_equal(AIC(total_ll, X_train.shape[1]), score)

    # AICc
    uoi = UoI_L1Logistic(estimation_score='AICc')
    assert(uoi._estimation_target == 0)
    uoi.classes_ = np.unique(y)
    score = -1 * uoi._score_predictions('AICc', fitter, X, y, support,
                                        (train_idxs, test_idxs))
    assert_equal(AICc(total_ll, *X_train.T.shape), score)


def test_estimation_target():
    """Verify the ability for the user to set the estimation taget variable"""

    # Assess r2 on train data
    uoi = UoI_Lasso(estimation_score='r2', estimation_target='train')

    # train gets converted to the index 0
    assert(uoi._estimation_target == 0)

    # Assess BIC on test data
    uoi = UoI_Lasso(estimation_score='BIC', estimation_target='test')

    # Assess r2 on train data
    uoi = UoI_ElasticNet(estimation_score='r2', estimation_target='train')

    # train gets converted to the index 0
    assert(uoi._estimation_target == 0)

    # Assess BIC on test data
    uoi = UoI_ElasticNet(estimation_score='BIC', estimation_target='test')

    assert(uoi._estimation_target == 1)

    uoi = UoI_L1Logistic(estimation_score='acc', estimation_target='train')

    assert(uoi._estimation_target == 0)

    uoi = UoI_L1Logistic(estimation_score='BIC', estimation_target='test')

    assert(uoi._estimation_target == 1)

    uoi = UoI_Poisson(estimation_score='acc', estimation_target='train')

    assert(uoi._estimation_target == 0)

    uoi = UoI_Poisson(estimation_score='BIC', estimation_target='test')

    assert(uoi._estimation_target == 1)
