from pyuoi.linear_model.lasso import PycLasso
from sklearn.datasets import make_regression
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.linear_model import lasso_path
import numpy as np
import pytest

try:
    import pycasso
except ImportError:
    pycasso = None


def l1_loss(y, X, coef, alpha):
    return np.linalg.norm(y - X @ coef)**2 + \
        alpha * np.linalg.norm(coef, 1)


@pytest.mark.skipif(pycasso is None, reason='pycasso not installed')
def test_pyclasso():
    """Test that pyclasso gives the same answers as sklearn lasso"""

    X, y, w = make_regression(coef=True, random_state=1)
    alphas = _alpha_grid(X, y, n_alphas=10)

    _, skl_coefs, _ = lasso_path(X, y, alphas=alphas)

    pyc_lasso = PycLasso(alphas=alphas)

    pyc_lasso.fit(X, y)

    # Compare l1 loss
    skl_losses = np.array([l1_loss(y, X, skl_coefs.T[i, :], alphas[i])
                           for i in range(alphas.size)])

    pyc_losses = np.array([l1_loss(y, X, pyc_lasso.coef_[i, :], alphas[i])
                           for i in range(alphas.size)])

    # Assert losses are within 0.5 %
    prcnt_diff = np.abs(np.array([((skl_losses[i] - pyc_losses[i])
                                 / skl_losses[i])
                                 for i in range(skl_losses.size)]))

    assert(np.all(prcnt_diff < 5e-2))
