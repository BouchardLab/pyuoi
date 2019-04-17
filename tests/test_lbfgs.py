import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from pyuoi.lbfgs import LBFGS, fmin_lbfgs
from pyuoi.lbfgs._lowlevel import LBFGSError
import numpy as np


def test_fmin_lbfgs():
    def f(x, g, *args):
        g[0] = 2 * x
        return x ** 2

    xmin = fmin_lbfgs(f, 100., line_search='armijo')
    assert_array_equal(xmin, [0])

    xmin = fmin_lbfgs(f, 100., line_search='strongwolfe')
    assert_array_equal(xmin, [0])


class TestOWLQN:

    def test_owl_qn_end(self):
        def f(x, g, *args):
            g[:] = 2. * (x - 1.)
            return np.sum((x - 1.) ** 2)

        xmin = fmin_lbfgs(f, np.zeros(10), orthantwise_c=1.,
                          orthantwise_end=5)
        assert_array_equal(xmin[5:], 1.)
        assert np.all(xmin[:5] < 1.)

    def test_owl_qn(self):
        def f(x, g, *args):
            g[0] = 2 * x
            return x ** 2

        xmin = fmin_lbfgs(f, 100., orthantwise_c=1, line_search='wolfe')
        assert_array_equal(xmin, [0])

    def test_owl_line_search_warning_explicit(self):
        def f(x, g, *args):
            g[0] = 2 * x
            return x ** 2

        with pytest.warns(UserWarning, match="OWL-QN"):
            fmin_lbfgs(f, 100., orthantwise_c=1, line_search='morethuente')
        with pytest.warns(UserWarning, match="OWL-QN"):
            fmin_lbfgs(f, 100., orthantwise_c=1, line_search='armijo')
        with pytest.warns(UserWarning, match="OWL-QN"):
            fmin_lbfgs(f, 100., orthantwise_c=1, line_search='strongwolfe')

    @pytest.mark.xfail(strict=True)
    def test_owl_wolfe_no_warning(self):
        """ This test is an attempt to show that wolfe throws no warnings.
        """

        def f(x, g, *args):
            g[0] = 2 * x
            return x ** 2

        with pytest.warns(UserWarning, match="OWL-QN"):
            fmin_lbfgs(f, 100., orthantwise_c=1, line_search='wolfe')


def test_2d():
    def f(x, g, f_calls):
        assert x.shape == (2, 2)
        assert g.shape == x.shape
        g[:] = 2 * x
        f_calls[0] += 1
        return (x ** 2).sum()

    def progress(x, g, fx, xnorm, gnorm, step, k, ls, *args):
        assert x.shape == (2, 2)
        assert g.shape == x.shape

        assert np.sqrt((x ** 2).sum()) == xnorm
        assert np.sqrt((g ** 2).sum()) == gnorm

        p_calls[0] += 1
        return 0

    f_calls = [0]
    p_calls = [0]

    xmin = fmin_lbfgs(f, [[10., 100.], [44., 55.]], progress, args=[f_calls])
    assert f_calls[0] > 0
    assert p_calls[0] > 0
    assert_array_almost_equal(xmin, [[0, 0], [0, 0]])


def test_class_interface():
    def f(x, g, *args):
        g[:] = 4 * x
        return x ** 4 + 1

    opt = LBFGS()
    opt.max_iterations = 3

    assert_array_equal(opt.minimize(f, 1e6), [0])

    opt.max_iterations = 1
    with pytest.raises(LBFGSError):
        opt.minimize(f, 1e7)


def test_input_validation():
    with pytest.raises(TypeError):
        fmin_lbfgs([], 1e4)
    with pytest.raises(TypeError):
        fmin_lbfgs(lambda x: x, 1e4, "ham")
    with pytest.raises(TypeError):
        fmin_lbfgs(lambda x: x, "spam")
