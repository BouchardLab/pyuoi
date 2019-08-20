.. PyUoI

========
Examples
========

Specific UoI algorithms can be imported similarly as done in ``scikit-learn``.
For example, UoI\ :sub:`Lasso` is imported from the ``linear_model`` library as

.. code:: python

    from pyuoi.linear_model import UoI_Lasso

while UoI\ :sub:`NMF` would be imported from the ``decomposition`` library:

.. code:: python

    from pyuoi.decomposition import UoI_NMF

Lasso Regression
----------------

PyUoI comes equipped with utility functions to generate synthetic data for
testing. For a simple linear model, we can use the ``make_linear_regression``
function:

.. code:: python

    from pyuoi.utils import make_linear_regression
    from pyuoi.linear_model import UoI_Lasso

    X, y, beta, intercept = make_linear_regression(n_samples=100, n_features=5,
                                                   n_informative=2)

Performing the fit
^^^^^^^^^^^^^^^^^^

To estimate the coefficients and intercept from the data, we create a
``UoI_Lasso`` object with desired hyperparameters and apply the ``fit``
function:

.. code:: python

    uoi = UoI_Lasso(n_boots_sel=30, n_boots_est=30,
                    estimation_frac=0.8, selection_Frac=0.8,
                    stability_selection=1.,
                    standardize=True, fit_intercept=True,
                    solver='cd')
    uoi.fit(X, y)

In the above fitting procedure, we use:

* 30 bootstraps in the selection module (``n_boots_sel``)

* 30 bootstraps in the estimation module (``n_boots_est``)

* 80% of the data in each selection bootstrap (``selection_frac``)

* 80% of the data in each estimation bootstrap (``estimation_frac``)

* A hard intersection (``stability_selection=1``).

* Coordinate descent solver (``solver='cd'``) provided by ``scikit-learn``.

Note that ``PyUoI`` uses the ``standardize`` argument in place of ``normalize``
as in ``scitkit-learn``, though the functionality is the same.
UoI\ :sub:`Lasso` can also be run with the Pycasso solver.

Accessing the fit
^^^^^^^^^^^^^^^^^

Logistic Regression
-------------------
Performing logistic regression in the PyUoI framework is almost exactly the
same as in UoI\ :sub:`Lasso`, with a few differences in keywords. The
``make_classification`` utility can be used to generate the data. First, we can
try binary logistic regression:

.. code:: python

    from pyuoi.utils import make_classification
    from pyuoi.linear_model import UoI_L1Logistic

    X, y, beta, intercept = make_classification(
        n_samples=200,
        random_state=6,
        n_informative=10,
        n_features=20,
        w_scale=4.,
        include_intercept=True)
    uoi = UoI_L1Logistic(n_boots_est=30, n_boots_sel=30,
                         estimation_score='log', n_C=30)
    uoi.fit(X, y)


A few differences from the Lasso case are worth pointing out:

* The estimation score, can be set to the either the log-likelihood (and penalized versions thereof) or the accuracy 

* hello

jello

Poisson Regression
------------------


Non-negative Matrix Factorization
---------------------------------

