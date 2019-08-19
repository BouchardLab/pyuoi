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

To estimate the coefficients and intercept from the data, we create a
``UoI_Lasso`` object with desired hyperparameters and apply the ``fit``
function:

.. code:: python

    uoi = UoI_Lasso(n_boots_sel=30, n_boots_est=30,
                    estimation_frac=0.8, selection_Frac=0.8,
                    stability_selection=1.,
                    standardize=True, fit_intercept=True)
    uoi.fit(X, y)

In the above fitting procedure, we use:

* 30 bootstraps in the selection module (``n_boots_sel``)

* 30 bootstraps in the estimation module (``n_boots_est``)

* 80% of the data in each selection bootstrap (``selection_frac``)

* 80% of the data in each estimation bootstrap (``estimation_frac``)

* A hard intersection (``stability_selection=1``).

Note that ``PyUoI`` uses the ``standardize`` argument in place of ``normalize``
as in ``scitkit-learn``, though the functionality is the same.

Logistic Regression
-------------------


Poisson Regression
------------------


Non-negative Matrix Factorization
---------------------------------

