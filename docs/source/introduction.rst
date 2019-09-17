.. PyUoI

======================================
Introduction to Union of Intersections
======================================

The Union of Intersections (UoI) is a flexible, modular, and scalable framework
capable of enhancing both the identification of features (model selection) as
well as the estimation of the contributions of these features
(model estimation) within a model-fitting procedure.

Methods built on top of the UoI framework (e.g. regression, classification,
dimensionality reduction) leverage stochastic data resampling and a range of
sparsity-inducing regularization parameters/dimensions to build families of
potential feature sets robust to resamples (i.e., perturbations) of the data,
and then average nearly unbiased parameter estimates of selected features to
maximize predictive accuracy.

In simpler terms: scientific data is often noisy, and we are generally
interested in identifying features in a model that are stable to that noise.
UoI uses novel aggregating procedures within a bootstrapping framework to do
both sparse (i.e., getting rid of features that aren't robust to the noise)
and predictive (i.e., keeping features that are informative) model fitting.

PyUoI has implementations of the UoI variants for several lasso or
elastic-net penalized generalized linear models (linear, logistic, and Poisson
regression) as well as dimensionality reduction techniques, including CUR
(column subset selection) and non-negative matrix factorization. Below, we
detail the UoI framework in these contexts.

Linear Models
-------------

We first consider the ``linear_model`` module, which contains implementations to
fit several penalized (generalized) linear models.

Lasso Regression
^^^^^^^^^^^^^^^^^

Consider a linear model with :math:`M` features:

.. math::

    \begin{align}
        y &= \sum_{i=1}^{M} x_i \beta_i + \epsilon,
    \end{align}

where :math:`\left\{ x_i \right\}_{i=1}^M` are the features, :math:`y` is the
response variable, and :math:`\epsilon` is variability uncaptured by our model.

A common approach to fitting the parameters :math:`\beta` relies on the lasso
penalty, which we apply to a typical mean squared (reconstruction) error:

.. math::

    \begin{align}
        \boldsymbol{\beta}^* &= \underset{\boldsymbol{\beta}}{\text{argmax }}
        \Big\{
        |\mathbf{y} - \mathbf{X} \boldsymbol{\beta}|^2 +
        \lambda |\boldsymbol{\beta}|_1
        \Big\}.
    \end{align}

The addition of the L1 penalty on the parameters has the impact of setting some
subset of them exactly equally to zero, thereby performing model selection. The
parameter :math:`\lambda` specifies how strongly we want to enforce sparsity,
and is typically chosen through cross-validation.

In UoI\ :sub:`Lasso`, we separate model selection and model estimation. As in
cross-validation, a sweep of :math:`\lambda` values is chosen to 


Logistic Regresion
^^^^^^^^^^^^^^^^^^


Poisson Regression
^^^^^^^^^^^^^^^^^^

Dimensionality Reduction
------------------------

CUR Decomposition (Column Subset Selection)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Non-negative Matrix Factorization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

