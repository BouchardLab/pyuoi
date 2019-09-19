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

PyUoI contains implementations to the UoI versions of lasso, elastic net,
logistic regression, and Poisson regression. We introduce the UoI framework for
lasso, and briefly elaborate on the cost functions for the other types of
regressions.

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
        \Bigg\{
        \frac{1}{N}\sum_{i=1}^N \left(y_i - \sum_{j=1}^M x_{ij}\beta_j\right)^2+
        \lambda \sum_{j=1}^M |\beta_j|
        \Bigg\}.
    \end{align}

The addition of the L1 penalty on the parameters has the impact of setting some
subset of them exactly equally to zero, thereby performing model selection. The
parameter :math:`\lambda` specifies how strongly we want to enforce sparsity,
and is typically chosen through cross-validation across a set of
hyperparameters :math:`\left\{\lambda_k\right\}_{k=1}^{H}`. Thus, model
selection (choosing the non-zero parameters) in addition to model estimation
(estimating the parameter values).

In the UoI framework, model selection and model estimation are performed
in separate modules. For UoI\ :sub:`Lasso`, the procedure is as follows:

* **Model Selection:** For each :math:`\lambda_k`, generate Lasso estimates
  on $N_S$ resamples of the data. The support :math:`S_k` (i.e., the set of
  non-zero parameters) for :math:`\lambda_k` consists of the features that
  persist in all model fits across the resamples.

* **Model Estimation:** For each support :math:`S_k`, perform Ordinary Least Squares
  (OLS) on :math:`N_E` resamples of the data, using only the features in
  :math:`S_k`. The final model is obtained by averaging across the fitted
  models that optimally predict (according to a specified metric) held-out data 
  for each resample.

Thus, the selection module ensures that, for each :math:`\lambda_j`, only
features that are stable to perturbations in the data (resamples) are allowed
in the support :math:`S_j`. Meanwhile, the estimation module ensures that only
the predictive supports are averaged together in the final model. The degree of
feature compression via intersections (quantified by :math:`N_S`) and the
degree of feature expansion via unions (quantified by :math:`N_E`) can be
balanced to maximize prediction accuracy for the response variable :math:`y`.

Elastic Net
^^^^^^^^^^^
The elastic net penalty includes an additional L2 penalty on the parameter
values:

.. math::

    \begin{align}
        \boldsymbol{\beta}^* &= \underset{\boldsymbol{\beta}}{\text{argmax }}
        \Bigg\{
        \frac{1}{N}\sum_{i=1}^N \left(y_i - \sum_{j=1}^M x_{ij}\beta_j\right)^2+
        \lambda \left(\alpha \sum_{j=1}^M |\beta_j| + \frac{1}{2}(1-\alpha)
        \sum_{j=1}^M \beta_j^2\right)
        \Bigg\}.
    \end{align}

Thus, there are two parameters :math:`\lambda` and :math:`\alpha` to
cross-validate over. In the UoI framework, instead of iterating over each
:math:`\lambda_k`, each pair of hyperparameters is iterated over. Then, the
selection module proceeds normally. The estimation module is unchanged, because
it operates only on unique supports produced by the selection module.

Logistic Regresion
^^^^^^^^^^^^^^^^^^
Logistic regression 

Poisson Regression
^^^^^^^^^^^^^^^^^^

Dimensionality Reduction
------------------------
Dimensionality reduction techniques do not fit as directly in the UoI framework,
as described above. 

CUR Decomposition (Column Subset Selection)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A common dimensionality reduction technique is *column subset selection* (CSS),
the selection of representative features from a data design matrix. Closely
related to CSS is CUR matrix decomposition, where the design matrix is written
as a decomposition of representative columns and rows. Here, we detail how CSS
procedures (and, by extension, CUR decomposition) naturally fit into the UoI
framework.

For a design matrix :math:`A` (with :math:`N` samples and :math:`M` features),
CSS is ordinarily performed by operating on the top right :math:`K` singular
vectors, represented by :math:`V_K`. Thus, :math:`K`, the number of singular
vectors to extract, is an initial hyperparameter of the problem.
To perform CSS, we operate on the *leverage scores* :math:`\ell_i` for each
column (feature) of :math:`A`. The leverage score of column :math:`i` is
defined as the norm of the :math:`i`th row in :math:`V_K`, normalized to
:math:`K`. Column selection is performed via importance sampling, using the
leverage scores, scaled by a constant :math:`c`, as the probability of
selection. The constant :math:`c` denotes the expected number of columns to
select, and is an additional parameter of the algorithm.

Non-negative Matrix Factorization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

