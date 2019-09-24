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

* **Model Estimation:** For each support :math:`S_k`, perform Ordinary Least
  Squares (OLS) on :math:`N_E` resamples of the data, using only the features
  in :math:`S_k`. The final model is obtained by averaging across the fitted
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
Logistic regression can be divided into two cases: **Bernoulli logistic
regression**, or a binary choice, and **multinomial logistic regression**,
where there are multiple classes. In the Bernoulli, or binary case, the model
is given by

.. math::
    \begin{align}
        P(y=1|\mathbf{x};\boldsymbol{\beta}) = \hat y &=
        \sigma\left(\beta_0 + \sum_{i=1}^M x_i \beta_i\right)\qquad \text{with}\\
        \sigma(r) &= \frac{1}{1+\exp(-r)}.
    \end{align}

Meanwhile, for multinomial logistic regression (multiclass), the model is
defined as

.. math::
    \begin{equation}
        P(y_i=1|\mathbf{x}, \boldsymbol{\beta}) = \hat y_i = \displaystyle\begin{cases}
        \frac{1}{C} \exp\left(\sum_{j=1}^M x_j \beta_{ij} \right), & i=1\\
        \frac{1}{C} \exp\left(\beta_i + \sum_{j=1}^M x_j \beta_{ij}\right), & i>1
        \end{cases},
    \end{equation}

where

.. math::
    \begin{align}
        C &= \exp\left(\sum_{j=1}^M x_j \beta_{1j}\right)
        + \sum_{i \neq 1} \exp\left(\beta_i + \sum_{j=1}^M x_j \beta_{ij} \right).
    \end{align}

Note that we have set the first class' intercept to zero to account for
overparameterization. The parameters are found by minimizing the penalized
negative log-likelihood. The penalized negative log-likelihood for one sample
is given by

.. math::
    \begin{align}
        \text{cost}(\mathbf{x}, y_i=1) = \begin{cases}
        \frac{1}{N}\left(\log C - \sum_{j=1}^M x_j \beta_{ij}\right) + \lambda \sum_{j=1}^M |\beta_{ij}|, & i=1\\
        \frac{1}{N}\left(\log C - \beta_i - \sum_{j=1}^M x_j \beta_{ij}\right) + \lambda \sum_{j=1}^M |\beta_{ij}|, & i>1
        \end{cases}.
    \end{align}

In UoI, this cost function is solved using a modified orthant-wise limited
memory quasi-Newton method.

Poisson Regression
^^^^^^^^^^^^^^^^^^
In Poisson Regression, we assume that the response variable follows a Poisson
distribution, with mean equal to the linear combination of the features and
the parameters:

.. math::
    \begin{align}
        P(y|\mathbf{x}, \boldsymbol{\beta}) &= \frac{1}{y!} \lambda^y e^{-\lambda} \\
        \lambda &= \beta_0 + \sum_{i=1}^M x_i \beta_i.
    \end{align}

Thus, we desire the parameters that minimize the penalized average
log-likelihood:

.. math::
    \begin{align}
        \boldsymbol{\beta}^* &= \underset{\boldsymbol{\beta}}{\text{argmin }}
        \left\{-\frac{1}{N}\sum_{i=1}^{N}\left[y_i \left(\beta_0 + \sum_{j=1}^M x_{ij} \beta_j\right)
        -\exp\left(\beta_0 + \sum_{j=1}^M x_{ij} \beta_j\right) \right] \right. \\
        & \qquad \qquad \qquad + \left. \lambda \left(\alpha \sum_{j=1}^M |\beta_j| + \frac{1}{2}(1-\alpha)
        \sum_{j=1}^M \beta_j^2\right)\right\}.
    \end{align}

Thus, the UoI fitting procedure proceeds similarly as with
UoI\ :sub:`ElasticNet`, except a different cost function is optimized over.
While this can be achieved with coordinate descent, similar to lasso, we have
found better results using an modified orthant-wise limited memory quasi-Newton
Method.

Dimensionality Reduction
------------------------
Dimensionality reduction techniques do not fit as directly in the UoI
framework, as described above.

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

The UoI procedure for CSS, is detailed in the pseudocode below. Briefly, the
algorithm extracts columns which persist across resamples of the data matrix
while combining columns selected across different SVD ranks. The algorithm can
accept any number of ranks of unionize over, though the default is to unionize
over :math:`k\in \left\{1, \ldots, K\right\}` where :math:`K` is some maximum
rank.

.. code:: python

    def UoI_CSS(A, K, c, n_bootstraps):
        # iterate over bootstraps
        for j in range(n_bootstraps):
            Aj = Generate resample of the data matrix A
            # iterate over ranks
            for k in K:
                Ci = CSS(Aj, k, c)
        C = union(intersection(Ci))
        return C

Non-negative Matrix Factorization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A non-negative matrix factorization consists of finding a parts-based
decomposition of some data matrix :math:`A \in \mathbb{R}^{m\times n}_+`.
This can be posed as a non-convex optimization problem, solving for the matrices
:math:`W \in \mathbb{R}_+^{m\times k}` and :math:`H \in \mathbb{R}_+^{k\times n}`
such that:

.. math::
    \begin{align}
        \min_{W\geq 0, H\geq 0} ||A - WH||_F
    \end{align}

where :math:`F` denotes the Frobenius norm. Here, the rows of :math:`H` form
some basis of the objects (of which there are :math:`k`) while the rows of
:math:`W` are the weights of the basis in :math:`A`. Importantly, :math:`W` and
:math:`H` are both non-negative, so the parts in NMF are often more
interpretable.

There are a variety of algorithms and approaches to both choose the correct
number of components and estimate the values of :math:`W` and :math:`H`. In the
UoI framework, basis estimation and weight estimation are separated into distinct
modules (similar to the linear models). NMF is fit to many bootstraps of the
data matrix, using a desired approach (in PyUoI, it defaults to a symmetric KL
divergence loss with multiplicative update rules). The fitted bases across
bootstraps are aggregated to form the final bases, which can then be used to
extract the weights.

Specifically, during basis estimation, NMF is fit to many bootstraps of the data matrix across
a variety of ranks. The bases fit will tend to form clusters near the bases
that would be fit if the entire data matrix is used. Then, the final rank is
chosen by evaluating a dissimilarity metric, which prefers ranks that result
in tight basis clusters. The final bases are chosen using a clustering algorithm
-- DBSCAN -- to identify clusters of bases across bootstraps. A consensus procedure,
such as the median, extracts the actual basis from each cluster. Finally, given
a set of bases :math:`H`, the weights can be determined by using non-negative
least squares.

The above procedure is detailed in the following pseudocode:

.. code:: python

    def UoI_NMF(A, ranks, n_bootstraps):
        # iterate over bootstraps
        for k in ranks:
            for i in range(n_bootstraps):
                Aj = Generate bootstrapped resample of the data matrix A
                Hi, Wi = NMF(A, k)
        
        Compute diss(Hi^k, H_j^k) and Gamma(k) and choose the best rank k_hat
        let Hk denote the bases fitted from rank k_hat

        # cluster the best set of bases
        Cluster Hk using DBSCAN
        Set centers of the clusters as the best bases H

        # fit W
        Fit W using non-negative least squares between A and H

        return W, H

Above, the dissimilarity between two sets of bases from different bootstrapped
data matrices :math:`H, H'` with rank :math:`k` is given by

.. math::
    \begin{align}
        \text{diss}(H, H') = 1 - \frac{1}{2k} \left(
            \sum_{j=1}^k \text{max}_i C_{ij} + \sum_{i=1}^k \text{max}_j C_{ij}
        \right)
    \end{align}

where :math:`C_{ij}` is the cross-correlation matrix between :math:`H` and
:math:`H'`. The discrepancy :math:`\Gamma(k)`, which aggregates dissimilarities
across pairs of bootstraps, is given by

.. math::
    \begin{align}
        \Gamma(k) = \sum_{1 \leq i \leq j \leq N_B} \text{diss}(H_i, H_j)
    \end{align}

where :math:`N_B` is the number of boostraps.
