.. PyUoI

======================================
Introduction to Union of Intersections
======================================

The Union of Intersections (UoI) is a flexible, modular, and scalable framework
capable of enhance both the identification of features (model selection) as
well as the estimation of the contributions of these features
(model estimation).

Methods built on top of the UoI framework (e.g. regression, classification,
dimensionality reduction) leverage stochastic data resampling and a range of
sparsity-inducing regularization parameters/dimensions to build families of
potential feature sets robust to resamples (i.e., perturbations) of the data,
and then average nearly unbiased parameter estimates of selected features to
maximize predictive accuracy.

PyUoI has implementations of the UoI variants for several lasso or
elastic-net penalized generalized linear models (linear, logistic, and Poisson
regression) as well as dimensionality reduction techniques, including CUR
(column subset selection) and non-negative matrix factorization. Below, we
detail the UoI framework in these contexts.

Linear Models
-------------

Dimensionality Reduction
------------------------
ruff
