---
title: 'PyUoI: The Union of Intersections Framework in Python'
tags:
    - generalized linear models
    - dimensionality reduction
    - sparsity
    - interpretability
    - Python
affiliations:
    - name: Redwood Center for Theoretical Neuroscience, University of California,
            Berkeley, Berkeley, California
      index: 1
    - name:
bibliography: paper.bib
header-includes:
    - \usepackage{algorithm}
    - \usepackage[noend]{algpseudocode}
---

# Summary

The increasing size and complexity of scientific data requires statistical
analysis methods that scale and produce models that are both interpretable and
predictive. Interpretability implies one can interpret the output of the model in
terms of processes generating the data [@murdoch2019]. This typically requires identification of
a small number of features in the actual data and accurate estimation of their
contributions [@bickel2006]. Meanwhile, achieving predictive power requires
optimizing the performance of some statistical measure such as precision,
mean squared error, etc. Across models, there is often a trade-off between interpretability and
predictive power. This trade-off is particularly acute for scientific
applications, where the output of the model is used to provide insight into the
underlying physical processes that generated the data.

The recently introduced Union of Intersections (UoI) is a flexible, modular, and
scalable framework designed to enhance both the identification of features
(model selection) as well as the estimation of the contributions of these
features (model estimation) [@bouchard2017]. UoI-based methods leverage
stochastic data resampling and a range of sparsity-inducing regularization
parameters to build families of potential feature sets robust to perturbations
of the data, and then average nearly unbiased parameter estimates of selected
features to maximize predictive accuracy. Models inferred through the UoI
framework are characterized by their usage of fewer parameters with little or no
loss in predictive accuracy relative to benchmark approaches.

`PyUoI` is a Python package containing implementations of a variety of UoI-based
algorithms, encompassing regression, classification, and dimensionality
reduction. In order to better facilitate its usage, `PyUoI`'s API is structured
similarly to the `scikit-learn` package, which is a commonly used Python machine
learning library [@sklearn_api]. Additionally, because the UoI framework is naturally
scalable, `PyUoI` is equipped with `mpi4py` functionality to parallelize model
fitting on large datasets.

# Background

The Union of Intersections is not a single method or algorithm, but a flexible
statistical framework into which other algorithms can be inserted. In this
section, we briefly describe UoI~Lasso~, the UoI implementation of
lasso penalized regression. UoI~Lasso~ is similar in structure to the
UoI versions of other lasso or elastic net penalized generalized linear models
(logistic and poisson).
We refer the user to existing literature on the UoI variants of column subset
selection and non-negative matrix factorization [@bouchard2017; @ubaru2017].

Linear regression consists of estimating parameters $\beta \in \mathbb{R}^p$
that map a $p$-dimensional vector of features $x \in \mathbb{R}^p$ to the
observation variable $y\in \mathbb{R}$, when the $N$ samples are corrupted by
i.i.d Gaussian noise:

\begin{equation}
y = \beta^T x + \epsilon
\end{equation}

where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ for each sample. When the true
$\beta$ is thought to be sparse (i.e., some subset of the $\beta$ are exactly
zero), an estimate of $\beta$ can be found by solving a constrained optimization
problem of the form

\begin{equation}
\hat{\beta} = \underset{\beta\in \mathbb{R}^p}{\text{argmin}}
                \frac{1}{N}\sum_{i=1}^N(y_i - \beta \cdot x_i)^2
                + \lambda |\beta|_1
\end{equation}

where $|\beta|_1$ is the $\ell_1$-norm of the parameters. The $\ell_1$-norm is a
convenient penalty because it will tend to force parameters to be set exactly
equal to zero, performing feature selection [@tibshirani1994]. Typically,
$\lambda$, the degree to which feature sparsity is enforced, is unknown and must
be determined through cross-validation across a set of hyperparameters
$\left\{\lambda_j\right\}_{j=1}^k$.

The key mathematical idea underlying UoI is to perform model selection through
intersection (compressive) operations and model estimation through union
(expansive) operations, in that order. For UoI~Lasso~, the procedure
is as follows (see Algorithm 1 for a more detailed pseudocode):

* **Model Selection:** For each $\lambda_j$, generate Lasso estimates on $N_S$
  resamples of the data (Line 2). The support $S_j$ (i.e., the set of non-zero
  parameters) for $\lambda_j$ consists of the features that persist in all model
  fits across the resamples (Line 7).
* **Model Estimation:** For each support $S_j$, perform Ordinary Least Squares
  (OLS) on $N_E$ resamples of the data. The final model is obtained by averaging
  across the supports chosen according to some model selection criteria, such as
  optimally predicting on held-out data for each resample (Lines 15-16).

Thus, the selection module ensures that, for each $\lambda_j$, only features
that are stable to perturbations in the data (resamples) are allowed in the
support $S_j$. Meanwhile, the estimation module ensures that only the predictive
supports are averaged together in the final model. The degree of feature
compression via intersections (quantified by $N_S$) and the degree of feature
expansion via unions (quantified by $N_E$) can be balanced to maximize
prediction accuracy for the response variable $y$.

\begin{algorithm}[t]
    \caption{\textsc{UoI-Lasso}}
    \label{alg:uoi}
    \hspace*{\algorithmicindent} \textbf{Input}:
    $X \in \mathbb{R}^{N\times p}$ design matrix \\
    \hspace*{4.5em} $y \in \mathbb{R}^{n}$ response variable \\
    \hspace*{4.5em} Regularization strengths $\left\{\lambda_j \right\}_{j=1}^{q}$ \\
    \hspace*{4.5em} Number of resamples $N_S$ and $N_E$ \\
    \hspace*{4.5em} Loss function $L(\beta; X, y)$

    \begin{algorithmic}[1]
        \Statex \textit{Model Selection}
        \For{$k = 1$ to $N_S$}
            \State Generate resample $X^k$, $y^k$
            \For{$j=1$ to $q$}
                \State $\hat{\beta}^{jk}\leftarrow$ Lasso regression (penalty $\lambda_j$) of $y^k$ on $X^k$
                \State $S_j^k \leftarrow \left\{i\right\}$ where $\hat{\beta}^{jk}_i \neq 0$
            \EndFor
        \EndFor
        \For{$j = 1$ to $q$}
            \State $\displaystyle S_j \leftarrow \bigcap_{k=1}^{N_S}S_j^k$ \hfill $\triangleright$ \textit{ Intersection}
        \EndFor
        \State \textit{Model Estimation}
        \For{$k=1$ to $N_E$}
            \State Generate training $\left(X_T^k, y_T^k\right)$ and evaluation $\left(X_E^k, y_E^k\right)$ resamples
            \For{$j=1$ to $q$}
                \State $X_{T, j}^k, X_{E, j}^k \leftarrow $ $X_T^k, X_E^k$ with features $S_j$ extracted.
                \State $\hat{\beta}^{jk} \leftarrow$ OLS Regression of $y_T^k$ on $X_{T,j}^k$
                \State $\ell^{jk} \leftarrow L(\hat{\beta}^{jk}; X^k_{E, j}, y_E^k)$
            \EndFor
            \State $\hat{\beta}^k \leftarrow \underset{\hat{\beta}^{jk}}{\text{argmin}} \ \ell^{jk}$
        \EndFor
        \State $\hat{\beta}^* = \underset{k}{\text{median}}\left(\hat{\beta}^k\right)$ \hfill $\triangleright$ \textit{ Union} \\
        \Return $\hat{\beta}^*$
    \end{algorithmic}
\end{algorithm}

# Features

`PyUoI` is split up into two modules, with the following UoI algorithms:


* `linear_model` (generalized linear models)
    * Lasso penalized linear regression UoI~Lasso~.
    * Elastic-net penalized linear regression (UoI~ElasticNet~).
    * Logistic regression (Bernoulli and multinomial) (UoI~Logistic~).
    * Poisson regression (UoI~Poisson~).
* `decomposition` (dimensionality reduction)
    * Column subset selection (UoI~CSS~) [@mahoney2009].
    * Non-negative matrix factorization (UoI~NMF~).

Similar to `scikit-learn`, each UoI algorithm has its own Python class. Instantiations
of these classes are created with specific hyperparameters and are fit to
user-provided datasets. The hyperparameters allow the user to fine-tune the
number of resamples, fraction of data in each resample, and the model selection
criteria used in the estimation module (in Algorithm 1, test set accuracy is
used, but the Akaike and Bayesian Information Criteria are also available).
Additionally, `PyUoI` is agnostic to the specific solver used for a given model.
For example, generalized linear models come equipped with a coordinate
descent solver (from `scikit-learn`), a built-in Orthant-Wise Limited memory
Quasi-Newton solver [@gong2015], and the `pycasso` solver [@ge2019]. The choice
of solver is left to the user as a hyperparameter.

# Applications

We have used `PyUoI` largely in the realm of neuroscience and genomics
[@bouchard2017; @ubaru2017]. A few applications include:

* Interpretable functional connectivity networks from neural populations in the
  visual, auditory, and motor cortices of various animal models;
* Sparse decoding of behavioral activity from spiking neural activity;
* Parts-based decomposition of electrocorticography recordings in rat auditory
  cortex that reflect functional cortical organization;
* Extraction of characteristic single nucleotide polymorphisms for the
  prediction of phenotypes in mice.

However, the algorithms implemented in `PyUoI` are broadly applicable and not
limited to these contexts.

# Acknowledgements

P.S.S. was supported by the Department of Defense (DoD) through the National
Defense Science \& Engineering Graduate Fellowship (NDSEG) Program. K.E.B. and
J.A.L. were supported through the Lawrence Berkeley National Laboratory-internal
LDRD "Deep Learning for Science" led by Prabhat. A.J.T. was supported by the
Department of Energy project “Co-design for artificial intelligence coupled with
computing at scale for extremely large, complex datasets” K.E.B. was funded by
Lawrence Berkeley National Laboratory-internal LDRD "Neuro/Nano-Technology for
BRAIN" led by Peter Denes.

# References
