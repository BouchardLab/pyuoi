#############
decomposition
#############

Abstract Base Class
-------------------

Decomposition classes are built through an ``AbstractDecompositionModel``, which
extends ``scikit-learn``'s ``BaseEstimator`` class to include methods that are
relevant for decomposition methods.

.. automodule:: pyuoi.decomposition.base
    :members: AbstractDecompositionModel

CUR Decomposition
-----------------

The ``pyuoi`` package includes a class to perform ordinary CUR decomposition in
addition to a class that performs UoI\ :sub:`CUR`.

.. automodule:: pyuoi.decomposition.CUR
    :members: CUR, UoI_CUR

Non-negative Matrix Factorization
---------------------------------

UoI\ :sub:`NMF` can be customized with various NMF, clustering,
non-negative least squares, and consensus algorithms. A base class accepts
general objects or functions to perform the desired NMF, clustering,
regression, and consensus grouping (provided that they have the correct
structure). A derived class which uses

* ``scikit-learn``'s NMF object

* DBSCAN for clustering

* ``scipy``'s non-negative least squares function

* the median function for consensus grouping

is also provided. This derived class accepts keyword arguments that correspond
to the keyword arguments of the above algorithms, so that the user does not
have to provide instantiated objects.

.. automodule:: pyuoi.decomposition.NMF
    :members: UoI_NMF_Base, UoI_NMF
