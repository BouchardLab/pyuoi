.. PyUoI

=====================================================
PyUoI: The Union of Intersections Framework in Python
=====================================================

PyUoI is a set of concrete implementations of the Union of Intersections (UoI)
framework, designed to produce sparse and predictive models for a variety of
machine learning algorithms. In general, the UoI framework leverages two
approaches to model fitting:

#. bootstrapping the data and fitting many models to find robust estimates and
#. separating the selection and estimation phases to reduce bias.

PyUoI contains implementations of the UoI framework for a variety of penalized
generalized linear models as well as dimensionality reductions techniques such
as CUR decomposition and non-negative matrix factorization.

PyUoI is designed to function similarly to scikit-learn, as it often builds
upon scikit-learn's implementations of the aforementioned algorithms.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   examples/index
   contributing
   mpi
   api


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
