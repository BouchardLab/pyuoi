.. PyUoI documentation master file, created by
   sphinx-quickstart on Thu Dec 13 11:27:32 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyUoI's documentation!
=================================

PyUoI is a set of concrete implementations of the algorthims described
in [Bouchard2017]_ and [Ubaru2017]_. In general, UoI versions of algorithms incorporate
two ideas into model fitting

#. bootstrapping the data and fitting many models to find robust estimates and
#. separating the selection and estimation phases to reduce bias.

The interface is meant to be similar to scikit-learn's versions of the models
and the implementations are often based on scikit-learn.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   pyuoi/pyuoi
   pyuoi/linear_model/linear_model
   pyuoi/decomposition/decomposition
   pyuoi/utils
   pyuoi/mpi_utils

.. rubric:: References

.. [Bouchard2017] Bouchard, K., Bujan, A., Roosta-Khorasani, F., Ubaru, S., Prabhat, M., Snijders, A., ... & Bhattacharya, S. (2017). Union of intersections (uoi) for interpretable data driven discovery and prediction. In Advances in Neural Information Processing Systems (pp. 1078-1086).

.. [Ubaru2017] Ubaru, S., Wu, K., & Bouchard, K. E. (2017, December). UoI-NMF cluster: a robust nonnegative matrix factorization algorithm for improved parts-based decomposition and reconstruction of noisy data. In 2017 16th IEEE International Conference on Machine Learning and Applications (ICMLA) (pp. 241-248). IEEE.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
