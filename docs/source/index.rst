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

Further details on the UoI framework can be found in [Bouchard2017]_ and
[Ubaru2017]_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   examples/index
   contributing
   mpi
   api

.. rubric:: References

.. [Bouchard2017] Bouchard, K., Bujan, A., Roosta-Khorasani, F., Ubaru, S.,
    Prabhat, M., Snijders, A., ... & Bhattacharya, S. (2017). Union of
    intersections (UoI) for interpretable data driven discovery and
    prediction. In Advances in Neural Information Processing
    Systems (pp. 1078-1086).
.. [Ubaru2017] Ubaru, S., Wu, K., & Bouchard, K. E. (2017, December). UoI-NMF
    cluster: a robust nonnegative matrix factorization algorithm for improved
    parts-based decomposition and reconstruction of noisy data. In 2017 16th
    IEEE International Conference on Machine Learning and Applications (ICMLA)
    (pp. 241-248). IEEE.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
