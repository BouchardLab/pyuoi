.. PyUoI

=====================================================
PyUoI: The Union of Intersections Framework in Python
=====================================================

PyUoI contains implementations of Union of Intersections framework for a variety
of penalized generalized linear models as well as dimensionality reductions
techniques such as column subset selection and non-negative matrix
factorization. In general, UoI is a statistical machine learning framework that
leverages two concepts in model inference:

#. Separating the selection and estimation problems to simultaneously achieve
sparse models with low-bias and low-variance parameter estimates.
#. Stability to perturbations in both selection and estimation.


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
