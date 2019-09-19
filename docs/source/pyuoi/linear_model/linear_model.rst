############
linear_model
############
All linear models operate through the basic structure provided by the base
class. The base class performs the necessary bootstrapping, fitting procedures,
intersection step, and model averaging. The derived classes simply provide
objects to the base class that perform the actual fits (e.g., UoI\ :sub:`Lasso`
provides ``Lasso`` and ``LinearRegression`` objects to the base class).

Base Classes
------------

The base class for all linear models is ``AbstractUoILinearModel``.
Intermediate derived classes, ``AbstractUoILinearRegressor`` (for lasso and
elastic net), and ``AbstractUoIGeneralizedLinearRegressor`` (for logistic and
Poisson regression) are also provided.

.. automodule:: pyuoi.linear_model.base
    :members: AbstractUoILinearModel, AbstractUoILinearRegressor,
              AbstractUoIGeneralizedLinearRegressor

Lasso
-----
The ``UoI_Lasso`` object provides the base class with a ``Lasso`` object for
the selection module and a ``LinearRegression`` object for the estimation
module. Additionally, the ``pycasso`` solver is provided as the ``PycLasso``
class.

.. automodule:: pyuoi.linear_model.lasso
   :members: UoI_Lasso, PycLasso

Elastic Net
-----------
The ``UoI_ElasticNet`` object provides the base class with an ``ElasticNet``
object for the selection module and a ``LinearRegression`` object for the
estimation module.

.. automodule:: pyuoi.linear_model.elasticnet
    :members: UoI_ElasticNet

Logistic Regression
-------------------
The ``UoI_L1Logistic`` module uses a custom logistic regression solver for both
the selection and estimation modules. This solver uses a modified orthant-wise
limited memory quasi-Newton algorithm. For estimation, no regularization is
performed.

.. automodule:: pyuoi.linear_model.logistic
    :members: UoI_L1Logistic

Poisson Regression
------------------
The ``poisson`` module provides a Poisson regression solver that uses either
coordinate descent or a modified orthant-wise limited memory quasi-Newton
solver. ``UoI_Poisson`` uses ``Poisson`` objects for both selection and
estimation; however, the estimation module uses no regularization penalties.

.. automodule:: pyuoi.linear_model.poisson
    :members: UoI_Poisson, Poisson
