"""Union of Intersection models with linear selection and estimation.

Provides both abstract base classes for creating user-defined UoI models
and several concrete implementations.
"""
from .base import (AbstractUoILinearModel, AbstractUoILinearRegressor,
                   AbstractUoIGeneralizedLinearRegressor)
from .lasso import UoI_Lasso
from .elasticnet import UoI_ElasticNet
from .logistic import MaskedCoefLogisticRegression, UoI_L1Logistic
from .poisson import Poisson, UoI_Poisson

__all__ = ["AbstractUoILinearModel",
           "AbstractUoILinearRegressor",
           "AbstractUoIGeneralizedLinearRegressor",
           "MaskedCoefLogisticRegression",
           "UoI_L1Logistic",
           "UoI_Lasso",
           "UoI_ElasticNet",
           "Poisson",
           "UoI_Poisson"]
