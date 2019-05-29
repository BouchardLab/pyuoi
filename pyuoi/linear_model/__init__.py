"""Union of Intersection models with linear selection and estimation.

Provides both abstract base classes for creating user-defined UoI models
and several concrete implementations.
"""
from .base import (AbstractUoILinearModel, AbstractUoILinearRegressor,
                   AbstractUoIGeneralizedLinearRegressor)
from .lasso import UoI_Lasso
from .elasticnet import UoI_ElasticNet
from .logistic import UoI_L1Logistic
from .poisson import Poisson
from .poisson import UoI_Poisson
from . import utils

__all__ = ["AbstractUoILinearModel",
           "AbstractUoILinearRegressor",
           "AbstractUoIGeneralizedLinearRegressor",
           "UoI_L1Logistic",
           "UoI_Lasso",
           "UoI_ElasticNet",
           "Poisson",
           "UoI_Poisson",
           "utils"]
