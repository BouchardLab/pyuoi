"""Union of Intersection models with linear selection and estimation.

Provides both abstract base classes for creating user-defined UoI models
and several concrete implementations.
"""
from .base import (AbstractUoILinearModel, AbstractUoILinearRegressor,
                   AbstractUoILinearClassifier)
from .lasso import UoI_Lasso
from .logistic import UoI_L1Logistic
from .poisson import Poisson
from . import utils


__all__ = ["AbstractUoILinearClassifier",
           "AbstractUoILinearModel",
           "AbstractUoILinearRegressor",
           "UoI_L1Logistic",
           "UoI_Lasso",
           "utils"]
from .poisson import Poisson
