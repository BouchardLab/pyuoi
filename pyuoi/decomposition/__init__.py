"""Union of Intersection models with matrix decomposition."""
from .CUR import CUR
from .NMF import UoINMF

__all__ = ["CUR",
           "NMF"]
