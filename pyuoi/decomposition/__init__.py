"""Union of Intersection models with matrix decomposition."""
from .CUR import UoI_CUR
from .CUR import CUR
from .NMF import UoI_NMF

__all__ = ["UoI_CUR",
           "CUR",
           "UoI_NMF"]
