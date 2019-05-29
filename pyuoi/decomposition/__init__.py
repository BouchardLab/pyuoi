"""Union of Intersection models with matrix decomposition."""
from .CUR import UoI_CUR
from .CUR import CUR
from .NMF import UoI_NMF
from .NMF import UoI_NMF_Base

__all__ = ["UoI_CUR",
           "CUR",
           "UoI_NMF",
           "UoI_NMF_Base"]
