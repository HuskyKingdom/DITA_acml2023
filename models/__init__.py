from .basemodel import BaseModel
from .gcn import GCN
from .savn import SAVN
from .mjolnir_o import MJOLNIR_O
from .JudgeModel import JudgeModel
from .dita import DITA

__all__ = ["BaseModel", "GCN", "SAVN", "MJOLNIR_O","DITA"]

variables = locals()
