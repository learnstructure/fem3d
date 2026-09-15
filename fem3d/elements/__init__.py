"""
Elements package for fem3d.
"""

from .element import ElementBase
from .frame import FrameElement
from .truss import TrussElement
from .spring import SpringElement

# Alias BeamElement -> FrameElement for familiarity and compatibility
BeamElement = FrameElement

__all__ = [
    "ElementBase",
    "FrameElement",
    "BeamElement",
    "TrussElement",
    "SpringElement",
]
