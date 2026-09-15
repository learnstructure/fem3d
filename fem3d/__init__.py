"""
fem3d: Finite Element Analysis for 3D Framed Structures and Trusses in Python.

Provides classes and utilities for performing 3D finite element analysis (FEA)
of general space frames, space trusses, and springs with 6 degrees of freedom per node.

Author: Abinash Mandal
"""

from .structure import Structure
from .nodes import Node
from .materials.elastic import ElasticMaterial
from .sections.section import Section
from .elements.frame import FrameElement, BeamElement
from .elements.truss import TrussElement
from .elements.spring import SpringElement
from .loads import PointLoad, DistributedLoad, ElementPointLoad
from .results import Results
from .buckling_analysis import buckling_analysis
from .utils.simple_frame import SimpleFrame, SimpleFrame3D

try:
    from .adapters.struct_core_adapter import (
        model_from_core,
        model_to_core,
        result_to_core,
    )
except ImportError:
    def _missing_adapter(*args, **kwargs):
        raise ImportError(
            "The 'struct_core' package is required for this feature.\n"
            "Install it with:  pip install struct_core\n"
            "or install optional extra:  pip install fem3d[ecosystem]"
        )
    model_from_core = model_to_core = result_to_core = _missing_adapter

__all__ = [
    "Structure",
    "Node",
    "ElasticMaterial",
    "Section",
    "FrameElement",
    "BeamElement",
    "TrussElement",
    "SpringElement",
    "PointLoad",
    "DistributedLoad",
    "ElementPointLoad",
    "Results",
    "buckling_analysis",
    "SimpleFrame",
    "SimpleFrame3D",
    "model_from_core",
    "model_to_core",
    "result_to_core",
]
