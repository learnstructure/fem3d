"""
Adapters package for fem3d.
"""

from .struct_core_adapter import model_from_core, model_to_core, result_to_core

__all__ = ["model_from_core", "model_to_core", "result_to_core"]
