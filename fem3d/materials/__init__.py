"""
Materials package for fem3d.
"""

from .material import Material
from .elastic import ElasticMaterial

__all__ = ["Material", "ElasticMaterial"]
