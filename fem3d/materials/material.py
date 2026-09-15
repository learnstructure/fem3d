"""
Base Material class for 3D finite element analysis.
"""

from abc import ABC, abstractmethod


class Material(ABC):
    """
    Abstract base class for all material models.
    """

    @abstractmethod
    def get_initial_tangent(self):
        """Return the initial elastic modulus (or equivalent)."""
        pass
