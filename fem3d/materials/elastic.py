"""
Elastic material model for 3D finite element analysis.
"""

from typing import Optional
from .material import Material


class ElasticMaterial(Material):
    """
    Isotropic linear elastic material definition.

    Attributes
    ----------
    E : float
        Young's Modulus of elasticity.
    nu : float
        Poisson's ratio. Defaults to 0.3.
    G : float
        Shear modulus of elasticity. Computed as E / (2 * (1 + nu)) if not provided.
    rho : float
        Mass density (mass per unit volume). Defaults to 0.0.
    """

    def __init__(
        self,
        E: float,
        nu: float = 0.3,
        G: Optional[float] = None,
        rho: float = 0.0,
    ):
        """
        Initialize an ElasticMaterial object.

        Parameters
        ----------
        E : float
            Young's Modulus of elasticity.
        nu : float, optional
            Poisson's ratio. Defaults to 0.3.
        G : float, optional
            Shear modulus. If None, computed as G = E / (2 * (1 + nu)).
        rho : float, optional
            Mass density. Defaults to 0.0.
        """
        self.E = float(E)
        self.nu = float(nu)
        if G is not None:
            self.G = float(G)
        else:
            self.G = self.E / (2.0 * (1.0 + self.nu))
        self.rho = float(rho)

    def get_initial_tangent(self) -> float:
        """Return Young's Modulus E."""
        return self.E

    def __repr__(self) -> str:
        return f"ElasticMaterial(E={self.E}, nu={self.nu}, G={self.G}, rho={self.rho})"
