"""
SpringElement module defining 3D elastic spring elements (translational and rotational).
"""

from typing import Optional, Union
import numpy as np

from .element import ElementBase


class SpringElement(ElementBase):
    """
    Elastic 3D spring element connecting two nodes or providing grounded elastic support.

    Attributes
    ----------
    kx, ky, kz : float
        Translational spring stiffnesses along global X, Y, Z.
    krx, kry, krz : float
        Rotational spring stiffnesses about global X, Y, Z.
    """

    def __init__(
        self,
        eid: Union[int, str],
        node_i,
        node_j,
        kx: float = 0.0,
        ky: float = 0.0,
        kz: float = 0.0,
        krx: float = 0.0,
        kry: float = 0.0,
        krz: float = 0.0,
    ):
        """
        Initialize a 3D SpringElement.

        Parameters
        ----------
        eid : int or str
            Unique element identifier.
        node_i : Node
            Start node.
        node_j : Node
            End node.
        kx, ky, kz : float, optional
            Translational stiffnesses along X, Y, Z. Defaults to 0.0.
        krx, kry, krz : float, optional
            Rotational stiffnesses about X, Y, Z. Defaults to 0.0.
        """
        # If node_i and node_j are at the same point, handle gracefully
        dx = node_j.x - node_i.x
        dy = node_j.y - node_i.y
        dz = node_j.z - node_i.z
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-12:
            # Coincident nodes (e.g. joint spring)
            self.id = eid
            self.node_i = node_i
            self.node_j = node_j
            self.length = 0.0
            self.structure = None
            self.R = np.eye(3)
        else:
            super().__init__(eid, node_i, node_j)

        self.k_diag = np.array(
            [kx, ky, kz, krx, kry, krz], dtype=float
        )
        self.eq_load = np.zeros(12, dtype=float)

    def global_stiffness(self) -> np.ndarray:
        """
        Return the 12x12 global stiffness matrix for the spring element.
        """
        K = np.zeros((12, 12), dtype=float)
        for d in range(6):
            val = self.k_diag[d]
            if val > 0.0:
                K[d, d] += val
                K[d, 6 + d] -= val
                K[6 + d, d] -= val
                K[6 + d, 6 + d] += val
        return K

    def local_stiffness(self) -> np.ndarray:
        """Return 12x12 stiffness."""
        return self.global_stiffness()

    def local_mass_matrix(self, lumped: bool = False) -> np.ndarray:
        """Spring has zero mass by default."""
        return np.zeros((12, 12), dtype=float)

    def get_local_forces(self) -> np.ndarray:
        """
        Compute spring forces: F = K_spring * (u_j - u_i).
        """
        if self.structure is None or self.structure.disp is None:
            raise ValueError("Structure has not been solved yet.")
        u_i = self.structure.disp[self.node_i.dofs]
        u_j = self.structure.disp[self.node_j.dofs]
        delta_u = u_j - u_i
        f_spring = self.k_diag * delta_u

        f = np.zeros(12, dtype=float)
        f[0:6] = -f_spring
        f[6:12] = f_spring
        return f

    def __repr__(self) -> str:
        return f"SpringElement(id={self.id}, nodes=({self.node_i.id}, {self.node_j.id}))"
