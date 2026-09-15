"""
TrussElement module defining 3D space truss (pin-jointed bar) finite elements.
"""

from typing import Union
import numpy as np

from .element import ElementBase
from ..materials.elastic import ElasticMaterial
from ..sections.section import Section


class TrussElement(ElementBase):
    """
    Elastic 3D space truss element with only axial stiffness.

    Connects two nodes with 6 DOFs per node, contributing stiffness
    strictly to the 3 translational DOFs at each node:
        [ux, uy, uz] at node_i and [ux, uy, uz] at node_j.
    Rotational DOFs [rx, ry, rz] have zero stiffness contribution.

    References:
    - Przemieniecki, J. S. (1968). Theory of Matrix Structural Analysis.
      McGraw-Hill, Section 5.3, pp. 71-73.
    - Weaver, W., & Gere, J. M. (1990). Matrix Analysis of Framed Structures.
      Section 4.4, pp. 210-215.
    """

    def __init__(
        self,
        eid: Union[int, str],
        node_i,
        node_j,
        material: Union[ElasticMaterial, float],
        section: Union[Section, float],
        extra_mass: float = 0.0,
    ):
        """
        Initialize a 3D TrussElement.

        Parameters
        ----------
        eid : int or str
            Unique element identifier.
        node_i : Node
            Start node.
        node_j : Node
            End node.
        material : ElasticMaterial or float
            Material definition or numeric Young's modulus E.
        section : Section or float
            Section definition or numeric cross-sectional area A.
        extra_mass : float, optional
            Additional distributed non-structural mass per unit length. Defaults to 0.0.
        """
        super().__init__(eid, node_i, node_j)

        if isinstance(material, (int, float)):
            self.material = ElasticMaterial(E=float(material))
        else:
            self.material = material

        if isinstance(section, (int, float)):
            self.section = Section(A=float(section))
        else:
            self.section = section

        self.area = self.section.A
        self.extra_mass = float(extra_mass)
        self.eq_load = np.zeros(12, dtype=float)

    def local_stiffness(self) -> np.ndarray:
        """
        Return the 12x12 local stiffness matrix.
        Only the axial terms at dofs 0 and 6 are non-zero.
        """
        E = self.material.E
        A = self.area
        L = self.length
        k = np.zeros((12, 12), dtype=float)
        EA_L = E * A / L
        k[0, 0] = EA_L
        k[0, 6] = -EA_L
        k[6, 0] = -EA_L
        k[6, 6] = EA_L
        return k

    def global_stiffness(self) -> np.ndarray:
        """
        Return the 12x12 global stiffness matrix for the truss element.
        Directly computed via outer product of member direction cosines:
            K_trans = (EA/L) * [ vx*vx^T  -vx*vx^T ]
                               [ -vx*vx^T  vx*vx^T ]
        """
        E = self.material.E
        A = self.area
        L = self.length
        EA_L = E * A / L

        vx = self.vx.reshape(3, 1)
        k33 = EA_L * (vx @ vx.T)

        K_g = np.zeros((12, 12), dtype=float)
        # Node i translations (dofs 0, 1, 2)
        K_g[0:3, 0:3] = k33
        K_g[0:3, 6:9] = -k33
        # Node j translations (dofs 6, 7, 8)
        K_g[6:9, 0:3] = -k33
        K_g[6:9, 6:9] = k33

        return K_g

    def local_mass_matrix(self, lumped: bool = False) -> np.ndarray:
        """Return 12x12 mass matrix in local coordinates."""
        L = self.length
        total_density = self.material.rho * self.area + self.extra_mass
        m_total = total_density * L
        m = np.zeros((12, 12), dtype=float)

        m_half = m_total / 2.0
        # Lumped translational mass at both nodes
        m[0, 0] = m_half
        m[1, 1] = m_half
        m[2, 2] = m_half
        m[6, 6] = m_half
        m[7, 7] = m_half
        m[8, 8] = m_half
        return m

    def geometric_stiffness(self, P: float) -> np.ndarray:
        """
        Geometric stiffness matrix for 3D truss under axial compressive force P.

        Reference:
        - Przemieniecki (1968), Section 12.2, Eq. (12.18).
        """
        L = self.length
        if abs(P) < 1e-12:
            return np.zeros((12, 12), dtype=float)

        vx = self.vx.reshape(3, 1)
        I3 = np.eye(3)
        kg33 = (P / L) * (I3 - vx @ vx.T)

        Kg = np.zeros((12, 12), dtype=float)
        Kg[0:3, 0:3] = kg33
        Kg[0:3, 6:9] = -kg33
        Kg[6:9, 0:3] = -kg33
        Kg[6:9, 6:9] = kg33
        return Kg

    def get_local_forces(self) -> np.ndarray:
        """
        Compute truss axial force in local coordinates.

        Returns
        -------
        numpy.ndarray
            12-element vector with non-zero values only at fx_i and fx_j.
        """
        if self.structure is None or self.structure.disp is None:
            raise ValueError("Structure has not been solved yet.")
        u_i = self.structure.disp[self.node_i.dofs[0:3]]
        u_j = self.structure.disp[self.node_j.dofs[0:3]]

        # Elongation along member vector
        delta_L = float(np.dot(self.vx, u_j - u_i))
        axial_tension = (self.material.E * self.area / self.length) * delta_L

        # Local forces: compression positive in fx_i, or standard tension convention
        # We follow standard convention: [fx_i, 0, 0, 0, 0, 0, fx_j, 0, ...]
        f_local = np.zeros(12, dtype=float)
        f_local[0] = -axial_tension  # tension pulls on node i (-x')
        f_local[6] = axial_tension   # tension pulls on node j (+x')
        return f_local

    def axial_force(self) -> float:
        """Return axial compressive force (positive in compression)."""
        f = self.get_local_forces()
        return -float(f[6])  # compression > 0

    def __repr__(self) -> str:
        return f"TrussElement(id={self.id}, nodes=({self.node_i.id}, {self.node_j.id}), L={self.length:.3f})"
