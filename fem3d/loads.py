"""
Loads module defining 3D point, distributed, and element loads.
"""

from typing import Union
import numpy as np


class PointLoad:
    """
    Represents a concentrated 3D force and moment load acting on a node.

    Attributes
    ----------
    node : Node
        Target node.
    fx, fy, fz : float
        Forces along global X, Y, Z.
    mx, my, mz : float
        Moments about global X, Y, Z.
    """

    def __init__(
        self,
        node,
        fx: float = 0.0,
        fy: float = 0.0,
        fz: float = 0.0,
        mx: float = 0.0,
        my: float = 0.0,
        mz: float = 0.0,
    ):
        """
        Initialize a 3D PointLoad.

        Parameters
        ----------
        node : Node
            The Node object to which the load is applied.
        fx, fy, fz : float, optional
            Concentrated forces in global X, Y, Z. Defaults to 0.0.
        mx, my, mz : float, optional
            Concentrated moments about global X, Y, Z. Defaults to 0.0.
        """
        self.node = node
        self.fx = float(fx)
        self.fy = float(fy)
        self.fz = float(fz)
        self.mx = float(mx)
        self.my = float(my)
        self.mz = float(mz)

    def as_array(self) -> np.ndarray:
        """Return 6-element load array [fx, fy, fz, mx, my, mz]."""
        return np.array(
            [self.fx, self.fy, self.fz, self.mx, self.my, self.mz], dtype=float
        )

    def __repr__(self) -> str:
        return f"PointLoad(node={self.node.id}, F=({self.fx}, {self.fy}, {self.fz}), M=({self.mx}, {self.my}, {self.mz}))"


class ElementLoad:
    """
    Base class for loads acting along the length of a 3D element.
    """

    def __init__(self, element):
        self.element = element

    def _compute_equivalent_loads(self) -> np.ndarray:
        """Compute 12-element equivalent local nodal load vector."""
        raise NotImplementedError

    def _transform_and_store_equivalent_loads(self, eq_local: np.ndarray):
        """
        Transform local equivalent nodal loads to global coordinates and
        accumulate in element.eq_load.
        """
        T = self.element.transformation_matrix()
        eq_global = T.T @ eq_local

        if not hasattr(self.element, "eq_load") or self.element.eq_load is None:
            self.element.eq_load = np.zeros(12, dtype=float)
        if not hasattr(self.element, "eq_load_local") or self.element.eq_load_local is None:
            self.element.eq_load_local = np.zeros(12, dtype=float)

        self.element.eq_load_local += eq_local
        self.element.eq_load += eq_global


class DistributedLoad(ElementLoad):
    """
    Uniformly distributed load acting on an element.

    References:
    - Przemieniecki, J. S. (1968). Theory of Matrix Structural Analysis.
      McGraw-Hill, Section 6.2, pp. 106-107.
    - Weaver, W., & Gere, J. M. (1990). Matrix Analysis of Framed Structures.
      Appendix B.
    """

    def __init__(
        self,
        element,
        wx: float = 0.0,
        wy: float = 0.0,
        wz: float = 0.0,
        coord_system: str = "local",
    ):
        """
        Initialize a DistributedLoad.

        Parameters
        ----------
        element : ElementBase
            The element on which the load acts.
        wx : float, optional
            Distributed load intensity along x-axis (axial). Defaults to 0.0.
        wy : float, optional
            Distributed load intensity along y-axis. Defaults to 0.0.
        wz : float, optional
            Distributed load intensity along z-axis. Defaults to 0.0.
        coord_system : str, optional
            'local' or 'global'. Defaults to 'local'.
        """
        super().__init__(element)
        self.coord_system = coord_system.lower()

        if self.coord_system == "global":
            # Transform global distributed load vector [wx, wy, wz] to local coordinates
            R = element.rotation_matrix_3x3()
            w_loc = R @ np.array([wx, wy, wz], dtype=float)
            self.wx, self.wy, self.wz = float(w_loc[0]), float(w_loc[1]), float(w_loc[2])
        else:
            self.wx = float(wx)
            self.wy = float(wy)
            self.wz = float(wz)

        eq_local = self._compute_equivalent_loads()
        self._transform_and_store_equivalent_loads(eq_local)

    def _compute_equivalent_loads(self) -> np.ndarray:
        """
        Compute work-equivalent nodal forces and moments in local coordinates.

        For uniform load:
        - Axial wx:
            f_x1 = wx * L / 2,  f_x2 = wx * L / 2
        - Transverse wy (bending about z'):
            f_y1 = wy * L / 2,  f_y2 = wy * L / 2
            m_z1 = wy * L^2 / 12,  m_z2 = -wy * L^2 / 12
        - Transverse wz (bending about y'):
            f_z1 = wz * L / 2,  f_z2 = wz * L / 2
            m_y1 = -wz * L^2 / 12,  m_y2 = wz * L^2 / 12
        """
        L = self.element.length
        eq = np.zeros(12, dtype=float)

        # Axial
        eq[0] = self.wx * L / 2.0
        eq[6] = self.wx * L / 2.0

        # Flexure in x-y (Iz)
        eq[1] = self.wy * L / 2.0
        eq[5] = self.wy * (L**2) / 12.0
        eq[7] = self.wy * L / 2.0
        eq[11] = -self.wy * (L**2) / 12.0

        # Flexure in x-z (Iy)
        # Note sign convention: positive theta_y is slope -du_z/dx
        eq[2] = self.wz * L / 2.0
        eq[4] = -self.wz * (L**2) / 12.0
        eq[8] = self.wz * L / 2.0
        eq[10] = self.wz * (L**2) / 12.0

        return eq


class ElementPointLoad(ElementLoad):
    """
    Concentrated point load acting on an element at distance x from start node.
    """

    def __init__(
        self,
        element,
        px: float = 0.0,
        py: float = 0.0,
        pz: float = 0.0,
        mx: float = 0.0,
        my: float = 0.0,
        mz: float = 0.0,
        x: float = 0.0,
        coord_system: str = "local",
    ):
        """
        Initialize an ElementPointLoad.

        Parameters
        ----------
        element : ElementBase
            The element.
        px, py, pz : float, optional
            Point force components. Defaults to 0.0.
        mx, my, mz : float, optional
            Point moment components. Defaults to 0.0.
        x : float, optional
            Distance from start node along element length.
        coord_system : str, optional
            'local' or 'global'. Defaults to 'local'.
        """
        super().__init__(element)
        self.coord_system = coord_system.lower()
        self.x = float(x)

        if self.coord_system == "global":
            R = element.rotation_matrix_3x3()
            f_loc = R @ np.array([px, py, pz], dtype=float)
            m_loc = R @ np.array([mx, my, mz], dtype=float)
            self.px, self.py, self.pz = float(f_loc[0]), float(f_loc[1]), float(f_loc[2])
            self.mx, self.my, self.mz = float(m_loc[0]), float(m_loc[1]), float(m_loc[2])
        else:
            self.px, self.py, self.pz = float(px), float(py), float(pz)
            self.mx, self.my, self.mz = float(mx), float(my), float(mz)

        eq_local = self._compute_equivalent_loads()
        self._transform_and_store_equivalent_loads(eq_local)

    def _compute_equivalent_loads(self) -> np.ndarray:
        L = self.element.length
        a = self.x
        b = L - a
        if a < 0.0 or a > L:
            raise ValueError(f"Load position x={a} is outside element span [0, {L}].")

        eq = np.zeros(12, dtype=float)

        # Axial px
        eq[0] = self.px * (b / L)
        eq[6] = self.px * (a / L)

        # Transverse py
        eq[1] = self.py * (b**2 * (3.0 * a + b)) / (L**3)
        eq[5] = self.py * (a * b**2) / (L**2)
        eq[7] = self.py * (a**2 * (a + 3.0 * b)) / (L**3)
        eq[11] = -self.py * (a**2 * b) / (L**2)

        # Transverse pz
        eq[2] = self.pz * (b**2 * (3.0 * a + b)) / (L**3)
        eq[4] = -self.pz * (a * b**2) / (L**2)
        eq[8] = self.pz * (a**2 * (a + 3.0 * b)) / (L**3)
        eq[10] = self.pz * (a**2 * b) / (L**2)

        # Torsion mx
        eq[3] = self.mx * (b / L)
        eq[9] = self.mx * (a / L)

        return eq
