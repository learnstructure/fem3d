"""
Loads module defining 3D point, distributed, and element loads.
"""

from typing import Union, Optional
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
    - In local coordinates: w1 (axial), w2 (transverse, web), w3 (transverse, flange)
    - In global coordinates: wx, wy, wz along global X, Y, Z

    References:
    - Przemieniecki, J. S. (1968). Theory of Matrix Structural Analysis.
      McGraw-Hill, Section 6.2, pp. 106-107.
    - Weaver, W., & Gere, J. M. (1990). Matrix Analysis of Framed Structures.
      Appendix B.
    """

    def __init__(
        self,
        element,
        w1: float = 0.0,
        w2: float = 0.0,
        w3: float = 0.0,
        coord_system: str = "local",
        wx: Optional[float] = None,
        wy: Optional[float] = None,
        wz: Optional[float] = None,
    ):
        """
        Initialize a DistributedLoad.

        Parameters
        ----------
        element : ElementBase
            The element on which the load acts.
        w1 : float, optional
            Distributed load intensity along local 1-axis (axial). Defaults to 0.0.
        w2 : float, optional
            Distributed load intensity along local 2-axis (transverse). Defaults to 0.0.
        w3 : float, optional
            Distributed load intensity along local 3-axis (transverse). Defaults to 0.0.
        coord_system : str, optional
            'local' or 'global'. Defaults to 'local'.
        """
        super().__init__(element)
        self.coord_system = coord_system.lower()

        if self.coord_system == "global":
            # Global loads passed via wx, wy, wz or w1, w2, w3
            gx = wx if wx is not None else w1
            gy = wy if wy is not None else w2
            gz = wz if wz is not None else w3
            R = element.rotation_matrix_3x3()
            w_loc = R @ np.array([gx, gy, gz], dtype=float)
            self.w1, self.w2, self.w3 = float(w_loc[0]), float(w_loc[1]), float(w_loc[2])
        else:
            self.w1 = float(wx if wx is not None else w1)
            self.w2 = float(wy if wy is not None else w2)
            self.w3 = float(wz if wz is not None else w3)

        eq_local = self._compute_equivalent_loads()
        self._transform_and_store_equivalent_loads(eq_local)

    def _compute_equivalent_loads(self) -> np.ndarray:
        """
        Compute work-equivalent nodal forces and moments in local coordinates.

        For uniform load:
        - Axial w1:
            f_1i = w1 * L / 2,  f_1j = w1 * L / 2
        - Transverse w2 (bending in 1-2 about axis 3):
            f_2i = w2 * L / 2,  f_2j = w2 * L / 2
            m_3i = w2 * L^2 / 12,  m_3j = -w2 * L^2 / 12
        - Transverse w3 (bending in 1-3 about axis 2):
            f_3i = w3 * L / 2,  f_3j = w3 * L / 2
            m_2i = -w3 * L^2 / 12,  m_2j = w3 * L^2 / 12
        """
        L = self.element.length
        eq = np.zeros(12, dtype=float)

        # Axial (local axis 1)
        eq[0] = self.w1 * L / 2.0
        eq[6] = self.w1 * L / 2.0

        # Flexure in 1-2 about axis 3 (resisted by I3)
        eq[1] = self.w2 * L / 2.0
        eq[5] = self.w2 * (L**2) / 12.0
        eq[7] = self.w2 * L / 2.0
        eq[11] = -self.w2 * (L**2) / 12.0

        # Flexure in 1-3 about axis 2 (resisted by I2)
        # Note sign convention: positive theta_2 is slope -du_3/dx1
        eq[2] = self.w3 * L / 2.0
        eq[4] = -self.w3 * (L**2) / 12.0
        eq[8] = self.w3 * L / 2.0
        eq[10] = self.w3 * (L**2) / 12.0

        return eq


class ElementPointLoad(ElementLoad):
    """
    Concentrated point load acting on an element at distance x from start node.
    - In local coordinates: p1, p2, p3 (forces), m1, m2, m3 (moments)
    - In global coordinates: px, py, pz (forces along X, Y, Z), mx, my, mz (moments about X, Y, Z)
    """

    def __init__(
        self,
        element,
        p1: float = 0.0,
        p2: float = 0.0,
        p3: float = 0.0,
        m1: float = 0.0,
        m2: float = 0.0,
        m3: float = 0.0,
        x: float = 0.0,
        coord_system: str = "local",
        px: Optional[float] = None,
        py: Optional[float] = None,
        pz: Optional[float] = None,
        mx: Optional[float] = None,
        my: Optional[float] = None,
        mz: Optional[float] = None,
    ):
        """
        Initialize an ElementPointLoad.

        Parameters
        ----------
        element : ElementBase
            The element.
        p1, p2, p3 : float, optional
            Point force components along local 1, 2, 3 axes. Defaults to 0.0.
        m1, m2, m3 : float, optional
            Point moment components about local 1, 2, 3 axes. Defaults to 0.0.
        x : float, optional
            Distance from start node along element length.
        coord_system : str, optional
            'local' or 'global'. Defaults to 'local'.
        """
        super().__init__(element)
        self.coord_system = coord_system.lower()
        self.x = float(x)

        if self.coord_system == "global":
            gx = px if px is not None else p1
            gy = py if py is not None else p2
            gz = pz if pz is not None else p3
            gmx = mx if mx is not None else m1
            gmy = my if my is not None else m2
            gmz = mz if mz is not None else m3
            R = element.rotation_matrix_3x3()
            f_loc = R @ np.array([gx, gy, gz], dtype=float)
            m_loc = R @ np.array([gmx, gmy, gmz], dtype=float)
            self.p1, self.p2, self.p3 = float(f_loc[0]), float(f_loc[1]), float(f_loc[2])
            self.m1, self.m2, self.m3 = float(m_loc[0]), float(m_loc[1]), float(m_loc[2])
        else:
            self.p1 = float(px if px is not None else p1)
            self.p2 = float(py if py is not None else p2)
            self.p3 = float(pz if pz is not None else p3)
            self.m1 = float(mx if mx is not None else m1)
            self.m2 = float(my if my is not None else m2)
            self.m3 = float(mz if mz is not None else m3)

        eq_local = self._compute_equivalent_loads()
        self._transform_and_store_equivalent_loads(eq_local)

    def _compute_equivalent_loads(self) -> np.ndarray:
        L = self.element.length
        a = self.x
        b = L - a
        if a < 0.0 or a > L:
            raise ValueError(f"Load position x={a} is outside element span [0, {L}].")

        eq = np.zeros(12, dtype=float)

        # Axial p1
        eq[0] = self.p1 * (b / L)
        eq[6] = self.p1 * (a / L)

        # Transverse p2 (bending in 1-2 about axis 3)
        eq[1] = self.p2 * (b**2 * (3.0 * a + b)) / (L**3)
        eq[5] = self.p2 * (a * b**2) / (L**2)
        eq[7] = self.p2 * (a**2 * (a + 3.0 * b)) / (L**3)
        eq[11] = -self.p2 * (a**2 * b) / (L**2)

        # Transverse p3 (bending in 1-3 about axis 2)
        eq[2] = self.p3 * (b**2 * (3.0 * a + b)) / (L**3)
        eq[4] = -self.p3 * (a * b**2) / (L**2)
        eq[8] = self.p3 * (a**2 * (a + 3.0 * b)) / (L**3)
        eq[10] = self.p3 * (a**2 * b) / (L**2)

        # Torsion m1
        eq[3] = self.m1 * (b / L)
        eq[9] = self.m1 * (a / L)

        return eq
