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

    Coordinates Convention:
    -----------------------
    - Local axes (1, 2, 3):
        w1: axial load along local axis 1.
        w2: transverse load along local axis 2 (bending in 1-2 plane, resisted by I3).
        w3: transverse load along local axis 3 (bending in 1-3 plane, resisted by I2).
    - Global axes (X, Y, Z):
        wx: global load intensity along global X.
        wy: global load intensity along global Y.
        wz: global load intensity along global Z.

    The load can be specified either in local coordinates (w1, w2, w3) or in
    global coordinates (wx, wy, wz). The 3x3 direction cosine rotation matrix R
    transforms between local and global components:
        w_local = R @ w_global
        w_global = R.T @ w_local

    Regardless of which coordinate system was used to define the load, both sets
    of attributes (w1, w2, w3 and wx, wy, wz) are always synchronized and accessible.

    References:
    - Przemieniecki, J. S. (1968). Theory of Matrix Structural Analysis.
      McGraw-Hill, Section 6.2, pp. 106-107.
    - Weaver, W., & Gere, J. M. (1990). Matrix Analysis of Framed Structures.
      Appendix B.
    """

    def __init__(
        self,
        element,
        w1: Optional[float] = None,
        w2: Optional[float] = None,
        w3: Optional[float] = None,
        coord_system: Optional[str] = None,
        wx: Optional[float] = None,
        wy: Optional[float] = None,
        wz: Optional[float] = None,
        is_self_weight: bool = False,
    ):
        """
        Initialize a DistributedLoad.

        Parameters
        ----------
        element : ElementBase
            The element on which the load acts.
        w1 : float, optional
            Distributed load intensity along local 1-axis (axial).
        w2 : float, optional
            Distributed load intensity along local 2-axis (transverse).
        w3 : float, optional
            Distributed load intensity along local 3-axis (transverse).
        coord_system : str, optional
            'local' or 'global'. If omitted, automatically inferred from the supplied parameters.
        wx : float, optional
            Distributed load intensity along global X-axis.
        wy : float, optional
            Distributed load intensity along global Y-axis.
        wz : float, optional
            Distributed load intensity along global Z-axis.
        is_self_weight : bool, optional
            Whether this load represents element self-weight body force. Defaults to False.
        """
        super().__init__(element)
        self.is_self_weight = bool(is_self_weight)

        has_local = any(v is not None for v in (w1, w2, w3))
        has_global = any(v is not None for v in (wx, wy, wz))

        # Check for conflicting inputs
        if has_local and has_global:
            loc_vals = [float(v or 0.0) for v in (w1, w2, w3)]
            glob_vals = [float(v or 0.0) for v in (wx, wy, wz)]
            if any(abs(v) > 1e-12 for v in loc_vals) and any(abs(v) > 1e-12 for v in glob_vals):
                raise ValueError(
                    "Cannot specify both local components (w1, w2, w3) and global components (wx, wy, wz) simultaneously. "
                    "Use w1, w2, w3 for local element axes, or wx, wy, wz for global axes."
                )

        sys_spec = coord_system.lower() if coord_system is not None else None

        if sys_spec == "global":
            # Explicitly global requested
            if has_local and not has_global:
                # Positional backward compatibility: DistributedLoad(el, 0, 0, -10, coord_system="global")
                gx = float(w1 or 0.0)
                gy = float(w2 or 0.0)
                gz = float(w3 or 0.0)
            else:
                gx = float(wx or 0.0)
                gy = float(wy or 0.0)
                gz = float(wz or 0.0)
            self._set_from_global(gx, gy, gz)
        elif sys_spec == "local":
            # Explicitly local requested
            if has_global and not has_local:
                raise ValueError(
                    f"coord_system='local' was specified, but global components (wx={wx}, wy={wy}, wz={wz}) were provided. "
                    "Use w1, w2, w3 for local element coordinates."
                )
            l1 = float(w1 or 0.0)
            l2 = float(w2 or 0.0)
            l3 = float(w3 or 0.0)
            self._set_from_local(l1, l2, l3)
        else:
            # Inferred coordinate system
            if has_global:
                gx = float(wx or 0.0)
                gy = float(wy or 0.0)
                gz = float(wz or 0.0)
                self._set_from_global(gx, gy, gz)
            else:
                l1 = float(w1 or 0.0)
                l2 = float(w2 or 0.0)
                l3 = float(w3 or 0.0)
                self._set_from_local(l1, l2, l3)

        eq_local = self._compute_equivalent_loads()
        self._transform_and_store_equivalent_loads(eq_local)

    def _set_from_global(self, gx: float, gy: float, gz: float):
        self.coord_system = "global"
        self.wx = float(gx)
        self.wy = float(gy)
        self.wz = float(gz)
        R = self.element.rotation_matrix_3x3()
        w_loc = R @ np.array([self.wx, self.wy, self.wz], dtype=float)
        self.w1 = float(w_loc[0])
        self.w2 = float(w_loc[1])
        self.w3 = float(w_loc[2])

    def _set_from_local(self, l1: float, l2: float, l3: float):
        self.coord_system = "local"
        self.w1 = float(l1)
        self.w2 = float(l2)
        self.w3 = float(l3)
        R = self.element.rotation_matrix_3x3()
        w_glob = R.T @ np.array([self.w1, self.w2, self.w3], dtype=float)
        self.wx = float(w_glob[0])
        self.wy = float(w_glob[1])
        self.wz = float(w_glob[2])

    @property
    def local_components(self) -> np.ndarray:
        """Return 3-element vector of local load components [w1, w2, w3]."""
        return np.array([self.w1, self.w2, self.w3], dtype=float)

    @property
    def global_components(self) -> np.ndarray:
        """Return 3-element vector of global load components [wx, wy, wz]."""
        return np.array([self.wx, self.wy, self.wz], dtype=float)

    @classmethod
    def local(
        cls,
        element,
        w1: float = 0.0,
        w2: float = 0.0,
        w3: float = 0.0,
        is_self_weight: bool = False,
    ) -> "DistributedLoad":
        """Factory method: create a DistributedLoad directly in element local coordinates (1, 2, 3)."""
        return cls(element, w1=w1, w2=w2, w3=w3, coord_system="local", is_self_weight=is_self_weight)

    @classmethod
    def global_load(
        cls,
        element,
        wx: float = 0.0,
        wy: float = 0.0,
        wz: float = 0.0,
        is_self_weight: bool = False,
    ) -> "DistributedLoad":
        """Factory method: create a DistributedLoad directly in global coordinates (X, Y, Z)."""
        return cls(element, wx=wx, wy=wy, wz=wz, coord_system="global", is_self_weight=is_self_weight)

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

    def __repr__(self) -> str:
        return (
            f"DistributedLoad(element={self.element.id}, "
            f"local=[{self.w1:.4g}, {self.w2:.4g}, {self.w3:.4g}], "
            f"global=[{self.wx:.4g}, {self.wy:.4g}, {self.wz:.4g}], "
            f"coord_system='{self.coord_system}')"
        )


class ElementPointLoad(ElementLoad):
    """
    Concentrated point load acting on an element at distance x from start node (0 <= x <= L).

    Coordinates Convention:
    -----------------------
    - Local axes (1, 2, 3):
        p1: point force along local axis 1 (axial).
        p2: point force along local axis 2 (transverse, flexure in 1-2).
        p3: point force along local axis 3 (transverse, flexure in 1-3).
        m1: point moment about local axis 1 (torsion).
        m2: point moment about local axis 2 (bending in 1-3).
        m3: point moment about local axis 3 (bending in 1-2).
    - Global axes (X, Y, Z):
        px, py, pz: global concentrated forces along X, Y, Z.
        mx, my, mz: global concentrated moments about X, Y, Z.

    The load can be specified either in local coordinates (p1..p3, m1..m3) or in
    global coordinates (px..pz, mx..mz). The 3x3 direction cosine rotation matrix R
    transforms between local and global components:
        f_local = R @ f_global,  m_local = R @ m_global
        f_global = R.T @ f_local,  m_global = R.T @ m_local

    Regardless of which coordinate system was used to define the load, both sets
    of attributes are always synchronized and accessible on the instance.
    """

    def __init__(
        self,
        element,
        p1: Optional[float] = None,
        p2: Optional[float] = None,
        p3: Optional[float] = None,
        m1: Optional[float] = None,
        m2: Optional[float] = None,
        m3: Optional[float] = None,
        x: float = 0.0,
        coord_system: Optional[str] = None,
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
            Point force components along local 1, 2, 3 axes.
        m1, m2, m3 : float, optional
            Point moment components about local 1, 2, 3 axes.
        x : float, optional
            Distance from start node along element length.
        coord_system : str, optional
            'local' or 'global'. If omitted, automatically inferred from provided arguments.
        px, py, pz : float, optional
            Point force components along global X, Y, Z axes.
        mx, my, mz : float, optional
            Point moment components about global X, Y, Z axes.
        """
        super().__init__(element)
        self.x = float(x)

        has_local = any(v is not None for v in (p1, p2, p3, m1, m2, m3))
        has_global = any(v is not None for v in (px, py, pz, mx, my, mz))

        if has_local and has_global:
            loc_vals = [float(v or 0.0) for v in (p1, p2, p3, m1, m2, m3)]
            glob_vals = [float(v or 0.0) for v in (px, py, pz, mx, my, mz)]
            if any(abs(v) > 1e-12 for v in loc_vals) and any(abs(v) > 1e-12 for v in glob_vals):
                raise ValueError(
                    "Cannot specify both local components (p1..p3, m1..m3) and global components (px..pz, mx..mz) simultaneously. "
                    "Use p1..p3, m1..m3 for local element axes, or px..pz, mx..mz for global axes."
                )

        sys_spec = coord_system.lower() if coord_system is not None else None

        if sys_spec == "global":
            if has_local and not has_global:
                # Positional backward compatibility with coord_system="global"
                gx = float(p1 or 0.0)
                gy = float(p2 or 0.0)
                gz = float(p3 or 0.0)
                gmx = float(m1 or 0.0)
                gmy = float(m2 or 0.0)
                gmz = float(m3 or 0.0)
            else:
                gx = float(px or 0.0)
                gy = float(py or 0.0)
                gz = float(pz or 0.0)
                gmx = float(mx or 0.0)
                gmy = float(my or 0.0)
                gmz = float(mz or 0.0)
            self._set_from_global(gx, gy, gz, gmx, gmy, gmz)
        elif sys_spec == "local":
            if has_global and not has_local:
                raise ValueError(
                    "coord_system='local' was specified, but global components were provided. "
                    "Use p1..p3, m1..m3 for local element coordinates."
                )
            self._set_from_local(
                float(p1 or 0.0),
                float(p2 or 0.0),
                float(p3 or 0.0),
                float(m1 or 0.0),
                float(m2 or 0.0),
                float(m3 or 0.0),
            )
        else:
            if has_global:
                gx = float(px or 0.0)
                gy = float(py or 0.0)
                gz = float(pz or 0.0)
                gmx = float(mx or 0.0)
                gmy = float(my or 0.0)
                gmz = float(mz or 0.0)
                self._set_from_global(gx, gy, gz, gmx, gmy, gmz)
            else:
                self._set_from_local(
                    float(p1 or 0.0),
                    float(p2 or 0.0),
                    float(p3 or 0.0),
                    float(m1 or 0.0),
                    float(m2 or 0.0),
                    float(m3 or 0.0),
                )

        eq_local = self._compute_equivalent_loads()
        self._transform_and_store_equivalent_loads(eq_local)

    def _set_from_global(self, gx: float, gy: float, gz: float, gmx: float, gmy: float, gmz: float):
        self.coord_system = "global"
        self.px = float(gx)
        self.py = float(gy)
        self.pz = float(gz)
        self.mx = float(gmx)
        self.my = float(gmy)
        self.mz = float(gmz)
        R = self.element.rotation_matrix_3x3()
        f_loc = R @ np.array([self.px, self.py, self.pz], dtype=float)
        m_loc = R @ np.array([self.mx, self.my, self.mz], dtype=float)
        self.p1, self.p2, self.p3 = float(f_loc[0]), float(f_loc[1]), float(f_loc[2])
        self.m1, self.m2, self.m3 = float(m_loc[0]), float(m_loc[1]), float(m_loc[2])

    def _set_from_local(self, l1: float, l2: float, l3: float, lm1: float, lm2: float, lm3: float):
        self.coord_system = "local"
        self.p1 = float(l1)
        self.p2 = float(l2)
        self.p3 = float(l3)
        self.m1 = float(lm1)
        self.m2 = float(lm2)
        self.m3 = float(lm3)
        R = self.element.rotation_matrix_3x3()
        f_glob = R.T @ np.array([self.p1, self.p2, self.p3], dtype=float)
        m_glob = R.T @ np.array([self.m1, self.m2, self.m3], dtype=float)
        self.px = float(f_glob[0])
        self.py = float(f_glob[1])
        self.pz = float(f_glob[2])
        self.mx = float(m_glob[0])
        self.my = float(m_glob[1])
        self.mz = float(m_glob[2])

    @property
    def local_forces(self) -> np.ndarray:
        """Return 3-element vector of local forces [p1, p2, p3]."""
        return np.array([self.p1, self.p2, self.p3], dtype=float)

    @property
    def local_moments(self) -> np.ndarray:
        """Return 3-element vector of local moments [m1, m2, m3]."""
        return np.array([self.m1, self.m2, self.m3], dtype=float)

    @property
    def global_forces(self) -> np.ndarray:
        """Return 3-element vector of global forces [px, py, pz]."""
        return np.array([self.px, self.py, self.pz], dtype=float)

    @property
    def global_moments(self) -> np.ndarray:
        """Return 3-element vector of global moments [mx, my, mz]."""
        return np.array([self.mx, self.my, self.mz], dtype=float)

    @classmethod
    def local(
        cls,
        element,
        p1: float = 0.0,
        p2: float = 0.0,
        p3: float = 0.0,
        m1: float = 0.0,
        m2: float = 0.0,
        m3: float = 0.0,
        x: float = 0.0,
    ) -> "ElementPointLoad":
        """Factory method: create an ElementPointLoad directly in element local coordinates (1, 2, 3)."""
        return cls(element, p1=p1, p2=p2, p3=p3, m1=m1, m2=m2, m3=m3, x=x, coord_system="local")

    @classmethod
    def global_load(
        cls,
        element,
        px: float = 0.0,
        py: float = 0.0,
        pz: float = 0.0,
        mx: float = 0.0,
        my: float = 0.0,
        mz: float = 0.0,
        x: float = 0.0,
    ) -> "ElementPointLoad":
        """Factory method: create an ElementPointLoad directly in global coordinates (X, Y, Z)."""
        return cls(element, px=px, py=py, pz=pz, mx=mx, my=my, mz=mz, x=x, coord_system="global")

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

    def __repr__(self) -> str:
        return (
            f"ElementPointLoad(element={self.element.id}, x={self.x}, "
            f"local_force=[{self.p1:.4g}, {self.p2:.4g}, {self.p3:.4g}], "
            f"global_force=[{self.px:.4g}, {self.py:.4g}, {self.pz:.4g}], "
            f"coord_system='{self.coord_system}')"
        )
