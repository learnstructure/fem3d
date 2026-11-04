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
        if hasattr(self.element, "add_equivalent_load"):
            self.element.add_equivalent_load(eq_local)
        else:
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
        w1_end: Optional[float] = None,
        w2_end: Optional[float] = None,
        w3_end: Optional[float] = None,
        coord_system: Optional[str] = None,
        wx: Optional[float] = None,
        wy: Optional[float] = None,
        wz: Optional[float] = None,
        wx_end: Optional[float] = None,
        wy_end: Optional[float] = None,
        wz_end: Optional[float] = None,
        start: float = 0.0,
        end: float = 1.0,
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
        if start < 0.0 or end > 1.0 or start >= end:
            raise ValueError(f"DistributedLoad relative span must satisfy 0 <= start < end <= 1, got start={start}, end={end}")
        self.start = float(start)
        self.end = float(end)

        has_local = any(v is not None for v in (w1, w2, w3, w1_end, w2_end, w3_end))
        has_global = any(v is not None for v in (wx, wy, wz, wx_end, wy_end, wz_end))

        # Check for conflicting inputs
        if has_local and has_global:
            loc_vals = [float(v or 0.0) for v in (w1, w2, w3, w1_end, w2_end, w3_end)]
            glob_vals = [float(v or 0.0) for v in (wx, wy, wz, wx_end, wy_end, wz_end)]
            if any(abs(v) > 1e-12 for v in loc_vals) and any(abs(v) > 1e-12 for v in glob_vals):
                raise ValueError(
                    "Cannot specify both local components and global components simultaneously. "
                    "Use w1, w2, w3 for local element axes, or wx, wy, wz for global axes."
                )

        sys_spec = coord_system.lower() if coord_system is not None else None

        if sys_spec == "global":
            if has_local and not has_global:
                raise ValueError(
                    f"coord_system='global' was specified, but local components were provided. "
                    "Use wx, wy, wz for global coordinates."
                )
            gx = float(wx or 0.0)
            gy = float(wy or 0.0)
            gz = float(wz or 0.0)
            gxe = float(gx if wx_end is None else wx_end)
            gye = float(gy if wy_end is None else wy_end)
            gze = float(gz if wz_end is None else wz_end)
            self._set_from_global(gx, gy, gz, gxe, gye, gze)
        elif sys_spec == "local":
            if has_global and not has_local:
                raise ValueError(
                    f"coord_system='local' was specified, but global components were provided. "
                    "Use w1, w2, w3 for local element coordinates."
                )
            l1 = float(w1 or 0.0)
            l2 = float(w2 or 0.0)
            l3 = float(w3 or 0.0)
            l1e = float(l1 if w1_end is None else w1_end)
            l2e = float(l2 if w2_end is None else w2_end)
            l3e = float(l3 if w3_end is None else w3_end)
            self._set_from_local(l1, l2, l3, l1e, l2e, l3e)
        else:
            if has_global:
                gx = float(wx or 0.0)
                gy = float(wy or 0.0)
                gz = float(wz or 0.0)
                gxe = float(gx if wx_end is None else wx_end)
                gye = float(gy if wy_end is None else wy_end)
                gze = float(gz if wz_end is None else wz_end)
                self._set_from_global(gx, gy, gz, gxe, gye, gze)
            else:
                l1 = float(w1 or 0.0)
                l2 = float(w2 or 0.0)
                l3 = float(w3 or 0.0)
                l1e = float(l1 if w1_end is None else w1_end)
                l2e = float(l2 if w2_end is None else w2_end)
                l3e = float(l3 if w3_end is None else w3_end)
                self._set_from_local(l1, l2, l3, l1e, l2e, l3e)

        eq_local = self._compute_equivalent_loads()
        self._transform_and_store_equivalent_loads(eq_local)

    def _set_from_global(self, gx: float, gy: float, gz: float, gxe: float, gye: float, gze: float):
        self.coord_system = "global"
        self.wx, self.wy, self.wz = float(gx), float(gy), float(gz)
        self.wx_end, self.wy_end, self.wz_end = float(gxe), float(gye), float(gze)
        R = self.element.rotation_matrix_3x3()
        w_loc = R @ np.array([self.wx, self.wy, self.wz], dtype=float)
        w_loc_end = R @ np.array([self.wx_end, self.wy_end, self.wz_end], dtype=float)
        self.w1, self.w2, self.w3 = float(w_loc[0]), float(w_loc[1]), float(w_loc[2])
        self.w1_end, self.w2_end, self.w3_end = float(w_loc_end[0]), float(w_loc_end[1]), float(w_loc_end[2])

    def _set_from_local(self, l1: float, l2: float, l3: float, l1e: float, l2e: float, l3e: float):
        self.coord_system = "local"
        self.w1, self.w2, self.w3 = float(l1), float(l2), float(l3)
        self.w1_end, self.w2_end, self.w3_end = float(l1e), float(l2e), float(l3e)
        R = self.element.rotation_matrix_3x3()
        w_glob = R.T @ np.array([self.w1, self.w2, self.w3], dtype=float)
        w_glob_end = R.T @ np.array([self.w1_end, self.w2_end, self.w3_end], dtype=float)
        self.wx, self.wy, self.wz = float(w_glob[0]), float(w_glob[1]), float(w_glob[2])
        self.wx_end, self.wy_end, self.wz_end = float(w_glob_end[0]), float(w_glob_end[1]), float(w_glob_end[2])

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
        Compute work-equivalent nodal forces and moments in local coordinates via 3-point Gauss-Legendre quadrature.
        Supports uniform and linearly varying distributed loads on full or partial spans [start, end].
        """
        L = self.element.length
        eq = np.zeros(12, dtype=float)
        gp = [-np.sqrt(0.6), 0.0, np.sqrt(0.6)]
        gw = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
        a = self.start * L
        b = self.end * L
        half_len = (b - a) / 2.0
        mid = (a + b) / 2.0

        for xi, w in zip(gp, gw):
            x = mid + half_len * xi
            mu = x / L
            alpha = (xi + 1.0) / 2.0
            cur_w1 = self.w1 + alpha * (self.w1_end - self.w1)
            cur_w2 = self.w2 + alpha * (self.w2_end - self.w2)
            cur_w3 = self.w3 + alpha * (self.w3_end - self.w3)

            n_ax1 = 1.0 - mu
            n_ax2 = mu

            n_tr1 = 1.0 - 3.0 * mu**2 + 2.0 * mu**3
            n_tr2 = x * (1.0 - mu)**2
            n_tr3 = 3.0 * mu**2 - 2.0 * mu**3
            n_tr4 = x * (mu**2 - mu)

            wt = w * half_len

            # Axial (local axis 1)
            eq[0] += wt * n_ax1 * cur_w1
            eq[6] += wt * n_ax2 * cur_w1

            # Transverse w2 (bending in 1-2 about axis 3)
            eq[1] += wt * n_tr1 * cur_w2
            eq[5] += wt * n_tr2 * cur_w2
            eq[7] += wt * n_tr3 * cur_w2
            eq[11] += wt * n_tr4 * cur_w2

            # Transverse w3 (bending in 1-3 about axis 2)
            eq[2] += wt * n_tr1 * cur_w3
            eq[4] += wt * (-n_tr2) * cur_w3
            eq[8] += wt * n_tr3 * cur_w3
            eq[10] += wt * (-n_tr4) * cur_w3

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
                raise ValueError(
                    "coord_system='global' was specified, but local components (p1, p2, p3, m1, m2, m3) were provided. "
                    "Use px, py, pz, mx, my, mz for global coordinates."
                )
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

        # Concentrated moment m3 (bending in 1-2 about axis 3)
        if abs(self.m3) > 1e-12:
            eq[1] -= (6.0 * a * b * self.m3) / (L**3)
            eq[7] += (6.0 * a * b * self.m3) / (L**3)
            eq[5] += (b * (2.0 * a - b) * self.m3) / (L**2)
            eq[11] += (a * (2.0 * b - a) * self.m3) / (L**2)

        # Concentrated moment m2 (bending in 1-3 about axis 2)
        if abs(self.m2) > 1e-12:
            eq[2] += (6.0 * a * b * self.m2) / (L**3)
            eq[8] -= (6.0 * a * b * self.m2) / (L**3)
            eq[4] += (b * (2.0 * a - b) * self.m2) / (L**2)
            eq[10] += (a * (2.0 * b - a) * self.m2) / (L**2)

        return eq

    def __repr__(self) -> str:
        return (
            f"ElementPointLoad(element={self.element.id}, x={self.x}, "
            f"local_force=[{self.p1:.4g}, {self.p2:.4g}, {self.p3:.4g}], "
            f"global_force=[{self.px:.4g}, {self.py:.4g}, {self.pz:.4g}], "
            f"coord_system='{self.coord_system}')"
        )
