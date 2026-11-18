"""Four-node flat shell element (fem3d/SHELL_PLAN.md, milestone M5).

The membrane with drilling rotations (``shell_membrane``) and the MITC4 plate (``shell_plate``) are combined in one
flat element with six DOFs per node (Wilson 2002, §10.2: "the 24 by 24 local element stiffness matrix ... is then
transformed to the global XYZ reference system"). Membrane and plate are independent in the element plane.

Three frames are kept apart:

- global X-Y-Z, in which nodes, loads, and displacements are given;
- the local axes 1-2-3 of each element, from ``struct_core.geometry.shell_local_axes`` (CSI Analysis Reference
  Manual §10.6-10.7), in which material directions, loads along local 3, and every result are expressed;
- the internal formulation frame: the corners projected on the element's mean plane (through the centroid of the
  corners, normal to axis 3), with coordinates measured along axes 1 and 2, and the natural coordinates (r, s) of the
  kernels. Because the kernels are invariant under in-plane rotation (tests of M3 and M4), measuring the projected
  coordinates along axes 1 and 2 gives stress resultants directly in the local 1-2-3 axes.

Warped elements (Wilson §10.3, Eqs. 10.1 and 10.2): a corner at height d above the mean plane is joined to its
projection by a rigid link, u_flat = u_node + theta x (-d e3), that is u1_flat = u1 - d theta2 and
u2_flat = u2 + d theta1 (Wilson Eq. 10.2, with d the offset of the mid-surface node from the flat-element node). The
shape checks of ``struct_core.geometry`` (CSI §10.2.1) refuse warping above 45 degrees.

Mass (CSI §10.11): mass per unit area rho x th (membrane thickness). Lumped mass is on the translations only, in
proportion to the diagonal of the consistent mass matrix (``struct_core.geometry.shell_corner_weights``, the same
fractions struct_load uses for the mass source); the consistent mass is rho th integral(N_i N_j) dA on each
translation. Self-weight (CSI §10.12) is the unit weight times th.

Only quadrilaterals are implemented; triangles (Allman membrane, DKT plate) are planned for a later session.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .shell_membrane import GAUSS_POINTS_2X2, _bilinear, membrane_resultants, membrane_stiffness, plane_stress_matrix
from .shell_plate import plate_pressure_loads, plate_resultants, plate_rigidities, plate_stiffness

SHELL_BEHAVIORS = ("shell", "membrane", "plate")

# Positions of the membrane DOFs (u1, u2, theta3) and plate DOFs (u3, theta1, theta2) among the six local DOFs of a
# node (u1, u2, u3, theta1, theta2, theta3)
_MEMBRANE_LOCAL = np.array([0, 1, 5])
_PLATE_LOCAL = np.array([2, 3, 4])
_MEMBRANE_DOFS = (6 * np.arange(4)[:, None] + _MEMBRANE_LOCAL).ravel()
_PLATE_DOFS = (6 * np.arange(4)[:, None] + _PLATE_LOCAL).ravel()

#: Extrapolation from the 2x2 Gauss points to the corners: corners = EXTRAPOLATION @ gauss_values. The bilinear
#: field through the Gauss-point values, evaluated at the corners (r, s = +-1, that is +-sqrt(3) in the coordinates
#: of the Gauss points).
EXTRAPOLATION = np.array([_bilinear(np.sqrt(3.0) * r, np.sqrt(3.0) * s)[0] for r, s in np.sign(GAUSS_POINTS_2X2)])


class ShellElement:
    """Four-node flat shell (membrane with drilling rotations + MITC4 plate), six DOFs per node.

    Parameters
    ----------
    eid : int or str
        Element ID.
    nodes : sequence of 4 Node
        Corner nodes in cyclic order; local axis 3 points toward the viewer when they appear counterclockwise.
    material : ElasticMaterial
        Isotropic material (E, nu, G, rho, unit_weight).
    thickness : float
        Membrane thickness th: membrane stiffness, mass, and self-weight (CSI §10.3).
    bending_thickness : float, optional
        Bending thickness thb: plate bending and transverse shear stiffness. Defaults to ``thickness``.
    behavior : {"shell", "membrane", "plate"}
        Full shell, membrane only, or plate only. A membrane gives no stiffness to the out-of-plane translation and
        the in-plane rotations, and a plate none to the in-plane translations and the drilling rotation: those DOFs
        must be restrained or connected to other elements (as in SAP2000).
    axis_angle : float
        CSI coordinate angle "ang" in degrees (``struct_core.geometry.shell_local_axes``).
    reference_vector : sequence of 3 floats, optional
        Vector in the local 3-2 plane (``struct_core.geometry.shell_local_axes``).
    """

    def __init__(
        self,
        eid: Any,
        nodes: Sequence[Any],
        material: Any,
        thickness: float,
        bending_thickness: Optional[float] = None,
        behavior: str = "shell",
        axis_angle: float = 0.0,
        reference_vector: Optional[Sequence[float]] = None,
    ):
        nodes = list(nodes)
        if len(nodes) != 4:
            raise ValueError(
                f"Shell element {eid!r}: fem3d implements four-node shells only; triangles are planned "
                f"(fem3d/SHELL_PLAN.md, later sessions), got {len(nodes)} nodes"
            )
        if len({id(n) for n in nodes}) != 4:
            raise ValueError(f"Shell element {eid!r}: the four nodes must be distinct")
        if behavior not in SHELL_BEHAVIORS:
            raise ValueError(f"Shell element {eid!r}: behavior must be one of {SHELL_BEHAVIORS}, got {behavior!r}")
        if not thickness > 0.0:
            raise ValueError(f"Shell element {eid!r}: thickness must be positive")
        if bending_thickness is not None and not bending_thickness > 0.0:
            raise ValueError(f"Shell element {eid!r}: bending_thickness must be positive")
        self.id = eid
        self.nodes = nodes
        self.material = material
        self.thickness = float(thickness)
        self.bending_thickness = float(bending_thickness) if bending_thickness is not None else None
        self.behavior = behavior
        self.axis_angle = float(axis_angle)
        self.reference_vector = None if reference_vector is None else tuple(float(v) for v in reference_vector)
        self.structure = None

    @property
    def effective_bending_thickness(self) -> float:
        return self.bending_thickness if self.bending_thickness is not None else self.thickness

    @property
    def coords(self) -> np.ndarray:
        return np.array([[n.x, n.y, n.z] for n in self.nodes], dtype=float)

    @property
    def dofs(self) -> List[int]:
        return [d for n in self.nodes for d in n.dofs]

    def __repr__(self) -> str:
        return (f"ShellElement(id={self.id!r}, nodes={[n.id for n in self.nodes]}, thickness={self.thickness}, "
                f"behavior={self.behavior!r})")


class ShellBatch:
    """The shell elements of a structure, computed together (all arrays have a leading dimension N).

    Attributes
    ----------
    ids : list of element IDs, in the order of the arrays
    coords : (N, 4, 3) corner coordinates
    axes : (N, 3, 3) local axes, rows e1, e2, e3 (``struct_core.geometry.shell_local_axes``)
    xy : (N, 4, 2) corners projected on the mean plane, measured along axes 1 and 2 from the centroid
    offsets : (N, 4) height of each corner above the mean plane, along axis 3
    transform : (N, 24, 24) from the global DOFs of the four nodes to the flat element's local DOFs
    """

    def __init__(self, shells: Sequence[ShellElement]):
        try:
            from struct_core.geometry import shell_local_axes
        except ImportError as err:  # the local axes of shells are defined once, in struct_core.geometry
            raise ImportError(
                "Shell elements need the 'struct_core' package for their local axes (struct_core.geometry). "
                "Install it with: pip install fem3d[ecosystem]"
            ) from err

        self.shells = list(shells)
        self.ids = [s.id for s in self.shells]
        n = len(self.shells)
        self.coords = np.array([s.coords for s in self.shells], dtype=float).reshape(n, 4, 3)
        ref = np.array([s.reference_vector if s.reference_vector is not None else (np.nan,) * 3 for s in self.shells])
        self.axes = shell_local_axes(self.coords, np.array([s.axis_angle for s in self.shells]), ref.reshape(n, 3))
        rel = self.coords - self.coords.mean(axis=1, keepdims=True)
        local = np.einsum("nij,nkj->nki", self.axes, rel)  # components along e1, e2, e3
        self.xy = local[:, :, :2]
        self.offsets = local[:, :, 2]
        self.dofs = np.array([s.dofs for s in self.shells], dtype=np.int64).reshape(n, 24)
        self.transform = self._transform()

        E = np.array([s.material.E for s in self.shells], dtype=float)
        nu = np.array([s.material.nu for s in self.shells], dtype=float)
        G = np.array([s.material.G for s in self.shells], dtype=float)
        self.thickness = np.array([s.thickness for s in self.shells], dtype=float)
        self.bending_thickness = np.array([s.effective_bending_thickness for s in self.shells], dtype=float)
        self.has_membrane = np.array([s.behavior in ("shell", "membrane") for s in self.shells])
        self.has_plate = np.array([s.behavior in ("shell", "plate") for s in self.shells])
        self.D = plane_stress_matrix(E, nu)
        self.Cb, self.Cs = plate_rigidities(E, nu, G, self.bending_thickness)

    def _transform(self) -> np.ndarray:
        """B = A R per node: R rotates global translations and rotations to local axes, A is the rigid link of a
        warped corner to its projection (u1 - d theta2, u2 + d theta1; Wilson Eq. 10.2)."""
        n = len(self.shells)
        T = np.zeros((n, 24, 24))
        for k in range(4):
            block = np.zeros((n, 6, 6))
            block[:, :3, :3] = self.axes
            block[:, 3:, 3:] = self.axes
            link = np.broadcast_to(np.eye(6), (n, 6, 6)).copy()
            link[:, 0, 4] = -self.offsets[:, k]
            link[:, 1, 3] = self.offsets[:, k]
            T[:, 6 * k : 6 * k + 6, 6 * k : 6 * k + 6] = link @ block
        return T

    # -- stiffness and mass ------------------------------------------------------------------------------------------
    def local_stiffness(self) -> np.ndarray:
        """Stiffness (N, 24, 24) on the flat element's local DOFs (u1, u2, u3, theta1, theta2, theta3 per node)."""
        n = len(self.shells)
        K = np.zeros((n, 24, 24))
        m = np.flatnonzero(self.has_membrane)
        if m.size:
            K[np.ix_(m, _MEMBRANE_DOFS, _MEMBRANE_DOFS)] = membrane_stiffness(self.xy[m], self.D[m], self.thickness[m])
        p = np.flatnonzero(self.has_plate)
        if p.size:
            K[np.ix_(p, _PLATE_DOFS, _PLATE_DOFS)] += plate_stiffness(self.xy[p], self.Cb[p], self.Cs[p])
        return K

    def global_stiffness(self) -> np.ndarray:
        """Stiffness (N, 24, 24) on the global DOFs of the four nodes: B^T K B."""
        B = self.transform
        return np.einsum("nai,nab,nbj->nij", B, self.local_stiffness(), B)

    def areas(self) -> np.ndarray:
        """Area of each element projected on its plane (normal to axis 3), the area of the flat element. It equals
        ``struct_core.geometry.shell_areas`` except for nearly planar warped elements, whose axis 3 is the normal of
        the plane through j1, j2, j3 (``shell_local_axes``, Notes): they differ by a factor of at least 0.9999."""
        return self.nodal_areas().sum(axis=1)

    def mass(self, lumped: bool) -> np.ndarray:
        """Mass (N, 24, 24) on the global DOFs, translations only (CSI §10.11)."""
        n = len(self.shells)
        rho_t = np.array([s.material.rho for s in self.shells], dtype=float) * self.thickness
        M = np.zeros((n, 24, 24))
        if lumped:
            from struct_core.geometry import shell_corner_weights

            per_corner = (rho_t * self.areas())[:, None] * shell_corner_weights(self.coords)
            for k in range(4):
                for d in range(3):
                    M[:, 6 * k + d, 6 * k + d] = per_corner[:, k]
            return M
        nn = np.zeros((n, 4, 4))
        for r, s in GAUSS_POINTS_2X2:
            N, dN_rs = _bilinear(r, s)
            J = np.einsum("ak,nkb->nab", dN_rs, self.xy)
            detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            nn += detJ[:, None, None] * np.outer(N, N)[None]
        nn *= rho_t[:, None, None]
        for d in range(3):
            idx = 6 * np.arange(4) + d
            M[:, idx[:, None], idx[None, :]] = nn
        return M

    # -- loads ------------------------------------------------------------------------------------------------------
    def nodal_areas(self) -> np.ndarray:
        """Consistent nodal shares of a unit uniform load, integral(N_i) dA on the flat element, shape (N, 4)."""
        return plate_pressure_loads(self.xy, 1.0)[:, 0::3]

    def uniform_load(self, index: int, pressure: float, direction: str) -> np.ndarray:
        """Consistent global nodal forces (24,) of a uniform load per unit area on shell ``index``: along local axis 3
        (``direction="local_3"``) or along a global axis ("x", "y", "z"); the sign of ``pressure`` gives the sense."""
        if direction == "local_3":
            vec = self.axes[index, 2]
        elif direction in ("x", "y", "z"):
            vec = np.eye(3)["xyz".index(direction)]
        else:
            raise ValueError(f"Shell load direction must be 'local_3', 'x', 'y', or 'z', got {direction!r}")
        shares = self.nodal_areas()[index]
        f = np.zeros(24)
        for k in range(4):
            f[6 * k : 6 * k + 3] = pressure * shares[k] * vec
        return f

    # -- results ----------------------------------------------------------------------------------------------------
    def resultants(self, u_global: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Stress resultants (F11, F22, F12, M11, M22, M12, V13, V23) per unit length in the local 1-2-3 axes, at the
        2x2 Gauss points (N, 4, 8) and extrapolated to the corners (N, 4, 8), from the global displacements of the
        structure (``u_global`` of size neq)."""
        n = len(self.shells)
        u_local = np.einsum("nij,nj->ni", self.transform, np.asarray(u_global, dtype=float)[self.dofs])
        gauss = np.zeros((n, 4, 8))
        m = np.flatnonzero(self.has_membrane)
        if m.size:
            gauss[m, :, 0:3] = membrane_resultants(self.xy[m], self.D[m], self.thickness[m], u_local[m][:, _MEMBRANE_DOFS])
        p = np.flatnonzero(self.has_plate)
        if p.size:
            gauss[p, :, 3:8] = plate_resultants(self.xy[p], self.Cb[p], self.Cs[p], u_local[p][:, _PLATE_DOFS])
        corners = np.einsum("ck,nkf->ncf", EXTRAPOLATION, gauss)
        return gauss, corners
