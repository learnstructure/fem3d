"""
ElementBase module defining the base class for 3D structural finite elements.
"""

import math
from typing import Optional, Union
import numpy as np


class ElementBase:
    """
    Base class for structural elements in a 3D finite element model.

    Handles 3D geometry calculation, element orientation vectors, and
    the 12x12 coordinate transformation matrix between local and global systems.

    Coordinate Transformation Theory
    --------------------------------
    References:
    - Weaver, W., & Gere, J. M. (1990). Matrix Analysis of Framed Structures (3rd ed.).
      Van Nostrand Reinhold, Section 4.10, pp. 240-244.
    - McGuire, W., Gallagher, R. H., & Ziemian, R. D. (2000). Matrix Structural Analysis
      (2nd ed.). John Wiley & Sons, Section 5.1.
    - Przemieniecki, J. S. (1968). Theory of Matrix Structural Analysis.
      McGraw-Hill, Section 4.5.

    Let the element span from node i (xi, yi, zi) to node j (xj, yj, zj).
    The local x'-axis is directed along the member centroidal axis:
        vx = (node_j - node_i) / L = [lx, mx, nx]
    
    The local principal y' and z' axes form a right-handed orthogonal triad:
        vx . vy = 0,   vx . vz = 0,   vy . vz = 0
        vx x vy = vz

    By default, global Z is considered the vertical axis (standard civil/structural convention).
    - For non-vertical members (not parallel to global Z):
        vz = (vx x [0, 0, 1]) / ||vx x [0, 0, 1]||
        vy = vz x vx
    - For vertical members parallel to global Z (lx = 0, mx = 0):
        If nx > 0 (pointing +Z):
            vy = [0, 1, 0], vz = [-1, 0, 0]
        If nx < 0 (pointing -Z):
            vy = [0, -1, 0], vz = [-1, 0, 0]
    
    If an explicit roll angle beta (in degrees or radians) or web_vector is supplied,
    the triad is rotated about vx accordingly.

    The 3x3 direction cosine rotation matrix R is:
        R = [ vx^T ]
            [ vy^T ]
            [ vz^T ]
    
    The 12x12 global-to-local transformation matrix T is block-diagonal:
        T = diag(R, R, R, R)
    """

    def __init__(
        self,
        eid: Union[int, str],
        node_i,
        node_j,
        roll_angle: float = 0.0,
        web_vector: Optional[np.ndarray] = None,
    ):
        """
        Initialize an ElementBase object.

        Parameters
        ----------
        eid : int or str
            Unique identifier of the element.
        node_i : Node
            Start node.
        node_j : Node
            End node.
        roll_angle : float, optional
            Member roll/web angle in degrees about local x'-axis. Defaults to 0.0.
        web_vector : numpy.ndarray, optional
            Reference orientation vector pointing toward local y' or z' axis.
        """
        self.id = eid
        self.node_i = node_i
        self.node_j = node_j
        self.roll_angle = float(roll_angle)
        self.web_vector = np.asarray(web_vector, dtype=float) if web_vector is not None else None
        self.structure = None

        self._update_geometry()

    def _update_geometry(self):
        """Compute length, orientation unit vectors vx, vy, vz, and rotation matrix R."""
        dx = self.node_j.x - self.node_i.x
        dy = self.node_j.y - self.node_i.y
        dz = self.node_j.z - self.node_i.z

        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.length = math.sqrt(dx * dx + dy * dy + dz * dz)

        if self.length < 1e-12:
            raise ValueError(
                f"Element {self.id} has zero length: node_i and node_j are coincident."
            )

        # Unit vector along member axis (local x')
        vx = np.array([dx, dy, dz], dtype=float) / self.length
        self.vx = vx

        # Determine reference local axes
        if self.web_vector is not None:
            # User supplied custom orientation vector
            v_ref = self.web_vector
            # Project reference vector perpendicular to vx
            vz_temp = np.cross(vx, v_ref)
            norm_z = np.linalg.norm(vz_temp)
            if norm_z < 1e-8:
                raise ValueError(
                    f"Element {self.id}: web_vector is parallel to the member axis."
                )
            vz = vz_temp / norm_z
            vy = np.cross(vz, vx)
        else:
            # Default orientation: global Z is vertical
            # Check if member is vertical (parallel to global Z)
            proj_xy = math.sqrt(vx[0] ** 2 + vx[1] ** 2)
            if proj_xy > 1e-6:
                # Non-vertical member: local z' is horizontal (in XY plane)
                # vz = (vx x [0, 0, 1]) / norm
                # vx x [0, 0, 1] = [vx[1], -vx[0], 0]
                vz = np.array([vx[1], -vx[0], 0.0], dtype=float) / proj_xy
                vy = np.cross(vz, vx)
            else:
                # Vertical member along global Z
                if vx[2] > 0:  # pointing +Z
                    vy = np.array([0.0, 1.0, 0.0], dtype=float)
                    vz = np.array([-1.0, 0.0, 0.0], dtype=float)
                else:  # pointing -Z
                    vy = np.array([0.0, -1.0, 0.0], dtype=float)
                    vz = np.array([-1.0, 0.0, 0.0], dtype=float)

        # Apply roll angle beta if non-zero (rotate vy and vz about vx)
        if abs(self.roll_angle) > 1e-9:
            beta = math.radians(self.roll_angle)
            cos_b = math.cos(beta)
            sin_b = math.sin(beta)
            vy_rot = cos_b * vy + sin_b * vz
            vz_rot = -sin_b * vy + cos_b * vz
            vy = vy_rot
            vz = vz_rot

        # Ensure unit lengths
        self.vy = vy / np.linalg.norm(vy)
        self.vz = vz / np.linalg.norm(vz)

        # 3x3 Direction Cosine Rotation Matrix R:
        # local_vector = R @ global_vector
        self.R = np.array([self.vx, self.vy, self.vz], dtype=float)

    def rotation_matrix_3x3(self) -> np.ndarray:
        """Return the 3x3 direction cosine rotation matrix."""
        return self.R.copy()

    def transformation_matrix(self, dof_per_node: int = 6) -> np.ndarray:
        """
        Return the transformation matrix from global to local coordinates.

        For a standard 3D frame element with 6 DOFs per node (total 12 DOFs):
            u_local = T @ u_global
            F_global = T.T @ F_local
            K_global = T.T @ K_local @ T

        Parameters
        ----------
        dof_per_node : int, optional
            Number of DOFs per node (3 for truss, 6 for frame). Defaults to 6.

        Returns
        -------
        numpy.ndarray
            Transformation matrix of size (2*dof_per_node, 2*dof_per_node).
        """
        R = self.R
        if dof_per_node == 6:
            # 12x12 block diagonal matrix
            T = np.zeros((12, 12), dtype=float)
            T[0:3, 0:3] = R
            T[3:6, 3:6] = R
            T[6:9, 6:9] = R
            T[9:12, 9:12] = R
            return T
        elif dof_per_node == 3:
            # 6x6 for 3D truss (translations only)
            T = np.zeros((6, 6), dtype=float)
            T[0:3, 0:3] = R
            T[3:6, 3:6] = R
            return T
        else:
            raise ValueError(f"Unsupported dof_per_node: {dof_per_node}")

    def local_stiffness(self) -> np.ndarray:
        """Return element stiffness matrix in local coordinates. Subclasses must implement."""
        raise NotImplementedError

    def global_stiffness(self) -> np.ndarray:
        """
        Assemble global stiffness matrix:
            K_global = T.T @ K_local @ T
        """
        k_local = self.local_stiffness()
        T = self.transformation_matrix()
        return T.T @ k_local @ T

    def local_mass_matrix(self, lumped: bool = False) -> np.ndarray:
        """Return element mass matrix in local coordinates. Subclasses implement."""
        raise NotImplementedError

    def mass_matrix(self, lumped: bool = False) -> np.ndarray:
        """Return element mass matrix in global coordinates."""
        m_local = self.local_mass_matrix(lumped=lumped)
        T = self.transformation_matrix()
        return T.T @ m_local @ T

    def equivalent_nodal_loads(self) -> np.ndarray:
        """Return equivalent nodal loads in global coordinates (size 12)."""
        if hasattr(self, "eq_load") and self.eq_load is not None:
            return self.eq_load
        return np.zeros(12, dtype=float)

    def get_local_displacements(self, global_disp: np.ndarray) -> np.ndarray:
        """
        Extract and transform nodal global displacements to local coordinates.

        Parameters
        ----------
        global_disp : numpy.ndarray
            Full global displacement vector of the structure.

        Returns
        -------
        numpy.ndarray
            Local displacement vector of size 12.
        """
        u_i = global_disp[self.node_i.dofs]
        u_j = global_disp[self.node_j.dofs]
        u_g = np.concatenate([u_i, u_j])
        T = self.transformation_matrix(dof_per_node=len(u_i))
        return T @ u_g

    def get_local_forces(self) -> np.ndarray:
        """
        Compute member end forces in local coordinates:
            f_local = K_local @ u_local - eq_load_local
        Subclasses must implement or specialize.
        """
        raise NotImplementedError
