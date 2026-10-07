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

    Coordinate Systems & Local Triad Definition
    -------------------------------------------
    `fem3d` uses a right-handed Cartesian global system (X, Y, Z) with **+Z vertical upward**.
    Each 3D element establishes a right-handed orthogonal local coordinate triad (v1, v2, v3):
      - **v1 (local 1-axis)**: Directed along the member centroidal axis from node_i to node_j:
            v1 = (node_j - node_i) / L = [v1_x, v1_y, v1_z]
      - **v2 (local 2-axis)**: Cross-sectional depth/web axis. Associated with bending moment I3
        and transverse displacement u_2.
      - **v3 (local 3-axis)**: Cross-sectional width/flange axis. Associated with bending moment I2
        and transverse displacement u_3.
      - Orthogonality and right-hand rule:
            v1 · v2 = 0,   v1 · v3 = 0,   v2 · v3 = 0
            v1 × v2 = v3,  v2 × v3 = v1,  v3 × v1 = v2

    Default Unrolled Local Axes (roll_angle = 0.0, web_vector = None):
    ------------------------------------------------------------------
    1. **Non-vertical members** (axis not parallel to global Z, sqrt(v1_x^2 + v1_y^2) > 0):
       - v3 is chosen to lie in the horizontal global XY plane (perpendicular to v1 and global Z):
             v3 = (v1 × [0, 0, 1]) / ||v1 × [0, 0, 1]|| = [v1_y, -v1_x, 0] / sqrt(v1_x^2 + v1_y^2)
       - v2 is chosen perpendicular to v3 and v1 (lying in the vertical plane containing the member,
         pointing generally upward):
             v2 = v3 × v1
       *Example*: For a beam along global +X (v1 = [1, 0, 0]):
         v3 = [0, -1, 0] (horizontal, along -Y)
         v2 = [0, 0, 1]  (vertical, along +Z)

    2. **Vertical members** (axis parallel to global Z, sqrt(v1_x^2 + v1_y^2) == 0):
       - If pointing upward along +Z (v1_z > 0):
             v2 = [0, 1, 0]   (along global +Y)
             v3 = [-1, 0, 0]  (along global -X)
       - If pointing downward along -Z (v1_z < 0):
             v2 = [0, -1, 0]  (along global -Y)
             v3 = [-1, 0, 0]  (along global -X)

    User-Specified Orientation:
    ---------------------------
    - **roll_angle** (beta in degrees):
      Rotates the default (v2, v3) triad about the longitudinal axis v1 by angle beta:
          v2_rot =  cos(beta) * v2 + sin(beta) * v3
          v3_rot = -sin(beta) * v2 + cos(beta) * v3
      Positive beta rotates from v2 toward v3 (right-hand screw rule along v1).
    - **web_vector** (v_ref):
      An explicit 3D reference vector lying in the member's local 1-2 plane
      (e.g., pointing along the web of an I-beam or depth of a beam):
          v3 = (v1 × v_ref) / ||v1 × v_ref||
          v2 = v3 × v1

    Coordinate Transformation:
    --------------------------
    The 3x3 direction cosine rotation matrix R maps global vectors to local vectors:
        R = [ v1^T ]
            [ v2^T ]
            [ v3^T ]
        v_local = R @ v_global
        v_global = R.T @ v_local

    The 12x12 global-to-local transformation matrix T is block-diagonal:
        T = diag(R, R, R, R)
        u_local = T @ u_global
        F_global = T.T @ F_local
        K_global = T.T @ K_local @ T

    References:
    - Weaver, W., & Gere, J. M. (1990). Matrix Analysis of Framed Structures (3rd ed.).
      Van Nostrand Reinhold, Section 4.10, pp. 240-244.
    - McGuire, W., Gallagher, R. H., & Ziemian, R. D. (2000). Matrix Structural Analysis
      (2nd ed.). John Wiley & Sons, Section 5.1.
    - Przemieniecki, J. S. (1968). Theory of Matrix Structural Analysis.
      McGraw-Hill, Section 4.5.
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
            Member roll angle in degrees about local 1-axis (v1).
            Rotates the local (2, 3) axes. Defaults to 0.0.
        web_vector : numpy.ndarray, optional
            Reference orientation vector pointing in the local 1-2 plane (e.g. web direction / v2).
            If provided, overrides default orientation. Defaults to None.
        """
        self.id = eid
        self.node_i = node_i
        self.node_j = node_j
        self.roll_angle = float(roll_angle)
        self.web_vector = np.asarray(web_vector, dtype=float) if web_vector is not None else None
        self.structure = None

        self._update_geometry()

    def _update_geometry(self):
        """Compute length, orientation unit vectors v1, v2, v3, and rotation matrix R."""
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

        # Unit vector along member axis (local 1-axis)
        v1 = np.array([dx, dy, dz], dtype=float) / self.length
        self.v1 = v1

        # Determine reference local axes
        if self.web_vector is not None:
            # User supplied custom orientation vector
            v_ref = self.web_vector
            # Project reference vector perpendicular to v1
            v3_temp = np.cross(v1, v_ref)
            norm_3 = np.linalg.norm(v3_temp)
            if norm_3 < 1e-8:
                raise ValueError(
                    f"Element {self.id}: web_vector is parallel to the member axis."
                )
            v3 = v3_temp / norm_3
            v2 = np.cross(v3, v1)
        else:
            # Default orientation: global Z is vertical
            # Check if member is vertical (parallel to global Z)
            proj_xy = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            if proj_xy > 1e-6:
                # Non-vertical member: local 3-axis is horizontal (in XY plane)
                # v3 = (v1 x [0, 0, 1]) / norm
                v3 = np.array([v1[1], -v1[0], 0.0], dtype=float) / proj_xy
                v2 = np.cross(v3, v1)
            else:
                # Vertical member along global Z
                if v1[2] > 0:  # pointing +Z
                    v2 = np.array([0.0, 1.0, 0.0], dtype=float)
                    v3 = np.array([-1.0, 0.0, 0.0], dtype=float)
                else:  # pointing -Z
                    v2 = np.array([0.0, -1.0, 0.0], dtype=float)
                    v3 = np.array([-1.0, 0.0, 0.0], dtype=float)

        # Apply roll angle beta if non-zero (rotate v2 and v3 about v1)
        if abs(self.roll_angle) > 1e-9:
            beta = math.radians(self.roll_angle)
            cos_b = math.cos(beta)
            sin_b = math.sin(beta)
            v2_rot = cos_b * v2 + sin_b * v3
            v3_rot = -sin_b * v2 + cos_b * v3
            v2 = v2_rot
            v3 = v3_rot

        # Ensure unit lengths
        self.v2 = v2 / np.linalg.norm(v2)
        self.v3 = v3 / np.linalg.norm(v3)

        # 3x3 Direction Cosine Rotation Matrix R:
        # local_vector = R @ global_vector
        self.R = np.array([self.v1, self.v2, self.v3], dtype=float)

    def rotation_matrix_3x3(self) -> np.ndarray:
        """Return the 3x3 direction cosine rotation matrix."""
        return self.R.copy()

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Alias for rotation_matrix_3x3()."""
        return self.rotation_matrix_3x3()

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

    def update_state(self, global_disp: np.ndarray):
        """
        Update internal element forces and state from the global displacement vector.

        Parameters
        ----------
        global_disp : numpy.ndarray
            Full global displacement vector of the structure.
        """
        u_i = global_disp[self.node_i.dofs]
        u_j = global_disp[self.node_j.dofs]
        u_global = np.concatenate([u_i, u_j])
        k_global = self.global_stiffness()
        f_global = k_global @ u_global
        if hasattr(self, "eq_load") and self.eq_load is not None:
            f_global = f_global - self.eq_load

        self.k_local = self.local_stiffness() if hasattr(self, "local_stiffness") else None
        self.K_global = k_global
        self.F_global = f_global
        T = self.transformation_matrix(dof_per_node=len(u_i))
        self.f_local = T @ f_global

    def get_tangent_stiffness(self) -> np.ndarray:
        """Return the global tangent stiffness matrix of the element."""
        if hasattr(self, "K_global") and self.K_global is not None:
            return self.K_global
        return self.global_stiffness()

    def get_internal_forces(self) -> np.ndarray:
        """Return the global internal resisting force vector of the element."""
        if hasattr(self, "F_global") and self.F_global is not None:
            return self.F_global
        if (
            hasattr(self, "structure")
            and self.structure is not None
            and getattr(self.structure, "disp", None) is not None
        ):
            self.update_state(self.structure.disp)
            return self.F_global
        num_dofs = (
            len(self.node_i.dofs) + len(self.node_j.dofs)
            if hasattr(self.node_i, "dofs") and self.node_i.dofs
            else 12
        )
        return np.zeros(num_dofs, dtype=float)

