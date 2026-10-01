"""
FrameElement module defining 3D elastic Euler-Bernoulli / Timoshenko frame elements.
"""

from typing import Optional, Union, List
import numpy as np

from .element import ElementBase
from ..materials.elastic import ElasticMaterial
from ..sections.section import Section


class FrameElement(ElementBase):
    """
    Elastic 3D beam-column frame element with 12 degrees of freedom.

    The formulation supports:
    - Axial extension / compression (EA)
    - St. Venant torsion (GJ)
    - Flexure in local 1-2 plane about local 3-axis (E*I3) and shear
    - Flexure in local 1-3 plane about local 2-axis (E*I2) and shear
    - Consistent and lumped mass matrices
    - 3D geometric stiffness matrix (Kg) for linear elastic buckling analysis
    - Optional end releases (hinges)

    Degrees of Freedom in Local Coordinates
    ---------------------------------------
    Local displacement vector (size 12):
        u_local = [
            u_1i, u_2i, u_3i, theta_1i, theta_2i, theta_3i,
            u_1j, u_2j, u_3j, theta_1j, theta_2j, theta_3j
        ]

    Stiffness Matrix Theory & Citations
    -----------------------------------
    References:
    - Przemieniecki, J. S. (1968). Theory of Matrix Structural Analysis.
      McGraw-Hill, Section 5.6, Eq. (5.58), pp. 79-82.
    - Weaver, W., & Gere, J. M. (1990). Matrix Analysis of Framed Structures (3rd ed.).
      Van Nostrand Reinhold, Section 4.11, pp. 244-249.
    - McGuire, W., Gallagher, R. H., & Ziemian, R. D. (2000). Matrix Structural Analysis
      (2nd ed.). John Wiley & Sons, Section 5.1, pp. 115-120.

    Sign Convention & Local Axes (1, 2, 3):
    ---------------------------------------
    - Local axis 1 (v1) is along the member centroidal axis from node_i to node_j.
    - Local axes 2 (v2) and 3 (v3) form a right-handed orthogonal triad with axis 1 (v1 x v2 = v3).
    - Right-hand rule applies to rotations, moments, and roll angle.
    - **I3 block (bending in local 1-2 plane)**:
        Deflection is u_2 (along local 2-axis), rotation is theta_3 (about local 3-axis).
        Resisted by I3 = ∫ x2^2 dA.
        Curvature = d^2(u_2)/dx1^2, slope = d(u_2)/dx1 = +theta_3.
    - **I2 block (bending in local 1-3 plane)**:
        Deflection is u_3 (along local 3-axis), rotation is theta_2 (about local 2-axis).
        Resisted by I2 = ∫ x3^2 dA.
        Curvature = d^2(u_3)/dx1^2, slope = d(u_3)/dx1 = -theta_2.
        (Note the negative sign: by right-hand rule about local 2-axis, a positive rotation
        theta_2 causes displacement in the negative 3-axis direction).
    - **Moments of Inertia**:
        I3 resists in-plane flexure (about local 3-axis), and I2 resists out-of-plane flexure (about local 2-axis).
    """

    def __init__(
        self,
        eid: Union[int, str],
        node_i,
        node_j,
        material: Union[ElasticMaterial, float],
        section: Union[Section, float],
        roll_angle: float = 0.0,
        web_vector: Optional[np.ndarray] = None,
        extra_mass: float = 0.0,
        releases_i: Optional[List[bool]] = None,
        releases_j: Optional[List[bool]] = None,
        include_shear_deformation: bool = False,
    ):
        """
        Initialize a 3D FrameElement.

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
        roll_angle : float, optional
            Member roll angle in degrees about local 1-axis (v1).
            Rotates the local (2, 3) axes. Defaults to 0.0.
        web_vector : numpy.ndarray, optional
            Reference orientation vector pointing in the local 1-2 plane (e.g. web direction / v2).
            If provided, overrides default orientation. Defaults to None.
        extra_mass : float, optional
            Additional distributed non-structural mass per unit length. Defaults to 0.0.
        releases_i : list of bool, optional
            End release fixity at start node [u1, u2, u3, r1, r2, r3].
            True indicates released (zero force/moment).
        releases_j : list of bool, optional
            End release fixity at end node [u1, u2, u3, r1, r2, r3].
        include_shear_deformation : bool, optional
            Whether to include Timoshenko shear deformation if shear areas As2, As3 are present.
        """
        super().__init__(
            eid, node_i, node_j, roll_angle=roll_angle, web_vector=web_vector
        )

        # Parse material
        if isinstance(material, (int, float)):
            self.material = ElasticMaterial(E=float(material))
        else:
            self.material = material

        # Parse section
        if isinstance(section, (int, float)):
            self.section = Section(A=float(section))
        else:
            self.section = section

        self.area = self.section.A
        self.I2 = self.section.I2
        self.I3 = self.section.I3
        self.J = self.section.J
        self.extra_mass = float(extra_mass)
        self.include_shear_deformation = include_shear_deformation

        self.releases_i = releases_i
        self.releases_j = releases_j

        # Local equivalent nodal loads (12-element vector)
        self.eq_load_local = np.zeros(12, dtype=float)
        self.eq_load = np.zeros(12, dtype=float)  # in global coordinates

    def local_stiffness(self) -> np.ndarray:
        """
        Return the 12x12 element stiffness matrix in local coordinates.

        Returns
        -------
        numpy.ndarray
            12x12 local stiffness matrix.
        """
        E = self.material.E
        G = self.material.G
        A = self.area
        I2 = self.I2
        I3 = self.I3
        J = self.J
        L = self.length

        # Timoshenko shear factors: phi_2 (for bending in 1-2 about 3) and phi_3 (for bending in 1-3 about 2)
        phi_2 = 0.0
        phi_3 = 0.0
        if self.include_shear_deformation:
            as2 = getattr(self.section, "As2", 0.0)
            as3 = getattr(self.section, "As3", 0.0)
            if as2 > 0:
                phi_2 = 12.0 * E * I3 / (G * as2 * L * L)
            if as3 > 0:
                phi_3 = 12.0 * E * I2 / (G * as3 * L * L)

        k = np.zeros((12, 12), dtype=float)

        # -------------------------------------------------------------
        # 1. Axial stiffness terms (dofs 0, 6: u_1i, u_1j)
        # Reference: Przemieniecki Eq. (5.58)
        # -------------------------------------------------------------
        EA_L = E * A / L
        k[0, 0] = EA_L
        k[0, 6] = -EA_L
        k[6, 0] = -EA_L
        k[6, 6] = EA_L

        # -------------------------------------------------------------
        # 2. Torsional stiffness terms (dofs 3, 9: theta_1i, theta_1j)
        # Reference: Przemieniecki Eq. (5.58)
        # -------------------------------------------------------------
        if J > 0.0 and G > 0.0:
            GJ_L = G * J / L
            k[3, 3] = GJ_L
            k[3, 9] = -GJ_L
            k[9, 3] = -GJ_L
            k[9, 9] = GJ_L

        # -------------------------------------------------------------
        # 3. Bending in 1-2 plane about local 3-axis (dofs 1, 5, 7, 11: u_2i, theta_3i, u_2j, theta_3j)
        # Resisted by I3.
        # Reference: Przemieniecki Eq. (5.58) & Weaver-Gere Section 4.11
        # -------------------------------------------------------------
        if I3 > 0.0:
            factor_2 = 1.0 + phi_2
            k12_3 = 12.0 * E * I3 / (L**3 * factor_2)
            k6_3 = 6.0 * E * I3 / (L**2 * factor_2)
            k4_3 = (4.0 + phi_2) * E * I3 / (L * factor_2)
            k2_3 = (2.0 - phi_2) * E * I3 / (L * factor_2)

            k[1, 1] = k12_3
            k[1, 5] = k6_3
            k[1, 7] = -k12_3
            k[1, 11] = k6_3

            k[5, 1] = k6_3
            k[5, 5] = k4_3
            k[5, 7] = -k6_3
            k[5, 11] = k2_3

            k[7, 1] = -k12_3
            k[7, 5] = -k6_3
            k[7, 7] = k12_3
            k[7, 11] = -k6_3

            k[11, 1] = k6_3
            k[11, 5] = k2_3
            k[11, 7] = -k6_3
            k[11, 11] = k4_3

        # -------------------------------------------------------------
        # 4. Bending in 1-3 plane about local 2-axis (dofs 2, 4, 8, 10: u_3i, theta_2i, u_3j, theta_2j)
        # Resisted by I2.
        # Sign convention note:
        # Rotation theta_2 is about local 2-axis.
        # By right-hand rule, positive theta_2 gives slope d(u_3)/dx1 = -theta_2.
        # Reference: Przemieniecki Eq. (5.58), Weaver & Gere Section 4.11
        # -------------------------------------------------------------
        if I2 > 0.0:
            factor_3 = 1.0 + phi_3
            k12_2 = 12.0 * E * I2 / (L**3 * factor_3)
            k6_2 = 6.0 * E * I2 / (L**2 * factor_3)
            k4_2 = (4.0 + phi_3) * E * I2 / (L * factor_3)
            k2_2 = (2.0 - phi_3) * E * I2 / (L * factor_3)

            k[2, 2] = k12_2
            k[2, 4] = -k6_2
            k[2, 8] = -k12_2
            k[2, 10] = -k6_2

            k[4, 2] = -k6_2
            k[4, 4] = k4_2
            k[4, 8] = k6_2
            k[4, 10] = k2_2

            k[8, 2] = -k12_2
            k[8, 4] = k6_2
            k[8, 8] = k12_2
            k[8, 10] = k6_2

            k[10, 2] = -k6_2
            k[10, 4] = k2_2
            k[10, 8] = k6_2
            k[10, 11] = 0.0
            k[10, 10] = k4_2

        # Handle optional end releases if specified
        if self.releases_i is not None or self.releases_j is not None:
            k = self._apply_releases(k)

        return k

    def _apply_releases(self, k: np.ndarray) -> np.ndarray:
        """
        Condense out released degrees of freedom via static condensation.
        releases_i: 6 booleans [ux, uy, uz, rx, ry, rz]
        releases_j: 6 booleans [ux, uy, uz, rx, ry, rz]
        """
        rel_flags = [False] * 12
        if self.releases_i:
            for i, r in enumerate(self.releases_i):
                if r:
                    rel_flags[i] = True
        if self.releases_j:
            for i, r in enumerate(self.releases_j):
                if r:
                    rel_flags[6 + i] = True

        released_dofs = [idx for idx, flag in enumerate(rel_flags) if flag]
        retained_dofs = [idx for idx, flag in enumerate(rel_flags) if not flag]

        if not released_dofs:
            return k

        # Static condensation:
        # [ K_rr  K_rc ] [ u_r ] = [ F_r ]
        # [ K_cr  K_cc ] [ u_c ]   [ 0   ]
        # K_condensed = K_rr - K_rc @ inv(K_cc) @ K_cr
        K_rr = k[np.ix_(retained_dofs, retained_dofs)]
        K_rc = k[np.ix_(retained_dofs, released_dofs)]
        K_cr = k[np.ix_(released_dofs, retained_dofs)]
        K_cc = k[np.ix_(released_dofs, released_dofs)]

        K_cc_inv = np.linalg.pinv(K_cc)
        K_r_star = K_rr - K_rc @ K_cc_inv @ K_cr

        k_new = np.zeros((12, 12), dtype=float)
        for i_idx, r_i in enumerate(retained_dofs):
            for j_idx, r_j in enumerate(retained_dofs):
                k_new[r_i, r_j] = K_r_star[i_idx, j_idx]
        return k_new

    def local_mass_matrix(self, lumped: bool = False) -> np.ndarray:
        """
        Return the 12x12 element mass matrix in local coordinates.

        References:
        - Przemieniecki, J. S. (1968). Theory of Matrix Structural Analysis.
          McGraw-Hill, Section 11.2, Eq. (11.23), pp. 294-297.
        - Archer, J. S. (1963). Consistent mass matrix for distributed mass systems.
          ASCE Journal of the Structural Division, 89(ST4), 161-178.
        """
        L = self.length
        total_density = self.material.rho * self.area + self.extra_mass
        m_total = total_density * L

        m = np.zeros((12, 12), dtype=float)

        if lumped or total_density <= 0.0:
            # Lumped mass matrix (diagonal)
            m_half = m_total / 2.0
            # Translation lumped masses
            m[0, 0] = m_half
            m[1, 1] = m_half
            m[2, 2] = m_half
            m[6, 6] = m_half
            m[7, 7] = m_half
            m[8, 8] = m_half

            # Rotational lumped inertias
            I_polar = self.I2 + self.I3
            m[3, 3] = total_density * I_polar * L / 24.0
            m[4, 4] = total_density * self.I2 * L / 24.0
            m[5, 5] = total_density * self.I3 * L / 24.0
            m[9, 9] = m[3, 3]
            m[10, 10] = m[4, 4]
            m[11, 11] = m[5, 5]
            return m

        # Consistent mass matrix (Archer / Przemieniecki Eq. 11.23)
        # 1. Axial mass
        m_axial = (m_total / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
        m[0, 0] = m_axial[0, 0]
        m[0, 6] = m_axial[0, 1]
        m[6, 0] = m_axial[1, 0]
        m[6, 6] = m_axial[1, 1]

        # 2. Torsional mass
        I_p = self.I2 + self.I3
        m_torsion = (total_density * I_p * L / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
        m[3, 3] = m_torsion[0, 0]
        m[3, 9] = m_torsion[0, 1]
        m[9, 3] = m_torsion[1, 0]
        m[9, 9] = m_torsion[1, 1]

        # 3. Bending mass in 1-2 plane (I3)
        coef_z = m_total / 420.0
        m[1, 1] = coef_z * 156.0
        m[1, 5] = coef_z * 22.0 * L
        m[1, 7] = coef_z * 54.0
        m[1, 11] = coef_z * -13.0 * L

        m[5, 1] = m[1, 5]
        m[5, 5] = coef_z * 4.0 * L**2
        m[5, 7] = coef_z * 13.0 * L
        m[5, 11] = coef_z * -3.0 * L**2

        m[7, 1] = m[1, 7]
        m[7, 5] = m[5, 7]
        m[7, 7] = coef_z * 156.0
        m[7, 11] = coef_z * -22.0 * L

        m[11, 1] = m[1, 11]
        m[11, 5] = m[5, 11]
        m[11, 7] = m[7, 11]
        m[11, 11] = coef_z * 4.0 * L**2

        # 4. Bending mass in 1-3 plane (I2)
        coef_y = m_total / 420.0
        m[2, 2] = coef_y * 156.0
        m[2, 4] = coef_y * -22.0 * L
        m[2, 8] = coef_y * 54.0
        m[2, 10] = coef_y * 13.0 * L

        m[4, 2] = m[2, 4]
        m[4, 4] = coef_y * 4.0 * L**2
        m[4, 8] = coef_y * -13.0 * L
        m[4, 10] = coef_y * -3.0 * L**2

        m[8, 2] = m[2, 8]
        m[8, 4] = m[4, 8]
        m[8, 8] = coef_y * 156.0
        m[8, 10] = coef_y * 22.0 * L

        m[10, 2] = m[2, 10]
        m[10, 4] = m[4, 10]
        m[10, 8] = m[8, 10]
        m[10, 10] = coef_y * 4.0 * L**2

        return m

    def geometric_stiffness(self, P: float) -> np.ndarray:
        """
        Return the 12x12 geometric stiffness matrix Kg in global coordinates.

        Axial load convention: P > 0 is compressive.

        References:
        - Przemieniecki, J. S. (1968). Theory of Matrix Structural Analysis.
          McGraw-Hill, Section 12.3, pp. 388-390.
        - McGuire, W., Gallagher, R. H., & Ziemian, R. D. (2000). Matrix Structural Analysis.
          Section 9.3, pp. 248-252.

        Parameters
        ----------
        P : float
            Axial compressive force (positive in compression).

        Returns
        -------
        numpy.ndarray
            12x12 geometric stiffness matrix in global coordinates.
        """
        L = self.length
        kg_local = np.zeros((12, 12), dtype=float)

        if abs(P) < 1e-12:
            return kg_local

        # Transverse bending terms in 1-2 plane (dofs 1, 5, 7, 11)
        factor = P / (30.0 * L)
        kg_local[1, 1] = 36.0 * factor
        kg_local[1, 5] = 3.0 * L * factor
        kg_local[1, 7] = -36.0 * factor
        kg_local[1, 11] = 3.0 * L * factor

        kg_local[5, 1] = 3.0 * L * factor
        kg_local[5, 5] = 4.0 * L**2 * factor
        kg_local[5, 7] = -3.0 * L * factor
        kg_local[5, 11] = -1.0 * L**2 * factor

        kg_local[7, 1] = -36.0 * factor
        kg_local[7, 5] = -3.0 * L * factor
        kg_local[7, 7] = 36.0 * factor
        kg_local[7, 11] = -3.0 * L * factor

        kg_local[11, 1] = 3.0 * L * factor
        kg_local[11, 5] = -1.0 * L**2 * factor
        kg_local[11, 7] = -3.0 * L * factor
        kg_local[11, 11] = 4.0 * L**2 * factor

        # Transverse bending terms in 1-3 plane (dofs 2, 4, 8, 10)
        kg_local[2, 2] = 36.0 * factor
        kg_local[2, 4] = -3.0 * L * factor
        kg_local[2, 8] = -36.0 * factor
        kg_local[2, 10] = -3.0 * L * factor

        kg_local[4, 2] = -3.0 * L * factor
        kg_local[4, 4] = 4.0 * L**2 * factor
        kg_local[4, 8] = 3.0 * L * factor
        kg_local[4, 10] = -1.0 * L**2 * factor

        kg_local[8, 2] = -36.0 * factor
        kg_local[8, 4] = 3.0 * L * factor
        kg_local[8, 8] = 36.0 * factor
        kg_local[8, 10] = 3.0 * L * factor

        kg_local[10, 2] = -3.0 * L * factor
        kg_local[10, 4] = -1.0 * L**2 * factor
        kg_local[10, 8] = 3.0 * L * factor
        kg_local[10, 10] = 4.0 * L**2 * factor

        # Torsional geometric stiffness (St. Venant approximation: P * (I2+I3)/A)
        if self.area > 0.0:
            I_p = self.I2 + self.I3
            r_g2 = I_p / self.area
            kg_local[3, 3] = (P * r_g2 / L)
            kg_local[3, 9] = -(P * r_g2 / L)
            kg_local[9, 3] = -(P * r_g2 / L)
            kg_local[9, 9] = (P * r_g2 / L)

        T = self.transformation_matrix()
        return T.T @ kg_local @ T

    def get_local_forces(self) -> np.ndarray:
        """
        Compute element end forces in local coordinates:
            f_local = K_local @ u_local - eq_load_local

        Returns
        -------
        numpy.ndarray
            Vector of 12 internal forces:
            [f1_i, f2_i, f3_i, m1_i, m2_i, m3_i, f1_j, f2_j, f3_j, m1_j, m2_j, m3_j]
        """
        if self.structure is None or self.structure.disp is None:
            raise ValueError("Structure has not been solved yet.")
        u_local = self.get_local_displacements(self.structure.disp)
        k_local = self.local_stiffness()
        f_local = k_local @ u_local - self.eq_load_local
        return f_local

    def axial_force(self) -> float:
        """
        Return the average axial force (compression positive).
        """
        f = self.get_local_forces()
        # f1_i is tension if negative; f1_j is tension if positive
        # Under compression, f1_i > 0 and f1_j < 0
        return float(f[0])

    def __repr__(self) -> str:
        return f"FrameElement(id={self.id}, nodes=({self.node_i.id}, {self.node_j.id}), L={self.length:.3f})"


# Convenient alias for 2D/ecosystem familiarity
BeamElement = FrameElement
