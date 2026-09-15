"""
Structure module defining the 3D finite element model container, assembly, boundary conditions, and solvers.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.linalg import eigh

from .nodes import Node
from .elements.element import ElementBase
from .elements.frame import FrameElement
from .elements.truss import TrussElement
from .elements.spring import SpringElement
from .loads import PointLoad


class Structure:
    """
    Represents a 3D finite element structural model.

    Manages nodes, elements, loads, degrees of freedom (6 per node),
    global matrix assembly, boundary condition partitioning, and linear static / modal analysis.

    Attributes
    ----------
    nodes : dict of {int/str: Node}
        Mapping of node IDs to Node objects.
    elements : dict of {int/str: ElementBase}
        Mapping of element IDs to ElementBase objects.
    loads : list
        List of load objects (PointLoad, ElementLoad, etc.).
    K : numpy.ndarray or None
        Global stiffness matrix (neq x neq).
    F : numpy.ndarray or None
        Global force vector (neq).
    M : numpy.ndarray or None
        Global mass matrix (neq x neq).
    disp : numpy.ndarray or None
        Global displacement vector (neq).
    reactions : numpy.ndarray or None
        Global reaction force vector at fixed DOFs (neq).
    neq : int
        Total number of global degrees of freedom (6 * number_of_nodes).
    fixed_dofs : list of int
        Indices of fixed/constrained degrees of freedom.
    free_dofs : list of int
        Indices of unconstrained degrees of freedom.
    modal_results : dict or None
        Dictionary containing frequencies, periods, and mode shapes from modal analysis.
    """

    def __init__(self):
        """Initialize an empty 3D Structure container."""
        self.nodes: Dict[Union[int, str], Node] = {}
        self.elements: Dict[Union[int, str], ElementBase] = {}
        self.loads: List = []
        self.K: Optional[np.ndarray] = None
        self.F: Optional[np.ndarray] = None
        self.M: Optional[np.ndarray] = None
        self.disp: Optional[np.ndarray] = None
        self.reactions: Optional[np.ndarray] = None
        self.modal_results: Optional[dict] = None
        self.neq: int = 0
        self.fixed_dofs: List[int] = []
        self.free_dofs: List[int] = []

    def add_node(self, node: Node):
        """Add a Node to the structure."""
        self.nodes[node.id] = node

    def add_element(self, element: ElementBase):
        """Add an element to the structure."""
        element.structure = self
        self.elements[element.id] = element

    def add_load(self, load):
        """Add a load (PointLoad, DistributedLoad, etc.) to the structure."""
        self.loads.append(load)

    def number_dofs(self):
        """
        Assign global DOF indices to each node.
        6 degrees of freedom per node: [ux, uy, uz, rx, ry, rz].
        """
        dof = 0
        for nid in sorted(self.nodes, key=lambda x: str(x)):
            self.nodes[nid].dofs = [dof + i for i in range(6)]
            dof += 6
        self.neq = dof

    def assemble_stiffness(self):
        """
        Assemble the global stiffness matrix K (neq x neq) from all elements.
        """
        if self.neq == 0:
            self.number_dofs()

        self.K = np.zeros((self.neq, self.neq), dtype=float)
        for el in self.elements.values():
            k_g = el.global_stiffness()
            dofs = el.node_i.dofs + el.node_j.dofs
            for i, ii in enumerate(dofs):
                for j, jj in enumerate(dofs):
                    self.K[ii, jj] += k_g[i, j]

    def assemble_loads(self):
        """
        Assemble the global force vector F (size neq) from nodal loads, point loads,
        and element equivalent distributed loads.
        """
        if self.neq == 0:
            self.number_dofs()

        self.F = np.zeros(self.neq, dtype=float)

        # 1. Concentrated loads stored on nodes
        for node in self.nodes.values():
            if any(abs(v) > 0.0 for v in node.load):
                self.F[node.dofs] += np.asarray(node.load, dtype=float)

        # 2. Point loads in self.loads
        for load in self.loads:
            if isinstance(load, PointLoad):
                self.F[load.node.dofs] += load.as_array()

        # 3. Equivalent nodal loads from element loads (distributed, etc.)
        for el in self.elements.values():
            if hasattr(el, "eq_load") and el.eq_load is not None:
                dofs = el.node_i.dofs + el.node_j.dofs
                self.F[dofs] += el.eq_load

    def _auto_fix_unstable_dofs(self):
        """
        Automatically constrain inactive degrees of freedom at nodes that
        connected only to truss or spring elements (no frame/beam elements).

        A 3D truss element provides stiffness only along its axial direction.
        Therefore, at a truss-only node:
          - All 3 rotational DOFs [rx, ry, rz] are zero-stiffness -> constrain.
          - Each translational DOF that has zero accumulated diagonal stiffness
            from all connected elements is also zero-stiffness -> constrain.
        This prevents singular K_ff without modifying user-specified supports.
        """
        if self.neq == 0:
            self.number_dofs()

        for node in self.nodes.values():
            connected = [
                el for el in self.elements.values() if node in (el.node_i, el.node_j)
            ]
            if not connected:
                continue

            has_frame = any(
                isinstance(el, FrameElement)
                or "Frame" in type(el).__name__
                or "Beam" in type(el).__name__
                for el in connected
            )
            if has_frame:
                continue

            # Truss-only or spring-only node:
            # Always constrain all 3 rotations.
            node.support[3] = True
            node.support[4] = True
            node.support[5] = True

            # Constrain translational DOFs that have zero diagonal stiffness
            # from all connected elements (i.e. no bar provides stiffness in that direction).
            # Accumulate diagonal stiffness contributions for global translations [ux, uy, uz].
            k_diag_trans = np.zeros(3, dtype=float)
            for el in connected:
                kg = el.global_stiffness()          # 12x12
                node_idx = 0 if el.node_i is node else 6
                for d in range(3):
                    k_diag_trans[d] += kg[node_idx + d, node_idx + d]

            for d in range(3):
                if k_diag_trans[d] < 1e-12:
                    node.support[d] = True

    def apply_boundary_conditions(self):
        """
        Determine free and fixed DOFs based on node support conditions.
        """
        self._auto_fix_unstable_dofs()

        fixed = []
        free = []
        for node in self.nodes.values():
            for i, fixed_flag in enumerate(node.support):
                dof_idx = node.dofs[i]
                if fixed_flag:
                    fixed.append(dof_idx)
                else:
                    free.append(dof_idx)

        self.fixed_dofs = fixed
        self.free_dofs = free

    def solve(self):
        """
        Perform linear static analysis.
        Numbers DOFs, assembles K and F, applies boundary conditions,
        solves for free displacements, and computes support reactions.

        Returns
        -------
        disp : numpy.ndarray
            Global displacement vector (size neq).
        reactions : numpy.ndarray
            Global reaction vector (size neq).
        """
        self.number_dofs()
        self.assemble_stiffness()
        self.assemble_loads()
        self.apply_boundary_conditions()

        # Partition system: K_ff * u_f = F_f
        K_ff = self.K[np.ix_(self.free_dofs, self.free_dofs)]
        F_f = self.F[self.free_dofs]

        self.disp = np.zeros(self.neq, dtype=float)
        if len(self.free_dofs) > 0:
            try:
                self.disp[self.free_dofs] = np.linalg.solve(K_ff, F_f)
            except np.linalg.LinAlgError as err:
                raise np.linalg.LinAlgError(
                    f"Stiffness matrix is singular or ill-conditioned: {err}. "
                    "Check for unconstrained DOFs or rigid body mechanisms."
                ) from err

        # Compute support reactions at fixed DOFs:
        # R_c = K_cf * u_f + K_cc * u_c - F_c
        self.reactions = np.zeros(self.neq, dtype=float)
        if self.fixed_dofs:
            if len(self.free_dofs) > 0:
                self.reactions[self.fixed_dofs] = (
                    self.K[np.ix_(self.fixed_dofs, self.free_dofs)]
                    @ self.disp[self.free_dofs]
                    - self.F[self.fixed_dofs]
                )
            else:
                self.reactions[self.fixed_dofs] = -self.F[self.fixed_dofs]

        return self.disp, self.reactions

    def assemble_mass_matrix(self, lumped: bool = False):
        """
        Assemble the global mass matrix M from element mass matrices and nodal masses.
        """
        if self.neq == 0:
            self.number_dofs()

        self.M = np.zeros((self.neq, self.neq), dtype=float)

        # Element mass contributions
        for el in self.elements.values():
            m_g = el.mass_matrix(lumped=lumped)
            dofs = el.node_i.dofs + el.node_j.dofs
            for i, ii in enumerate(dofs):
                for j, jj in enumerate(dofs):
                    self.M[ii, jj] += m_g[i, j]

        # Nodal lumped masses and inertias
        for node in self.nodes.values():
            for i in range(3):
                if node.mass[i] > 0.0:
                    dof = node.dofs[i]
                    self.M[dof, dof] += node.mass[i]
            for i in range(3):
                if node.inertia[i] > 0.0:
                    dof = node.dofs[3 + i]
                    self.M[dof, dof] += node.inertia[i]

    def get_reduced_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (K_ff, M_ff) reduced to free degrees of freedom.
        """
        if not self.free_dofs:
            self.apply_boundary_conditions()
        if self.K is None:
            self.assemble_stiffness()
        if self.M is None:
            self.assemble_mass_matrix()

        K_ff = self.K[np.ix_(self.free_dofs, self.free_dofs)]
        M_ff = self.M[np.ix_(self.free_dofs, self.free_dofs)]
        return K_ff, M_ff

    def modal_analysis(self, num_modes: int = 6) -> dict:
        """
        Perform 3D undamped eigenvalue modal analysis:
            K_ff * phi = omega^2 * M_ff * phi

        Parameters
        ----------
        num_modes : int, optional
            Number of lowest natural modes to calculate. Defaults to 6.

        Returns
        -------
        dict
            - 'frequencies': cyclic frequencies in Hz (1D array)
            - 'periods': natural periods in seconds (1D array)
            - 'omega': circular frequencies in rad/s (1D array)
            - 'modes': mode shapes expanded to full global DOF space (neq x num_modes)
        """
        if self.disp is None:
            self.solve()

        if self.M is None:
            self.assemble_mass_matrix()

        K_ff, M_ff = self.get_reduced_matrices()
        n_free = len(self.free_dofs)
        modes_to_calc = min(num_modes, n_free)

        # Solve symmetric generalized eigenvalue problem
        eigenvalues, eigenvectors = eigh(K_ff, M_ff)

        # Filter out negligible numerical negatives
        eigenvalues = np.maximum(eigenvalues, 0.0)
        omega = np.sqrt(eigenvalues[:modes_to_calc])
        frequencies = omega / (2.0 * np.pi)
        periods = np.zeros_like(frequencies)
        non_zero = frequencies > 1e-10
        periods[non_zero] = 1.0 / frequencies[non_zero]

        # Expand mode shapes to full DOF space (neq x modes)
        full_modes = np.zeros((self.neq, modes_to_calc), dtype=float)
        for m_idx in range(modes_to_calc):
            full_modes[self.free_dofs, m_idx] = eigenvectors[:, m_idx]
            # Normalize mode shape such that maximum displacement component is 1.0
            max_val = np.max(np.abs(full_modes[:, m_idx]))
            if max_val > 1e-12:
                full_modes[:, m_idx] /= max_val

        self.modal_results = {
            "omega": omega,
            "frequencies": frequencies,
            "periods": periods,
            "modes": full_modes,
        }
        return self.modal_results
