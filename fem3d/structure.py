"""
Structure module defining the 3D finite element model container, assembly, boundary conditions, and solvers.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .diaphragm import (
    DIAPHRAGM_AXES,
    RigidDiaphragm,
    build_constraint_matrix,
    validate_diaphragm_geometry,
)
from .nodes import Node
from .elements.element import ElementBase
from .elements.frame import FrameElement
from .elements.truss import TrussElement
from .elements.spring import SpringElement
from .elements.shell import ShellBatch, ShellElement
from .loads import PointLoad, ShellLoad


class Structure:
    """
    Represents a 3D finite element structural model.

    Manages nodes, elements, loads, degrees of freedom (6 per node),
    global matrix assembly, boundary condition partitioning, and linear static / modal analysis.

    The global stiffness and mass matrices are sparse (``scipy.sparse`` CSR): element blocks are assembled in
    coordinate (COO) form, with duplicate entries summed, and converted once to CSR. The static system is solved by a
    sparse LU factorization (SuperLU).

    Attributes
    ----------
    nodes : dict of {int/str: Node}
        Mapping of node IDs to Node objects.
    elements : dict of {int/str: ElementBase}
        Mapping of element IDs to two-node elements (frames, trusses, springs).
    shells : dict of {int/str: ShellElement}
        Mapping of element IDs to four-node shell elements, kept apart from ``elements`` so that code written for
        two-node elements never meets them. IDs are unique across both.
    loads : list
        List of load objects (PointLoad, ElementLoad, etc.).
    K : scipy.sparse.csr_matrix or None
        Global stiffness matrix (neq x neq).
    F : numpy.ndarray or None
        Global force vector (neq).
    M : scipy.sparse.csr_matrix or None
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

    def __init__(self, units: Optional[Union[Any, str]] = None):
        """Initialize an empty 3D Structure container."""
        self.units = units
        self.nodes: Dict[Union[int, str], Node] = {}
        self.elements: Dict[Union[int, str], ElementBase] = {}
        self.shells: Dict[Union[int, str], ShellElement] = {}
        self._shell_batch: Optional[ShellBatch] = None
        self.diaphragms: Dict[str, RigidDiaphragm] = {}
        self._constraint_T: Optional[sp.csc_matrix] = None
        self._constraint_has_mass: bool = False
        self._independent_dofs: List[int] = []
        self.diaphragm_masters: Dict[str, dict] = {}
        self.loads: List = []
        self.K: Optional[sp.csr_matrix] = None
        self.F: Optional[np.ndarray] = None
        self.M: Optional[sp.csr_matrix] = None
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
        """Add a two-node element (frame, truss, spring) to the structure."""
        if element.id in self.shells:
            raise ValueError(f"Element ID {element.id!r} is already used by a shell element")
        element.structure = self
        self.elements[element.id] = element

    def add_shell(self, shell: ShellElement):
        """Add a shell element to the structure."""
        if not isinstance(shell, ShellElement):
            raise TypeError(f"add_shell expects a ShellElement, got {type(shell).__name__}")
        if shell.id in self.elements or shell.id in self.shells:
            raise ValueError(f"Shell ID {shell.id!r} is already used by another element")
        shell.structure = self
        self.shells[shell.id] = shell
        self._shell_batch = None

    def shell_batch(self) -> Optional[ShellBatch]:
        """The shell elements computed together (local axes, transformations, stiffness, results), or None."""
        if not self.shells:
            return None
        if self.neq == 0:
            self.number_dofs()
        if self._shell_batch is None:
            self._shell_batch = ShellBatch(list(self.shells.values()))
        return self._shell_batch

    def add_load(self, load):
        """Add a load (PointLoad, DistributedLoad, etc.) to the structure."""
        self.loads.append(load)

    def add_diaphragm(
        self,
        diaphragm_id: Union[str, RigidDiaphragm],
        node_ids: Optional[Iterable[Union[int, str]]] = None,
        axis: str = "Z",
        name: Optional[str] = None,
    ) -> RigidDiaphragm:
        """Add a rigid planar diaphragm constraining ``node_ids`` perpendicular to ``axis``."""
        if isinstance(diaphragm_id, RigidDiaphragm):
            d = diaphragm_id
        else:
            if node_ids is None:
                raise ValueError("node_ids is required when adding a diaphragm.")
            d = RigidDiaphragm(id=diaphragm_id, node_ids=tuple(node_ids), axis=axis, name=name)

        if d.id in self.diaphragms:
            raise ValueError(f"Duplicate diaphragm ID {d.id!r}.")
        validate_diaphragm_geometry(d, self.nodes)

        used = {nid: existing.id for existing in self.diaphragms.values() for nid in existing.node_ids}
        for nid in d.node_ids:
            if nid in used:
                raise ValueError(
                    f"Node {nid!r} is already assigned to diaphragm {used[nid]!r}; "
                    f"a node may belong to at most one diaphragm."
                )

        self.diaphragms[d.id] = d
        self._constraint_T = None
        self._constraint_has_mass = False
        self._independent_dofs = []
        self.diaphragm_masters = {}
        return d

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
        self._shell_batch = None  # shell DOF indices follow the numbering
        self._constraint_T = None
        self._constraint_has_mass = False
        self._independent_dofs = []
        self.diaphragm_masters = {}

    def _assemble_sparse(
        self,
        blocks: Iterable[Tuple[np.ndarray, np.ndarray]],
        diagonal: Iterable[Tuple[int, float]] = (),
    ) -> sp.csr_matrix:
        """
        Assemble a global (neq x neq) matrix from element blocks and diagonal terms.

        Parameters
        ----------
        blocks : iterable of (dofs, matrix)
            ``dofs`` of shape (n,) and ``matrix`` of shape (n, n), or a group of elements of one type at once:
            ``dofs`` of shape (N, n) and ``matrix`` of shape (N, n, n).
        diagonal : iterable of (dof, value)
            Terms added on the diagonal (support springs, nodal masses).
        """
        rows, cols, vals = [], [], []
        for dofs, mat in blocks:
            d = np.asarray(dofs, dtype=np.int64)
            m = np.asarray(mat, dtype=float)
            if d.ndim == 1:
                d, m = d[None], m[None]
            n = d.shape[1]
            rows.append(np.repeat(d, n, axis=1).ravel())
            cols.append(np.tile(d, (1, n)).ravel())
            vals.append(m.reshape(-1))
        diag = list(diagonal)
        if diag:
            dd = np.array([i for i, _ in diag], dtype=np.int64)
            rows.append(dd)
            cols.append(dd)
            vals.append(np.array([v for _, v in diag], dtype=float))
        if not rows:
            return sp.csr_matrix((self.neq, self.neq), dtype=float)
        coo = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.neq, self.neq)
        )
        return coo.tocsr()  # duplicate entries are summed

    def _element_dofs(self, el) -> List[int]:
        return el.node_i.dofs + el.node_j.dofs

    def assemble_stiffness(self):
        """
        Assemble the global stiffness matrix K (neq x neq, sparse CSR) from all elements and support springs.
        """
        if self.neq == 0:
            self.number_dofs()

        blocks = [(self._element_dofs(el), el.global_stiffness()) for el in self.elements.values()]
        batch = self.shell_batch()
        if batch is not None:
            blocks.append((batch.dofs, batch.global_stiffness()))
        springs = [
            (node.dofs[i], float(k_val))
            for node in self.nodes.values()
            if hasattr(node, "springs") and any(node.springs)
            for i, k_val in enumerate(node.springs)
            if k_val > 0.0
        ]
        self.K = self._assemble_sparse(blocks, springs)

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

        # 4. Uniform loads on shell elements: consistent nodal forces
        shell_loads = [load for load in self.loads if isinstance(load, ShellLoad)]
        if shell_loads:
            batch = self.shell_batch()
            index = {sid: i for i, sid in enumerate(batch.ids)} if batch is not None else {}
            for load in shell_loads:
                if load.shell.id not in index or self.shells[load.shell.id] is not load.shell:
                    raise ValueError(f"ShellLoad on shell {load.shell.id!r}, which is not in this structure")
                i = index[load.shell.id]
                self.F[batch.dofs[i]] += batch.uniform_load(i, load.pressure, load.direction)

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

        # Nodes of shell elements are never fixed automatically: a shell gives them rotational stiffness (or, for a
        # membrane or plate alone, leaves DOFs for the user to restrain, as SAP2000 does)
        shell_nodes = {id(n) for sh in self.shells.values() for n in sh.nodes}
        diaph_constrained_dofs: Dict[int, set] = {}
        for d in self.diaphragms.values():
            local_dofs = set(DIAPHRAGM_AXES[d.axis]["dof_indices"])
            for nid in d.node_ids:
                if nid in self.nodes:
                    diaph_constrained_dofs.setdefault(id(self.nodes[nid]), set()).update(local_dofs)

        # Elements connected to each node, in element order (one pass over the elements)
        connected_to: Dict[int, List[ElementBase]] = {}
        for el in self.elements.values():
            connected_to.setdefault(id(el.node_i), []).append(el)
            if el.node_j is not el.node_i:
                connected_to.setdefault(id(el.node_j), []).append(el)

        for node in self.nodes.values():
            connected = connected_to.get(id(node), [])
            if not connected or id(node) in shell_nodes:
                continue

            has_frame = any(
                isinstance(el, FrameElement)
                or "Frame" in type(el).__name__
                or "Beam" in type(el).__name__
                for el in connected
            )
            if has_frame:
                continue

            constrained = diaph_constrained_dofs.get(id(node), set())

            # Truss-only or spring-only node:
            # Constrain rotations that are not coupled into a rigid diaphragm.
            for r_idx in (3, 4, 5):
                if r_idx not in constrained:
                    node.support[r_idx] = True

            # Constrain translational DOFs that have zero diagonal stiffness
            # from all connected elements (i.e. no bar provides stiffness in that direction),
            # excluding DOFs constrained by a rigid diaphragm.
            k_diag_trans = np.zeros(3, dtype=float)
            for el in connected:
                kg = el.global_stiffness()          # 12x12
                node_idx = 0 if el.node_i is node else 6
                for d in range(3):
                    k_diag_trans[d] += kg[node_idx + d, node_idx + d]

            for d in range(3):
                if d not in constrained and k_diag_trans[d] < 1e-12:
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
                k_val = node.springs[i] if hasattr(node, "springs") else 0.0
                if fixed_flag and k_val > 0.0:
                    dof_name = ["ux", "uy", "uz", "rx", "ry", "rz"][i]
                    raise ValueError(
                        f"Node {node.id} DOF {dof_name} cannot be both restrained (fixed) and have a spring stiffness ({k_val})."
                    )
                if fixed_flag:
                    fixed.append(dof_idx)
                else:
                    free.append(dof_idx)

        self.fixed_dofs = fixed
        self.free_dofs = free
        self._constraint_T = None
        self._constraint_has_mass = False
        self._independent_dofs = []
        self.diaphragm_masters = {}
        if self.diaphragms:
            self.constraint_matrix(use_mass=False)

    def constraint_matrix(self, use_mass: bool = False) -> sp.csc_matrix:
        """
        Return the sparse kinematic transformation matrix T (neq x n_reduced) mapping
        reduced coordinates q to full nodal displacements u = T q.
        """
        if self.neq == 0 and self.nodes:
            self.number_dofs()
        if not self.free_dofs and not self.fixed_dofs and self.nodes:
            self.apply_boundary_conditions()
        if not self.diaphragms:
            n_free = len(self.free_dofs)
            return sp.csc_matrix(
                (np.ones(n_free, dtype=float), (np.asarray(self.free_dofs, dtype=np.int64), np.arange(n_free, dtype=np.int64))),
                shape=(self.neq, n_free),
                dtype=float,
            )
        want_mass = bool(use_mass and self.M is not None)
        if self._constraint_T is None or (want_mass and not self._constraint_has_mass):
            T, indep, masters = build_constraint_matrix(
                self.nodes,
                self.free_dofs,
                self.diaphragms.values(),
                mass_matrix=self.M if want_mass else None,
            )
            self._constraint_T = T
            self._constraint_has_mass = want_mass
            self._independent_dofs = indep
            self.diaphragm_masters = masters
        return self._constraint_T

    @property
    def n_reduced_dofs(self) -> int:
        """Number of independent + master DOFs in the reduced system."""
        if not self.free_dofs and not self.fixed_dofs and self.nodes:
            self.apply_boundary_conditions()
        if not self.diaphragms:
            return len(self.free_dofs)
        return int(self.constraint_matrix(use_mass=self._constraint_has_mass).shape[1])

    def reduce_matrix(self, mat):
        """Transform a full (neq x neq) matrix into the reduced DOF space (T^T A T)."""
        if not self.free_dofs and not self.fixed_dofs and self.nodes:
            self.apply_boundary_conditions()
        if not self.diaphragms:
            free = np.asarray(self.free_dofs, dtype=np.int64)
            return mat[free][:, free] if sp.issparse(mat) else np.asarray(mat, dtype=float)[np.ix_(free, free)]
        T = self.constraint_matrix(use_mass=self._constraint_has_mass)
        if sp.issparse(mat):
            return (T.T @ mat @ T).tocsr()
        arr = np.asarray(mat, dtype=float)
        return np.asarray(T.T @ (T.T @ arr.T).T, dtype=float)

    def reduce_vector(self, vec: np.ndarray) -> np.ndarray:
        """Transform a full (neq,) generalized force vector into the reduced DOF space (T^T v)."""
        if not self.free_dofs and not self.fixed_dofs and self.nodes:
            self.apply_boundary_conditions()
        arr = np.asarray(vec, dtype=float)
        if not self.diaphragms:
            return arr[self.free_dofs]
        T = self.constraint_matrix(use_mass=self._constraint_has_mass)
        return np.asarray(T.T @ arr, dtype=float)

    def expand_displacements(self, u_reduced: np.ndarray) -> np.ndarray:
        """Recover full nodal displacements (neq,) or (neq, k) from reduced coordinates (u = T q)."""
        if not self.free_dofs and not self.fixed_dofs and self.nodes:
            self.apply_boundary_conditions()
        arr = np.asarray(u_reduced, dtype=float)
        if arr.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D reduced displacement array, got shape {arr.shape}.")
        expected = self.n_reduced_dofs
        if arr.shape[0] != expected:
            raise ValueError(
                f"Reduced displacement length ({arr.shape[0]}) does not match n_reduced_dofs ({expected})."
            )
        if not self.diaphragms:
            shape = (self.neq,) if arr.ndim == 1 else (self.neq, arr.shape[1])
            full = np.zeros(shape, dtype=float)
            if expected > 0:
                if arr.ndim == 1:
                    full[self.free_dofs] = arr
                else:
                    full[self.free_dofs, :] = arr
            return full
        T = self.constraint_matrix(use_mass=self._constraint_has_mass)
        return np.asarray(T @ arr, dtype=float)

    def influence_vector(self, direction: str = "x") -> np.ndarray:
        """Return the rigid-body ground-motion influence vector in the reduced DOF space."""
        dir_key = str(direction).strip().lower()
        target_map = {"x": 0, "0": 0, "y": 1, "1": 1, "z": 2, "2": 2}
        if dir_key not in target_map:
            raise ValueError(f"Unknown direction {direction!r}; expected 'x', 'y', or 'z'.")
        target = target_map[dir_key]
        if not self.free_dofs and not self.fixed_dofs and self.nodes:
            self.apply_boundary_conditions()
        if not self.diaphragms:
            vec = np.zeros(len(self.free_dofs), dtype=float)
            for i, dof in enumerate(self.free_dofs):
                if dof % 6 == target:
                    vec[i] = 1.0
            return vec
        T = self.constraint_matrix(use_mass=self._constraint_has_mass)
        vec = np.zeros(T.shape[1], dtype=float)
        for i, dof in enumerate(self._independent_dofs):
            if dof % 6 == target:
                vec[i] = 1.0
        for meta in self.diaphragm_masters.values():
            trans_map = meta["master_trans_dofs"]
            if target in trans_map:
                vec[trans_map[target]] = 1.0
        return vec

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

        free = np.asarray(self.free_dofs, dtype=np.int64)
        fixed = np.asarray(self.fixed_dofs, dtype=np.int64)
        self.disp = np.zeros(self.neq, dtype=float)

        if not self.diaphragms:
            # Partition system: K_ff * u_f = F_f, solved by sparse LU (SuperLU)
            F_f = self.F[free]
            if len(free) > 0:
                K_ff = self.K[free][:, free].tocsc()
                try:
                    lu = splu(K_ff)
                except RuntimeError as err:  # SuperLU: "Factor is exactly singular"
                    raise np.linalg.LinAlgError(
                        f"Stiffness matrix is singular or ill-conditioned: {err}. "
                        "Check for unconstrained DOFs or rigid body mechanisms."
                    ) from err
                u_f = lu.solve(F_f)
                if not np.all(np.isfinite(u_f)):
                    raise np.linalg.LinAlgError(
                        "Stiffness matrix is singular or ill-conditioned: the solution is not finite. "
                        "Check for unconstrained DOFs or rigid body mechanisms."
                    )
                self.disp[free] = u_f
        else:
            T = self.constraint_matrix(use_mass=False)
            if T.shape[1] > 0:
                K_r = (T.T @ self.K @ T).tocsc()
                F_r = np.asarray(T.T @ self.F, dtype=float)
                try:
                    lu = splu(K_r)
                except RuntimeError as err:
                    raise np.linalg.LinAlgError(
                        f"Stiffness matrix is singular or ill-conditioned: {err}. "
                        "Check for unconstrained DOFs or rigid body mechanisms."
                    ) from err
                q = lu.solve(F_r)
                if not np.all(np.isfinite(q)):
                    raise np.linalg.LinAlgError(
                        "Stiffness matrix is singular or ill-conditioned: the solution is not finite. "
                        "Check for unconstrained DOFs or rigid body mechanisms."
                    )
                self.disp = np.asarray(T @ q, dtype=float)

        # Compute support reactions at fixed DOFs:
        # R_c = K_cf * u_f + K_cc * u_c - F_c
        self.reactions = np.zeros(self.neq, dtype=float)
        if len(fixed):
            if len(free) > 0:
                self.reactions[fixed] = (
                    self.K[fixed][:, free] @ self.disp[free]
                    - self.F[fixed]
                )
            else:
                self.reactions[self.fixed_dofs] = -self.F[self.fixed_dofs]

        # Add spring reactions (-k * u) at spring-supported DOFs
        for node in self.nodes.values():
            if hasattr(node, "springs") and any(node.springs):
                for i, k_val in enumerate(node.springs):
                    if k_val > 0.0:
                        dof = node.dofs[i]
                        self.reactions[dof] -= float(k_val) * self.disp[dof]

        return self.disp, self.reactions

    def assemble_mass_matrix(self, lumped: bool = False, include_rotational_inertia: bool = False):
        """
        Assemble the global mass matrix M from element mass matrices and nodal masses.

        Parameters
        ----------
        lumped : bool, optional
            If True, uses lumped element mass matrices. Defaults to False.
        include_rotational_inertia : bool, optional
            If True and lumped=True, includes rotational lumped mass for frame elements.
            If False (default), rotational mass is zero, matching standard building frame
            software (SAP2000, ETABS) where rotational DOFs are Guyan condensed.
        """
        if self.neq == 0:
            self.number_dofs()

        self._mass_lumped = lumped
        self._mass_rotational = include_rotational_inertia

        # Element mass contributions (shells: translations only, CSI §10.11)
        blocks = [
            (self._element_dofs(el), el.mass_matrix(lumped=lumped, include_rotational_inertia=include_rotational_inertia))
            for el in self.elements.values()
        ]
        batch = self.shell_batch()
        if batch is not None:
            blocks.append((batch.dofs, batch.mass(lumped=lumped)))
        # Nodal lumped masses and inertias
        nodal = []
        for node in self.nodes.values():
            for i in range(3):
                if node.mass[i] > 0.0:
                    nodal.append((node.dofs[i], float(node.mass[i])))
            for i in range(3):
                if node.inertia[i] > 0.0:
                    nodal.append((node.dofs[3 + i], float(node.inertia[i])))
        self.M = self._assemble_sparse(blocks, nodal)
        self._constraint_T = None
        self._constraint_has_mass = False
        self._independent_dofs = []
        self.diaphragm_masters = {}

    def get_reduced_matrices(
        self,
        lumped: Optional[bool] = None,
        include_rotational_inertia: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (K_r, M_r) reduced to active/master degrees of freedom, as sparse CSR matrices.
        """
        if not self.free_dofs and not self.fixed_dofs and self.nodes:
            self.apply_boundary_conditions()
        if self.K is None:
            self.assemble_stiffness()
        if lumped is not None:
            if (
                self.M is None
                or getattr(self, "_mass_lumped", None) != lumped
                or getattr(self, "_mass_rotational", None) != include_rotational_inertia
            ):
                self.assemble_mass_matrix(lumped=lumped, include_rotational_inertia=include_rotational_inertia)
        elif self.M is None:
            self.assemble_mass_matrix(lumped=False, include_rotational_inertia=include_rotational_inertia)

        if not self.diaphragms:
            free = np.asarray(self.free_dofs, dtype=np.int64)
            K_ff = self.K[free][:, free].tocsr()
            M_ff = self.M[free][:, free].tocsr()
            return K_ff, M_ff

        T = self.constraint_matrix(use_mass=True)
        K_r = (T.T @ self.K @ T).tocsr()
        M_r = (T.T @ self.M @ T).tocsr()
        return K_r, M_r

    def modal_analysis(
        self,
        num_modes: int = 6,
        lumped: bool = False,
        include_rotational_inertia: bool = False,
    ) -> dict:
        """
        Perform 3D undamped eigenvalue modal analysis:
            K_ff * phi = omega^2 * M_ff * phi
        delegating matrix regularization and eigensolution to structdyn.ModalAnalysis.

        Parameters
        ----------
        num_modes : int, optional
            Number of lowest natural modes to calculate. Defaults to 6.
        lumped : bool, optional
            Whether to use lumped mass formulation. Defaults to False.
        include_rotational_inertia : bool, optional
            Whether to include frame rotational lumped inertia when lumped=True. Defaults to False.

        Returns
        -------
        dict
            - 'frequencies': cyclic frequencies in Hz (1D array)
            - 'periods': natural periods in seconds (1D array)
            - 'omega': circular frequencies in rad/s (1D array)
            - 'modes': mode shapes expanded to full global DOF space (neq x num_modes), normalized to peak 1.0
            - 'phi': reduced mode shape matrix on free DOFs (n_free x num_modes)
        """
        from structdyn.mdf.mdf import MDF

        if self.disp is None:
            self.solve()

        n_red = self.n_reduced_dofs
        modes_to_calc = min(num_modes, n_red)
        if modes_to_calc <= 0:
            self.modal_results = {
                "omega": np.array([], dtype=float),
                "frequencies": np.array([], dtype=float),
                "periods": np.array([], dtype=float),
                "modes": np.zeros((self.neq, 0), dtype=float),
                "phi": np.zeros((n_red, 0), dtype=float),
            }
            return self.modal_results

        mdf = MDF.from_fem3d(
            self,
            lumped=lumped,
            include_rotational_inertia=include_rotational_inertia,
        )
        omega, phi = mdf.modal.modal_analysis(n_modes=modes_to_calc)

        frequencies = omega / (2.0 * np.pi)
        periods = np.zeros_like(frequencies)
        non_zero = frequencies > 1e-10
        periods[non_zero] = 1.0 / frequencies[non_zero]

        # Expand mode shapes to full DOF space (neq x modes)
        full_modes = self.expand_displacements(phi)
        for m_idx in range(len(omega)):
            max_val = np.max(np.abs(full_modes[:, m_idx]))
            if max_val > 1e-12:
                full_modes[:, m_idx] /= max_val

        self.modal_results = {
            "omega": omega,
            "frequencies": frequencies,
            "periods": periods,
            "modes": full_modes,
            "phi": phi,
        }
        return self.modal_results

    def set_modal_results(self, omega, phi):
        """
        Register modal analysis results on the structure for later use by plotting utilities.

        Parameters
        ----------
        omega : numpy.ndarray
            Array of natural circular frequencies (rad/s), one per mode.
        phi : numpy.ndarray
            Mode shape matrix of shape (n_reduced_dofs, n_modes) or (neq, n_modes).
        """
        omega_arr = np.asarray(omega, dtype=float)
        phi_arr = np.asarray(phi, dtype=float)
        periods = np.zeros_like(omega_arr)
        non_zero = omega_arr > 1e-10
        periods[non_zero] = 2.0 * np.pi / omega_arr[non_zero]

        if not self.free_dofs and not self.fixed_dofs and self.nodes:
            self.apply_boundary_conditions()
        n_red = self.n_reduced_dofs

        if phi_arr.ndim == 2:
            if phi_arr.shape[0] == n_red:
                full_modes = self.expand_displacements(phi_arr)
            elif phi_arr.shape[0] == self.neq:
                full_modes = phi_arr.copy()
            else:
                full_modes = phi_arr
        elif phi_arr.ndim == 1:
            if len(phi_arr) == n_red:
                full_modes = self.expand_displacements(phi_arr).reshape(-1, 1)
            else:
                full_modes = phi_arr.reshape(-1, 1)
        else:
            full_modes = phi_arr

        self.modal_results = {
            "omega": omega_arr,
            "periods": periods,
            "frequencies": omega_arr / (2.0 * np.pi),
            "phi": phi_arr,
            "modes": full_modes,
        }
        return self.modal_results

    def shell_resultants(self) -> Dict[Union[int, str], Dict[str, np.ndarray]]:
        """Stress resultants of the shell elements for the current displacements (``self.disp``).

        Returns ``{shell_id: {"gauss_points": (4, 8), "corners": (4, 8)}}``, the eight resultants F11, F22, F12,
        M11, M22, M12, V13, V23 per unit length in the element's local 1-2-3 axes (CSI Analysis Reference Manual
        §10.18), at the 2x2 Gauss points and extrapolated to the corners in node order.
        """
        batch = self.shell_batch()
        if batch is None:
            return {}
        if self.disp is None:
            raise RuntimeError("Solve the structure before asking for shell resultants")
        gauss, corners = batch.resultants(self.disp)
        return {sid: {"gauss_points": gauss[i], "corners": corners[i]} for i, sid in enumerate(batch.ids)}

    def commit_state(self):
        """Commit the state of all elements after a converged step."""
        for el in self.elements.values():
            if hasattr(el, "commit"):
                el.commit()

    def response_spectrum_analysis(
        self,
        params=None,
        spectrum_curve=None,
        gm=None,
        gm_component=None,
        Sa=None,
        Sd=None,
        n_modes=None,
        ji=None,
        direction="x",
        scale_factor=1.0,
        combination="CQC",
        lumped: bool = True,
        include_rotational_inertia: bool = False,
    ):
        """
        Perform linear dynamic Response Spectrum Analysis (RSA) on the 3D structure.

        Parameters
        ----------
        params : SeismicParameters, optional
            SeismicParameters object from struct_load providing design response spectrum.
        spectrum_curve : callable or tuple or dict, optional
            Continuous response spectrum curve Sa(T).
        gm : GroundMotion or str, optional
            Ground motion object, built-in event name (e.g. 'elcentro', 'northridge',
            'loma_prieta', 'san_fernando', 'elcentro_chopra'), or composite string (e.g. 'northridge:hor2').
        gm_component : str, optional
            Specific component when gm is a string ('hor1', 'hor2', 'up').
            If None, explicitly matched to excitation direction:
            - direction='x' -> 'hor1' (horizontal 1 / X)
            - direction='y' -> 'hor2' (horizontal 2 / Y)
            - direction='z' -> 'up' (vertical / Z)
        Sa : array-like, optional
            Array of modal spectral accelerations.
        Sd : array-like, optional
            Array of modal spectral displacements.
        n_modes : int, optional
            Number of modes to include. Defaults to all available free DOFs.
        ji : float or array-like, optional
            Modal damping ratio(s). If None, defaults to params.damping_ratio or 0.05 (5%).
        direction : str, optional
            Excitation direction ('x', 'y', or 'z'). Defaults to 'x'.
        scale_factor : float, optional
            Scale factor for spectral values (e.g. 9.80665 or 386.09 if Sa is in g).
        combination : str, optional
            Modal combination method: 'CQC', 'SRSS', or 'ABS'. Defaults to 'CQC'.
        lumped : bool, optional
            Whether to use lumped mass formulation. Defaults to True (commercial standard for RSA).
        include_rotational_inertia : bool, optional
            Whether to include rotational mass inertia when lumped=True. Defaults to False (matching SAP2000/ETABS).

        Returns
        -------
        dict
            Dictionary containing rsa object, modal and combined displacements,
            element forces, base shear, and mass participation metrics.
        """
        from structdyn.mdf.mdf import MDF
        from structdyn.mdf.analytical_methods.response_spectrum_analysis import (
            ResponseSpectrumAnalysis,
        )

        if self.neq == 0:
            self.number_dofs()
        self.apply_boundary_conditions()

        if ji is None:
            ji = getattr(params, "damping_ratio", 0.05) if params is not None else 0.05

        if spectrum_curve is None and params is not None and getattr(params, "sa_spectrum", None) is not None:
            spectrum_curve = params.sa_spectrum

        mdf = MDF.from_fem3d(
            self,
            zeta=ji,
            n_modes=n_modes,
            lumped=lumped,
            include_rotational_inertia=include_rotational_inertia,
        )
        rsa = ResponseSpectrumAnalysis(
            mdf=mdf,
            ji=ji,
            n_modes=n_modes,
            gm=gm,
            gm_component=gm_component,
            Sd=Sd,
            Sa=Sa,
            spectrum_curve=spectrum_curve,
            params=params,
            direction=direction,
            scale_factor=scale_factor,
        )

        modal_u = rsa.modal_displacements()
        combined_u = rsa.combine_modal_responses(modal_u, method=combination)

        # Expand each mode to original nodal DOFs before CQC/SRSS/ABS combination
        full_modal_u = self.expand_displacements(modal_u)
        full_combined_u = rsa.combine_modal_responses(full_modal_u, method=combination)
        self.disp = full_combined_u

        elem_forces = rsa.compute_modal_element_forces(method=combination)

        # Shell resultants of every mode, combined component by component with the same rule (peak values, not
        # concurrent; in each element's local axes)
        combined_shells: Dict[Union[int, str], Dict[str, np.ndarray]] = {}
        batch = self.shell_batch()
        if batch is not None:
            per_mode = [batch.resultants(full_modal_u[:, m]) for m in range(full_modal_u.shape[1])]
            parts = {}
            for key, k in (("gauss_points", 0), ("corners", 1)):
                stacked = np.stack([pm[k] for pm in per_mode], axis=-1)  # (N, 4, 8, modes)
                parts[key] = np.asarray(
                    rsa.combine_modal_responses(stacked.reshape(-1, stacked.shape[-1]), method=combination)
                ).reshape(stacked.shape[:-1])
            combined_shells = {
                sid: {"gauss_points": parts["gauss_points"][i], "corners": parts["corners"][i]}
                for i, sid in enumerate(batch.ids)
            }
        modal_vb = rsa.modal_base_shear()
        combined_vb = rsa.combine_modal_responses(modal_vb, method=combination)

        self.set_modal_results(rsa.omega, rsa.phi)

        self.rsa_results = {
            "rsa": rsa,
            "omega": rsa.omega,
            "periods": rsa.T,
            "phi": rsa.phi,
            "participation_factors": rsa.lambda_,
            "effective_mass": rsa.effective_mass,
            "mass_participation_ratio": rsa.mass_participation_ratio,
            "modal_displacements": modal_u,
            "combined_displacements": combined_u,
            "full_modal_displacements": full_modal_u,
            "full_combined_displacements": full_combined_u,
            "modal_element_forces": elem_forces["modal"],
            "combined_element_forces": elem_forces["combined"],
            "combined_shell_resultants": combined_shells,
            "modal_base_shear": modal_vb,
            "combined_base_shear": combined_vb,
            "combination_method": combination,
        }
        return self.rsa_results

    def apply_self_weight(
        self,
        direction: str = "-z",
        gravity_accel: Optional[float] = None,
        units: Optional[Any] = None,
    ) -> "Structure":
        """
        Compute and apply element self-weight distributed loads directly to the structure.

        Parameters
        ----------
        direction : str, optional
            Gravity direction in global coordinates ('-z', '-y', '-x'). Defaults to '-z'.
        gravity_accel : float, optional
            Acceleration due to gravity. If None, derived from structure.units.
        units : str or Units, optional
            Units for calculation. If None, uses structure.units.

        Returns
        -------
        Structure
            self (for chaining).
        """
        from struct_load import apply_self_weight as _apply_sw
        return _apply_sw(self, direction=direction, gravity_accel=gravity_accel, units=units)



