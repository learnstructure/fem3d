"""
Structure module defining the 3D finite element model container, assembly, boundary conditions, and solvers.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

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

    The global stiffness and mass matrices are sparse (``scipy.sparse`` CSR): element blocks are assembled in
    coordinate (COO) form, with duplicate entries summed, and converted once to CSR. The static system is solved by a
    sparse LU factorization (SuperLU).

    Attributes
    ----------
    nodes : dict of {int/str: Node}
        Mapping of node IDs to Node objects.
    elements : dict of {int/str: ElementBase}
        Mapping of element IDs to ElementBase objects.
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

        blocks = ((self._element_dofs(el), el.global_stiffness()) for el in self.elements.values())
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

        # Elements connected to each node, in element order (one pass over the elements)
        connected_to: Dict[int, List[ElementBase]] = {}
        for el in self.elements.values():
            connected_to.setdefault(id(el.node_i), []).append(el)
            if el.node_j is not el.node_i:
                connected_to.setdefault(id(el.node_j), []).append(el)

        for node in self.nodes.values():
            connected = connected_to.get(id(node), [])
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

        # Partition system: K_ff * u_f = F_f, solved by sparse LU (SuperLU)
        free = np.asarray(self.free_dofs, dtype=np.int64)
        fixed = np.asarray(self.fixed_dofs, dtype=np.int64)
        F_f = self.F[free]

        self.disp = np.zeros(self.neq, dtype=float)
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

        # Element mass contributions
        blocks = (
            (self._element_dofs(el), el.mass_matrix(lumped=lumped, include_rotational_inertia=include_rotational_inertia))
            for el in self.elements.values()
        )
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

    def get_reduced_matrices(
        self,
        lumped: Optional[bool] = None,
        include_rotational_inertia: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (K_ff, M_ff) reduced to the free degrees of freedom, as sparse CSR matrices.
        """
        if not self.free_dofs:
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

        free = np.asarray(self.free_dofs, dtype=np.int64)
        K_ff = self.K[free][:, free].tocsr()
        M_ff = self.M[free][:, free].tocsr()
        return K_ff, M_ff

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

        n_free = len(self.free_dofs)
        modes_to_calc = min(num_modes, n_free)
        if modes_to_calc <= 0:
            self.modal_results = {
                "omega": np.array([], dtype=float),
                "frequencies": np.array([], dtype=float),
                "periods": np.array([], dtype=float),
                "modes": np.zeros((self.neq, 0), dtype=float),
                "phi": np.zeros((n_free, 0), dtype=float),
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
        full_modes = np.zeros((self.neq, len(omega)), dtype=float)
        for m_idx in range(len(omega)):
            full_modes[self.free_dofs, m_idx] = phi[:, m_idx]
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
            Mode shape matrix of shape (n_free_dofs, n_modes) or (neq, n_modes).
        """
        omega_arr = np.asarray(omega, dtype=float)
        phi_arr = np.asarray(phi, dtype=float)
        periods = np.zeros_like(omega_arr)
        non_zero = omega_arr > 1e-10
        periods[non_zero] = 2.0 * np.pi / omega_arr[non_zero]

        if not self.free_dofs:
            self.apply_boundary_conditions()

        if phi_arr.ndim == 2:
            num_modes = phi_arr.shape[1]
            if phi_arr.shape[0] == len(self.free_dofs):
                full_modes = np.zeros((self.neq, num_modes), dtype=float)
                full_modes[self.free_dofs, :] = phi_arr
            elif phi_arr.shape[0] == self.neq:
                full_modes = phi_arr.copy()
            else:
                full_modes = phi_arr
        elif phi_arr.ndim == 1:
            if len(phi_arr) == len(self.free_dofs):
                full_modes = np.zeros((self.neq, 1), dtype=float)
                full_modes[self.free_dofs, 0] = phi_arr
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

        d_comb = np.zeros(self.neq, dtype=float)
        d_comb[self.free_dofs] = combined_u
        self.disp = d_comb

        elem_forces = rsa.compute_modal_element_forces(method=combination)
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
            "modal_element_forces": elem_forces["modal"],
            "combined_element_forces": elem_forces["combined"],
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



