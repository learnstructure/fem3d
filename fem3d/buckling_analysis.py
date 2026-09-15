"""
Buckling analysis module for 3D finite element frame and truss structures.

Solves the generalized eigenvalue problem:
    K_ff * phi = lambda * Kg_ff * phi
where Kg is the 3D geometric stiffness matrix assembled with compression positive.
"""

from typing import Tuple
import numpy as np
from scipy.linalg import eig

from .elements.frame import FrameElement
from .elements.truss import TrussElement


def assemble_geometric_stiffness(structure) -> np.ndarray:
    """
    Assemble the global 3D geometric stiffness matrix Kg (neq x neq).

    Axial force is taken positive in compression.
    """
    Kg = np.zeros((structure.neq, structure.neq), dtype=float)
    for el in structure.elements.values():
        if not isinstance(el, (FrameElement, TrussElement)):
            continue

        # Axial compressive force:
        P = el.axial_force()
        if abs(P) < 1e-12:
            continue

        Kg_el = el.geometric_stiffness(P)
        dofs = el.node_i.dofs + el.node_j.dofs
        for i, ii in enumerate(dofs):
            for j, jj in enumerate(dofs):
                Kg[ii, jj] += Kg_el[i, j]

    return Kg


def buckling_analysis(structure, num_modes: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform linear elastic 3D frame/truss buckling analysis.

    Solves:
        K_ff * phi = lambda * Kg_ff * phi

    Parameters
    ----------
    structure : Structure
        The solved structure under reference design load case.
    num_modes : int, optional
        Number of positive buckling modes to return. Defaults to 1.

    Returns
    -------
    buckling_factors : numpy.ndarray
        Array of positive critical load multipliers lambda (sorted ascending).
    buckling_modes : numpy.ndarray
        Array (neq x num_modes) containing 3D mode shapes in full global DOF space.
    """
    if structure.disp is None:
        structure.solve()

    if structure.free_dofs is None:
        structure.apply_boundary_conditions()

    free = structure.free_dofs
    K_ff = structure.K[np.ix_(free, free)]

    Kg = assemble_geometric_stiffness(structure)
    Kg_ff = Kg[np.ix_(free, free)]

    # Solve generalized eigenvalue problem: K_ff * phi = lambda * Kg_ff * phi
    # Alternatively: inv(K_ff) * Kg_ff * phi = (1/lambda) * phi
    vals, vecs = eig(Kg_ff, K_ff)

    # Note: eig returns (alpha, v) such that Kg_ff * v = alpha * K_ff * v
    # where alpha = 1 / lambda
    real_mask = np.isreal(vals) & (np.real(vals) > 1e-12)
    pos_alphas = np.real(vals[real_mask])
    pos_vecs = np.real(vecs[:, real_mask])

    if len(pos_alphas) == 0:
        raise ValueError(
            "No positive buckling modes found. Structure may be entirely in tension "
            "or reference loads do not induce compression."
        )

    # lambda = 1 / alpha
    lambdas = 1.0 / pos_alphas
    sort_indices = np.argsort(lambdas)
    lambdas_sorted = lambdas[sort_indices]
    vecs_sorted = pos_vecs[:, sort_indices]

    modes_to_return = min(num_modes, len(lambdas_sorted))
    buckling_factors = lambdas_sorted[:modes_to_return]

    full_modes = np.zeros((structure.neq, modes_to_return), dtype=float)
    for m in range(modes_to_return):
        full_modes[free, m] = vecs_sorted[:, m]
        max_v = np.max(np.abs(full_modes[:, m]))
        if max_v > 1e-12:
            full_modes[:, m] /= max_v

    return buckling_factors, full_modes
