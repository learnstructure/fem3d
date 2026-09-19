"""
Mode shape and buckling mode visualization utilities for 3D structures.
"""

from typing import Optional, Union, Tuple
import numpy as np


def prepare_mode_displacement(
    structure,
    mode_vec: np.ndarray,
    span: float,
    scale: Optional[float] = None,
    auto_scale: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Construct the full global displacement vector from a mode eigenvector
    and calculate an appropriate visual magnification factor.
    """
    # Ensure DOFs are numbered
    if not hasattr(structure, "neq") or structure.neq == 0:
        structure.number_dofs()

    if getattr(structure, "free_dofs", None) is None:
        structure.apply_boundary_conditions()

    mode_vec = np.asarray(mode_vec, dtype=float).flatten()

    # Reconstruct full displacement vector
    if len(mode_vec) == len(structure.free_dofs):
        disp_full = np.zeros(structure.neq, dtype=float)
        disp_full[structure.free_dofs] = mode_vec
    elif len(mode_vec) == structure.neq:
        disp_full = mode_vec.copy()
    else:
        raise ValueError(
            f"Mode vector length ({len(mode_vec)}) does not match free DOFs ({len(structure.free_dofs)}) or total DOFs ({structure.neq})."
        )

    # Calculate effective visual scale
    if scale is not None:
        effective_scale = scale
    elif auto_scale:
        max_trans = 0.0
        for node in structure.nodes.values():
            ux, uy, uz = disp_full[node.dofs[:3]]
            mag = np.hypot(ux, np.hypot(uy, uz))
            if mag > max_trans:
                max_trans = mag

        if max_trans > 1e-12:
            effective_scale = (0.12 * span) / max_trans
        else:
            effective_scale = 1.0
    else:
        effective_scale = 1.0

    return disp_full, effective_scale
