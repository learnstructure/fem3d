"""
Rigid floor diaphragm constraint definitions and sparse transformation builder.

Formulation follows Wilson, *Three-Dimensional Static and Dynamic Analysis of Structures*, Ch. 7 §7.5
and the CSI Analysis Reference Manual, Ch. 5 §§5.3–5.4, §5.12.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp


DOF_NAMES = ("ux", "uy", "uz", "rx", "ry", "rz")

# Right-handed triad (1, 2, 3) with e1 x e2 = e3:
#   axis 3 is normal to the diaphragm plane;
#   (u1_i, u2_i, r3_i) are constrained by u1_i = u1_m - (x2_i - x2_m) * r3_m,
#   u2_i = u2_m + (x1_i - x1_m) * r3_m, r3_i = r3_m;
#   out-of-plane DOFs remain independent at each node.
DIAPHRAGM_AXES: Dict[str, Dict[str, Any]] = {
    "Z": {
        "in_plane_coords": ("x", "y"),
        "normal_coord": "z",
        "dof_indices": (0, 1, 5),
        "trans_dofs": (0, 1),
        "rot_dof": 5,
    },
    "X": {
        "in_plane_coords": ("y", "z"),
        "normal_coord": "x",
        "dof_indices": (1, 2, 3),
        "trans_dofs": (1, 2),
        "rot_dof": 3,
    },
    "Y": {
        "in_plane_coords": ("z", "x"),
        "normal_coord": "y",
        "dof_indices": (2, 0, 4),
        "trans_dofs": (2, 0),
        "rot_dof": 4,
    },
}


@dataclass
class RigidDiaphragm:
    """Rigid planar diaphragm coupling in-plane translations and normal-axis rotation."""

    id: str
    node_ids: Tuple[Union[int, str], ...]
    axis: str = "Z"
    name: Optional[str] = None

    def __post_init__(self) -> None:
        self.id = str(self.id).strip()
        if not self.id:
            raise ValueError("RigidDiaphragm id must be non-empty.")

        self.axis = str(self.axis).strip().upper()
        if self.axis not in DIAPHRAGM_AXES:
            raise ValueError(f"Unsupported diaphragm axis {self.axis!r}; expected 'Z', 'X', or 'Y'.")

        nids = tuple(self.node_ids)
        if len(nids) < 2:
            raise ValueError(f"RigidDiaphragm {self.id!r} requires at least 2 distinct nodes, got {len(nids)}.")
        seen = set()
        dups: List[Union[int, str]] = []
        for nid in nids:
            if nid in seen and nid not in dups:
                dups.append(nid)
            seen.add(nid)
        if dups:
            raise ValueError(f"RigidDiaphragm {self.id!r} contains duplicate node ID(s): {dups}.")
        self.node_ids = nids


def validate_diaphragm_geometry(diaphragm: RigidDiaphragm, nodes: Dict[Union[int, str], Any]) -> None:
    """Validate node existence and coplanarity along the diaphragm normal axis (CSI §5.4.1)."""
    missing = [nid for nid in diaphragm.node_ids if nid not in nodes]
    if missing:
        raise ValueError(f"Diaphragm {diaphragm.id!r} references missing node(s): {missing}.")

    spec = DIAPHRAGM_AXES[diaphragm.axis]
    a_attr, b_attr = spec["in_plane_coords"]
    n_attr = spec["normal_coord"]

    a_vals = [float(getattr(nodes[nid], a_attr)) for nid in diaphragm.node_ids]
    b_vals = [float(getattr(nodes[nid], b_attr)) for nid in diaphragm.node_ids]
    normals = [float(getattr(nodes[nid], n_attr)) for nid in diaphragm.node_ids]

    span = max(max(a_vals) - min(a_vals), max(b_vals) - min(b_vals))
    tol = max(1e-6, 1e-6 * span)
    delta = max(normals) - min(normals)
    if delta > tol:
        raise ValueError(
            f"Diaphragm {diaphragm.id!r} (axis={diaphragm.axis!r}) nodes are not coplanar: "
            f"{diaphragm.axis} coordinates vary by {delta:.6g} (tolerance {tol:.6g}); "
            f"CSI §5.4.1 requires coplanar nodes."
        )


def build_constraint_matrix(
    nodes: Dict[Union[int, str], Any],
    free_dofs: Iterable[int],
    diaphragms: Iterable[RigidDiaphragm],
    mass_matrix: Optional[sp.spmatrix] = None,
) -> Tuple[sp.csc_matrix, List[int], Dict[str, Dict[str, Any]]]:
    """
    Build the exact sparse kinematic constraint transformation matrix T (neq x n_reduced)
    mapping reduced generalized coordinates q to full nodal displacements u = T q.

    Each diaphragm contributes three master DOFs [u1_m, u2_m, L_d * r3_m], where the master
    point is placed at the diaphragm translational center of mass (or geometric centroid when
    massless) and master rotation is scaled by the diaphragm RMS radius L_d so K_r and M_r
    stay well-conditioned across all length units.
    """
    neq = 6 * len(nodes)
    diaph_list = list(diaphragms)

    seen_nodes: Dict[Union[int, str], str] = {}
    constrained_global_dofs = set()

    for d in diaph_list:
        validate_diaphragm_geometry(d, nodes)
        spec = DIAPHRAGM_AXES[d.axis]
        local_dofs = spec["dof_indices"]

        for nid in d.node_ids:
            if nid in seen_nodes:
                raise ValueError(
                    f"Node {nid!r} is assigned to multiple diaphragms ({seen_nodes[nid]!r} and {d.id!r}); "
                    f"a node may belong to at most one diaphragm."
                )
            seen_nodes[nid] = d.id

            node = nodes[nid]
            fixed = [DOF_NAMES[idx] for idx in local_dofs if bool(node.support[idx])]
            if fixed:
                raise ValueError(
                    f"Diaphragm {d.id!r} (axis={d.axis!r}) node {nid!r} has fixed in-plane support DOF(s) "
                    f"{fixed}; fixed in-plane restraints on rigid diaphragm nodes are not supported "
                    f"(out-of-plane fixed supports and elastic springs are allowed)."
                )
            for idx in local_dofs:
                constrained_global_dofs.add(node.dofs[idx])

    independent_dofs = [int(dof) for dof in free_dofs if dof not in constrained_global_dofs]
    n_indep = len(independent_dofs)
    n_reduced = n_indep + 3 * len(diaph_list)

    rows: List[int] = list(independent_dofs)
    cols: List[int] = list(range(n_indep))
    vals: List[float] = [1.0] * n_indep

    mass_diag: Optional[np.ndarray] = None
    if mass_matrix is not None:
        mass_diag = np.asarray(mass_matrix.diagonal(), dtype=float)

    masters: Dict[str, Dict[str, Any]] = {}
    for d_idx, d in enumerate(diaph_list):
        spec = DIAPHRAGM_AXES[d.axis]
        a_attr, b_attr = spec["in_plane_coords"]
        n_attr = spec["normal_coord"]
        d1_loc, d2_loc = spec["trans_dofs"]
        d3_loc = spec["rot_dof"]

        diaph_nodes = [nodes[nid] for nid in d.node_ids]
        x1 = np.array([float(getattr(nd, a_attr)) for nd in diaph_nodes], dtype=float)
        x2 = np.array([float(getattr(nd, b_attr)) for nd in diaph_nodes], dtype=float)
        x3_m = float(np.mean([float(getattr(nd, n_attr)) for nd in diaph_nodes]))

        if mass_diag is not None:
            m1 = np.array([max(0.0, float(mass_diag[nd.dofs[d1_loc]])) for nd in diaph_nodes], dtype=float)
            m2 = np.array([max(0.0, float(mass_diag[nd.dofs[d2_loc]])) for nd in diaph_nodes], dtype=float)
            # Moment-of-inertia coupling terms:
            #   sum_i m1_i (x2_i - x2_m) = 0  ->  x2_m = sum(m1_i x2_i) / sum(m1_i)
            #   sum_i m2_i (x1_i - x1_m) = 0  ->  x1_m = sum(m2_i x1_i) / sum(m2_i)
            sum_m1 = float(np.sum(m1))
            sum_m2 = float(np.sum(m2))
            x1_m = float(np.dot(m2, x1) / sum_m2) if sum_m2 > 0.0 else float(np.mean(x1))
            x2_m = float(np.dot(m1, x2) / sum_m1) if sum_m1 > 0.0 else float(np.mean(x2))
        else:
            x1_m = float(np.mean(x1))
            x2_m = float(np.mean(x2))

        dx1 = x1 - x1_m
        dx2 = x2 - x2_m
        rms_radius = math.sqrt(float(np.mean(dx1**2 + dx2**2)))
        scale_radius = rms_radius if rms_radius > 1e-12 else 1.0

        col_u1 = n_indep + 3 * d_idx
        col_u2 = col_u1 + 1
        col_r3 = col_u1 + 2

        for i, nd in enumerate(diaph_nodes):
            g1 = int(nd.dofs[d1_loc])
            g2 = int(nd.dofs[d2_loc])
            gr = int(nd.dofs[d3_loc])

            rows.extend((g1, g1, g2, g2, gr))
            cols.extend((col_u1, col_r3, col_u2, col_r3, col_r3))
            vals.extend((1.0, -float(dx2[i]) / scale_radius, 1.0, float(dx1[i]) / scale_radius, 1.0 / scale_radius))

        if d.axis == "Z":
            master_coords = (x1_m, x2_m, x3_m)
        elif d.axis == "X":
            master_coords = (x3_m, x1_m, x2_m)
        else:  # "Y"
            master_coords = (x2_m, x3_m, x1_m)

        masters[d.id] = {
            "axis": d.axis,
            "master_coords": master_coords,
            "scale_radius": scale_radius,
            "master_dofs": (col_u1, col_u2, col_r3),
            "master_trans_dofs": {d1_loc: col_u1, d2_loc: col_u2},
            "node_ids": d.node_ids,
        }

    T = sp.csc_matrix((vals, (rows, cols)), shape=(neq, n_reduced), dtype=float)
    return T, independent_dofs, masters
