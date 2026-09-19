"""
3D Element drawing utilities:
- Undeformed frame, truss, and spring elements
- 3D Cubic Hermite spline deformed curves
- Axial force color mapping (tension blue, compression red, neutral gray)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
except ImportError:  # pragma: no cover
    cm = mcolors = None

from ..elements.frame import FrameElement
from ..elements.truss import TrussElement
from ..elements.spring import SpringElement


def hermite_cubic_3d(el: FrameElement, u_i_glob: np.ndarray, u_j_glob: np.ndarray, scale: float, n_points: int = 40) -> np.ndarray:
    """
    Compute 3D deformed coordinates along a FrameElement using cubic Hermite polynomials.
    """
    if hasattr(el, "rotation_matrix_3x3"):
        R = el.rotation_matrix_3x3()
    elif hasattr(el, "R"):
        R = el.R
    else:
        R = el.transformation_matrix()[:3, :3]
    L = el.length

    # Transform global displacements to local coordinates:
    # u_i_glob: [ux, uy, uz, rx, ry, rz]
    d_i_trans_loc = R @ u_i_glob[:3]
    d_i_rot_loc = R @ u_i_glob[3:6]
    d_j_trans_loc = R @ u_j_glob[:3]
    d_j_rot_loc = R @ u_j_glob[3:6]

    xi = np.linspace(0.0, 1.0, n_points)

    # Hermite shape functions
    N1 = 1.0 - 3.0 * xi**2 + 2.0 * xi**3
    N2 = L * (xi - 2.0 * xi**2 + xi**3)
    N3 = 3.0 * xi**2 - 2.0 * xi**3
    N4 = L * (-xi**2 + xi**3)

    # Local axial displacement (linear)
    u_loc = (1.0 - xi) * d_i_trans_loc[0] + xi * d_j_trans_loc[0]

    # Local transverse deflections
    # y' deflection bending with rotation theta_z'
    v_loc = N1 * d_i_trans_loc[1] + N2 * d_i_rot_loc[2] + N3 * d_j_trans_loc[1] + N4 * d_j_rot_loc[2]
    # z' deflection bending with rotation theta_y' (right-hand rule slope: dw/dx = -theta_y)
    w_loc = N1 * d_i_trans_loc[2] - N2 * d_i_rot_loc[1] + N3 * d_j_trans_loc[2] - N4 * d_j_rot_loc[1]

    # Combine local deflections: shape (3, n_points)
    disp_loc = np.vstack([u_loc, v_loc, w_loc]) * scale

    # Rotate deflections back to global: disp_glob = R.T @ disp_loc
    disp_glob = R.T @ disp_loc

    # Base chord coordinates in global space
    p_i = np.array([el.node_i.x, el.node_i.y, el.node_i.z], dtype=float)
    p_j = np.array([el.node_j.x, el.node_j.y, el.node_j.z], dtype=float)
    chord_glob = np.outer(p_i, (1.0 - xi)) + np.outer(p_j, xi)

    # Deformed coordinates: shape (n_points, 3)
    coords_def = (chord_glob + disp_glob).T
    return coords_def


def get_element_axial_force(el) -> float:
    """
    Extract or compute internal axial force (tension > 0, compression < 0).
    """
    if hasattr(el, "axial_force"):
        af = el.axial_force() if callable(el.axial_force) else el.axial_force
        return float(af)

    # If force vector is already computed on element:
    if hasattr(el, "force") and el.force is not None:
        if isinstance(el, TrussElement):
            # In TrussElement, local fx_j is tension if positive
            return float(el.force[6]) if len(el.force) > 6 else float(el.force[-1])
        elif isinstance(el, FrameElement):
            # In FrameElement, local fx_j is tension
            return float(el.force[6])

    # Fallback: compute from internal forces
    try:
        f = el.internal_forces()
        if len(f) >= 7:
            return float(f[6])
        return float(f[0])
    except Exception:
        return 0.0


def draw_3d_elements(
    ax,
    structure,
    show_undeformed: bool = True,
    show_deformed: bool = True,
    color_by_force: bool = False,
    scale: float = 1.0,
    n_points: int = 40,
) -> Optional[Tuple[cm.ScalarMappable, float, float]]:
    """
    Draw 3D elements in undeformed and/or deformed configurations.
    Returns a ScalarMappable if color_by_force is active.
    """
    mappable = None

    # Setup force colormap if requested
    if color_by_force and cm is not None:
        forces = {}
        for eid, el in structure.elements.items():
            forces[eid] = get_element_axial_force(el)

        max_f = max(abs(min(forces.values(), default=0.0)), abs(max(forces.values(), default=0.0)))
        if max_f < 1e-6:
            max_f = 1.0

        norm = mcolors.Normalize(vmin=-max_f, vmax=max_f)
        try:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("coolwarm")
        except Exception:
            import matplotlib
            cmap = matplotlib.colormaps["coolwarm"]
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

    # 1. Undeformed Geometry
    if show_undeformed:
        first_frame = True
        first_truss = True
        first_spring = True

        for el in structure.elements.values():
            p_i = [el.node_i.x, el.node_i.y, el.node_i.z]
            p_j = [el.node_j.x, el.node_j.y, el.node_j.z]

            if isinstance(el, TrussElement):
                ax.plot(
                    [p_i[0], p_j[0]], [p_i[1], p_j[1]], [p_i[2], p_j[2]],
                    color="#16A085",
                    linestyle="--",
                    linewidth=1.8,
                    label="Undeformed Truss" if first_truss else "",
                    zorder=3,
                )
                first_truss = False
            elif isinstance(el, SpringElement):
                # 3D helical/zigzag representation
                n_coils = 8
                t = np.linspace(0, 1, n_coils * 4)
                base = np.outer(p_i, 1 - t) + np.outer(p_j, t)
                ax.plot(
                    base[0], base[1], base[2],
                    color="#E67E22",
                    linestyle=":",
                    linewidth=2.0,
                    label="Spring" if first_spring else "",
                    zorder=3,
                )
                first_spring = False
            else:
                # FrameElement
                ax.plot(
                    [p_i[0], p_j[0]], [p_i[1], p_j[1]], [p_i[2], p_j[2]],
                    color="#2C3E50",
                    linestyle="-",
                    linewidth=2.0,
                    label="Undeformed Frame" if first_frame else "",
                    zorder=3,
                )
                first_frame = False

    # 2. Deformed Geometry
    if show_deformed and getattr(structure, "disp", None) is not None:
        first_def = True
        disp = structure.disp

        for eid, el in structure.elements.items():
            u_i = disp[el.node_i.dofs]
            u_j = disp[el.node_j.dofs]

            # Choose line color
            if color_by_force and mappable is not None:
                el_f = get_element_axial_force(el)
                elem_color = mappable.to_rgba(el_f)
            else:
                elem_color = "#2980B9"

            if isinstance(el, FrameElement):
                coords = hermite_cubic_3d(el, u_i, u_j, scale=scale, n_points=n_points)
                ax.plot(
                    coords[:, 0], coords[:, 1], coords[:, 2],
                    color=elem_color,
                    linewidth=2.5,
                    label="Deformed" if first_def and not color_by_force else "",
                    zorder=4,
                )
                first_def = False
            else:
                # Truss or Spring: linear displaced chord
                p_i_def = np.array([el.node_i.x, el.node_i.y, el.node_i.z]) + scale * u_i[:3]
                p_j_def = np.array([el.node_j.x, el.node_j.y, el.node_j.z]) + scale * u_j[:3]
                ax.plot(
                    [p_i_def[0], p_j_def[0]], [p_i_def[1], p_j_def[1]], [p_i_def[2], p_j_def[2]],
                    color=elem_color,
                    linewidth=2.5,
                    label="Deformed" if first_def and not color_by_force else "",
                    zorder=4,
                )
                first_def = False

    return mappable
