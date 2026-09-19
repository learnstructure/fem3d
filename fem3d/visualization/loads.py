"""
3D Applied loads visualization:
- Point forces (3D arrows with magnitude labels)
- Point moments (double-headed 3D arrows with magnitude labels)
- Distributed member loads (vector arrays with profile boundary line)
"""

from typing import List, Optional
import numpy as np


def draw_3d_point_load(ax, pos: np.ndarray, force_vec: np.ndarray, arrow_len: float, color="#E74C3C", label: Optional[str] = None):
    """
    Draw a 3D point force vector arrow pointing towards or from the node.
    """
    f_norm = np.linalg.norm(force_vec)
    if f_norm < 1e-9:
        return

    u_vec = force_vec / f_norm
    # The arrow points INTO the node from an offset start position:
    start_pos = pos - u_vec * arrow_len

    ax.quiver(
        start_pos[0], start_pos[1], start_pos[2],
        u_vec[0] * arrow_len, u_vec[1] * arrow_len, u_vec[2] * arrow_len,
        color=color,
        arrow_length_ratio=0.25,
        linewidth=2.0,
        zorder=7,
    )

    if label:
        ax.text(
            start_pos[0], start_pos[1], start_pos[2],
            label,
            color=color,
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.8, linewidth=0.8),
            zorder=8,
        )


def draw_3d_moment(ax, pos: np.ndarray, moment_vec: np.ndarray, arrow_len: float, color="#8E44AD", label: Optional[str] = None):
    """
    Draw a 3D moment as a DOUBLE-HEADED 3D arrow along the moment vector axis.
    """
    m_norm = np.linalg.norm(moment_vec)
    if m_norm < 1e-9:
        return

    u_vec = moment_vec / m_norm

    # Full arrow
    ax.quiver(
        pos[0], pos[1], pos[2],
        u_vec[0] * arrow_len, u_vec[1] * arrow_len, u_vec[2] * arrow_len,
        color=color,
        arrow_length_ratio=0.25,
        linewidth=2.4,
        zorder=7,
    )

    # Second arrowhead along the shaft (giving the characteristic double-headed moment symbol)
    retracted_len = arrow_len * 0.72
    ax.quiver(
        pos[0], pos[1], pos[2],
        u_vec[0] * retracted_len, u_vec[1] * retracted_len, u_vec[2] * retracted_len,
        color=color,
        arrow_length_ratio=0.32,
        linewidth=2.4,
        zorder=7,
    )

    if label:
        tip_pos = pos + u_vec * (arrow_len * 1.05)
        ax.text(
            tip_pos[0], tip_pos[1], tip_pos[2],
            label,
            color=color,
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.8, linewidth=0.8),
            zorder=8,
        )


def draw_3d_loads(ax, structure, span: float, arrow_scale: float = 0.12):
    """
    Draw all applied nodal point loads, point moments, and member distributed loads.
    """
    base_arrow_len = max(span * arrow_scale, 0.1)

    # 1. Nodal Concentrated Loads & Moments
    for node in structure.nodes.values():
        if not any(abs(v) > 1e-9 for v in node.load):
            continue

        fx, fy, fz, mx, my, mz = node.load
        pos = np.array([node.x, node.y, node.z], dtype=float)

        # Force vector
        f_vec = np.array([fx, fy, fz], dtype=float)
        f_mag = np.linalg.norm(f_vec)
        if f_mag > 1e-9:
            lbl = f"{f_mag:.2f}"
            draw_3d_point_load(ax, pos, f_vec, base_arrow_len, color="#E74C3C", label=lbl)

        # Moment vector (double-headed arrow)
        m_vec = np.array([mx, my, mz], dtype=float)
        m_mag = np.linalg.norm(m_vec)
        if m_mag > 1e-9:
            lbl = f"{m_mag:.2f} (M)"
            draw_3d_moment(ax, pos, m_vec, base_arrow_len, color="#8E44AD", label=lbl)

    # 2. Member Distributed Loads
    for load in getattr(structure, "loads", []):
        el = getattr(load, "element", None)
        if el is None:
            continue

        # Check for distributed load components
        wx = getattr(load, "wx", 0.0)
        wy = getattr(load, "wy", 0.0)
        wz = getattr(load, "wz", 0.0)
        w_mag = np.hypot(wx, np.hypot(wy, wz))
        if w_mag < 1e-9:
            continue

        # Compute load vector in global coordinates
        w_local = np.array([wx, wy, wz], dtype=float)
        R = el.rotation_matrix_3x3()  # 3x3 direction cosine rotation matrix
        w_global = R.T @ w_local
        w_dir = w_global / w_mag

        # Draw 5 arrows along the element
        n_arrows = 5
        tails = []
        L = el.length
        p_i = np.array([el.node_i.x, el.node_i.y, el.node_i.z], dtype=float)
        p_j = np.array([el.node_j.x, el.node_j.y, el.node_j.z], dtype=float)

        for s in np.linspace(0.1, 0.9, n_arrows):
            pt = p_i + s * (p_j - p_i)
            tail = pt - w_dir * (base_arrow_len * 0.7)
            tails.append(tail)
            ax.quiver(
                tail[0], tail[1], tail[2],
                w_dir[0] * (base_arrow_len * 0.7),
                w_dir[1] * (base_arrow_len * 0.7),
                w_dir[2] * (base_arrow_len * 0.7),
                color="#2980B9",
                arrow_length_ratio=0.3,
                linewidth=1.4,
                zorder=6,
            )

        # Boundary line connecting arrow tails
        if len(tails) > 1:
            tails_arr = np.array(tails)
            ax.plot(tails_arr[:, 0], tails_arr[:, 1], tails_arr[:, 2], color="#2980B9", linestyle="--", linewidth=1.2, zorder=6)
