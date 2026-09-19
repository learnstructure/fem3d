"""
3D Boundary support symbols for structural visualization.
Supports fixed pads, pinned pyramids/cones, and roller bearings.
"""

from typing import Tuple, List, Optional
import numpy as np

try:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:  # pragma: no cover
    Poly3DCollection = None


def get_support_normal(structure, node) -> np.ndarray:
    """
    Determine the unit normal vector pointing away from the structure into the foundation/ground.
    """
    connected_vecs = []
    for el in structure.elements.values():
        if el.node_i.id == node.id:
            vec = np.array([el.node_j.x - node.x, el.node_j.y - node.y, el.node_j.z - node.z], dtype=float)
            length = np.linalg.norm(vec)
            if length > 1e-9:
                connected_vecs.append(vec / length)
        elif el.node_j.id == node.id:
            vec = np.array([el.node_i.x - node.x, el.node_i.y - node.y, el.node_i.z - node.z], dtype=float)
            length = np.linalg.norm(vec)
            if length > 1e-9:
                connected_vecs.append(vec / length)

    if not connected_vecs:
        return np.array([0.0, 0.0, -1.0])

    sum_vec = np.sum(connected_vecs, axis=0)
    norm = np.linalg.norm(sum_vec)
    if norm < 1e-6:
        # Balanced or collinear members; default to downward
        return np.array([0.0, 0.0, -1.0])

    # Foundation points opposite to connected members
    normal = -sum_vec / norm

    # Snap to dominant Cartesian axis if close to vertical
    if abs(normal[2]) > 0.7:
        return np.array([0.0, 0.0, -1.0 if normal[2] <= 0 else 1.0])

    return normal


def draw_3d_box(ax, center: np.ndarray, size: np.ndarray, facecolor="#95A5A6", edgecolor="#2C3E50", alpha=0.8):
    """Draw a 3D rectangular box/pad."""
    if Poly3DCollection is None:
        return

    cx, cy, cz = center
    dx, dy, dz = size / 2.0

    # 8 vertices
    vertices = np.array([
        [cx - dx, cy - dy, cz - dz],
        [cx + dx, cy - dy, cz - dz],
        [cx + dx, cy + dy, cz - dz],
        [cx - dx, cy + dy, cz - dz],
        [cx - dx, cy - dy, cz + dz],
        [cx + dx, cy - dy, cz + dz],
        [cx + dx, cy + dy, cz + dz],
        [cx - dx, cy + dy, cz + dz],
    ])

    # 6 faces
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
        [vertices[3], vertices[0], vertices[4], vertices[7]],  # left
    ]

    poly = Poly3DCollection(faces, facecolors=facecolor, edgecolors=edgecolor, linewidths=1.0, alpha=alpha)
    ax.add_collection3d(poly)


def draw_3d_pyramid(ax, apex: np.ndarray, base_center: np.ndarray, base_size: float, facecolor="#BDC3C7", edgecolor="#2C3E50", alpha=0.8):
    """Draw a 3D pyramid (pinned base) with apex at the node."""
    if Poly3DCollection is None:
        return

    bx, by, bz = base_center
    d = base_size / 2.0

    # Base square vertices (assuming ground plane roughly horizontal or oriented)
    b_verts = [
        np.array([bx - d, by - d, bz]),
        np.array([bx + d, by - d, bz]),
        np.array([bx + d, by + d, bz]),
        np.array([bx - d, by + d, bz]),
    ]

    faces = [
        [b_verts[0], b_verts[1], b_verts[2], b_verts[3]],  # base
        [apex, b_verts[0], b_verts[1]],                   # side 1
        [apex, b_verts[1], b_verts[2]],                   # side 2
        [apex, b_verts[2], b_verts[3]],                   # side 3
        [apex, b_verts[3], b_verts[0]],                   # side 4
    ]

    poly = Poly3DCollection(faces, facecolors=facecolor, edgecolors=edgecolor, linewidths=1.0, alpha=alpha)
    ax.add_collection3d(poly)


def draw_3d_supports(ax, structure, span: float, support_scale: float = 0.05):
    """
    Draw 3D support symbols at all restrained nodes.
    """
    if Poly3DCollection is None:
        return

    size = max(span * support_scale, 0.05)

    for node in structure.nodes.values():
        if not any(node.support):
            continue

        ux, uy, uz, rx, ry, rz = node.support
        pos = np.array([node.x, node.y, node.z], dtype=float)
        normal = get_support_normal(structure, node)

        # 1. Fixed Support (all 6 DOFs or all 3 translations + rotations fixed)
        if ux and uy and uz and (rx or ry or rz):
            pad_center = pos + normal * (size * 0.6)
            box_size = np.array([size * 1.4, size * 1.4, size * 0.6])
            draw_3d_box(ax, pad_center, box_size, facecolor="#7F8C8D", edgecolor="#1A252F", alpha=0.85)

            # Ground hash plate beneath the pad
            ground_center = pad_center + normal * (size * 0.35)
            ground_size = np.array([size * 1.8, size * 1.8, size * 0.15])
            draw_3d_box(ax, ground_center, ground_size, facecolor="#BDC3C7", edgecolor="#2C3E50", alpha=0.6)

        # 2. Pinned Support (translations fixed, rotations free)
        elif ux and uy and uz:
            base_center = pos + normal * size
            draw_3d_pyramid(ax, apex=pos, base_center=base_center, base_size=size * 1.1, facecolor="#BDC3C7", edgecolor="#2C3E50", alpha=0.85)
            # Small foundation footing plate underneath pyramid base
            footing_center = base_center + normal * (size * 0.2)
            footing_size = np.array([size * 1.5, size * 1.5, size * 0.2])
            draw_3d_box(ax, footing_center, footing_size, facecolor="#7F8C8D", edgecolor="#1A252F", alpha=0.7)

        # 3. Roller / Partial Support (1 or 2 translational restraints)
        else:
            base_center = pos + normal * (size * 0.8)
            footing_size = np.array([size * 1.2, size * 1.2, size * 0.25])
            draw_3d_box(ax, base_center, footing_size, facecolor="#95A5A6", edgecolor="#2C3E50", alpha=0.8)
            # Small roller sphere indicator
            sphere_pos = pos + normal * (size * 0.4)
            ax.scatter([sphere_pos[0]], [sphere_pos[1]], [sphere_pos[2]], s=40, color="#E67E22", edgecolors="#2C3E50", zorder=5)
