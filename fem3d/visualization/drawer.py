"""
DrawStructure: 3D structural model visualizer using Matplotlib 3D.
Renders realistic 3D boundary supports, applied point/distributed loads,
double-headed moment arrows, 3D cubic Hermite deformed shapes,
axial force color-mapping, and vibration/buckling mode shapes.
"""

from typing import Optional, Tuple, Union
import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:  # pragma: no cover
    plt = None
    Axes3D = None

from .supports import draw_3d_supports
from .loads import draw_3d_loads
from .elements import draw_3d_elements
from .modes import prepare_mode_displacement


class DrawStructure:
    """
    Visualizes 3D space frames and trusses with publication-quality engineering aesthetics.
    """

    def __init__(self, structure, scale: float = 1.0, arrow_scale: float = 0.12, support_scale: float = 0.05):
        """
        Initialize the 3D structure drawer.

        Parameters
        ----------
        structure : Structure
            The 3D Structure instance to visualize.
        scale : float, optional
            Displacement magnification factor for deformed shapes. Defaults to 1.0.
        arrow_scale : float, optional
            Scaling factor for load arrows relative to structure span. Defaults to 0.12.
        support_scale : float, optional
            Scaling factor for support symbols relative to structure span. Defaults to 0.05.
        """
        self.structure = structure
        self.scale = scale
        self.arrow_scale = arrow_scale
        self.support_scale = support_scale

    def _get_span(self) -> float:
        """Calculate characteristic span (maximum bounding dimension)."""
        if not self.structure.nodes:
            return 1.0
        xs = [n.x for n in self.structure.nodes.values()]
        ys = [n.y for n in self.structure.nodes.values()]
        zs = [n.z for n in self.structure.nodes.values()]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        dz = max(zs) - min(zs)
        span = max(dx, dy, dz)
        return span if span > 1e-6 else 1.0

    def _get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_coords, max_coords) arrays across all nodes."""
        xs = [n.x for n in self.structure.nodes.values()]
        ys = [n.y for n in self.structure.nodes.values()]
        zs = [n.z for n in self.structure.nodes.values()]
        min_c = np.array([min(xs), min(ys), min(zs)], dtype=float)
        max_c = np.array([max(xs), max(ys), max(zs)], dtype=float)
        return min_c, max_c

    def draw(
        self,
        show_undeformed: bool = True,
        show_deformed: bool = True,
        show_loads: bool = True,
        show_supports: bool = True,
        show_node_labels: bool = True,
        show_element_labels: bool = False,
        color_by_force: bool = False,
        elev: float = 25.0,
        azim: float = -60.0,
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 8),
        save_path: Optional[str] = None,
        show: bool = True,
        ax: Optional[Axes3D] = None,
        n_points: int = 40,
    ) -> Axes3D:
        """
        Draw the 3D structure in perspective view.

        Parameters
        ----------
        show_undeformed : bool, optional
            Whether to draw undeformed wireframe. Defaults to True.
        show_deformed : bool, optional
            Whether to draw deformed shape. Defaults to True.
        show_loads : bool, optional
            Whether to draw applied point loads and moments. Defaults to True.
        show_supports : bool, optional
            Whether to draw 3D boundary support symbols. Defaults to True.
        show_node_labels : bool, optional
            Whether to annotate node IDs. Defaults to True.
        show_element_labels : bool, optional
            Whether to annotate element IDs at member midpoints. Defaults to False.
        color_by_force : bool, optional
            Whether to color deformed members by axial force (tension blue, compression red). Defaults to False.
        elev : float, optional
            3D viewing camera elevation angle in degrees. Defaults to 25.0.
        azim : float, optional
            3D viewing camera azimuth angle in degrees. Defaults to -60.0.
        title : str, optional
            Custom figure title.
        figsize : tuple of float, optional
            Figure width and height in inches. Defaults to (10, 8).
        save_path : str, optional
            Filepath to save figure (PNG, PDF, SVG). Defaults to None.
        show : bool, optional
            Whether to display figure using plt.show(). Defaults to True.
        ax : Axes3D, optional
            Existing 3D axes to draw onto. Defaults to None.
        n_points : int, optional
            Number of points for 3D Hermite spline curve rendering. Defaults to 40.

        Returns
        -------
        Axes3D
            The active 3D matplotlib axes.
        """
        if plt is None:
            raise ImportError("matplotlib is required to draw 3D structures.")

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        span = self._get_span()

        # 1. Elements (Undeformed & Deformed, with optional axial force coloring)
        mappable = draw_3d_elements(
            ax=ax,
            structure=self.structure,
            show_undeformed=show_undeformed,
            show_deformed=show_deformed,
            color_by_force=color_by_force,
            scale=self.scale,
            n_points=n_points,
        )

        # 2. Boundary Supports
        if show_supports:
            draw_3d_supports(ax=ax, structure=self.structure, span=span, support_scale=self.support_scale)

        # 3. Applied Loads (forces & double-headed moments)
        if show_loads:
            draw_3d_loads(ax=ax, structure=self.structure, span=span, arrow_scale=self.arrow_scale)

        # 4. Node Markers & Labels
        if show_node_labels:
            for node in self.structure.nodes.values():
                ax.scatter([node.x], [node.y], [node.z], s=35, color="#2C3E50", edgecolors="white", linewidths=0.8, zorder=5)
                # Offset label slightly in +Z and +X
                ax.text(
                    node.x + 0.02 * span,
                    node.y + 0.02 * span,
                    node.z + 0.02 * span,
                    f"N{node.id}",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#2C3E50",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#BDC3C7", alpha=0.85, linewidth=0.7),
                    zorder=6,
                )

        # 5. Element Labels
        if show_element_labels:
            for el in self.structure.elements.values():
                mx = 0.5 * (el.node_i.x + el.node_j.x)
                my = 0.5 * (el.node_i.y + el.node_j.y)
                mz = 0.5 * (el.node_i.z + el.node_j.z)
                ax.text(
                    mx, my, mz,
                    f"E{el.id}",
                    fontsize=7.5,
                    color="#D35400",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#FEF9E7", edgecolor="#F39C12", alpha=0.85, linewidth=0.7),
                    zorder=6,
                )

        # 6. Colorbar for Axial Force
        if color_by_force and mappable is not None:
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.08)
            cbar.set_label("Axial Force (Tension > 0 / Compression < 0)", rotation=270, labelpad=18, fontsize=9)

        # 7. Aspect Ratio & Coordinate Bounds
        min_c, max_c = self._get_bounds()
        ranges = np.maximum(max_c - min_c, 0.15 * span)
        ax.set_box_aspect([ranges[0], ranges[1], ranges[2]])

        # Add visual margin
        pad = 0.1 * span
        ax.set_xlim(min_c[0] - pad, max_c[0] + pad)
        ax.set_ylim(min_c[1] - pad, max_c[1] + pad)
        ax.set_zlim(min_c[2] - pad, max_c[2] + pad)

        # 8. View Angle & Labels
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X (m)", fontsize=9, labelpad=8)
        ax.set_ylabel("Y (m)", fontsize=9, labelpad=8)
        ax.set_zlabel("Z (m)", fontsize=9, labelpad=8)
        ax.grid(True, linestyle=":", alpha=0.4)

        if title is not None:
            ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

        # Legend if not color-by-force and labels exist
        handles, labels = ax.get_legend_handles_labels()
        if handles and not color_by_force:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        return ax

    def draw_mode_shape(
        self,
        mode: int = 1,
        phi: Optional[np.ndarray] = None,
        period: Optional[float] = None,
        omega: Optional[float] = None,
        scale: Optional[float] = None,
        auto_scale: bool = True,
        title: Optional[str] = None,
        **kwargs,
    ) -> Axes3D:
        """
        Draw the 3D vibration mode shape.
        """
        # If phi not provided, run modal analysis on structure
        if phi is None:
            if not hasattr(self.structure, "modal_results") or self.structure.modal_results is None:
                self.structure.modal_analysis(num_modes=max(mode, 3))
            res = self.structure.modal_results
            mode_idx = mode - 1
            mode_vec = res["modes"][:, mode_idx]
            if omega is None and "omega" in res:
                omega = float(res["omega"][mode_idx])
            if period is None and "periods" in res:
                period = float(res["periods"][mode_idx])
        else:
            mode_vec = phi[:, mode - 1] if phi.ndim == 2 else phi

        span = self._get_span()
        disp_full, eff_scale = prepare_mode_displacement(
            self.structure, mode_vec=mode_vec, span=span, scale=scale, auto_scale=auto_scale
        )

        if title is None:
            if period is not None and np.isfinite(period):
                title = f"Mode {mode} Shape (T = {period:.4f} s, ω = {omega:.2f} rad/s)"
            elif omega is not None and np.isfinite(omega):
                title = f"Mode {mode} Shape (ω = {omega:.2f} rad/s)"
            else:
                title = f"Mode {mode} Shape"

        old_disp = getattr(self.structure, "disp", None)
        old_scale = self.scale

        self.structure.disp = disp_full
        self.scale = eff_scale
        show_loads = kwargs.pop("show_loads", False)

        try:
            ax = self.draw(
                show_undeformed=kwargs.pop("show_undeformed", True),
                show_deformed=True,
                show_loads=show_loads,
                title=title,
                **kwargs,
            )
        finally:
            self.structure.disp = old_disp
            self.scale = old_scale

        return ax

    def draw_buckling_mode(
        self,
        mode: int = 1,
        phi: Optional[np.ndarray] = None,
        load_factor: Optional[float] = None,
        scale: Optional[float] = None,
        auto_scale: bool = True,
        title: Optional[str] = None,
        **kwargs,
    ) -> Axes3D:
        """
        Draw the 3D linear elastic buckling mode shape.
        """
        if phi is None:
            from ..buckling_analysis import buckling_analysis
            factors, modes = buckling_analysis(self.structure, num_modes=max(mode, 2))
            mode_idx = mode - 1
            mode_vec = modes[:, mode_idx]
            if load_factor is None and len(factors) > mode_idx:
                load_factor = float(factors[mode_idx])
        else:
            mode_vec = phi[:, mode - 1] if phi.ndim == 2 else phi

        span = self._get_span()
        disp_full, eff_scale = prepare_mode_displacement(
            self.structure, mode_vec=mode_vec, span=span, scale=scale, auto_scale=auto_scale
        )

        if title is None:
            if load_factor is not None:
                title = f"Buckling Mode {mode} (λ_cr = {load_factor:.3f})"
            else:
                title = f"Buckling Mode {mode}"

        old_disp = getattr(self.structure, "disp", None)
        old_scale = self.scale

        self.structure.disp = disp_full
        self.scale = eff_scale
        show_loads = kwargs.pop("show_loads", False)

        try:
            ax = self.draw(
                show_undeformed=kwargs.pop("show_undeformed", True),
                show_deformed=True,
                show_loads=show_loads,
                title=title,
                **kwargs,
            )
        finally:
            self.structure.disp = old_disp
            self.scale = old_scale

        return ax


def draw_structure(structure_or_frame, **kwargs):
    """
    Convenience function to draw a 3D structure.
    Accepts either a Structure or SimpleFrame/SimpleFrame3D instance.
    """
    struct = structure_or_frame.structure if hasattr(structure_or_frame, "structure") else structure_or_frame
    drawer = DrawStructure(struct)
    return drawer.draw(**kwargs)


def plot_mode_shape(structure_or_frame, mode=1, **kwargs):
    """
    Convenience function to plot a 3D vibration mode shape.
    """
    struct = structure_or_frame.structure if hasattr(structure_or_frame, "structure") else structure_or_frame
    drawer = DrawStructure(struct)
    return drawer.draw_mode_shape(mode=mode, **kwargs)


def plot_buckling_mode(structure_or_frame, mode=1, **kwargs):
    """
    Convenience function to plot a 3D elastic buckling mode shape.
    """
    struct = structure_or_frame.structure if hasattr(structure_or_frame, "structure") else structure_or_frame
    drawer = DrawStructure(struct)
    return drawer.draw_buckling_mode(mode=mode, **kwargs)
