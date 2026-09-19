"""
Tests for 3D structural visualization module (fem3d.visualization).
"""

import os
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for automated testing

from fem3d import (
    SimpleFrame,
    DrawStructure,
    draw_structure,
    plot_mode_shape,
    plot_buckling_mode,
)


@pytest.fixture
def solved_portal_frame():
    """Create and solve a 3D portal space frame."""
    frame = SimpleFrame()
    # 4 base nodes
    frame.add_node(1, 0.0, 0.0, 0.0)
    frame.add_node(2, 6.0, 0.0, 0.0)
    frame.add_node(3, 6.0, 4.0, 0.0)
    frame.add_node(4, 0.0, 4.0, 0.0)
    # 4 roof nodes
    frame.add_node(5, 0.0, 0.0, 3.5)
    frame.add_node(6, 6.0, 0.0, 3.5)
    frame.add_node(7, 6.0, 4.0, 3.5)
    frame.add_node(8, 0.0, 4.0, 3.5)

    E = 200e6
    A = 0.01
    Iy = 1e-4
    Iz = 2e-4
    J = 3e-4

    # Columns
    frame.add_frame(1, 1, 5, E, A, Iy, Iz, J, extra_mass=25.0)
    frame.add_frame(2, 2, 6, E, A, Iy, Iz, J, extra_mass=25.0)
    frame.add_frame(3, 3, 7, E, A, Iy, Iz, J, extra_mass=25.0)
    frame.add_frame(4, 4, 8, E, A, Iy, Iz, J, extra_mass=25.0)
    # Roof Beams
    frame.add_frame(5, 5, 6, E, A, Iy, Iz, J, extra_mass=25.0)
    frame.add_frame(6, 6, 7, E, A, Iy, Iz, J, extra_mass=25.0)
    frame.add_frame(7, 7, 8, E, A, Iy, Iz, J, extra_mass=25.0)
    frame.add_frame(8, 8, 5, E, A, Iy, Iz, J, extra_mass=25.0)

    # Fixed base supports
    for nid in (1, 2, 3, 4):
        frame.add_support(nid, [1, 1, 1, 1, 1, 1])

    # Nodal loads and moments
    frame.add_node_load(5, [25.0, 0.0, -10.0, 15.0, 0.0, 0.0])
    frame.add_node_load(6, [0.0, 20.0, -10.0, 0.0, 10.0, 0.0])

    frame.solve()
    return frame


@pytest.fixture
def solved_space_truss():
    """Create and solve a 3D space truss."""
    frame = SimpleFrame()
    frame.add_node(1, -2.0, 0.0, 0.0)
    frame.add_node(2, 2.0, 0.0, 0.0)
    frame.add_node(3, 0.0, 3.0, 0.0)
    frame.add_node(4, 0.0, 1.0, 2.5)

    E = 70e6
    A = 0.005

    frame.add_truss(1, 1, 4, E, A)
    frame.add_truss(2, 2, 4, E, A)
    frame.add_truss(3, 3, 4, E, A)

    # Pin supports at base
    frame.add_support(1, [1, 1, 1, 0, 0, 0])
    frame.add_support(2, [1, 1, 1, 0, 0, 0])
    frame.add_support(3, [1, 1, 1, 0, 0, 0])

    # Load at apex
    frame.add_node_load(4, [10.0, 5.0, -30.0, 0.0, 0.0, 0.0])

    frame.solve()
    return frame


def test_draw_structure_frame(solved_portal_frame, tmp_path):
    """Test 3D frame visualization and figure saving."""
    img_path = str(tmp_path / "portal_frame.png")
    drawer = DrawStructure(solved_portal_frame.structure, scale=50.0)
    ax = drawer.draw(
        show_undeformed=True,
        show_deformed=True,
        show_loads=True,
        show_supports=True,
        show_node_labels=True,
        show_element_labels=True,
        title="3D Space Portal Frame",
        save_path=img_path,
        show=False,
    )
    assert ax is not None
    assert os.path.exists(img_path)
    assert os.path.getsize(img_path) > 1000


def test_draw_color_by_force(solved_space_truss, tmp_path):
    """Test 3D truss visualization with color-by-axial-force option."""
    img_path = str(tmp_path / "truss_forces.png")
    ax = solved_space_truss.draw(
        color_by_force=True,
        show_loads=True,
        show_supports=True,
        save_path=img_path,
        show=False,
    )
    assert ax is not None
    assert os.path.exists(img_path)
    assert os.path.getsize(img_path) > 1000


def test_draw_mode_shape(solved_portal_frame, tmp_path):
    """Test 3D modal vibration shape plotting."""
    img_path = str(tmp_path / "mode1.png")
    ax = solved_portal_frame.plot_mode_shape(mode=1, save_path=img_path, show=False)
    assert ax is not None
    assert os.path.exists(img_path)
    assert os.path.getsize(img_path) > 1000


def test_draw_buckling_mode(solved_portal_frame, tmp_path):
    """Test 3D elastic buckling mode plotting."""
    img_path = str(tmp_path / "buckling1.png")
    ax = solved_portal_frame.plot_buckling_mode(mode=1, save_path=img_path, show=False)
    assert ax is not None
    assert os.path.exists(img_path)
    assert os.path.getsize(img_path) > 1000


def test_convenience_functions(solved_space_truss):
    """Test module-level convenience functions."""
    ax1 = draw_structure(solved_space_truss, show=False)
    assert ax1 is not None
