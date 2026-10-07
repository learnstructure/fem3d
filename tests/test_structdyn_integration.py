"""
Unit tests for fem3d integration with structdyn.
Tests MDF creation, modal analysis, damping, and time-history analysis.
"""

import math
import numpy as np
import pytest

from fem3d import (
    Structure,
    Node,
    ElasticMaterial,
    Section,
    FrameElement,
    SimpleFrame,
    DrawStructure,
    plot_mode_shape,
)
from structdyn import MDF, GroundMotion, LoadHistory


@pytest.fixture
def sample_3d_portal_frame():
    """Build a simple 3D single-story space frame."""
    frame = SimpleFrame()
    L = 120.0  # in
    H = 100.0  # in
    E = 30000.0
    G = 11500.0
    A = 10.0
    I2 = 80.0
    I3 = 150.0
    J = 30.0

    # Base nodes (1-4)
    frame.add_node(1, 0, 0, 0)
    frame.add_node(2, L, 0, 0)
    frame.add_node(3, L, L, 0)
    frame.add_node(4, 0, L, 0)

    # Roof nodes (5-8)
    frame.add_node(5, 0, 0, H)
    frame.add_node(6, L, 0, H)
    frame.add_node(7, L, L, H)
    frame.add_node(8, 0, L, H)

    # Base supports
    for nid in [1, 2, 3, 4]:
        frame.add_support(nid, [1, 1, 1, 1, 1, 1])

    # Columns
    eid = 1
    for b_id, t_id in [(1, 5), (2, 6), (3, 7), (4, 8)]:
        frame.add_frame(eid, b_id, t_id, E, A, I2=I2, I3=I3, G=G, J=J)
        eid += 1

    # Beams
    for n_a, n_b in [(5, 6), (6, 7), (7, 8), (8, 5)]:
        frame.add_frame(eid, n_a, n_b, E, A, I2=I2, I3=I3, G=G, J=J)
        eid += 1

    # Lumped masses at roof nodes
    mass_val = 10.0 / 386.4  # kips*s^2/in
    for nid in [5, 6, 7, 8]:
        frame.structure.nodes[nid].set_mass(
            mass=[mass_val, mass_val, mass_val],
            inertia=[1e-4, 1e-4, 1e-4],
        )

    return frame


def test_mdf_from_fem3d_modal_frequencies(sample_3d_portal_frame):
    """
    Verify MDF.from_fem3d produces natural frequencies identical to fem3d native modal analysis.
    """
    frame = sample_3d_portal_frame
    struct = frame.structure

    # Native fem3d modal analysis
    modal_native = struct.modal_analysis(num_modes=4)
    omega_native = modal_native["omega"]

    # structdyn MDF modal analysis
    mdf = MDF.from_fem3d(struct)
    omega_structdyn, phi = mdf.modal.modal_analysis(dof_normalize=None)

    assert len(omega_structdyn) >= 4
    for i in range(4):
        assert abs(omega_structdyn[i] - omega_native[i]) < 1e-6


def test_mdf_from_fem3d_simple_frame_forwarding(sample_3d_portal_frame):
    """
    Verify passing SimpleFrame directly to MDF.from_fem3d or MDF.from_fem.
    """
    frame = sample_3d_portal_frame

    mdf1 = MDF.from_fem3d(frame)
    mdf2 = MDF.from_fem(frame)

    assert mdf1.ndof == len(frame.structure.free_dofs)
    assert mdf2.ndof == len(frame.structure.free_dofs)
    assert np.allclose(mdf1.K, mdf2.K)
    assert np.allclose(mdf1.M, mdf2.M)


def test_mdf_from_fem3d_damping(sample_3d_portal_frame):
    """
    Verify that specifying modal damping creates a valid non-zero damping matrix C.
    """
    frame = sample_3d_portal_frame
    mdf = MDF.from_fem3d(frame, zeta=0.05, n_modes=3)

    assert mdf.C is not None
    assert mdf.C.shape == (mdf.ndof, mdf.ndof)
    assert np.count_nonzero(mdf.C) > 0
    # Damping matrix should be symmetric
    assert np.allclose(mdf.C, mdf.C.T)


def test_mdf_from_fem3d_time_history_force(sample_3d_portal_frame):
    """
    Verify dynamic response under external force history using Newmark-Beta.
    """
    frame = sample_3d_portal_frame
    mdf = MDF.from_fem3d(frame, zeta=0.03, n_modes=3)

    # 1 second sinusoidal force at roof node 5 in X (free DOF 0)
    dt = 0.01
    time_steps = np.arange(0.0, 1.0 + dt, dt)
    force = 10.0 * np.sin(2.0 * np.pi * 3.0 * time_steps)  # 3 Hz excitation
    load = LoadHistory(time_steps=time_steps, load_values=force, dof=[0])

    response = mdf.find_response(load, method="newmark_beta")

    assert response is not None
    assert len(response) == len(time_steps)
    assert "u1" in response.columns
    # Check that roof displacement is non-zero and finite
    max_disp = response["u1"].abs().max()
    assert max_disp > 1e-4
    assert np.all(np.isfinite(response["u1"]))


def test_mdf_from_fem3d_ground_motion_directions(sample_3d_portal_frame):
    """
    Verify that ground motion in X vs Y directions correctly excites corresponding translational DOFs.
    """
    frame = sample_3d_portal_frame
    mdf = MDF.from_fem3d(frame, zeta=0.05, n_modes=3)

    dt = 0.02
    t = np.arange(0.0, 0.5 + dt, dt)
    # Short Ricker-type pulse acceleration
    acc = 38.6 * np.sin(2.0 * np.pi * 5.0 * t)  # ~0.1 g
    gm = GroundMotion.from_arrays(acc, dt=dt)

    resp_x = mdf.find_response_ground_motion(gm, direction="x", method="newmark_beta")
    resp_y = mdf.find_response_ground_motion(gm, direction="y", method="newmark_beta")

    assert resp_x is not None
    assert resp_y is not None

    # Node 5 has DOFs [ux, uy, uz, rx, ry, rz]
    # In free_dofs, find indices for ux and uy
    node5 = frame.structure.nodes[5]
    idx_ux = frame.structure.free_dofs.index(node5.dofs[0])
    idx_uy = frame.structure.free_dofs.index(node5.dofs[1])

    col_ux = f"u{idx_ux + 1}"
    col_uy = f"u{idx_uy + 1}"

    # Under X excitation, X displacement should dominate Y displacement
    assert resp_x[col_ux].abs().max() > 10.0 * resp_x[col_uy].abs().max()
    # Under Y excitation, Y displacement should dominate X displacement
    assert resp_y[col_uy].abs().max() > 10.0 * resp_y[col_ux].abs().max()


def test_set_modal_results_and_mode_shape(sample_3d_portal_frame):
    """
    Verify that set_modal_results stores results on structure and DrawStructure can access them.
    """
    frame = sample_3d_portal_frame
    struct = frame.structure
    mdf = MDF.from_fem3d(struct)

    omega, phi = mdf.modal.modal_analysis(dof_normalize=None)
    struct.set_modal_results(omega, phi)

    assert "modes" in struct.modal_results
    assert "omega" in struct.modal_results
    assert struct.modal_results["modes"].shape[0] == struct.neq

    drawer = DrawStructure(struct)
    assert drawer is not None
