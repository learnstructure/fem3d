"""
Test suite validating 3D multi-story space frame modal analysis against SAP2000 benchmarks.

Validates:
1. Euler-Bernoulli 3D frame formulation (without shear deformation) matching SAP2000:
   - Modes 1 & 2 (Lateral translation): 0.523277 s
   - Mode 3 (Torsion):                  0.474247 s
   - Modes 4 & 5 (Higher translation):   0.172492 s
   - Mode 6 (Higher torsion):           0.159199 s

2. Timoshenko 3D frame formulation (with shear deformation) matching SAP2000:
   - Modes 1 & 2 (Lateral translation): 0.542639 s
   - Mode 3 (Torsion):                  0.491825 s
   - Modes 4 & 5 (Higher translation):   0.178885 s
   - Mode 6 (Higher torsion):           0.165114 s

Also verifies exact Guyan dynamic condensation of massless rotational DOFs.
"""

import math
import numpy as np
import pytest

from fem3d import SimpleFrame, Section, ElasticMaterial
from structdyn.mdf.mdf import MDF


def build_3d_frame(include_shear_deformation: bool = False):
    """
    Build the 4-story, 1-bay x 1-bay 3D space frame model.
    Units: kips, inches, seconds.
    """
    g = 386.09
    bay_x = 16.0 * 12.0  # 192 in
    bay_y = 16.0 * 12.0  # 192 in
    story_h = 10.0 * 12.0  # 120 in
    n_stories = 4

    E = 6000.0  # ksi
    nu = 0.2
    G = E / (2.0 * (1.0 + nu))  # 2500 ksi

    beam_section = Section.from_rectangle(width=20.0, depth=35.0)
    col_section = Section.from_rectangle(width=20.0, depth=20.0)

    frame = SimpleFrame()
    coords_xy = [
        (0.0, 0.0),
        (bay_x, 0.0),
        (bay_x, bay_y),
        (0.0, bay_y),
    ]

    # Ground nodes (fixed)
    for i, (x, y) in enumerate(coords_xy, start=1):
        frame.add_node(i, x, y, 0.0)
        frame.add_support(i, [1, 1, 1, 1, 1, 1])

    # Story nodes
    node_id = 5
    for story in range(1, n_stories + 1):
        z = story * story_h
        for x, y in coords_xy:
            frame.add_node(node_id, x, y, z)
            node_id += 1

    # Columns
    elem_id = 1
    for story in range(1, n_stories + 1):
        lower_base = (story - 1) * 4 + 1
        upper_base = story * 4 + 1
        for k in range(4):
            bot_nid = lower_base + k
            top_nid = upper_base + k
            frame.add_frame(
                elem_id,
                bot_nid,
                top_nid,
                E=E,
                G=G,
                A=col_section,
                include_shear_deformation=include_shear_deformation,
            )
            elem_id += 1

    # Beams
    for story in range(1, n_stories + 1):
        story_base = story * 4 + 1
        floor_pairs = [
            (story_base + 0, story_base + 1),
            (story_base + 1, story_base + 2),
            (story_base + 2, story_base + 3),
            (story_base + 3, story_base + 0),
        ]
        for n1, n2 in floor_pairs:
            frame.add_frame(
                elem_id,
                n1,
                n2,
                E=E,
                G=G,
                A=beam_section,
                include_shear_deformation=include_shear_deformation,
            )
            elem_id += 1

    # Lumped floor masses: 450 kips per story distributed to 4 corner nodes
    mass_per_node = (450.0 / 4.0) / g
    for story in range(1, n_stories + 1):
        story_base = story * 4 + 1
        for k in range(4):
            nid = story_base + k
            frame.structure.nodes[nid].set_mass(
                mass=[mass_per_node, mass_per_node, mass_per_node]
            )

    return frame


def test_frame_modal_euler_bernoulli_sap2000():
    """
    Validate standard Euler-Bernoulli modal analysis against SAP2000 benchmark.
    Tests exact Guyan condensation of massless rotational DOFs.
    """
    frame = build_3d_frame(include_shear_deformation=False)
    mdf = MDF.from_fem3d(frame)
    omega, phi = mdf.modal.modal_analysis(dof_normalize=None)
    periods = 2.0 * math.pi / omega

    # Benchmark SAP2000 periods (Euler-Bernoulli, without shear deformation)
    sap2000_expected = np.array([
        0.523277,
        0.523277,
        0.474247,
        0.172492,
        0.172492,
        0.159199,
    ])

    for i in range(6):
        assert periods[i] == pytest.approx(sap2000_expected[i], rel=1e-4)

    # Verify through high-level Structure.modal_analysis API as well
    modal_dict = frame.structure.modal_analysis(num_modes=6)
    for i in range(6):
        assert modal_dict["periods"][i] == pytest.approx(sap2000_expected[i], rel=1e-4)

    # Verify modal orthogonality: phi^T K phi = omega^2 * (phi^T M phi)
    K_ff, M_ff = frame.structure.get_reduced_matrices()
    for i in range(6):
        phi_i = phi[:, i]
        Kn = float(phi_i.T @ K_ff @ phi_i)
        Mn = float(phi_i.T @ M_ff @ phi_i)
        assert np.sqrt(Kn / Mn) == pytest.approx(omega[i], rel=1e-9)


def test_frame_modal_timoshenko_sap2000():
    """
    Validate Timoshenko space frame modal analysis against SAP2000 benchmark.
    Includes shear deformation for both beams (As=583.33) and columns (As=333.33).
    """
    frame = build_3d_frame(include_shear_deformation=True)
    mdf = MDF.from_fem3d(frame)
    omega, phi = mdf.modal.modal_analysis(dof_normalize=None)
    periods = 2.0 * math.pi / omega

    # Benchmark SAP2000 periods (Timoshenko, with shear deformation)
    sap2000_expected = np.array([
        0.542639,
        0.542639,
        0.491825,
        0.178885,
        0.178885,
        0.165114,
    ])

    for i in range(6):
        assert periods[i] == pytest.approx(sap2000_expected[i], rel=1e-4)

    # Verify modal orthogonality
    K_ff, M_ff = frame.structure.get_reduced_matrices()
    for i in range(6):
        phi_i = phi[:, i]
        Kn = float(phi_i.T @ K_ff @ phi_i)
        Mn = float(phi_i.T @ M_ff @ phi_i)
        assert np.sqrt(Kn / Mn) == pytest.approx(omega[i], rel=1e-9)
