"""
Unit tests for 3D Space Truss elements and auto-stabilization of rotational DOFs.
"""

import numpy as np
import pytest

from fem3d import SimpleFrame, Results


def test_space_truss_tripod_equilibrium():
    """
    Test 3D tripod space truss:
    Verify global equilibrium of reactions against applied apex load.
    """
    frame = SimpleFrame()

    # Base pinned nodes
    frame.add_node(1, -30.0, 0.0, 0.0)
    frame.add_node(2, 15.0, 25.98076, 0.0)
    frame.add_node(3, 15.0, -25.98076, 0.0)

    # Apex node
    frame.add_node(4, 0.0, 0.0, 40.0)

    E = 10000.0
    A = 2.5

    frame.add_truss(1, 1, 4, E=E, A=A)
    frame.add_truss(2, 2, 4, E=E, A=A)
    frame.add_truss(3, 3, 4, E=E, A=A)

    frame.add_support(1, [1, 1, 1, 1, 1, 1])
    frame.add_support(2, [1, 1, 1, 1, 1, 1])
    frame.add_support(3, [1, 1, 1, 1, 1, 1])

    # 3D apex load
    Fx, Fy, Fz = 12.0, -8.0, -25.0
    frame.add_node_load(4, [Fx, Fy, Fz, 0.0, 0.0, 0.0])

    disp, reac = frame.solve()
    results = Results(frame)

    reac_df = results.reactions()
    sum_fx = reac_df["Fx"].sum()
    sum_fy = reac_df["Fy"].sum()
    sum_fz = reac_df["Fz"].sum()

    # Equilibrium: Sum of reactions + applied load = 0
    assert pytest.approx(sum_fx, abs=1e-5) == -Fx
    assert pytest.approx(sum_fy, abs=1e-5) == -Fy
    assert pytest.approx(sum_fz, abs=1e-5) == -Fz


def test_truss_auto_fix_unstable_rotations():
    """
    Ensure that Structure automatically constrains inactive rotational DOFs
    at truss-only nodes so that the stiffness matrix remains non-singular.
    """
    frame = SimpleFrame()
    frame.add_node(1, 0.0, 0.0, 0.0)
    frame.add_node(2, 50.0, 0.0, 0.0)

    frame.add_truss(1, 1, 2, E=29000.0, A=5.0)

    # Base node 1 fully fixed
    frame.add_support(1, [1, 1, 1, 1, 1, 1])
    # Node 2 is unconstrained by user, but since it only has a truss attached,
    # its rotational DOFs should be auto-fixed by Structure._auto_fix_unstable_dofs()
    frame.add_node_load(2, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    disp, reac = frame.solve()
    # If this succeeds without a LinAlgError singular matrix, auto-fix worked!
    assert frame.structure.disp is not None
    assert frame.structure.disp[frame.structure.nodes[2].dofs[0]] > 0.0


def test_kassimali_example_8_1_space_truss():
    """
    Example 8.1 from Kassimali A. (2022), Matrix Analysis of Structures.
    Tests nodal displacements at the apex (node 5) and reaction equilibrium.
    Units: kN, m.
    """
    frame = SimpleFrame()

    # Nodes
    frame.add_node(1, -1.5, 0.0, 2.0)
    frame.add_node(2, 3.0, 0.0, 2.0)
    frame.add_node(3, 1.5, 0.0, -2.0)
    frame.add_node(4, -3.0, 0.0, -2.0)
    frame.add_node(5, 0.0, 6.0, 0.0)

    E = 70e6     # kN/m^2
    A = 3700e-6  # m^2

    # 3D truss elements
    frame.add_truss(1, 1, 5, E, A)
    frame.add_truss(2, 2, 5, E, A)
    frame.add_truss(3, 3, 5, E, A)
    frame.add_truss(4, 4, 5, E, A)

    # Pin supports at base [ux, uy, uz]
    frame.add_support(1, [1, 1, 1])
    frame.add_support(2, [1, 1, 1])
    frame.add_support(3, [1, 1, 1])
    frame.add_support(4, [1, 1, 1])

    # Concentrated 3D loads at apex [Fx, Fy, Fz, Mx, My, Mz]
    frame.add_node_load(5, [0.0, -400.0, -200.0, 0.0, 0.0, 0.0])

    disp, reac = frame.solve()
    results = Results(frame)

    # Verify Node 5 displacements: ux=0.002949, uy=-0.003271, uz=-0.01546
    node_disp = results.node_displacements().set_index("node").loc[5]
    assert pytest.approx(node_disp["ux"], abs=1e-5) == 0.002949
    assert pytest.approx(node_disp["uy"], abs=1e-5) == -0.003271
    assert pytest.approx(node_disp["uz"], abs=1e-5) == -0.015460

    # Verify equilibrium of reactions
    r_sum = results.reactions()[["Fx", "Fy", "Fz"]].sum()
    assert pytest.approx(r_sum["Fx"], abs=1e-3) == 0.0
    assert pytest.approx(r_sum["Fy"], abs=1e-3) == 400.0
    assert pytest.approx(r_sum["Fz"], abs=1e-3) == 200.0

