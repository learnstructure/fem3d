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
