"""
Unit tests for 3D Spring elements.
"""

import pytest
from fem3d import SimpleFrame, Results


def test_3d_spring_displacement():
    frame = SimpleFrame()
    frame.add_node(1, 0.0, 0.0, 0.0)
    frame.add_node(2, 10.0, 0.0, 0.0)

    # Add translational springs kx=50, ky=100, kz=200
    frame.add_spring(1, 1, 2, kx=50.0, ky=100.0, kz=200.0)

    # Node 1 fixed
    frame.add_support(1, [1, 1, 1, 1, 1, 1])

    # Node 2 loaded along X and Z
    Fx = 25.0
    Fz = 100.0
    frame.add_node_load(2, [Fx, 0.0, Fz, 0.0, 0.0, 0.0])

    disp, reac = frame.solve()
    results = Results(frame)
    df_disp = results.node_displacements()
    df_reac = results.reactions()

    # u = F / k
    assert pytest.approx(df_disp.iloc[1]["ux"], rel=1e-6) == Fx / 50.0
    assert pytest.approx(df_disp.iloc[1]["uz"], rel=1e-6) == Fz / 200.0
    assert pytest.approx(df_reac.iloc[0]["Fx"], rel=1e-6) == -Fx
    assert pytest.approx(df_reac.iloc[0]["Fz"], rel=1e-6) == -Fz
