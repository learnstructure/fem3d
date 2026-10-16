"""
Unit tests for Results post-processor and report generation.
"""

from fem3d import SimpleFrame, Results


def test_results_and_create_report():
    frame = SimpleFrame()
    frame.add_node(1, 0.0, 0.0, 0.0)
    frame.add_node(2, 60.0, 0.0, 0.0)
    frame.add_node(3, 60.0, 60.0, 0.0)

    E, A, I2, I3, J = 29000.0, 10.0, 50.0, 100.0, 20.0
    frame.add_frame(1, 1, 2, E, A, I2=I2, I3=I3, J=J)
    frame.add_frame(2, 2, 3, E, A, I2=I2, I3=I3, J=J)

    frame.add_support(1, [1, 1, 1, 1, 1, 1])
    frame.add_support(3, [1, 1, 1, 1, 1, 1])

    frame.add_node_load(2, [0.0, 5.0, 10.0, 0.0, 0.0, 0.0])

    frame.solve()

    results = Results(frame)
    disp_df = results.node_displacements()
    reac_df = results.reactions()
    forces_df = results.element_forces()

    assert len(disp_df) == 3
    assert "ux" in disp_df.columns and "uz" in disp_df.columns
    assert len(reac_df) == 3
    assert "Fx" in reac_df.columns and "Fz" in reac_df.columns
    assert len(forces_df) == 2
    assert "f3_i" in forces_df.columns and "m3_j" in forces_df.columns
    assert "V3_i" in forces_df.columns and "M3_j" in forces_df.columns

    report = results.create_report(print_report=False)
    assert "3D Structural Analysis Report" in report
    assert "Node Support & Load Conditions" in report
    assert "Global Structure Stiffness Matrix K" in report
    assert "Element Local End Forces" in report


def test_simple_frame_units_and_self_weight_3d():
    """Verify 3D SimpleFrame accepts units and can materialize self-weight."""
    frame = SimpleFrame(units="kN, m")
    assert frame.units == "kN, m"
    assert frame.structure.units == "kN, m"

    frame.add_node(1, 0.0, 0.0, 0.0)
    frame.add_node(2, 6.0, 0.0, 0.0)
    frame.add_frame(1, 1, 2, E=30e6, A=0.12, I2=0.0016, I3=0.0025, unit_weight=25.0)
    frame.add_support(1, [1, 1, 1, 1, 1, 1])
    frame.add_support(2, [1, 1, 1, 1, 1, 1])

    frame.apply_self_weight(direction="-z")
    assert len(frame.structure.loads) == 1
    load = frame.structure.loads[0]
    # w = 0.12 * 25.0 = 3.0 kN/m
    assert abs(load.w3 - (-3.0)) < 1e-9 or abs(load.w2 - (-3.0)) < 1e-9

    disp, reac = frame.solve()
    assert disp is not None
    assert reac is not None

