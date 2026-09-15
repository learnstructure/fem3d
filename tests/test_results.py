"""
Unit tests for Results post-processor and report generation.
"""

from fem3d import SimpleFrame, Results


def test_results_and_create_report():
    frame = SimpleFrame()
    frame.add_node(1, 0.0, 0.0, 0.0)
    frame.add_node(2, 60.0, 0.0, 0.0)
    frame.add_node(3, 60.0, 60.0, 0.0)

    E, A, Iy, Iz, J = 29000.0, 10.0, 50.0, 100.0, 20.0
    frame.add_frame(1, 1, 2, E, A, Iy, Iz, J)
    frame.add_frame(2, 2, 3, E, A, Iy, Iz, J)

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
    assert "fz_i" in forces_df.columns and "mz_j" in forces_df.columns

    report = results.create_report(print_report=False)
    assert "3D Structural Analysis Report" in report
    assert "Node Support & Load Conditions" in report
    assert "Global Structure Stiffness Matrix K" in report
    assert "Element Local End Forces" in report
