"""
Unit tests for 3D Frame Elements, coordinate transformations, and static solutions.
"""

import math
import numpy as np
import pytest

from fem3d import SimpleFrame, Results, Structure, Node, ElasticMaterial, Section, FrameElement
from fem3d.loads import PointLoad, DistributedLoad


def test_cantilever_axial_extension():
    """Test axial elongation of 3D frame: delta = P * L / (E * A)."""
    frame = SimpleFrame()
    frame.add_node(1, 0.0, 0.0, 0.0)
    frame.add_node(2, 100.0, 0.0, 0.0)

    E, A = 30000.0, 10.0
    frame.add_frame(1, 1, 2, E=E, A=A, I2=100.0, I3=100.0, J=50.0)
    frame.add_support(1, [1, 1, 1, 1, 1, 1])

    P = 15.0
    frame.add_node_load(2, [P, 0.0, 0.0, 0.0, 0.0, 0.0])

    disp, reac = frame.solve()
    results = Results(frame)
    df_disp = results.node_displacements()
    df_reac = results.reactions()

    expected_ux = P * 100.0 / (E * A)
    assert pytest.approx(df_disp.iloc[1]["ux"], rel=1e-6) == expected_ux
    assert pytest.approx(df_reac.iloc[0]["Fx"], rel=1e-6) == -P


def test_cantilever_pure_torsion():
    """Test pure torsion of 3D frame: theta = T * L / (G * J)."""
    frame = SimpleFrame()
    frame.add_node(1, 0.0, 0.0, 0.0)
    frame.add_node(2, 120.0, 0.0, 0.0)

    E, nu = 30000.0, 0.25
    G = E / (2.0 * (1.0 + nu))
    A, J = 20.0, 80.0
    frame.add_frame(1, 1, 2, E=E, A=A, I2=100.0, I3=100.0, J=J, nu=nu)
    frame.add_support(1, [1, 1, 1, 1, 1, 1])

    T_torque = 60.0
    frame.add_node_load(2, [0.0, 0.0, 0.0, T_torque, 0.0, 0.0])

    frame.solve()
    results = Results(frame)
    df_disp = results.node_displacements()
    df_reac = results.reactions()

    expected_rx = T_torque * 120.0 / (G * J)
    assert pytest.approx(df_disp.iloc[1]["rx"], rel=1e-6) == expected_rx
    assert pytest.approx(df_reac.iloc[0]["Mx"], rel=1e-6) == -T_torque


def test_cantilever_biaxial_bending():
    """
    Test biaxial bending of 3D cantilever:
    Lateral load in Y bends about local 2: delta_y = Py * L^3 / (3*E*I2), theta_z = Py * L^2 / (2*E*I2)
    Lateral load in Z bends about local 3: delta_z = Pz * L^3 / (3*E*I3), theta_y = -Pz * L^2 / (2*E*I3)
    """
    frame = SimpleFrame()
    frame.add_node(1, 0.0, 0.0, 0.0)
    frame.add_node(2, 100.0, 0.0, 0.0)

    E = 29000.0
    A = 15.0
    I3 = 200.0  # Major axis
    I2 = 50.0   # Minor axis
    J = 25.0
    L = 100.0

    frame.add_frame(1, 1, 2, E=E, A=A, I2=I2, I3=I3, J=J)
    frame.add_support(1, [1, 1, 1, 1, 1, 1])

    Py = 8.0
    Pz = 4.0
    frame.add_node_load(2, [0.0, Py, Pz, 0.0, 0.0, 0.0])

    frame.solve()
    results = Results(frame)
    df_disp = results.node_displacements()
    df_reac = results.reactions()

    tip = df_disp.iloc[1]
    reac = df_reac.iloc[0]

    expected_uy = Py * (L**3) / (3.0 * E * I2)   # global Y deflection from Py
    expected_rz = Py * (L**2) / (2.0 * E * I2)   # rotation rz

    expected_uz = Pz * (L**3) / (3.0 * E * I3)   # global Z deflection from Pz
    expected_ry = Pz * (L**2) / (2.0 * E * I3)   # rotation ry from Pz

    assert abs(tip["uy"] - expected_uy) < 1e-4 * abs(expected_uy) + 1e-10
    assert abs(tip["rz"] - expected_rz) < 1e-4 * abs(expected_rz) + 1e-10
    assert abs(tip["uz"] - expected_uz) < 1e-4 * abs(expected_uz) + 1e-10

    # Force equilibrium
    assert abs(reac["Fy"] + Py) < 1e-6 * abs(Py) + 1e-10
    assert abs(reac["Fz"] + Pz) < 1e-6 * abs(Pz) + 1e-10


def test_vertical_column_orientation():
    """
    Test a vertical column (parallel to global Z) to ensure rotation matrices
    and bending responses align properly.
    """
    frame = SimpleFrame()
    frame.add_node(1, 0.0, 0.0, 0.0)
    frame.add_node(2, 0.0, 0.0, 150.0)

    E = 29000.0
    A = 20.0
    I2, I3, J = 100.0, 250.0, 50.0
    L = 150.0

    frame.add_frame(1, 1, 2, E=E, A=A, I2=I2, I3=I3, J=J)
    frame.add_support(1, [1, 1, 1, 1, 1, 1])

    # Horizontal shear load along global X
    Fx = 10.0
    frame.add_node_load(2, [Fx, 0.0, 0.0, 0.0, 0.0, 0.0])

    frame.solve()
    results = Results(frame)
    reac = results.reactions().iloc[0]

    # Equilibrium in global X and moment
    assert pytest.approx(reac["Fx"], rel=1e-5) == -Fx
    assert pytest.approx(abs(reac["My"]), rel=1e-5) == Fx * L


def test_distributed_load_equilibrium():
    """
    Test uniformly distributed load on a fixed-fixed 3D beam.
    Fixed end reactions should equal w*L/2, and fixed end moments w*L^2/12.
    """
    frame = SimpleFrame()
    frame.add_node(1, 0.0, 0.0, 0.0)
    frame.add_node(2, 120.0, 0.0, 0.0)

    E = 29000.0
    A = 10.0
    I2, I3, J = 80.0, 150.0, 40.0
    L = 120.0

    frame.add_frame(1, 1, 2, E=E, A=A, I2=I2, I3=I3, J=J)
    frame.add_support(1, [1, 1, 1, 1, 1, 1])
    frame.add_support(2, [1, 1, 1, 1, 1, 1])

    w3 = 0.5  # kips/in along local 3 (wz)
    frame.add_distributed_load(1, w3=w3)

    frame.solve()
    results = Results(frame)
    reac = results.reactions()

    assert abs(reac.iloc[0]["Fy"] - w3 * L / 2.0) < 1e-4 * abs(w3 * L / 2.0) + 1e-10
    assert abs(reac.iloc[1]["Fy"] - w3 * L / 2.0) < 1e-4 * abs(w3 * L / 2.0) + 1e-10


def test_logan_example_5_8_space_frame():
    """
    Example 5.8 from Logan D. L. (2022), A First Course in the Finite Element Method.
    Tests nodal displacements at the apex (node 1) and reaction equilibrium.
    Units: kN, m.
    """
    frame = SimpleFrame()

    # Nodes
    frame.add_node(1, 2.5, 0.0, 0.0)
    frame.add_node(2, 0.0, 0.0, 0.0)
    frame.add_node(3, 2.5, 2.5, 0.0)
    frame.add_node(4, 2.5, 0.0, -2.5)

    # Material & Section properties
    E = 200e6
    A = 6.25e-3
    I2 = 40e-6
    I3 = 40e-6
    G = 60e6
    J = 20e-6

    # 3D frame elements
    frame.add_frame(1, 2, 1, E=E, A=A, I2=I2, I3=I3, G=G, J=J)
    frame.add_frame(2, 3, 1, E=E, A=A, I2=I2, I3=I3, G=G, J=J)
    frame.add_frame(3, 4, 1, E=E, A=A, I2=I2, I3=I3, G=G, J=J)

    # Fixed supports at 2, 3, 4
    frame.add_support(2, [1, 1, 1, 1, 1, 1])
    frame.add_support(3, [1, 1, 1, 1, 1, 1])
    frame.add_support(4, [1, 1, 1, 1, 1, 1])

    # Concentrated 3D loads at apex [Fx, Fy, Fz, Mx, My, Mz]
    frame.add_node_load(1, [0.0, 0.0, -200.0, -100.0, 0.0, 0.0])

    disp, reac = frame.solve()
    results = Results(frame)

    # Verify Node 1 displacements:
    node_disp = results.node_displacements().set_index("node").loc[1]
    assert pytest.approx(node_disp["ux"], rel=1e-3) == 1.747e-06
    assert pytest.approx(node_disp["uy"], rel=1e-3) == 5.651e-05
    assert pytest.approx(node_disp["uz"], rel=1e-3) == -3.356e-04
    assert pytest.approx(node_disp["rx"], rel=1e-3) == -3.752e-03
    assert pytest.approx(node_disp["ry"], rel=1e-3) == 9.935e-05
    assert pytest.approx(node_disp["rz"], rel=1e-3) == 1.715e-05

    # Verify reaction equilibrium
    r_sum = results.reactions()[["Fx", "Fy", "Fz"]].sum()
    assert pytest.approx(r_sum["Fx"], abs=1e-4) == 0.0
    assert pytest.approx(r_sum["Fy"], abs=1e-4) == 0.0
    assert pytest.approx(r_sum["Fz"], abs=1e-4) == 200.0


def test_kassimali_example_8_4_space_frame():
    """
    Example 8.4 from Kassimali A. (2022), Matrix Analysis of Structures.
    Tests nodal displacements at the junction (node 1), support reactions,
    and member local internal forces.
    Units: kN, m.
    """
    frame = SimpleFrame()

    # Nodes
    frame.add_node(1, 0.0, 0.0, 0.0)
    frame.add_node(2, -5.0, 0.0, 0.0)
    frame.add_node(3, 0.0, 0.0, -5.0)
    frame.add_node(4, 0.0, 5.0, 0.0)

    # Material & Section properties
    E = 200e6       # kN/m2 (200 GPa)
    G = 79.31e6     # kN/m2 (79.31 GPa)
    A = 73500e-6    # m2 (73,500 mm2)
    I3 = 710e-6     # m4 (710 x 10^6 mm4) - about local axis 3
    I2 = 234e-6     # m4 (234 x 10^6 mm4) - about local axis 2
    J = 15e-6       # m4 (15 x 10^6 mm4)

    # Add 3D frame elements with roll angles
    frame.add_frame(1, 2, 1, E=E, A=A, I2=I2, I3=I3, G=G, J=J, roll_angle=0)
    frame.add_frame(2, 3, 1, E=E, A=A, I2=I2, I3=I3, G=G, J=J, roll_angle=180)
    frame.add_frame(3, 4, 1, E=E, A=A, I2=I2, I3=I3, G=G, J=J, roll_angle=30)

    # Fixed supports at 2, 3, 4
    frame.add_support(2, [1, 1, 1, 1, 1, 1])
    frame.add_support(3, [1, 1, 1, 1, 1, 1])
    frame.add_support(4, [1, 1, 1, 1, 1, 1])

    # Concentrated moment at node 1 & distributed load on element 1
    frame.add_node_load(1, [0.0, 0.0, 0.0, -150.0, -150.0, 0.0])
    frame.add_distributed_load(1, wz=-48.0, coord_system='global')

    disp, reac = frame.solve()
    results = Results(frame)

    # 1. Verify Node 1 displacements:
    node_disp = results.node_displacements().set_index("node").loc[1]
    assert pytest.approx(node_disp["ux"], rel=1e-3) == -7.313e-06
    assert pytest.approx(node_disp["uy"], rel=1e-3) == 9.799e-06
    assert pytest.approx(node_disp["uz"], rel=1e-3) == -1.512e-05
    assert pytest.approx(node_disp["rx"], rel=1e-3) == -7.621e-04
    assert pytest.approx(node_disp["ry"], rel=1e-3) == -1.650e-03
    assert pytest.approx(node_disp["rz"], rel=1e-3) == 2.684e-04

    # 2. Verify support reactions
    reactions = results.reactions().set_index("node")
    # Node 2
    assert pytest.approx(reactions.loc[2]["Fx"], abs=0.01) == 21.500
    assert pytest.approx(reactions.loc[2]["Fy"], abs=0.01) == 2.970
    assert pytest.approx(reactions.loc[2]["Fz"], abs=0.01) == 176.429
    assert pytest.approx(reactions.loc[2]["Mx"], abs=0.01) == 0.181
    assert pytest.approx(reactions.loc[2]["My"], abs=0.01) == -194.220
    assert pytest.approx(reactions.loc[2]["Mz"], abs=0.01) == 4.914

    # Node 3
    assert pytest.approx(reactions.loc[3]["Fx"], abs=0.01) == -18.497
    assert pytest.approx(reactions.loc[3]["Fy"], abs=0.01) == 25.840
    assert pytest.approx(reactions.loc[3]["Fz"], abs=0.01) == 44.463
    assert pytest.approx(reactions.loc[3]["Mx"], abs=0.01) == -42.955
    assert pytest.approx(reactions.loc[3]["My"], abs=0.01) == -30.801
    assert pytest.approx(reactions.loc[3]["Mz"], abs=0.01) == -0.064

    # Node 4
    assert pytest.approx(reactions.loc[4]["Fx"], abs=0.01) == -3.003
    assert pytest.approx(reactions.loc[4]["Fy"], abs=0.01) == -28.810
    assert pytest.approx(reactions.loc[4]["Fz"], abs=0.01) == 19.108
    assert pytest.approx(reactions.loc[4]["Mx"], abs=0.01) == -31.965
    assert pytest.approx(reactions.loc[4]["My"], abs=0.01) == 0.393
    assert pytest.approx(reactions.loc[4]["Mz"], abs=0.01) == -5.014

    # Global equilibrium
    r_sum = results.reactions()[["Fx", "Fy", "Fz"]].sum()
    assert pytest.approx(r_sum["Fx"], abs=1e-3) == 0.0
    assert pytest.approx(r_sum["Fy"], abs=1e-3) == 0.0
    assert pytest.approx(r_sum["Fz"], abs=1e-3) == 240.0

    # 3. Verify element internal forces (local coordinates 1, 2, 3)
    el_forces = results.element_forces().set_index("element")
    # Member 1 (checking both f1/f2/f3 and P/V2/V3)
    assert pytest.approx(el_forces.loc[1]["f1_i"], abs=0.01) == 21.500
    assert pytest.approx(el_forces.loc[1]["f2_i"], abs=0.01) == 176.429
    assert pytest.approx(el_forces.loc[1]["f3_i"], abs=0.01) == -2.970
    assert pytest.approx(el_forces.loc[1]["m3_i"], abs=0.01) == 194.220
    assert pytest.approx(el_forces.loc[1]["f1_j"], abs=0.01) == -21.500
    assert pytest.approx(el_forces.loc[1]["f2_j"], abs=0.01) == 63.571

    assert pytest.approx(el_forces.loc[1]["P_i"], abs=0.01) == 21.500
    assert pytest.approx(el_forces.loc[1]["V2_i"], abs=0.01) == 176.429
    assert pytest.approx(el_forces.loc[1]["V3_i"], abs=0.01) == -2.970
    assert pytest.approx(el_forces.loc[1]["M3_i"], abs=0.01) == 194.220

    # Member 2
    assert pytest.approx(el_forces.loc[2]["f1_i"], abs=0.01) == 44.463
    assert pytest.approx(el_forces.loc[2]["f2_i"], abs=0.01) == -25.840
    assert pytest.approx(el_forces.loc[2]["f3_i"], abs=0.01) == -18.497

    # Member 3
    assert pytest.approx(el_forces.loc[3]["f1_i"], abs=0.01) == 28.810
    assert pytest.approx(el_forces.loc[3]["f2_i"], abs=0.01) == 18.049
    assert pytest.approx(el_forces.loc[3]["f3_i"], abs=0.01) == -6.953


def test_aliases():
    """Verify BeamElement and SimpleFrame3D aliases function identically."""
    from fem3d import BeamElement as Beam, SimpleFrame3D as SF3D
    assert Beam is FrameElement
    assert SF3D is SimpleFrame
