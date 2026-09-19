# Example 8.4: Kassimali A. (2022), Matrix analysis of structures
# units in kN, m

from fem3d import SimpleFrame, Results

frame = SimpleFrame()

# nodes
frame.add_node(1, 0, 0, 0)
frame.add_node(2, -5, 0, 0)
frame.add_node(3, 0, 0, -5)
frame.add_node(4, 0, 5, 0)

# Material & Section properties
# In Kassimali: Iz = 710(10^6) mm4 (strong axis), Iy = 234(10^6) mm4 (weak axis)
E = 200e6       # kN/m2 (200 GPa)
G = 79.31e6     # kN/m2 (79.31 GPa)
A = 73500e-6    # m2 (73,500 mm2)
Iz = 710e-6     # m4 (710 x 10^6 mm4) - major/strong axis
Iy = 234e-6     # m4 (234 x 10^6 mm4) - minor/weak axis
J = 15e-6       # m4 (15 x 10^6 mm4)

# Add 3D frame elements
# Coordinate transformation from Kassimali (Y-up) to fem3d (Z-up):
# [X_fem, Y_fem, Z_fem] = [X_k, -Z_k, Y_k]  (a 90-deg rotation about X)
# Member 1: along +X, roll_angle = 0
# Member 2: vertical along +Z, Kassimali roll Psi=90 deg relative to Y-up corresponds to roll_angle = 180 deg in fem3d
# Member 3: along -Y, Kassimali roll Psi=30 deg corresponds to roll_angle = +30 deg in fem3d
frame.add_frame(1, 2, 1, E, A, Iy=Iy, Iz=Iz, G=G, J=J, roll_angle=0)
frame.add_frame(2, 3, 1, E, A, Iy=Iy, Iz=Iz, G=G, J=J, roll_angle=180)
frame.add_frame(3, 4, 1, E, A, Iy=Iy, Iz=Iz, G=G, J=J, roll_angle=30)

# Fixed supports at 2, 3, 4
frame.add_support(2, [1, 1, 1, 1, 1, 1])
frame.add_support(3, [1, 1, 1, 1, 1, 1])
frame.add_support(4, [1, 1, 1, 1, 1, 1])

# Concentrated 3D loads at apex [Fx, Fy, Fz, Mx, My, Mz]
# In Kassimali: Mx_k = -150, My_k = 0, Mz_k = 150 -> [Mx, My, Mz]_fem = [-150, -150, 0]
frame.add_node_load(1, [0, 0, 0, -150, -150, 0])
frame.add_distributed_load(1, wz=-48, coord_system='global')

disp, reac = frame.solve()

results = Results(frame)
print("=== 3D Space Frame Results (fem3d vs Kassimali Ex. 8.4) ===")
print("\nNode Displacements:")
print(results.node_displacements())
print("\nReactions:")
print(results.reactions())
print("\nElement Forces (local coordinates):")
import pandas as pd
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)
print(results.element_forces())

# Check equilibrium of reactions vs applied load
r_sum = results.reactions()[["Fx", "Fy", "Fz"]].sum()
print(f"\nSum of reactions: Fx={r_sum['Fx']:.4f}, Fy={r_sum['Fy']:.4f}, Fz={r_sum['Fz']:.4f}")

# Node 1 displacements
print("\nNode 1 displacements:")
print(results.node_displacements().set_index("node").loc[1][["ux", "uy", "uz", "rx", "ry", "rz"]])
# ux=-7.313e-06, uy=9.799e-06, uz=-1.512e-05, rx=-7.621e-04, ry=-1.650e-03, rz=2.684e-04

# 3D Visualization
# frame.draw(
#     show_undeformed=True,
#     show_deformed=False,
#     show_loads=True,
#     show_supports=True,
#     color_by_force=True,
#     title="Example 8.4 - 3D Space Frame (Kassimali)",
# )

