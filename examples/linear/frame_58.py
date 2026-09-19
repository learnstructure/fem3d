# Example 5.8: Logan D. L. (2022), A First Course in the Finite Element Method
# units in kN, m

from fem3d import SimpleFrame, Results

frame = SimpleFrame()

# nodes
frame.add_node(1, 2.5, 0, 0)
frame.add_node(2, 0, 0, 0)
frame.add_node(3, 2.5, 2.5, 0)
frame.add_node(4, 2.5, 0, -2.5)

# Material & Section properties
E = 200e6
A = 6.25e-3
Iy = 40e-6
Iz = 40e-6
G = 60e6
J = 20e-6

# Add 3D frame elements
frame.add_frame(1, 2, 1, E, A, Iy, Iz, G=G, J=J)
frame.add_frame(2, 3, 1, E, A, Iy, Iz, G=G, J=J)
frame.add_frame(3, 4, 1, E, A, Iy, Iz, G=G, J=J)

# Fixed supports at 2, 3, 4
frame.add_support(2, [1, 1, 1, 1, 1, 1])
frame.add_support(3, [1, 1, 1, 1, 1, 1])
frame.add_support(4, [1, 1, 1, 1, 1, 1])

# Concentrated 3D loads at apex [Fx, Fy, Fz, Mx, My, Mz]
frame.add_node_load(1, [0, 0, -200, -100, 0.0, 0])

disp, reac = frame.solve()

results = Results(frame)
print("=== 3D Space Frame Results (Logan Ex. 5.8) ===")
print("\nNode Displacements:")
print(results.node_displacements())
print("\nReactions:")
print(results.reactions())
print("\nElement Bar Forces (local fx_i is tension if negative):")
print(results.element_forces())

# Check equilibrium of reactions vs applied load
r_sum = results.reactions()[["Fx", "Fy", "Fz"]].sum()
print(f"\nSum of reactions: Fx={r_sum['Fx']:.4f}, Fy={r_sum['Fy']:.4f}, Fz={r_sum['Fz']:.4f}")

# Node 1 displacements
print("\nNode 1 displacements:")
print(results.node_displacements().set_index("node").loc[1][["ux", "uy", "uz", "rx", "ry", "rz"]])
# ux=1.747e-06, uy=5.651e-05, uz=-3.356e-04, rx=-3.752e-03, ry=9.935e-05, rz=1.715e-05



# 3D Visualization
# frame.draw(
#     show_undeformed=True,
#     show_deformed=False,
#     show_loads=True,
#     show_supports=True,
#     color_by_force=True,
#     title="Example 5.8: Logan D. L. (2022), A First Course in the Finite Element Method",
# )

