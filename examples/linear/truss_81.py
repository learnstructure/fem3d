# Example 8.1: Kassimali A. (2022), Matrix analysis of structures
# units in kN, m

from fem3d import SimpleFrame, Results

frame = SimpleFrame()

# nodes
frame.add_node(1, -1.5, 0, 2)
frame.add_node(2, 3, 0, 2)
frame.add_node(3, 1.5,0, -2)
frame.add_node(4, -3, 0, -2)
frame.add_node(5, 0, 6, 0)


E = 70e6                                 #kN/m2
A = 3700e-6                             # m2

# Add 3D truss elements
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
frame.add_node_load(5, [0, -400, -200,   0.0, 0.0, 0.0])

disp, reac = frame.solve()

results = Results(frame)
print("=== 3D Space Truss Results ===")
print("\nNode Displacements:")
print(results.node_displacements())
print("\nReactions:")
print(results.reactions())
print("\nElement Bar Forces (local fx_i is tension if negative):")
print(results.element_forces())

# Check equilibrium of reactions vs applied load
r_sum = results.reactions()[["Fx", "Fy", "Fz"]].sum()
print(f"\nSum of reactions: Fx={r_sum['Fx']:.4f}, Fy={r_sum['Fy']:.4f}, Fz={r_sum['Fz']:.4f}")

#Node 5 displacements
print("\nNode 5 displacements:")
print(results.node_displacements().loc[4][["ux","uy","uz"]])        #ux=0.002949 , uy=-0.003271 , uz=-0.01546

# 3D Visualization
frame.draw(
    show_undeformed=True,
    show_deformed=False,
    show_loads=True,
    show_supports=True,
    color_by_force=True,
    title="Example 8.1 - 3D Space Truss (Kassimali)",
)

