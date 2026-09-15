"""
Example: 3D Space Truss (Tripod) Benchmark.

A 3-bar space truss pinned at 3 base nodes on the ground (Z = 0)
meeting at a common apex (Node 4) under a 3D concentrated load.

Geometry:
- Node 1: (-40, 0, 0)
- Node 2: (20, 34.641, 0)
- Node 3: (20, -34.641, 0)
- Node 4 (apex): (0, 0, 60)

Bars:
- Bar 1: 1 - 4
- Bar 2: 2 - 4
- Bar 3: 3 - 4
"""

from fem3d import SimpleFrame, Results

def main():
    frame = SimpleFrame()

    # Base nodes
    frame.add_node(1, -40.0, 0.0, 0.0)
    frame.add_node(2, 20.0, 34.641016, 0.0)
    frame.add_node(3, 20.0, -34.641016, 0.0)

    # Apex node
    frame.add_node(4, 0.0, 0.0, 60.0)

    E = 10000.0  # ksi (Aluminum)
    A = 2.0      # in^2

    # Add 3D truss elements
    frame.add_truss(1, 1, 4, E=E, A=A)
    frame.add_truss(2, 2, 4, E=E, A=A)
    frame.add_truss(3, 3, 4, E=E, A=A)

    # Pin supports at base [ux, uy, uz, rx, ry, rz]
    frame.add_support(1, [1, 1, 1, 1, 1, 1])
    frame.add_support(2, [1, 1, 1, 1, 1, 1])
    frame.add_support(3, [1, 1, 1, 1, 1, 1])

    # Concentrated 3D load at apex [Fx, Fy, Fz, Mx, My, Mz]
    # Downward vertical load and lateral force
    frame.add_node_load(4, [10.0, 5.0, -20.0, 0.0, 0.0, 0.0])

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
    print("Expected: Fx=-10.0, Fy=-5.0, Fz=+20.0")


if __name__ == "__main__":
    main()
