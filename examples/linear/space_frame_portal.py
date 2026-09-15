"""
Example: 3D Portal Space Frame Structure.

A 4-column, 4-beam 3D building bay frame under vertical and lateral loads.

Geometry:
Base nodes (Z = 0):
- Node 1: (0, 0, 0)
- Node 2: (120, 0, 0)
- Node 3: (120, 120, 0)
- Node 4: (0, 120, 0)

Top roof nodes (Z = 144 in):
- Node 5: (0, 0, 144)
- Node 6: (120, 0, 144)
- Node 7: (120, 120, 144)
- Node 8: (0, 120, 144)

Members:
- 4 Columns: 1-5, 2-6, 3-7, 4-8
- 4 Roof Beams: 5-6, 6-7, 7-8, 8-5
"""

from fem3d import SimpleFrame, Results

def main():
    frame = SimpleFrame()

    # Define base nodes
    frame.add_node(1, 0.0, 0.0, 0.0)
    frame.add_node(2, 120.0, 0.0, 0.0)
    frame.add_node(3, 120.0, 120.0, 0.0)
    frame.add_node(4, 0.0, 120.0, 0.0)

    # Define roof nodes
    H = 144.0
    frame.add_node(5, 0.0, 0.0, H)
    frame.add_node(6, 120.0, 0.0, H)
    frame.add_node(7, 120.0, 120.0, H)
    frame.add_node(8, 0.0, 120.0, H)

    # Material & Section properties
    E = 29000.0  # ksi
    A_col = 15.0  # in^2
    Iy_col = 100.0
    Iz_col = 100.0
    J_col = 50.0

    A_beam = 12.0
    Iy_beam = 80.0
    Iz_beam = 250.0
    J_beam = 30.0

    # Add columns (vertical members)
    frame.add_frame("C1", 1, 5, E, A_col, Iy_col, Iz_col, J_col)
    frame.add_frame("C2", 2, 6, E, A_col, Iy_col, Iz_col, J_col)
    frame.add_frame("C3", 3, 7, E, A_col, Iy_col, Iz_col, J_col)
    frame.add_frame("C4", 4, 8, E, A_col, Iy_col, Iz_col, J_col)

    # Add roof beams
    frame.add_frame("B1", 5, 6, E, A_beam, Iy_beam, Iz_beam, J_beam)
    frame.add_frame("B2", 6, 7, E, A_beam, Iy_beam, Iz_beam, J_beam)
    frame.add_frame("B3", 7, 8, E, A_beam, Iy_beam, Iz_beam, J_beam)
    frame.add_frame("B4", 8, 5, E, A_beam, Iy_beam, Iz_beam, J_beam)

    # Fixed boundary supports at all 4 column bases
    for n in [1, 2, 3, 4]:
        frame.add_support(n, [1, 1, 1, 1, 1, 1])

    # Apply lateral wind/seismic load in X and Y, and gravity in Z at roof node 5
    frame.add_node_load(5, [20.0, 15.0, -30.0, 0.0, 0.0, 0.0])
    frame.add_node_load(6, [0.0, 0.0, -30.0, 0.0, 0.0, 0.0])
    frame.add_node_load(7, [0.0, 0.0, -30.0, 0.0, 0.0, 0.0])
    frame.add_node_load(8, [0.0, 0.0, -30.0, 0.0, 0.0, 0.0])

    disp, reac = frame.solve()

    results = Results(frame)
    print("=== 3D Portal Space Frame Report ===")
    results.create_report(print_report=True)


if __name__ == "__main__":
    main()
