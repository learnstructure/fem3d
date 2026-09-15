"""
Example: 3D Cantilever Beam under Combined Axial, Torsion, and Biaxial Bending.

Analytical Verification:
Length L = 100 in, Section: Rectangular b = 4 in (along y'), h = 6 in (along z')
E = 30000 ksi, nu = 0.3 -> G = E / (2*(1+nu)) = 11538.46 ksi
A = 24 in^2
Iz = b * h^3 / 12 = 72 in^4 (strong axis)
Iy = h * b^3 / 12 = 32 in^4 (weak axis)
J = 4 * (4^3) * (1/3 - 0.21*(4/6)*(1 - 4^4/(12*6^4))) ~ 52.32 in^4

Loads at free tip (Node 2):
Fx = 24 kips (axial tension)
Fy = 5 kips (lateral shear bending about z')
Fz = 10 kips (vertical shear bending about y')
Mx = 50 kip-in (torsion)

Theoretical tip displacements:
ux = Fx * L / (E * A) = 24 * 100 / (30000 * 24) = 0.003333 in
uy = Fy * L^3 / (3 * E * Iz) = 5 * 100^3 / (3 * 30000 * 72) = 0.771605 in
rz = Fy * L^2 / (2 * E * Iz) = 5 * 100^2 / (2 * 30000 * 72) = 0.011574 rad
uz = Fz * L^3 / (3 * E * Iy) = 10 * 100^3 / (3 * 30000 * 32) = 3.472222 in
ry = -Fz * L^2 / (2 * E * Iy) = -10 * 100^2 / (2 * 30000 * 32) = -0.052083 rad
rx = Mx * L / (G * J)
"""

from fem3d import SimpleFrame, Results

def main():
    frame = SimpleFrame()

    # Fixed base at Node 1 (0, 0, 0), free tip at Node 2 (100, 0, 0)
    frame.add_node(1, 0.0, 0.0, 0.0)
    frame.add_node(2, 100.0, 0.0, 0.0)

    E = 30000.0
    nu = 0.3
    b = 4.0
    h = 6.0
    A = b * h
    Iz = b * (h**3) / 12.0
    Iy = h * (b**3) / 12.0
    # Torsional constant J
    a, c = max(b, h), min(b, h)
    J = a * (c**3) * (1.0 / 3.0 - 0.21 * (c / a) * (1.0 - (c**4) / (12.0 * a**4)))

    frame.add_frame(1, 1, 2, E=E, A=A, Iy=Iy, Iz=Iz, J=J, nu=nu)

    # Fully fixed base [ux, uy, uz, rx, ry, rz]
    frame.add_support(1, [1, 1, 1, 1, 1, 1])

    # Tip loads at node 2 [Fx, Fy, Fz, Mx, My, Mz]
    frame.add_node_load(2, [24.0, 5.0, 10.0, 50.0, 0.0, 0.0])

    disp, reac = frame.solve()

    results = Results(frame)
    print("=== 3D Cantilever Results ===")
    print("\nNode Displacements:")
    print(results.node_displacements())
    print("\nReactions:")
    print(results.reactions())
    print("\nElement End Forces:")
    print(results.element_forces())

    # Theoretical validation checks
    G = E / (2.0 * (1.0 + nu))
    expected_ux = 24.0 * 100.0 / (E * A)
    expected_uy = 5.0 * (100.0**3) / (3.0 * E * Iz)
    expected_uz = 10.0 * (100.0**3) / (3.0 * E * Iy)
    expected_rx = 50.0 * 100.0 / (G * J)

    u_tip = results.node_displacements().iloc[1]
    print(f"\nTip ux: computed={u_tip['ux']:.6f}, exact={expected_ux:.6f}")
    print(f"Tip uy: computed={u_tip['uy']:.6f}, exact={expected_uy:.6f}")
    print(f"Tip uz: computed={u_tip['uz']:.6f}, exact={expected_uz:.6f}")
    print(f"Tip rx: computed={u_tip['rx']:.6f}, exact={expected_rx:.6f}")


if __name__ == "__main__":
    main()
