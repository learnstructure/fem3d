"""Simply supported square slab under a uniform load, modeled with fem3d shell elements.

Compares the center deflection and the center moment with the Kirchhoff plate solution of Timoshenko and
Woinowsky-Krieger (Table 8, nu = 0.3): w = 0.00406 q a^4 / D and M = 0.0479 q a^2. The shell (MITC4) includes shear
deformation, which adds about 0.7 percent to the deflection of this slab (span / thickness = 20); the moment of a
simply supported plate is the same in both theories.
"""

import numpy as np

from fem3d import ElasticMaterial, Node, ShellElement, ShellLoad, Structure

a, t, q, n = 4.0, 0.2, 5.0, 8  # m, m, kPa, elements per side
concrete = ElasticMaterial(E=25e6, nu=0.3)  # kN/m2

s = Structure()
nodes = {}
for j in range(n + 1):
    for i in range(n + 1):
        node = Node(nid=j * (n + 1) + i, x=a * i / n, y=a * j / n, z=0.0)
        on_x, on_y = i in (0, n), j in (0, n)
        # simple support: w = 0 and no rotation about the edge normal; two corners hold the in-plane rigid motion
        node.set_support(ux_fixed=(i, j) == (0, 0), uy_fixed=(i, j) in ((0, 0), (n, 0)), uz_fixed=on_x or on_y,
                         rx_fixed=on_x, ry_fixed=on_y)
        nodes[(i, j)] = node
        s.add_node(node)

for j in range(n):
    for i in range(n):
        slab = ShellElement(f"S{j}_{i}", [nodes[(i, j)], nodes[(i + 1, j)], nodes[(i + 1, j + 1)], nodes[(i, j + 1)]],
                            concrete, thickness=t)
        s.add_shell(slab)
        s.add_load(ShellLoad(slab, pressure=-q, direction="z"))

s.solve()

D = concrete.E * t**3 / (12 * (1 - concrete.nu**2))
w_center = s.disp[nodes[(n // 2, n // 2)].dofs[2]]
# M11 at the corner of the four central elements, averaged over them
corner = [s.shell_resultants()[f"S{j}_{i}"]["corners"][k, 3]
          for (i, j, k) in ((n // 2 - 1, n // 2 - 1, 2), (n // 2, n // 2 - 1, 3), (n // 2, n // 2, 0), (n // 2 - 1, n // 2, 1))]
print(f"center deflection {w_center:.6f} m   (Kirchhoff {-0.00406 * q * a**4 / D:.6f} m)")
print(f"center moment M11 {np.mean(corner):.3f} kN-m/m   (Kirchhoff {0.0479 * q * a**2:.3f} kN-m/m)")
