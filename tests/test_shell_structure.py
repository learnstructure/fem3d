"""The flat shell in fem3d's Structure (fem3d/SHELL_PLAN.md, milestone M5).

Pass/fail values are closed forms (patch tests, plate frequency, load totals) or the reference values of the standard
problems (Scordelis-Lo roof, CSI Problem 2-006; twisted beam, Problem 2-004) as MacNeal and Harder (1985) give them,
with the tolerances stated in each test.
"""

import math

import numpy as np
import pytest

from struct_core.geometry import rotate_shell_resultants

from fem3d import ElasticMaterial, Node, PointLoad, Structure
from fem3d.elements.shell import ShellElement
from fem3d.loads import ShellLoad


def rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


Q = rotation([0.3, -0.5, 0.8], 1.1)  # an arbitrary rigid rotation


def structure_from(points, quads, material, thickness, supports=(), **shell_kw):
    s = Structure()
    nodes = [Node(nid=i, x=p[0], y=p[1], z=p[2]) for i, p in enumerate(points)]
    for n in nodes:
        s.add_node(n)
    for nid, flags in supports:
        nodes[nid].set_support(*flags)
    for k, q in enumerate(quads):
        s.add_shell(ShellElement(k, [nodes[i] for i in q], material, thickness, **shell_kw))
    return s, nodes


# --------------------------------------------------------------------------------------------------------------------
# one element
# --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("warp", [0.0, 0.15], ids=["flat", "warped"])
@pytest.mark.parametrize("rotated", [False, True], ids=["xy plane", "rotated"])
def test_free_shell_has_six_rigid_modes(warp, rotated):
    """Six zero eigenvalues, and the six rigid motions give no forces; for a warped element this checks the sign of
    the rigid links to the mean plane (Wilson Eq. 10.2)."""
    pts = np.array([(0, 0, warp), (2, 0.3, -warp), (1.6, 1.4, warp), (-0.2, 1, -warp)], float)
    if rotated:
        pts = pts @ Q.T + [5.0, -3.0, 2.0]
    s, _ = structure_from(pts, [[0, 1, 2, 3]], ElasticMaterial(E=30e6, nu=0.3), 0.1)
    s.number_dofs()
    K = s.shell_batch().global_stiffness()[0]
    assert np.allclose(K, K.T, atol=1e-13 * np.abs(K).max())
    w = np.linalg.eigvalsh(K)
    assert np.sum(np.abs(w) < 1e-9 * w[-1]) == 6
    for k in range(6):
        u = np.zeros(24)
        for c in range(4):
            if k < 3:
                u[6 * c + k] = 1.0
            else:
                omega = np.eye(3)[k - 3]
                u[6 * c : 6 * c + 3] = np.cross(omega, pts[c])
                u[6 * c + 3 : 6 * c + 6] = omega
        assert np.abs(K @ u).max() < 1e-9 * np.abs(K).max() * np.abs(u).max()


def test_ids_and_inputs_are_checked():
    s, nodes = structure_from([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [[0, 1, 2, 3]],
                              ElasticMaterial(E=1.0), 0.1)
    with pytest.raises(ValueError, match="already used"):
        s.add_shell(ShellElement(0, nodes, ElasticMaterial(E=1.0), 0.1))
    from fem3d import TrussElement, Section

    with pytest.raises(ValueError, match="already used"):
        s.add_element(TrussElement(0, nodes[0], nodes[1], Section(A=1.0, I2=1.0, I3=1.0, J=1.0,
                                                                  material=ElasticMaterial(E=1.0))))
    with pytest.raises(ValueError, match="behavior"):
        ShellElement(5, nodes, ElasticMaterial(E=1.0), 0.1, behavior="slab")
    other, other_nodes = structure_from([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [[0, 1, 2, 3]],
                                        ElasticMaterial(E=1.0), 0.1)
    s.add_load(ShellLoad(other.shells[0], 1.0))
    with pytest.raises(ValueError, match="not in this structure"):
        s.solve()


# --------------------------------------------------------------------------------------------------------------------
# patch test in an arbitrary plane (CSI Problem 2-001 mesh)
# --------------------------------------------------------------------------------------------------------------------
PATCH = np.array([[0, 0], [0, 0.12], [0.04, 0.02], [0.08, 0.08], [0.18, 0.03], [0.16, 0.08], [0.24, 0], [0.24, 0.12]])
PATCH_QUADS = [[0, 6, 4, 2], [2, 4, 5, 3], [4, 6, 7, 5], [1, 3, 5, 7], [0, 2, 3, 1]]


@pytest.mark.parametrize("rotated", [False, True], ids=["xy plane", "rotated"])
def test_patch_test_with_edge_loads_in_any_plane(rotated):
    """Uniform tension sigma and uniform moment M_xx = m applied as consistent edge loads on x = 0 and x = 0.24, one
    corner fully fixed, every other DOF free. In the axes of the patch (x along the load), every Gauss point has
    F11 = sigma t, M11 = m and zero for the six other resultants. The patch is then turned and moved rigidly in
    space, so the local axes, the transformations, and the results are all exercised."""
    E, nu, t, sigma, m, h = 1.0e6, 0.25, 0.001, 1000.0, 2.0e-4, 0.12
    R = Q if rotated else np.eye(3)
    pts = np.column_stack([PATCH, np.zeros(8)]) @ R.T + (np.array([1.0, 2.0, 3.0]) if rotated else 0.0)
    s, nodes = structure_from(pts, PATCH_QUADS, ElasticMaterial(E=E, nu=nu), t, supports=[(0, [True] * 6)])
    ex, ey = R[:, 0], R[:, 1]
    # the moment vector of M_xx = m on the face x = 0.24 is -m e_y per unit length (CSI §10.18: M = -int z sigma)
    for nid, side in ((0, -1), (1, -1), (6, 1), (7, 1)):
        f = side * sigma * t * h / 2 * ex
        mom = -side * m * h / 2 * ey
        s.add_load(PointLoad(nodes[nid], *f, *mom))
    s.solve()
    batch = s.shell_batch()
    gauss = np.array([r["gauss_points"] for r in s.shell_resultants().values()])
    flip = batch.axes[:, 2] @ R[:, 2] < 0
    in_patch_axes = rotate_shell_resultants(gauss, batch.axes[:, None], ex, flip[:, None])
    expected = np.array([sigma * t, 0, 0, m, 0, 0, 0, 0])
    assert np.allclose(in_patch_axes, expected, rtol=0, atol=1e-8 * np.abs(expected).max())


# --------------------------------------------------------------------------------------------------------------------
# Scordelis-Lo roof (CSI Problem 2-006)
# --------------------------------------------------------------------------------------------------------------------
def scordelis_lo(n, transform=None, renumber=0, axis_angle=0.0):
    """Quarter roof, n x n mesh: R = 25 ft, L = 50 ft, 40 degrees, t = 0.25 ft, E = 4.32e8 psf, nu = 0, 90 psf
    gravity. Diaphragm at y = 0 (ux = uz = 0); symmetry at y = 25 (uy = rx = rz = 0) and at the crown x = 0
    (ux = ry = rz = 0). Returns the structure and the free-edge midspan node."""
    R, L, phi0 = 25.0, 50.0, np.radians(40.0)
    pts, sup = [], []
    for i in range(n + 1):
        for j in range(n + 1):
            phi = phi0 * i / n
            pts.append((R * np.sin(phi), L / 2 * j / n, R * np.cos(phi)))
            f = [False] * 6
            if j == 0:
                f[0] = f[2] = True
            if j == n:
                f[1] = f[3] = f[5] = True
            if i == 0:
                f[0] = f[4] = f[5] = True
            sup.append(f)
    pts = np.array(pts)
    if transform is not None:
        pts = pts @ transform.T
        sup = [[True] * 6 if any(f) else f for f in sup]  # rotated symmetry planes are not axis-aligned
    quads = []
    for i in range(n):
        for j in range(n):
            q = [i * (n + 1) + j, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1]
            quads.append(q[renumber:] + q[:renumber])
    s, nodes = structure_from(pts, quads, ElasticMaterial(E=4.32e8, nu=0.0), 0.25,
                              supports=list(enumerate(sup)), axis_angle=axis_angle)
    gravity = np.array([0.0, 0.0, -1.0]) if transform is None else transform @ [0.0, 0.0, -1.0]
    for sh in s.shells.values():
        for axis in range(3):
            if abs(gravity[axis]) > 0:
                s.add_load(ShellLoad(sh, 90.0 * gravity[axis], "xyz"[axis]))
    return s, nodes[(n + 1) * (n + 1) - 1]


def test_scordelis_lo_roof_converges():
    """Vertical deflection at the middle of the free edge. MacNeal and Harder (1985) normalize with 0.3024 ft and
    note that most elements converge slightly below the 0.3086 ft of Scordelis and Lo; here 4x4, 8x8, 16x16 give
    0.3160, 0.3039, 0.3017 (32x32: 0.3014), within 0.5 percent of 0.3024 from 8x8 on and decreasing monotonically."""
    values = []
    for n in (4, 8, 16):
        s, tip = scordelis_lo(n)
        s.solve()
        values.append(-s.disp[tip.dofs[2]])
    assert values[0] > values[1] > values[2]
    assert values[1] == pytest.approx(0.3024, rel=0.005)
    assert values[2] == pytest.approx(0.3024, rel=0.005)


def test_results_are_invariant_under_rigid_rotation_renumbering_and_axis_angle():
    """The 4x4 roof (all supports fixed where any was, so that the rotated copy keeps them): rotating the model
    rotates the displacements; renumbering the corners cyclically permutes the corner results; an axis angle of
    30 degrees rotates the resultants (struct_core.geometry.rotate_shell_resultants) and leaves the displacements."""
    base, _ = scordelis_lo(4, transform=np.eye(3))
    base.solve()
    rb = base.shell_resultants()
    axes_b = base.shell_batch().axes

    turned, _ = scordelis_lo(4, transform=Q)
    turned.solve()
    for nid in base.nodes:
        db, dt = base.disp[base.nodes[nid].dofs], turned.disp[turned.nodes[nid].dofs]
        assert np.allclose(Q @ db[:3], dt[:3], atol=1e-9) and np.allclose(Q @ db[3:], dt[3:], atol=1e-9)
    axes_t = turned.shell_batch().axes
    rt = turned.shell_resultants()
    for i, sid in enumerate(base.shells):
        a = rotate_shell_resultants(rb[sid]["corners"], axes_b[i], axes_b[i, 0])
        b = rotate_shell_resultants(rt[sid]["corners"], axes_t[i], Q @ axes_b[i, 0])
        assert np.allclose(a, b, atol=1e-6 * np.abs(a).max())

    for shift in (1, 2, 3):
        shifted, _ = scordelis_lo(4, transform=np.eye(3), renumber=shift)
        shifted.solve()
        assert np.allclose(shifted.disp, base.disp, atol=1e-12)
        rs = shifted.shell_resultants()
        for sid in base.shells:
            assert np.allclose(rs[sid]["corners"], np.roll(rb[sid]["corners"], -shift, axis=0), atol=1e-6)

    angled, _ = scordelis_lo(4, transform=np.eye(3), axis_angle=30.0)
    angled.solve()
    assert np.allclose(angled.disp, base.disp, atol=1e-12)
    axes_a = angled.shell_batch().axes
    ra = angled.shell_resultants()
    for i, sid in enumerate(base.shells):
        expected = rotate_shell_resultants(rb[sid]["corners"], axes_b[i], axes_a[i, 0])
        assert np.allclose(ra[sid]["corners"], expected, atol=1e-6 * np.abs(expected).max())


# --------------------------------------------------------------------------------------------------------------------
# twisted beam (CSI Problem 2-004)
# --------------------------------------------------------------------------------------------------------------------
def twisted_beam(n_long, n_wide, load):
    """L = 12, width 1.1, t = 0.32, E = 29e6, nu = 0.22, twisted 90 degrees from the root (width along Z) to the
    tip (width along Y), root fixed, unit tip load along Y (in-plane at the tip) or Z (out-of-plane)."""
    L, W = 12.0, 1.1
    pts = []
    for i in range(n_long + 1):
        th = np.pi / 2 * (1 - i / n_long)
        for j in range(n_wide + 1):
            w = -W / 2 + W * j / n_wide
            pts.append((L * i / n_long, w * np.cos(th), w * np.sin(th)))
    quads = [[i * (n_wide + 1) + j, (i + 1) * (n_wide + 1) + j, (i + 1) * (n_wide + 1) + j + 1, i * (n_wide + 1) + j + 1]
             for i in range(n_long) for j in range(n_wide)]
    s, nodes = structure_from(pts, quads, ElasticMaterial(E=29e6, nu=0.22), 0.32,
                              supports=[(j, [True] * 6) for j in range(n_wide + 1)])
    tip = nodes[-(n_wide + 1):]
    weights = np.full(n_wide + 1, 1.0 / n_wide)
    weights[[0, -1]] /= 2
    for node, wgt in zip(tip, weights):
        s.add_load(PointLoad(node, fy=wgt if load == "in" else 0.0, fz=wgt if load == "out" else 0.0))
    s.solve()
    return np.mean([s.disp[n.dofs[1 if load == "in" else 2]] for n in tip])


@pytest.mark.parametrize("mesh", [(12, 2), (24, 4)], ids=["12x2", "24x4"])
def test_twisted_beam(mesh):
    """MacNeal and Harder (1985) Table 3: 0.005424 (in-plane) and 0.001754 (out-of-plane), from beam theory; CSI's
    hand calculation gives 0.005429 and 0.001749. Each element is warped by 7.5 degrees. Results are within 1.5
    percent (12x2: 0.005445, 0.001772; 48x8: 0.005480, 0.001771); the good elements of MacNeal and Harder's Table 10
    are within 2 percent on the 12x2 mesh."""
    assert twisted_beam(*mesh, "in") == pytest.approx(0.005424, rel=0.015)
    assert twisted_beam(*mesh, "out") == pytest.approx(0.001754, rel=0.015)


# --------------------------------------------------------------------------------------------------------------------
# loads, self-weight, mass, and dynamics
# --------------------------------------------------------------------------------------------------------------------
def flat_plate(n, a=2.0, h=0.02, E=1.0e7, nu=0.3, rho=1.0e-3, unit_weight=None, supported=True):
    pts = [(a * i / n, a * j / n, 0.0) for j in range(n + 1) for i in range(n + 1)]
    quads = [[j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i]
             for j in range(n) for i in range(n)]
    sup = []
    if supported:
        for k, (x, y, _) in enumerate(pts):
            f = [False] * 6
            on_x, on_y = np.isclose(x, 0) or np.isclose(x, a), np.isclose(y, 0) or np.isclose(y, a)
            if on_x or on_y:
                f[2] = True
                f[3] = f[3] or on_x  # hard simple support: no rotation about the edge normal
                f[4] = f[4] or on_y
            sup.append((k, f))
        sup[0][1][0] = sup[0][1][1] = True  # in-plane rigid motion
        sup[n][1][1] = True
    mat = ElasticMaterial(E=E, nu=nu, rho=rho, unit_weight=unit_weight)
    return structure_from(pts, quads, mat, h, supports=sup)


def test_uniform_loads_and_self_weight_sum_to_the_applied_totals():
    from struct_load.engine.self_weight import apply_self_weight

    pts = np.array([(0, 0, 0), (2, 0, 0.4), (2.2, 1.5, 0.6), (0, 1.2, 0.1)])  # sloped and slightly warped
    s, _ = structure_from(pts, [[0, 1, 2, 3]], ElasticMaterial(E=1.0, nu=0.2, unit_weight=24.0), 0.2)
    s.number_dofs()
    area = s.shell_batch().areas()[0]
    normal = s.shell_batch().axes[0, 2]
    s.add_load(ShellLoad(s.shells[0], 5.0, "local_3"))
    s.add_load(ShellLoad(s.shells[0], -3.0, "y"))
    apply_self_weight(s)
    sw = [ld for ld in s.loads if isinstance(ld, ShellLoad) and ld.is_self_weight]
    assert len(sw) == 1 and sw[0].direction == "z" and sw[0].pressure == pytest.approx(-24.0 * 0.2)
    apply_self_weight(s)  # not applied twice
    assert len([ld for ld in s.loads if isinstance(ld, ShellLoad) and ld.is_self_weight]) == 1
    s.assemble_loads()
    total = s.F.reshape(-1, 6)[:, :3].sum(axis=0)
    expected = area * (5.0 * normal + np.array([0.0, -3.0, -24.0 * 0.2]))
    assert np.allclose(total, expected, rtol=1e-12)


@pytest.mark.parametrize("lumped", [True, False])
def test_mass_totals_and_plate_frequency(lumped):
    """Total translational mass rho t A in each direction (rotational mass zero, CSI §10.11). The first frequency of
    a simply supported square plate (a/h = 100) against the Kirchhoff closed form
    omega_11 = 2 pi^2 / a^2 sqrt(D / (rho h)) (Timoshenko and Woinowsky-Krieger; Reddy Ch. 9). Lumped mass converges
    from below (0.9913 and 0.9976 of exact on 8x8 and 16x16), consistent mass from above (1.0171 and 1.0040)."""
    a, h, E, nu, rho = 2.0, 0.02, 1.0e7, 0.3, 1.0e-3
    D = E * h**3 / (12 * (1 - nu**2))
    exact = 2 * np.pi**2 / a**2 * np.sqrt(D / (rho * h))
    ratios = []
    for n in (8, 16):
        s, _ = flat_plate(n, a, h, E, nu, rho)
        s.solve()
        s.assemble_mass_matrix(lumped=lumped)
        M = s.M.toarray()
        for d in range(6):
            idx = np.arange(d, s.neq, 6)
            expected = rho * h * a * a if d < 3 else 0.0
            assert M[np.ix_(idx, idx)].sum() == pytest.approx(expected, abs=1e-12)
        ratios.append(s.modal_analysis(num_modes=1, lumped=lumped)["omega"][0] / exact)
    if lumped:
        assert ratios[0] < ratios[1] < 1.0
    else:
        assert ratios[0] > ratios[1] > 1.0
    assert ratios[1] == pytest.approx(1.0, abs=0.005)


def test_response_spectrum_combines_shell_resultants():
    """With one mode the combined resultants are the absolute values of the modal ones."""
    s, _ = flat_plate(4)
    s.solve()
    out = s.response_spectrum_analysis(Sa=[2.0], n_modes=1, direction="z", combination="SRSS")
    combined = out["combined_shell_resultants"]
    assert set(combined) == set(s.shells)
    modal = np.zeros(s.neq)
    modal[s.free_dofs] = out["modal_displacements"][:, 0]
    gauss, corners = s.shell_batch().resultants(modal)
    for i, sid in enumerate(s.shell_batch().ids):
        assert np.allclose(combined[sid]["gauss_points"], np.abs(gauss[i]))
        assert np.allclose(combined[sid]["corners"], np.abs(corners[i]))
    assert max(np.abs(c["gauss_points"]).max() for c in combined.values()) > 0


def test_twist_of_a_thick_strip_converges_to_reissner_mindlin_torsion():
    """CSI Problem 2-002, load case 4: strip 6 x 0.2 in, t = 0.1 in, E = 1e7 psi, nu = 0.3, root fixed, unit torque.

    Closed form of a Reissner-Mindlin strip of width b with free edges under a twist rate a (Bathe Eqs. 5.88 to 5.97
    with w = a x y, beta_y = a x): moment equilibrium d M12 / d y = V13 with M12 = (G t^3 / 12)(d beta_x / dy + a)
    and V13 = k G t (a y - beta_x) gives beta_x = a y + C sinh(mu y), mu^2 = 12 k / t^2; M12 = 0 at the free edges
    fixes C, and the torque (twisting moments plus the couple of the transverse shears) is G a J with
    J = (b t^3 / 3)(1 - tanh(mu b / 2) / (mu b / 2)). For b / t = 2 this is 0.2284 b t^3, within 0.3 percent of the
    Saint-Venant constant 0.229 b t^3 of CSI's hand calculation (0.00341 in), and well below the shell-theory value
    b t^3 / 3 (0.00234 in) that SAP2000 converges to (struct_link/tests/test_sap2000_shells.py)."""
    E, nu, L, b, t = 1.0e7, 0.3, 6.0, 0.2, 0.1
    G = E / (2 * (1 + nu))
    mu = math.sqrt(12 * 5 / 6) / t
    J = b * t**3 / 3 * (1 - math.tanh(mu * b / 2) / (mu * b / 2))
    exact = L / (G * J) * (b / 2)  # Y displacement of the top tip node under a unit torque

    def twist(n_long, n_deep):
        pts = [(L * i / n_long, 0.0, b * j / n_deep) for i in range(n_long + 1) for j in range(n_deep + 1)]
        nid = lambda i, j: i * (n_deep + 1) + j  # noqa: E731
        quads = [[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
                 for i in range(n_long) for j in range(n_deep)]
        s, nodes = structure_from(pts, quads, ElasticMaterial(E=E, nu=nu), t,
                                  supports=[(nid(0, j), [True] * 6) for j in range(n_deep + 1)])
        s.add_load(PointLoad(nodes[nid(n_long, 0)], fy=-5.0))
        s.add_load(PointLoad(nodes[nid(n_long, n_deep)], fy=5.0))
        s.solve()
        return s.disp[nodes[nid(n_long, n_deep)].dofs[1]]

    values = [twist(24, 4), twist(48, 8), twist(96, 16)]
    assert values[0] < values[1] < values[2]
    assert values[2] == pytest.approx(exact, rel=0.001)

