"""Verification of the MITC4 plate bending element (fem3d/SHELL_PLAN.md, milestone M4).

Pass/fail values are closed-form solutions: Navier series of the Kirchhoff plate (Timoshenko and Woinowsky-Krieger
Ch. 5) and of the first-order shear deformation plate (Reddy 2006, Eqs. 10.2.10 to 10.2.14), Timoshenko's tables for
clamped plates, and beam theory. SAP2000 values are not used as tolerances (SHELL_PLAN.md, Decisions).

The element is tested on its own, in its plane, with a small assembler; fem3d's Structure accepts shell elements from
milestone M5.
"""

import numpy as np
import pytest
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from fem3d.elements import shell_plate as sp


# --------------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------------
def grid(xs, ys):
    """Nodes and counterclockwise quadrilaterals of a structured mesh."""
    nodes = np.array([[x, y] for y in ys for x in xs], float)
    nx = len(xs)
    quads = np.array(
        [[j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i] for j in range(len(ys) - 1) for i in range(nx - 1)]
    )
    return nodes, quads


def element_dofs(quads):
    return (3 * np.asarray(quads)[:, :, None] + np.arange(3)).reshape(len(quads), 12)


def assemble(nodes, quads, Cb, Cs, pressure=0.0):
    """Sparse stiffness and consistent pressure loads, 3 DOFs per node (w, theta_x, theta_y)."""
    dofs = element_dofs(quads)
    Ke = sp.plate_stiffness(nodes[quads], Cb, Cs)
    rows = np.repeat(dofs, 12, axis=1).ravel()
    cols = np.tile(dofs, (1, 12)).ravel()
    K = sps.coo_matrix((Ke.ravel(), (rows, cols)), shape=(3 * len(nodes),) * 2).tocsc()
    F = np.zeros(3 * len(nodes))
    np.add.at(F, dofs.ravel(), sp.plate_pressure_loads(nodes[quads], pressure).ravel())
    return K, F


def solve(K, F, fixed, prescribed=None):
    u = np.zeros(K.shape[0])
    for dof, value in (prescribed or {}).items():
        u[dof] = value
    fixed = np.union1d(np.asarray(sorted(fixed), int), np.asarray(list(prescribed or {}), int))
    free = np.setdiff1d(np.arange(K.shape[0]), fixed)
    K = sps.csc_matrix(K)
    rhs = F[free] - K[free][:, fixed] @ u[fixed]
    u[free] = spla.spsolve(K[free][:, free], rhs)
    return u


def isotropic(E, nu, h):
    return sp.plate_rigidities(E, nu, E / (2 * (1 + nu)), h)


SHAPES = {
    "square": [[0, 0], [1, 0], [1, 1], [0, 1]],
    "rectangle": [[0, 0], [3, 0], [3, 1], [0, 1]],
    "parallelogram": [[0, 0], [2, 0], [2.6, 1], [0.6, 1]],
    "general": [[0, 0], [2, 0.3], [1.6, 1.4], [-0.2, 1]],
}


def rigid_modes(xy):
    """w = a + b x + c y with beta_x = b = -theta_y and beta_y = c = theta_x."""
    x, y = np.asarray(xy, float).T
    modes = np.zeros((3, 12))
    modes[0, 0::3] = 1.0
    modes[1, 0::3], modes[1, 2::3] = x, -1.0
    modes[2, 0::3], modes[2, 1::3] = y, 1.0
    return modes


# --------------------------------------------------------------------------------------------------------------------
# single element
# --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("thickness", [0.5, 1e-3])
def test_free_element_has_three_rigid_modes(shape, thickness):
    xy = np.array(SHAPES[shape], float)
    K = sp.plate_stiffness(xy, *isotropic(30e6, 0.3, thickness))[0]
    assert np.allclose(K, K.T, rtol=0, atol=1e-13 * np.abs(K).max())
    w = np.linalg.eigvalsh(K)
    assert np.sum(np.abs(w) < 1e-10 * w[-1]) == 3
    for mode in rigid_modes(xy):
        assert np.abs(K @ mode).max() < 1e-9 * np.abs(K).max() * np.abs(mode).max()


def test_assumed_shear_of_the_2x2_element_is_bathe_eq_5_101():
    """Bathe Eq. 5.101 (element of geometry 2 x 2; Bathe numbers the corners 1 (+,+), 2 (-,+), 3 (-,-), 4 (+,-))."""
    xy = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    to_bathe = [2, 3, 0, 1]  # Bathe node k is node to_bathe[k-1] here
    rng = np.random.default_rng(1)
    u = rng.normal(size=12)
    w, tx, ty = (u[0::3][to_bathe], u[1::3][to_bathe], u[2::3][to_bathe])
    _, Bs, _ = sp.plate_strain_matrices(xy)
    for g, (r, s) in enumerate(sp.GAUSS_POINTS_2X2):
        g_xz = 0.5 * (1 + s) * ((w[0] - w[1]) / 2 + (ty[0] + ty[1]) / 2) + 0.5 * (1 - s) * (
            (w[3] - w[2]) / 2 + (ty[3] + ty[2]) / 2
        )
        g_yz = 0.5 * (1 + r) * ((w[0] - w[3]) / 2 - (tx[0] + tx[3]) / 2) + 0.5 * (1 - r) * (
            (w[1] - w[2]) / 2 - (tx[1] + tx[2]) / 2
        )
        assert np.allclose(Bs[0, g] @ u, [g_xz, g_yz], atol=1e-13)


def test_assumed_shear_of_a_general_element_is_bathe_eqs_5_102_to_5_105():
    xy = np.array(SHAPES["general"], float)
    to_bathe = [2, 3, 0, 1]
    xb, yb = xy[to_bathe, 0], xy[to_bathe, 1]
    rng = np.random.default_rng(2)
    u = rng.normal(size=12)
    w, tx, ty = (u[0::3][to_bathe], u[1::3][to_bathe], u[2::3][to_bathe])
    Ax, Bx, Cx = xb[0] - xb[1] - xb[2] + xb[3], xb[0] - xb[1] + xb[2] - xb[3], xb[0] + xb[1] - xb[2] - xb[3]
    Ay, By, Cy = yb[0] - yb[1] - yb[2] + yb[3], yb[0] - yb[1] + yb[2] - yb[3], yb[0] + yb[1] - yb[2] - yb[3]

    def side(i, j):  # bracketed terms of Eq. 5.103 for the side from Bathe node i to node j
        return (w[i] - w[j]) / 2 + (xb[i] - xb[j]) / 4 * (ty[i] + ty[j]) - (yb[i] - yb[j]) / 4 * (tx[i] + tx[j])

    _, Bs, _ = sp.plate_strain_matrices(xy)
    for g, (r, s) in enumerate(sp.GAUSS_POINTS_2X2):
        g_r = np.array([Ax + s * Bx, Ay + s * By]) / 4  # dx/dr, dy/dr
        g_s = np.array([Cx + r * Bx, Cy + r * By]) / 4  # dx/ds, dy/ds
        detJ = g_r[0] * g_s[1] - g_r[1] * g_s[0]
        gamma_rz = np.hypot(Cx + r * Bx, Cy + r * By) / (8 * detJ) * ((1 + s) * side(0, 1) + (1 - s) * side(3, 2))
        gamma_sz = np.hypot(Ax + s * Bx, Ay + s * By) / (8 * detJ) * ((1 + r) * side(0, 3) + (1 - r) * side(1, 2))
        alpha, beta = np.arctan2(g_r[1], g_r[0]), np.arctan2(g_s[1], g_s[0])
        expected = [
            gamma_rz * np.sin(beta) - gamma_sz * np.sin(alpha),
            -gamma_rz * np.cos(beta) + gamma_sz * np.cos(alpha),
        ]
        assert np.allclose(Bs[0, g] @ u, expected, atol=1e-12)


def test_stiffness_is_invariant_under_rigid_motion_and_cyclic_renumbering():
    xy = np.array(SHAPES["general"], float)
    Cb, Cs = isotropic(1.0, 0.3, 0.2)
    K = sp.plate_stiffness(xy, Cb, Cs)[0]
    a = 0.7
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    K_rot = sp.plate_stiffness(xy @ R.T + [3.0, -2.0], Cb, Cs)[0]
    T = np.zeros((12, 12))
    for i in range(4):
        T[3 * i, 3 * i] = 1.0
        T[3 * i + 1 : 3 * i + 3, 3 * i + 1 : 3 * i + 3] = R  # the rotation vector turns with the element
    assert np.allclose(K_rot, T @ K @ T.T, atol=1e-12 * np.abs(K).max())
    for shift in (1, 2, 3):
        order = np.roll(np.arange(4), -shift)
        dofs = (3 * order[:, None] + np.arange(3)).ravel()
        assert np.allclose(sp.plate_stiffness(xy[order], Cb, Cs)[0], K[np.ix_(dofs, dofs)], atol=1e-12 * np.abs(K).max())


def test_vectorized_matches_element_by_element_and_bad_input_is_refused():
    shapes = np.array([SHAPES[s] for s in ("square", "parallelogram", "general")], float)
    Cb, Cs = sp.plate_rigidities([1.0, 2.0, 3.0], [0.2, 0.25, 0.3], [0.4, 0.8, 1.2], [0.1, 0.2, 0.3])
    K = sp.plate_stiffness(shapes, Cb, Cs)
    for i in range(3):
        assert np.allclose(K[i], sp.plate_stiffness(shapes[i], Cb[i], Cs[i])[0])
    with pytest.raises(ValueError, match="counterclockwise"):
        sp.plate_stiffness(shapes[0][::-1], Cb[0], Cs[0])
    with pytest.raises(ValueError, match="thickness"):
        sp.plate_rigidities(1.0, 0.3, 0.4, 0.0)


def test_pressure_loads_sum_to_the_total_load():
    xy = np.array(SHAPES["general"], float)
    x, y = xy.T
    area = 0.5 * abs((x[2] - x[0]) * (y[3] - y[1]) - (x[3] - x[1]) * (y[2] - y[0]))
    loads = sp.plate_pressure_loads(xy, 2.5)[0]
    assert loads[0::3].sum() == pytest.approx(2.5 * area)
    assert np.all(loads[1::3] == 0) and np.all(loads[2::3] == 0)


# --------------------------------------------------------------------------------------------------------------------
# bending patch test: CSI Problem 2-001 (MacNeal and Harder 1985)
# --------------------------------------------------------------------------------------------------------------------
PATCH_NODES = np.array(
    [[0, 0], [0, 0.12], [0.04, 0.02], [0.08, 0.08], [0.18, 0.03], [0.16, 0.08], [0.24, 0], [0.24, 0.12]], float
)
PATCH_QUADS = np.array([[0, 6, 4, 2], [2, 4, 5, 3], [4, 6, 7, 5], [1, 3, 5, 7], [0, 2, 3, 1]])


@pytest.mark.parametrize("thickness", [0.001, 0.05])
def test_bending_patch_test_with_prescribed_edge_displacements(thickness):
    """Uz = (x^2 + x y + y^2)/2000, Rx = (y + x/2)/1000, Ry = -(x + y/2)/1000 at joints 1, 2, 7, 8. Constant
    curvature: M11 = M22 = 1.111E-07 and M12 = 3.333E-08 lb-in/in for t = 0.001 in, no transverse shear. The thicker
    plate checks that the assumed shear stays exactly zero."""
    E, nu = 1.0e6, 0.25
    Cb, Cs = isotropic(E, nu, thickness)
    K, F = assemble(PATCH_NODES, PATCH_QUADS, Cb, Cs)
    x, y = PATCH_NODES.T
    w, tx, ty = (x * x + x * y + y * y) / 2000, (y + x / 2) / 1000, -(x + y / 2) / 1000
    prescribed = {}
    for n in (0, 1, 6, 7):
        prescribed.update({3 * n: w[n], 3 * n + 1: tx[n], 3 * n + 2: ty[n]})
    u = solve(K, F, [], prescribed)
    interior = [2, 3, 4, 5]
    assert np.allclose(u[1::3][interior], tx[interior], atol=1e-14)
    assert np.allclose(u[2::3][interior], ty[interior], atol=1e-14)
    resultants = sp.plate_resultants(PATCH_NODES[PATCH_QUADS], Cb, Cs, u[element_dofs(PATCH_QUADS)])
    D = E * thickness**3 / (12 * (1 - nu**2))
    expected_M = D * 1e-3 * np.array([1 + nu, 1 + nu, (1 - nu) / 2])
    assert np.allclose(resultants[:, :, :3], expected_M, rtol=1e-8, atol=0)
    assert np.allclose(resultants[:, :, 3:], 0.0, atol=1e-10 * D)
    if thickness == 0.001:
        assert expected_M == pytest.approx([1.111e-7, 1.111e-7, 3.333e-8], rel=1e-3)


# --------------------------------------------------------------------------------------------------------------------
# closed forms for rectangular plates
# --------------------------------------------------------------------------------------------------------------------
def navier_center_deflection(a, b, E, nu, h, load, shear, terms=99, shear_factor=5 / 6):
    """Center deflection of a simply supported a x b plate under a unit uniform load or a unit central point load.

    Kirchhoff (Timoshenko and Woinowsky-Krieger Ch. 5, Navier): W_mn = Q_mn / (D (alpha^2 + beta^2)^2). First-order
    shear deformation (Reddy 2006, Eqs. 10.2.10 to 10.2.14, isotropic, no foundation): W_mn from the 3x3 system.
    Q_mn = 16 / (pi^2 m n) (odd m, n) for the uniform load, 4 / (a b) sin(m pi / 2) sin(n pi / 2) for the point load.
    """
    D = E * h**3 / (12 * (1 - nu**2))
    A = shear_factor * E / (2 * (1 + nu)) * h
    w = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            al, be = m * np.pi / a, n * np.pi / b
            Q = 16 / (np.pi**2 * m * n) if load == "uniform" else 4 / (a * b) * np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2)
            if shear:
                S = np.array(
                    [
                        [A * (al**2 + be**2), A * al, A * be],
                        [A * al, D * al**2 + (1 - nu) / 2 * D * be**2 + A, (1 + nu) / 2 * D * al * be],
                        [A * be, (1 + nu) / 2 * D * al * be, (1 - nu) / 2 * D * al**2 + D * be**2 + A],
                    ]
                )
                W = np.linalg.solve(S, [Q, 0.0, 0.0])[0]
            else:
                W = Q / (D * (al**2 + be**2) ** 2)
            w += W * np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2)
    return w


def test_navier_series_reproduce_reddy_table_10_2_1():
    """Isotropic square plate, nu = 0.25, uniform load, w_hat = w D / (q b^4) * 100 (19 terms in each direction)."""
    for b_over_h, expected in ((10, 0.4259), (20, 0.4111)):
        h = 1.0 / b_over_h
        D = h**3 / (12 * (1 - 0.25**2))
        w = navier_center_deflection(1.0, 1.0, 1.0, 0.25, h, "uniform", shear=True, terms=37)
        assert w * D * 100 == pytest.approx(expected, abs=5e-5)
    D = 0.01**3 / (12 * (1 - 0.25**2))
    assert navier_center_deflection(1.0, 1.0, 1.0, 0.25, 0.01, "uniform", shear=False, terms=37) * D * 100 == (
        pytest.approx(0.4062, abs=5e-5)
    )


def quarter_plate(a, b, nx, ny, E, nu, h, support, load):
    """Quarter of an a x b plate (CSI Problem 2-005): supported edges at x = 0 and y = 0, symmetry at x = a/2
    (theta_y = 0) and y = b/2 (theta_x = 0). Simple support is the 'hard' support of the Navier solutions (Reddy
    Fig. 10.2.1): w = 0 and zero rotation about the edge normal, which is the rotation that tilts the normal along
    the edge (theta_x on x = 0, theta_y on y = 0). Clamped edges fix w and both rotations. Unit uniform load, or unit
    central point load (a quarter of it on the quarter model)."""
    nodes, quads = grid(np.linspace(0, a / 2, nx + 1), np.linspace(0, b / 2, ny + 1))
    K, F = assemble(nodes, quads, *isotropic(E, nu, h), pressure=1.0 if load == "uniform" else 0.0)
    center = len(nodes) - 1
    if load == "point":
        F[3 * center] = 0.25
    x, y = nodes.T
    on_x0, on_y0 = np.isclose(x, 0), np.isclose(y, 0)
    fixed = set(3 * np.flatnonzero(on_x0 | on_y0))
    if support == "clamped":
        fixed |= set(3 * np.flatnonzero(on_x0 | on_y0) + 1) | set(3 * np.flatnonzero(on_x0 | on_y0) + 2)
    else:
        fixed |= set(3 * np.flatnonzero(on_x0) + 1) | set(3 * np.flatnonzero(on_y0) + 2)
    fixed |= set(3 * np.flatnonzero(np.isclose(x, a / 2)) + 2) | set(3 * np.flatnonzero(np.isclose(y, b / 2)) + 1)
    return solve(K, F, fixed)[3 * center]


# CSI Problem 2-005: E = 17,472,000 psi, nu = 0.3, t = 0.0001 in (span / thickness = 20,000)
E_2005, NU_2005, T_2005 = 17_472_000.0, 0.3, 1.0e-4
D_2005 = E_2005 * T_2005**3 / (12 * (1 - NU_2005**2))


@pytest.mark.parametrize("load", ["uniform", "point"])
def test_2005_simply_supported_square_plate_converges_to_navier(load):
    exact = navier_center_deflection(2.0, 2.0, E_2005, NU_2005, T_2005, load, shear=False, terms=199)
    coarse = quarter_plate(2.0, 2.0, 4, 4, E_2005, NU_2005, T_2005, "simple", load)
    fine = quarter_plate(2.0, 2.0, 16, 16, E_2005, NU_2005, T_2005, "simple", load)
    assert abs(fine / exact - 1) < 0.001
    assert abs(fine / exact - 1) < abs(coarse / exact - 1)
    # Timoshenko Table 8 / Eq. 147 as used by Problem 2-005: 0.00406 q a^4 / D and 0.01160 P a^2 / D
    assert exact * D_2005 / (16 if load == "uniform" else 4) == pytest.approx(
        0.00406 if load == "uniform" else 0.01160, abs=6e-6
    )


@pytest.mark.parametrize(
    "load, coefficient", [("uniform", 0.00126), ("point", 0.00560)], ids=["Table 35", "Table 37"]
)
def test_2005_clamped_square_plate_matches_timoshenko_tables(load, coefficient):
    """Timoshenko and Woinowsky-Krieger Tables 35 and 37 (nu = 0.3, three significant digits, so 1 percent)."""
    fine = quarter_plate(2.0, 2.0, 16, 16, E_2005, NU_2005, T_2005, "clamped", load)
    a2 = 4.0 if load == "point" else 16.0  # P a^2 / D or q a^4 / D with a = 2
    assert fine * D_2005 / a2 == pytest.approx(coefficient, rel=0.01)


@pytest.mark.parametrize("support", ["simple", "clamped"])
@pytest.mark.parametrize("load", ["uniform", "point"])
def test_2005_long_plate_converges(support, load):
    """2 x 10 in plate on square elements (quarter meshes 4x20, 8x40, 16x80). Simple support against the Navier
    series; clamped against Timoshenko Tables 35 and 37 for b/a = infinity (0.00260 q a^4 / D, 0.00725 P a^2 / D;
    at b/a = 2 they are already 0.00254 and 0.00722)."""
    results = [quarter_plate(2.0, 10.0, n, 5 * n, E_2005, NU_2005, T_2005, support, load) for n in (4, 8, 16)]
    if support == "simple":
        exact = navier_center_deflection(2.0, 10.0, E_2005, NU_2005, T_2005, load, shear=False, terms=199)
        tolerance = 0.005
    else:
        exact = (0.00260 * 16.0 if load == "uniform" else 0.00725 * 4.0) / D_2005
        tolerance = 0.01
    errors = [abs(r / exact - 1) for r in results]
    assert errors[-1] < tolerance
    assert errors[0] > errors[1] > errors[2]


# --------------------------------------------------------------------------------------------------------------------
# thick plates and shear locking
# --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("span_to_thickness", [5, 10, 20])
def test_thick_simply_supported_plate_converges_to_first_order_shear_deformation_navier(span_to_thickness):
    h = 2.0 / span_to_thickness
    exact = navier_center_deflection(2.0, 2.0, 1.0, 0.3, h, "uniform", shear=True)
    assert quarter_plate(2.0, 2.0, 8, 8, 1.0, 0.3, h, "simple", "uniform") == pytest.approx(exact, rel=0.002)


def test_no_shear_locking_as_the_plate_becomes_thin():
    """On a fixed 8x8 quarter mesh, the ratio to the first-order shear deformation solution stays the same from
    span/thickness 10 to 100,000 (a locking element would stiffen without bound)."""
    ratios = []
    for span_to_thickness in (10, 1e2, 1e3, 1e4, 1e5):
        h = 2.0 / span_to_thickness
        exact = navier_center_deflection(2.0, 2.0, 1.0, 0.3, h, "uniform", shear=True)
        ratios.append(quarter_plate(2.0, 2.0, 8, 8, 1.0, 0.3, h, "simple", "uniform") / exact)
    assert np.ptp(ratios) < 0.001
    assert all(abs(r - 1) < 0.002 for r in ratios)


# --------------------------------------------------------------------------------------------------------------------
# CSI Problem 2-002: straight cantilever, out-of-plane cases
# --------------------------------------------------------------------------------------------------------------------
def cantilever_out_of_plane(n_long, n_wide):
    """Strip 6 x 0.2 in, t = 0.1 in, E = 1e7 psi, nu = 0.3; root: w and the bending rotation fixed (Problem 2-002
    restrains Uy and Rz at joints 1 and 8; the twist rotation is free). Unit tip load and unit tip moment, spread
    over the tip nodes with trapezoidal weights."""
    E, nu = 1.0e7, 0.3
    nodes, quads = grid(np.linspace(0, 6, n_long + 1), np.linspace(0, 0.2, n_wide + 1))
    K, _ = assemble(nodes, quads, *isotropic(E, nu, 0.1))
    nx = n_long + 1
    root = np.arange(n_wide + 1) * nx
    tip = root + n_long
    weights = np.full(n_wide + 1, 1.0 / n_wide)
    weights[[0, -1]] /= 2
    fixed = set(3 * root) | set(3 * root + 2)
    F = np.zeros(K.shape[0])
    F[3 * tip] = weights
    deflection = solve(K, F, fixed)[3 * tip].mean()
    F = np.zeros(K.shape[0])
    F[3 * tip + 2] = weights
    rotation = solve(K, F, fixed)[3 * tip + 2].mean()
    return deflection, rotation


def test_2002_out_of_plane_load_and_moment():
    """Hand calculations of Problem 2-002: tip deflection P L^3 / (3 E I) + P L / (k G A) = 0.4321 in and tip
    rotation M L / (E I) = 0.036 rad, I = 0.2 x 0.1^3 / 12. The strip converges from below to within 0.2 percent."""
    I = 0.2 * 0.1**3 / 12
    hand_deflection = 6.0**3 / (3 * 1.0e7 * I) + 6.0 / (5 / 6 * 1.0e7 / 2.6 * 0.02)
    assert hand_deflection == pytest.approx(0.4321, abs=5e-5)
    results = [cantilever_out_of_plane(n, m) for n, m in ((6, 1), (24, 4), (48, 8))]
    errors = [abs(d / hand_deflection - 1) for d, _ in results]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.002
    for _, rotation in results:
        assert rotation == pytest.approx(6.0 / (1.0e7 * I), rel=0.002)
