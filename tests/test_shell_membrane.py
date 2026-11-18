"""Verification of the quadrilateral membrane with drilling rotations (fem3d/SHELL_PLAN.md, milestone M3).

Pass/fail values are closed-form solutions. Coarse-mesh SAP2000 values from the CSI verification manuals are not used
as tolerances (SHELL_PLAN.md, Decisions); the SAP2000 comparison under mesh refinement is milestone M6.

The element is tested on its own, in its plane, with a small dense assembler; fem3d's Structure accepts shell
elements from milestone M5.
"""

import numpy as np
import pytest

from fem3d.elements import shell_membrane as sm


# --------------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------------
def assemble(nodes, quads, D, thickness):
    """Dense stiffness of a plane mesh, 3 DOFs per node (u, v, theta)."""
    nodes = np.asarray(nodes, float)
    quads = np.asarray(quads)
    K = np.zeros((3 * len(nodes),) * 2)
    for q, k in zip(quads, sm.membrane_stiffness(nodes[quads], D, thickness)):
        dofs = (3 * q[:, None] + np.arange(3)).ravel()
        K[np.ix_(dofs, dofs)] += k
    return K


def solve(K, F, fixed, prescribed=None):
    """Displacements with the DOFs in ``fixed`` held at zero or at the values of ``prescribed``."""
    u = np.zeros(K.shape[0])
    for dof, value in (prescribed or {}).items():
        u[dof] = value
    fixed = np.union1d(np.asarray(fixed, int), np.asarray(list(prescribed or {}), int))
    free = np.setdiff1d(np.arange(K.shape[0]), fixed)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free] - K[np.ix_(free, fixed)] @ u[fixed])
    return u


def element_dofs(quads):
    return np.array([(3 * np.asarray(q)[:, None] + np.arange(3)).ravel() for q in quads])


def strip_mesh(x_bottom, x_top, depth, n_depth=1):
    """Quadrilaterals of a straight strip: bottom and top node rows at the given x, n_depth elements deep."""
    nx = len(x_bottom)
    nodes = []
    for k in range(n_depth + 1):
        f = k / n_depth
        nodes += [[x_bottom[i] * (1 - f) + x_top[i] * f, depth * f] for i in range(nx)]
    quads = [
        [k * nx + i, k * nx + i + 1, (k + 1) * nx + i + 1, (k + 1) * nx + i]
        for k in range(n_depth)
        for i in range(nx - 1)
    ]
    return np.array(nodes), quads, nx


def subdivide(x, m):
    return np.concatenate([np.linspace(x[i], x[i + 1], m + 1)[:-1] for i in range(len(x) - 1)] + [[x[-1]]])


def edge_loads(ys, traction):
    """Consistent nodal loads of a traction (force per unit length) along a straight edge with nodes at ys."""
    loads = np.zeros(len(ys))
    g, w = np.polynomial.legendre.leggauss(6)
    for a in range(len(ys) - 1):
        h = ys[a + 1] - ys[a]
        for gi, wi in zip(g, w):
            n1 = (1 - gi) / 2
            y = ys[a] + (1 - n1) * h
            loads[a] += wi * h / 2 * traction(y) * n1
            loads[a + 1] += wi * h / 2 * traction(y) * (1 - n1)
    return loads


def zero_eigenvalues(K, rel=1e-10):
    w = np.linalg.eigvalsh(K)
    return int(np.sum(np.abs(w) < rel * w[-1])), w


SHAPES = {
    "square": [[0, 0], [1, 0], [1, 1], [0, 1]],
    "rectangle": [[0, 0], [2, 0], [2, 1], [0, 1]],
    "parallelogram": [[0, 0], [2, 0], [2.6, 1], [0.6, 1]],
    "trapezoid": [[0, 0], [2, 0], [1.5, 1], [0.5, 1]],
    "general": [[0, 0], [2, 0.3], [1.6, 1.4], [-0.2, 1]],
    "large and offset": [[1000, 500], [1004, 500.5], [1003.5, 503], [999, 502]],
}


def rigid_modes(xy):
    x, y = np.asarray(xy, float).T
    modes = np.zeros((3, 12))
    modes[0, 0::3] = 1.0
    modes[1, 1::3] = 1.0
    modes[2, 0::3], modes[2, 1::3], modes[2, 2::3] = -y, x, 1.0  # counterclockwise rotation about local 3
    return modes


# --------------------------------------------------------------------------------------------------------------------
# single element
# --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
def test_free_element_has_three_rigid_modes_and_no_spurious_mode(shape):
    xy = np.array(SHAPES[shape], float)
    K = sm.membrane_stiffness(xy, sm.plane_stress_matrix(30e6, 0.25), 0.1)[0]
    assert np.allclose(K, K.T, rtol=0, atol=1e-12 * np.abs(K).max())
    n_zero, w = zero_eigenvalues(K)
    assert n_zero == 3
    assert w[3] > 1e-6 * w[-1]
    for mode in rigid_modes(xy):
        assert np.abs(K @ mode).max() < 1e-9 * np.abs(K).max() * np.abs(mode).max()


def test_drilling_rotation_is_counterclockwise():
    """A rigid rotation with clockwise nodal rotations strains the drilling penalty (Wilson Eq. 9.11 misprint)."""
    xy = np.array(SHAPES["general"], float)
    K = sm.membrane_stiffness(xy, sm.plane_stress_matrix(1.0, 0.25), 1.0)[0]
    wrong = rigid_modes(xy)[2]
    wrong[2::3] = -1.0
    assert wrong @ K @ wrong > 1e-3 * np.abs(K).max()


def test_penalties_remove_the_two_spurious_modes(monkeypatch):
    """Without the drilling penalty, equal nodal rotations cost no energy (Wilson §9.7, Fig. 9.3). Without the
    alternating-rotation penalty, a rectangle keeps the mode (+, -, +, -) (Cook et al. §3.10)."""
    D = sm.plane_stress_matrix(1.0, 0.25)
    monkeypatch.setattr(sm, "DRILLING_PENALTY_FACTOR", 0.0)
    K = sm.membrane_stiffness(np.array(SHAPES["general"], float), D, 1.0)[0]
    equal = np.zeros(12)
    equal[2::3] = 1.0
    assert zero_eigenvalues(K)[0] == 4
    assert abs(equal @ K @ equal) < 1e-12 * np.abs(K).max()

    monkeypatch.setattr(sm, "DRILLING_PENALTY_FACTOR", 0.025)
    monkeypatch.setattr(sm, "ROTATION_HOURGLASS_FACTOR", 0.0)
    K = sm.membrane_stiffness(np.array(SHAPES["rectangle"], float), D, 1.0)[0]
    alternating = np.zeros(12)
    alternating[2::3] = [1.0, -1.0, 1.0, -1.0]
    assert zero_eigenvalues(K)[0] == 4
    assert abs(alternating @ K @ alternating) < 1e-12 * np.abs(K).max()


def test_stiffness_is_invariant_under_rigid_motion_and_cyclic_renumbering():
    xy = np.array(SHAPES["general"], float)
    D = sm.plane_stress_matrix(1.0, 0.3)
    K = sm.membrane_stiffness(xy, D, 0.5)[0]
    # rotating and translating the element: K' = T K T^T, with T rotating (u, v) and leaving theta
    a = 0.7
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    K_rot = sm.membrane_stiffness(xy @ R.T + [3.0, -2.0], D, 0.5)[0]
    T = np.zeros((12, 12))
    for i in range(4):
        T[3 * i : 3 * i + 2, 3 * i : 3 * i + 2] = R
        T[3 * i + 2, 3 * i + 2] = 1.0
    assert np.allclose(K_rot, T @ K @ T.T, atol=1e-12 * np.abs(K).max())
    # starting the node list at another corner permutes K
    for shift in (1, 2, 3):
        order = np.roll(np.arange(4), -shift)
        K_shift = sm.membrane_stiffness(xy[order], D, 0.5)[0]
        dofs = (3 * order[:, None] + np.arange(3)).ravel()
        assert np.allclose(K_shift, K[np.ix_(dofs, dofs)], atol=1e-12 * np.abs(K).max())


def test_vectorized_matches_element_by_element():
    shapes = np.array([SHAPES[s] for s in ("square", "trapezoid", "general")], float)
    D = sm.plane_stress_matrix([1.0, 2.0, 3.0], [0.2, 0.25, 0.3])
    K = sm.membrane_stiffness(shapes, D, [0.1, 0.2, 0.3])
    for i in range(3):
        assert np.allclose(K[i], sm.membrane_stiffness(shapes[i], D[i], [0.1, 0.2, 0.3][i])[0])


def test_clockwise_or_reentrant_element_is_refused():
    D = sm.plane_stress_matrix(1.0, 0.25)
    with pytest.raises(ValueError, match="counterclockwise"):
        sm.membrane_stiffness(np.array(SHAPES["square"], float)[::-1], D, 1.0)
    with pytest.raises(ValueError, match="counterclockwise"):
        sm.membrane_stiffness(np.array([[0, 0], [2, 0], [0.3, 0.3], [0, 2]], float), D, 1.0)
    with pytest.raises(ValueError, match="thickness"):
        sm.membrane_stiffness(np.array(SHAPES["square"], float), D, 0.0)


# --------------------------------------------------------------------------------------------------------------------
# patch tests: CSI Problem 2-001 (MacNeal and Harder 1985)
# --------------------------------------------------------------------------------------------------------------------
PATCH_NODES = np.array(
    [[0, 0], [0, 0.12], [0.04, 0.02], [0.08, 0.08], [0.18, 0.03], [0.16, 0.08], [0.24, 0], [0.24, 0.12]], float
)
PATCH_QUADS = [[0, 6, 4, 2], [2, 4, 5, 3], [4, 6, 7, 5], [1, 3, 5, 7], [0, 2, 3, 1]]  # joints 1..8 -> 0..7
PATCH_BOUNDARY = [0, 1, 6, 7]  # joints 1, 2, 7, 8
PATCH_E, PATCH_NU, PATCH_T = 1.0e6, 0.25, 0.001


def patch_field(xy, rotation=0.0):
    """MacNeal-Harder displacement field (CSI Problem 2-001) plus a rigid rotation about local 3."""
    x, y = xy[:, 0], xy[:, 1]
    return (x + y / 2) / 1000 - rotation * y, (y + x / 2) / 1000 + rotation * x


@pytest.mark.parametrize("rotation", [0.0, 2.0e-3])
def test_patch_test_with_prescribed_edge_displacements(rotation):
    """Constant stress in every element; the free drilling rotations equal the rotation of the field."""
    D = sm.plane_stress_matrix(PATCH_E, PATCH_NU)
    K = assemble(PATCH_NODES, PATCH_QUADS, D, PATCH_T)
    ux, uy = patch_field(PATCH_NODES, rotation)
    prescribed = {}
    for n in PATCH_BOUNDARY:
        prescribed[3 * n], prescribed[3 * n + 1] = ux[n], uy[n]
    u = solve(K, np.zeros(K.shape[0]), [], prescribed)
    assert np.allclose(u[0::3], ux, rtol=0, atol=1e-12)
    assert np.allclose(u[1::3], uy, rtol=0, atol=1e-12)
    assert np.allclose(u[2::3], rotation, rtol=0, atol=1e-12)
    F = sm.membrane_resultants(PATCH_NODES[np.array(PATCH_QUADS)], D, PATCH_T, u[element_dofs(PATCH_QUADS)])
    # CSI Problem 2-001: sigma_xx = sigma_yy = 1333 lb/in2, sigma_xy = 400 lb/in2 (Timoshenko and Goodier)
    expected = PATCH_T * np.array([PATCH_E / (1 - PATCH_NU) * 1e-3, PATCH_E / (1 - PATCH_NU) * 1e-3, 400.0])
    assert np.allclose(F, expected, rtol=1e-9, atol=0)


def test_patch_test_with_edge_tractions_and_minimal_supports():
    """Uniform tension applied as consistent loads, three translations restrained, every rotation free."""
    D = sm.plane_stress_matrix(PATCH_E, PATCH_NU)
    K = assemble(PATCH_NODES, PATCH_QUADS, D, PATCH_T)
    sigma, height = 1000.0, 0.12
    F = np.zeros(K.shape[0])
    F[3 * 6] = F[3 * 7] = sigma * PATCH_T * height / 2  # joints 7 and 8 on x = 0.24
    u = solve(K, F, [0, 1, 3 * 1])  # joint 1: u, v; joint 2: u
    resultants = sm.membrane_resultants(PATCH_NODES[np.array(PATCH_QUADS)], D, PATCH_T, u[element_dofs(PATCH_QUADS)])
    assert np.allclose(resultants, [sigma * PATCH_T, 0.0, 0.0], rtol=0, atol=1e-9 * sigma * PATCH_T)
    assert np.allclose(u[2::3], 0.0, atol=1e-12)


# --------------------------------------------------------------------------------------------------------------------
# pure bending (higher-order patch test, Ibrahimbegovic and Frey 1992 §6.2; Wilson §9.9)
# --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("nu", [0.0, 0.25])
def test_pure_bending_of_rectangles_is_exact(nu):
    """Simply supported beam (L = 10, h = 1, E = 100, unit thickness) under unit end couples, six rectangles, all
    rotations free. Beam theory: midspan deflection M L^2 / (8 E I) = 1.5, end rotation M L / (2 E I) = 0.6, and
    sigma_x = M y / I at every Gauss point."""
    E, L, h, M = 100.0, 10.0, 1.0, 1.0
    I = h**3 / 12
    D = sm.plane_stress_matrix(E, nu)
    nodes, quads, nx = strip_mesh(np.linspace(0, L, 7), np.linspace(0, L, 7), h)
    K = assemble(nodes, quads, D, 1.0)
    F = np.zeros(K.shape[0])
    F[[3 * nx, 3 * (nx - 1)]] = M / h  # sagging couples: tension at the bottom, compression at the top
    F[[0, 3 * (2 * nx - 1)]] = -M / h
    u = solve(K, F, [0, 1, 3 * (nx - 1) + 1])
    assert u[3 * 3 + 1] == pytest.approx(-M * L**2 / (8 * E * I), rel=1e-9)
    assert np.allclose(u[[2, 3 * nx + 2]], -M * L / (2 * E * I), rtol=1e-9)
    assert np.allclose(u[[3 * (nx - 1) + 2, 3 * (2 * nx - 1) + 2]], M * L / (2 * E * I), rtol=1e-9)
    xy = nodes[np.array(quads)]
    F_gp = sm.membrane_resultants(xy, D, 1.0, u[element_dofs(quads)])
    y_gp = np.einsum("gk,nk->ng", np.array([sm._bilinear(r, s)[0] for r, s in sm.GAUSS_POINTS_2X2]), xy[:, :, 1])
    assert np.allclose(F_gp[:, :, 0], -M * (y_gp - h / 2) / I, rtol=0, atol=1e-9)
    assert np.allclose(F_gp[:, :, 1:], 0.0, atol=1e-9)


def test_pure_bending_by_drilling_moments():
    """The same beam loaded by moments on the drilling rotations (half the couple at each end node). Wilson §9.9
    reports nearly the same deflection as the force couple; the drilling moment is not a consistent load, so the
    deflection is within 1 percent of beam theory rather than exact."""
    E, L, h, M = 100.0, 10.0, 1.0, 1.0
    D = sm.plane_stress_matrix(E, 0.0)
    nodes, quads, nx = strip_mesh(np.linspace(0, L, 7), np.linspace(0, L, 7), h)
    K = assemble(nodes, quads, D, 1.0)
    F = np.zeros(K.shape[0])
    F[[2, 3 * nx + 2]] = -M / 2
    F[[3 * (nx - 1) + 2, 3 * (2 * nx - 1) + 2]] = M / 2
    u = solve(K, F, [0, 1, 3 * (nx - 1) + 1])
    assert -u[3 * 3 + 1] == pytest.approx(M * L**2 / (8 * E * h**3 / 12), rel=0.01)


# --------------------------------------------------------------------------------------------------------------------
# CSI Problem 2-002: straight cantilever, in-plane cases (MacNeal and Harder 1985)
# --------------------------------------------------------------------------------------------------------------------
E_2002, NU_2002, T_2002, DEPTH_2002, L_2002 = 1.0e7, 0.3, 0.1, 0.2, 6.0
I_2002 = T_2002 * DEPTH_2002**3 / 12
# Hand calculations of Problem 2-002 (unit-load method, shear area 5/6 A)
AXIAL_2002 = 1.0 * L_2002 / (E_2002 * T_2002 * DEPTH_2002)  # 3.0e-5 in
SHEAR_2002 = L_2002**3 / (3 * E_2002 * I_2002) + L_2002 / (5 / 6 * E_2002 / (2 * (1 + NU_2002)) * T_2002 * DEPTH_2002)
MOMENT_2002 = 1.0 * L_2002 / (E_2002 * I_2002) * DEPTH_2002 / 2  # 9.0e-4 in: Ux at the tip corners under 1 lb-in

MESH_A = (np.linspace(0, 6, 7), np.linspace(0, 6, 7))
MESH_B = (np.cumsum([0, 1.1, 0.8, 1.2, 0.8, 1.2, 0.9]), np.cumsum([0, 0.9, 1.2, 0.8, 1.2, 0.8, 1.1]))
MESH_C = (np.cumsum([0, 0.9, 1, 1, 1, 1, 1.1]), np.cumsum([0, 1.1, 1, 1, 1, 1, 0.9]))
# Model F: Model C with each trapezoidal end element divided into four
MESH_F = (
    np.concatenate([np.linspace(0, 0.9, 5), [1.9, 2.9, 3.9, 4.9], np.linspace(4.9, 6, 5)[1:]]),
    np.concatenate([np.linspace(0, 1.1, 5), [2.1, 3.1, 4.1, 5.1], np.linspace(5.1, 6, 5)[1:]]),
)


def cantilever_2002(x_bottom, x_top, n_depth=1):
    """Tip displacements under the three in-plane load cases of Problem 2-002.

    Root: u restrained at every root node, v at the bottom root node; the tip loads are consistent nodal loads of
    uniform axial stress, of the parabolic shear stress (with the equal and opposite shear at the root, which the
    Problem applies as -0.5 lb at joint 8 for its one-element-deep mesh), and of a linear bending stress.
    """
    D = sm.plane_stress_matrix(E_2002, NU_2002)
    nodes, quads, nx = strip_mesh(x_bottom, x_top, DEPTH_2002, n_depth)
    K = assemble(nodes, quads, D, T_2002)
    root = np.arange(n_depth + 1) * nx
    tip = root + nx - 1
    ys = nodes[tip, 1]
    fixed = np.concatenate([3 * root, [1]])
    out = {}
    F = np.zeros(K.shape[0])
    F[3 * tip] = edge_loads(ys, lambda y: 1.0 / DEPTH_2002)
    out["axial"] = solve(K, F, fixed)[3 * tip].mean()
    F = np.zeros(K.shape[0])
    tau = lambda y: 6.0 / DEPTH_2002**3 * y * (DEPTH_2002 - y)  # noqa: E731, unit resultant
    F[3 * tip + 1] = edge_loads(ys, tau)
    F[3 * root + 1] = -edge_loads(ys, tau)
    u = solve(K, F, fixed)
    out["shear"] = u[3 * tip[[0, -1]] + 1].mean()
    F = np.zeros(K.shape[0])
    F[3 * tip] = edge_loads(ys, lambda y: (y - DEPTH_2002 / 2) / I_2002 * T_2002)  # unit moment
    u = solve(K, F, fixed)
    out["moment"] = np.abs(u[3 * tip[[0, -1]]]).mean()
    return out


@pytest.mark.parametrize("mesh", [MESH_A, MESH_B, MESH_C])
def test_2002_axial_extension_is_exact_on_every_mesh(mesh):
    assert cantilever_2002(*mesh)["axial"] == pytest.approx(AXIAL_2002, rel=1e-9)


def test_2002_in_plane_moment_is_exact_on_rectangles():
    """Exact to round-off: the condition number of the reduced K grows to 8e7 on the 24x4 mesh (cond * eps = 2e-8),
    so the tolerance is 1e-7."""
    for n_depth, n_long in ((1, 6), (2, 12), (4, 24)):
        x = np.linspace(0, L_2002, n_long + 1)
        assert cantilever_2002(x, x, n_depth)["moment"] == pytest.approx(MOMENT_2002, rel=1e-7)


def test_2002_in_plane_shear_converges_on_rectangles():
    """Error below 1 percent on the 6x1 mesh and decreasing monotonically to below 0.1 percent on 48x8."""
    errors = []
    for n_depth, n_long in ((1, 6), (2, 12), (4, 24), (8, 48)):
        x = np.linspace(0, L_2002, n_long + 1)
        errors.append(abs(cantilever_2002(x, x, n_depth)["shear"] / SHEAR_2002 - 1))
    assert errors[0] < 0.01
    assert errors[-1] < 0.001
    assert all(b < a for a, b in zip(errors, errors[1:]))


@pytest.mark.parametrize(
    "mesh, n_depth",
    [(MESH_F, 1), ((subdivide(MESH_B[0], 10), subdivide(MESH_B[1], 10)), 2)],
    ids=["model F (parallelograms)", "model G (trapezoids 60x2)"],
)
def test_2002_distorted_meshes_converge(mesh, n_depth):
    """The refinements of Problem 2-002 (Models F and G): shear and moment within 2 percent of the hand values."""
    out = cantilever_2002(*mesh, n_depth)
    assert out["shear"] == pytest.approx(SHEAR_2002, rel=0.02)
    assert out["moment"] == pytest.approx(MOMENT_2002, rel=0.02)


# --------------------------------------------------------------------------------------------------------------------
# CSI Problem 2-003: curved cantilever, in-plane load (MacNeal and Harder 1985)
# --------------------------------------------------------------------------------------------------------------------
def curved_cantilever_2003(n_arc, n_radial):
    """Tip displacement along Y under Fy = 1 at the tip of a quarter ring (radii 4.12 and 4.32 in, t = 0.1 in,
    E = 1e7 psi, nu = 0.25), root at Y = 0 fully fixed in the plane, tip at X = 0. Nodes lie on the arcs."""
    D = sm.plane_stress_matrix(1.0e7, 0.25)
    angles = np.linspace(0, np.pi / 2, n_arc + 1)
    radii = np.linspace(4.12, 4.32, n_radial + 1)
    nodes = np.array([[r * np.cos(a), r * np.sin(a)] for r in radii for a in angles])
    nx = n_arc + 1
    quads = [
        [k * nx + i, (k + 1) * nx + i, (k + 1) * nx + i + 1, k * nx + i + 1]
        for k in range(n_radial)
        for i in range(n_arc)
    ]
    K = assemble(nodes, quads, D, 0.1)
    root = np.arange(n_radial + 1) * nx
    tip = root + n_arc
    F = np.zeros(K.shape[0])
    F[3 * tip[[0, -1]] + 1] = 0.5
    u = solve(K, F, np.concatenate([3 * root, 3 * root + 1]))
    return u[3 * tip[[0, -1]] + 1].mean()


def test_2003_curved_cantilever_converges_to_hand_calculation():
    """Hand calculation of Problem 2-003: pi R^3 F / (4 E I) + pi R F / (4 G Av) + pi R F / (4 E A) = 0.08860 in,
    R = 4.22 in. The 90x1 and 90x3 meshes of the Problem are within 0.5 percent."""
    R, E, G, t, d = 4.22, 1.0e7, 4.0e6, 0.1, 0.2
    hand = np.pi * R**3 / (4 * E * t * d**3 / 12) + np.pi * R / (4 * G * 5 / 6 * d * t) + np.pi * R / (4 * E * d * t)
    assert hand == pytest.approx(0.08860, abs=5e-6)
    assert curved_cantilever_2003(90, 1) == pytest.approx(hand, rel=0.005)
    assert curved_cantilever_2003(90, 3) == pytest.approx(hand, rel=0.005)
    assert abs(curved_cantilever_2003(30, 1) / hand - 1) < abs(curved_cantilever_2003(6, 1) / hand - 1)


# --------------------------------------------------------------------------------------------------------------------
# penalty factors: rank of meshes, what the penalties act on, and sensitivity to their values
# --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape", ["rectangle", "general"])
def test_single_element_has_exactly_three_zero_eigenvalues(shape):
    K = sm.membrane_stiffness(np.array(SHAPES[shape], float), sm.plane_stress_matrix(1.0, 0.25), 1.0)[0]
    assert zero_eigenvalues(K)[0] == 3


@pytest.mark.parametrize("distortion", [0.0, 0.3], ids=["rectangles", "distorted"])
def test_free_two_by_two_mesh_has_exactly_three_zero_eigenvalues(distortion):
    """No support, every rotation free: only the three rigid modes of the mesh (the alternating-rotation mode of a
    rectangular mesh is suppressed)."""
    x, y = np.meshgrid([0.0, 1.0, 2.0], [0.0, 0.5, 1.0])
    nodes = np.column_stack([x.ravel(), y.ravel()])
    nodes[4] += distortion * np.array([1.0, 0.4])  # move the center node
    quads = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
    K = assemble(nodes, quads, sm.plane_stress_matrix(1.0, 0.25), 1.0)
    assert zero_eigenvalues(K)[0] == 3


def test_penalties_vanish_for_rigid_motion_constant_strain_and_linear_rotation():
    """On a distorted quadrilateral, both penalty rows are zero for the rigid modes and for constant strain (nodal
    rotations equal to the rotation of the field); the alternating-rotation row is also zero for any linear rotation
    field theta = a + b x + c y. The rows are the ones built into the stiffness."""
    xy = np.array(SHAPES["general"], float)
    x, y = xy.T
    drilling, hourglass = (row[0] for row in sm.membrane_penalty_vectors(xy))
    fields = list(rigid_modes(xy))
    for ux, uy in ((x, 0 * x), (0 * x, y), (y / 2, x / 2)):  # constant eps_x, eps_y, gamma_xy; rotation zero
        u = np.zeros(12)
        u[0::3], u[1::3] = ux, uy
        fields.append(u)
    for u in fields:
        assert abs(drilling @ u) < 1e-12
        assert abs(hourglass @ u) < 1e-12
    for a, b, c in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.3, -2.0, 1.7)):
        u = np.zeros(12)
        u[2::3] = a + b * x + c * y
        assert abs(hourglass @ u) < 1e-12
    alternating = np.zeros(12)
    alternating[2::3] = [1.0, -1.0, 1.0, -1.0]
    assert abs(hourglass @ alternating) > 0.1
    # the hourglass row is the term the stiffness carries: K(factor) - K(0) = factor * G * t * A * h h^T
    D = sm.plane_stress_matrix(1.0, 0.25)
    area = 0.5 * abs((x[2] - x[0]) * (y[3] - y[1]) - (x[3] - x[1]) * (y[2] - y[0]))
    K = sm.membrane_stiffness(xy, D, 1.0)[0]
    factor = sm.ROTATION_HOURGLASS_FACTOR
    try:
        sm.ROTATION_HOURGLASS_FACTOR = 0.0
        K0 = sm.membrane_stiffness(xy, D, 1.0)[0]
    finally:
        sm.ROTATION_HOURGLASS_FACTOR = factor
    assert np.allclose(K - K0, factor * D[0, 2, 2] * area * np.outer(hourglass, hourglass), atol=1e-14)


def _refined_results():
    out = {}
    for name, mesh, n_depth in (
        ("2-002 F", MESH_F, 1),
        ("2-002 G", (subdivide(MESH_B[0], 10), subdivide(MESH_B[1], 10)), 2),
    ):
        result = cantilever_2002(*mesh, n_depth)
        out[name + " shear"], out[name + " moment"] = result["shear"], result["moment"]
    out["2-003 90x1"] = curved_cantilever_2003(90, 1)
    out["2-003 90x3"] = curved_cantilever_2003(90, 3)
    return out


@pytest.mark.parametrize("constant", ["ROTATION_HOURGLASS_FACTOR", "DRILLING_PENALTY_FACTOR"])
@pytest.mark.parametrize("scale", [0.1, 10.0])
def test_results_are_insensitive_to_the_penalty_factors(monkeypatch, constant, scale):
    """Scaling a penalty factor by 0.1 or 10 keeps the patch tests (Problem 2-001) and pure bending exact, and moves
    the refined meshes of Problems 2-002 and 2-003 by less than 1 percent. The coarse distorted 6x1 meshes (Models B
    and C) are more sensitive; their changes are recorded in SHELL_PLAN.md, M3."""
    reference = _refined_results()
    monkeypatch.setattr(sm, constant, getattr(sm, constant) * scale)
    test_patch_test_with_prescribed_edge_displacements(0.0)
    test_patch_test_with_prescribed_edge_displacements(2.0e-3)
    test_patch_test_with_edge_tractions_and_minimal_supports()
    test_pure_bending_of_rectangles_is_exact(0.25)
    test_2002_in_plane_moment_is_exact_on_rectangles()
    scaled = _refined_results()
    for key, value in reference.items():
        assert scaled[key] == pytest.approx(value, rel=0.01), key
