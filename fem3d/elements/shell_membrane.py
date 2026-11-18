"""Quadrilateral membrane element with drilling rotations (fem3d/SHELL_PLAN.md, milestone M3).

The formulation that CSI cites for the membrane part of the SAP2000 shell (CSI Analysis Reference Manual §10.1.1:
isoparametric, quadratic in-plane displacements, drilling rotation normal to the plane): Ibrahimbegovic, Taylor and
Wilson (1990), as presented in Wilson (2002), *Three-Dimensional Static and Dynamic Analysis of Structures*,
Chapter 9, and in Ibrahimbegovic and Frey (1992), *Engineering Fracture Mechanics* 43(1), 13-24, Section 3 (element
MQ2). SAP2000's own implementation is not published; its results for distorted meshes differ from this element by a
few percent, and both converge to the closed-form solutions (tests/test_shell_membrane.py).

1. Bilinear in-plane displacements plus the hierarchical mid-side functions N5 to N8 (Wilson Eqs. 9.1 and 9.2).
2. The relative mid-side displacement normal to each side is parabolic, set by the drilling rotations at the two
   ends of the side: (l_JK / 8)(theta_K - theta_J) along the outward normal (Wilson Eqs. 9.3 and 9.4;
   Ibrahimbegovic and Frey Eq. 24).
3. The relative tangential mid-side displacements are kept as four incompatible modes N_I * t_JK and condensed at
   element level (Ibrahimbegovic and Frey Eqs. 24 and 37). Without them the element cannot represent the Poisson
   strain of pure bending and is 1/(1 - nu^2) too stiff; with them, rectangular meshes are exact in pure bending, as
   SAP2000 is in CSI verification Problem 2-002.
4. The strain matrices of the rotations and of the incompatible modes are purified of constant strain: their
   element average, from the same 2x2 Gauss rule as the stiffness, is subtracted (Wilson Eqs. 9.8 and 6.7;
   Ibrahimbegovic and Frey Eqs. 39 and 40). This is what makes the element pass the patch test.
5. A rank-one penalty ties the nodal rotations to the continuum rotation at the element center (Wilson Eqs. 9.11 to
   9.13), with k0 = 0.025 D33 (Eq. 9.14). The continuum rotation includes the incompatible modes (Ibrahimbegovic and
   Frey Eqs. 41 and 45).
6. With 2x2 integration, rectangles and parallelograms keep one more zero-energy mode: alternating rotations
   (+, -, +, -) at the corners. After step 4 that mode has no strain at the Gauss points and no center rotation, and
   it spreads through a mesh, so a wall of rectangles with free drilling rotations would be singular. Cook et al.
   (2002), *Concepts and Applications of Finite Element Analysis*, §3.10, give the cure: a penalty on the function
   theta1 - theta2 + theta3 - theta4. Here that function is orthogonalized against linear rotation fields (it
   vanishes for every rotation a + b x + c y on any quadrilateral), so the penalty does not affect rigid-body motion,
   constant strain, or pure bending. Its factor is the same 0.025 D33.

Sign convention: the drilling rotation is positive counterclockwise about local 3 (right-hand rule), as are the frame
rotations it connects to, and the continuum rotation is theta0 = (dv/dx - du/dy) / 2 (Ibrahimbegovic and Frey
Eq. 42). Wilson prints Eq. 9.11 with the opposite sign, which would strain the penalty under a rigid in-plane rotation
(tests/test_shell_membrane.py checks the rigid modes and a rotated patch test).

Everything here works in the element's internal formulation frame: two-dimensional coordinates in the element plane,
nodes in counterclockwise order. All functions are vectorized over N elements. DOF order per node: (u, v, theta).
"""

from __future__ import annotations

import numpy as np

GAUSS_POINT = 1.0 / np.sqrt(3.0)
#: Natural coordinates (r, s) of the 2x2 Gauss points, in the order of the corner nodes they lie next to.
GAUSS_POINTS_2X2 = np.array(
    [[-GAUSS_POINT, -GAUSS_POINT], [GAUSS_POINT, -GAUSS_POINT], [GAUSS_POINT, GAUSS_POINT], [-GAUSS_POINT, GAUSS_POINT]]
)
#: Natural coordinates of the corner nodes 1 to 4.
CORNER_POINTS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
#: Penalty factor of the drilling rotation: k0 = 0.025 D33 (Wilson Eq. 9.14).
DRILLING_PENALTY_FACTOR = 0.025
#: Penalty factor of the alternating-rotation mode (Cook et al. §3.10 give the penalty, not its value). The value is
#: Wilson's k0 = 0.025 D33 (Eq. 9.14). Results are insensitive to it: scaling it by 0.1 or 10 leaves the patch test
#: and pure bending exact (tests/test_shell_membrane.py; changes of the cantilevers in SHELL_PLAN.md, M3).
ROTATION_HOURGLASS_FACTOR = 0.025

# Sides 5 to 8 (Wilson Fig. 9.1) as (J, K) corner indices.
_SIDE_J = np.array([0, 1, 2, 3])
_SIDE_K = np.array([1, 2, 3, 0])
# Side-difference operator: (S @ theta)[m] = theta_K - theta_J for side m.
_SIDE_DIFFERENCE = np.zeros((4, 4))
_SIDE_DIFFERENCE[np.arange(4), _SIDE_J] = -1.0
_SIDE_DIFFERENCE[np.arange(4), _SIDE_K] = 1.0
_ALTERNATING = np.array([1.0, -1.0, 1.0, -1.0])


def plane_stress_matrix(E, nu) -> np.ndarray:
    """Isotropic plane-stress matrix D (Wilson Eq. 9.9), shape (N, 3, 3) for arrays E and nu of length N."""
    E = np.atleast_1d(np.asarray(E, dtype=float))
    nu = np.broadcast_to(np.asarray(nu, dtype=float), E.shape)
    c = E / (1.0 - nu**2)
    D = np.zeros(E.shape + (3, 3))
    D[:, 0, 0] = D[:, 1, 1] = c
    D[:, 0, 1] = D[:, 1, 0] = c * nu
    D[:, 2, 2] = E / (2.0 * (1.0 + nu))
    return D


def _bilinear(r: float, s: float):
    """Bilinear shape functions N1..N4 and their (r, s) derivatives, shapes (4,) and (2, 4)."""
    N = 0.25 * np.array([(1 - r) * (1 - s), (1 + r) * (1 - s), (1 + r) * (1 + s), (1 - r) * (1 + s)])
    dN = 0.25 * np.array(
        [[-(1 - s), (1 - s), (1 + s), -(1 + s)], [-(1 - r), -(1 + r), (1 + r), (1 - r)]]
    )
    return N, dN


def _hierarchical_derivatives(r: float, s: float) -> np.ndarray:
    """(r, s) derivatives of the mid-side functions N5..N8 (Wilson Eq. 9.2), shape (2, 4)."""
    return np.array(
        [
            [-r * (1 - s), 0.5 * (1 - s * s), -r * (1 + s), -0.5 * (1 - s * s)],
            [-0.5 * (1 - r * r), -(1 + r) * s, 0.5 * (1 - r * r), -(1 - r) * s],
        ]
    )


def _as_quads(xy) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if xy.ndim == 2:
        xy = xy[None]
    if xy.ndim != 3 or xy.shape[1:] != (4, 2):
        raise ValueError(f"membrane coordinates must have shape (N, 4, 2), got {xy.shape}")
    return xy


def _point_derivatives(xy: np.ndarray, r: float, s: float):
    """Cartesian derivatives at (r, s) of the bilinear and the mid-side functions, and det J.

    Returns dN (N, 2, 4), dH (N, 2, 4), detJ (N,). Raises ValueError where det J <= 0 (nodes not in
    counterclockwise order, or a corner angle of 180 degrees or more).
    """
    _, dN_rs = _bilinear(r, s)
    dH_rs = _hierarchical_derivatives(r, s)
    J = np.einsum("ak,nkb->nab", dN_rs, xy)  # J[a, b] = d x_b / d r_a
    detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    bad = np.flatnonzero(detJ <= 0.0)
    if bad.size:
        raise ValueError(
            f"membrane element(s) {bad.tolist()} have a non-positive Jacobian at (r, s) = ({r:.3f}, {s:.3f}): "
            "nodes must be in counterclockwise order in the element plane, with every inside angle below 180 degrees"
        )
    Jinv = np.empty_like(J)
    Jinv[:, 0, 0] = J[:, 1, 1] / detJ
    Jinv[:, 1, 1] = J[:, 0, 0] / detJ
    Jinv[:, 0, 1] = -J[:, 0, 1] / detJ
    Jinv[:, 1, 0] = -J[:, 1, 0] / detJ
    dN = np.einsum("nab,bk->nak", Jinv, dN_rs)
    dH = np.einsum("nab,bk->nak", Jinv, dH_rs)
    return dN, dH, detJ


def _side_vectors(xy: np.ndarray):
    """Side operators of the drilling rotations and unit side tangents.

    For side m from corner J to corner K, the relative mid-side displacement from the rotations is
    du_m = (y_K - y_J) / 8 * (theta_K - theta_J) and dv_m = (x_J - x_K) / 8 * (theta_K - theta_J), that is
    (l / 8)(theta_K - theta_J) along the outward normal of a counterclockwise element. Returns Pu, Pv (N, 4, 4) with
    du = Pu @ theta, dv = Pv @ theta, and the unit tangents t (N, 4, 2) of the incompatible modes.
    """
    d = xy[:, _SIDE_K] - xy[:, _SIDE_J]
    Pu = (d[:, :, 1] / 8.0)[:, :, None] * _SIDE_DIFFERENCE[None]
    Pv = (-d[:, :, 0] / 8.0)[:, :, None] * _SIDE_DIFFERENCE[None]
    t = d / np.linalg.norm(d, axis=2, keepdims=True)
    return Pu, Pv, t


def _point_matrices(xy: np.ndarray, r: float, s: float, Pu, Pv, t):
    """Strain rows (N, 3, 16) and rotation row (N, 16) at (r, s), before the constant-strain purification.

    Columns: the 12 nodal DOFs (u, v, theta per node), then the 4 incompatible tangential modes.
    """
    n = xy.shape[0]
    dN, dH, detJ = _point_derivatives(xy, r, s)
    N, _ = _bilinear(r, s)
    # drilling part: u_h = sum_m H_m (Pu theta)_m, v_h = sum_m H_m (Pv theta)_m
    Hu_x = np.einsum("nm,nmk->nk", dH[:, 0], Pu)
    Hu_y = np.einsum("nm,nmk->nk", dH[:, 1], Pu)
    Hv_x = np.einsum("nm,nmk->nk", dH[:, 0], Pv)
    Hv_y = np.einsum("nm,nmk->nk", dH[:, 1], Pv)
    B = np.zeros((n, 3, 16))
    B[:, 0, 0:12:3] = dN[:, 0]
    B[:, 1, 1:12:3] = dN[:, 1]
    B[:, 2, 0:12:3] = dN[:, 1]
    B[:, 2, 1:12:3] = dN[:, 0]
    B[:, 0, 2:12:3] = Hu_x
    B[:, 1, 2:12:3] = Hv_y
    B[:, 2, 2:12:3] = Hu_y + Hv_x
    # incompatible tangential modes: u_I = sum_m H_m t_m alpha_m
    B[:, 0, 12:] = dH[:, 0] * t[:, :, 0]
    B[:, 1, 12:] = dH[:, 1] * t[:, :, 1]
    B[:, 2, 12:] = dH[:, 1] * t[:, :, 0] + dH[:, 0] * t[:, :, 1]
    # continuum rotation (dv/dx - du/dy) / 2 minus the bilinear interpolation of the nodal rotations
    rot = np.zeros((n, 16))
    rot[:, 0:12:3] = -0.5 * dN[:, 1]
    rot[:, 1:12:3] = 0.5 * dN[:, 0]
    rot[:, 2:12:3] = 0.5 * (Hv_x - Hu_y) - N
    rot[:, 12:] = 0.5 * (dH[:, 0] * t[:, :, 1] - dH[:, 1] * t[:, :, 0])
    return B, rot, detJ, dN


def _rotation_hourglass(xy: np.ndarray, dN0: np.ndarray) -> np.ndarray:
    """Weights g (N, 4) of the alternating-rotation penalty, g @ theta = (theta1 - theta2 + theta3 - theta4) / 4
    orthogonalized against linear fields: g @ (a + b x + c y) = 0 at the corners of any quadrilateral.

    With h = (1, -1, 1, -1) and dN0 the bilinear derivatives at the center (sum dN0_x = 0, dN0_x @ x = 1,
    dN0_x @ y = 0, and likewise for y): g = (h - (h @ x) dN0_x - (h @ y) dN0_y) / 4.
    """
    hx = xy[:, :, 0] @ _ALTERNATING
    hy = xy[:, :, 1] @ _ALTERNATING
    return 0.25 * (_ALTERNATING[None] - hx[:, None] * dN0[:, 0] - hy[:, None] * dN0[:, 1])


def membrane_penalty_vectors(xy) -> tuple[np.ndarray, np.ndarray]:
    """Rows of the two penalty terms on the 12 nodal DOFs, before condensation of the incompatible modes.

    Returns ``drilling`` (N, 12): the continuum rotation at the element center minus the bilinear average of the
    nodal rotations (Wilson Eq. 9.12; its incompatible-mode part is omitted here), and ``hourglass`` (N, 12): the
    alternating-rotation function of ``_rotation_hourglass`` on the rotation DOFs. The penalty energies are
    0.5 * factor * D33 * t * A * (row @ u)^2.
    """
    xy = _as_quads(xy)
    Pu, Pv, t = _side_vectors(xy)
    _, rot0, _, dN0 = _point_matrices(xy, 0.0, 0.0, Pu, Pv, t)
    hourglass = np.zeros((xy.shape[0], 12))
    hourglass[:, 2::3] = _rotation_hourglass(xy, dN0)
    return rot0[:, :12], hourglass


def _membrane_matrices(xy, D):
    """Condensed stiffness per unit thickness (N, 12, 12), condensed strain matrices at the Gauss points
    (N, 4, 3, 12), and det J at the Gauss points (N, 4). The thickness factors out of the condensation."""
    xy = _as_quads(xy)
    n = xy.shape[0]
    D = np.broadcast_to(np.asarray(D, dtype=float), (n, 3, 3))
    Pu, Pv, t = _side_vectors(xy)
    rows = [_point_matrices(xy, r, s, Pu, Pv, t) for r, s in GAUSS_POINTS_2X2]
    B = np.stack([row[0] for row in rows], axis=1)  # (N, 4, 3, 16)
    detJ = np.stack([row[2] for row in rows], axis=1)
    area = detJ.sum(axis=1)
    # purify the strains of the rotations and of the incompatible modes of constant strain
    mean = np.einsum("ng,ngij->nij", detJ, B[..., 2:12:3]) / area[:, None, None]
    B[..., 2:12:3] -= mean[:, None]
    mean = np.einsum("ng,ngij->nij", detJ, B[..., 12:]) / area[:, None, None]
    B[..., 12:] -= mean[:, None]

    K = np.einsum("ng,ngai,nab,ngbj->nij", detJ, B, D, B)
    _, rot0, _, dN0 = _point_matrices(xy, 0.0, 0.0, Pu, Pv, t)
    G = D[:, 2, 2] * area
    K += (DRILLING_PENALTY_FACTOR * G)[:, None, None] * np.einsum("ni,nj->nij", rot0, rot0)
    # alternating rotations, orthogonal to every linear rotation field (Cook et al. §3.10)
    hourglass = np.zeros((n, 16))
    hourglass[:, 2:12:3] = _rotation_hourglass(xy, dN0)
    K += (ROTATION_HOURGLASS_FACTOR * G)[:, None, None] * np.einsum("ni,nj->nij", hourglass, hourglass)

    # static condensation of the incompatible modes: alpha = -Kii^-1 Kic u
    Kcc, Kci, Kii = K[:, :12, :12], K[:, :12, 12:], K[:, 12:, 12:]
    T = -np.linalg.solve(Kii, np.transpose(Kci, (0, 2, 1)))  # (N, 4, 12)
    K_condensed = Kcc + Kci @ T
    K_condensed = 0.5 * (K_condensed + np.transpose(K_condensed, (0, 2, 1)))
    B_condensed = B[..., :12] + np.einsum("ngak,nkj->ngaj", B[..., 12:], T)
    return K_condensed, B_condensed, detJ


def _thickness(thickness, n: int) -> np.ndarray:
    t = np.broadcast_to(np.asarray(thickness, dtype=float), (n,))
    if np.any(t <= 0.0):
        raise ValueError("membrane thickness must be positive")
    return t


def membrane_stiffness(xy, D, thickness) -> np.ndarray:
    """Stiffness matrices (N, 12, 12) of quadrilateral membranes with drilling rotations.

    Parameters
    ----------
    xy : array (N, 4, 2) or (4, 2)
        Corner coordinates in the element plane, counterclockwise.
    D : array (N, 3, 3) or (3, 3)
        Plane-stress matrix (stress from strain), symmetric positive definite.
    thickness : float or array (N,)
        Membrane thickness.
    """
    K, _, _ = _membrane_matrices(xy, D)
    return _thickness(thickness, K.shape[0])[:, None, None] * K


def membrane_strain_matrices(xy, D) -> tuple[np.ndarray, np.ndarray]:
    """Strain matrices at the 2x2 Gauss points, with the incompatible modes condensed.

    Returns
    -------
    B : array (N, 4, 3, 12)
        Strains (eps_11, eps_22, gamma_12) at each Gauss point (order of ``GAUSS_POINTS_2X2``) from the 12 nodal DOFs.
    detJ : array (N, 4)
        Jacobian determinants at the Gauss points (Gauss weights are 1).
    """
    _, B, detJ = _membrane_matrices(xy, D)
    return B, detJ


def membrane_resultants(xy, D, thickness, u) -> np.ndarray:
    """Membrane forces per unit length (F11, F22, F12) at the 2x2 Gauss points, shape (N, 4, 3).

    ``u`` (N, 12) holds the element DOFs (u, v, theta per node) in the element plane. F = t D B u.
    """
    _, B, _ = _membrane_matrices(xy, D)
    n = B.shape[0]
    D = np.broadcast_to(np.asarray(D, dtype=float), (n, 3, 3))
    t = _thickness(thickness, n)
    u = np.asarray(u, dtype=float).reshape(n, 12)
    return t[:, None, None] * np.einsum("nab,ngbj,nj->nga", D, B, u)
