"""Four-node plate bending element MITC4 (fem3d/SHELL_PLAN.md, milestone M4).

Reissner-Mindlin plate with mixed interpolation of the transverse shear strains: Dvorkin and Bathe (1984); Bathe
(2014), *Finite Element Procedures*, 2nd ed., §5.4.2, Eqs. 5.88 to 5.105 and Example 5.30.

- Kinematics (Bathe Eq. 5.88): u = -z beta_x, v = -z beta_y, w = w(x, y), with beta_x = -theta_y and
  beta_y = theta_x (Eq. 5.99), where theta_x and theta_y are the nodal rotations about the local x and y axes
  (right-hand rule). w, theta_x, and theta_y are interpolated bilinearly.
- Curvatures (Eq. 5.95): kappa = (d beta_x/dx, d beta_y/dy, d beta_x/dy + d beta_y/dx), from the bilinear
  rotations.
- Transverse shear (Eqs. 5.100 to 5.103, Example 5.30): the covariant shear strains
  gamma_r = dw/dr - beta . dx/dr and gamma_s = dw/ds - beta . dx/ds are evaluated from the displacement
  interpolation at the midpoints of the sides (tying points A, C on s = +1, -1 for gamma_r; D, B on r = +1, -1 for
  gamma_s) and interpolated linearly between them; the Cartesian strains are gamma = J^-1 (gamma_r, gamma_s) at the
  point considered. This is Bathe's Eq. 5.103 written with the Jacobian instead of the angles alpha and beta.
- Stress resultants (Eqs. 5.94 to 5.97): M = C_b kappa and V = C_s gamma, with C_b = E h^3 / (12 (1 - nu^2)) times
  the plane-stress pattern and C_s = k G h I, k = 5/6 (Eq. 5.93). These are the CSI resultants M11, M22, M12, V13,
  V23 (CSI Analysis Reference Manual §10.18, Eqns. 10-8d to 10-8h: M = -integral of x3 sigma, V = integral of
  sigma_13), so positive moments put the bottom face (-3) in tension.
- 2x2 Gauss integration of both parts. One element has exactly three zero-energy modes, the rigid modes
  w = a + b x + c y.

Everything here works in the element's internal formulation frame: two-dimensional coordinates in the element plane,
nodes in counterclockwise order. All functions are vectorized over N elements. DOF order per node:
(w, theta_x, theta_y). This is the thick-plate (Mindlin) formulation only; a thin-plate (Kirchhoff) element (DKQ)
is planned for a later session.
"""

from __future__ import annotations

import numpy as np

from .shell_membrane import GAUSS_POINTS_2X2, _as_quads, _bilinear

#: Shear correction factor k of the transverse shear rigidity k G h (Bathe Eq. 5.93).
SHEAR_CORRECTION_FACTOR = 5.0 / 6.0

# Tying points (Bathe Fig. 5.26 and Eq. 5.100, in this module's node order 1(-1,-1), 2(1,-1), 3(1,1), 4(-1,1)):
# gamma_r at A = (0, +1) and C = (0, -1); gamma_s at D = (+1, 0) and B = (-1, 0).
_TYING_R = ((0.0, 1.0), (0.0, -1.0))
_TYING_S = ((1.0, 0.0), (-1.0, 0.0))


def plate_rigidities(E, nu, G, thickness) -> tuple[np.ndarray, np.ndarray]:
    """Bending rigidity C_b (N, 3, 3) and transverse shear rigidity C_s (N, 2, 2) of isotropic plates (Bathe
    Eq. 5.97), with the shear correction factor ``SHEAR_CORRECTION_FACTOR``. ``G`` is the material shear modulus
    (CSI g13 = g23)."""
    E = np.atleast_1d(np.asarray(E, dtype=float))
    nu = np.broadcast_to(np.asarray(nu, dtype=float), E.shape)
    G = np.broadcast_to(np.asarray(G, dtype=float), E.shape)
    h = np.broadcast_to(np.asarray(thickness, dtype=float), E.shape)
    if np.any(h <= 0.0):
        raise ValueError("plate thickness must be positive")
    d = E * h**3 / (12.0 * (1.0 - nu**2))
    Cb = np.zeros(E.shape + (3, 3))
    Cb[:, 0, 0] = Cb[:, 1, 1] = d
    Cb[:, 0, 1] = Cb[:, 1, 0] = d * nu
    Cb[:, 2, 2] = d * (1.0 - nu) / 2.0
    Cs = np.zeros(E.shape + (2, 2))
    Cs[:, 0, 0] = Cs[:, 1, 1] = SHEAR_CORRECTION_FACTOR * G * h
    return Cb, Cs


def _jacobian(xy: np.ndarray, r: float, s: float):
    """Bilinear shape functions (4,), their Cartesian derivatives (N, 2, 4), J (N, 2, 2), and det J (N,)."""
    N, dN_rs = _bilinear(r, s)
    J = np.einsum("ak,nkb->nab", dN_rs, xy)  # J[a, b] = d x_b / d r_a
    detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    bad = np.flatnonzero(detJ <= 0.0)
    if bad.size:
        raise ValueError(
            f"plate element(s) {bad.tolist()} have a non-positive Jacobian at (r, s) = ({r:.3f}, {s:.3f}): "
            "nodes must be in counterclockwise order in the element plane, with every inside angle below 180 degrees"
        )
    dN = np.einsum("nab,bk->nak", np.linalg.inv(J), dN_rs)
    return N, dN_rs, J, detJ, dN


def _covariant_shear(xy: np.ndarray, r: float, s: float) -> np.ndarray:
    """Rows (N, 2, 12) of the covariant shear strains (gamma_r, gamma_s) from the displacement interpolation at
    (r, s): gamma_a = dw/dr_a - beta_x dx/dr_a - beta_y dy/dr_a, with beta_x = -theta_y and beta_y = theta_x."""
    N, dN_rs, J, _, _ = _jacobian(xy, r, s)
    rows = np.zeros((xy.shape[0], 2, 12))
    for a in range(2):
        rows[:, a, 0::3] = dN_rs[a]  # dw/dr_a
        rows[:, a, 1::3] = -J[:, a, 1][:, None] * N  # -beta_y dy/dr_a = -theta_x dy/dr_a
        rows[:, a, 2::3] = J[:, a, 0][:, None] * N  # -beta_x dx/dr_a = +theta_y dx/dr_a
    return rows


def plate_strain_matrices(xy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curvature and assumed shear strain matrices at the 2x2 Gauss points.

    Returns
    -------
    Bb : array (N, 4, 3, 12)
        Curvatures (kappa_x, kappa_y, kappa_xy) from the 12 DOFs (w, theta_x, theta_y per node).
    Bs : array (N, 4, 2, 12)
        Transverse shear strains (gamma_xz, gamma_yz) of the MITC4 interpolation.
    detJ : array (N, 4)
        Jacobian determinants at the Gauss points (Gauss weights are 1).
    """
    xy = _as_quads(xy)
    n = xy.shape[0]
    tie_r = [_covariant_shear(xy, r, s)[:, 0] for r, s in _TYING_R]  # at A (s = +1) and C (s = -1)
    tie_s = [_covariant_shear(xy, r, s)[:, 1] for r, s in _TYING_S]  # at D (r = +1) and B (r = -1)
    Bb, Bs, dets = [], [], []
    for r, s in GAUSS_POINTS_2X2:
        _, _, J, detJ, dN = _jacobian(xy, r, s)
        bb = np.zeros((n, 3, 12))
        bb[:, 0, 2::3] = -dN[:, 0]  # d beta_x / dx = -d theta_y / dx
        bb[:, 1, 1::3] = dN[:, 1]  # d beta_y / dy = d theta_x / dy
        bb[:, 2, 2::3] = -dN[:, 1]
        bb[:, 2, 1::3] = dN[:, 0]
        covariant = np.stack(
            [0.5 * (1 + s) * tie_r[0] + 0.5 * (1 - s) * tie_r[1], 0.5 * (1 + r) * tie_s[0] + 0.5 * (1 - r) * tie_s[1]],
            axis=1,
        )
        Bs.append(np.einsum("nab,nbj->naj", np.linalg.inv(J), covariant))
        Bb.append(bb)
        dets.append(detJ)
    return np.stack(Bb, axis=1), np.stack(Bs, axis=1), np.stack(dets, axis=1)


def _rigidities(Cb, Cs, n: int):
    return (
        np.broadcast_to(np.asarray(Cb, dtype=float), (n, 3, 3)),
        np.broadcast_to(np.asarray(Cs, dtype=float), (n, 2, 2)),
    )


def plate_stiffness(xy, Cb, Cs) -> np.ndarray:
    """Stiffness matrices (N, 12, 12) of MITC4 plates: sum over the 2x2 Gauss points of
    (Bb^T C_b Bb + Bs^T C_s Bs) det J (Bathe Eq. 5.94)."""
    Bb, Bs, detJ = plate_strain_matrices(xy)
    Cb, Cs = _rigidities(Cb, Cs, Bb.shape[0])
    return np.einsum("ng,ngai,nab,ngbj->nij", detJ, Bb, Cb, Bb) + np.einsum("ng,ngai,nab,ngbj->nij", detJ, Bs, Cs, Bs)


def plate_resultants(xy, Cb, Cs, u) -> np.ndarray:
    """Plate resultants per unit length (M11, M22, M12, V13, V23) at the 2x2 Gauss points, shape (N, 4, 5)."""
    Bb, Bs, _ = plate_strain_matrices(xy)
    n = Bb.shape[0]
    Cb, Cs = _rigidities(Cb, Cs, n)
    u = np.asarray(u, dtype=float).reshape(n, 12)
    M = np.einsum("nab,ngbj,nj->nga", Cb, Bb, u)
    V = np.einsum("nab,ngbj,nj->nga", Cs, Bs, u)
    return np.concatenate([M, V], axis=2)


def plate_pressure_loads(xy, pressure) -> np.ndarray:
    """Consistent nodal loads (N, 12) of a uniform pressure along +z (force per area): integral of N_i p dA on the
    w DOFs, zero on the rotations (Bathe Example 5.29, Eq. b)."""
    xy = _as_quads(xy)
    p = np.broadcast_to(np.asarray(pressure, dtype=float), (xy.shape[0],))
    loads = np.zeros((xy.shape[0], 12))
    for r, s in GAUSS_POINTS_2X2:
        N, _, _, detJ, _ = _jacobian(xy, r, s)
        loads[:, 0::3] += (p * detJ)[:, None] * N
    return loads
