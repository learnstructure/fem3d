"""
Equivalent nodal loads of member loads on Timoshenko (shear-deformable) frame members.

The equivalent nodal loads are the work of the member load on the exact deflected shapes of the element
(``fem3d.loads.bending_shape_functions``). Three independent checks:

1. The shapes are exact: they take the unit end values and solve the unloaded Timoshenko beam equations
   EI psi'' + kGA (v' - psi) = 0 and (v' - psi)' = 0; with phi = 0 they are the Hermite cubics.
2. Fixed-end forces from the flexibility (unit-load) method, which uses neither shape functions nor the stiffness
   matrix: a cantilever under the load, with the end force and moment as redundants, and the flexibilities
   integral(M m / EI) + integral(V v / kGA) by Gauss quadrature on each load segment (exact for these polynomials).
3. A member point load gives the same solution as a mesh with a node at the load point, where the load is a nodal
   load (exact for any element whose stiffness is exact), with and without end releases.

The comparison with SAP2000 (struct_link/tests/test_sap2000_live.py) checks the same loads in a 3D frame.
"""

import numpy as np
import pytest

from fem3d import ElasticMaterial, FrameElement, Node, Results, Section, Structure
from fem3d.loads import DistributedLoad, ElementPointLoad, PointLoad, bending_shape_functions

E, NU = 25.0e6, 0.2
L = 2.0          # short, deep member: phi_2 = 0.46, phi_3 = 0.12
WIDTH, DEPTH = 0.4, 0.8


def _member(include_shear_deformation=True, releases_j=None, length=L, eid=1, node_i=None, node_j=None):
    node_i = node_i or Node(1, 0.0, 0.0, 0.0)
    node_j = node_j or Node(2, length, 0.0, 0.0)
    material = ElasticMaterial(E=E, nu=NU)
    section = Section.from_rectangle(width=WIDTH, depth=DEPTH, material=material)
    return FrameElement(eid, node_i, node_j, material=material, section=section,
                        include_shear_deformation=include_shear_deformation, releases_j=releases_j)


# ── 1. exact shapes ──────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("phi", [0.0, 0.46, 3.0])
def test_shape_functions_solve_the_timoshenko_beam(phi):
    ei, kga = 1.0, 12.0 / (phi * L**2) if phi > 0 else None
    ends = [bending_shape_functions(x, L, phi) for x in (0.0, L)]
    values = np.array([[nv[k], npsi[k]] for nv, npsi in ends for k in range(4)]).reshape(2, 4, 2)
    # unit end values: (v_i, psi_i, v_j, psi_j) of shape k
    np.testing.assert_allclose(
        np.array([values[0, :, 0], values[0, :, 1], values[1, :, 0], values[1, :, 1]]), np.eye(4), atol=1e-14)
    h = 1e-4 * L
    for x in np.linspace(0.1, 1.9, 7):
        for k in range(4):
            v = lambda s: bending_shape_functions(s, L, phi)[0][k]
            psi = lambda s: bending_shape_functions(s, L, phi)[1][k]
            gamma = lambda s: (v(s + h) - v(s - h)) / (2 * h) - psi(s)       # shear strain v' - psi
            psi_xx = (psi(x + h) - 2 * psi(x) + psi(x - h)) / h**2
            assert gamma(x) == pytest.approx(gamma(0.5 * L), abs=1e-7)        # constant shear
            if kga is None:
                assert gamma(x) == pytest.approx(0.0, abs=1e-7)               # Euler-Bernoulli: psi = v'
            else:
                assert ei * psi_xx == pytest.approx(-kga * gamma(x), abs=1e-5)


def test_shape_functions_without_shear_are_hermite():
    for x in np.linspace(0.0, L, 9):
        mu = x / L
        nv, npsi = bending_shape_functions(x, L, 0.0)
        hermite = (1 - 3 * mu**2 + 2 * mu**3, x * (1 - mu)**2, 3 * mu**2 - 2 * mu**3, x * (mu**2 - mu))
        slopes = (6 * (mu**2 - mu) / L, 1 - 4 * mu + 3 * mu**2, -6 * (mu**2 - mu) / L, 3 * mu**2 - 2 * mu)
        np.testing.assert_allclose(nv, hermite, atol=1e-14)
        np.testing.assert_allclose(npsi, slopes, atol=1e-14)


# ── 2. flexibility-method reference ──────────────────────────────────────────────────────────────────────────────


def fixed_end_loads(length, ei, kga, distributed=(), forces=(), moments=()):
    """Equivalent nodal loads (v_i, psi_i, v_j, psi_j) of a fixed-fixed beam in one bending plane.

    ``distributed``: (w_start, w_end, start, end) with relative start and end; ``forces``: (P, a) transverse;
    ``moments``: (m, a) about the plane normal; a is the distance from end i. ``kga = None``: no shear deformation.
    The beam is released at j (a cantilever); the end force R and moment Mj at j are found from v_j = psi_j = 0 with
    the unit-load method. M(x) and V(x) are the moment and force resultants of the actions on [x, L].
    """
    gauss_x, gauss_w = np.polynomial.legendre.leggauss(12)
    breaks = sorted({0.0, length, *[s * length for _, _, s, _ in distributed],
                     *[e * length for _, _, _, e in distributed], *[a for _, a in forces], *[a for _, a in moments]})

    def integrate(f, lo, hi):
        total = 0.0
        pts = [lo] + [b for b in breaks if lo < b < hi] + [hi]
        for a, b in zip(pts[:-1], pts[1:]):
            total += 0.5 * (b - a) * sum(w * f(0.5 * (b - a) * t + 0.5 * (a + b)) for t, w in zip(gauss_x, gauss_w))
        return total

    def q(s):
        total = 0.0
        for w0, w1, st, en in distributed:
            a, b = st * length, en * length
            if a <= s <= b:
                total += w0 + (w1 - w0) * (s - a) / (b - a)
        return total

    def shear0(x):
        return integrate(q, x, length) + sum(p for p, a in forces if a > x)

    def moment0(x):
        return (integrate(lambda s: q(s) * (s - x), x, length) + sum(p * (a - x) for p, a in forces if a > x)
                + sum(m for m, a in moments if a > x))

    flex_v = (lambda x: 0.0) if kga is None else (lambda x: 1.0 / kga)
    f11 = integrate(lambda x: (length - x) ** 2 / ei + flex_v(x), 0.0, length)
    f12 = integrate(lambda x: (length - x) / ei, 0.0, length)
    f22 = length / ei
    d1 = integrate(lambda x: moment0(x) * (length - x) / ei + shear0(x) * flex_v(x), 0.0, length)
    d2 = integrate(lambda x: moment0(x) / ei, 0.0, length)
    r_j, m_j = np.linalg.solve([[f11, f12], [f12, f22]], [-d1, -d2])
    # equilibrium of the whole member (loads at x = 0 included) gives the reactions at i
    total_force = integrate(q, 0.0, length) + sum(p for p, _ in forces)
    total_moment_i = (integrate(lambda s: q(s) * s, 0.0, length) + sum(p * a for p, a in forces)
                      + sum(m for m, _ in moments))
    r_i = -(total_force + r_j)
    m_i = -(total_moment_i + r_j * length + m_j)
    return np.array([-r_i, -m_i, -r_j, -m_j])  # equivalent nodal loads = - fixed-end reactions


_DISTRIBUTED = [(-12.0, -4.0, 0.15, 0.7), (5.0, 5.0, 0.0, 1.0), (-8.0, 0.0, 0.4, 1.0)]
_POINTS = [(0.3, -15.0, 7.0), (0.0, 10.0, -4.0), (0.85, 6.0, 9.0)]  # (relative position, force, moment)


@pytest.mark.parametrize("shear", [True, False])
@pytest.mark.parametrize("w0, w1, start, end", _DISTRIBUTED)
def test_distributed_load_matches_flexibility_method(shear, w0, w1, start, end):
    member = _member(include_shear_deformation=shear)
    sec, G = member.section, member.material.G
    eq = DistributedLoad(member, w1=0.0, w2=w0, w3=0.5 * w0, w2_end=w1, w3_end=0.5 * w1, w1_end=0.0,
                         start=start, end=end, coord_system="local")._compute_equivalent_loads()
    # plane 1-2 (I3, As2): DOFs v2, theta3 = psi
    ref2 = fixed_end_loads(L, E * sec.I3, G * sec.As2 if shear else None, distributed=[(w0, w1, start, end)])
    np.testing.assert_allclose(eq[[1, 5, 7, 11]], ref2, rtol=1e-10, atol=1e-10)
    # plane 1-3 (I2, As3): DOFs v3, theta2 = -psi
    ref3 = fixed_end_loads(L, E * sec.I2, G * sec.As3 if shear else None,
                           distributed=[(0.5 * w0, 0.5 * w1, start, end)])
    np.testing.assert_allclose(eq[[2, 4, 8, 10]], ref3 * [1, -1, 1, -1], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("shear", [True, False])
@pytest.mark.parametrize("pos, force, moment", _POINTS)
def test_point_force_and_moment_match_flexibility_method(shear, pos, force, moment):
    member = _member(include_shear_deformation=shear)
    sec, G = member.section, member.material.G
    a = pos * L
    eq = ElementPointLoad(member, p1=0.0, p2=force, p3=-force, m1=0.0, m2=-moment, m3=moment, x=a,
                          coord_system="local")._compute_equivalent_loads()
    ref2 = fixed_end_loads(L, E * sec.I3, G * sec.As2 if shear else None, forces=[(force, a)], moments=[(moment, a)])
    np.testing.assert_allclose(eq[[1, 5, 7, 11]], ref2, rtol=1e-10, atol=1e-10)
    # plane 1-3: a moment m2 about +2 turns the 1-3 plane the other way (its normal is 1 x 3 = -2)
    ref3 = fixed_end_loads(L, E * sec.I2, G * sec.As3 if shear else None, forces=[(-force, a)],
                           moments=[(moment, a)])
    np.testing.assert_allclose(eq[[2, 4, 8, 10]], ref3 * [1, -1, 1, -1], rtol=1e-10, atol=1e-10)


def test_moment_at_the_start_node_goes_to_its_rotation():
    """Regression: a member moment at x = 0 was applied to the start node with the wrong sign."""
    member = _member(include_shear_deformation=False)
    eq = ElementPointLoad(member, m2=3.0, m3=5.0, x=0.0, coord_system="local")._compute_equivalent_loads()
    expected = np.zeros(12)
    expected[4], expected[5] = 3.0, 5.0
    np.testing.assert_allclose(eq, expected, atol=1e-14)


# ── 3. member point load = mesh with a node at the load ─────────────────────────────────────────────────────────


def _solve(with_mesh, releases_j):
    structure = Structure()
    a, b = Node("A", 0.0, 0.0, 0.0), Node("B", L, 0.0, 0.0)
    nodes = [a, b] + ([Node("P", 0.3 * L, 0.0, 0.0)] if with_mesh else [])
    for n in nodes:
        structure.add_node(n)
    a.set_support(ux_fixed=True, uy_fixed=True, uz_fixed=True, rx_fixed=True, ry_fixed=True, rz_fixed=True)
    # B: free axially; its rotations are restrained so that a moment release at B leaves no free rotation
    b.set_support(uy_fixed=True, uz_fixed=True, rx_fixed=True, ry_fixed=True, rz_fixed=True)
    load = dict(fx=2.0, fy=-6.0, fz=-15.0, mx=1.5, my=4.0, mz=-7.0)
    if with_mesh:
        structure.add_element(_member(eid=1, node_i=a, node_j=nodes[2], length=None))
        structure.add_element(_member(eid=2, node_i=nodes[2], node_j=b, releases_j=releases_j))
        structure.add_load(PointLoad(nodes[2], **load))
    else:
        member = _member(eid=1, node_i=a, node_j=b, releases_j=releases_j)
        structure.add_element(member)
        structure.add_load(ElementPointLoad(member, px=load["fx"], py=load["fy"], pz=load["fz"], mx=load["mx"],
                                            my=load["my"], mz=load["mz"], x=0.3 * L, coord_system="global"))
    structure.solve()
    results = Results(structure)
    reactions = results.reactions().set_index("node").loc[["A", "B"], ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]]
    displacements = results.node_displacements().set_index("node").loc[["B"], ["ux"]]
    return reactions.to_numpy(), displacements.to_numpy()


@pytest.mark.parametrize("releases_j", [None, [False, False, False, False, True, True]])
def test_member_point_load_equals_nodal_load_on_a_mesh(releases_j):
    reactions_member, disp_member = _solve(False, releases_j)
    reactions_mesh, disp_mesh = _solve(True, releases_j)
    np.testing.assert_allclose(reactions_member, reactions_mesh, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(disp_member, disp_mesh, rtol=1e-9, atol=1e-15)
