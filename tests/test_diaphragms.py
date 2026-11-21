"""Verification tests for Milestone 2 rigid floor diaphragms in fem3d."""

import copy
import math

import numpy as np
import pytest
from scipy.linalg import eigh

from struct_core import (
    BeamElement,
    ConcreteMaterial,
    Diaphragm,
    DistributedLoad,
    ElasticMaterial as ScElasticMaterial,
    GeneralSection,
    LoadCase,
    MassSource,
    MassSourceFactor,
    Node as ScNode,
    PointLoad as ScPointLoad,
    Project,
    RectangularSection,
    StructuralModel,
    Support,
    convert_project_units,
)

from fem3d import (
    DistributedLoad as FemDistributedLoad,
    ElasticMaterial,
    FrameElement,
    Node,
    PointLoad,
    RigidDiaphragm,
    Section,
    Structure,
    TrussElement,
    buckling_analysis,
    model_from_core,
    model_to_core,
    result_to_core,
)


def _permute_vec(xyz, axis: str):
    x, y, z = xyz
    if axis == "Z":
        return (x, y, z)
    if axis == "X":
        # (X_new, Y_new, Z_new) <- (Z_old, X_old, Y_old): normal is X_new
        return (z, x, y)
    if axis == "Y":
        # (X_new, Y_new, Z_new) <- (Y_old, Z_old, X_old): normal is Y_new
        return (y, z, x)
    raise ValueError(axis)


def _eccentric_single_story_system(*, axis: str = "Z"):
    """
    Build an exact 3-DOF (coupled u1, u2, r3) single-story 4-column system with:
      - zero frame density (mass lumped purely at roof nodes so M_3x3 is exact),
      - unequal isotropic column flexural stiffnesses (I2 == I3 per column) across corners,
      - asymmetric roof nodal masses and torsional inertias,
      - out-of-plane DOFs (u3, r1, r2) fixed at the roof so exact 3x3 closed-form equations hold.
    """
    E = 3.0e7
    nu = 0.25
    G = E / (2.0 * (1.0 + nu))
    H = 4.0
    Lx, Ly = 8.0, 6.0

    # Distinct column properties on the 4 corners (creating stiffness eccentricity in both X and Y)
    col_specs = [
        # (x, y, I_bend, J_torsion, nodal_mass, nodal_Iz)
        (0.0, 0.0, 4.0e-3, 1.2e-3, 30.0, 18.0),
        (Lx, 0.0, 1.5e-3, 0.5e-3, 15.0, 7.0),
        (Lx, Ly, 1.0e-3, 0.4e-3, 10.0, 5.0),
        (0.0, Ly, 2.5e-3, 0.9e-3, 25.0, 12.0),
    ]

    s = Structure(units="kN, m")
    mat = ElasticMaterial(E=E, nu=nu, rho=0.0)

    top_nodes = []
    columns = []
    for idx, (x, y, I_b, J_t, m_val, iz_val) in enumerate(col_specs, start=1):
        b_node = Node(idx, *_permute_vec((x, y, 0.0), axis))
        b_node.set_support(True, True, True, True, True, True)
        t_node = Node(10 + idx, *_permute_vec((x, y, H), axis))
        # Restrain out-of-plane DOFs [u3, r1, r2], leaving in-plane DOFs free for diaphragm D1
        if axis == "Z":
            t_node.set_support(uz_fixed=True, rx_fixed=True, ry_fixed=True)
            t_node.set_mass(mass=[m_val, m_val, m_val], inertia=[0.0, 0.0, iz_val])
        elif axis == "X":
            t_node.set_support(ux_fixed=True, ry_fixed=True, rz_fixed=True)
            t_node.set_mass(mass=[m_val, m_val, m_val], inertia=[iz_val, 0.0, 0.0])
        else:  # "Y"
            t_node.set_support(uy_fixed=True, rx_fixed=True, rz_fixed=True)
            t_node.set_mass(mass=[m_val, m_val, m_val], inertia=[0.0, iz_val, 0.0])

        s.add_node(b_node)
        s.add_node(t_node)
        sec = Section(A=0.25, I2=I_b, I3=I_b, J=J_t, material=mat)
        el = FrameElement(f"C{idx}", b_node, t_node, mat, sec, include_shear_deformation=False)
        s.add_element(el)
        top_nodes.append(t_node)
        columns.append((x, y, I_b, J_t, m_val, iz_val))

    d = s.add_diaphragm("D1", [n.id for n in top_nodes], axis=axis, name="Roof")
    assert isinstance(d, RigidDiaphragm)

    # Asymmetric roof loads expressed in diaphragm (1, 2, 3) axes, then permuted to global
    loads_123 = [
        (40.0, -15.0, 12.0),
        (25.0, 30.0, -5.0),
        (-10.0, 20.0, 8.0),
        (5.0, 15.0, 3.0),
    ]
    for nd, (f1, f2, m3) in zip(top_nodes, loads_123):
        fx, fy, fz = _permute_vec((f1, f2, 0.0), axis)
        mx, my, mz = _permute_vec((0.0, 0.0, m3), axis)
        s.add_load(PointLoad(nd, fx=fx, fy=fy, fz=fz, mx=mx, my=my, mz=mz))

    # Analytical 3x3 system about diaphragm center of mass in (1, 2, 3)
    masses = np.array([c[4] for c in columns])
    xs = np.array([c[0] for c in columns])
    ys = np.array([c[1] for c in columns])
    izs = np.array([c[5] for c in columns])
    m_tot = float(np.sum(masses))
    x_cm = float(np.dot(masses, xs) / m_tot)
    y_cm = float(np.dot(masses, ys) / m_tot)

    K3 = np.zeros((3, 3), dtype=float)
    M3 = np.zeros((3, 3), dtype=float)
    F3 = np.zeros(3, dtype=float)
    for (x, y, I_b, J_t, m_val, iz_val), (f1, f2, m3) in zip(columns, loads_123):
        dx = x - x_cm
        dy = y - y_cm
        Ti = np.array([[1.0, 0.0, -dy], [0.0, 1.0, dx], [0.0, 0.0, 1.0]])
        k_col = np.diag([12.0 * E * I_b / H**3, 12.0 * E * I_b / H**3, G * J_t / H])
        m_col = np.diag([m_val, m_val, iz_val])
        K3 += Ti.T @ k_col @ Ti
        M3 += Ti.T @ m_col @ Ti
        F3 += Ti.T @ np.array([f1, f2, m3])

    q_static = np.linalg.solve(K3, F3)
    return s, top_nodes, columns, (x_cm, y_cm, H), K3, M3, F3, q_static


# ── 1. Closed-form single-story eccentric diaphragm (static, modal, RSA, 3 planes) ────────────────────────


@pytest.mark.parametrize("axis", ["Z", "X", "Y"])
def test_closed_form_single_story_eccentric_diaphragm_static_modal_and_rsa(axis: str):
    s, top_nodes, columns, (x_cm, y_cm, _H), K3, M3, _F3, q_static = _eccentric_single_story_system(axis=axis)
    axis_123 = {"Z": (0, 1, 5), "X": (1, 2, 3), "Y": (2, 0, 4)}[axis]
    d1_idx, d2_idx, r3_idx = axis_123

    # Static solve + displacement recovery at every corner + global 6-DOF equilibrium
    disp, reactions = s.solve()
    assert disp.shape == (s.neq,)
    u1_m, u2_m, r3_m = q_static
    for nd, (x, y, *_rest) in zip(top_nodes, columns):
        assert disp[nd.dofs[d1_idx]] == pytest.approx(u1_m - (y - y_cm) * r3_m, rel=1e-12, abs=1e-14)
        assert disp[nd.dofs[d2_idx]] == pytest.approx(u2_m + (x - x_cm) * r3_m, rel=1e-12, abs=1e-14)
        assert disp[nd.dofs[r3_idx]] == pytest.approx(r3_m, rel=1e-12, abs=1e-14)

    # 6-component global equilibrium including moments about the origin
    total_reac = np.zeros(6, dtype=float)
    total_load = np.zeros(6, dtype=float)
    for nd in s.nodes.values():
        r_nd = reactions[nd.dofs]
        pos = np.array([nd.x, nd.y, nd.z])
        total_reac[:3] += r_nd[:3]
        total_reac[3:] += r_nd[3:] + np.cross(pos, r_nd[:3])
        f_nd = s.F[nd.dofs]
        total_load[:3] += f_nd[:3]
        total_load[3:] += f_nd[3:] + np.cross(pos, f_nd[:3])
    np.testing.assert_allclose(total_reac + total_load, 0.0, atol=1e-10)

    # Modal verification against exact 3x3 generalized eigensolution
    modal = s.modal_analysis(num_modes=3, lumped=True, include_rotational_inertia=False)
    evals, evecs = eigh(K3, M3)
    omega_exact = np.sqrt(evals)
    np.testing.assert_allclose(modal["omega"], omega_exact, rtol=1e-12)
    np.testing.assert_allclose(modal["periods"], 2.0 * np.pi / omega_exact, rtol=1e-12)

    # Check master placed at exact COM and scaled rotational mass matches I_m / L_d^2
    meta = s.diaphragm_masters["D1"]
    mx_cm, my_cm, mz_cm = _permute_vec((x_cm, y_cm, 4.0), axis)
    np.testing.assert_allclose(meta["master_coords"], (mx_cm, my_cm, mz_cm), rtol=1e-12, atol=1e-12)
    _, M_red = s.get_reduced_matrices(lumped=True)
    L_d = meta["scale_radius"]
    assert M_red[2, 2] * L_d**2 == pytest.approx(M3[2, 2], rel=1e-12)
    assert abs(M_red[0, 2]) < 1e-12 and abs(M_red[1, 2]) < 1e-12

    # RSA along in-plane axis 1 (with 5% damping, CQC and SRSS)
    sa_vals = np.array([1.2, 0.85, 0.60])
    dir1 = {"Z": "x", "X": "y", "Y": "z"}[axis]
    r_3 = np.array([1.0, 0.0, 0.0])

    for combo in ("CQC", "SRSS"):
        rsa_res = s.response_spectrum_analysis(Sa=sa_vals, ji=0.05, direction=dir1, combination=combo)
        assert rsa_res["effective_mass"].sum() == pytest.approx(M3[0, 0], rel=1e-12)
        assert rsa_res["mass_participation_ratio"].sum() == pytest.approx(1.0, rel=1e-12)

        # Analytical 3-mode RSA displacements at corner nodes
        gamma = np.array([(evecs[:, m] @ M3 @ r_3) / (evecs[:, m] @ M3 @ evecs[:, m]) for m in range(3)])
        sd = sa_vals / (omega_exact**2)
        u3_modal = evecs * (gamma * sd)[None, :]
        if combo == "CQC":
            rho = rsa_res["rsa"].compute_cqc_matrix()
        else:
            rho = np.eye(3)

        u1_corners = []
        for nd, (x, y, *_rest) in zip(top_nodes, columns):
            Ti = np.array([[1.0, 0.0, -(y - y_cm)], [0.0, 1.0, x - x_cm], [0.0, 0.0, 1.0]])
            nodal_modal = Ti @ u3_modal
            expected_comb = np.sqrt(np.maximum(0.0, np.einsum("ki,ij,kj->k", nodal_modal, rho, nodal_modal)))
            assert s.disp[nd.dofs[d1_idx]] == pytest.approx(expected_comb[0], rel=1e-12, abs=1e-14)
            assert s.disp[nd.dofs[d2_idx]] == pytest.approx(expected_comb[1], rel=1e-12, abs=1e-14)
            assert s.disp[nd.dofs[r3_idx]] == pytest.approx(expected_comb[2], rel=1e-12, abs=1e-14)
            u1_corners.append(expected_comb[0])

        # Flexible edge must experience larger displacement than stiff edge (expansion before combination!)
        assert u1_corners[2] > u1_corners[0]

        vb_modes = np.array([(r_3 @ M3 @ evecs[:, m]) * gamma[m] * (omega_exact[m] ** 2) * sd[m] for m in range(3)])
        vb_expected = math.sqrt(float(vb_modes @ rho @ vb_modes))
        assert rsa_res["combined_base_shear"][0] == pytest.approx(vb_expected, rel=1e-12)


# ── 2. Multi-story frame, kinematics, origin shift, units, springs, truss nodes ───────────────────────────


def _two_story_core_model(*, with_diaphragms: bool = True, asymmetric_mass: bool = True) -> Project:
    Lx, Ly = 6.0, 4.0
    zs = [0.0, 3.5, 7.0]
    nodes = []
    for k, z in enumerate(zs):
        for j, y in enumerate((0.0, Ly)):
            for i, x in enumerate((0.0, Lx)):
                nid = 100 * k + 10 * j + i + 1
                m = 0.0
                it = 0.0
                if k > 0 and asymmetric_mass:
                    m = 12.0 if i == 0 else 4.0
                    it = 6.0 if j == 0 else 2.0
                nodes.append(ScNode(id=nid, x=x, y=y, z=z, mass=m, inertia=it))

    supports = [Support.fixed_3d(1), Support.fixed_3d(2), Support.fixed_3d(11), Support.fixed_3d(12)]
    elems = []
    # Columns
    for k in (0, 1):
        for offset in (1, 2, 11, 12):
            elems.append(BeamElement(
                id=f"C_{k}_{offset}",
                start_node=100 * k + offset,
                end_node=100 * (k + 1) + offset,
                section_id="COL",
            ))
    # Beams at levels 1 and 2
    for k in (1, 2):
        b = 100 * k
        elems.extend([
            BeamElement(id=f"BX1_{k}", start_node=b + 1, end_node=b + 2, section_id="BM"),
            BeamElement(id=f"BX2_{k}", start_node=b + 11, end_node=b + 12, section_id="BM"),
            BeamElement(id=f"BY1_{k}", start_node=b + 1, end_node=b + 11, section_id="BM"),
            BeamElement(id=f"BY2_{k}", start_node=b + 2, end_node=b + 12, section_id="BM"),
        ])

    diaphragms = (
        [
            Diaphragm(id="D1", name="Floor_1", node_ids=[101, 102, 111, 112], axis="Z"),
            Diaphragm(id="D2", name="Roof", node_ids=[201, 202, 211, 212], axis="Z"),
        ]
        if with_diaphragms
        else []
    )
    model = StructuralModel(
        dimension="3D",
        vertical_axis="Z",
        units="kN, m",
        nodes=nodes,
        supports=supports,
        materials=[ConcreteMaterial(id="C30", E=25.0e6, nu=0.2, fc=30.0e3, unit_weight=24.0)],
        sections=[
            RectangularSection(id="COL", material_id="C30", b=0.45, h=0.45),
            RectangularSection(id="BM", material_id="C30", b=0.30, h=0.55),
        ],
        elements=elems,
        diaphragms=diaphragms,
        load_cases=[
            LoadCase(
                id="LATERAL",
                load_type="wind",
                point_loads=[
                    ScPointLoad(node_id=101, fx=50.0, fy=15.0, fz=-40.0),
                    ScPointLoad(node_id=202, fx=80.0, fy=-10.0, fz=-60.0),
                ],
                element_loads=[DistributedLoad(element_id="BX1_2", wz=-12.0)],
            )
        ],
    )
    return Project(model=model, units="kN, m")


def test_multistory_kinematics_independent_out_of_plane_origin_and_units():
    proj = _two_story_core_model(with_diaphragms=True)
    s_m = model_from_core(proj, include_self_weight=False)
    disp_m, _ = s_m.solve()

    # Check exact rigid-body in-plane kinematics while out-of-plane uz, rx, ry remain non-uniform
    for floor_nodes in ([101, 102, 111, 112], [201, 202, 211, 212]):
        n0 = s_m.nodes[floor_nodes[0]]
        u0 = disp_m[n0.dofs]
        for nid in floor_nodes[1:]:
            ni = s_m.nodes[nid]
            ui = disp_m[ni.dofs]
            assert ui[0] == pytest.approx(u0[0] - (ni.y - n0.y) * u0[5], abs=1e-12)
            assert ui[1] == pytest.approx(u0[1] + (ni.x - n0.x) * u0[5], abs=1e-12)
            assert ui[5] == pytest.approx(u0[5], abs=1e-12)
        uzs = [disp_m[s_m.nodes[nid].dofs[2]] for nid in floor_nodes]
        assert max(uzs) - min(uzs) > 1e-6

    modal_m = s_m.modal_analysis(num_modes=6, lumped=True)

    # Consistent mass matrix also runs cleanly with diaphragms
    modal_cons = s_m.modal_analysis(num_modes=6, lumped=False)
    assert len(modal_cons["periods"]) == 6
    assert np.all(np.diff(modal_cons["omega"]) >= -1e-12)

    # Coordinate origin translation invariance
    shifted = proj.model_copy(deep=True)
    for n in shifted.model.nodes:
        n.x += 123.4
        n.y -= 56.7
        n.z += 10.0
    s_shift = model_from_core(shifted, include_self_weight=False)
    disp_shift, _ = s_shift.solve()
    modal_shift = s_shift.modal_analysis(num_modes=6, lumped=True)
    np.testing.assert_allclose(disp_shift, disp_m, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(modal_shift["periods"], modal_m["periods"], rtol=1e-11)

    # Unit invariance: N, mm and kip, in (especially checking L_d scaling in mm)
    for unit_target, len_scale in (("N, mm", 1000.0), ("kip, in", 1.0 / 0.0254)):
        proj_u = convert_project_units(proj, unit_target, in_place=False)
        s_u = model_from_core(proj_u, include_self_weight=False)
        disp_u, _ = s_u.solve()
        modal_u = s_u.modal_analysis(num_modes=6, lumped=True)
        np.testing.assert_allclose(modal_u["periods"], modal_m["periods"], rtol=1e-11)
        for nid in s_m.nodes:
            u_orig = disp_m[s_m.nodes[nid].dofs]
            u_conv = disp_u[s_u.nodes[nid].dofs]
            np.testing.assert_allclose(u_conv[:3], u_orig[:3] * len_scale, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(u_conv[3:], u_orig[3:], rtol=1e-10, atol=1e-10)


def test_beam_axial_force_inside_rigid_diaphragm():
    """Wilson Ch. 7 §7.4–7.5: unloaded beams in a rigid diaphragm have P = 0; axially loaded beams carry -f_eq."""
    s = Structure(units="kN, m")
    mat = ElasticMaterial(E=25.0e6, nu=0.2, rho=0.0)
    col_sec = Section.from_rectangle(0.4, 0.4, mat)
    bm_sec = Section.from_rectangle(0.3, 0.5, mat)
    b1, b2 = Node(1, 0.0, 0.0, 0.0), Node(2, 6.0, 0.0, 0.0)
    t1, t2 = Node(3, 0.0, 0.0, 3.5), Node(4, 6.0, 0.0, 3.5)
    b1.set_support(True, True, True, True, True, True)
    b2.set_support(True, True, True, True, True, True)
    for n in (b1, b2, t1, t2):
        s.add_node(n)

    c1 = FrameElement("C1", b1, t1, mat, col_sec)
    c2 = FrameElement("C2", b2, t2, mat, col_sec)
    bm = FrameElement("B1", t1, t2, mat, bm_sec)
    for el in (c1, c2, bm):
        s.add_element(el)
    s.add_diaphragm("D1", [3, 4], axis="Z")
    # Apply lateral load ONLY to node 3 plus gravity load on B1:
    # Without a diaphragm, the lateral push compresses B1 (~50% of H). With a rigid diaphragm, P == 0.
    s.add_load(PointLoad(t1, fx=100.0, fy=30.0))
    s.add_load(FemDistributedLoad(bm, wz=-15.0, coord_system="global"))

    s.solve()
    f_bm = bm.get_local_forces()
    assert abs(f_bm[0]) < 1e-10
    assert abs(f_bm[6]) < 1e-10
    assert abs(f_bm[2]) < 1e-10 and abs(f_bm[8]) < 1e-10  # minor-axis V3 = 0
    assert abs(f_bm[4]) < 1e-10 and abs(f_bm[10]) < 1e-10  # minor-axis M2 = 0

    # Now add direct local axial distributed load w1 = +10 kN/m along the beam:
    # Since u1_j - u1_i == 0, local forces equal fixed-end axial reactions f1_i = f1_j = -w1 * L / 2.
    bm2_load = FemDistributedLoad(bm, w1=10.0, coord_system="local")
    s.add_load(bm2_load)
    bm.f_local = None
    s.solve()
    f_bm_axial = bm.get_local_forces()
    assert f_bm_axial[0] == pytest.approx(-10.0 * 6.0 / 2.0, rel=1e-12)
    assert f_bm_axial[6] == pytest.approx(-10.0 * 6.0 / 2.0, rel=1e-12)


def test_elastic_springs_and_truss_node_inside_rigid_diaphragm():
    proj = _two_story_core_model(with_diaphragms=True, asymmetric_mass=False)
    # Add elastic horizontal & rotational spring support at diaphragm node 101
    proj.model.supports.append(Support(node_id=101, uz=True, kx=5000.0, ky=3000.0, krz=1000.0))
    s = model_from_core(proj, include_self_weight=False)
    disp, reactions = s.solve()

    u101 = disp[s.nodes[101].dofs]
    r101 = reactions[s.nodes[101].dofs]
    assert r101[0] == pytest.approx(-5000.0 * u101[0], rel=1e-12)
    assert r101[1] == pytest.approx(-3000.0 * u101[1], rel=1e-12)
    assert r101[5] == pytest.approx(-1000.0 * u101[5], rel=1e-12)

    # Check global horizontal + vertical equilibrium across fixed supports + springs
    total_reac = np.zeros(3, dtype=float)
    total_load = np.zeros(3, dtype=float)
    for nd in s.nodes.values():
        total_reac += reactions[nd.dofs][:3]
        total_load += s.F[nd.dofs][:3]
    np.testing.assert_allclose(total_reac + total_load, 0.0, atol=1e-10)

    # Truss-only node inside a diaphragm must not have its in-plane DOFs auto-fixed
    s2 = Structure(units="kN, m")
    mat = ElasticMaterial(E=2.0e8, nu=0.3, rho=7.85)
    sec = Section.from_rectangle(0.2, 0.2, mat)
    n1, n2, n3, n4 = Node(1, 0, 0, 0), Node(2, 4, 0, 0), Node(3, 0, 0, 3), Node(4, 4, 0, 3)
    n1.set_support(True, True, True, True, True, True)
    n2.set_support(True, True, True, False, False, False)
    for n in (n1, n2, n3, n4):
        s2.add_node(n)
    s2.add_element(FrameElement("C1", n1, n3, mat, sec))
    s2.add_element(TrussElement("T1", n2, n4, sec))
    s2.add_element(TrussElement("B1", n3, n4, sec))
    s2.add_diaphragm("D1", [3, 4], axis="Z")
    s2.add_load(PointLoad(n3, fx=20.0))
    disp2, _ = s2.solve()
    assert disp2[n4.dofs[0]] == pytest.approx(disp2[n3.dofs[0]], rel=1e-12)
    assert abs(disp2[n4.dofs[0]]) > 1e-6


def test_buckling_analysis_with_rigid_diaphragm():
    E = 2.0e7
    I_b = 2.0e-3
    H = 4.0
    Lx, Ly = 6.0, 4.0
    P0 = 100.0

    s = Structure(units="kN, m")
    mat = ElasticMaterial(E=E, nu=0.2, rho=0.0)
    sec = Section(A=0.25, I2=I_b, I3=I_b, J=1.0e-3, material=mat)
    top_nodes = []
    for idx, (x, y) in enumerate([(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)], start=1):
        bn = Node(idx, x, y, 0.0)
        bn.set_support(True, True, True, True, True, True)
        tn = Node(10 + idx, x, y, H)
        tn.set_support(uz_fixed=False, rx_fixed=True, ry_fixed=True)
        s.add_node(bn)
        s.add_node(tn)
        s.add_element(FrameElement(f"C{idx}", bn, tn, mat, sec, include_shear_deformation=False))
        s.add_load(PointLoad(tn, fz=-P0))
        top_nodes.append(tn)

    s.add_diaphragm("D1", [n.id for n in top_nodes], axis="Z")
    factors, modes = buckling_analysis(s, num_modes=3)
    assert modes.shape == (s.neq, 3)

    # Pure translational sway critical multiplier for 1-element fixed-guided column = 10*E*I / (P0 * H^2)
    lambda_sway = 10.0 * E * I_b / (P0 * H**2)
    assert factors[0] == pytest.approx(lambda_sway, rel=1e-12)

    # Buckling mode shapes satisfy exact rigid diaphragm kinematics
    for m in range(3):
        u0 = modes[top_nodes[0].dofs, m]
        for tn in top_nodes[1:]:
            ui = modes[tn.dofs, m]
            assert ui[0] == pytest.approx(u0[0] - (tn.y - top_nodes[0].y) * u0[5], abs=1e-11)
            assert ui[1] == pytest.approx(u0[1] + (tn.x - top_nodes[0].x) * u0[5], abs=1e-11)
            assert ui[5] == pytest.approx(u0[5], abs=1e-11)


def test_adapter_roundtrip_and_zero_diaphragm_identity():
    # Adapter round-trip with diaphragms
    proj = _two_story_core_model(with_diaphragms=True)
    s = model_from_core(proj, include_self_weight=False)
    s.solve()
    res = result_to_core(s)
    assert set(res.node_results.keys()) == {n.id for n in proj.model.nodes}
    back_proj = model_to_core(s)
    assert len(back_proj.model.diaphragms) == 2
    assert back_proj.model.get_diaphragm("D1").node_ids == [101, 102, 111, 112]
    assert back_proj.model.get_diaphragm("D2").name == "Roof"

    # Zero-diaphragm identity regression: model without diaphragms takes free_dofs fast path identically
    proj_no_d = _two_story_core_model(with_diaphragms=False)
    s_no_d = model_from_core(proj_no_d, include_self_weight=False)
    assert not s_no_d.diaphragms
    disp_nd, reac_nd = s_no_d.solve()
    assert s_no_d.n_reduced_dofs == len(s_no_d.free_dofs)
    K_ff, M_ff = s_no_d.get_reduced_matrices(lumped=True)
    free = np.asarray(s_no_d.free_dofs, dtype=np.int64)
    np.testing.assert_array_equal(K_ff.toarray(), s_no_d.K[free][:, free].toarray())
    np.testing.assert_array_equal(M_ff.toarray(), s_no_d.M[free][:, free].toarray())
    modal_nd = s_no_d.modal_analysis(num_modes=4, lumped=True)
    rsa_nd = s_no_d.response_spectrum_analysis(Sa=[1.0, 0.8, 0.5, 0.3], n_modes=4, direction="x")
    assert len(modal_nd["periods"]) == 4
    assert rsa_nd["combined_displacements"].shape == (len(free),)
    np.testing.assert_array_equal(s_no_d.expand_displacements(disp_nd[free]), disp_nd)
    assert np.sum(np.abs(reac_nd)) > 0.0
