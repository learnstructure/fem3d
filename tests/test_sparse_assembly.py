"""
Sparse assembly and solution (SHELL_PLAN.md, M2). The sparse global matrices must equal the dense assembly of the
same element matrices to round-off, and the static and modal results must equal the dense solutions.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import structdyn.mdf.mdf as mdf_module
from fem3d import SimpleFrame
from structdyn.mdf.mdf import DENSE_DOF_LIMIT, MDF
sys.path.insert(0, str(Path(__file__).parent))
from test_frame_modal_sap2000 import build_3d_frame  # noqa: E402


def dense_stiffness(structure):
    """The former dense assembly: element by element, then support springs on the diagonal."""
    K = np.zeros((structure.neq, structure.neq))
    for el in structure.elements.values():
        dofs = el.node_i.dofs + el.node_j.dofs
        K[np.ix_(dofs, dofs)] += el.global_stiffness()
    for node in structure.nodes.values():
        for i, k in enumerate(getattr(node, "springs", [])):
            if k > 0:
                K[node.dofs[i], node.dofs[i]] += k
    return K


def dense_mass(structure, lumped):
    M = np.zeros((structure.neq, structure.neq))
    for el in structure.elements.values():
        dofs = el.node_i.dofs + el.node_j.dofs
        M[np.ix_(dofs, dofs)] += el.mass_matrix(lumped=lumped)
    for node in structure.nodes.values():
        for i in range(3):
            M[node.dofs[i], node.dofs[i]] += node.mass[i]
            M[node.dofs[3 + i], node.dofs[3 + i]] += node.inertia[i]
    return M


def mixed_model():
    """A frame with a truss diagonal, a spring element, and support springs (kip, in)."""
    f = SimpleFrame()
    for nid, (x, y, z) in enumerate([(0, 0, 0), (200, 0, 0), (0, 0, 120), (200, 0, 120), (100, 150, 120),
                                     (300, 0, 0)], start=1):
        f.add_node(nid, x, y, z)
    f.add_support(1, [1, 1, 1, 1, 1, 1])
    f.add_support(2, [1, 1, 1, 0, 0, 0])
    f.add_support(6, [1, 1, 1, 1, 1, 1])
    f.structure.nodes[2].set_springs(krx=5e5, kry=5e5)
    f.add_frame(1, 1, 3, E=29000.0, A=20.0, I2=300.0, I3=500.0, J=40.0)
    f.add_frame(2, 2, 4, E=29000.0, A=20.0, I2=300.0, I3=500.0, J=40.0, roll_angle=30.0)
    f.add_frame(3, 3, 4, E=29000.0, A=25.0, I2=400.0, I3=900.0, J=60.0)
    f.add_truss(4, 1, 4, E=29000.0, A=5.0)
    f.add_truss(5, 3, 5, E=29000.0, A=5.0)
    f.add_truss(6, 4, 5, E=29000.0, A=5.0)
    f.add_truss(7, 2, 5, E=29000.0, A=5.0)
    f.add_spring(8, 2, 6, kx=1e3, ky=2e3, kz=3e3, krx=1e4, kry=1e4, krz=1e4)
    f.add_node_load(3, [10.0, 5.0, -20.0, 0.0, 0.0, 100.0])
    f.add_node_load(5, [0.0, 0.0, -15.0, 0.0, 0.0, 0.0])
    return f


def test_sparse_stiffness_equals_dense_assembly():
    for f in (mixed_model(), build_3d_frame(), build_3d_frame(include_shear_deformation=True)):
        s = f.structure
        s.solve()
        assert sp.isspmatrix_csr(s.K)
        ref = dense_stiffness(s)
        assert np.max(np.abs(s.K.toarray() - ref)) <= 1e-12 * np.max(np.abs(ref))


@pytest.mark.parametrize("lumped", [False, True])
def test_sparse_mass_equals_dense_assembly(lumped):
    s = build_3d_frame().structure
    s.number_dofs()
    s.assemble_mass_matrix(lumped=lumped)
    ref = dense_mass(s, lumped)
    assert sp.isspmatrix_csr(s.M)
    assert np.max(np.abs(s.M.toarray() - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_static_solution_and_reactions_equal_the_dense_solution():
    f = mixed_model()
    disp, reac = f.solve()
    s = f.structure
    K = dense_stiffness(s)
    free, fixed = s.free_dofs, s.fixed_dofs
    u = np.zeros(s.neq)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], s.F[free])
    assert np.allclose(disp, u, rtol=1e-10, atol=1e-14 * np.abs(u).max())
    r = np.zeros(s.neq)
    r[fixed] = K[np.ix_(fixed, free)] @ u[free] - s.F[fixed]
    for node in s.nodes.values():
        for i, k in enumerate(node.springs):
            if k > 0:
                r[node.dofs[i]] -= k * u[node.dofs[i]]
    assert np.allclose(reac, r, rtol=1e-10, atol=1e-10 * np.abs(r).max())


def test_truss_only_node_rotations_are_restrained_as_before():
    f = mixed_model()
    f.solve()
    assert f.structure.nodes[5].support[3:] == [True, True, True]  # node 5 has only trusses
    assert f.structure.nodes[3].support == [False] * 6


def test_singular_stiffness_is_reported():
    f = SimpleFrame()
    f.add_node(1, 0, 0, 0)
    f.add_node(2, 100, 0, 0)
    f.add_frame(1, 1, 2, E=29000.0, A=10.0, I2=50.0, I3=100.0, J=30.0)
    f.add_node_load(2, [1.0, 0, 0, 0, 0, 0])
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        f.solve()


def test_modal_analysis_is_the_same_on_both_sides_of_the_dense_limit(monkeypatch):
    # The SAP2000 4-story frame (test_frame_modal_sap2000.py) with lumped mass: massless rotations are condensed in
    # the dense path and handled by shift-invert in the sparse path.
    frame = build_3d_frame()
    dense = MDF.from_fem3d(frame, lumped=True)
    assert not dense.is_sparse
    w_dense, _ = dense.modal.modal_analysis(n_modes=6)
    monkeypatch.setattr(mdf_module, "DENSE_DOF_LIMIT", 10)
    sparse = MDF.from_fem3d(build_3d_frame(), lumped=True)
    assert sparse.is_sparse
    w_sparse, _ = sparse.modal.modal_analysis(n_modes=6)
    assert np.allclose(w_sparse, w_dense, rtol=1e-9)


def grid_frame():
    """A 10 x 10 bay, 4-story frame with 0.5 kip-s2/in at every floor node: 2904 free DOFs."""
    f = SimpleFrame()
    n = 11
    nid = lambda i, j, k: (k * n + j) * n + i + 1
    for k in range(5):
        for j in range(n):
            for i in range(n):
                f.add_node(nid(i, j, k), 240.0 * i, 240.0 * j, 144.0 * k)
                if k == 0:
                    f.add_support(nid(i, j, k), [1, 1, 1, 1, 1, 1])
    eid = 1
    props = dict(E=4000.0, A=400.0, I2=13333.0, I3=13333.0, J=22000.0)
    for k in range(1, 5):
        for j in range(n):
            for i in range(n):
                f.add_frame(eid, nid(i, j, k - 1), nid(i, j, k), **props)
                eid += 1
                if i < n - 1:
                    f.add_frame(eid, nid(i, j, k), nid(i + 1, j, k), **props)
                    eid += 1
                if j < n - 1:
                    f.add_frame(eid, nid(i, j, k), nid(i, j + 1, k), **props)
                    eid += 1
                f.structure.nodes[nid(i, j, k)].set_mass(0.5)
    return f


def test_large_frame_uses_the_sparse_eigensolver():
    f = grid_frame()
    t0 = time.perf_counter()
    res = f.structure.modal_analysis(num_modes=6, lumped=True)
    elapsed = time.perf_counter() - t0
    s = f.structure
    assert len(s.free_dofs) == 2904 > DENSE_DOF_LIMIT
    K_ff, M_ff = s.get_reduced_matrices(lumped=True)
    phi, omega = res["phi"], res["omega"]
    for i in range(6):
        r = K_ff @ phi[:, i] - omega[i] ** 2 * (M_ff @ phi[:, i])
        assert np.linalg.norm(r) <= 1e-8 * np.linalg.norm(K_ff @ phi[:, i])
    assert np.all(np.diff(omega) >= 0) and omega[0] > 0
    assert elapsed < 30.0


def test_response_spectrum_analysis_of_a_large_sparse_frame():
    f = grid_frame()
    sa = 0.5 * 386.09
    res = f.structure.response_spectrum_analysis(spectrum_curve=lambda T: sa, n_modes=12, direction="x")
    total_mass = 0.5 * 4 * 121
    participating = float(np.sum(res["mass_participation_ratio"]))
    assert 0.5 < participating <= 1.0
    # A constant spectrum: the combined base shear cannot exceed the participating mass times Sa
    vb = float(np.ravel(res["combined_base_shear"])[0])
    assert 0.5 * participating * total_mass * sa < vb <= participating * total_mass * sa * (1 + 1e-9)
