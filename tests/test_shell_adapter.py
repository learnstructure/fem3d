"""Shell elements through the struct_core adapter (fem3d/SHELL_PLAN.md, M5): model_from_core, result_to_core, and
model_to_core, with shell loads, self-weight, and the mass source."""

import numpy as np
import pytest

from struct_core import (
    ConcreteMaterial,
    LoadCase,
    MassSource,
    MassSourceFactor,
    Node,
    ShellElement,
    ShellLoad,
    ShellSection,
    StructuralModel,
    Support,
)

from fem3d.adapters.struct_core_adapter import model_from_core, model_to_core, result_to_core

A, N, H, Q_LIVE, GAMMA = 4.0, 4, 0.2, 5.0, 24.0  # kN, m


def plate_model(**model_kw):
    """Simply supported 4 x 4 m slab (t = 0.2 m, concrete 24 kN/m3), 4 x 4 elements, live load 5 kPa downward."""
    nodes, supports = [], []
    for j in range(N + 1):
        for i in range(N + 1):
            nid = j * (N + 1) + i + 1
            x, y = A * i / N, A * j / N
            nodes.append(Node(id=nid, x=x, y=y, z=0.0))
            on_x, on_y = i in (0, N), j in (0, N)
            if on_x or on_y:
                supports.append(Support(node_id=nid, ux=nid == 1, uy=nid in (1, N + 1), uz=True, rx=on_x, ry=on_y))
    shells = [
        ShellElement(id=f"S{j * N + i + 1}", section_id="SLAB",
                     node_ids=[j * (N + 1) + i + 1, j * (N + 1) + i + 2, (j + 1) * (N + 1) + i + 2, (j + 1) * (N + 1) + i + 1])
        for j in range(N) for i in range(N)
    ]
    return StructuralModel(
        units="kN, m",
        nodes=nodes,
        supports=supports,
        materials=[ConcreteMaterial(id="C30", E=25e6, fc=30e3, unit_weight=GAMMA)],
        shell_sections=[ShellSection(id="SLAB", material_id="C30", thickness=H)],
        shell_elements=shells,
        load_cases=[
            LoadCase(id="DEAD", include_self_weight=True),
            LoadCase(id="LIVE", shell_loads=[ShellLoad(element_ids=[s.id for s in shells], pressure=-Q_LIVE,
                                                       direction="z")]),
        ],
        **model_kw,
    )


@pytest.mark.parametrize("case, total", [("DEAD", GAMMA * H * A * A), ("LIVE", Q_LIVE * A * A)])
def test_shell_loads_and_self_weight_reach_the_solver(case, total):
    structure = model_from_core(plate_model(), load_case_id=case)
    assert len(structure.shells) == N * N
    structure.solve()
    reactions = structure.reactions.reshape(-1, 6)
    assert reactions[:, 2].sum() == pytest.approx(total, rel=1e-10)
    assert np.allclose(reactions[:, :2].sum(axis=0), 0.0, atol=1e-8 * total)


def test_result_to_core_reports_gauss_corner_and_node_resultants():
    structure = model_from_core(plate_model(), load_case_id="LIVE")
    structure.solve()
    result = result_to_core(structure)
    direct = structure.shell_resultants()
    assert set(result.shell_results) == set(direct)
    for sid, res in result.shell_results.items():
        assert np.allclose([f.as_list() for f in res.gauss_points], direct[sid]["gauss_points"])
        assert np.allclose([f.as_list() for f in res.corners], direct[sid]["corners"])
    center = (N // 2) * (N + 1) + N // 2 + 1
    groups = result.shell_node_results[center]
    assert len(groups) == 1 and len(groups[0].element_ids) == 4  # one plane, four elements averaged
    assert groups[0].forces.M11 > 0 and groups[0].forces.M11 == pytest.approx(groups[0].forces.M22, rel=1e-9)


def test_model_to_core_round_trip_keeps_shells_and_their_loads():
    structure = model_from_core(plate_model(), load_case_id="LIVE")
    structure.solve()
    project = model_to_core(structure, units="kN, m")
    model = project.model
    assert sorted(s.id for s in model.shell_elements) == sorted(structure.shells)
    assert len(model.shell_sections) == 1 and model.shell_sections[0].thickness == H
    assert sum(len(sl.element_ids) for lc in model.load_cases for sl in lc.shell_loads) == N * N
    again = model_from_core(project)
    again.solve()
    for nid, node in structure.nodes.items():
        assert np.allclose(again.disp[again.nodes[nid].dofs], structure.disp[node.dofs], atol=1e-14)


def test_mass_source_carries_the_shell_mass_once():
    """With a mass source, the shells' self-mass is lumped at the nodes by struct_load (CSI §10.11) and the shell
    elements carry none, so the total mass is gamma t A / g, not twice that."""
    model = plate_model(mass_source=MassSource(include_element_mass=True, factors=[]))
    structure = model_from_core(model, load_case_id="LIVE")
    assert all(sh.material.rho == 0.0 for sh in structure.shells.values())
    structure.number_dofs()
    structure.assemble_mass_matrix(lumped=True)
    M = structure.M.toarray()
    g = 9.80665
    for d in range(3):
        idx = np.arange(d, structure.neq, 6)
        assert M[np.ix_(idx, idx)].sum() == pytest.approx(GAMMA * H * A * A / g, rel=1e-6)

    without = model_from_core(plate_model(), load_case_id="LIVE")  # no mass source: the material density
    assert all(sh.material.rho == pytest.approx(GAMMA / g) for sh in without.shells.values())
