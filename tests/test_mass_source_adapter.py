"""Regression tests for MassSource nodal inertia preservation and include_element_mass in fem3d."""

from __future__ import annotations

import math

import numpy as np
import pytest
from struct_core import (
    BeamElement,
    DistributedLoad,
    ElasticMaterial,
    LoadCase,
    MassSource,
    MassSourceFactor,
    Node,
    PointLoad,
    Project,
    RectangularSection,
    ShellElement,
    ShellLoad,
    ShellSection,
    StructuralModel,
    Support,
)

from fem3d.adapters import model_from_core


def _cantilever_3d(
    *,
    node_mass: float = 0.0,
    node_inertia: float = 0.0,
    mass_source: MassSource,
    stiffness_modifiers: dict | None = None,
) -> Project:
    model = StructuralModel(
        dimension="3D",
        vertical_axis="Z",
        units="kN, m",
        nodes=[
            Node(id="1", x=0.0, y=0.0, z=0.0),
            Node(id="2", x=0.0, y=0.0, z=4.0, mass=node_mass, inertia=node_inertia),
        ],
        supports=[Support.fixed_3d("1")],
        materials=[ElasticMaterial(id="M", E=30.0e6, nu=0.2, unit_weight=24.0)],
        sections=[RectangularSection(id="C", material_id="M", b=0.4, h=0.4)],
        elements=[
            BeamElement(
                id="E1",
                start_node="1",
                end_node="2",
                section_id="C",
                stiffness_modifiers=stiffness_modifiers or {},
            )
        ],
        load_cases=[
            LoadCase(
                id="SDL",
                point_loads=[PointLoad(node_id="2", fz=-98.0665)],  # 10.0 tonnes
                element_loads=[DistributedLoad(element_id="E1", wz=-9.80665)],  # 1.0 tonne/m * 4 m = 4.0 tonnes
            )
        ],
        mass_source=mass_source,
    )
    return Project(model=model, units="kN, m")


def test_mass_source_preserves_nodal_inertia_and_counts_added_mass_once():
    ms = MassSource(
        id="MS1",
        include_element_mass=False,
        include_added_mass=True,
        factors=[MassSourceFactor(load_case_id="SDL", factor=1.0)],
    )
    proj = _cantilever_3d(node_mass=5000.0, node_inertia=1200.0, mass_source=ms)
    st = model_from_core(proj)

    n2 = st.nodes["2"]
    assert n2.mass == pytest.approx([15.0, 15.0, 15.0], rel=1e-12)
    assert n2.inertia == pytest.approx([1.2, 1.2, 1.2], rel=1e-12)

    st.number_dofs()
    st.apply_boundary_conditions()
    _, M_ff = st.get_reduced_matrices(lumped=True, include_rotational_inertia=False)
    # 15.0 nodal + 2.0 half distributed SDL on [ux, uy, uz], 1.2 on [rx, ry, rz]
    assert M_ff.diagonal() == pytest.approx([17.0, 17.0, 17.0, 1.2, 1.2, 1.2], rel=1e-12)


def test_mass_source_exclude_added_mass_excludes_nodal_mass_and_inertia():
    ms = MassSource(
        id="MS2",
        include_element_mass=False,
        include_added_mass=False,
        factors=[MassSourceFactor(load_case_id="SDL", factor=1.0)],
    )
    proj = _cantilever_3d(node_mass=5000.0, node_inertia=1200.0, mass_source=ms)
    st = model_from_core(proj)

    n2 = st.nodes["2"]
    assert n2.mass == pytest.approx([10.0, 10.0, 10.0], rel=1e-12)
    assert n2.inertia == pytest.approx([0.0, 0.0, 0.0], abs=1e-15)

    st.number_dofs()
    st.apply_boundary_conditions()
    _, M_ff = st.get_reduced_matrices(lumped=True, include_rotational_inertia=False)
    assert M_ff.diagonal() == pytest.approx([12.0, 12.0, 12.0, 0.0, 0.0, 0.0], rel=1e-12)


def test_include_element_mass_false_removes_frame_self_mass_even_with_modifiers():
    ms = MassSource(id="MS3", include_element_mass=False, include_added_mass=True, factors=[])
    proj = _cantilever_3d(
        node_mass=10000.0,
        node_inertia=0.0,
        mass_source=ms,
        stiffness_modifiers={"A": 0.6, "mass": 2.0},
    )
    st = model_from_core(proj)
    st.number_dofs()
    st.apply_boundary_conditions()
    _, M_ff = st.get_reduced_matrices(lumped=True, include_rotational_inertia=False)
    assert M_ff.diagonal() == pytest.approx([10.0, 10.0, 10.0, 0.0, 0.0, 0.0], rel=1e-12)

    E, I, L, m = 30.0e6, 0.4**4 / 12.0, 4.0, 10.0
    omega_exact = math.sqrt(3.0 * E * I / (m * L**3))
    from structdyn import MDF

    omega, _ = MDF.from_fem3d(st, lumped=True).modal.modal_analysis(n_modes=1)
    assert float(omega[0]) == pytest.approx(omega_exact, rel=1e-12)


def test_include_element_mass_false_consistent_for_frames_and_shells():
    # Single 1x1 m shell panel on two columns; include_element_mass=False removes both frame and shell self-mass
    model = StructuralModel(
        dimension="3D",
        vertical_axis="Z",
        units="kN, m",
        nodes=[
            Node(id="1", x=0.0, y=0.0, z=0.0),
            Node(id="2", x=1.0, y=0.0, z=0.0),
            Node(id="3", x=1.0, y=1.0, z=0.0),
            Node(id="4", x=0.0, y=1.0, z=0.0),
        ],
        supports=[Support.fixed_3d("1"), Support.fixed_3d("4")],
        materials=[ElasticMaterial(id="M", E=30.0e6, nu=0.2, unit_weight=24.0)],
        sections=[RectangularSection(id="B", material_id="M", b=0.3, h=0.4)],
        elements=[BeamElement(id="E1", start_node="1", end_node="2", section_id="B")],
        shell_sections=[ShellSection(id="S", material_id="M", thickness=0.2)],
        shell_elements=[ShellElement(id="A1", node_ids=["1", "2", "3", "4"], section_id="S")],
        load_cases=[
            LoadCase(id="SDL", shell_loads=[ShellLoad(element_ids=["A1"], pressure=-39.2266, direction="z")])
        ],  # 4.0 tonnes total -> 1.0 tonne per corner
        mass_source=MassSource(
            id="MS_NO_ELEM",
            include_element_mass=False,
            include_added_mass=True,
            factors=[MassSourceFactor(load_case_id="SDL", factor=1.0)],
        ),
    )
    st = model_from_core(Project(model=model, units="kN, m"))
    st.number_dofs()
    st.apply_boundary_conditions()
    _, M_ff = st.get_reduced_matrices(lumped=True, include_rotational_inertia=False)
    # Free nodes are 2 and 3 (6 DOFs each): 1.0 tonne on ux, uy, uz from shell SDL only, 0.0 on rotations
    assert M_ff.diagonal() == pytest.approx([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], rel=1e-12)
