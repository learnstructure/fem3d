"""
Unit tests for struct_core adapter with fem3d.
"""

import pytest
from fem3d import SimpleFrame, Results

try:
    import struct_core
    from fem3d.adapters.struct_core_adapter import model_from_core, model_to_core, result_to_core
    HAS_STRUCT_CORE = True
except ImportError:
    HAS_STRUCT_CORE = False


@pytest.mark.skipif(not HAS_STRUCT_CORE, reason="struct_core not installed")
def test_struct_core_adapter_roundtrip():
    frame = SimpleFrame()
    frame.add_node(1, 0.0, 0.0, 0.0)
    frame.add_node(2, 100.0, 0.0, 0.0)

    frame.add_frame(1, 1, 2, E=29000.0, A=10.0, I2=50.0, I3=100.0, J=30.0)
    frame.add_support(1, [1, 1, 1, 1, 1, 1])
    frame.add_node_load(2, [10.0, 5.0, -2.0, 0.0, 0.0, 0.0])

    disp, reac = frame.solve()

    # Convert to struct_core model
    sc_model = model_to_core(frame)
    assert len(sc_model.nodes) == 2
    assert len(sc_model.elements) == 1
    assert sc_model.sections[0].I2 == 50.0
    assert sc_model.sections[0].I3 == 100.0

    # Convert back from struct_core model
    rebuilt_struct = model_from_core(sc_model)
    assert len(rebuilt_struct.nodes) == 2
    assert len(rebuilt_struct.elements) == 1

    # Convert analysis results to struct_core.AnalysisResult
    results = Results(frame)
    sc_result = result_to_core(results)
    assert len(sc_result.node_results) == 2
    assert len(sc_result.element_results) == 1
    # Node 1 is the fixed support: all displacements are zero.
    # Node 2 is the free tip with applied loads [Fx=10, Fy=5, Fz=-2]:
    #   ux (axial) at node 2 should be nonzero.
    assert sc_result.node_results[2].displacement.ux != 0.0
    # Node 1 should have nonzero reactions (it's the fixed support).
    assert sc_result.node_results[1].reaction is not None
    assert sc_result.node_results[1].reaction.fx != 0.0
    assert sc_result.element_results[1].forces[0].P != 0.0


@pytest.mark.skipif(not HAS_STRUCT_CORE, reason="struct_core not installed")
def test_model_from_core_with_self_weight():
    from struct_core import (
        StructuralModel,
        Node,
        Support,
        ElasticMaterial,
        RectangularSection,
        BeamElement,
        LoadCase,
        LoadCombination,
        LoadCaseFactor,
    )

    nodes = [
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=10.0, y=0.0, z=0.0),
    ]
    mat = ElasticMaterial(id="Conc", E=3e7, rho=2400.0)
    sec = RectangularSection(id="B1", b=0.5, h=1.0)
    elements = [
        BeamElement(id=1, start_node=1, end_node=2, material_id="Conc", section_id="B1"),
    ]
    supports = [Support.fixed_3d(node_id=1)]
    lc_dead = LoadCase(id="DEAD", name="Dead Load", include_self_weight=True)
    comb_14d = LoadCombination(
        id="1.4D",
        factors=[LoadCaseFactor(load_case_id="DEAD", factor=1.4)],
    )

    model = StructuralModel(
        nodes=nodes,
        materials=[mat],
        sections=[sec],
        elements=elements,
        supports=supports,
        load_cases=[lc_dead],
        load_combinations=[comb_14d],
    )

    # 1. Automatic self-weight included
    struct = model_from_core(model, load_case_id="DEAD", include_self_weight=True)
    assert len(struct.loads) == 1
    struct.solve()
    reac_df = Results(struct).reactions()
    node1_fz = reac_df.loc[reac_df["node"] == 1, "Fz"].values[0]
    expected_w = (2400.0 * 0.5 * 9.80665 / 1000.0) * 10.0
    assert pytest.approx(node1_fz, rel=1e-3) == expected_w

    # 2. Self-weight disabled
    struct_no_sw = model_from_core(model, load_case_id="DEAD", include_self_weight=False)
    assert len(struct_no_sw.loads) == 0

    # 3. In load combination with factor 1.4
    struct_comb = model_from_core(model, load_combination_id="1.4D", include_self_weight=True)
    assert len(struct_comb.loads) == 1
    struct_comb.solve()
    comb_reac_df = Results(struct_comb).reactions()
    comb_node1_fz = comb_reac_df.loc[comb_reac_df["node"] == 1, "Fz"].values[0]
    assert pytest.approx(comb_node1_fz, rel=1e-3) == expected_w * 1.4


@pytest.mark.skipif(not HAS_STRUCT_CORE, reason="struct_core not installed")
def test_model_from_core_shear_deformation():
    from struct_core import (
        StructuralModel,
        Node,
        Support,
        ElasticMaterial,
        GeneralSection,
        BeamElement,
        PointLoad,
        LoadCase,
    )

    nodes = [
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=0.0, y=0.0, z=5.0),
    ]
    mat = ElasticMaterial(id="Steel", E=200e6, nu=0.3)
    sec = GeneralSection(id="Sec1", A=0.01, I3=1e-4, I2=1e-4, J=1e-4, As2=0.008, As3=0.008)
    elements = [
        BeamElement(id=1, start_node=1, end_node=2, material_id="Steel", section_id="Sec1"),
    ]
    supports = [Support.fixed_3d(node_id=1)]
    lc = LoadCase(id="LC1", point_loads=[PointLoad(node_id=2, fx=100.0)])
    model = StructuralModel(nodes=nodes, materials=[mat], sections=[sec], elements=elements, supports=supports, load_cases=[lc])

    # 1. With shear deformation explicitly disabled
    struct_eb = model_from_core(model, include_shear_deformation=False)
    assert struct_eb.elements[1].include_shear_deformation is False
    struct_eb.solve()
    disp_eb = Results(struct_eb).node_displacements()
    tip_dx_eb = disp_eb.loc[disp_eb["node"] == 2, "ux"].values[0]

    # 2. With shear deformation explicitly enabled
    struct_timo = model_from_core(model, include_shear_deformation=True)
    assert struct_timo.elements[1].include_shear_deformation is True
    assert struct_timo.elements[1].section.As2 == 0.008
    struct_timo.solve()
    disp_timo = Results(struct_timo).node_displacements()
    tip_dx_timo = disp_timo.loc[disp_timo["node"] == 2, "ux"].values[0]

    # Timoshenko beam includes additional shear flexibility: tip deflection must be strictly greater
    assert tip_dx_timo > tip_dx_eb


@pytest.mark.skipif(not HAS_STRUCT_CORE, reason="struct_core not installed")
def test_model_from_core_mass_source():
    from struct_core import (
        BeamElement,
        DistributedLoad,
        ElasticMaterial,
        GeneralSection,
        LoadCase,
        MassSource,
        MassSourceFactor,
        Node,
        PointLoad,
        StructuralModel,
        Support,
    )

    nodes = [
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=0.0, y=0.0, z=4.0),
    ]
    mat = ElasticMaterial(id="Steel", E=200e6, rho=7850.0)
    sec = GeneralSection(id="ColSec", A=0.02, I2=2e-4, I3=2e-4, J=4e-4)
    elements = [
        BeamElement(id=1, start_node=1, end_node=2, material_id="Steel", section_id="ColSec"),
    ]
    supports = [Support.fixed_3d(node_id=1)]

    # Dead load: 100 kN tip vertical load
    # Live load: 50 kN tip vertical load
    lc_dead = LoadCase(id="DEAD", point_loads=[PointLoad(node_id=2, fz=-100.0)])
    lc_live = LoadCase(id="LIVE", point_loads=[PointLoad(node_id=2, fz=-50.0)])

    ms = MassSource(
        id="MS1",
        include_element_mass=True,
        include_added_mass=True,
        lateral_only=False,
        factors=[
            MassSourceFactor(load_case_id="DEAD", factor=1.0),
            MassSourceFactor(load_case_id="LIVE", factor=0.25),
        ],
    )

    model = StructuralModel(
        nodes=nodes,
        materials=[mat],
        sections=[sec],
        elements=elements,
        supports=supports,
        load_cases=[lc_dead, lc_live],
        mass_source=ms,
    )

    struct = model_from_core(model, mass_source=ms)
    assert hasattr(struct, "mass_source_result")
    # Total tip load = 100 + 0.25 * 50 = 112.5 kN
    # Lumped mass at node 2 should be approx 112.5 / 9.80665 = 11.4717 tonne
    m_node2 = struct.nodes[2].mass[0]
    assert pytest.approx(m_node2, rel=1e-3) == 112.5 / 9.80665

    # Run modal analysis
    modal = struct.modal_analysis(num_modes=2)
    assert len(modal["frequencies"]) == 2
    assert modal["periods"][0] > 0.0


@pytest.mark.skipif(not HAS_STRUCT_CORE, reason="struct_core not installed")
def test_model_from_core_3d_consistent_mass_conversion():
    """Verify 3D materials with rho in kg/m^3 and nodes with mass in kg convert to consistent metric tonnes."""
    from struct_core import (
        StructuralModel,
        Node,
        Support,
        ElasticMaterial,
        RectangularSection,
        BeamElement,
    )

    mat = ElasticMaterial(id="Conc", E=2.5e7, rho=2400.0)
    sec = RectangularSection(id="ColSec", b=0.4, h=0.4)
    nodes = [
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=0.0, y=0.0, z=3.5, mass=5000.0),
    ]
    elements = [
        BeamElement(id=1, start_node=1, end_node=2, material_id="Conc", section_id="ColSec"),
    ]
    supports = [Support(node_id=1, ux=True, uy=True, uz=True, rx=True, ry=True, rz=True)]
    model = StructuralModel(nodes=nodes, materials=[mat], sections=[sec], elements=elements, supports=supports)

    struct = model_from_core(model)
    # Density should be converted from 2400.0 kg/m^3 to 2.4 tonne/m^3
    assert pytest.approx(struct.elements[1].material.rho, rel=1e-5) == 2.4
    # Node mass should be converted from 5000.0 kg to 5.0 tonnes
    assert pytest.approx(struct.nodes[2].mass[0], rel=1e-5) == 5.0

    modal = struct.modal_analysis(num_modes=1)
    T1 = modal["periods"][0]
    assert 0.05 < T1 < 0.3, f"Expected realistic period ~0.1s, got {T1}s"


@pytest.mark.skipif(not HAS_STRUCT_CORE, reason="struct_core not installed")
def test_model_from_core_load_cases_filtering():
    """Verify default sum of all load cases and explicit load case selection."""
    from struct_core import (
        StructuralModel,
        Node,
        Support,
        ElasticMaterial,
        RectangularSection,
        BeamElement,
        PointLoad,
        LoadCase,
    )

    nodes = [
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=0.0, y=0.0, z=3.0),
    ]
    mat = ElasticMaterial(id="M1", E=3e7, nu=0.2)
    sec = RectangularSection(id="S1", b=0.3, h=0.3)
    elements = [
        BeamElement(id=1, start_node=1, end_node=2, material_id="M1", section_id="S1"),
    ]
    supports = [Support(node_id=1, ux=True, uy=True, uz=True, rx=True, ry=True, rz=True)]

    lc_dead = LoadCase(id="DEAD", name="Dead Load", point_loads=[PointLoad(node_id=2, fz=-100.0)])
    lc_live = LoadCase(id="LIVE", name="Live Load", point_loads=[PointLoad(node_id=2, fz=-50.0)])

    model = StructuralModel(
        nodes=nodes,
        materials=[mat],
        sections=[sec],
        elements=elements,
        supports=supports,
        load_cases=[lc_dead, lc_live],
    )

    # 1. Default: all cases transferred (summed: -100 + -50 = -150)
    struct_all = model_from_core(model)
    struct_all.solve()
    assert pytest.approx(struct_all.reactions[struct_all.nodes[1].dofs[2]], rel=1e-5) == 150.0

    # 2. Explicit 'DEAD' only
    struct_dead = model_from_core(model, load_case_id="DEAD")
    struct_dead.solve()
    assert pytest.approx(struct_dead.reactions[struct_dead.nodes[1].dofs[2]], rel=1e-5) == 100.0

    # 3. Explicit 'LIVE' only
    struct_live = model_from_core(model, load_case_id="LIVE")
    struct_live.solve()
    assert pytest.approx(struct_live.reactions[struct_live.nodes[1].dofs[2]], rel=1e-5) == 50.0

    # 4. Explicit list of cases ['DEAD', 'LIVE']
    struct_list = model_from_core(model, load_case_id=["DEAD", "LIVE"])
    struct_list.solve()
    assert pytest.approx(struct_list.reactions[struct_list.nodes[1].dofs[2]], rel=1e-5) == 150.0

    # 5. Invalid case raises ValueError
    with pytest.raises(ValueError):
        model_from_core(model, load_case_id="WIND")


@pytest.mark.skipif(not HAS_STRUCT_CORE, reason="struct_core not installed")
def test_ecosystem_roundtrip_units_and_unit_weight():
    """Verify units and unit_weight survive model_to_core -> apply_self_weight -> model_from_core."""
    from fem3d import Structure, Node, ElasticMaterial, Section, FrameElement, DistributedLoad
    from fem3d.adapters import model_to_core, model_from_core
    from struct_load import apply_self_weight

    struct = Structure(units="kN, m")
    n1 = Node(1, 0.0, 0.0, 0.0)
    n1.set_support([1, 1, 1, 1, 1, 1])
    n2 = Node(2, 5.0, 0.0, 0.0)
    struct.add_node(n1)
    struct.add_node(n2)

    mat = ElasticMaterial(E=30e6, nu=0.2, unit_weight=24.0)
    sec = Section.from_rectangle(width=0.3, depth=0.5)  # A = 0.15 m^2
    elem = FrameElement(1, n1, n2, material=mat, section=sec)
    struct.add_element(elem)

    # Superimposed live load
    struct.add_load(DistributedLoad(elem, wz=-10.0, coord_system="global"))

    # Convert to struct_core
    core_model = model_to_core(struct)
    assert core_model.units.force == "kN"
    assert core_model.units.length == "m"
    assert core_model.materials[0].unit_weight == 24.0

    # Apply self weight: w = 24.0 * 0.15 = 3.6 kN/m
    core_model = apply_self_weight(core_model)
    dead_case = next(lc for lc in core_model.load_cases if lc.id == "DEAD")
    assert len(dead_case.element_loads) == 1
    assert pytest.approx(dead_case.element_loads[0].wz, rel=1e-5) == -3.6

    # Convert back to fem3d Structure
    struct_rebuilt = model_from_core(core_model)
    assert struct_rebuilt.units.force == "kN"
    assert struct_rebuilt.elements[1].material.unit_weight == 24.0
    # Both DEAD and LIVE transferred: total w = -10.0 + -3.6 = -13.6 kN/m, total load = 5.0 * 13.6 = 68.0 kN
    disp, reac = struct_rebuilt.solve()
    assert pytest.approx(reac[struct_rebuilt.nodes[1].dofs[2]], rel=1e-5) == 68.0




