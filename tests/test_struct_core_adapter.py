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


