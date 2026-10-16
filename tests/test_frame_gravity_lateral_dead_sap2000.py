"""
Test suite validating 3D RC space frame linear static analysis under combined
gravity, lateral, and automated element self-weight (dead load) against SAP2000 benchmark results.

Model Description:
- 2 bays along X (4.0 m, 6.0 m), 1 bay along Y (5.0 m), 2 stories (3.5 m each).
- Columns: 0.30 m (along X) x 0.40 m (along Y).
- Beams: 0.25 m width x 0.40 m depth.
- Material: Concrete C30 (E = 30 GPa, nu = 0.20, unit_weight = 24.0 kN/m^3).
- Supports: All 6 base nodes at ground level (Z = 0) are fully fixed [1, 1, 1, 1, 1, 1].
- Loads:
  * Superimposed gravity: 10 kN/m downward on all 14 beams (total = 700 kN).
  * Lateral loads (+X on x = 0 face): 40 kN each at story 1, 20 kN each at story 2 (total = 120 kN).
  * Self-weight (dead load): 24 kN/m^3 * section area on all columns & beams (total = 288.96 kN).
  * Total vertical load = 700.0 + 288.96 = 988.96 kN.

Validated against SAP2000 with Timoshenko beam formulation (shear deformation included).
"""

import pytest
import numpy as np

from fem3d import (
    Structure,
    Node,
    ElasticMaterial,
    Section,
    FrameElement,
    PointLoad,
    DistributedLoad,
    Results,
)
from fem3d.adapters import model_to_core, model_from_core
from struct_load import apply_self_weight


def build_frame_model(include_shear_deformation: bool = True) -> Structure:
    """Build the 3D space frame structure before self-weight is applied."""
    material = ElasticMaterial(E=30.0e6, nu=0.20, unit_weight=24.0)

    # Column: 0.3 m (along X) x 0.4 m (along Y)
    col_section = Section.from_rectangle(width=0.30, depth=0.40)

    # Beam: 0.25 m width x 0.40 m depth
    beam_section = Section.from_rectangle(width=0.25, depth=0.40)

    struct = Structure(units="kN, m")
    x_coords = [0.0, 4.0, 10.0]
    y_coords = [0.0, 5.0]
    z_coords = [0.0, 3.5, 7.0]

    node_grid = {}
    nid = 1
    for z in z_coords:
        for y in y_coords:
            for x in x_coords:
                node = Node(nid, x, y, z)
                if z == 0.0:
                    node.set_support([1, 1, 1, 1, 1, 1])
                struct.add_node(node)
                node_grid[(x, y, z)] = node
                nid += 1

    # 12 columns
    elem_id = 1
    for story in range(2):
        z_bot = z_coords[story]
        z_top = z_coords[story + 1]
        for y in y_coords:
            for x in x_coords:
                col_elem = FrameElement(
                    eid=elem_id,
                    node_i=node_grid[(x, y, z_bot)],
                    node_j=node_grid[(x, y, z_top)],
                    material=material,
                    section=col_section,
                    include_shear_deformation=include_shear_deformation,
                )
                struct.add_element(col_elem)
                elem_id += 1

    # 14 beams
    beams = []
    # X-direction beams
    for story in range(1, 3):
        z = z_coords[story]
        for y in y_coords:
            for bay in range(2):
                bm_elem = FrameElement(
                    eid=elem_id,
                    node_i=node_grid[(x_coords[bay], y, z)],
                    node_j=node_grid[(x_coords[bay + 1], y, z)],
                    material=material,
                    section=beam_section,
                    include_shear_deformation=include_shear_deformation,
                )
                struct.add_element(bm_elem)
                beams.append(bm_elem)
                elem_id += 1

    # Y-direction beams
    for story in range(1, 3):
        z = z_coords[story]
        for x in x_coords:
            bm_elem = FrameElement(
                eid=elem_id,
                node_i=node_grid[(x, y_coords[0], z)],
                node_j=node_grid[(x, y_coords[1], z)],
                material=material,
                section=beam_section,
                include_shear_deformation=include_shear_deformation,
            )
            struct.add_element(bm_elem)
            beams.append(bm_elem)
            elem_id += 1

    # Superimposed gravity load: 10 kN/m downward on all beams
    for bm in beams:
        struct.add_load(DistributedLoad(bm, wz=-10.0, coord_system="global"))

    # Lateral loads along +X applied at x = 0.0 facade
    for y in y_coords:
        struct.add_load(PointLoad(node_grid[(0.0, y, 3.5)], fx=40.0))
        struct.add_load(PointLoad(node_grid[(0.0, y, 7.0)], fx=20.0))

    return struct


def build_analyzed_model_roundtrip() -> Structure:
    """Build analyzed model via struct_core round-trip (matching frame_gravity_lateral_dead.py)."""
    struct = build_frame_model(include_shear_deformation=True)
    core_model = model_to_core(struct)
    core_model = apply_self_weight(core_model)
    struct_analyzed = model_from_core(core_model)
    struct_analyzed.solve()
    return struct_analyzed


def test_frame_gravity_lateral_dead_equilibrium():
    """Validate global equilibrium (Fx = -120 kN, Fy = 0, Fz = +988.96 kN)."""
    struct = build_analyzed_model_roundtrip()
    results = Results(struct)
    df_rx = results.reactions()
    r_sum = df_rx[["Fx", "Fy", "Fz"]].sum()

    assert r_sum["Fx"] == pytest.approx(-120.0, rel=1e-5)
    assert r_sum["Fy"] == pytest.approx(0.0, abs=1e-5)
    assert r_sum["Fz"] == pytest.approx(988.96, rel=1e-5)


def test_frame_gravity_lateral_dead_sap2000_reactions():
    """Validate base reactions against SAP2000 benchmark results."""
    struct = build_analyzed_model_roundtrip()
    results = Results(struct)
    df_rx = results.reactions().set_index("node")

    # Node 1 at (0, 0, 0)
    assert df_rx.loc[1, "Fx"] == pytest.approx(-17.6877, abs=0.01)
    assert df_rx.loc[1, "Fy"] == pytest.approx(3.3330, abs=0.01)
    assert df_rx.loc[1, "Fz"] == pytest.approx(102.2278, abs=0.01)
    assert df_rx.loc[1, "Mx"] == pytest.approx(-3.8614, abs=0.01)
    assert df_rx.loc[1, "My"] == pytest.approx(-37.0302, abs=0.01)

    # Node 2 at (4, 0, 0)
    assert df_rx.loc[2, "Fx"] == pytest.approx(-20.2623, abs=0.01)
    assert df_rx.loc[2, "Fy"] == pytest.approx(3.3330, abs=0.01)
    assert df_rx.loc[2, "Fz"] == pytest.approx(228.5083, abs=0.01)
    assert df_rx.loc[2, "Mx"] == pytest.approx(-3.8614, abs=0.01)
    assert df_rx.loc[2, "My"] == pytest.approx(-39.8657, abs=0.01)

    # Node 3 at (10, 0, 0)
    assert df_rx.loc[3, "Fx"] == pytest.approx(-22.0500, abs=0.01)
    assert df_rx.loc[3, "Fy"] == pytest.approx(3.3330, abs=0.01)
    assert df_rx.loc[3, "Fz"] == pytest.approx(163.7439, abs=0.01)
    assert df_rx.loc[3, "Mx"] == pytest.approx(-3.8614, abs=0.01)
    assert df_rx.loc[3, "My"] == pytest.approx(-41.8720, abs=0.01)


def test_frame_gravity_lateral_dead_sap2000_displacements():
    """Validate story displacements against SAP2000 benchmark results."""
    struct = build_analyzed_model_roundtrip()
    results = Results(struct)
    df_disp = results.node_displacements().set_index("node")

    # Roof (Story 2, Z = 7.0 m):
    # Node 13 at (0, 0, 7.0): ux = 5.9713 mm, uz = -0.1433 mm
    assert df_disp.loc[13, "ux"] * 1000.0 == pytest.approx(5.9713, abs=0.005)
    assert df_disp.loc[13, "uy"] * 1000.0 == pytest.approx(0.0087, abs=0.005)
    assert df_disp.loc[13, "uz"] * 1000.0 == pytest.approx(-0.1433, abs=0.005)

    # Node 14 at (4, 0, 7.0): ux = 5.9458 mm, uz = -0.3224 mm
    assert df_disp.loc[14, "ux"] * 1000.0 == pytest.approx(5.9458, abs=0.005)
    assert df_disp.loc[14, "uz"] * 1000.0 == pytest.approx(-0.3224, abs=0.005)

    # Node 15 at (10, 0, 7.0): ux = 5.9148 mm, uz = -0.2260 mm
    assert df_disp.loc[15, "ux"] * 1000.0 == pytest.approx(5.9148, abs=0.005)
    assert df_disp.loc[15, "uz"] * 1000.0 == pytest.approx(-0.2260, abs=0.005)

    # Story 1 (Middle, Z = 3.5 m):
    # Node 7 at (0, 0, 3.5): ux = 3.7687 mm, uz = -0.0945 mm
    assert df_disp.loc[7, "ux"] * 1000.0 == pytest.approx(3.7687, abs=0.005)
    assert df_disp.loc[7, "uy"] * 1000.0 == pytest.approx(-0.0059, abs=0.005)
    assert df_disp.loc[7, "uz"] * 1000.0 == pytest.approx(-0.0945, abs=0.005)

    # Node 8 at (4, 0, 3.5): ux = 3.7377 mm, uz = -0.2173 mm
    assert df_disp.loc[8, "ux"] * 1000.0 == pytest.approx(3.7377, abs=0.005)
    assert df_disp.loc[8, "uz"] * 1000.0 == pytest.approx(-0.2173, abs=0.005)

    # Node 9 at (10, 0, 3.5): ux = 3.7247 mm, uz = -0.1543 mm
    assert df_disp.loc[9, "ux"] * 1000.0 == pytest.approx(3.7247, abs=0.005)
    assert df_disp.loc[9, "uz"] * 1000.0 == pytest.approx(-0.1543, abs=0.005)


def test_direct_self_weight_matches_roundtrip_and_sap2000():
    """
    Verify that direct Structure.apply_self_weight() produces results identical
    to the struct_core roundtrip model and matches SAP2000 benchmarks.
    """
    s_direct = build_frame_model(include_shear_deformation=True)
    s_direct.apply_self_weight()
    disp_direct, rx_direct = s_direct.solve()

    s_rt = build_analyzed_model_roundtrip()
    disp_rt, rx_rt = s_rt.solve()

    # Exact equality across all DOFs
    max_disp_diff = np.max(np.abs(disp_direct - disp_rt))
    assert max_disp_diff < 1e-10, f"Max displacement diff between direct and round-trip: {max_disp_diff}"

    # Verify key nodes on direct model
    n13 = s_direct.nodes[13]
    n7 = s_direct.nodes[7]
    assert s_direct.disp[n13.dofs[0]] * 1000.0 == pytest.approx(5.9713, abs=0.005)
    assert s_direct.disp[n7.dofs[0]] * 1000.0 == pytest.approx(3.7687, abs=0.005)
