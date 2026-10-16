"""
Test suite validating 3D RC space frame linear static analysis (gravity + lateral)
against benchmark results obtained from SAP2000.

Model:
- 2-bay (4 m, 6 m) along X, 1-bay (5 m) along Y, 2-story (3.5 m each) space frame.
- Columns: 0.3 m (along X) x 0.4 m (along Y).
- Beams: 0.25 m width x 0.4 m depth.
- Material: Concrete C30 (E = 30 GPa, nu = 0.2, G = 12.5 GPa).
- All base nodes fixed.
- Beam gravity load: 10 kN/m downward on all beams.
- Lateral loads along +X applied at x = 0 facade:
  - Middle story (z = 3.5 m): 40 kN each at (0, 0, 3.5) and (0, 5, 3.5) -> 80 kN
  - Top story    (z = 7.0 m): 20 kN each at (0, 0, 7.0) and (0, 5, 7.0) -> 40 kN
  - Total lateral load = 120 kN.

Validated with SAP2000 under Timoshenko frame formulation (shear deformation included).
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


def build_frame_gravity_lateral(include_shear_deformation: bool = True) -> Structure:
    """Build and return the 3D space frame structure."""
    E = 30.0e6  # kPa (30 GPa)
    nu = 0.20
    G = E / (2.0 * (1.0 + nu))  # 12.5 GPa
    material = ElasticMaterial(E=E, G=G, rho=0.0)

    # Column: 0.3 m (along X) x 0.4 m (along Y)
    # Depth (local 2, along +Y) = 0.40 m, Width (local 3, along -X) = 0.30 m
    col_section = Section.from_rectangle(width=0.30, depth=0.40)

    # Beam: 0.25 m width x 0.40 m depth
    # Depth (local 2, along +Z) = 0.40 m, Width (local 3, horizontal) = 0.25 m
    beam_section = Section.from_rectangle(width=0.25, depth=0.40)

    struct = Structure()
    x_coords = [0.0, 4.0, 10.0]
    y_coords = [0.0, 5.0]
    z_coords = [0.0, 3.5, 7.0]

    # Nodes:
    # Ground  (z=0.0): Nodes 1 to 6 (fixed)
    # Story 1 (z=3.5): Nodes 7 to 12
    # Story 2 (z=7.0): Nodes 13 to 18
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

    # Columns (12 elements)
    elem_id = 1
    for story in range(2):
        z_bot = z_coords[story]
        z_top = z_coords[story + 1]
        for y in y_coords:
            for x in x_coords:
                n_bot = node_grid[(x, y, z_bot)]
                n_top = node_grid[(x, y, z_top)]
                col_elem = FrameElement(
                    eid=elem_id,
                    node_i=n_bot,
                    node_j=n_top,
                    material=material,
                    section=col_section,
                    include_shear_deformation=include_shear_deformation,
                )
                struct.add_element(col_elem)
                elem_id += 1

    # Beams (14 elements: 8 along X, 6 along Y)
    beams = []
    # X-direction beams
    for story in range(1, 3):
        z = z_coords[story]
        for y in y_coords:
            for bay in range(2):
                n1 = node_grid[(x_coords[bay], y, z)]
                n2 = node_grid[(x_coords[bay + 1], y, z)]
                bm_elem = FrameElement(
                    eid=elem_id,
                    node_i=n1,
                    node_j=n2,
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
            n1 = node_grid[(x, y_coords[0], z)]
            n2 = node_grid[(x, y_coords[1], z)]
            bm_elem = FrameElement(
                eid=elem_id,
                node_i=n1,
                node_j=n2,
                material=material,
                section=beam_section,
                include_shear_deformation=include_shear_deformation,
            )
            struct.add_element(bm_elem)
            beams.append(bm_elem)
            elem_id += 1

    # Gravity load: 10 kN/m downwards on all 14 beams
    for bm in beams:
        struct.add_load(DistributedLoad(bm, wz=-10.0))

    # Lateral loads: along +X on the x = 0.0 face
    # Story 1 (z = 3.5 m): 40 kN each at (0, 0, 3.5) and (0, 5, 3.5)
    for y in y_coords:
        node_mid = node_grid[(0.0, y, 3.5)]
        struct.add_load(PointLoad(node_mid, fx=40.0))

    # Story 2 (z = 7.0 m): 20 kN each at (0, 0, 7.0) and (0, 5, 7.0)
    for y in y_coords:
        node_top = node_grid[(0.0, y, 7.0)]
        struct.add_load(PointLoad(node_top, fx=20.0))

    return struct


def test_frame_gravity_lateral_equilibrium():
    """Verify global static equilibrium of base reactions."""
    struct = build_frame_gravity_lateral(include_shear_deformation=True)
    struct.solve()

    results = Results(struct)
    df_reactions = results.reactions()
    r_sum = df_reactions[["Fx", "Fy", "Fz"]].sum()

    # Total applied: Fx = +120 kN, Fy = 0 kN, Fz = -700 kN
    assert r_sum["Fx"] == pytest.approx(-120.0, rel=1e-5)
    assert r_sum["Fy"] == pytest.approx(0.0, abs=1e-5)
    assert r_sum["Fz"] == pytest.approx(700.0, rel=1e-5)


def test_frame_gravity_lateral_sap2000_reactions():
    """
    Validate base reactions against SAP2000 benchmark results
    (with shear deformation).
    """
    struct = build_frame_gravity_lateral(include_shear_deformation=True)
    struct.solve()

    results = Results(struct)
    df_reactions = results.reactions().set_index("node")

    # Node 1 at (0, 0, 0)
    assert df_reactions.loc[1, "Fx"] == pytest.approx(-18.0591, abs=0.01)
    assert df_reactions.loc[1, "Fy"] == pytest.approx(2.6879, abs=0.01)
    assert df_reactions.loc[1, "Fz"] == pytest.approx(61.7994, abs=0.01)
    assert df_reactions.loc[1, "Mx"] == pytest.approx(-3.1141, abs=0.01)
    assert df_reactions.loc[1, "My"] == pytest.approx(-37.5547, abs=0.01)

    # Node 2 at (4, 0, 0)
    assert df_reactions.loc[2, "Fx"] == pytest.approx(-20.7444, abs=0.01)
    assert df_reactions.loc[2, "Fy"] == pytest.approx(2.6879, abs=0.01)
    assert df_reactions.loc[2, "Fz"] == pytest.approx(170.1736, abs=0.01)
    assert df_reactions.loc[2, "Mx"] == pytest.approx(-3.1141, abs=0.01)
    assert df_reactions.loc[2, "My"] == pytest.approx(-40.5147, abs=0.01)

    # Node 3 at (10, 0, 0)
    assert df_reactions.loc[3, "Fx"] == pytest.approx(-21.1965, abs=0.01)
    assert df_reactions.loc[3, "Fy"] == pytest.approx(2.6879, abs=0.01)
    assert df_reactions.loc[3, "Fz"] == pytest.approx(118.0269, abs=0.01)
    assert df_reactions.loc[3, "Mx"] == pytest.approx(-3.1141, abs=0.01)
    assert df_reactions.loc[3, "My"] == pytest.approx(-40.9667, abs=0.01)


def test_frame_gravity_lateral_sap2000_displacements():
    """
    Validate story displacements against SAP2000 benchmark results
    (with shear deformation).
    """
    struct = build_frame_gravity_lateral(include_shear_deformation=True)
    struct.solve()

    results = Results(struct)
    df_disp = results.node_displacements().set_index("node")

    # Roof (Story 2, Z = 7.0 m):
    # Node 13 at (0, 0, 7.0): ux = 6.0456 mm, uy = 0.0070 mm, uz = -0.0943 mm
    assert df_disp.loc[13, "ux"] * 1000.0 == pytest.approx(6.0456, abs=0.005)
    assert df_disp.loc[13, "uy"] * 1000.0 == pytest.approx(0.0070, abs=0.005)
    assert df_disp.loc[13, "uz"] * 1000.0 == pytest.approx(-0.0943, abs=0.005)

    # Node 15 at (10, 0, 7.0): ux = 5.9948 mm, uy = 0.0070 mm, uz = -0.1692 mm
    assert df_disp.loc[15, "ux"] * 1000.0 == pytest.approx(5.9948, abs=0.005)
    assert df_disp.loc[15, "uy"] * 1000.0 == pytest.approx(0.0070, abs=0.005)
    assert df_disp.loc[15, "uz"] * 1000.0 == pytest.approx(-0.1692, abs=0.005)

    # Middle floor (Story 1, Z = 3.5 m):
    # Node 7 at (0, 0, 3.5): ux = 3.7904 mm, uy = -0.0047 mm, uz = -0.0601 mm
    assert df_disp.loc[7, "ux"] * 1000.0 == pytest.approx(3.7904, abs=0.005)
    assert df_disp.loc[7, "uy"] * 1000.0 == pytest.approx(-0.0047, abs=0.005)
    assert df_disp.loc[7, "uz"] * 1000.0 == pytest.approx(-0.0601, abs=0.005)

    # Node 9 at (10, 0, 3.5): ux = 3.7429 mm, uy = -0.0047 mm, uz = -0.1147 mm
    assert df_disp.loc[9, "ux"] * 1000.0 == pytest.approx(3.7429, abs=0.005)
    assert df_disp.loc[9, "uy"] * 1000.0 == pytest.approx(-0.0047, abs=0.005)
    assert df_disp.loc[9, "uz"] * 1000.0 == pytest.approx(-0.1147, abs=0.005)
