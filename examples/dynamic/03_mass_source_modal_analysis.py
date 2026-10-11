"""
Example 03: 3D Modal Analysis with Commercial Mass Source Definition
=====================================================================

Demonstrates:
1. Building a 3D steel space frame via `struct_core`.
2. Defining Dead and Live load cases.
3. Converting the model to `fem3d.Structure` using two mass configurations:
   - Configuration A: Self-Mass only
   - Configuration B: Commercial Mass Source (1.0 DEAD + 0.25 LIVE + Self-Mass)
4. Solving 3D eigenvalue modal analysis via `structure.modal_analysis()` (delegating to structdyn).
5. Comparing the natural periods and frequencies to evaluate the period elongation.
"""

import numpy as np
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
from fem3d.adapters.struct_core_adapter import model_from_core


def create_3d_frame_model():
    """Construct a 1-story 3D space frame model in SI units."""
    # Nodes: 4 at ground (Z=0), 4 at roof (Z=3.5 m)
    nodes = [
        # Ground nodes
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=6.0, y=0.0, z=0.0),
        Node(id=3, x=6.0, y=5.0, z=0.0),
        Node(id=4, x=0.0, y=5.0, z=0.0),
        # Roof nodes
        Node(id=5, x=0.0, y=0.0, z=3.5),
        Node(id=6, x=6.0, y=0.0, z=3.5),
        Node(id=7, x=6.0, y=5.0, z=3.5),
        Node(id=8, x=0.0, y=5.0, z=3.5),
    ]

    mat_steel = ElasticMaterial(id="STEEL", E=200e6, rho=7850.0)  # 200 GPa, 7850 kg/m^3
    sec_col = GeneralSection(id="COL", A=0.015, I2=1.5e-4, I3=1.5e-4, J=3e-4)
    sec_beam = GeneralSection(id="BEAM", A=0.012, I2=2.0e-4, I3=1.0e-4, J=2e-4)

    elements = [
        # 4 Columns
        BeamElement(id="C1", start_node=1, end_node=5, material_id="STEEL", section_id="COL"),
        BeamElement(id="C2", start_node=2, end_node=6, material_id="STEEL", section_id="COL"),
        BeamElement(id="C3", start_node=3, end_node=7, material_id="STEEL", section_id="COL"),
        BeamElement(id="C4", start_node=4, end_node=8, material_id="STEEL", section_id="COL"),
        # 4 Roof perimeter beams
        BeamElement(id="B1", start_node=5, end_node=6, material_id="STEEL", section_id="BEAM"),
        BeamElement(id="B2", start_node=6, end_node=7, material_id="STEEL", section_id="BEAM"),
        BeamElement(id="B3", start_node=7, end_node=8, material_id="STEEL", section_id="BEAM"),
        BeamElement(id="B4", start_node=8, end_node=5, material_id="STEEL", section_id="BEAM"),
    ]

    supports = [
        Support.fixed_3d(node_id=1),
        Support.fixed_3d(node_id=2),
        Support.fixed_3d(node_id=3),
        Support.fixed_3d(node_id=4),
    ]

    # Dead Load (Finishes + MEP = 18 kN/m on all beams)
    beam_ids = ["B1", "B2", "B3", "B4"]
    dead_loads = [DistributedLoad(element_id=bid, wz=-18.0) for bid in beam_ids]
    lc_dead = LoadCase(id="DEAD", name="Superimposed Dead Load", element_loads=dead_loads)

    # Live Load (Office live load = 10 kN/m on all beams)
    live_loads = [DistributedLoad(element_id=bid, wz=-10.0) for bid in beam_ids]
    lc_live = LoadCase(id="LIVE", name="Live Load", element_loads=live_loads)

    model = StructuralModel(
        nodes=nodes,
        elements=elements,
        materials=[mat_steel],
        sections=[sec_col, sec_beam],
        supports=supports,
        load_cases=[lc_dead, lc_live],
        story_elevations=[3.5],
    )
    return model


def main():
    print("=" * 72)
    print("Example 03: 3D Modal Analysis with Mass Source (fem3d + structdyn)")
    print("=" * 72)

    model = create_3d_frame_model()

    # Configuration A: Self-Mass only
    ms_self_only = MassSource(
        id="MS_SELF",
        name="Self-Mass Only",
        include_element_mass=True,
        include_added_mass=False,
        factors=[],
    )

    # Configuration B: ASCE 7-22 Mass Source: 1.0 DEAD + 0.25 LIVE + Self-Mass
    ms_asce7 = MassSource(
        id="MS_ASCE7",
        name="ASCE 7-22 Mass Source (1.0 D + 0.25 L + Self)",
        include_element_mass=True,
        include_added_mass=True,
        lateral_only=False,
        factors=[
            MassSourceFactor(load_case_id="DEAD", factor=1.0),
            MassSourceFactor(load_case_id="LIVE", factor=0.25),
        ],
    )

    # Build and solve Configuration A
    struct_a = model_from_core(model, mass_source=ms_self_only)
    modal_a = struct_a.modal_analysis(num_modes=4)

    # Build and solve Configuration B
    struct_b = model_from_core(model, mass_source=ms_asce7)
    modal_b = struct_b.modal_analysis(num_modes=4)

    print("\n[Modal Analysis Comparison]")
    print(f"{'Mode':<6} {'Self-Mass Only Period (s)':<26} {'With Mass Source Period (s)':<28} {'Period Ratio (B/A)':<20}")
    print("-" * 78)
    for m in range(4):
        t_a = modal_a["periods"][m]
        t_b = modal_b["periods"][m]
        ratio = t_b / t_a if t_a > 0 else 0.0
        print(f"{m+1:<6} {t_a:<26.4f} {t_b:<28.4f} {ratio:<20.2f}")

    print("\n[Mass Source B Summary]")
    ms_res = getattr(struct_b, "mass_source_result", None)
    if ms_res:
        print(f"  Total Effective Seismic Weight W: {ms_res.total_seismic_weight:10.2f} kN")
        print(f"  Total Inertial Mass M           : {ms_res.total_mass:10.2f} tonnes")
        print(f"  Center of Mass (CM)             : ({ms_res.center_of_mass[0]:.2f} m, "
              f"{ms_res.center_of_mass[1]:.2f} m, {ms_res.center_of_mass[2]:.2f} m)")

    print("\n" + "=" * 72)
    print("3D Mass Source modal analysis completed successfully.")
    print("=" * 72)


if __name__ == "__main__":
    main()
