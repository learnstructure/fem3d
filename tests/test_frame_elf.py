"""
Regression test suite for 3D RC Space Frame Equivalent Lateral Force (ELF) procedure.

Validates:
1. Reusable model generation via `rc_frame_model.create_rc_frame` and `create_rc_frame_core`.
2. Mass source multipliers evaluation and story seismic weight aggregation.
3. ASCE 7-16 / 7-22 Equivalent Lateral Force (ELF) procedure (Ta, Cs, V, story forces Fx).
4. Finite element static analysis and global base equilibrium under lateral seismic loads.
"""

import pytest
import numpy as np

from fem3d.examples.linear.rc_frame_model import (
    create_rc_frame,
    create_rc_frame_core,
    STORY_ELEVATIONS,
)
from fem3d.adapters import model_to_core, model_from_core
from fem3d import Results
from struct_core import (
    DistributedLoad as ScDistributedLoad,
    LoadCase,
    LoadCaseFactor,
    LoadCombination,
    MassSource,
    MassSourceFactor,
    PointLoad as ScPointLoad,
)
from struct_load import LoadGenerator, apply_self_weight, calculate_mass_source
from struct_load.codes.asce7_seismic import (
    SeismicParameters,
    calculate_seismic_elf,
)


def test_rc_frame_model_creation():
    """Verify reusable model geometry, nodes, elements, and attached properties."""
    struct = create_rc_frame(include_shear_deformation=True)
    assert len(struct.nodes) == 18
    assert len(struct.elements) == 26
    assert len(struct.columns) == 12
    assert len(struct.beams) == 14
    assert struct.story_elevations == [3.5, 7.0]

    # Check fixed base supports at z=0 (nodes 1..6)
    for nid in range(1, 7):
        assert struct.nodes[nid].support == [1, 1, 1, 1, 1, 1]

    # Verify struct_core model creation
    core_model = create_rc_frame_core(include_shear_deformation=True)
    assert len(core_model.nodes) == 18
    assert len(core_model.elements) == 26
    assert len(core_model.supports) == 6
    assert core_model.story_elevations == [3.5, 7.0]


def test_mass_source_and_elf_calculation():
    """Verify mass source evaluation and ASCE 7 ELF lateral forces."""
    struct = create_rc_frame(include_shear_deformation=True)
    beams = struct.beams
    core_model = model_to_core(struct)

    # Apply 10 kN/m superimposed dead and 4 kN/m live load
    core_sdl = [ScDistributedLoad(element_id=bm.id, wz=-10.0, is_local=False) for bm in beams]
    core_live = [ScDistributedLoad(element_id=bm.id, wz=-4.0, is_local=False) for bm in beams]

    core_model.load_cases = [
        LoadCase(id="DEAD", name="Dead Load", element_loads=core_sdl),
        LoadCase(id="LIVE", name="Live Load", element_loads=core_live),
    ]

    # Materialize self-weight into DEAD
    core_model = apply_self_weight(core_model, target_case_id="DEAD")

    # Define mass source: 1.0 DEAD + 0.25 LIVE
    ms = MassSource(
        id="MS_TEST",
        include_element_mass=False,  # Already in DEAD
        factors=[
            MassSourceFactor(load_case_id="DEAD", factor=1.0),
            MassSourceFactor(load_case_id="LIVE", factor=0.25),
        ],
    )
    core_model.mass_source = ms

    res = calculate_mass_source(core_model, mass_source=ms)
    # Expected: Self-weight (288.96) + SDL (700.0) + 0.25 * LIVE (280 * 0.25 = 70.0) = 1058.96 kN
    assert pytest.approx(res.total_seismic_weight, rel=1e-3) == 1058.96
    assert pytest.approx(sum(res.story_weights.values()), rel=1e-3) == 1058.96

    # Test ASCE 7 ELF calculation
    params = SeismicParameters(
        sds=1.00,
        sd1=0.40,
        r_factor=8.0,
        importance_factor=1.0,
        system_type="rc_smf",
    )
    story_elevations = [3.5, 7.0]
    story_weights = [res.story_weights[h] for h in story_elevations]

    elf_res = calculate_seismic_elf(
        story_elevations=story_elevations,
        story_weights=story_weights,
        params=params,
        units="kN, m",
    )

    # Period Ta = 0.0466 * 7.0^0.9 = 0.2685 s
    assert pytest.approx(elf_res.ta, abs=0.01) == 0.269
    # Cs = SDS / (R/Ie) = 1.0 / 8.0 = 0.1250
    assert pytest.approx(elf_res.cs, rel=1e-3) == 0.1250
    # Base shear V = 0.1250 * 1058.96 = 132.37 kN
    assert pytest.approx(elf_res.base_shear, rel=1e-3) == 132.37

    # Story force sum must equal base shear
    assert pytest.approx(sum(elf_res.story_forces.values()), rel=1e-3) == elf_res.base_shear


def test_frame_elf_equilibrium():
    """Verify finite element solution under ELF seismic point loads satisfies global base equilibrium."""
    struct = create_rc_frame(include_shear_deformation=True)
    node_grid = struct.node_grid
    beams = struct.beams
    core_model = model_to_core(struct)

    core_sdl = [ScDistributedLoad(element_id=bm.id, wz=-10.0, is_local=False) for bm in beams]
    core_model.load_cases = [LoadCase(id="DEAD", element_loads=core_sdl)]
    core_model = apply_self_weight(core_model, target_case_id="DEAD")

    ms = MassSource(
        id="MS_EQ",
        include_element_mass=False,
        factors=[MassSourceFactor(load_case_id="DEAD", factor=1.0)],
    )
    res = calculate_mass_source(core_model, mass_source=ms)

    params = SeismicParameters(sds=1.00, sd1=0.40, r_factor=8.0, importance_factor=1.0)
    story_elevations = [3.5, 7.0]
    story_weights = [res.story_weights[h] for h in story_elevations]

    elf_res = calculate_seismic_elf(
        story_elevations=story_elevations,
        story_weights=story_weights,
        params=params,
        units="kN, m",
    )

    # Apply lateral loads to face nodes at x=0
    eq_loads = []
    for idx, elev in enumerate(story_elevations, start=1):
        fx_tot = elf_res.story_forces[idx]
        face_nodes = [node.id for (x, y, z), node in node_grid.items() if abs(z - elev) < 1e-3 and abs(x - 0.0) < 1e-3]
        for nid in face_nodes:
            eq_loads.append(ScPointLoad(node_id=nid, fx=fx_tot / len(face_nodes)))

    core_model.load_cases.append(LoadCase(id="EQ_X", point_loads=eq_loads))

    struct_eq = model_from_core(core_model, load_case_id="EQ_X")
    disp, reac = struct_eq.solve()
    results = Results(struct_eq)
    df_reac = results.reactions()

    # Base reactions sum in X must equal -V
    sum_rx = float(df_reac["Fx"].sum())
    assert pytest.approx(sum_rx + elf_res.base_shear, abs=1e-4) == 0.0

    # Upper floor displacements must be positive and monotonic
    df_disp = results.node_displacements()
    roof_nodes = [node.id for (x, y, z), node in node_grid.items() if abs(z - 7.0) < 1e-3]
    mid_nodes = [node.id for (x, y, z), node in node_grid.items() if abs(z - 3.5) < 1e-3]

    roof_ux = float(df_disp[df_disp["node"].isin(roof_nodes)]["ux"].mean())
    mid_ux = float(df_disp[df_disp["node"].isin(mid_nodes)]["ux"].mean())

    assert roof_ux > mid_ux > 0.0


def test_frame_elf_example():
    """
    Verify frame_elf.py example matching SAP2000 benchmarks.
    Assertions strictly for approximate period, base shear, and top story one node displacement.
    """
    core_model = create_rc_frame_core(include_shear_deformation=True)
    beam_ids = [el.id for el in core_model.elements if el.section_id == "Beam_250x400"]

    # 1. Superimposed Dead and LIVE loads on beams
    sdl_loads = [ScDistributedLoad(element_id=eid, wz=-10.0) for eid in beam_ids]
    core_model.load_cases.append(LoadCase(id="SuperDead", name="Superimposed Dead Load", element_loads=sdl_loads))

    live_loads = [ScDistributedLoad(element_id=eid, wz=-4.0) for eid in beam_ids]
    core_model.load_cases.append(LoadCase(id="LIVE", name="Live Load", element_loads=live_loads))

    # 2. Mass source: 1.0 * ElementSelfMass + 1.0 * SuperDead + 0.25 * LIVE
    mass_source = MassSource(
        id="MS_ELF",
        include_element_mass=True,
        factors=[
            MassSourceFactor(load_case_id="SuperDead", factor=1.0),
            MassSourceFactor(load_case_id="LIVE", factor=0.25),
        ],
    )
    core_model.mass_source = mass_source

    # 3. ASCE 7-22 Seismic Parameters
    params = SeismicParameters(
        sds=1.00,
        sd1=0.40,
        s1=0.35,
        tl=8.0,
        r_factor=8.0,
        importance_factor=1.0,
        system_type="rc_smf",
        site_class="D",
        risk_category="I",
        method=2,
    )

    # 4. Generate lateral seismic loads via LoadGenerator
    generator = LoadGenerator(units="kN, m")
    eq_case, elf_res = generator.apply_lateral_seismic(
        core_model,
        params=params,
        direction="X",
        seismic_case_id="EQ_X",
        mass_source=mass_source,
        distribution="face",
    )

    # 5. Solve in fem3d
    struct = model_from_core(core_model, load_case_id="EQ_X")
    struct.solve()
    results = Results(struct)

    # 6. Top story one node displacement (Node 13 at x=0, y=0, z=7.0 m)
    df_disp = results.node_displacements()
    node13_ux_mm = float(df_disp[df_disp["node"] == 13]["ux"].iloc[0]) * 1000.0

    # Assertions: period, base shear, and top story one node displacement
    assert pytest.approx(elf_res.ta, rel=1e-3) == 0.269
    assert pytest.approx(elf_res.base_shear, rel=1e-3) == 132.37
    assert pytest.approx(node13_ux_mm, rel=1e-3) == 8.9326


