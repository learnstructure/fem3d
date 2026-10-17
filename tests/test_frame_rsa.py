"""
Test suite validating 3D RC Space Frame Response Spectrum Analysis (RSA) per ASCE 7-22.
"""

import pytest
import numpy as np

from fem3d.examples.linear.rc_frame_model import create_rc_frame_core
from fem3d.adapters import model_from_core
from struct_core import (
    DistributedLoad as ScDistributedLoad,
    LoadCase,
    MassSource,
    MassSourceFactor,
)
from struct_load import (
    SeismicParameters,
    calculate_design_response_spectrum,
)


def test_frame_rsa_asce7_22():
    """Verify ASCE 7-22 Response Spectrum Analysis on 3D RC Space Frame."""
    core_model = create_rc_frame_core(include_shear_deformation=True)
    beam_ids = [el.id for el in core_model.elements if el.section_id == "Beam_250x400"]

    sdl_loads = [ScDistributedLoad(element_id=eid, wz=-10.0) for eid in beam_ids]
    core_model.load_cases.append(LoadCase(id="SuperDead", name="Superimposed Dead Load", element_loads=sdl_loads))

    live_loads = [ScDistributedLoad(element_id=eid, wz=-4.0) for eid in beam_ids]
    core_model.load_cases.append(LoadCase(id="LIVE", name="Live Load", element_loads=live_loads))

    mass_source = MassSource(
        id="MS_RSA",
        include_element_mass=True,
        factors=[
            MassSourceFactor(load_case_id="SuperDead", factor=1.0),
            MassSourceFactor(load_case_id="LIVE", factor=0.25),
        ],
    )
    core_model.mass_source = mass_source

    params = SeismicParameters(
        sds=1.00,
        sd1=0.40,
        tl=8.0,
        site_class="D",
        damping_ratio=0.05,
    )

    # Test response spectrum generation in struct_load
    spec = calculate_design_response_spectrum(params)
    assert spec["t0"] == 0.08
    assert spec["ts"] == 0.40
    assert spec["sds"] == 1.00
    assert spec["site_class"] == "D"
    assert spec["damping_ratio"] == 0.05
    assert len(spec["periods"]) > 50

    struct = model_from_core(core_model)
    g = 9.80665

    rsa_x = struct.response_spectrum_analysis(
        params=params,
        n_modes=12,
        direction="x",
        scale_factor=g,
        combination="CQC",
    )

    rsa_y = struct.response_spectrum_analysis(
        params=params,
        n_modes=12,
        direction="y",
        scale_factor=g,
        combination="CQC",
    )

    # 1. Fundamental periods (matching SAP2000 benchmarks: 0.457663 s, 0.416672 s, 0.387409 s)
    assert pytest.approx(rsa_x["periods"][0], rel=1e-3) == 0.4577
    assert pytest.approx(rsa_x["periods"][1], rel=1e-3) == 0.4167
    assert pytest.approx(rsa_x["periods"][2], rel=1e-3) == 0.3874

    # 2. Modal mass participation ratios
    assert pytest.approx(rsa_x["mass_participation_ratio"][0] * 100.0, rel=1e-2) == 90.33
    assert pytest.approx(rsa_y["mass_participation_ratio"][1] * 100.0, rel=1e-2) == 83.67
    assert sum(rsa_x["mass_participation_ratio"]) >= 0.90  # ASCE 7 >= 90%
    assert sum(rsa_y["mass_participation_ratio"]) >= 0.90  # ASCE 7 >= 90%

    # 3. Combined Base shears (elastic)
    vb_x = float(rsa_x["combined_base_shear"][0])
    vb_y = float(rsa_y["combined_base_shear"][0])
    assert pytest.approx(vb_x, rel=1e-3) == 818.81
    assert pytest.approx(vb_y, rel=1e-3) == 854.47

    # 4. Roof displacement at Node 13 under RSA-X
    u_comb_x = rsa_x["combined_displacements"]
    dof_x_node13 = struct.nodes[13].dofs[0]
    ux_13_mm = float(u_comb_x[struct.free_dofs.index(dof_x_node13)]) * 1000.0
    assert pytest.approx(ux_13_mm, rel=1e-3) == 54.897
