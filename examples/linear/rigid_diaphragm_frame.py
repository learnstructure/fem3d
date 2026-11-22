"""Minimal example: 3D RC space frame with rigid floor diaphragms, asymmetric mass, ELF, modal, and RSA."""

from pathlib import Path
import sys

import numpy as np

_curr_dir = Path(__file__).resolve().parent
_repo_dir = _curr_dir.parent.parent
for p in (str(_curr_dir), str(_repo_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fem3d import Results
from fem3d.adapters import model_from_core
from rc_frame_model import create_rc_frame_core
from struct_core import (
    Diaphragm,
    DistributedLoad as ScDistributedLoad,
    LoadCase,
    MassSource,
    MassSourceFactor,
    validate_structural_model,
)
from struct_load import LoadGenerator, SeismicParameters, calculate_mass_source


def main() -> None:
    model = create_rc_frame_core(include_shear_deformation=True)

    # Assign rigid Z-diaphragms at both elevated floors (Story1 at Z=3.5 m, Story2 at Z=7.0 m)
    model.diaphragms = [
        Diaphragm(id="D1", name="Story1", node_ids=[7, 8, 9, 10, 11, 12], axis="Z"),
        Diaphragm(id="D2", name="Story2", node_ids=[13, 14, 15, 16, 17, 18], axis="Z"),
    ]

    # Add asymmetric lumped mass along Y = 5.0 m so the floor center of mass shifts off the centroid (5.0, 2.5)
    for node in model.nodes:
        if node.z > 0.1 and abs(node.y - 5.0) < 1e-6:
            node.mass = 8000.0  # kg

    beam_ids = [el.id for el in model.elements if el.section_id == "Beam_250x400"]
    model.load_cases.extend([
        LoadCase(id="SuperDead", load_type="dead", element_loads=[ScDistributedLoad(element_id=eid, wz=-10.0) for eid in beam_ids]),
        LoadCase(id="LIVE", load_type="live", element_loads=[ScDistributedLoad(element_id=eid, wz=-4.0) for eid in beam_ids]),
    ])
    ms = MassSource(
        id="MS_ELF",
        include_element_mass=True,
        include_added_mass=True,
        factors=[MassSourceFactor(load_case_id="SuperDead", factor=1.0), MassSourceFactor(load_case_id="LIVE", factor=0.25)],
    )
    model.mass_source = ms
    validate_structural_model(model)

    # Mass summary and mass-weighted ASCE 7-22 ELF forces (resultant acts at floor COM)
    ms_res = calculate_mass_source(model, mass_source=ms)
    params = SeismicParameters(
        sds=1.0, sd1=0.4, s1=0.35, tl=8.0, r_factor=8.0, system_type="rc_smf", site_class="D", risk_category="II"
    )
    _, elf_res = LoadGenerator(units="kN, m").apply_lateral_seismic(
        model, params=params, direction="X", seismic_case_id="EQ_X", mass_source=ms
    )

    # Static ELF analysis in fem3d (exact u = T q transformation)
    struct = model_from_core(model, load_case_id="EQ_X", mass_source=ms, include_self_weight=False)
    struct.solve()
    res = Results(struct)
    _, M_r = struct.get_reduced_matrices(lumped=True, include_rotational_inertia=False)

    nodes_by_id = {n.id: n for n in model.nodes}
    print(f"ELF total weight W = {elf_res.total_weight:.2f} kN, base shear V = {elf_res.base_shear:.2f} kN")
    for did in ("D1", "D2"):
        dm = ms_res.diaphragm_masses[did]
        c1, _, cr = struct.diaphragm_masters[did]["master_dofs"]
        L_d = struct.diaphragm_masters[did]["scale_radius"]
        print(
            f"{did}: mass={dm.mass:.3f} t (fem3d={M_r[c1, c1]:.3f}), "
            f"COM=({dm.center_of_mass[0]:.3f}, {dm.center_of_mass[1]:.3f}) m, "
            f"I_zz={dm.rotational_inertia:.1f} t*m^2 (fem3d={M_r[cr, cr] * L_d**2:.1f})"
        )

    # Check rigid in-plane kinematics at Roof (D2) relative to node 13
    df_d = res.node_displacements().set_index("node")
    x0, y0 = nodes_by_id[13].x, nodes_by_id[13].y
    ux0, uy0, rz0 = df_d.loc[13, ["ux", "uy", "rz"]]
    max_kin_err = max(
        max(
            abs(df_d.loc[nid, "ux"] - (ux0 - (nodes_by_id[nid].y - y0) * rz0)),
            abs(df_d.loc[nid, "uy"] - (uy0 + (nodes_by_id[nid].x - x0) * rz0)),
            abs(df_d.loc[nid, "rz"] - rz0),
        )
        for nid in model.get_diaphragm("D2").node_ids
    )
    beam_max_axial = res.element_forces().loc[lambda df: df["element"].isin(beam_ids), "f1_i"].abs().max()
    print(f"Roof max rigid-kinematics residual = {max_kin_err:.3e} m")
    print(f"Max beam axial force under EQ_X    = {beam_max_axial:.3e} kN")

    # Modal & Response Spectrum Analysis (torsionally coupled modes)
    modal = struct.modal_analysis(num_modes=6, lumped=True, include_rotational_inertia=False)
    rsa = struct.response_spectrum_analysis(params=params, n_modes=6, direction="x", combination="CQC", scale_factor=9.80665)
    print("First 6 modal periods (s):", np.round(modal["periods"], 4).tolist())
    print(f"RSA combined X base shear  = {float(rsa['combined_base_shear'][0]):.2f} kN")


if __name__ == "__main__":
    main()
