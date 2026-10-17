"""
Example: 3D Reinforced Concrete Space Frame - ASCE 7-22 Equivalent Lateral Force (ELF) Procedure.

Demonstrates:
1. Reusable model geometry imported from `rc_frame_model.py` via `create_rc_frame_core`.
2. Defining Superimposed Dead ("SuperDead") and LIVE gravity load cases directly on `struct_core.StructuralModel`.
3. Specifying Mass Source with `include_element_mass=True` and load multipliers (1.0 * SuperDead + 0.25 * LIVE).
4. Automated ASCE 7-22 ELF seismic load generation via `struct_load.LoadGenerator.apply_lateral_seismic`,
   incorporating short-period SDS, 1-sec SD1, mapped S1, transition TL, site class, and Method 2.
5. Converting the core model to a `fem3d.Structure` via `fem3d.adapters.model_from_core` for FEA solution.
6. Displaying package-calculated base shear, story shears, base reactions, and upper story nodal displacements
   for direct verification against SAP2000.

Units:
    Forces:       kN
    Lengths:      m
    Stresses / E: kPa (kN/m^2)
    Moments:      kN*m
"""

from pathlib import Path
import sys

# Ensure local example modules and ecosystem packages are importable
_curr_dir = Path(__file__).resolve().parent
_repo_dir = _curr_dir.parent.parent
for p in (str(_curr_dir), str(_repo_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fem3d import Results
from fem3d.adapters import model_from_core
from struct_core import (
    DistributedLoad as ScDistributedLoad,
    LoadCase,
    MassSource,
    MassSourceFactor,
)
from struct_load import LoadGenerator, SeismicParameters
from rc_frame_model import create_rc_frame_core


def main():
    # -------------------------------------------------------------------
    # 1. Build Reusable Frame Model in struct_core
    # -------------------------------------------------------------------
    core_model = create_rc_frame_core(include_shear_deformation=True)

    # -------------------------------------------------------------------
    # 2. Define Gravity Loads directly on core_model (SuperDead & LIVE)
    # -------------------------------------------------------------------
    # Identify beam element IDs (section: Beam_250x400)
    beam_ids = [el.id for el in core_model.elements if el.section_id == "Beam_250x400"]

    # Superimposed Dead = 10 kN/m on all beams
    sdl_loads = [ScDistributedLoad(element_id=eid, wz=-10.0) for eid in beam_ids]
    core_model.load_cases.append(
        LoadCase(id="SuperDead", name="Superimposed Dead Load", element_loads=sdl_loads)
    )

    # Live Load = 4 kN/m on all beams
    live_loads = [ScDistributedLoad(element_id=eid, wz=-4.0) for eid in beam_ids]
    core_model.load_cases.append(
        LoadCase(id="LIVE", name="Live Load", element_loads=live_loads)
    )

    # -------------------------------------------------------------------
    # 3. Define Mass Source as Load Case Multipliers (ASCE 7 Sec 12.7.2)
    # -------------------------------------------------------------------
    # Self-weight is accounted for directly via include_element_mass=True.
    # Effective seismic weight: W = 1.0 * ElementSelfMass + 1.0 * SuperDead + 0.25 * LIVE
    mass_source = MassSource(
        id="MS_ELF",
        include_element_mass=True,
        factors=[
            MassSourceFactor(load_case_id="SuperDead", factor=1.0),
            MassSourceFactor(load_case_id="LIVE", factor=0.25),
        ],
    )
    core_model.mass_source = mass_source

    # -------------------------------------------------------------------
    # 4. Generate Lateral Seismic Loads via struct_load.LoadGenerator
    # -------------------------------------------------------------------
    # Complete ASCE 7-22 Section 12.8 Seismic Parameters:
    #   - SDS = 1.00g, SD1 = 0.40g, S1 = 0.35g, TL = 8.0s
    #   - Site Class D, Risk Category I (Ie = 1.0), RC Special Moment Frame (R = 8.0)
    #   - Method 2 (Eq. 12.8-3 to 12.8-5)
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

    generator = LoadGenerator(units="kN, m")
    eq_case, elf_result = generator.apply_lateral_seismic(
        core_model,
        params=params,
        direction="X",
        seismic_case_id="EQ_X",
        mass_source=mass_source,
        distribution="face",
    )

    # -------------------------------------------------------------------
    # 5. Convert to fem3d.Structure via Adapter and Solve Linear Static FEA
    # -------------------------------------------------------------------
    struct = model_from_core(core_model, load_case_id="EQ_X")
    struct.solve()
    results = Results(struct)

    # -------------------------------------------------------------------
    # 6. Output Results for SAP2000 Verification
    # -------------------------------------------------------------------
    print("=" * 80)
    print("3D RC SPACE FRAME: ASCE 7-22 EQUIVALENT LATERAL FORCE (ELF) SEISMIC PROCEDURE")
    print("=" * 80)
    print("ASCE 7-22 Parameters:")
    print(f"  SDS = {params.sds:.2f}g, SD1 = {params.sd1:.2f}g, S1 = {params.s1:.2f}g, TL = {params.tl:.1f}s")
    print(f"  Site Class: {params.site_class}, Risk Category: {params.risk_category}")
    print(f"  System: RC SMF (R = {params.r_factor:.1f}), Risk Cat {params.risk_category} (Ie = {params.importance_factor:.2f}), Method: {elf_result.method_used}")
    print(f"\nELF Analysis Summary:")
    print(f"  Total Effective Seismic Weight (W) : {elf_result.total_weight:10.2f} kN")
    print(f"  Approximate Period (Ta)            : {elf_result.ta:10.3f} s  (Cu = {elf_result.cu:.2f})")
    print(f"  Vertical Exponent (k)              : {elf_result.k:10.2f}")
    print(f"  Seismic Response Coeff (Cs)        : {elf_result.cs:10.4f}")
    print(f"  Design Base Shear (V)              : {elf_result.base_shear:10.2f} kN")

    print("\n" + "-" * 80)
    print(f"{'Story':<8} | {'Elev (m)':<10} | {'Weight Wi (kN)':<16} | {'Cvx':<10} | {'Force Fx (kN)':<15} | {'Shear Vx (kN)':<15}")
    print("-" * 80)
    for idx, elev in enumerate(core_model.story_elevations, start=1):
        w_i = elf_result.story_weights[idx]
        cvx_i = elf_result.cvx[idx] if elf_result.cvx else 0.0
        f_i = elf_result.story_forces[idx]
        v_i = elf_result.story_shears[idx] if elf_result.story_shears else f_i
        print(f"Level {idx:<2} | {elev:<10.1f} | {w_i:<16.2f} | {cvx_i:<10.4f} | {f_i:<15.2f} | {v_i:<15.2f}")
    print("-" * 80)

    # Support Reactions at Base (Z = 0.0 m)
    print("\n" + "-" * 80)
    print("SUPPORT REACTIONS AT FIXED BASE (Z = 0.0 m):")
    print("-" * 80)
    df_reactions = results.reactions()
    base_df = df_reactions[df_reactions["node"] <= 6].copy()
    base_df["Coord (x, y)"] = [
        f"({struct.nodes[nid].x:.1f}, {struct.nodes[nid].y:.1f})" for nid in base_df["node"]
    ]
    cols = ["node", "Coord (x, y)", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    print(base_df[cols].to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    total_rx = base_df["Fx"].sum()
    print(f"\nSum of Base Fx Reactions : {total_rx:10.2f} kN (Equilibrium check vs V = {elf_result.base_shear:.2f} kN)")

    # Nodal Displacements for Upper Stories
    df_disp = results.node_displacements()

    print("\n" + "-" * 80)
    print("ROOF DISPLACEMENTS (STORY 2, Z = 7.0 m):")
    print("-" * 80)
    roof_df = df_disp[df_disp["node"] >= 13].copy()
    roof_df["Coord (x, y)"] = [
        f"({struct.nodes[nid].x:.1f}, {struct.nodes[nid].y:.1f})" for nid in roof_df["node"]
    ]
    roof_df["ux (mm)"] = roof_df["ux"] * 1000.0
    roof_df["uy (mm)"] = roof_df["uy"] * 1000.0
    roof_df["uz (mm)"] = roof_df["uz"] * 1000.0
    disp_cols = ["node", "Coord (x, y)", "ux (mm)", "uy (mm)", "uz (mm)", "rx", "ry", "rz"]
    print(roof_df[disp_cols].to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    print("\n" + "-" * 80)
    print("MIDDLE STORY DISPLACEMENTS (STORY 1, Z = 3.5 m):")
    print("-" * 80)
    mid_df = df_disp[(df_disp["node"] >= 7) & (df_disp["node"] <= 12)].copy()
    mid_df["Coord (x, y)"] = [
        f"({struct.nodes[nid].x:.1f}, {struct.nodes[nid].y:.1f})" for nid in mid_df["node"]
    ]
    mid_df["ux (mm)"] = mid_df["ux"] * 1000.0
    mid_df["uy (mm)"] = mid_df["uy"] * 1000.0
    mid_df["uz (mm)"] = mid_df["uz"] * 1000.0
    print(mid_df[disp_cols].to_string(index=False, float_format=lambda x: f"{x:10.4f}"))
    print("=" * 80)


if __name__ == "__main__":
    main()
