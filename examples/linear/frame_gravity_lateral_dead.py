"""
Example: 3D Reinforced Concrete Space Frame under Combined Gravity, Lateral,
and Automated Element Self-Weight (Dead Load).

Constructed using the core Structure class (object-oriented direct stiffness API),
with element self-weights computed and materialized through the ecosystem bridge:
    fem3d.Structure -> struct_core.StructuralModel -> struct_load.apply_self_weight

Units:
    Forces:       kN
    Lengths:      m
    Stresses / E: kPa (kN/m^2)
    Moments:      kN*m

Structure Description:
    - 2 stories, height h = 3.5 m each (total height H = 7.0 m).
    - 2 bays along X: Bay 1 = 4.0 m, Bay 2 = 6.0 m (total length L_x = 10.0 m).
    - 1 bay along Y:  Bay width = 5.0 m (total width L_y = 5.0 m).
    - Columns: 0.30 m (along X) x 0.40 m (along Y) rectangular section.
        * A  = 0.1200 m^2
        * I2 = 0.0009 m^4 (bending about Y, resisting X-translation)
        * I3 = 0.0016 m^4 (bending about X, resisting Y-translation)
        * J  = 0.001944 m^4 (St. Venant torsion)
    - Beams: 0.25 m (width) x 0.40 m (depth) rectangular section.
        * A  = 0.1000 m^2
        * I3 = 0.001333 m^4 (strong-axis vertical bending)
        * I2 = 0.000521 m^4 (weak-axis horizontal bending)
        * J  = 0.001273 m^4 (St. Venant torsion)
    - Material: Standard concrete grade (C30):
        * E  = 30,000,000 kPa (30 GPa)
        * nu = 0.20
        * G  = 12,500,000 kPa (12.5 GPa)
        * unit_weight = 24.0 kN/m^3 (or rho = 2400 kg/m^3)
    - Supports: All 6 base nodes at ground level (Z = 0) are fully fixed [1, 1, 1, 1, 1, 1].

Loads:
    1. Superimposed Gravity Loads:
       - Uniform distributed downward load on all beams: wz = -10.0 kN/m.
       - Total beam length = 70.0 m -> Total superimposed gravity = 700.0 kN.
    2. Lateral Loads (along +X applied to x = 0.0 face):
       - Middle story (Z = 3.5 m): Fx = 40.0 kN at each node along x = 0 (2 nodes -> 80.0 kN).
       - Top story    (Z = 7.0 m): Fx = 20.0 kN at each node along x = 0 (2 nodes -> 40.0 kN).
       - Total lateral load = 80.0 + 40.0 = 120.0 kN.
    3. Self-Weight (Dead Load):
       - Computed automatically via `struct_load` engine on the `struct_core` model representation:
         * Columns: A = 0.12 m^2 * 24.0 kN/m^3 = 2.880 kN/m (total = 12 * 3.5m * 2.88 = 120.96 kN)
         * Beams:   A = 0.10 m^2 * 24.0 kN/m^3 = 2.400 kN/m (total = 70m * 2.40 = 168.00 kN)
         * Total self-weight = 288.96 kN.
       - Total vertical load = 700.0 + 288.96 = 988.96 kN.
"""

import math
import numpy as np
import pandas as pd

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

# -------------------------------------------------------------------
# 1. Configuration & Model Options
# -------------------------------------------------------------------
# Set to True to include Timoshenko shear deformation (matching SAP2000 default).
# Set to False for classical Euler-Bernoulli beam-column theory.
INCLUDE_SHEAR_DEFORMATION = True

# -------------------------------------------------------------------
# 2. Material & Section Properties (kN, m)
# -------------------------------------------------------------------
# Standard C30 Concrete: E = 30 GPa, nu = 0.20, unit_weight = 24.0 kN/m^3.
# (Shear modulus G = E / [2 * (1 + nu)] = 12.5 GPa is automatically derived)
material = ElasticMaterial(E=30.0e6, nu=0.20, unit_weight=24.0)

# Column: 0.3 m (along X) x 0.4 m (along Y)
# Vertical member local triad: local 2 is along global +Y, local 3 is along global -X.
# Therefore depth (local 2) = 0.4 m, width (local 3) = 0.3 m.
col_section = Section.from_rectangle(width=0.30, depth=0.40)

# Beam: 0.25 m width (horizontal) x 0.40 m depth (vertical)
# Horizontal member local triad: local 2 is vertical (+Z), local 3 is horizontal.
# Therefore depth (local 2) = 0.4 m, width (local 3) = 0.25 m.
beam_section = Section.from_rectangle(width=0.25, depth=0.40)

# -------------------------------------------------------------------
# 3. Initialize Structure and Grid Geometry (Explicit Units)
# -------------------------------------------------------------------
struct = Structure(units="kN, m")

x_coords = [0.0, 4.0, 10.0]  # 2 bays: 4m and 6m
y_coords = [0.0, 5.0]        # 1 bay: 5m
z_coords = [0.0, 3.5, 7.0]   # 2 stories: 3.5m each

# Create nodes (total = 3 * 2 * 3 = 18 nodes)
node_grid = {}
nid = 1
for z in z_coords:
    for y in y_coords:
        for x in x_coords:
            node = Node(nid, x, y, z)
            if z == 0.0:
                # All bases fully fixed: [ux, uy, uz, rx, ry, rz]
                node.set_support([1, 1, 1, 1, 1, 1])
            struct.add_node(node)
            node_grid[(x, y, z)] = node
            nid += 1

# -------------------------------------------------------------------
# 4. Create Column Elements (Vertical Members)
# -------------------------------------------------------------------
# 6 column lines across 2 stories = 12 column elements
elem_id = 1
columns = []
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
                include_shear_deformation=INCLUDE_SHEAR_DEFORMATION,
            )
            struct.add_element(col_elem)
            columns.append(col_elem)
            elem_id += 1

# -------------------------------------------------------------------
# 5. Create Beam Elements (Horizontal Members)
# -------------------------------------------------------------------
# 14 beam elements (7 per floor level):
# - 4 beams along X per floor (2 lines along Y * 2 bays along X)
# - 3 beams along Y per floor (3 lines along X * 1 bay along Y)
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
                include_shear_deformation=INCLUDE_SHEAR_DEFORMATION,
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
            include_shear_deformation=INCLUDE_SHEAR_DEFORMATION,
        )
        struct.add_element(bm_elem)
        beams.append(bm_elem)
        elem_id += 1

# -------------------------------------------------------------------
# 6. Apply Superimposed Gravity & Lateral Loads
# -------------------------------------------------------------------
# (A) Superimposed Gravity Load: 10 kN/m downwards on all 14 beams
for bm in beams:
    grav_load = DistributedLoad(bm, wz=-10.0, coord_system="global")
    struct.add_load(grav_load)

# (B) Lateral Loads: along +X on the x = 0.0 face
# Middle story (z = 3.5 m): 40 kN at nodes along x = 0 (Nodes 7, 10)
for y in y_coords:
    node_mid = node_grid[(0.0, y, 3.5)]
    struct.add_load(PointLoad(node_mid, fx=40.0))

# Top story (z = 7.0 m): 20 kN at nodes along x = 0 (Nodes 13, 16)
for y in y_coords:
    node_top = node_grid[(0.0, y, 7.0)]
    struct.add_load(PointLoad(node_top, fx=20.0))

# -------------------------------------------------------------------
# 7. Automated Self-Weight & Ecosystem Round-Trip
# -------------------------------------------------------------------
# Convert to standard struct_core representation (units & loads mapped to LIVE)
core_model = model_to_core(struct)

# Materialize self-weight distributed loads into DEAD load case
# (Infers gravity direction '-z' and unit system from core_model automatically)
core_model = apply_self_weight(core_model)

# Re-import from struct_core into analysis model.
# By default, model_from_core transfers and sums all load cases (DEAD + LIVE).
# Users can also specify a case explicitly: model_from_core(core_model, load_case_id="DEAD")
struct_analyzed = model_from_core(core_model)

# -------------------------------------------------------------------
# 8. Solve Linear Static Analysis
# -------------------------------------------------------------------
disp, reactions = struct_analyzed.solve()

# -------------------------------------------------------------------
# 9. Post-Processing & Result Display
# -------------------------------------------------------------------
results = Results(struct_analyzed)
df_displacements = results.node_displacements()
df_reactions = results.reactions()
df_forces = results.element_forces()

dead_case = next(lc for lc in core_model.load_cases if lc.id == "DEAD")
total_sw = sum(abs(getattr(dl, "wz", 0.0)) * struct_analyzed.elements[dl.element_id].length for dl in dead_case.element_loads)

print("=" * 80)
print("3D RC SPACE FRAME: COMBINED GRAVITY, LATERAL & AUTOMATED SELF-WEIGHT (DEAD)")
print("=" * 80)
shear_mode = "Timoshenko (Shear On)" if INCLUDE_SHEAR_DEFORMATION else "Euler-Bernoulli (Shear Off)"
print(f"Theory:           {shear_mode}")
print(f"Total Nodes:      {len(struct_analyzed.nodes)}, Total Elements: {len(struct_analyzed.elements)} (12 columns, 14 beams)")
print(f"Total DOFs:       {struct_analyzed.neq}, Free DOFs: {len(struct_analyzed.free_dofs)}")
print(f"Self-Weight Mode: Materialized via struct_load engine ({len(dead_case.element_loads)} elements loaded)")
print(f"Concrete Density: Unit Weight = {material.unit_weight:.2f} kN/m^3")
print(f"Total Self-Weight: {total_sw:.4f} kN (Columns: {12 * 3.5 * (col_section.A * material.unit_weight):.2f} kN, Beams: {70.0 * (beam_section.A * material.unit_weight):.2f} kN)")

# Global Equilibrium Verification
r_sum = df_reactions[["Fx", "Fy", "Fz"]].sum()
expected_fx = -120.0
expected_fy = 0.0
expected_fz = 700.0 + total_sw

print("\n" + "-" * 80)
print("BASE EQUILIBRIUM CHECK:")
print("-" * 80)
print(f"Sum of Base Reactions:  Rx = {r_sum['Fx']:+10.4f} kN   (Applied: +120.0000 kN)")
print(f"                        Ry = {r_sum['Fy']:+10.4f} kN   (Applied:    0.0000 kN)")
print(f"                        Rz = {r_sum['Fz']:+10.4f} kN   (Applied: -{expected_fz:.4f} kN)")
eq_err_x = abs(r_sum['Fx'] - expected_fx)
eq_err_y = abs(r_sum['Fy'] - expected_fy)
eq_err_z = abs(r_sum['Fz'] - expected_fz)
print(f"Equilibrium Residuals:  |dFx| = {eq_err_x:.1e} kN, |dFy| = {eq_err_y:.1e} kN, |dFz| = {eq_err_z:.1e} kN")

# Support Reactions Table
print("\n" + "-" * 80)
print("SUPPORT REACTIONS AT FIXED BASE (Z = 0.0 m):")
print("-" * 80)
base_df = df_reactions[df_reactions["node"] <= 6].copy()
base_nodes_info = []
for nid in base_df["node"]:
    n = struct_analyzed.nodes[nid]
    base_nodes_info.append(f"({n.x:.1f}, {n.y:.1f})")
base_df["Coord (x, y)"] = base_nodes_info
cols_order = ["node", "Coord (x, y)", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]
print(base_df[cols_order].to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

# Nodal Displacements Table for Upper Floors
print("\n" + "-" * 80)
print("ROOF DISPLACEMENTS (STORY 2, Z = 7.0 m):")
print("-" * 80)
roof_df = df_displacements[df_displacements["node"] >= 13].copy()
coords_info = []
for nid in roof_df["node"]:
    n = struct_analyzed.nodes[nid]
    coords_info.append(f"({n.x:.1f}, {n.y:.1f})")
roof_df["Coord (x, y)"] = coords_info
roof_df["ux (mm)"] = roof_df["ux"] * 1000.0
roof_df["uy (mm)"] = roof_df["uy"] * 1000.0
roof_df["uz (mm)"] = roof_df["uz"] * 1000.0
roof_cols = ["node", "Coord (x, y)", "ux (mm)", "uy (mm)", "uz (mm)", "rx", "ry", "rz"]
print(roof_df[roof_cols].to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

print("\n" + "-" * 80)
print("MIDDLE STORY DISPLACEMENTS (STORY 1, Z = 3.5 m):")
print("-" * 80)
mid_df = df_displacements[(df_displacements["node"] >= 7) & (df_displacements["node"] <= 12)].copy()
coords_info_mid = []
for nid in mid_df["node"]:
    n = struct_analyzed.nodes[nid]
    coords_info_mid.append(f"({n.x:.1f}, {n.y:.1f})")
mid_df["Coord (x, y)"] = coords_info_mid
mid_df["ux (mm)"] = mid_df["ux"] * 1000.0
mid_df["uy (mm)"] = mid_df["uy"] * 1000.0
mid_df["uz (mm)"] = mid_df["uz"] * 1000.0
print(mid_df[roof_cols].to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

print("=" * 80)


# Results with shear deformation (SAP2000 benchmark):
# --------------------------------------------------------------------------------
# SUPPORT REACTIONS AT FIXED BASE (Z = 0.0 m):
# --------------------------------------------------------------------------------
#  node Coord (x, y)         Fx         Fy         Fz         Mx         My         Mz
#     1   (0.0, 0.0)   -17.6877     3.3330   102.2278    -3.8614   -37.0302    -0.0000
#     2   (4.0, 0.0)   -20.2623     3.3330   228.5083    -3.8614   -39.8657    -0.0000
#     3  (10.0, 0.0)   -22.0500     3.3330   163.7439    -3.8614   -41.8720    -0.0000
#     4   (0.0, 5.0)   -17.6877    -3.3330   102.2278     3.8614   -37.0302    -0.0000
#     5   (4.0, 5.0)   -20.2623    -3.3330   228.5083     3.8614   -39.8657    -0.0000
#     6  (10.0, 5.0)   -22.0500    -3.3330   163.7439     3.8614   -41.8720    -0.0000
#
# --------------------------------------------------------------------------------
# ROOF DISPLACEMENTS (STORY 2, Z = 7.0 m):
# --------------------------------------------------------------------------------
#  node Coord (x, y)    ux (mm)    uy (mm)    uz (mm)         rx         ry         rz
#    13   (0.0, 0.0)     5.9713     0.0087    -0.1433    -0.0003     0.0004     0.0000
#    14   (4.0, 0.0)     5.9458     0.0087    -0.3224    -0.0003     0.0004     0.0000
#    15  (10.0, 0.0)     5.9148     0.0087    -0.2260    -0.0003    -0.0004     0.0000
#    16   (0.0, 5.0)     5.9713    -0.0087    -0.1433     0.0003     0.0004     0.0000
#    17   (4.0, 5.0)     5.9458    -0.0087    -0.3224     0.0003     0.0004     0.0000
#    18  (10.0, 5.0)     5.9148    -0.0087    -0.2260     0.0003    -0.0004     0.0000
#
# --------------------------------------------------------------------------------
# MIDDLE STORY DISPLACEMENTS (STORY 1, Z = 3.5 m):
# --------------------------------------------------------------------------------
#  node Coord (x, y)    ux (mm)    uy (mm)    uz (mm)         rx         ry         rz
#     7   (0.0, 0.0)     3.7687    -0.0059    -0.0945    -0.0001     0.0008     0.0000
#     8   (4.0, 0.0)     3.7377    -0.0059    -0.2173    -0.0001     0.0006     0.0000
#     9  (10.0, 0.0)     3.7247    -0.0059    -0.1543    -0.0001     0.0004     0.0000
#    10   (0.0, 5.0)     3.7687     0.0059    -0.0945     0.0001     0.0008     0.0000
#    11   (4.0, 5.0)     3.7377     0.0059    -0.2173     0.0001     0.0006     0.0000
#    12  (10.0, 5.0)     3.7247     0.0059    -0.1543     0.0001     0.0004     0.0000
# ================================================================================

