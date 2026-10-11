"""
Example: 3D Multi-Story Space Frame Response Spectrum Analysis (RSA).

Demonstrates:
1. Building a 4-story 3D reinforced concrete space frame model.
2. Response Spectrum Analysis along X direction using ASCE 7 design spectrum.
3. Response Spectrum Analysis along Y direction.
4. Calculation of 3D modal mass participation ratios (MMPR).
5. Evaluation of 3D member end forces (P, V2, V3, T, M2, M3).
6. Orthogonal directional combination: 100% X + 30% Y (ASCE 7 Section 12.5.3).
"""

import numpy as np
from fem3d import (
    Section,
    ElasticMaterial,
    SimpleFrame,
)

# -----------------------------------------------------------------------------
# 1. Geometry and Material Properties (Units: kips, inches, seconds)
# -----------------------------------------------------------------------------
g = 386.09  # in/s²
bay_x = 16.0 * 12.0  # 192 in (16 ft)
bay_y = 16.0 * 12.0  # 192 in (16 ft)
story_h = 10.0 * 12.0  # 120 in (10 ft)
n_stories = 4

E = 6000.0  # ksi
nu = 0.2
G = E / (2.0 * (1.0 + nu))  # 2500 ksi
material = ElasticMaterial(E=E, G=G, rho=0.0)

# Concrete Cross Sections
# Beams: 20 in x 35 in
b_bm, d_bm = 20.0, 35.0
beam_section = Section(
    A=b_bm * d_bm,
    I2=(d_bm * b_bm**3) / 12.0,
    I3=(b_bm * d_bm**3) / 12.0,
    J=0.3 * (b_bm**3) * d_bm,
)

# Columns: 20 in x 20 in
b_col = 20.0
col_section = Section(
    A=b_col * b_col,
    I2=(b_col**4) / 12.0,
    I3=(b_col**4) / 12.0,
    J=0.141 * (b_col**4),
)

# -----------------------------------------------------------------------------
# 2. Build 3D Structural Model using SimpleFrame
# -----------------------------------------------------------------------------
frame = SimpleFrame()

coords_xy = [
    (0.0, 0.0),
    (bay_x, 0.0),
    (bay_x, bay_y),
    (0.0, bay_y),
]

# Ground nodes (Story 0, Z=0)
for i, (x, y) in enumerate(coords_xy, start=1):
    frame.add_node(i, x, y, 0.0)
    frame.add_support(i, [1, 1, 1, 1, 1, 1])  # Fully fixed base

# Story nodes and lumped masses
# 450 kips total floor weight / 4 corner columns = 112.5 kips/node
mass_per_node = (450.0 / 4.0) / g  # kips*s²/in
node_id = 5

for story in range(1, n_stories + 1):
    z = story * story_h
    floor_nodes = []
    for x, y in coords_xy:
        node = frame.add_node(node_id, x, y, z)
        # Assign translational lumped mass and rotational inertia
        frame.structure.nodes[node_id].set_mass(
            mass=[mass_per_node, mass_per_node, mass_per_node],
            inertia=[1e-4, 1e-4, 1e-4],
        )
        floor_nodes.append(node_id)
        node_id += 1

elem_id = 1

# Columns
for story in range(n_stories):
    bot_offset = story * 4 + 1
    top_offset = (story + 1) * 4 + 1
    for c in range(4):
        bot_nid = bot_offset + c
        top_nid = top_offset + c
        frame.add_frame(elem_id, bot_nid, top_nid, material, col_section)
        elem_id += 1

# Beams in X and Y directions
for story in range(1, n_stories + 1):
    floor_base = story * 4 + 1
    # Beams parallel to X: (1-2) and (4-3)
    frame.add_frame(elem_id, floor_base, floor_base + 1, material, beam_section)
    elem_id += 1
    frame.add_frame(elem_id, floor_base + 3, floor_base + 2, material, beam_section)
    elem_id += 1

    # Beams parallel to Y: (1-4) and (2-3)
    frame.add_frame(elem_id, floor_base, floor_base + 3, material, beam_section)
    elem_id += 1
    frame.add_frame(elem_id, floor_base + 1, floor_base + 2, material, beam_section)
    elem_id += 1

print("=" * 70)
print("FEM3D: RESPONSE SPECTRUM ANALYSIS OF 4-STORY SPACE FRAME")
print("=" * 70)

# -----------------------------------------------------------------------------
# 3. RSA along Global X Direction (ASCE 7 Design Spectrum)
# -----------------------------------------------------------------------------
print("\n--- [Analysis 1] Seismic Response in X Direction (ASCE 7) ---")
rsa_x = frame.response_spectrum_analysis(
    sds=1.0,
    sd1=0.6,
    tl=8.0,
    n_modes=12,
    direction="x",
    scale_factor=g,
    combination="CQC",
)

periods = rsa_x["periods"]
gammas_x = rsa_x["participation_factors"]
mmpr_x = rsa_x["mass_participation_ratio"]

print("Modal Periods and X-Mass Participation:")
for i in range(min(6, len(periods))):
    print(
        f"  Mode {i+1}: T = {periods[i]:.4f} s | Gamma_X = {gammas_x[i]:.4f} | "
        f"MMPR_X = {mmpr_x[i]*100:.2f}%"
    )

vb_x = float(rsa_x["combined_base_shear"][0])
print(f"Total Combined Base Shear in X (Vbx): {vb_x:.2f} kips")

# -----------------------------------------------------------------------------
# 4. RSA along Global Y Direction (ASCE 7 Design Spectrum)
# -----------------------------------------------------------------------------
print("\n--- [Analysis 2] Seismic Response in Y Direction (ASCE 7) ---")
rsa_y = frame.response_spectrum_analysis(
    sds=1.0,
    sd1=0.6,
    tl=8.0,
    n_modes=12,
    direction="y",
    scale_factor=g,
    combination="CQC",
)

gammas_y = rsa_y["participation_factors"]
mmpr_y = rsa_y["mass_participation_ratio"]
vb_y = float(rsa_y["combined_base_shear"][0])

print(f"Total Combined Base Shear in Y (Vby): {vb_y:.2f} kips")

# -----------------------------------------------------------------------------
# 5. Member Internal Forces and 100% X + 30% Y Orthogonal Combination
# -----------------------------------------------------------------------------
# Base corner column (Element 1)
f_col_x = rsa_x["combined_element_forces"][1]
f_col_y = rsa_y["combined_element_forces"][1]

# 12 DOFs: [P_i, V2_i, V3_i, T_i, M2_i, M3_i, P_j, V2_j, V3_j, T_j, M2_j, M3_j]
# 100% X + 30% Y combination
f_comb = f_col_x + 0.3 * f_col_y

print("\n--- Base Column (Element 1) Response & Combination ---")
print(f"Under X-Spectrum: P = {abs(f_col_x[0]):.2f} k, V2 = {abs(f_col_x[1]):.2f} k, M3 = {abs(f_col_x[5]):.2f} k-in")
print(f"Under Y-Spectrum: P = {abs(f_col_y[0]):.2f} k, V3 = {abs(f_col_y[2]):.2f} k, M2 = {abs(f_col_y[4]):.2f} k-in")
print(f"ASCE 7 Orthogonal (100% X + 30% Y):")
print(f"  Design Axial P:    {abs(f_comb[0]):.2f} kips")
print(f"  Design Shear V2:   {abs(f_comb[1]):.2f} kips")
print(f"  Design Moment M3:  {abs(f_comb[5]):.2f} kip-in ({abs(f_comb[5])/12:.2f} kip-ft)")
print("=" * 70)
