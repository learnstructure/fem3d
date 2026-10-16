"""
Example: 3D Multi-Story Space Frame Time-History Analysis using structdyn.
Mirrors fem2d/examples/dynamic/frame_th.py for 3D structures.

Units: kips, inches, seconds

Model Description:
A 4-story, 1-bay by 1-bay 3D space frame subjected to the 1940 El Centro
earthquake ground acceleration record.
- 5% modal damping (zeta = 0.05) applied to the first 3 modes
- Solved via Newmark-Beta average acceleration method (beta=1/4, gamma=1/2)
- Compares response under X-direction excitation vs Y-direction excitation
"""

import numpy as np
import pandas as pd
from fem3d import SimpleFrame, Section
from structdyn import MDF, GroundMotion, elcentro_chopra

# -------------------------------------------------------------------
# 1. Model parameters (kips, inches, seconds)
# -------------------------------------------------------------------
g = 386.09          # in/s^2 (gravity)
bay_x = 16.0 * 12   # 192 in
bay_y = 16.0 * 12   # 192 in
story_h = 10.0 * 12 # 120 in
n_stories = 4

E = 6000.0  # ksi
nu = 0.2
G = E / (2.0 * (1.0 + nu))  # 2500 ksi

# Section properties using exact rectangular elasticity theory (matching SAP2000):
# Beams: 20 in x 35 in (J = 60,031.9 in^4)
beam_section = Section.from_rectangle(width=20.0, depth=35.0)

# Columns: 20 in x 20 in (J = 22,533.3 in^4)
col_section = Section.from_rectangle(width=20.0, depth=20.0)

# -------------------------------------------------------------------
# 2. Build 3D structure using SimpleFrame
# -------------------------------------------------------------------
frame = SimpleFrame()

coords_xy = [
    (0.0, 0.0),
    (bay_x, 0.0),
    (bay_x, bay_y),
    (0.0, bay_y),
]

# Ground nodes (fixed)
for i, (x, y) in enumerate(coords_xy, start=1):
    frame.add_node(i, x, y, 0.0)
    frame.add_support(i, [1, 1, 1, 1, 1, 1])

# Story nodes
node_id = 5
for story in range(1, n_stories + 1):
    z = story * story_h
    for x, y in coords_xy:
        frame.add_node(node_id, x, y, z)
        node_id += 1

# Columns
elem_id = 1
for story in range(1, n_stories + 1):
    lower_base = (story - 1) * 4 + 1
    upper_base = story * 4 + 1
    for k in range(4):
        bot_nid = lower_base + k
        top_nid = upper_base + k
        frame.add_frame(
            elem_id, bot_nid, top_nid,
            E=E, G=G,
            A=col_section.A, I2=col_section.I2, I3=col_section.I3, J=col_section.J,
        )
        elem_id += 1

# Beams
for story in range(1, n_stories + 1):
    story_base = story * 4 + 1
    floor_pairs = [
        (story_base + 0, story_base + 1),
        (story_base + 1, story_base + 2),
        (story_base + 2, story_base + 3),
        (story_base + 3, story_base + 0),
    ]
    for n1, n2 in floor_pairs:
        frame.add_frame(
            elem_id, n1, n2,
            E=E, G=G,
            A=beam_section.A, I2=beam_section.I2, I3=beam_section.I3, J=beam_section.J,
        )
        elem_id += 1

# Lumped story masses (450 kips per story distributed equally)
mass_per_node = (450.0 / 4.0) / g

for story in range(1, n_stories + 1):
    story_base = story * 4 + 1
    for k in range(4):
        nid = story_base + k
        frame.structure.nodes[nid].set_mass(
            mass=[mass_per_node, mass_per_node, mass_per_node],
        )

# -------------------------------------------------------------------
# 3. Create MDF with 5% modal damping on first 3 modes
# -------------------------------------------------------------------
mdf = MDF.from_fem3d(frame, zeta=0.05, n_modes=3)

# -------------------------------------------------------------------
# 4. Load El Centro earthquake record
# -------------------------------------------------------------------
elc = elcentro_chopra()
gm = GroundMotion.from_arrays(elc["acc (g)"], dt=0.02, scale_factor=g)

print("=" * 65)
print("3D SPACE FRAME TIME-HISTORY ANALYSIS (via structdyn)")
print("=" * 65)
print(f"Ground Motion: El Centro 1940, Duration: {gm.time[-1]:.2f} s ({len(gm.time)} steps)")
print(f"Peak Ground Acceleration: {np.max(np.abs(gm.acceleration)) / g:.3f} g")

# -------------------------------------------------------------------
# 5. Run dynamic analysis in X direction
# -------------------------------------------------------------------
print("\nRunning Newmark-Beta integration for X-direction excitation...")
resp_x = mdf.find_response_ground_motion(gm, direction="x", method="newmark_beta")

# Find the roof node (corner node at story 4: node 17)
# Node 17 has global DOFs: frame.structure.nodes[17].dofs
roof_node = frame.structure.nodes[17]
# In mdf free DOFs, find the index corresponding to roof ux
roof_ux_global = roof_node.dofs[0]  # ux global DOF
free_dof_idx_x = frame.structure.free_dofs.index(roof_ux_global)
col_name_ux = f"u{free_dof_idx_x + 1}"

peak_ux_pos = resp_x[col_name_ux].max()
peak_ux_neg = resp_x[col_name_ux].min()
peak_ux_abs = resp_x[col_name_ux].abs().max()

print(f"Roof Node {roof_node.id} X-displacement Response:")
print(f"  Max (Positive): {peak_ux_pos:+.4f} in")
print(f"  Min (Negative): {peak_ux_neg:+.4f} in")
print(f"  Peak Absolute : {peak_ux_abs:.4f} in")

# -------------------------------------------------------------------
# 6. Run dynamic analysis in Y direction
# -------------------------------------------------------------------
print("\nRunning Newmark-Beta integration for Y-direction excitation...")
resp_y = mdf.find_response_ground_motion(gm, direction="y", method="newmark_beta")

roof_uy_global = roof_node.dofs[1]  # uy global DOF
free_dof_idx_y = frame.structure.free_dofs.index(roof_uy_global)
col_name_uy = f"u{free_dof_idx_y + 1}"

peak_uy_pos = resp_y[col_name_uy].max()
peak_uy_neg = resp_y[col_name_uy].min()
peak_uy_abs = resp_y[col_name_uy].abs().max()

print(f"Roof Node {roof_node.id} Y-displacement Response:")
print(f"  Max (Positive): {peak_uy_pos:+.4f} in")
print(f"  Min (Negative): {peak_uy_neg:+.4f} in")
print(f"  Peak Absolute : {peak_uy_abs:.4f} in")
print("=" * 65)
