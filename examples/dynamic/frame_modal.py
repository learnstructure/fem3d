"""
Example: 3D Multi-Story Space Frame Modal Analysis using structdyn.
Mirrors fem2d/examples/dynamic/hw4.py for 3D structures.

Units: kips, inches, seconds

Model Description:
A 4-story, 1-bay by 1-bay 3D reinforced concrete space frame:
- Bay width in X: 16 ft = 192 in
- Bay width in Y: 16 ft = 192 in
- Story height:   10 ft = 120 in (4 stories, total height = 480 in)
- Fixed at ground base (Z = 0)
- Concrete E = 6,000 ksi, nu = 0.2, G = 2,500 ksi
- Columns: 20 in x 20 in square section
- Beams:   20 in x 35 in rectangular section
- Floor lumped mass: 450 kips per story distributed to the 4 corner nodes
"""

import math
import numpy as np
from fem3d import (
    Structure,
    Node,
    ElasticMaterial,
    Section,
    FrameElement,
    SimpleFrame,
    DrawStructure,
    plot_mode_shape,
)
from structdyn.mdf.mdf import MDF

# -------------------------------------------------------------------
# 1. Model parameters (kips, inches, seconds)
# -------------------------------------------------------------------
g = 386.09        # in/s^2 (gravity)
bay_x = 16.0 * 12 # 192 in
bay_y = 16.0 * 12 # 192 in
story_h = 10.0 * 12 # 120 in
n_stories = 4

# Concrete material properties
E = 6000.0  # ksi
nu = 0.2
G = E / (2.0 * (1.0 + nu))  # 2500 ksi
material = ElasticMaterial(E=E, G=G, rho=0.0)

# Section properties
# Beams: 20 in x 35 in
b_bm, d_bm = 20.0, 35.0
A_bm = b_bm * d_bm
I3_bm = (b_bm * d_bm**3) / 12.0  # Strong axis bending (Z-axis transverse)
I2_bm = (d_bm * b_bm**3) / 12.0  # Weak axis bending
J_bm = 0.3 * (b_bm**3) * d_bm
beam_section = Section(A=A_bm, I2=I2_bm, I3=I3_bm, J=J_bm)

# Columns: 20 in x 20 in
b_col = 20.0
A_col = b_col * b_col
I_col = (b_col**4) / 12.0
J_col = 0.141 * (b_col**4)
col_section = Section(A=A_col, I2=I_col, I3=I_col, J=J_col)

# -------------------------------------------------------------------
# 2. Build 3D structure using SimpleFrame
# -------------------------------------------------------------------
frame = SimpleFrame()

# Node layout per floor: 4 corner columns
# 1: (0, 0), 2: (bay_x, 0), 3: (bay_x, bay_y), 4: (0, bay_y)
coords_xy = [
    (0.0, 0.0),
    (bay_x, 0.0),
    (bay_x, bay_y),
    (0.0, bay_y),
]

# Ground nodes (story 0)
for i, (x, y) in enumerate(coords_xy, start=1):
    frame.add_node(i, x, y, 0.0)
    frame.add_support(i, [1, 1, 1, 1, 1, 1])  # fully fixed base

# Story nodes (stories 1 to n_stories)
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
            E=E, A=A_col, I2=I_col, I3=I_col, J=J_col, G=G,
        )
        elem_id += 1

# Beams (perimeter at each floor level)
for story in range(1, n_stories + 1):
    story_base = story * 4 + 1
    floor_pairs = [
        (story_base + 0, story_base + 1),  # X-beam bottom
        (story_base + 1, story_base + 2),  # Y-beam right
        (story_base + 2, story_base + 3),  # X-beam top
        (story_base + 3, story_base + 0),  # Y-beam left
    ]
    for n1, n2 in floor_pairs:
        frame.add_frame(
            elem_id, n1, n2,
            E=E, A=A_bm, I2=I2_bm, I3=I3_bm, J=J_bm, G=G,
        )
        elem_id += 1

# -------------------------------------------------------------------
# 3. Lumped floor masses
# -------------------------------------------------------------------
# Each floor carries 450 kips, distributed equally among 4 corner nodes
mass_per_node = (450.0 / 4.0) / g  # kips*s^2/in
small_inertia = 1e-4

for story in range(1, n_stories + 1):
    story_base = story * 4 + 1
    for k in range(4):
        nid = story_base + k
        frame.structure.nodes[nid].set_mass(
            mass=[mass_per_node, mass_per_node, mass_per_node],
            inertia=[small_inertia, small_inertia, small_inertia],
        )

# -------------------------------------------------------------------
# 4. Create MDF dynamic system using structdyn
# -------------------------------------------------------------------
mdf = MDF.from_fem3d(frame)

# -------------------------------------------------------------------
# 5. Eigenvalue modal analysis
# -------------------------------------------------------------------
omega, phi = mdf.modal.modal_analysis(dof_normalize=None)
T = 2.0 * math.pi / omega
frequencies = omega / (2.0 * math.pi)

print("=" * 65)
print("3D MULTI-STORY SPACE FRAME MODAL ANALYSIS (via structdyn)")
print("=" * 65)
print(f"Total DOFs = {frame.structure.neq}, Free DOFs = {len(frame.structure.free_dofs)}")
print("\nFirst 6 Modes:")
print(f"{'Mode':<8} {'Period (s)':<14} {'Freq (Hz)':<14} {'Omega (rad/s)':<14}")
print("-" * 50)
for i in range(min(6, len(T))):
    print(f"{i+1:<8} {T[i]:<14.5f} {frequencies[i]:<14.5f} {omega[i]:<14.5f}")
print("=" * 65)

# -------------------------------------------------------------------
# 6. Register modal results on structure for 3D visualization
# -------------------------------------------------------------------
frame.set_modal_results(omega, phi)

# Option 1: Quick function call
# plot_mode_shape(frame.structure, mode=1)

# Option 2: Object-oriented style with DrawStructure
# drawer = DrawStructure(frame.structure)
# drawer.draw_mode_shape(mode=1)
