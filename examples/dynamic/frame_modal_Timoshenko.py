"""
Example: 3D Multi-Story Space Frame Modal Analysis with Timoshenko Shear Deformation.
Compares natural vibration periods with SAP2000 results including shear deformation.

Units: kips, inches, seconds

Model Description:
A 4-story, 1-bay by 1-bay 3D reinforced concrete space frame:
- Bay width in X: 16 ft = 192 in
- Bay width in Y: 16 ft = 192 in
- Story height:   10 ft = 120 in (4 stories, total height = 480 in)
- Fixed at ground base (Z = 0)
- Concrete E = 6,000 ksi, nu = 0.2, G = 2,500 ksi
- Columns: 20 in x 20 in square section (As2 = As3 = 5/6 * A = 333.33 in^2)
- Beams:   20 in x 35 in rectangular section (As2 = As3 = 5/6 * A = 583.33 in^2)
- Floor lumped mass: 450 kips per story distributed to the 4 corner nodes
- Massless rotational DOFs condensed via exact Guyan reduction in structdyn
- Periods from SAP2000 (with shear deformation):
    Mode 1: 0.542639 s (Translation X)
    Mode 2: 0.542639 s (Translation Y)
    Mode 3: 0.491825 s (Torsion Z)
    Mode 4: 0.178885 s (Higher Translation X)
    Mode 5: 0.178885 s (Higher Translation Y)
    Mode 6: 0.165114 s (Higher Torsion Z)
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

# Section properties using exact rectangular elasticity theory (matching SAP2000):
# Includes Timoshenko shear areas As2 = As3 = 5/6 * A
# Beams: 20 in x 35 in (A = 700 in^2, As2 = As3 = 583.33 in^2, J = 60,031.9 in^4)
beam_section = Section.from_rectangle(width=20.0, depth=35.0)

# Columns: 20 in x 20 in (A = 400 in^2, As2 = As3 = 333.33 in^2, J = 22,533.3 in^4)
col_section = Section.from_rectangle(width=20.0, depth=20.0)

# -------------------------------------------------------------------
# 2. Build 3D structure using SimpleFrame with shear deformation enabled
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

# Columns (include_shear_deformation=True)
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
            A=col_section,
            include_shear_deformation=True,
        )
        elem_id += 1

# Beams (perimeter at each floor level, include_shear_deformation=True)
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
            E=E, G=G,
            A=beam_section,
            include_shear_deformation=True,
        )
        elem_id += 1

# -------------------------------------------------------------------
# 3. Lumped floor masses
# -------------------------------------------------------------------
# Each floor carries 450 kips, distributed equally among 4 corner nodes
# Rotational inertia is zero (massless rotational DOFs condensed via exact Guyan reduction)
mass_per_node = (450.0 / 4.0) / g  # kips*s^2/in

for story in range(1, n_stories + 1):
    story_base = story * 4 + 1
    for k in range(4):
        nid = story_base + k
        frame.structure.nodes[nid].set_mass(
            mass=[mass_per_node, mass_per_node, mass_per_node],
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

# Benchmark SAP2000 periods with shear deformation
sap2000_T = [0.542639, 0.542639, 0.491825, 0.178885, 0.178885, 0.165114]

print("=" * 78)
print("3D SPACE FRAME MODAL ANALYSIS WITH TIMOSHENKO SHEAR DEFORMATION")
print("=" * 78)
print(f"Total DOFs = {frame.structure.neq}, Free DOFs = {len(frame.structure.free_dofs)}")
print("\nFirst 6 Modes Comparison with SAP2000:")
print(f"{'Mode':<6} {'structdyn (s)':<16} {'SAP2000 (s)':<16} {'Diff (%)':<12} {'Freq (Hz)':<14} {'Omega (rad/s)':<14}")
print("-" * 78)
for i in range(min(6, len(T))):
    diff_pct = (T[i] - sap2000_T[i]) / sap2000_T[i] * 100.0 if i < len(sap2000_T) else 0.0
    sap_str = f"{sap2000_T[i]:.6f}" if i < len(sap2000_T) else "N/A"
    print(f"{i+1:<6} {T[i]:<16.6f} {sap_str:<16} {diff_pct:<+12.4f} {frequencies[i]:<14.5f} {omega[i]:<14.5f}")
print("=" * 78)

# -------------------------------------------------------------------
# 6. Register modal results on structure for 3D visualization
# -------------------------------------------------------------------
frame.set_modal_results(omega, phi)
