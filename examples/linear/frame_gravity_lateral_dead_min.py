"""
3D RC Space Frame: Gravity, Lateral, and Direct Element Self-Weight (Dead Load).
Minimal replica of frame_gravity_lateral_dead.py applying self-weight directly to fem3d.Structure.

Units: kN, m, kPa (kN/m^2)
"""

from fem3d import Structure, Node, ElasticMaterial, Section, FrameElement, PointLoad, DistributedLoad, Results
from struct_load import apply_self_weight

# 1. Material & Sections
mat = ElasticMaterial(E=30.0e6, nu=0.20, unit_weight=24.0)  # C30 Concrete
col_sec = Section.from_rectangle(width=0.30, depth=0.40)      # Column: 0.3m x 0.4m
beam_sec = Section.from_rectangle(width=0.25, depth=0.40)     # Beam: 0.25m x 0.40m

# 2. Structure & Grid Setup
struct = Structure(units="kN, m")
x_coords = [0.0, 4.0, 10.0]  # 2 bays: 4m, 6m
y_coords = [0.0, 5.0]        # 1 bay: 5m
z_coords = [0.0, 3.5, 7.0]   # 2 stories: 3.5m each

node_grid = {}
nid = 1
for z in z_coords:
    for y in y_coords:
        for x in x_coords:
            node = Node(nid, x, y, z)
            if z == 0.0:
                node.set_support([1, 1, 1, 1, 1, 1])  # Fixed base
            struct.add_node(node)
            node_grid[(x, y, z)] = node
            nid += 1

# 3. Columns (12 elements)
eid = 1
for story in range(2):
    for y in y_coords:
        for x in x_coords:
            col = FrameElement(
                eid=eid,
                node_i=node_grid[(x, y, z_coords[story])],
                node_j=node_grid[(x, y, z_coords[story + 1])],
                material=mat,
                section=col_sec,
                include_shear_deformation=True,
            )
            struct.add_element(col)
            eid += 1

# 4. Beams (14 elements)
beams = []
for story in (1, 2):
    z = z_coords[story]
    # X-direction beams
    for y in y_coords:
        for bay in (0, 1):
            bm = FrameElement(
                eid=eid,
                node_i=node_grid[(x_coords[bay], y, z)],
                node_j=node_grid[(x_coords[bay + 1], y, z)],
                material=mat,
                section=beam_sec,
                include_shear_deformation=True,
            )
            struct.add_element(bm)
            beams.append(bm)
            eid += 1
    # Y-direction beams
    for x in x_coords:
        bm = FrameElement(
            eid=eid,
            node_i=node_grid[(x, y_coords[0], z)],
            node_j=node_grid[(x, y_coords[1], z)],
            material=mat,
            section=beam_sec,
            include_shear_deformation=True,
        )
        struct.add_element(bm)
        beams.append(bm)
        eid += 1

# 5. Superimposed Gravity Loads (10 kN/m downward on all 14 beams)
for bm in beams:
    struct.add_load(DistributedLoad(bm, wz=-10.0, coord_system="global"))

# 6. Lateral Loads (+X on x=0 face: 40 kN at story 1, 20 kN at story 2)
for y in y_coords:
    struct.add_load(PointLoad(node_grid[(0.0, y, 3.5)], fx=40.0))
    struct.add_load(PointLoad(node_grid[(0.0, y, 7.0)], fx=20.0))

# 7. Apply Self-Weight Directly to fem3d Structure
# (Can be called as struct.apply_self_weight() or apply_self_weight(struct))
struct.apply_self_weight()

# 8. Solve
disp, reactions = struct.solve()
results = Results(struct)

# 9. Summary & Validation Output
df_rx = results.reactions()
sum_rx = float(df_rx["Fx"].sum())
sum_rz = float(df_rx["Fz"].sum())

print("================================================================================")
print("3D RC SPACE FRAME: DIRECT SELF-WEIGHT (MINIMAL REPLICA)")
print("================================================================================")
print(f"Total Nodes:     {len(struct.nodes)}, Total Elements: {len(struct.elements)}")
print(f"Total Applied:   Fx = +120.00 kN, Fz = -988.96 kN (700 kN Gravity + 288.96 kN SW)")
print(f"Total Reactions: Rx = {sum_rx:8.2f} kN, Rz = {sum_rz:8.2f} kN")
print(f"Equilibrium:     dFx = {abs(sum_rx + 120.0):.2e} kN, dFz = {abs(sum_rz - 988.96):.2e} kN\n")

print("Key Displacements (Timoshenko with shear deformation):")
for nid, label in [(13, "Story 2 (x=0, y=0)"), (7, "Story 1 (x=0, y=0)")]:
    n = struct.nodes[nid]
    ux = struct.disp[n.dofs[0]] * 1e3
    uz = struct.disp[n.dofs[2]] * 1e3
    print(f"  Node {nid:2d} [{label:<20}]: ux = {ux:6.4f} mm, uz = {uz:7.4f} mm")

print("\nVerification:")
# Timoshenko analysis with concrete nu=0.20 and direct self-weight:
# Node 13 ux = 5.9713 mm, Node 7 ux = 3.7687 mm, Base Rz = 988.96 kN
n13 = struct.nodes[13]
assert abs(struct.disp[n13.dofs[0]] * 1e3 - 5.9713) < 1e-3, "Node 13 ux does not match expected"
assert abs(sum_rz - 988.96) < 1e-2, "Base Rz does not match expected"
print("PASS: Results match reference analysis identically.")
print("================================================================================")

