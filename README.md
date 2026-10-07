# fem3d

**A 3D Finite Element Analysis library for general space frames and trusses in Python, based on the Direct Stiffness Method.**

`fem3d` provides a full implementation of the 3D elastic beam-column frame element (12 DOF), space truss, and spring elements. It supports linear static analysis, modal analysis, and linear elastic buckling analysis. It mirrors the architecture of the `fem2d` package, extended to 3D.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Coordinate System & Sign Convention](#coordinate-system--sign-convention)
4. [Degrees of Freedom Convention](#degrees-of-freedom-convention)
5. [Local Axes & Orientation](#local-axes--orientation)
6. [Element Formulations](#element-formulations)
7. [Package Structure](#package-structure)
8. [Core API Reference](#core-api-reference)
9. [High-Level API: `SimpleFrame`](#high-level-api-simpleframe)
10. [Section Library](#section-library)
11. [Load Types](#load-types)
12. [Post-Processing: `Results`](#post-processing-results)
13. [Modal Analysis](#modal-analysis)
14. [Buckling Analysis](#buckling-analysis)
15. [struct_core Adapter](#struct_core-adapter)
16. [Quick Examples](#quick-examples)
17. [Textbook References](#textbook-references)

---

## Features

- **6 DOFs per node** — Translation $(u_x, u_y, u_z)$ and rotation $(\theta_x, \theta_y, \theta_z)$ in global coordinates.
- **3D Frame / Beam-Column Element (12 DOF)** — Euler-Bernoulli formulation with axial, St. Venant torsion, and biaxial bending $(EI_y, EI_z)$. Optional Timoshenko shear deformation.
- **3D Space Truss Element** — Pin-jointed bar members. Axial stiffness only.
- **3D Spring Element** — Translational and rotational springs along 3D axes.
- **Arbitrary 3D Member Orientation** — Automatic local axes computation from member geometry. Supports user-specified roll angle $\beta$ (degrees) or web reference vector.
- **End Releases** — Optional moment/torsion/shear releases at either end of a frame element (e.g. pinned connections).
- **Nodal & Element Loads** — Concentrated 3D nodal loads; uniform and partial distributed loads on elements with exact fixed-end force/moment calculation; element point loads.
- **Modal Analysis** — Generalized eigenvalue solver for natural frequencies and mode shapes.
- **Linear Elastic Buckling Analysis** — Geometric stiffness matrix $K_g$ with eigenvalue solver.
- **Post-Processing** — pandas DataFrames for displacements, reactions, and internal forces. Textbook-style text report generator.
- **High-Level API (`SimpleFrame`)** — Fluent model construction API.
- **`struct_core` Integration** — Bidirectional conversion: `model_from_core`, `model_to_core`, `result_to_core`.

---

## Installation

```bash
# Inside the fem3d directory, install in editable mode using the structeng conda environment:
conda activate structeng
pip install -e .
```

---

## Coordinate System & Sign Convention

### Global Coordinate System

`fem3d` uses a standard **right-handed Cartesian** global coordinate system $(X, Y, Z)$ satisfying $\vec{X} \times \vec{Y} = \vec{Z}$:

```text
         Z (vertical / upward)
         ^
         |    Y (horizontal / transverse / depth)
         |   /
         |  /
         | /
         +----------------> X (horizontal / longitudinal / span)
```

- **X (Global X)** — horizontal axis, positive to the right (primary frame span or length).
- **Y (Global Y)** — horizontal axis, positive along the transverse / depth direction (into the plane).
- **Z (Global Z)** — vertical axis, positive upward (column height / elevation).

#### Standard Orthographic Views

- **Plan View** (looking down from $+Z$ toward ground):
  - **X** points to the right (East / longitudinal span)
  - **Y** points upward on the page (North / transverse depth)
  - **Z** points out of the page toward the viewer (Up)

  ```text
       Y (transverse / North)
       ^
       |
       |
       +--------------> X (longitudinal / East)
      (.) Z (out of page / Up)
  ```

- **Elevation View** (looking along $+Y$ toward the structural face):
  - **X** points to the right (span)
  - **Z** points upward (height)
  - **Y** points into the page (away from viewer)

  ```text
       Z (height / elevation)
       ^
       |
       |
       +--------------> X (longitudinal / span)
       x  Y (into page / depth)
  ```

> **Note on Vertical Axis:** Global **Z** is treated as the vertical upward axis by default. All automatic local orientation routines treat $+Z$ as the vertical reference direction, and gravity load acts in the $-Z$ direction.


### Forces and Moments

All forces and moments in global coordinates follow the right-hand rule:

| Symbol | Meaning |
|--------|---------|
| $F_x, F_y, F_z$ | Forces along global X, Y, Z |
| $M_x, M_y, M_z$ | Moments about global X, Y, Z |

Positive moments: right-hand rule about the positive axis direction.

### Support Reactions

Reported reactions are the **forces and moments that the structure exerts on the support** (equal and opposite to what the support applies to the structure). A downward load on a pinned base produces a positive $F_z$ reaction.

---

## Degrees of Freedom Convention

Each node has **6 degrees of freedom** in global coordinates:

```
DOF index:   0     1     2     3      4      5
             ux    uy    uz    rx     ry     rz
             (Tx)  (Ty)  (Tz)  (Rx)   (Ry)   (Rz)
```

| Index | DOF | Meaning |
|-------|-----|---------|
| 0 | $u_x$ | Translation along global X |
| 1 | $u_y$ | Translation along global Y |
| 2 | $u_z$ | Translation along global Z |
| 3 | $\theta_x$ | Rotation about global X (rx) |
| 4 | $\theta_y$ | Rotation about global Y (ry) |
| 5 | $\theta_z$ | Rotation about global Z (rz) |

A **12-DOF element** connects node $i$ to node $j$:

```
u_element = [ux_i, uy_i, uz_i, rx_i, ry_i, rz_i,
             ux_j, uy_j, uz_j, rx_j, ry_j, rz_j]
  index:      0     1     2     3     4     5
              6     7     8     9    10    11
```

Support fixity list format: `[ux, uy, uz, rx, ry, rz]` where `1` (or `True`) = fixed, `0` = free.

---

## Local Axes & Orientation

### Local Axes Convention (1, 2, 3)

For each element, the local coordinate triad $(\vec{v}_1, \vec{v}_2, \vec{v}_3)$ is a right-handed orthogonal system:

- **$\vec{v}_1$ (Local Axis 1 — Longitudinal / Axial):** directed along the member centroidal axis from node $i$ to node $j$:
  $$\vec{v}_1 = \frac{1}{L}(x_j - x_i,\; y_j - y_i,\; z_j - z_i)^T$$

- **$\vec{v}_2$ (Local Axis 2 — Transverse / Web / Depth):** cross-section axis in the local triad. Associated with bending in the local 1-2 plane (governed by $I_3$, deflection $u_2$, rotation $\theta_3$).

- **$\vec{v}_3$ (Local Axis 3 — Transverse / Flange / Width):** cross-section axis such that $\vec{v}_3 = \vec{v}_1 \times \vec{v}_2$. Associated with bending in the local 1-3 plane (governed by $I_2$, deflection $u_3$, rotation $\theta_2$).

### Default Automatic Orientation

For **non-vertical members** (member axis not parallel to global Z):
$$\vec{v}_3 = \frac{\vec{v}_1 \times [0, 0, 1]}{|\vec{v}_1 \times [0, 0, 1]|}, \quad \vec{v}_2 = \vec{v}_3 \times \vec{v}_1$$

This places $\vec{v}_3$ in the horizontal plane and $\vec{v}_2$ as close to vertical as possible (oriented with the web vertical).

For **vertical members** (member axis parallel to global Z):
- Member pointing +Z: $\vec{v}_2 = [0, 1, 0]$, $\vec{v}_3 = [-1, 0, 0]$
- Member pointing −Z: $\vec{v}_2 = [0, -1, 0]$, $\vec{v}_3 = [-1, 0, 0]$

### Horizontal Member Example

For a beam along **global X** (node i at origin, node j along +X):

| Local axis | Global direction |
|-----------|-----------------|
| $\vec{v}_1$ (axial) | $+X$ (along member) |
| $\vec{v}_2$ (web / depth) | $+Z$ (upward vertical) |
| $\vec{v}_3$ (flange / width) | $-Y$ |

This means:
- A **global Y load** at a node maps to local 3 → bending about axis 2 uses $E I_2$
- A **global Z load** at a node maps to local 2 → bending about axis 3 uses $E I_3$

### User-Specified Orientation

Override the automatic orientation with:

- **Roll angle** `roll_angle` (degrees): rotates the local 2-3 plane about $\vec{v}_1$.
- **Web vector** `web_vector` (3D array): explicit reference vector pointing in the local 1-2 plane (direction of axis 2).

### Coordinate Transformation

The $3 \times 3$ direction cosine matrix $R$:
$$R = \begin{bmatrix} \vec{v}_1^T \\ \vec{v}_2^T \\ \vec{v}_3^T \end{bmatrix}$$

The $12 \times 12$ global-to-local transformation matrix:
$$T = \text{diag}(R, R, R, R)$$

Stiffness transformation:
$$K_{global} = T^T \; K_{local} \; T$$

*Reference: Weaver & Gere (1990), Section 4.10; McGuire et al. (2000), Section 5.1.*

---

## Element Formulations

### 3D Frame Element (`FrameElement`)

The $12 \times 12$ local stiffness matrix has four uncoupled blocks:

#### 1. Axial (DOFs 0, 6 — $u_{1i}, u_{1j}$)

$$K_{axial} = \frac{EA}{L} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$$

#### 2. Torsion (DOFs 3, 9 — $\theta_{1i}, \theta_{1j}$)

$$K_{torsion} = \frac{GJ}{L} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$$

where $G = E / [2(1+\nu)]$.

#### 3. Bending in 1-2 plane about local axis 3 — $EI_3$ block (DOFs 1, 5, 7, 11 — $u_{2i}, \theta_{3i}, u_{2j}, \theta_{3j}$)

Sign convention: slope $du_2/dx_1 = +\theta_3$ (right-hand rule about local axis 3).

$$K_{3} = \begin{bmatrix}
\frac{12EI_3}{L^3} & \frac{6EI_3}{L^2} & -\frac{12EI_3}{L^3} & \frac{6EI_3}{L^2} \\
\frac{6EI_3}{L^2} & \frac{4EI_3}{L} & -\frac{6EI_3}{L^2} & \frac{2EI_3}{L} \\
-\frac{12EI_3}{L^3} & -\frac{6EI_3}{L^2} & \frac{12EI_3}{L^3} & -\frac{6EI_3}{L^2} \\
\frac{6EI_3}{L^2} & \frac{2EI_3}{L} & -\frac{6EI_3}{L^2} & \frac{4EI_3}{L}
\end{bmatrix}$$

#### 4. Bending in 1-3 plane about local axis 2 — $EI_2$ block (DOFs 2, 4, 8, 10 — $u_{3i}, \theta_{2i}, u_{3j}, \theta_{2j}$)

Sign convention: slope $du_3/dx_1 = -\theta_2$ (right-hand rule about local axis 2; **note the sign flip** relative to the axis 3 bending block).

$$K_{2} = \begin{bmatrix}
\frac{12EI_2}{L^3} & -\frac{6EI_2}{L^2} & -\frac{12EI_2}{L^3} & -\frac{6EI_2}{L^2} \\
-\frac{6EI_2}{L^2} & \frac{4EI_2}{L} & \frac{6EI_2}{L^2} & \frac{2EI_2}{L} \\
-\frac{12EI_2}{L^3} & \frac{6EI_2}{L^2} & \frac{12EI_2}{L^3} & \frac{6EI_2}{L^2} \\
-\frac{6EI_2}{L^2} & \frac{2EI_2}{L} & \frac{6EI_2}{L^2} & \frac{4EI_2}{L}
\end{bmatrix}$$

*References: Przemieniecki (1968), Eq. 5.58; Weaver & Gere (1990), Sec. 4.11; McGuire et al. (2000), Sec. 5.1.*

> **⚠️ Verify:** These stiffness blocks should be cross-checked with your textbook, especially the sign convention for the $K_z$ block (the $du_z/dx = -\theta_y$ convention varies between references).

#### Internal Forces

Local internal forces recovered at each end: $[P_i, V_{y,i}, V_{z,i}, T_i, M_{y,i}, M_{z,i}, P_j, V_{y,j}, V_{z,j}, T_j, M_{y,j}, M_{z,j}]$.

### 3D Truss Element (`TrussElement`)

Pure axial stiffness only. Local $2 \times 2$ system expanded to $12 \times 12$ via transformation. Rotation DOFs are automatically constrained at truss-only nodes (see Auto-Stabilization below).

### 3D Spring Element (`SpringElement`)

Diagonal stiffness in global coordinates: $(k_x, k_y, k_z, k_{rx}, k_{ry}, k_{rz})$. No transformation needed.

### Auto-Stabilization of Truss-Only Nodes

At nodes connected only to truss or spring elements (no frame elements), there is zero rotational stiffness and potentially zero translational stiffness in transverse directions. `fem3d` automatically detects these and constrains the zero-stiffness DOFs to prevent singular $K_{ff}$, without modifying user-specified supports.

---

## Package Structure

```
fem3d/
├── pyproject.toml
├── README.md
├── LICENSE
├── fem3d/
│   ├── __init__.py            # Public API exports
│   ├── nodes.py               # Node class (6 DOF)
│   ├── materials/
│   │   ├── material.py        # Abstract MaterialBase
│   │   └── elastic.py         # ElasticMaterial (E, nu, G, rho)
│   ├── sections/
│   │   └── section.py         # Section (A, I2, I3, J) with factory methods
│   ├── elements/
│   │   ├── element.py         # ElementBase (geometry, local axes v1, v2, v3, T matrix)
│   │   ├── frame.py           # FrameElement (12x12 K, M, Kg with I2, I3)
│   │   ├── truss.py           # TrussElement (axial only)
│   │   └── spring.py          # SpringElement (diagonal 12x12)
│   ├── loads.py               # PointLoad, DistributedLoad (w1, w2, w3), ElementPointLoad
│   ├── structure.py           # Structure: assembly, BCs, linear solver, modal
│   ├── results.py             # Results: DataFrames + text report
│   ├── buckling_analysis.py   # Linear elastic buckling
│   ├── utils/
│   │   └── simple_frame.py    # SimpleFrame / SimpleFrame3D high-level API
│   └── adapters/
│       └── struct_core_adapter.py  # struct_core bidirectional adapter
├── examples/
│   └── linear/
│       ├── frame_58.py
│       └── frame_84.py
└── tests/
    ├── conftest.py
    ├── test_frame_3d.py
    ├── test_truss_3d.py
    ├── test_spring_3d.py
    ├── test_modal_buckling.py
    ├── test_results.py
    └── test_struct_core_adapter.py
```

---

## Core API Reference

### `Node`

```python
from fem3d import Node

n = Node(nid=1, x=0.0, y=0.0, z=0.0)

# Set support (6-element list: [ux, uy, uz, rx, ry, rz])
n.set_support([1, 1, 1, 1, 1, 1])     # Fully fixed
n.set_support([1, 1, 1, 0, 0, 0])     # Pinned (pin base)
n.set_support(ux_fixed=True, uy_fixed=True, uz_fixed=True)  # keyword form

# Set nodal load [Fx, Fy, Fz, Mx, My, Mz]
n.set_load([0.0, -10.0, 0.0, 0.0, 0.0, 0.0])

# Set lumped mass
n.set_mass(mass=1500.0, inertia=0.0)   # scalar (applied to all 3 translations)
n.set_mass(mass=[m1, m2, m3], inertia=[I1, I2, I3])  # per-component
```

### `ElasticMaterial`

```python
from fem3d import ElasticMaterial

steel = ElasticMaterial(E=29000.0, nu=0.3, rho=0.000284)
# G is computed automatically as G = E / (2*(1+nu))
print(steel.G)  # 11153.8 ksi
```

### `Section`

```python
from fem3d import Section

sec = Section(A=10.0, I2=50.0, I3=200.0, J=30.0)

# Factory methods:
sec = Section.from_rectangle(width=4.0, depth=8.0)
sec = Section.from_circle(diameter=6.0)
sec = Section.from_pipe(outer_d=8.0, thickness=0.5)
sec = Section.from_tube(width=6.0, depth=8.0, thickness=0.375)
sec = Section.from_i_shape(bf=6.0, tf=0.5, d=10.0, tw=0.3)
```

### `Structure`

```python
from fem3d import Structure, Node, ElasticMaterial, Section, FrameElement, PointLoad

s = Structure()

# Add nodes
n1 = Node(1, 0, 0, 0)
n2 = Node(2, 0, 0, 120)   # vertical column, 120 in. tall
s.add_node(n1)
s.add_node(n2)

# Set support
n1.set_support([1, 1, 1, 1, 1, 1])   # fixed base

# Add element
mat = ElasticMaterial(E=29000.0)
sec = Section(A=14.7, I2=272.0, I3=984.0, J=3.01)
el = FrameElement(1, n1, n2, mat, sec)
s.add_element(el)

# Add load
s.add_load(PointLoad(n2, fx=10.0, fy=5.0))

# Solve
disp, reactions = s.solve()

# Modal analysis
freqs, modes = s.modal_analysis(num_modes=3)
```

---

## High-Level API: `SimpleFrame`

`SimpleFrame` (alias: `SimpleFrame3D`) wraps `Structure` for rapid model building:

```python
from fem3d import SimpleFrame, Results

frame = SimpleFrame()

# Nodes: (id, x, y, z)
frame.add_node(1, 0.0, 0.0, 0.0)
frame.add_node(2, 0.0, 0.0, 120.0)
frame.add_node(3, 120.0, 0.0, 120.0)

# Frame elements: (id, node_i, node_j, E, A, I2, I3, J)
frame.add_frame(1, 1, 2, E=29000, A=14.7, I2=272, I3=984, J=3.01)
frame.add_frame(2, 2, 3, E=29000, A=10.0, I2=100, I3=300, J=2.0, roll_angle=0.0)

# Truss elements: (id, node_i, node_j, E, A)
frame.add_truss(3, 1, 3, E=29000, A=5.0)

# Spring elements: (id, node_i, node_j, kx, ky, kz, krx, kry, krz)
frame.add_spring(4, 1, 2, kx=1000.0, ky=0.0, kz=500.0)

# Supports: (node_id, [ux, uy, uz, rx, ry, rz])
frame.add_support(1, [1, 1, 1, 1, 1, 1])   # fixed

# Nodal loads: (node_id, [Fx, Fy, Fz, Mx, My, Mz])
frame.add_node_load(3, [10.0, 0.0, -5.0, 0.0, 0.0, 0.0])

# Distributed loads: (elem_id, w1=0, w2=0, w3=0)
# w1, w2, w3 are in local element coordinates
frame.add_distributed_load(2, w3=-0.5)   # uniform load along local 3

# Solve
disp, reactions = frame.solve()

# Post-processing
results = Results(frame)
print(results.node_displacements())
print(results.reactions())
print(results.element_forces())
results.create_report()
```

---

## Section Library

`Section` provides factory class methods for common cross-sections. All formulas are standard elastic section properties.

| Factory method | Shape | Parameters |
|---------------|-------|------------|
| `Section(A, I2, I3, J)` | Generic | Explicit values |
| `Section.from_rectangle(width, depth)` | Solid rectangle | $b \times h$ |
| `Section.from_circle(diameter)` | Solid circle | $d$ |
| `Section.from_pipe(outer_d, thickness)` | Hollow circle | $D_o$, $t$ |
| `Section.from_tube(width, depth, thickness)` | Hollow rectangle (HSS) | $b$, $h$, $t$ |
| `Section.from_i_shape(bf, tf, d, tw)` | I-beam / W-shape | flange & web dims |

**Moments of Inertia Convention:**
- $I_3$ = second moment of area about the local **3-axis** ($I_3 = \int x_2^2 dA$), resisting flexure in the local 1-2 plane (depth / major axis in standard orientation).
- $I_2$ = second moment of area about the local **2-axis** ($I_2 = \int x_3^2 dA$), resisting flexure in the local 1-3 plane (width / minor axis in standard orientation).

---

## Load Types

### `PointLoad`

Concentrated force and moment at a node:

```python
from fem3d import PointLoad
load = PointLoad(node, fx=0.0, fy=0.0, fz=-10.0, mx=0.0, my=0.0, mz=0.0)
structure.add_load(load)
```

### `DistributedLoad`

Uniformly distributed load along an element **in local element coordinates** ($w_1, w_2, w_3$) or global coordinates ($w_X, w_Y, w_Z$):

```python
from fem3d import DistributedLoad
load = DistributedLoad(element, w1=0.0, w2=0.0, w3=-0.5)
structure.add_load(load)
```

Fixed-end equivalent nodal forces (exact for uniform load, Euler-Bernoulli):

| Direction | Shear reactions | End moments |
|-----------|----------------|-------------|
| $w_2$ (local 2) | $\pm w_2 L / 2$ | $\pm w_2 L^2 / 12$ about local axis 3 |
| $w_3$ (local 3) | $\pm w_3 L / 2$ | $\mp w_3 L^2 / 12$ about local axis 2 |
| $w_1$ (local 1) | $\pm w_1 L / 2$ | none |

Loads are transformed to global DOFs via the element's $T$ matrix.

### `ElementPointLoad`

Concentrated load at a point along an element at local position $x$ (in element length units):

```python
from fem3d import ElementPointLoad
load = ElementPointLoad(element, p1=0.0, p2=-5.0, p3=0.0, m1=0.0, m2=0.0, m3=0.0, x=60.0)
```

---

## Post-Processing: `Results`

```python
results = Results(frame)   # or Results(structure)

# Node displacements — pandas DataFrame
df_disp = results.node_displacements()
# Columns: ['node', 'ux', 'uy', 'uz', 'rx', 'ry', 'rz']

# Support reactions — pandas DataFrame
df_reac = results.reactions()
# Columns: ['node', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']

# Element internal forces — pandas DataFrame (local coordinates)
df_forces = results.element_forces()
# Primary local force columns:
#   ['f1_i', 'f2_i', 'f3_i', 'm1_i', 'm2_i', 'm3_i',
#    'f1_j', 'f2_j', 'f3_j', 'm1_j', 'm2_j', 'm3_j']
# Engineering nomenclature columns:
#   ['P_i', 'V2_i', 'V3_i', 'T_i', 'M2_i', 'M3_i',
#    'P_j', 'V2_j', 'V3_j', 'T_j', 'M2_j', 'M3_j']
# where i = start node end, j = end node end

# Full formatted text report
results.create_report(print_report=True)
```

**Element forces sign convention (local coordinates):**

| Symbol | Primary Column | Meaning (positive) |
|--------|---------------|--------------------|
| $P$ | `f1_i` / `P_i` | Axial tension at start end |
| $V_2$ | `f2_i` / `V2_i` | Shear along local 2 at start end |
| $V_3$ | `f3_i` / `V3_i` | Shear along local 3 at start end |
| $T$ | `m1_i` / `T_i` | Torsional moment about local 1 at start end |
| $M_2$ | `m2_i` / `M2_i` | Bending moment about local 2 at start end |
| $M_3$ | `m3_i` / `M3_i` | Bending moment about local 3 at start end |

---

## Modal Analysis

```python
structure.solve()   # must solve statics first (builds K, F)

freqs_hz, modes = structure.modal_analysis(num_modes=6)
# freqs_hz: natural frequencies in Hz (sorted ascending)
# modes: shape (neq, num_modes) eigenvectors (mass-normalized)
```

The solver uses the **consistent mass matrix** by default. It assembles the reduced free-DOF mass matrix $M_{ff}$ and solves:
$$K_{ff} \phi = \omega^2 M_{ff} \phi$$

Natural frequency: $f_n = \omega_n / (2\pi)$ Hz.

---

## Buckling Analysis

```python
from fem3d import buckling_analysis

# Must solve statics first to obtain element axial forces
disp, reac = frame.solve()

lambdas, modes = buckling_analysis(frame, num_modes=3)
# lambdas: critical load multipliers (sorted ascending by abs value)
# modes: shape (neq, num_modes) buckling mode shapes
```

Solves the generalized eigenvalue problem:
$$K_{ff} \phi = \lambda \, K_{g,ff} \phi$$

The critical load = $\lambda_{cr} \times$ applied load. The geometric stiffness matrix $K_g$ is assembled from element axial forces recovered from the linear static solution.

---

## struct_core Adapter

Bidirectional conversion with `struct_core` models and results:

```python
from fem3d import model_from_core, model_to_core, result_to_core
import struct_core as sc

# fem3d -> struct_core
sc_model = model_to_core(frame)          # SimpleFrame or Structure
sc_result = result_to_core(results)      # Results object or structure

# struct_core -> fem3d
structure = model_from_core(sc_model)    # StructuralModel or Project

# Support mapping (struct_core.Support now supports full 6 DOF):
#   fem3d [ux, uy, uz, rx, ry, rz] <-> struct_core Support {ux, uy, uz, rx, ry, rz}

# AnalysisResult mapping:
#   node_results: Dict[NodeId, NodeResult] with NodeDisplacement (6 DOF) + NodeReaction (6 DOF)
#   element_results: Dict[ElementId, ElementResult] with BeamSectionForce at i and j ends
```

---

## Quick Examples

### 3D Cantilever — Axial, Torsion, and Biaxial Bending

```python
from fem3d import SimpleFrame, Results

frame = SimpleFrame()
frame.add_node(1, 0, 0, 0)
frame.add_node(2, 120, 0, 0)   # 10-ft beam along global X

frame.add_frame(1, 1, 2, E=29000, A=14.7, I2=272, I3=984, J=15.0)
frame.add_support(1, [1, 1, 1, 1, 1, 1])   # fixed base

# Combined loads at tip
frame.add_node_load(2, [5.0, 3.0, -2.0, 1.0, 0.0, 0.0])
frame.solve()

results = Results(frame)
results.create_report()
```

### Space Truss

```python
from fem3d import SimpleFrame, Results

frame = SimpleFrame()
frame.add_node(1,   0,   0, 60)   # apex
frame.add_node(2,  60,  60,  0)
frame.add_node(3, -60,  60,  0)
frame.add_node(4,   0, -60,  0)

for nid in [2, 3, 4]:
    frame.add_support(nid, [1, 1, 1, 0, 0, 0])   # pinned base

frame.add_truss(1, 1, 2, E=29000, A=2.0)
frame.add_truss(2, 1, 3, E=29000, A=2.0)
frame.add_truss(3, 1, 4, E=29000, A=2.0)

frame.add_node_load(1, [0, 0, -50, 0, 0, 0])
disp, reac = frame.solve()
```

### Modal Analysis of a Cantilever

```python
from fem3d import SimpleFrame, Results

frame = SimpleFrame()
for i in range(5):
    frame.add_node(i+1, 0, 0, i*24.0)

frame.add_support(1, [1, 1, 1, 1, 1, 1])

for i in range(4):
    frame.add_frame(i+1, i+1, i+2, E=29000, A=14.7, I2=272, I3=984, J=3.0)
    # Set lumped mass at each node
    frame.structure.nodes[i+2].set_mass(mass=0.005)

frame.solve()
freqs, modes = frame.structure.modal_analysis(num_modes=4)
print("Natural frequencies (Hz):", freqs)
```

### 3D Structure Visualization (`DrawStructure`)

`fem3d` includes a publication-quality 3D visualization module powered by Matplotlib 3D:

```python
from fem3d import SimpleFrame, DrawStructure, draw_structure

frame = SimpleFrame()
# ... build geometry and solve ...
frame.solve()

# 1. Direct interactive 3D plot
frame.draw(
    show_undeformed=True,
    show_deformed=True,
    show_loads=True,
    show_supports=True,
    color_by_force=True,     # Tension in blue, compression in red
    title="3D Space Structure",
    save_path="structure_3d.png",
)

# 2. Vibration mode shape plot
frame.plot_mode_shape(mode=1, title="Fundamental Vibration Mode")

# 3. Linear elastic buckling mode plot
frame.plot_buckling_mode(mode=1, title="First Buckling Mode")
```

**Key Visualization Features:**
- **Realistic 3D Supports:** Fixed anchor pads with ground stubs, 3D pinned pyramids, and roller bearings.
- **Double-Headed Moment Arrows:** 3D moments drawn with double arrowheads along the moment vector axis.
- **Cubic Hermite Splines:** 3D deformed beam curves showing real member bending curvature in space.
- **Axial Force Colormap:** Elements color-coded with a synchronized colorbar (blue = tension, red = compression).


---

## Textbook References

The stiffness matrices, mass matrices, and transformation procedures in `fem3d` follow these standard references. Code comments cite specific equations and pages.

| Reference | Used for |
|-----------|----------|
| **Przemieniecki, J. S.** (1968). *Theory of Matrix Structural Analysis*. McGraw-Hill. Sec. 5.6, Eq. 5.58. | Local stiffness matrix, consistent mass |
| **Weaver, W. & Gere, J. M.** (1990). *Matrix Analysis of Framed Structures* (3rd ed.). Van Nostrand Reinhold. Sec. 4.10–4.11. | Local axes, transformation matrix, sign convention |
| **McGuire, W., Gallagher, R. H., & Ziemian, R. D.** (2000). *Matrix Structural Analysis* (2nd ed.). Wiley. Sec. 5.1. | 3D beam stiffness, end releases |
| **Cook, R. D., Malkus, D. S., Plesha, M. E., & Witt, R. J.** (2002). *Concepts and Applications of Finite Element Analysis* (4th ed.). Wiley. | Consistent mass, geometric stiffness |
| **Bathe, K. J.** (1996). *Finite Element Procedures*. Prentice Hall. | Eigenvalue solvers, buckling |

> **Note:** Items marked `# TODO: verify` in the source code or test files have computed values that need manual cross-check against your textbook, particularly for biaxial bending sign conventions (the $\theta_y$ sign) and moment reaction directions.

---

## Ecosystem Integration with `struct_core`

`fem3d` provides bidirectional adapters to `struct_core` for 3D frames:

```python
from fem3d.adapters import model_from_core, model_to_core, result_to_core

# 1. Build a 3D fem3d.Structure from a struct_core.StructuralModel or Project
# Automatically computes member self-weight line loads via struct_load for cases
# with include_self_weight=True (e.g. DEAD). Supports load cases or combinations.
structure = model_from_core(
    project,
    load_case_id="DEAD",
    load_combination_id="COMB_1.2D+1.6L",
    include_self_weight=True,
)
structure.solve()

# 2. Export 3D analysis results to struct_core.AnalysisResult
analysis_result = result_to_core(structure, analysis_case_id="Static_3D")
```

---

## Running Tests

```bash
conda activate structeng
cd fem3d
pytest -v
```

All 28 tests pass cleanly.

