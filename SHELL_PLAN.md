# Shell elements in fem3d: implementation plan

Plan for ROADMAP.md §4 (shell elements for slabs and walls), approved 17 November 2026. Sources, local copies, and
verification problems: `references/notes/shell_elements_reading_list.md` (root folder, not tracked).

Status marks: `[ ]` planned, `[~]` in progress, `[x]` done (with date). Each milestone ends with its tests passing.
Changes are recorded in the `CHANGELOG.md` of each package touched.

## Decisions

- **Names.** `ShellSection`, `ShellElement`, `ShellLoad`, `ShellForces` in `struct_core`. The existing `AreaLoad` (a
  floor load distributed to frames) is unchanged; its docstring states that it is not a load on shell elements.
- **Shell elements are a separate list**, `StructuralModel.shell_elements`, not members of the `Element` union, so code
  that assumes two-node elements is not affected.
- **Local axes** follow SAP2000 exactly (CSI Analysis Reference Manual §10.6–10.7). Arguments: `axis_angle` (degrees,
  rotates axes 1 and 2 about +3, CSI "ang") and `reference_vector` (parallel to the 3-2 plane, positive projection on
  local 2).
- **One canonical, vectorized axes function** in a new module `struct_core.geometry` (numpy becomes a dependency of
  `struct_core`). Element stiffness, loads, results, `struct_link`, the GUI, and slab design use it. Axes are never
  computed anywhere else.
- **Three frames are kept separate:** global X-Y-Z; the user-facing local 1-2-3 (material, loads, output); the
  element's internal formulation frame (mean-plane projection, natural coordinates). Stiffness is computed in the
  internal frame and transformed once; every result is reported in 1-2-3.
- **Shells are never dropped silently.** Every package that reads a model either supports shells or refuses a model
  with shells, with a clear message.
- **Pass/fail tests are closed-form solutions.** SAP2000 is a convergence check: both programs approach the same value
  under mesh refinement, within a stated tolerance. Tolerances are not fitted to coarse-mesh SAP2000 results.
- `rebar_material_id` and `cover` stay off `ShellSection` until slab design.

## M1. struct_core schemas, geometry, and no silent drops `[x]` 17 November 2026

Done: all items below. Tests: `struct_core/tests/test_shell_geometry.py`, `test_shell_schema.py`;
`struct_load/tests/test_shell_weight.py`; `struct_link/tests/test_shell_refusal.py`; `fem2d/tests/test_shell_refusal.py`;
`fem3d/tests/test_shell_refusal.py` (fem3d refuses shells until M5); `struct_agent/tests/test_shell_reporting.py`. The
comparison of axes with SAP2000 follows in M6.

Schemas (`struct_core`):

- `ShellSection` (`type="shell"`): `material_id`, `thickness`, optional `bending_thickness` (CSI th and thb; default
  equal to `thickness`), `behavior` (`membrane`, `plate`, `shell`), `plate_theory` (`thick`, `thin`).
- `ShellElement`: `id`, 3 or 4 distinct `node_ids`, `section_id`, `axis_angle`, optional `reference_vector`.
- `ShellLoad` in `LoadCase.shell_loads`: element IDs, uniform pressure (force per area), direction `local_3` or a global
  direction. Self-weight uses `LoadCase.include_self_weight`.
- Results: `ShellForces` (F11, F22, F12, M11, M22, M12, V13, V23, per unit length; CSI §10.18 names and signs),
  `ShellElementResult` (values at the 2×2 Gauss points and extrapolated to the element corners, one to one with the
  SAP2000 `AreaForceShell` table), node-averaged values, and superposition for combinations and envelopes.
- Units: `convert_project_units` converts thickness and pressure. Validation: references, duplicate IDs, node count,
  zero area.

Geometry (`struct_core.geometry`):

- `shell_local_axes(coords, axis_angle, reference_vector)` returns an (N, 3, 3) array for N elements at once.
- Shape checks (CSI §10.2.1), with the clause cited in every message:
  - warping: largest angle between corner normals; warn above 30°, refuse above 45°;
  - inside angle at each corner: refuse at 180° or more, warn outside 45°–135°;
  - aspect ratio (triangle: longest to shortest side; quadrilateral: ratio of the distances between midpoints of
    opposite sides): warn above 4, refuse above 10.
- `rotate_shell_resultants`: N and M as tensors, V as a vector, into any in-plane direction.
- `check_normal_consistency`: flags adjacent elements with flipped normals, and elements whose normal is not a required
  direction (a meshed floor must have all normals at +Z).
- Node averaging: the elements at a node are grouped by plane; each group is rotated into a common direction with
  `rotate_shell_resultants` and then averaged. Values are never averaged across a slab-wall junction; each plane is
  reported separately. A warning is given only for flipped normals.

No silent drops (each with a test):

- `struct_load` self-weight and mass source include shell elements (otherwise the ELF seismic weight is understated).
- The `.s2k` and `.e2k` writers of `struct_link`, and `fem2d`, refuse models with shells, with a clear message, until
  they support them.
- The `struct_agent` model summary and validation report shell elements; they neither fail nor ignore them.

Tests for the axes: horizontal, vertical, sloped, and nearly horizontal elements (both sides of the 1e-3 tolerance);
angle and reference-vector cases; cyclic renumbering. Comparison with SAP2000 axes follows in M6.

## M2. Sparse assembly and sparse modal analysis `[x]` 17 November 2026

Done: all items below. Tests: `fem3d/tests/test_sparse_assembly.py`, `structdyn/tests/test_sparse_modal.py`. Oracle:
largest relative difference between the sparse and the former dense solver 7.6e-13; stored references unchanged
(`research_paper/experiment_run.md`, 17 November 2026).

- fem3d: every element type supplies (DOF indices, matrix) blocks; shell blocks are vectorized, (N, 24, 24). Assembly
  in COO, then CSR. Static solution with `scipy.sparse.linalg`; sparse reactions.
- `_auto_fix_unstable_dofs` rewritten with a node-to-element map (linear in model size), with the same rules for
  frames, trusses, and springs.
- `buckling_analysis` and `Results.create_report` updated for sparse matrices.
- structdyn: `MDF` keeps sparse `K` and `M`. One named constant sets the dense/sparse switch: at or below it, the
  matrices are made dense and the current `eigh` path runs unchanged; above it, `scipy.sparse.linalg.eigsh` in
  shift-invert mode at σ = 0. A singular `K` gives a clear error. Time-history and damping-matrix methods stay dense and
  give a clear error for large sparse models.
- Tests: dense and sparse `K` agree to round-off on a frame model; the switch is tested on both sides of the constant;
  `eigsh` agrees with `eigh` on a medium model; singular `K` error.
- Checks before stopping: all fem3d and structdyn tests, the SAP2000 frame tests, struct_agent tests, research_paper
  tests, and the oracle. Round-off changes logged in `research_paper/experiment_run.md`.
- **Stop after M2** so that it can be committed on its own.

## M3. Membrane with drilling rotation `[x]` 17 November 2026

Done: `fem3d/elements/shell_membrane.py` (vectorized kernel in the element plane: stiffness, condensed strain
matrices, membrane forces at the Gauss points); tests: `fem3d/tests/test_shell_membrane.py`. The element is not yet
reachable from `Structure` (M5).

- Ibrahimbegovic, Taylor and Wilson (1990), as presented in Wilson Ch. 9, coded from the rendered page images.
- Tests: one free element has six zero eigenvalues, no spurious zero-energy modes, and a symmetric stiffness; membrane
  patch test (CSI Problem 2-001, MacNeal and Harder 1985); in-plane cases of the straight, curved, and twisted
  cantilevers (Problems 2-002 to 2-004) against their hand calculations.

Findings that changed the formulation (all checked on page images):

- Wilson Eq. 9.11 is printed with the sign of a clockwise rotation. With counterclockwise drilling rotations (needed to
  connect to frames) the continuum rotation is (dv/dx - du/dy)/2, as in Ibrahimbegovic and Frey (1992) Eq. 42. A
  test shows that the printed sign would strain the penalty under a rigid rotation.
- Wilson's element as printed (tangential mid-side displacements set to zero) cannot represent the Poisson strain of
  pure bending: rectangles are 1/(1 - nu^2) too stiff (0.9375 of the exact deflection for nu = 0.25, against Wilson's
  own Table 9.1 value 1.000 and SAP2000's 0.999 in Problem 2-002). The tangential mid-side displacements are kept as
  four incompatible modes, condensed at element level, as in Ibrahimbegovic and Frey (1992) Eqs. 24, 37, 39, and 45.
  Pure bending of rectangles is then exact for any nu.
- With 2x2 integration, rectangles and parallelograms keep a fourth zero-energy mode (alternating rotations at the
  corners), which spreads through a mesh: SAP2000's Model A of Problem 2-002 (6x1 rectangles, drilling rotations
  free) is singular with Wilson's element as printed (condition number 5e17). The cure of Cook et al. §3.10, a
  penalty on theta1 - theta2 + theta3 - theta4, is added, orthogonalized against linear rotation fields so that it
  does not affect rigid motion, constant strain, or pure bending. Its factor, 0.025 D33, is Wilson's k0; it is a
  constant of the formulation (`ROTATION_HOURGLASS_FACTOR`), not a user input.
- Wilson's Table 9.1 values for distorted elements (0.502 and 0.280 for a = 1 and 2) are not reproduced by any
  variant tried (with or without the incompatible modes, the purification, 3x3 integration, or other penalty
  locations and factors), so they are not used as test values.
- The one-element test has 3 zero eigenvalues for the membrane alone (6 in the full shell, M5).
- The twisted cantilever (2-004) involves bending even under its in-plane load, so it moves to M5.

Results (pass/fail against closed forms; SAP2000 values from the CSI manuals for information only):

| Case | This element | SAP2000 | Closed form |
|---|---|---|---|
| 2-001 patch test, prescribed edges or tractions, rotations free | exact | exact | 1333, 1333, 400 psi |
| 2-002 axial, Models A, B, C | exact | 3E-05 | 3.0E-05 in |
| 2-002 in-plane shear, Model A (6x1) / 48x8 / F / G | 0.1073 / 0.1081 / 0.1068 / 0.1076 | 0.1072 / - / 0.1061 / 0.1071 | 0.1081 in |
| 2-002 in-plane moment, Model A / F / G | 9.000E-04 / 8.956E-04 / 8.958E-04 | 8.990E-04 / 8.886E-04 / 8.825E-04 | 9.000E-04 in |
| 2-002 Models B and C (6x1, distorted), shear | 0.0222, 0.0840 | 0.0227, 0.0804 | 0.1081 in |
| 2-003 in-plane, 90x1 / 90x3 | 0.08858 / 0.08851 | 0.0845 / 0.0884 | 0.08860 in |
| Simple beam, unit end couples (Ibrahimbegovic and Frey §6.2) | 1.5000, 0.6000 | - | 1.5, 0.6 |

## M4. MITC4 plate bending `[ ]`

- Dvorkin and Bathe (1984); Bathe §5.4.2. Assumed transverse shear strains at the tying points, shear correction 5/6.
  `plate_theory="thin"` raises a clear error until DKQ is implemented.
- Tests: bending patch test (2-001); out-of-plane cantilever cases (2-002 to 2-004); rectangular plate under uniform
  and point load (2-005) against Navier thin-plate solutions (Timoshenko Ch. 5 or Reddy Ch. 6) and first-order shear
  deformation solutions (Reddy Ch. 10); no shear locking as the thickness decreases.

## M5. Flat shell, loads, mass, and results `[ ]`

- Membrane and bending combined in a flat facet element; warped elements projected onto the mean plane (Wilson Ch. 10,
  checked on the page images); pure membrane, pure plate, or full shell.
- Loads: uniform pressure in local 3 or a global direction, self-weight, consistent nodal loads.
- Mass: lumped as in CSI §10.11 (proportional to the diagonal of the consistent mass, translations only); consistent
  translational mass for `lumped=False`.
- Results: resultants at the Gauss points, extrapolated to corners, and averaged at nodes as in M1.
- `struct_core` adapter in both directions.
- Tests: Scordelis–Lo roof (Problem 2-006) with a convergence table against Belytschko et al. (1985) and Wilson Ch. 10;
  twisted cantilever (Problem 2-004, both loads; moved from M3); one free element has six zero eigenvalues;
  invariance under cyclic renumbering, the axis angle (isotropic material), and a rigid rotation of the model.

## M6. Comparison with SAP2000 (struct_link live session) `[ ]`

- Push shell sections (`PropArea.SetShell_1`), area objects, local axes, pressure loads, and self-weight; pull
  `AreaForceShell` and `AreaElm.GetTransformationMatrix`.
- Read the axes of SAP2000's own verification models (`Verification\Analysis Examples\*.sdb`, opened from a temporary
  copy) for the axis tests.
- SAP2000 results stored as JSON fixtures, so that the tests also run without SAP2000; live tests regenerate them
  where SAP2000 is installed. Convergence checks only, as stated in Decisions.
- `.s2k` and `.e2k` exchange of area objects stays in ROADMAP.md §1b.

## M7. Documentation and checks `[ ]`

- `CHANGELOG.md` of every package touched, package READMEs, the root README, and one minimal example.
- ROADMAP.md §4 marked in progress; agent tools for shells listed as the next step.
- Tests of fem3d, structdyn, struct_core, struct_link, and struct_agent; oracle check. Log in
  `research_paper/experiment_run.md` only if a benchmark or oracle result changes.

## Later sessions

DKQ and DKT thin plates, triangles (Allman membrane), mesh generation, `.s2k`/`.e2k` area exchange, agent tools, and
slab design.
