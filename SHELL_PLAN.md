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

Sensitivity to the penalty factors (accepted 17 November 2026: both stay at 0.025 D33, the hourglass factor citing
Wilson Eq. 9.14 and Cook et al. §3.10). Each factor was scaled by 0.1 and 10 with the other at 0.025
(`test_results_are_insensitive_to_the_penalty_factors`):

- The patch tests of 2-001 (prescribed edges, with and without a rigid rotation, and edge tractions) and pure bending
  of rectangles stay exact to round-off. Rectangular meshes (2-002 Model A) do not change at all.
- Change of the tip displacements (percent):

| Case | Hourglass x0.1 | Hourglass x10 | Drilling x0.1 | Drilling x10 |
|---|---|---|---|---|
| 2-002 Model B (6x1 trapezoids), shear / moment | +3.7 / +3.5 | -13.3 / -12.3 | +0.4 / +0.4 | -1.1 / -1.3 |
| 2-002 Model C (6x1 parallelograms, trapezoidal ends), shear / moment | +0.8 / +0.7 | -2.9 / -2.3 | +0.9 / +0.7 | -2.2 / -1.6 |
| 2-002 Model F, shear / moment | +0.14 / +0.03 | -0.86 / -0.05 | +0.01 / 0.00 | -0.01 / -0.01 |
| 2-002 Model G (60x2), shear / moment | 0.00 / 0.00 | -0.03 / -0.03 | +0.13 / +0.14 | -0.82 / -0.84 |
| 2-003, 6x1 | +0.03 | -0.31 | +0.10 | -0.09 |
| 2-003, 90x1 / 90x3 | +0.02 / 0.00 | -0.01 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

  The factors matter only for coarse, strongly distorted elements (Model B has 45-degree trapezoids with aspect ratio
  near 5); every refined mesh moves by less than 1 percent, which the test asserts.
- Also tested: a single rectangle, a single distorted quadrilateral, and a free 2x2 mesh (rectangular and distorted)
  each have exactly three zero eigenvalues; on a distorted quadrilateral both penalty rows vanish for the rigid modes
  and constant strain, and the hourglass row also for any linear rotation field (`membrane_penalty_vectors`).

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

## M4. MITC4 plate bending `[x]` 17 November 2026

Done: `fem3d/elements/shell_plate.py` (vectorized kernel in the element plane: rigidities, stiffness, curvature and
assumed shear strain matrices, resultants M11, M22, M12, V13, V23 at the Gauss points, consistent pressure loads);
tests: `fem3d/tests/test_shell_plate.py`. Not yet reachable from `Structure` (M5).

- Dvorkin and Bathe (1984); Bathe §5.4.2. Assumed transverse shear strains at the tying points, shear correction 5/6.
  `plate_theory="thin"` raises a clear error until DKQ is implemented.
- Tests: bending patch test (2-001); out-of-plane cantilever cases (2-002 to 2-004); rectangular plate under uniform
  and point load (2-005) against Navier thin-plate solutions (Timoshenko Ch. 5 or Reddy Ch. 6) and first-order shear
  deformation solutions (Reddy Ch. 10); no shear locking as the thickness decreases.

Notes (formulas checked on the page images of Bathe pp. 420-429, Reddy pp. 367-370, Timoshenko pp. 202 and 206, and
CSI Analysis Reference Manual §10.18):

- Kinematics and signs follow Bathe Eqs. 5.88 and 5.99 (beta_x = -theta_y, beta_y = theta_x). The resultants
  M = C_b kappa and V = C_s gamma equal the CSI definitions (M = -integral of x3 sigma, V = integral of sigma_13).
- The MITC4 shear interpolation is coded in covariant form, gamma = J^-1 (gamma_r, gamma_s). Tests compare it with
  Bathe's explicit Eq. 5.101 (2 x 2 element) and Eqs. 5.102 to 5.105 (general quadrilateral), coded independently in
  the test; both agree to round-off.
- The thin-plate refusal (`plate_theory="thin"`) belongs to the adapter that reads `ShellSection` (M5); the kernel is
  the thick-plate formulation only. CSI uses the bending thickness thb for bending and transverse shear
  (§10.3, "Section Thickness"); M5 passes `ShellSection.effective_bending_thickness`.
- The out-of-plane twist case of 2-002 (load case 4) and the out-of-plane load of 2-003 are not closed-form tests:
  their hand calculations use the beam torsion constant 0.229 b d^3, while shell theory gives b d^3 / 3 (CSI's own
  discussion of both problems). They move to M5 and M6 as comparisons, together with 2-004.
- The Navier implementation is checked against Reddy Table 10.2.1 (0.4259 and 0.4111 for b/h = 10 and 20, 0.4062 for
  classical plate theory).

Results (pass/fail against closed forms; SAP2000 thick-plate values from the CSI manual for information only):

| Case | This element | SAP2000 | Closed form |
|---|---|---|---|
| 2-001 bending patch test (t = 0.001 and 0.05 in) | exact, V = 0 | exact | M = 1.111E-07, 1.111E-07, 3.333E-08 |
| 2-005 square, simple, uniform: quarter 4x4 / 16x16 | 4.041 / 4.061 | 4.03 / 4.06 | 4.062 (Navier) |
| 2-005 square, simple, point: 4x4 / 16x16 | 11.54 / 11.59 | 11.61 / 11.61 | 11.60 (Navier) |
| 2-005 square, clamped, uniform: 4x4 / 16x16 | 1.251 / 1.264 | 1.25 / 1.27 | 1.26 (Table 35) |
| 2-005 square, clamped, point: 4x4 / 16x16 | 5.404 / 5.596 | 5.38 / 5.61 | 5.60 (Table 37) |
| 2-005 2x10, simple, uniform: 4x20 / 8x40 / 16x80 | 12.65 / 12.89 / 12.95 | - | 12.97 (Navier) |
| 2-005 2x10, clamped, point: 4x20 / 8x40 / 16x80 | 6.89 / 7.15 / 7.22 | - | 7.25 (Table 37) |
| Square, simple, uniform, 8x8 quarter, a/h = 10 to 100,000 | 0.9987 to 0.9987 of FSDT | - | Reddy Eq. 10.2.14 |
| 2-002 out-of-plane tip load: 6x1 / 24x4 / 48x8 | 0.4288 / 0.4310 / 0.4315 | 0.4321 (6x1) | 0.4321 |
| 2-002 out-of-plane tip moment (rad) | 0.0360 | 0.0360 | 0.036 |

The 2x10 meshes of the Problem (4x4 and 16x16 quarters) have elements of aspect ratio 5; the tests use square
elements and show convergence. With the Problem's 16x16 mesh the clamped point load gives 7.12 (-1.8 percent).

Stress resultants (added 18 November 2026; Gauss-point values, before the extrapolation and averaging of M5):

- Signs against CSI §10.18. Strip of 2-002 (L = 6, b = 0.2, t = 0.1) with a tip load P = +1 along +3: at every Gauss
  point V13 = +P/b, V23 = 0, M22 = M12 = 0, and M11 = P(L - x_c)/b > 0 (tension at the bottom), to round-off on the
  6x1 and 24x4 meshes. M11 is constant along each element (constant curvature), so a Gauss point carries the moment
  at the element centroid x_c; equilibrium makes that value exact. The test uses nu = 0, which makes the strip an
  exact beam. With nu = 0.3 the strip is a plate: the clamped root restrains the anticlastic curvature (M22 up to
  4.6, V13 from -30 to +28 in the root elements of the 24x4 mesh, against P/b = 5) and the free edges carry the
  Mindlin shear boundary layer (V13 within 33 percent of P/b beyond x = 1). These are plate effects, not errors.
- Simply supported square plate (a = 2, nu = 0.3), uniform downward load, quarter meshes, against the Kirchhoff
  Navier series at the same Gauss points (for a simply supported plate the first-order shear deformation moments and
  shears are identical; checked with Reddy Eqs. 10.2.14 to 10.2.17). The series reproduce Timoshenko Table 8:
  0.0479 q a^2 at the center, and 0.338 q a at the middle of the side (Timoshenko Eq. p gives 0.33766).

| Quarter mesh | 4x4 | 8x8 | 16x16 |
|---|---|---|---|
| Gauss point nearest the center | (0.947, 0.947) | (0.974, 0.974) | (0.987, 0.987) |
| M11 = M22 / (q a^2): MITC4 | 0.04715 | 0.04770 | 0.04784 |
| M11 = M22 / (q a^2): Navier at the point | 0.04766 | 0.04783 | 0.04787 |
| Error | 1.07 % | 0.27 % | 0.06 % |
| Gauss point nearest the side middle (0, 1) | (0.053, 0.947) | (0.026, 0.974) | (0.013, 0.987) |
| V13 / (q a): MITC4 | -0.2813 | -0.3080 | -0.3224 |
| V13 / (q a): Navier at the point | -0.3112 | -0.3245 | -0.3310 |
| Error | 9.6 % | 5.1 % | 2.6 % |

  The moments converge at second order and the transverse shear at first order, as Bathe §5.4.2 notes for the shear
  of MITC4. The values are the same for span/thickness 100, 1000, and 20,000 (no locking of the shear resultants).
  V13 is negative at the side x = 0 under a downward load (CSI Eqn. 10-9a).

## M5. Flat shell, loads, mass, and results `[x]` 18 November 2026

Done: `fem3d/elements/shell.py` (`ShellElement`, `ShellBatch`), `Structure.shells` and `add_shell`, `ShellLoad`,
`Structure.shell_resultants()`, shell resultants in response spectrum analysis, the adapter in both directions,
shell self-weight in struct_load's direct `apply_self_weight(Structure)`. Tests: `fem3d/tests/test_shell_structure.py`,
`test_shell_adapter.py`, `test_shell_refusal.py`; `struct_agent/tests/test_shell_reporting.py`.

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

Decisions and notes (formulas checked on the page images of Wilson Ch. 10; CSI §10.3, §10.11, §10.12, and §10.18
read in the manual's text):

- The internal frame is the projection of the corners on the mean plane (through their centroid, normal to axis 3),
  measured along axes 1 and 2. Since the kernels are invariant under in-plane rotation, the resultants come out in
  the local 1-2-3 axes without a further rotation.
- Rigid links of warped corners: u1_flat = u1 - d theta2, u2_flat = u2 + d theta1, with d the height of the corner
  above the mean plane (Wilson Eq. 10.2, his d being the offset of the mid-surface node from the flat node). A free
  warped element has exactly six zero eigenvalues and no force under any rigid rotation, which fixes the sign.
- Areas are those of the mean-plane element (equal to `struct_core.geometry.shell_areas`); lumped mass uses
  `shell_corner_weights`, so fem3d and struct_load's mass source lump shell mass identically.
- With a mass source, struct_load already lumps the shells' self-mass at the nodes (M1), so the adapter gives the
  shells a copy of their material with rho = 0; without a mass source they carry rho x th. Frames are unchanged.
- The agent's analysis tools refuse models with shells (`struct_agent.tools.analysis.shell_model_refusal`): its
  load, result, and design tools are frame-only, and shell results would be computed and ignored. This stays until
  the shell tools are added (ROADMAP.md §4).
- A membrane alone gives no stiffness to the out-of-plane DOFs and a plate alone none to the in-plane DOFs; as in
  SAP2000, the user restrains them. Shell nodes are never fixed automatically.

Results (pass/fail tolerances in the tests):

| Case | Result | Reference |
|---|---|---|
| Free element, flat and warped, any orientation | 6 zero eigenvalues, rigid motions free of force | exact |
| 2-001 patch (tension and pure bending) in an arbitrary 3D plane | exact at every Gauss point | closed form |
| Scordelis-Lo roof, free-edge deflection: 4x4 / 8x8 / 16x16 / 32x32 | 0.3160 / 0.3039 / 0.3017 / 0.3014 ft | 0.3024 (MacNeal and Harder); 0.3086 (Scordelis and Lo) |
| Twisted beam, in-plane: 12x2 / 24x4 / 48x8 | 0.005445 / 0.005467 / 0.005480 | 0.005424 (beam theory; CSI hand 0.005429) |
| Twisted beam, out-of-plane: 12x2 / 24x4 / 48x8 | 0.001772 / 0.001770 / 0.001771 | 0.001754 (CSI hand 0.001749) |
| Simply supported plate, first frequency, lumped: 8x8 / 16x16 | 0.9913 / 0.9976 of exact | Kirchhoff closed form |
| Same, consistent mass: 8x8 / 16x16 | 1.0171 / 1.0040 of exact | Kirchhoff closed form |
| Example slab (a/t = 20), center M11 | 3.832 kN-m/m | 0.0479 q a^2 = 3.832 |

SAP2000 gives 0.3068 ft for Scordelis-Lo (6x6, thin) and 0.005402 / 0.001760 for the twisted beam (12x2, thick). The
twisted beam converges about 1 percent above MacNeal and Harder's beam-theory values; their own good elements are
within 2 percent on the 12x2 mesh (their Table 10), and the strip is thick (t/w = 0.29), so the test tolerance is 1.5
percent. The comparison with SAP2000 under mesh refinement is M6.

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
