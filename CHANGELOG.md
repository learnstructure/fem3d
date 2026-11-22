# Changelog

All notable changes to `fem3d` are recorded in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). The code of the paper is the git tag `v2.1-paper`.

## [Unreleased]

### Changed
- Updated frame ELF and dynamic examples and `tests/test_frame_elf.py` from `story_elevations` to `Story` (`stories`
  and explicit `base_elevation`), and updated the ELF frame base shear expectation from `132.37 kN` to `128.59 kN`
  (`node13_ux_mm` `8.9326 -> 8.7545 mm`) after unifying restrained-base mass exclusion in `struct_load`.

### Added
- **Rigid floor diaphragms (`fem3d/DIAPHRAGM_PLAN.md`, M2, M5 & M7):** exact sparse kinematic transformation
  (`u = T q`, `K_r = T^T K T`, `M_r = T^T M T`, `F_r = T^T F`, internal RMS-radius scaling `L_d`) on `Structure`
  (`add_diaphragm`, `constraint_matrix`, `expand_displacements`, `influence_vector`, `reduce_matrix`, `reduce_vector`,
  `get_reduced_matrices`, `diaphragm_masters`) across `Z`, `X`, and `Y` planes, placing the master at the floor
  center of mass (centroid when massless) so rotational inertia follows from `T^T M T`. Supported across linear
  static, modal, RSA (expanding modal vectors to original nodes before CQC/SRSS/ABS combination), linear elastic
  buckling, visualization, and `model_from_core` / `model_to_core` / `result_to_core`. `_auto_fix_unstable_dofs` leaves
  in-plane diaphragm DOFs coupled on standalone diaphragm joints and restrains only unsupported zero-stiffness
  out-of-plane DOFs without reporting artificial reactions. Example: `examples/linear/rigid_diaphragm_frame.py`.
  Tests: `tests/test_diaphragms.py`, `tests/test_diaphragm_refusal.py`.
- `tests/test_shell_structure.py`: invariance of a warped mesh (twisted beam 12x2) under cyclic renumbering of
  every element: the stiffness is the same to round-off, and the corner resultants agree after rotation into a
  common direction, although the in-plane axes turn (SAP2000's rule for warped elements). README and
  SHELL_PLAN.md record the known differences: corner moments at clamped edges, torsion of narrow strips, and the
  local axes of warped elements.
- `tests/test_shell_structure.py`: the twist of a thick strip (CSI Problem 2-002, load case 4) converges to the
  closed-form torsion of a Reissner-Mindlin strip with free edges (0.1 percent on 96x16), which for this strip is
  within 0.3 percent of the Saint-Venant value; SAP2000 converges to the shell-theory value b t^3 / 3 instead
  (SHELL_PLAN.md, M6).
- Shell elements in `Structure` (SHELL_PLAN.md, M5): `ShellElement` (`fem3d.elements.shell`), a four-node flat shell
  that combines the membrane (M3) and the MITC4 plate (M4), six DOFs per node, with `behavior` "shell",
  "membrane", or "plate", membrane and bending thicknesses, and the SAP2000 local axes of
  `struct_core.geometry.shell_local_axes` (`axis_angle`, `reference_vector`). Warped elements are projected on their
  mean plane with rigid links (Wilson Eq. 10.2). Shells are kept in `Structure.shells` (`add_shell`), apart from the
  two-node elements; IDs are unique across both. `ShellLoad` (uniform load along local 3 or a global axis, consistent
  nodal forces); mass on the translations, lumped as CSI §10.11 or consistent; `Structure.shell_resultants()` (F11
  ... V23 at the Gauss points and the corners, local axes, CSI §10.18 signs); `combined_shell_resultants` from
  `response_spectrum_analysis`. The adapter converts shell elements, sections, and loads in both directions, adds the
  shells' self-weight, reports `shell_results` and node averages per plane (`shell_node_results`), and gives the
  shells no mass of their own when a mass source lumps it at the nodes. Triangles and thin-plate (Kirchhoff)
  sections are refused with the reason. Example: `examples/linear/shell_plate.py`. Tests:
  `tests/test_shell_structure.py` (six rigid modes, flat and warped; the 2-001 patch in an arbitrary plane;
  invariance under rigid rotation, cyclic renumbering, and axis angle; Scordelis-Lo roof; twisted beam; load totals
  and self-weight; mass totals and the plate frequency against the closed form; response spectrum),
  `tests/test_shell_adapter.py`, `tests/test_shell_refusal.py`.
- `fem3d.elements.shell_plate` (SHELL_PLAN.md, M4): the MITC4 plate bending element of Dvorkin and Bathe (1984), as
  in Bathe (2014) §5.4.2, vectorized, in the element plane: isotropic rigidities (shear correction 5/6), stiffness,
  curvature and assumed shear strain matrices, the CSI resultants M11, M22, M12, V13, V23 at the Gauss points, and
  consistent pressure loads. Not yet used by `Structure` (M5). Tests: `tests/test_shell_plate.py` (three rigid modes
  of one element, the shear interpolation against Bathe Eqs. 5.101 to 5.105, invariance, the bending patch test of
  CSI Problem 2-001, rectangular plates of Problem 2-005 against Navier series and Timoshenko's tables, thick plates
  against the first-order shear deformation Navier solution of Reddy, no shear locking up to span/thickness 100,000,
  the out-of-plane cases of Problem 2-002, the signs of V13 and M11 on a cantilever strip against CSI §10.18, and the
  convergence of M11, M22, and V13 at the Gauss points of a simply supported plate against the Navier series and
  Timoshenko Table 8).
- `fem3d.elements.shell_membrane` (SHELL_PLAN.md, M3): a vectorized quadrilateral membrane with drilling rotations,
  in the element plane (stiffness, strain matrices, and membrane forces at the 2x2 Gauss points). Formulation:
  Ibrahimbegovic, Taylor and Wilson (1990) as in Wilson (2002) Ch. 9, with the tangential incompatible modes of
  Ibrahimbegovic and Frey (1992) and a penalty on the alternating-rotation mode (Cook et al. 2002, §3.10). The
  reasons for the two additions and the sign of Wilson Eq. 9.11 are recorded in SHELL_PLAN.md. Not yet used by
  `Structure` (M5). Tests: `tests/test_shell_membrane.py` (rank and rigid modes of one element, invariance, the
  patch tests of CSI Problem 2-001, exact pure bending, convergence to the hand calculations of Problems 2-002
  and 2-003, three zero eigenvalues of single elements and of a free 2x2 mesh, the penalty rows
  (`membrane_penalty_vectors`) vanishing for rigid motion, constant strain, and linear rotation fields, and the
  sensitivity of the results to the two penalty factors, recorded in SHELL_PLAN.md).
- Sparse assembly and solution (SHELL_PLAN.md, M2): `Structure.K` and `Structure.M` are `scipy.sparse` CSR
  matrices, assembled from element blocks in COO form (`Structure._assemble_sparse`, which also takes vectorized
  blocks of N elements for the shell elements to come). The static system is solved by sparse LU (SuperLU);
  reactions are computed from the sparse partitions. `get_reduced_matrices` returns sparse matrices, which
  structdyn makes dense for models up to `structdyn.mdf.mdf.DENSE_DOF_LIMIT` DOFs, so frame models follow the
  same modal path as before. Frame results are unchanged to round-off: `tests/test_sparse_assembly.py` compares
  K, M, displacements, and reactions with the former dense assembly and solution, the modal periods on both
  sides of the dense limit, and the modal and response spectrum analysis of a 2904-DOF frame.
- `buckling_analysis.assemble_geometric_stiffness` returns a sparse Kg; the reduced eigenproblem is still
  solved densely. `Results.create_report` formats sparse K.
- `SHELL_PLAN.md`: the implementation plan of shell elements (ROADMAP §4).
- `fem3d.loads.bending_shape_functions`: the exact interpolation (deflection and cross-section rotation) of a
  two-node Timoshenko beam, which reduces to the Hermite cubics without shear deformation.
- `FrameElement.shear_parameters()`: the shear parameters (phi_2, phi_3) used by both the stiffness matrix and the
  equivalent nodal loads.
- `tests/test_timoshenko_member_loads.py`: the shapes solve the Timoshenko beam equations; the equivalent nodal
  loads of distributed loads (full, partial, trapezoidal) and of point forces and moments equal the fixed-end forces
  of the flexibility (unit-load) method, with and without shear deformation, in both bending planes; and a member
  point load gives the same solution as a mesh with a node at the load, with and without end releases.

### Changed
- `ShellBatch.areas()` is the area of the flat element (projected on the plane normal to axis 3); it differs from
  `struct_core.geometry.shell_areas` by at most 1e-4 for nearly planar warped elements, whose axis 3 is now the
  normal of the plane through j1, j2, j3 (struct_core 'Changed').
- `Structure._auto_fix_unstable_dofs` finds the elements at each node in one pass over the elements (it compared
  every node with every element); the rules are unchanged.
- A singular stiffness matrix is reported from the sparse LU factorization, with the same `LinAlgError` message.

### Fixed
- `model_from_core` with a `MassSource`: preserved `Node.inertia` when `include_added_mass=True` (previously reset to
  zero when `set_mass(mass=...)` was called) and cleared both nodal mass and inertia when `include_added_mass=False`;
  when `include_element_mass=False`, removed frame and truss element self-mass (consistent with shells) so element
  self-mass is not included in `M`. Test: `tests/test_mass_source_adapter.py`.
- Member loads on Timoshenko members (`include_shear_deformation=True`): the equivalent nodal loads of
  `DistributedLoad` and `ElementPointLoad` used Euler-Bernoulli (Hermite) shape functions, which are exact only for
  full-span uniform loads (and loads symmetric about midspan). Partial, trapezoidal, and off-center point loads gave
  wrong fixed-end forces (0.4 % error in displacements and reactions of a 3D test frame compared with SAP2000). They
  now use the exact Timoshenko shapes. Results of Euler-Bernoulli members and of full-span uniform loads do not change.
- Member point moments (`ElementPointLoad` m2, m3): the equivalent nodal moments at the rotational DOFs had the wrong
  sign (a moment at the start node was applied as its negative). The translational terms were correct.
- Torsional mass of frame elements: the consistent mass matrix used the mass per length (rho A) instead of the mass
  density (rho) with the polar moment of area, so the torsional inertia was A times too small (wrong units). It is now
  rho I_p L / 6 [[2, 1], [1, 2]]. The lumped rotational inertia (`include_rotational_inertia=True`) had the same
  error; it is now rho I_p L / 2 for torsion and rho I L / 24 for bending, as in fem2d. Verified against the
  closed-form torsional frequencies of a fixed-free shaft (`tests/test_mass_and_modifiers.py`). Lumped analyses
  without rotational inertia, which the struct_agent tools use, do not change.
- `model_from_core` applies the element `mass` modifier (it was ignored) and no longer lets the `A` modifier scale
  the mass (the element mass was rho times the modified area). As in SAP2000 27.1, measured in the struct_link tests,
  the A modifier scales the axial stiffness only and the mass modifier the whole member mass.
- `examples/dynamic/02_response_spectrum_analysis_3d.py` now passes `system_type="rc_smf"` to `SeismicParameters`
  (required when `r_factor` is not given).
