# Rigid floor diaphragms: implementation plan

Plan for `ROADMAP.md` §1 (P1), approved 21 November 2026. Theory and code sources checked on rendered PDF page images:

- Wilson, *Three-Dimensional Static and Dynamic Analysis of Structures*, Ch. 7 §7.5 "Floor Diaphragm Constraints" (PDF pp. 104–109; noting printed misprint in Eq. 7.8 entry `(3,3)`).
- CSI Analysis Reference Manual, Ch. 5 §§5.3–5.4, §5.12, and §5.13 "Diaphragm Constraint" (`SAP2000 27/Manuals`, PDF pp. 61–66 and 75–76).
- ASCE 7-22 §12.3.1.2, Table 12.3-1, §12.8.4.1, and §12.8.4.2.1 (`references/standards`, PDF pp. 179–180).

Status marks: `[ ]` planned, `[~]` in progress, `[x]` done (with date). Each milestone ends with its tests passing. Changes are recorded in the `CHANGELOG.md` of each package touched.

## Decisions

- **Exact kinematic transformation.** Enforce rigid diaphragms via exact DOF transformation `u = T q`, `K_r = T^T K T`, `M_r = T^T M T`, and `F_r = T^T F`, never penalty stiffness. For normal axis `3` (`Z`, `X`, or `Y`) and in-plane right-handed axes `(1, 2)`, each constrained node satisfies `u1_i = u1_m - (x2_i - x2_m) r3_m`, `u2_i = u2_m + (x1_i - x1_m) r3_m`, `r3_i = r3_m`, while out-of-plane DOFs `(u3_i, r1_i, r2_i)` remain independent.
- **Master point, mass transformation, and scaling.** Place the diaphragm master at the center of mass computed from the assembled translational masses (falling back to geometric centroid when massless), matching CSI §5.12. Transform assembled `M` through `M_r = T^T M T` so master rotational inertia follows automatically (`sum(I_3i + m_1i (x2_i - x2_m)^2 + m_2i (x1_i - x1_m)^2)`). Internally scale the master rotation column of `T` by the diaphragm RMS radius `L_d` so `K_r` and `M_r` stay well-conditioned across all unit systems (`m`, `mm`, `in`, `ft`). Models with no diaphragms take the existing `free_dofs` fast path unchanged.
- **Rigid only (`struct_core.Diaphragm`).** Support `rigidity="rigid"` only. Explicitly refuse `"semi_rigid"` / `"semirigid"` because semi-rigid diaphragms require ETABS slab shells, which are deferred.
- **Supports and overlapping constraints.** A node may belong to at most one diaphragm. Out-of-plane fixed supports and elastic support springs on diaphragm nodes are allowed. Fixed in-plane supports on diaphragm nodes are refused with a clear validation error (Option A; check in Milestone 5 whether any ETABS verification model needs them).
- **Beam axial forces inside a rigid diaphragm.** By Wilson §7.4–7.5 and CSI, beams with both ends in the same rigid diaphragm have zero axial deformation (`u1_j - u1_i = 0`), giving `P = 0` for axially unloaded beams and only the fixed-end axial force `-f_eq,local` for axially loaded beams. ETABS and SAP2000 report the same values.
- **ELF lateral forces (`struct_load`).** Models without diaphragms keep today's load distribution bit-for-bit unchanged. On diaphragm stories, distribute ELF story force in proportion to nodal seismic mass so the resultant acts at the floor center of mass per ASCE 7-22 §12.8.4.1 (Option B).
- **Agent default (`struct_agent`, Milestone 6).** Automatically apply rigid diaphragms (`Option B`) on eligible 3D models only when the deterministic checker confirms every ASCE 7-22 §12.3.1.2 condition from model data (concrete slab, span-to-depth `<= 3.0`, no Type 2–5 horizontal irregularity in Table 12.3-1). When any condition cannot be confirmed, do not apply rigid diaphragms and report which condition could not be determined. Always report the assumption and keep `rigid_diaphragm=False` as opt-out.
- **Rigor and verification.** Tolerances come from round-off or theory, never fitted to results. No difference is labeled "known" without a checked explanation. Copy CSI files to temporary folders before opening and close all started CSI processes.

## M1. `struct_core` schema, validation, units, and no silent drops `[x]` 21 November 2026

Done: all items below. Test runs after final code edits: `struct_core` 193 passed, `fem2d` 115 passed, `fem3d` 192 passed, offline `struct_link` 99 passed (599 passed total). New tests: `struct_core/tests/test_diaphragms.py`, `fem2d/tests/test_diaphragm_refusal.py`, `fem3d/tests/test_diaphragm_refusal.py`, `struct_link/tests/test_diaphragm_refusal.py`.

- `struct_core`: `Diaphragm` schema (`id`, `name`, `node_ids`, `axis`, `rigidity="rigid"`), `StructuralModel.diaphragms`, `StructuralModel.get_diaphragm()`, and export from `struct_core` and `struct_core.structural_model`. Semi-rigid and flexible rigidities refused at validation.
- Validation (`validate_structural_model`): unique diaphragm IDs, 3D-only refusal (`dimension="2D"` or `vertical_axis="Y"`), existing nodes, at least 2 distinct nodes, no intra-diaphragm duplicates, no node in more than one diaphragm, coplanarity along `axis` (within `max(1e-6, 1e-6 * span)`), and refusal of fixed in-plane support DOFs while allowing out-of-plane supports and elastic springs.
- Units: `convert_project_units` preserves diaphragms cleanly across unit systems.
- Refusals: `fem2d` refuses diaphragms permanently (2D plane solver); `fem3d` and `struct_link` writers hold temporary M1 refusals until M2 and M4.
- Updates to `ROADMAP.md`: accidental torsion moved to §1a P1 as next task; new §4 P2 item for mass distribution on floors without diaphragms.

## M2. `fem3d` and `structdyn` solver support `[x]` 21 November 2026

Done: all items below. Test runs after final code edits: `fem3d` 202 passed, `structdyn` 111 passed, `struct_core` 193 passed (506 passed total). New/updated tests: `fem3d/tests/test_diaphragms.py`, `fem3d/tests/test_diaphragm_refusal.py`, `structdyn/tests/test_diaphragms.py`.

- `fem3d.Structure` sparse transformation `T`, `expand_displacements`, `influence_vector`, `reduce_matrix`, `reduce_vector`, `get_reduced_matrices` (`T^T K T`, `T^T M T`).
- Static, modal, RSA (expanding modal displacements to original nodes before CQC/SRSS/ABS combination), buckling, visualization helper, and `struct_core` adapter support; `structdyn` (`MDF`, `ResponseSpectrumAnalysis`) updated to use `expand_displacements`, `influence_vector`, `reduce_matrix`, and `reduce_vector`.
- Closed-form eccentric diaphragm (Z, X, Y planes), multi-story kinematics/units/origin shift, beam axial force, elastic springs, truss nodes, buckling, sparse vs dense modal (`eigsh` vs `eigh`), Newmark time-history, and zero-diaphragm identity regression tests.

## M3. `struct_load` integration `[x]` 21 November 2026

Done: all items below. Test runs after final code edits: `struct_load` 85 passed, `fem3d` 202 passed, `structdyn` 111 passed, `struct_agent` 174 passed (572 passed total). New tests: `struct_load/tests/test_diaphragms.py`.

- `calculate_mass_source` and `calculate_model_seismic_weight` keep identical values without or with diaphragms; added read-only `DiaphragmMassSummary` (`MassSourceResult.diaphragm_masses`) and `nodal_lumped_masses()` matching `fem3d`'s `T^T M T` master mass, center of mass, and rotational inertia across `Z`, `X`, and `Y` planes and `lateral_only=True`/`False`.
- `LoadGenerator.apply_lateral_seismic` distributes ELF story forces in proportion to nodal seismic mass on rigid Z-diaphragm floors so the resultant acts at the floor center of mass (ASCE 7-22 §12.8.4.1), while floors without diaphragms keep the exact existing `face` / `uniform` distribution.
- Verified with an asymmetric-mass floor (ELF resultant at COM, not centroid), static `fem3d` comparison against equivalent loads applied at the master COM node, and zero-diaphragm / mixed-floor regression tests.

## M4. `struct_link` exchange and RSA comparison `[x]` 21 November 2026

Done: all items below (offline tests and mocked COM sessions; live verification runs in M5). Test runs after final code edits: `struct_link` offline 119 passed (32 live deselected), `struct_core` 193 passed, `fem3d` 202 passed, `structdyn` 111 passed, `struct_load` 86 passed, `struct_agent/tests/test_link_tools.py` 21 passed (732 passed total). New/updated tests: `struct_link/tests/test_diaphragms.py`, `struct_link/tests/test_diaphragm_refusal.py`, `struct_link/tests/test_etabs_e2k_fidelity.py`, `struct_link/tests/test_sap2000_roundtrip.py` (`diff_models`).

- Capabilities & writers: removed temporary `refuse_diaphragms`, added case-insensitive diaphragm ID collision check; ETABS `.e2k` and `EtabsLiveSession` write rigid Z diaphragms (`$ DIAPHRAGM NAMES`, `DIAPH`, `Diaphragm.SetDiaphragm`, `PointObj.SetDiaphragm`), refusing non-Z or reserved `"None"`; SAP2000 `.s2k` and `Sap2000LiveSession` write rigid `Z`/`X`/`Y` diaphragms (`CONSTRAINT DEFINITIONS - DIAPHRAGM`, `JOINT CONSTRAINT ASSIGNMENTS`, `ConstraintDef.SetDiaphragm`, `PointObj.SetConstraint`), refusing 2D models and semi-rigid diaphragms.
- Readers: ETABS `.e2k` (`$ DIAPHRAGM NAMES`, story/point `DIAPH` with explicit `DIAPH "None"` override) and SAP2000 `.s2k` / live tables (`Axis=Auto`, `MultiLevel=Yes`, coplanarity checks) materialize coplanar rigid `Diaphragm` objects, split multi-story/multi-level shared names per plane, and continue refusing semi-rigid, area-assigned, non-`GLOBAL`, non-coplanar, overlapping, and non-diaphragm constraints under `allow_partial_import=False`.
- RSA pulling & `compare.py`: `Sap2000LiveSession` and `EtabsLiveSession` push zero eccentricity and pull `_CASE_RESPONSE_SPECTRUM` as unsigned `AnalysisResult(result_type="rsa")` envelopes; `_estimate_stiffness_condition_number` uses `reduce_matrix`, `_solve_modal_case` uses `influence_vector`, and `_solve_rsa_case` / `_compare_rsa_case` combine modal frame forces at each station and compare unsigned envelopes at first-principles RSA tolerances (`1e-7` Timoshenko, `1e-6` Euler-Bernoulli).
- Pre-M4 fixes also completed: unified `LoadGenerator.calculate_model_seismic_weight()` and `apply_lateral_seismic()` with `MS_ASCE7` (`test_benchmark_suite.py`: 261 passed, 2 skipped, zero reference shifts) and added `ROADMAP.md` §4 P2 lateral-only mass source item.

## M5. Live verification against ETABS 23 and SAP2000 27 `[ ]`

- Same 3D static, modal, and RSA diaphragm models verified in both ETABS 23 and SAP2000 27 to round-off tolerances, including beam axial forces.
- ETABS 23 verification models with diaphragms (temporary copies) imported with `allow_partial_import=False`, round-tripped, and compared; report whether any ETABS verification model uses in-plane fixed supports on diaphragm nodes.
- Regression suite for non-diaphragm models.

## M6. `struct_agent` tools and ASCE 7-22 §12.3.1.2 eligibility `[ ]`

- Deterministic ASCE 7-22 §12.3.1.2 checker integrated into `create_frame_model`, `create_model_from_plan_draft`, `create_project`, `inspect_project`, and analysis tools (`run_analysis`, `run_response_spectrum`, `run_time_history`, `verify_with_csi`).
- Test a model with diaphragms through every agent analysis tool (`run_analysis`, `run_response_spectrum`, `run_time_history`, `verify_with_csi`, and the story-shear and reaction helpers) so none still indexes `free_dofs` directly.
- Log tool and schema changes in `research_paper/experiment_run.md`.

## M7. Documentation, examples, and full suite regression `[ ]`

- Update `CHANGELOG.md`, `README.md`, `ROADMAP.md`, and add a minimal runnable example; run full `pytest` across all touched packages.
