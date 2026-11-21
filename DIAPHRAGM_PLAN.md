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

## M2. `fem3d` and `structdyn` solver support `[ ]`

- `fem3d.Structure` sparse transformation `T`, `expand_displacements`, `influence_vector`, `reduce_matrix`, `get_reduced_matrices` (`T^T K T`, `T^T M T`).
- Static, modal, RSA, buckling, and `struct_core` adapter support; `structdyn` (`MDF`, `ModalAnalysis`, `ResponseSpectrumAnalysis`) updated to eliminate direct `free_dofs` slicing assumptions.
- Closed-form, multi-story, beam axial force, unit-invariance, and zero-diaphragm identity regression tests.

## M3. `struct_load` integration `[ ]`

- Mass source and seismic weight verified identical without diaphragms and consistent with diaphragms.
- `apply_lateral_seismic` distributes story forces by nodal seismic weight on diaphragm floors (ASCE 7-22 §12.8.4.1) while keeping non-diaphragm behavior unchanged.

## M4. `struct_link` exchange and RSA comparison `[ ]`

- ETABS `.e2k` and `EtabsLiveSession` read/write of rigid diaphragms; SAP2000 `.s2k` and `Sap2000LiveSession` read/write of diaphragm constraints; keep refusing semi-rigid diaphragms and non-diaphragm constraints under `allow_partial_import=False`.
- Extend `pull_results` and `compare_results` to compare RSA cases in addition to static and modal cases.

## M5. Live verification against ETABS 23 and SAP2000 27 `[ ]`

- Same 3D static, modal, and RSA diaphragm models verified in both ETABS 23 and SAP2000 27 to round-off tolerances, including beam axial forces.
- ETABS 23 verification models with diaphragms (temporary copies) imported with `allow_partial_import=False`, round-tripped, and compared; report whether any ETABS verification model uses in-plane fixed supports on diaphragm nodes.
- Regression suite for non-diaphragm models.

## M6. `struct_agent` tools and ASCE 7-22 §12.3.1.2 eligibility `[ ]`

- Deterministic ASCE 7-22 §12.3.1.2 checker integrated into `create_frame_model`, `create_model_from_plan_draft`, `create_project`, `inspect_project`, and analysis tools (`run_analysis`, `run_response_spectrum`, `run_time_history`, `verify_with_csi`).
- Log tool and schema changes in `research_paper/experiment_run.md`.

## M7. Documentation, examples, and full suite regression `[ ]`

- Update `CHANGELOG.md`, `README.md`, `ROADMAP.md`, and add a minimal runnable example; run full `pytest` across all touched packages.
