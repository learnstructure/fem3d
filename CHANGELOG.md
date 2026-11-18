# Changelog

All notable changes to `fem3d` are recorded in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). The code of the paper is the git tag `v2.1-paper`.

## [Unreleased]

### Added
- `fem3d.loads.bending_shape_functions`: the exact interpolation (deflection and cross-section rotation) of a
  two-node Timoshenko beam, which reduces to the Hermite cubics without shear deformation.
- `FrameElement.shear_parameters()`: the shear parameters (phi_2, phi_3) used by both the stiffness matrix and the
  equivalent nodal loads.
- `tests/test_timoshenko_member_loads.py`: the shapes solve the Timoshenko beam equations; the equivalent nodal
  loads of distributed loads (full, partial, trapezoidal) and of point forces and moments equal the fixed-end forces
  of the flexibility (unit-load) method, with and without shear deformation, in both bending planes; and a member
  point load gives the same solution as a mesh with a node at the load, with and without end releases.

### Fixed
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
