"""
Element mass: torsional inertia, and the mass, weight, and area modifiers of struct_core elements.

1. Torsional mass. A fixed-free circular shaft has the torsional frequencies f_n = (2n - 1) / (4 L) sqrt(G J / (rho I_p))
   (I_p = J for a circle). The consistent (rho I_p L / 6 [[2, 1], [1, 2]]) and lumped (rho I_p L / 2 per node)
   torsional masses converge to it, from above and from below. The former torsional mass used the mass per length
   rho A instead of rho, so it was A times too small.
2. Modifiers, as SAP2000 27.1 applies them (measured on a cantilever, struct_link tests): the A modifier scales the
   axial stiffness only (bending frequencies unchanged, axial frequency / sqrt(2) for A = 0.5); the mass modifier
   scales the mass only (all frequencies x sqrt(2) for mass = 0.5); the weight modifier scales the self-weight only.
"""

import math

import numpy as np
import pytest

from fem3d import ElasticMaterial, FrameElement, Node, Section, Structure

E, NU, RHO = 200.0e6, 0.3, 7.85  # kN, m, t
L, D = 2.0, 0.1


def _torsional_frequencies(n_elements, lumped):
    material = ElasticMaterial(E=E, nu=NU, rho=RHO)
    section = Section.from_circle(diameter=D, material=material)
    structure = Structure()
    nodes = [Node(i, i * L / n_elements, 0.0, 0.0) for i in range(n_elements + 1)]
    for i, n in enumerate(nodes):
        structure.add_node(n)
        # Only the twist about the shaft axis is free; the base is fixed
        n.set_support(ux_fixed=True, uy_fixed=True, uz_fixed=True, rx_fixed=(i == 0), ry_fixed=True, rz_fixed=True)
    for i in range(n_elements):
        structure.add_element(FrameElement(i + 1, nodes[i], nodes[i + 1], material=material, section=section))
    result = structure.modal_analysis(num_modes=2, lumped=lumped, include_rotational_inertia=lumped)
    exact = [(2 * n - 1) / (4 * L) * math.sqrt(material.G * section.J / (RHO * (section.I2 + section.I3)))
             for n in (1, 2)]
    return np.asarray(result["frequencies"]), np.asarray(exact)


@pytest.mark.parametrize("lumped", [False, True])
def test_torsional_frequencies_of_a_shaft_match_the_closed_form(lumped):
    fem, exact = _torsional_frequencies(40, lumped)
    np.testing.assert_allclose(fem, exact, rtol=2e-3)
    coarse, _ = _torsional_frequencies(10, lumped)
    # consistent mass converges from above, lumped mass from below
    assert (coarse[0] > fem[0] > exact[0]) if not lumped else (coarse[0] < fem[0] < exact[0])


def _cantilever_frequencies(stiffness_modifiers):
    from struct_core import BeamElement, ConcreteMaterial, Node as ScNode, Project, RectangularSection, \
        StructuralModel, Support
    from fem3d.adapters import model_from_core

    model = StructuralModel(
        dimension="3D",
        nodes=[ScNode(id=i, x=i * 1.0, y=0.0, z=0.0) for i in range(5)],
        materials=[ConcreteMaterial(id="C", E=25e6, nu=0.2, fc=28e3, unit_weight=24.0)],
        sections=[RectangularSection(id="R", material_id="C", b=0.3, h=0.5)],
        elements=[BeamElement(id=i, start_node=i, end_node=i + 1, section_id="R",
                              stiffness_modifiers=stiffness_modifiers) for i in range(4)],
        supports=[Support.fixed_3d(0)],
    )
    project = Project(model=model)
    project.metadata.units = "kN, m"
    structure = model_from_core(project)
    result = structure.modal_analysis(num_modes=24)  # all modes of the 4-element cantilever
    ux_tip = structure.nodes[4].dofs[0]
    axial = np.abs(np.asarray(result["modes"])[ux_tip, :]) > 0.99  # modes normalized to a peak of 1
    frequencies = np.asarray(result["frequencies"])
    return frequencies[~axial], frequencies[axial]


def test_mass_modifier_scales_the_mass_only():
    other, axial = _cantilever_frequencies(None)
    other_m, axial_m = _cantilever_frequencies({"mass": 0.5})
    np.testing.assert_allclose(other_m, other * math.sqrt(2.0), rtol=1e-9)
    np.testing.assert_allclose(axial_m, axial * math.sqrt(2.0), rtol=1e-9)


def test_area_modifier_scales_the_axial_stiffness_only():
    """Regression: the A modifier also scaled the element mass (rho x modified area)."""
    other, axial = _cantilever_frequencies(None)
    other_a, axial_a = _cantilever_frequencies({"A": 0.5})
    np.testing.assert_allclose(other_a, other, rtol=1e-9)              # bending and torsion unchanged
    np.testing.assert_allclose(axial_a, axial / math.sqrt(2.0), rtol=1e-9)
    # the fundamental axial frequency is close to the closed form 1 / (4 L) sqrt(E / rho) of a cantilever bar
    assert axial[0] == pytest.approx(math.sqrt(25e6 / (24.0 / 9.80665)) / 16.0, rel=0.02)
