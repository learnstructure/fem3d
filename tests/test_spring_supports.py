"""
Tests for support springs and shear area requirements in fem3d.
"""

import pytest
import numpy as np

from fem3d import Structure, Node, FrameElement, Section, ElasticMaterial
from struct_core import (
    StructuralModel, Node as ScNode, BeamElement as ScBeamElement,
    GeneralSection, ConcreteMaterial, Support as ScSupport,
    PointLoad as ScPointLoad, LoadCase as ScLoadCase, Units
)
from fem3d.adapters.struct_core_adapter import model_from_core


def test_fem3d_cantilever_tip_spring():
    """
    3D Cantilever along X: L = 4.0 m, EI3 = 20,000 kN*m^2.
    Tip spring: kz = 1,000 kN/m.
    Tip load: Fz = -10 kN.
    k_cant = 3 * EI / L^3 = 3 * 20,000 / 64 = 937.5 kN/m.
    k_tot = 937.5 + 1000 = 1937.5 kN/m.
    delta = 10 / 1937.5 = 0.00516129 m.
    R_spring = 1000 * 0.00516129 = 5.16129 kN (upward).
    R_base = 937.5 * 0.00516129 = 4.83871 kN (upward).
    """
    struct = Structure()
    n1 = Node(1, 0.0, 0.0, 0.0)
    n2 = Node(2, 4.0, 0.0, 0.0)
    struct.add_node(n1)
    struct.add_node(n2)
    n1.set_support([1, 1, 1, 1, 1, 1])
    n2.set_springs(kz=1000.0)

    mat = ElasticMaterial(E=2.0e7, nu=0.2)
    # Strong axis bending: I3 = 0.001 -> EI = 20,000
    sec = Section(A=0.1, I2=0.001, I3=0.001, J=0.001, material=mat)
    struct.add_element(FrameElement(1, n1, n2, sec))

    n2.set_load(fz=-10.0)
    struct.solve()

    disp_z = struct.disp[n2.dofs[2]]
    assert abs(disp_z) == pytest.approx(0.00516129, rel=1e-4)

    rxn_base_z = struct.reactions[n1.dofs[2]]
    rxn_tip_z = struct.reactions[n2.dofs[2]]
    assert rxn_tip_z == pytest.approx(5.16129, rel=1e-4)
    assert rxn_base_z == pytest.approx(4.83871, rel=1e-4)
    assert rxn_base_z + rxn_tip_z == pytest.approx(10.0, rel=1e-5)


def test_fem3d_adapter_support_springs():
    """Verify that struct_core.Support with spring stiffnesses transfers to fem3d correctly."""
    model = StructuralModel(
        dimension="3D",
        units=Units(force="kN", length="m"),
        nodes=[
            ScNode(id=1, x=0.0, y=0.0, z=0.0),
            ScNode(id=2, x=4.0, y=0.0, z=0.0),
        ],
        materials=[
            ConcreteMaterial(id="C1", fc=30000.0, E=2.0e7, unit_weight=24.0),
        ],
        sections=[
            GeneralSection(id="S1", A=0.1, I2=0.001, I3=0.001, J=0.001, material_id="C1"),
        ],
        elements=[
            ScBeamElement(id="B1", start_node=1, end_node=2, section_id="S1"),
        ],
        supports=[
            ScSupport(node_id=1, ux=True, uy=True, uz=True, rx=True, ry=True, rz=True),
            ScSupport(node_id=2, kz=1000.0),
        ],
        load_cases=[
            ScLoadCase(
                id="L1",
                point_loads=[ScPointLoad(node_id=2, fz=-10.0)],
            )
        ]
    )

    struct = model_from_core(model, load_case_id="L1")
    struct.solve()

    disp_z = struct.disp[struct.nodes[2].dofs[2]]
    assert abs(disp_z) == pytest.approx(0.00516129, rel=1e-4)
    assert struct.reactions[struct.nodes[2].dofs[2]] == pytest.approx(5.16129, rel=1e-4)
    assert struct.reactions[struct.nodes[1].dofs[2]] == pytest.approx(4.83871, rel=1e-4)


def test_fem3d_support_cannot_be_both_fixed_and_spring():
    """Verify that a 3D DOF cannot be both fixed and have a spring."""
    struct = Structure()
    n1 = Node(1, 0.0, 0.0, 0.0)
    struct.add_node(n1)
    n1.set_support(uz_fixed=True)
    n1.set_springs(kz=1000.0)

    struct.number_dofs()
    with pytest.raises(ValueError, match="cannot be both restrained"):
        struct.apply_boundary_conditions()


def test_fem3d_general_section_missing_shear_area_raises_when_shear_deformation_requested():
    """Verify F12: GeneralSection without As2/As3 raises ValueError when shear deformation is requested."""
    model = StructuralModel(
        dimension="3D",
        units=Units(force="kN", length="m"),
        nodes=[ScNode(id=1, x=0.0, y=0.0, z=0.0), ScNode(id=2, x=4.0, y=0.0, z=0.0)],
        materials=[ConcreteMaterial(id="C1", fc=30000.0, E=2.0e7, unit_weight=24.0)],
        sections=[GeneralSection(id="G1", A=0.1, I2=0.001, I3=0.001, J=0.001, As2=0.08, As3=None, material_id="C1")],
        elements=[ScBeamElement(id="B1", start_node=1, end_node=2, section_id="G1", include_shear_deformation=True)],
        supports=[ScSupport(node_id=1, ux=True, uy=True, uz=True, rx=True, ry=True, rz=True)],
    )
    with pytest.raises(ValueError, match="without shear area.*As3"):
        model_from_core(model)

