"""Permanent Milestone 2 refusal tests for invalid or unsupported diaphragm definitions in fem3d."""

import pytest

from struct_core import (
    BeamElement,
    ConcreteMaterial,
    Diaphragm,
    Node as ScNode,
    RectangularSection,
    StructuralModel,
    Support,
)

from fem3d import ElasticMaterial, FrameElement, Node, Section, Structure
from fem3d.adapters.struct_core_adapter import model_from_core


def _basic_3d_structure(*, fix_top_in_plane: bool = False) -> Structure:
    s = Structure(units="kN, m")
    mat = ElasticMaterial(E=25.0e6, nu=0.2, rho=2.4)
    sec = Section.from_rectangle(width=0.4, depth=0.4, material=mat)
    n1 = Node(1, 0.0, 0.0, 0.0)
    n2 = Node(2, 5.0, 0.0, 0.0)
    n3 = Node(3, 0.0, 0.0, 3.5)
    n4 = Node(4, 5.0, 0.0, 3.5)
    n1.set_support(True, True, True, True, True, True)
    n2.set_support(True, True, True, True, True, True)
    if fix_top_in_plane:
        n3.set_support(ux_fixed=True)
    for n in (n1, n2, n3, n4):
        s.add_node(n)
    s.add_element(FrameElement("C1", n1, n3, mat, sec))
    s.add_element(FrameElement("C2", n2, n4, mat, sec))
    s.add_element(FrameElement("B1", n3, n4, mat, sec))
    return s


def test_fem3d_refuses_fixed_in_plane_support_on_diaphragm_node():
    s = _basic_3d_structure(fix_top_in_plane=True)
    s.add_diaphragm("D1", [3, 4], axis="Z")
    with pytest.raises(ValueError, match="fixed in-plane support DOF"):
        s.apply_boundary_conditions()


def test_fem3d_refuses_overlapping_node_duplicate_id_and_bad_geometry():
    s = _basic_3d_structure()
    s.add_diaphragm("D1", [3, 4], axis="Z")
    with pytest.raises(ValueError, match="Duplicate diaphragm ID"):
        s.add_diaphragm("D1", [3, 4], axis="Z")
    s.nodes[5] = Node(5, 2.5, 2.0, 3.5)
    with pytest.raises(ValueError, match="already assigned to diaphragm 'D1'"):
        s.add_diaphragm("D2", [4, 5], axis="Z")
    with pytest.raises(ValueError, match="at least 2 distinct nodes"):
        s.add_diaphragm("D_ONE", [5], axis="Z")
    with pytest.raises(ValueError, match="duplicate node ID"):
        s.add_diaphragm("D_DUP", [5, 5], axis="Z")
    with pytest.raises(ValueError, match="missing node"):
        s.add_diaphragm("D_MISS", [5, 99], axis="Z")
    with pytest.raises(ValueError, match="Unsupported diaphragm axis"):
        s.add_diaphragm("D_AXIS", [4, 5], axis="W")

    s.nodes[6] = Node(6, 5.0, 2.0, 3.8)
    with pytest.raises(ValueError, match="not coplanar.*CSI §5.4.1"):
        s.add_diaphragm("D_WARP", [5, 6], axis="Z")


def test_model_from_core_refuses_semi_rigid_diaphragm():
    model = StructuralModel(
        dimension="3D",
        vertical_axis="Z",
        units="kN, m",
        nodes=[
            ScNode(id=1, x=0.0, y=0.0, z=0.0),
            ScNode(id=2, x=5.0, y=0.0, z=0.0),
            ScNode(id=3, x=0.0, y=0.0, z=3.5),
            ScNode(id=4, x=5.0, y=0.0, z=3.5),
        ],
        supports=[Support.fixed_3d(1), Support.fixed_3d(2)],
        materials=[ConcreteMaterial(id="C30", E=25.0e6, fc=30.0e3)],
        sections=[RectangularSection(id="S1", material_id="C30", b=0.4, h=0.4)],
        elements=[
            BeamElement(id="C1", start_node=1, end_node=3, section_id="S1"),
            BeamElement(id="C2", start_node=2, end_node=4, section_id="S1"),
            BeamElement(id="B1", start_node=3, end_node=4, section_id="S1"),
        ],
        diaphragms=[
            Diaphragm.model_construct(id="D1", node_ids=[3, 4], axis="Z", rigidity="semi_rigid"),
        ],
    )
    with pytest.raises(ValueError, match="slab shells"):
        model_from_core(model)
