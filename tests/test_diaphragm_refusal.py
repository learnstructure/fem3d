"""Temporary Milestone 1 refusal until fem3d implements rigid diaphragm constraints in Milestone 2."""

import pytest

from struct_core import (
    BeamElement,
    ConcreteMaterial,
    Diaphragm,
    Node,
    RectangularSection,
    StructuralModel,
    Support,
)

from fem3d.adapters.struct_core_adapter import model_from_core


def test_fem3d_temporarily_refuses_diaphragms_until_m2():
    model = StructuralModel(
        dimension="3D",
        vertical_axis="Z",
        units="kN, m",
        nodes=[
            Node(id=1, x=0.0, y=0.0, z=0.0),
            Node(id=2, x=5.0, y=0.0, z=0.0),
            Node(id=3, x=0.0, y=0.0, z=3.5),
            Node(id=4, x=5.0, y=0.0, z=3.5),
        ],
        supports=[Support.fixed_3d(1), Support.fixed_3d(2)],
        materials=[ConcreteMaterial(id="C30", E=25.0e6, fc=30.0e3)],
        sections=[RectangularSection(id="S1", material_id="C30", b=0.4, h=0.4)],
        elements=[
            BeamElement(id="C1", start_node=1, end_node=3, section_id="S1"),
            BeamElement(id="C2", start_node=2, end_node=4, section_id="S1"),
            BeamElement(id="B1", start_node=3, end_node=4, section_id="S1"),
        ],
        diaphragms=[Diaphragm(id="D1", node_ids=[3, 4], axis="Z")],
    )
    with pytest.raises(ValueError, match="does not support diaphragms yet"):
        model_from_core(model)
