"""Shell elements that fem3d does not formulate yet are refused with the reason, never analyzed approximately:
triangles (planned) and thin-plate (Kirchhoff) sections (DKQ, planned). fem3d/SHELL_PLAN.md, M5."""

import pytest

from struct_core import ConcreteMaterial, Node, ShellElement, ShellSection, StructuralModel, Support

from fem3d import ElasticMaterial, Node as FemNode
from fem3d.adapters.struct_core_adapter import model_from_core
from fem3d.elements.shell import ShellElement as FemShellElement


def _model(node_ids, **section):
    return StructuralModel(
        units="kN, m",
        nodes=[Node(id=i + 1, x=x, y=y, z=0.0) for i, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)])],
        materials=[ConcreteMaterial(id="C", E=25e6, fc=30e3)],
        shell_sections=[ShellSection(id="S", material_id="C", thickness=0.2, **section)],
        shell_elements=[ShellElement(id="P1", node_ids=node_ids, section_id="S")],
        supports=[Support(node_id=n, ux=True, uy=True, uz=True) for n in (1, 2, 3, 4)],
    )


def test_triangular_shells_are_refused():
    with pytest.raises(NotImplementedError, match="four-node shell elements only"):
        model_from_core(_model([1, 2, 3]))
    nodes = [FemNode(nid=i, x=x, y=y, z=0.0) for i, (x, y) in enumerate([(0, 0), (1, 0), (0, 1)])]
    with pytest.raises(ValueError, match="four-node shells only"):
        FemShellElement(1, nodes, ElasticMaterial(E=1.0), 0.1)


def test_thin_plate_sections_are_refused_but_thin_membranes_are_not():
    with pytest.raises(NotImplementedError, match="thin-plate"):
        model_from_core(_model([1, 2, 3, 4], plate_theory="thin"))
    structure = model_from_core(_model([1, 2, 3, 4], plate_theory="thin", behavior="membrane"))
    assert list(structure.shells) == ["P1"]
