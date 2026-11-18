"""Until fem3d analyzes shell elements (SHELL_PLAN.md, M5), a model with shells is refused, never analyzed without
them."""

import pytest

from struct_core import ConcreteMaterial, Node, ShellElement, ShellSection, StructuralModel, Support

from fem3d.adapters.struct_core_adapter import model_from_core


def test_model_with_shells_is_refused():
    model = StructuralModel(
        units="kN, m",
        nodes=[Node(id=i + 1, x=x, y=y, z=0.0) for i, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)])],
        materials=[ConcreteMaterial(id="C", E=25e6, fc=30e3)],
        shell_sections=[ShellSection(id="S", material_id="C", thickness=0.2)],
        shell_elements=[ShellElement(id="P1", node_ids=[1, 2, 3, 4], section_id="S")],
        supports=[Support(node_id=n, ux=True, uy=True, uz=True) for n in (1, 2, 3, 4)],
    )
    with pytest.raises(NotImplementedError, match="shell elements"):
        model_from_core(model)
