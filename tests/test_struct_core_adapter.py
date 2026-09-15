"""
Unit tests for struct_core adapter with fem3d.
"""

import pytest
from fem3d import SimpleFrame, Results

try:
    import struct_core
    from fem3d.adapters.struct_core_adapter import model_from_core, model_to_core, result_to_core
    HAS_STRUCT_CORE = True
except ImportError:
    HAS_STRUCT_CORE = False


@pytest.mark.skipif(not HAS_STRUCT_CORE, reason="struct_core not installed")
def test_struct_core_adapter_roundtrip():
    frame = SimpleFrame()
    frame.add_node(1, 0.0, 0.0, 0.0)
    frame.add_node(2, 100.0, 0.0, 0.0)

    frame.add_frame(1, 1, 2, E=29000.0, A=10.0, Iy=50.0, Iz=100.0, J=30.0)
    frame.add_support(1, [1, 1, 1, 1, 1, 1])
    frame.add_node_load(2, [10.0, 5.0, -2.0, 0.0, 0.0, 0.0])

    disp, reac = frame.solve()

    # Convert to struct_core model
    sc_model = model_to_core(frame)
    assert len(sc_model.nodes) == 2
    assert len(sc_model.elements) == 1

    # Convert back from struct_core model
    rebuilt_struct = model_from_core(sc_model)
    assert len(rebuilt_struct.nodes) == 2
    assert len(rebuilt_struct.elements) == 1

    # Convert analysis results to struct_core.AnalysisResult
    results = Results(frame)
    sc_result = result_to_core(results)
    assert len(sc_result.node_results) == 2
    assert len(sc_result.element_results) == 1
    # Node 1 is the fixed support: all displacements are zero.
    # Node 2 is the free tip with applied loads [Fx=10, Fy=5, Fz=-2]:
    #   ux (axial) at node 2 should be nonzero.
    assert sc_result.node_results[2].displacement.ux != 0.0
    # Node 1 should have nonzero reactions (it's the fixed support).
    assert sc_result.node_results[1].reaction is not None
    assert sc_result.node_results[1].reaction.fx != 0.0
