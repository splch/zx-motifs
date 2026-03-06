"""Tests for circuit-to-ZX conversion pipeline."""
import numpy as np
import pyzx as zx

from zx_motifs.pipeline.converter import (
    SimplificationLevel,
    convert_at_all_levels,
    count_t_gates,
    qiskit_to_zx,
)


class TestQiskitToZx:
    def test_bell_conversion(self, bell_circuit):
        zx_circ = qiskit_to_zx(bell_circuit)
        g = zx_circ.to_graph()
        assert g.num_vertices() > 0
        assert g.num_edges() > 0

    def test_roundtrip_semantics(self, bell_circuit):
        """Verify the ZX diagram represents the same unitary as the circuit."""
        zx_circ = qiskit_to_zx(bell_circuit)
        g = zx_circ.to_graph()
        # Compare original circuit tensor with the graph tensor
        assert zx.compare_tensors(zx_circ, g)

    def test_ghz_conversion(self, ghz3_circuit):
        zx_circ = qiskit_to_zx(ghz3_circuit)
        g = zx_circ.to_graph()
        # 3 inputs + 3 outputs + internal gates
        assert g.num_vertices() >= 6


class TestConvertAtAllLevels:
    def test_produces_all_levels(self, bell_circuit):
        snapshots = convert_at_all_levels(bell_circuit, "bell_q2")
        assert len(snapshots) == 6
        levels = {s.level for s in snapshots}
        assert levels == set(SimplificationLevel)

    def test_simplification_reduces_vertices(self, ghz3_circuit):
        snapshots = convert_at_all_levels(ghz3_circuit, "ghz_q3")
        raw = next(s for s in snapshots if s.level == SimplificationLevel.RAW)
        full = next(s for s in snapshots if s.level == SimplificationLevel.FULL_REDUCE)
        # Full reduce should have equal or fewer vertices than raw
        assert full.num_vertices <= raw.num_vertices

    def test_snapshot_metadata(self, bell_circuit):
        snapshots = convert_at_all_levels(bell_circuit, "bell_q2")
        for s in snapshots:
            assert s.algorithm_name == "bell_q2"
            assert s.n_qubits == 2
            assert s.num_vertices > 0
            d = s.to_dict()
            assert "algorithm" in d
            assert "level" in d

    def test_all_levels_preserve_semantics(self, bell_circuit):
        """Every simplification level should preserve the tensor."""
        snapshots = convert_at_all_levels(bell_circuit, "bell_q2")
        raw_graph = snapshots[0].graph
        for s in snapshots[1:]:
            assert zx.compare_tensors(raw_graph, s.graph), (
                f"Level {s.level.value} changed the semantics"
            )


class TestCountTGates:
    def test_no_t_gates(self, bell_zx_graph):
        assert count_t_gates(bell_zx_graph) == 0

    def test_with_t_gates(self, t_gate_circuit):
        zx_circ = qiskit_to_zx(t_gate_circuit)
        g = zx_circ.to_graph()
        assert count_t_gates(g) >= 1
