"""Tests for ZX box composition."""
import pyzx as zx
import pytest
from qiskit import QuantumCircuit

from zx_motifs.pipeline.composer import (
    compose_parallel,
    compose_sequential,
    make_box_from_circuit,
    simplify_box,
)


class TestMakeBoxFromCircuit:
    def test_basic_box(self, bell_circuit):
        box = make_box_from_circuit("bell", bell_circuit)
        assert box.n_inputs == 2
        assert box.n_outputs == 2
        assert box.name == "bell"

    def test_boundary_vertices_exist(self, bell_circuit):
        box = make_box_from_circuit("bell", bell_circuit)
        vertices = set(box.graph.vertices())
        for v in box.left_boundary:
            assert v in vertices
        for v in box.right_boundary:
            assert v in vertices


class TestComposeSequential:
    def test_identity_composition(self):
        """Composing two identity-like circuits should work."""
        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc2 = QuantumCircuit(2)
        qc2.h(0)

        box1 = make_box_from_circuit("h1", qc1)
        box2 = make_box_from_circuit("h2", qc2)
        composed = compose_sequential(box1, box2)

        assert composed.n_inputs == 2
        assert composed.n_outputs == 2

    def test_boundary_mismatch_raises(self):
        qc2 = QuantumCircuit(2)
        qc3 = QuantumCircuit(3)
        box2 = make_box_from_circuit("b2", qc2)
        box3 = make_box_from_circuit("b3", qc3)
        with pytest.raises(ValueError):
            compose_sequential(box2, box3)

    def test_semantic_preservation(self):
        """Composing H then CNOT should match a Bell state circuit."""
        qc_h = QuantumCircuit(2)
        qc_h.h(0)

        qc_cx = QuantumCircuit(2)
        qc_cx.cx(0, 1)

        qc_bell = QuantumCircuit(2)
        qc_bell.h(0)
        qc_bell.cx(0, 1)

        box_h = make_box_from_circuit("hadamard", qc_h)
        box_cx = make_box_from_circuit("cnot", qc_cx)
        box_bell = make_box_from_circuit("bell", qc_bell)

        composed = compose_sequential(box_h, box_cx)
        assert zx.compare_tensors(composed.graph, box_bell.graph)

    def test_three_way_composition(self):
        """A ; B ; C should preserve semantics."""
        qc_a = QuantumCircuit(2)
        qc_a.h(0)

        qc_b = QuantumCircuit(2)
        qc_b.cx(0, 1)

        qc_c = QuantumCircuit(2)
        qc_c.h(1)

        # Full circuit for reference
        qc_full = QuantumCircuit(2)
        qc_full.h(0)
        qc_full.cx(0, 1)
        qc_full.h(1)

        box_a = make_box_from_circuit("a", qc_a)
        box_b = make_box_from_circuit("b", qc_b)
        box_c = make_box_from_circuit("c", qc_c)
        box_full = make_box_from_circuit("full", qc_full)

        composed = compose_sequential(compose_sequential(box_a, box_b), box_c)
        assert zx.compare_tensors(composed.graph, box_full.graph)


class TestComposeParallel:
    def test_parallel_doubles_wires(self, bell_circuit):
        box = make_box_from_circuit("bell", bell_circuit)
        par = compose_parallel(box, box)
        assert par.n_inputs == 4
        assert par.n_outputs == 4


class TestSimplifyBox:
    def test_spider_fusion_preserves_boundaries(self, bell_circuit):
        box = make_box_from_circuit("bell", bell_circuit)
        simplified = simplify_box(box, level="spider_fusion")
        assert simplified.n_inputs == box.n_inputs
        assert simplified.n_outputs == box.n_outputs

    def test_interior_clifford_preserves_boundaries(self, bell_circuit):
        box = make_box_from_circuit("bell", bell_circuit)
        simplified = simplify_box(box, level="interior_clifford")
        assert simplified.n_inputs == box.n_inputs
        assert simplified.n_outputs == box.n_outputs

    def test_simplification_preserves_semantics(self, bell_circuit):
        box = make_box_from_circuit("bell", bell_circuit)
        simplified = simplify_box(box, level="interior_clifford")
        assert zx.compare_tensors(box.graph, simplified.graph)

    def test_unknown_level_raises(self, bell_circuit):
        box = make_box_from_circuit("bell", bell_circuit)
        with pytest.raises(ValueError):
            simplify_box(box, level="nonexistent")
