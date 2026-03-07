"""Tests for phase polynomial extraction, arithmetic, and symmetry tags."""
from fractions import Fraction

import pytest
from qiskit import QuantumCircuit

from zx_motifs.pipeline.phase_poly import (
    PhaseGadget,
    PhasePolynomial,
    compute_symmetry_tags,
    compute_symmetry_tags_from_pauli_strings,
    extract_from_circuit,
    extract_from_networkx,
    extract_from_zx_graph,
)


# ── PhaseGadget ─────────────────────────────────────────────────────


class TestPhaseGadget:
    def test_pauli_string(self):
        g = PhaseGadget(frozenset([0, 1]), Fraction(1, 4), n_qubits=4)
        assert g.pauli_string() == "ZZII"

    def test_support_vector(self):
        g = PhaseGadget(frozenset([1, 3]), Fraction(1, 2), n_qubits=4)
        assert g.support_vector() == [0, 1, 0, 1]

    def test_weight(self):
        g = PhaseGadget(frozenset([0, 2, 3]), Fraction(1, 4), n_qubits=4)
        assert g.weight == 3

    def test_phase_class(self):
        assert PhaseGadget(frozenset([0]), Fraction(1, 4), 2).phase_class == "t_like"
        assert PhaseGadget(frozenset([0]), Fraction(1, 2), 2).phase_class == "clifford"
        assert PhaseGadget(frozenset([0]), Fraction(1, 1), 2).phase_class == "pauli"

    def test_symmetry_properties(self):
        g = PhaseGadget(frozenset([0, 1]), Fraction(1, 4), n_qubits=4)
        assert g.is_particle_number_preserving()
        assert g.is_z2_parity_preserving()
        assert g.is_real_valued()

    def test_odd_weight_not_z2_preserving(self):
        g = PhaseGadget(frozenset([0]), Fraction(1, 4), n_qubits=4)
        assert not g.is_z2_parity_preserving()


# ── PhasePolynomial ─────────────────────────────────────────────────


class TestPhasePolynomial:
    def test_add_gadget(self):
        poly = PhasePolynomial(n_qubits=4)
        poly.add_gadget(frozenset([0, 1]), Fraction(1, 4))
        assert len(poly.gadgets) == 1
        assert poly.gadgets[0].pauli_string() == "ZZII"

    def test_simplify_merges_same_support(self):
        poly = PhasePolynomial(n_qubits=4)
        poly.add_gadget(frozenset([0, 1]), Fraction(1, 4))
        poly.add_gadget(frozenset([0, 1]), Fraction(1, 4))
        simplified = poly.simplify()
        assert len(simplified.gadgets) == 1
        assert simplified.gadgets[0].phase == Fraction(1, 2)

    def test_simplify_cancels_to_zero(self):
        poly = PhasePolynomial(n_qubits=4)
        poly.add_gadget(frozenset([0, 1]), Fraction(1, 1))
        poly.add_gadget(frozenset([0, 1]), Fraction(1, 1))
        simplified = poly.simplify()
        assert len(simplified.gadgets) == 0

    def test_compose(self):
        p1 = PhasePolynomial(n_qubits=4)
        p1.add_gadget(frozenset([0, 1]), Fraction(1, 4))
        p2 = PhasePolynomial(n_qubits=4)
        p2.add_gadget(frozenset([2, 3]), Fraction(1, 4))
        composed = p1.compose(p2)
        assert len(composed.gadgets) == 2

    def test_cnot_propagation(self):
        """CNOT(c=0, t=1) transforms Z_1 -> Z_0*Z_1."""
        poly = PhasePolynomial(n_qubits=3)
        poly.add_gadget(frozenset([1]), Fraction(1, 4))
        propagated = poly.cnot_propagate(control=0, target=1)
        assert propagated.gadgets[0].support == frozenset([0, 1])

    def test_cnot_propagation_no_effect(self):
        """CNOT(c=0, t=1) does NOT change Z_2."""
        poly = PhasePolynomial(n_qubits=3)
        poly.add_gadget(frozenset([2]), Fraction(1, 4))
        propagated = poly.cnot_propagate(control=0, target=1)
        assert propagated.gadgets[0].support == frozenset([2])

    def test_t_count(self):
        poly = PhasePolynomial(n_qubits=4)
        poly.add_gadget(frozenset([0]), Fraction(1, 4))
        poly.add_gadget(frozenset([1]), Fraction(1, 2))
        poly.add_gadget(frozenset([2]), Fraction(1, 4))
        assert poly.t_count == 2

    def test_support_matrix(self):
        poly = PhasePolynomial(n_qubits=3)
        poly.add_gadget(frozenset([0, 1]), Fraction(1, 4))
        poly.add_gadget(frozenset([1, 2]), Fraction(1, 4))
        assert poly.support_matrix() == [[1, 1, 0], [0, 1, 1]]

    def test_serialization_roundtrip(self):
        poly = PhasePolynomial(n_qubits=4)
        poly.add_gadget(frozenset([0, 1]), Fraction(1, 4))
        poly.add_gadget(frozenset([2, 3]), Fraction(1, 8))
        d = poly.to_dict()
        restored = PhasePolynomial.from_dict(d)
        assert len(restored.gadgets) == 2
        assert restored.gadgets[0].support == frozenset([0, 1])
        assert restored.gadgets[0].phase == Fraction(1, 4)
        assert restored.n_qubits == 4


# ── Circuit Extraction ──────────────────────────────────────────────


class TestCircuitExtraction:
    def test_cnot_t_cnot_gives_zz_gadget(self):
        """CNOT(0,1) T(1) CNOT(0,1) = ZZ phase gadget on qubits 0,1."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.t(1)
        qc.cx(0, 1)
        poly = extract_from_circuit(qc)
        assert len(poly.gadgets) == 1
        assert poly.gadgets[0].support == frozenset([0, 1])
        assert poly.gadgets[0].phase == Fraction(1, 4)

    def test_single_t_gate(self):
        qc = QuantumCircuit(2)
        qc.t(0)
        poly = extract_from_circuit(qc)
        assert len(poly.gadgets) == 1
        assert poly.gadgets[0].support == frozenset([0])
        assert poly.gadgets[0].phase == Fraction(1, 4)

    def test_multiple_gadgets(self):
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.t(1)
        qc.cx(0, 1)
        qc.cx(2, 3)
        qc.t(3)
        qc.cx(2, 3)
        poly = extract_from_circuit(qc)
        simplified = poly.simplify()
        assert len(simplified.gadgets) == 2
        supports = {g.support for g in simplified.gadgets}
        assert frozenset([0, 1]) in supports
        assert frozenset([2, 3]) in supports


# ── ZX Graph Extraction ─────────────────────────────────────────────


class TestZXExtraction:
    def test_extracts_nonzero_gadgets(self):
        """ZX extraction should find phase gadgets from simplified graph."""
        import copy
        import pyzx as zx
        from zx_motifs.pipeline.converter import qiskit_to_zx

        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.t(1)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.t(2)
        qc.cx(1, 2)

        zx_circ = qiskit_to_zx(qc)
        g = zx_circ.to_graph()
        g2 = copy.deepcopy(g)
        zx.simplify.full_reduce(g2)

        poly = extract_from_zx_graph(g2)
        assert len(poly.gadgets) > 0
        assert all(g.phase != 0 for g in poly.gadgets)


# ── NetworkX Extraction ─────────────────────────────────────────────


class TestNetworkXExtraction:
    def test_extracts_from_featurized_graph(self):
        """Should find gadgets in a featurized ZX graph."""
        import networkx as nx

        nxg = nx.Graph()
        nxg.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
        nxg.add_node(1, vertex_type="Z", phase_class="t_like", is_boundary=False)
        nxg.add_node(2, vertex_type="Z", phase_class="zero", is_boundary=False)
        nxg.add_edge(0, 1, edge_type="HADAMARD")
        nxg.add_edge(1, 2, edge_type="HADAMARD")

        gadgets = extract_from_networkx(nxg)
        assert len(gadgets) == 1
        assert gadgets[0].support == frozenset([0, 2])

    def test_skips_boundary_nodes(self):
        import networkx as nx

        nxg = nx.Graph()
        nxg.add_node(0, vertex_type="BOUNDARY", phase_class="zero", is_boundary=True)
        nxg.add_node(1, vertex_type="Z", phase_class="t_like", is_boundary=False)
        nxg.add_edge(0, 1, edge_type="SIMPLE")

        gadgets = extract_from_networkx(nxg)
        assert len(gadgets) == 1
        assert gadgets[0].support == frozenset([1])


# ── Symmetry Tags ──────────────────────────────────────────────────


class TestSymmetryTags:
    def test_all_even_weight(self):
        poly = PhasePolynomial(n_qubits=4)
        poly.add_gadget(frozenset([0, 1]), Fraction(1, 4))
        poly.add_gadget(frozenset([2, 3]), Fraction(1, 4))
        tags = compute_symmetry_tags(poly)
        assert tags["z2_parity_preserving"]
        assert tags["particle_number_preserving"]
        assert tags["real_valued"]

    def test_odd_weight_breaks_z2(self):
        poly = PhasePolynomial(n_qubits=4)
        poly.add_gadget(frozenset([0]), Fraction(1, 4))
        tags = compute_symmetry_tags(poly)
        assert not tags["z2_parity_preserving"]

    def test_pauli_string_symmetry_xy(self):
        tags = compute_symmetry_tags_from_pauli_strings(["XYZZ"])
        assert tags["particle_number_preserving"]  # 1 X, 1 Y
        assert tags["z2_parity_preserving"]  # weight 4

    def test_pauli_string_y_breaks_real(self):
        tags = compute_symmetry_tags_from_pauli_strings(["YIII"])
        assert not tags["real_valued"]
        assert not tags["particle_number_preserving"]
