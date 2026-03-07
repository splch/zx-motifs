"""Tests for interaction graph construction."""
from fractions import Fraction

from zx_motifs.pipeline.interaction import (
    build_interaction_graph,
    compare_interaction_graphs,
    interaction_degree_sequence,
    interaction_density,
    max_interaction_weight,
)
from zx_motifs.pipeline.phase_poly import PhasePolynomial


class TestInteractionGraph:
    def test_basic_construction(self):
        poly = PhasePolynomial(n_qubits=4)
        poly.add_gadget(frozenset([0, 1]), Fraction(1, 4))
        poly.add_gadget(frozenset([1, 2]), Fraction(1, 4))
        g = build_interaction_graph(poly)
        assert set(g.nodes()) == {0, 1, 2, 3}
        assert g.has_edge(0, 1)
        assert g.has_edge(1, 2)
        assert not g.has_edge(0, 2)
        assert not g.has_edge(2, 3)

    def test_edge_weights(self):
        poly = PhasePolynomial(n_qubits=3)
        poly.add_gadget(frozenset([0, 1]), Fraction(1, 4))
        poly.add_gadget(frozenset([0, 1]), Fraction(1, 2))
        g = build_interaction_graph(poly)
        assert g[0][1]["weight"] == 2

    def test_three_qubit_gadget(self):
        poly = PhasePolynomial(n_qubits=4)
        poly.add_gadget(frozenset([0, 1, 2]), Fraction(1, 4))
        g = build_interaction_graph(poly)
        assert g.has_edge(0, 1)
        assert g.has_edge(0, 2)
        assert g.has_edge(1, 2)

    def test_single_qubit_ignored(self):
        poly = PhasePolynomial(n_qubits=3)
        poly.add_gadget(frozenset([0]), Fraction(1, 4))
        g = build_interaction_graph(poly)
        assert g.number_of_edges() == 0

    def test_empty_polynomial(self):
        poly = PhasePolynomial(n_qubits=4)
        g = build_interaction_graph(poly)
        assert g.number_of_nodes() == 4
        assert g.number_of_edges() == 0


class TestInteractionMetrics:
    def test_density(self):
        poly = PhasePolynomial(n_qubits=4)
        poly.add_gadget(frozenset([0, 1]), Fraction(1, 4))
        poly.add_gadget(frozenset([1, 2]), Fraction(1, 4))
        poly.add_gadget(frozenset([2, 3]), Fraction(1, 4))
        g = build_interaction_graph(poly)
        assert interaction_density(g) == 0.5

    def test_max_weight(self):
        poly = PhasePolynomial(n_qubits=3)
        poly.add_gadget(frozenset([0, 1]), Fraction(1, 4))
        poly.add_gadget(frozenset([0, 1]), Fraction(1, 2))
        poly.add_gadget(frozenset([1, 2]), Fraction(1, 4))
        g = build_interaction_graph(poly)
        assert max_interaction_weight(g) == 2

    def test_degree_sequence(self):
        poly = PhasePolynomial(n_qubits=3)
        poly.add_gadget(frozenset([0, 1]), Fraction(1, 4))
        poly.add_gadget(frozenset([1, 2]), Fraction(1, 4))
        g = build_interaction_graph(poly)
        degrees = interaction_degree_sequence(g)
        assert degrees == [1, 2, 1]

    def test_compare_identical(self):
        poly = PhasePolynomial(n_qubits=3)
        poly.add_gadget(frozenset([0, 1]), Fraction(1, 4))
        g1 = build_interaction_graph(poly)
        g2 = build_interaction_graph(poly)
        assert compare_interaction_graphs(g1, g2) == 1.0

    def test_compare_disjoint(self):
        p1 = PhasePolynomial(n_qubits=4)
        p1.add_gadget(frozenset([0, 1]), Fraction(1, 4))
        p2 = PhasePolynomial(n_qubits=4)
        p2.add_gadget(frozenset([2, 3]), Fraction(1, 4))
        g1 = build_interaction_graph(p1)
        g2 = build_interaction_graph(p2)
        assert compare_interaction_graphs(g1, g2) == 0.0
