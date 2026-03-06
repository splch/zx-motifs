"""Tests for phase classification and ZX-to-NetworkX conversion."""
from fractions import Fraction

import networkx as nx

from zx_motifs.pipeline.featurizer import (
    classify_phase,
    compute_graph_features,
    pyzx_to_networkx,
)


class TestClassifyPhase:
    def test_zero(self):
        assert classify_phase(Fraction(0)) == "zero"

    def test_pauli_pi(self):
        assert classify_phase(Fraction(1, 1)) == "pauli"

    def test_clifford_pi_over_2(self):
        assert classify_phase(Fraction(1, 2)) == "clifford"

    def test_clifford_3pi_over_2(self):
        assert classify_phase(Fraction(3, 2)) == "clifford"

    def test_t_like_pi_over_4(self):
        assert classify_phase(Fraction(1, 4)) == "t_like"

    def test_t_like_7pi_over_4(self):
        assert classify_phase(Fraction(7, 4)) == "t_like"

    def test_t_like_3pi_over_4(self):
        assert classify_phase(Fraction(3, 4)) == "t_like"

    def test_arbitrary(self):
        assert classify_phase(Fraction(1, 3)) == "arbitrary"
        assert classify_phase(Fraction(1, 5)) == "arbitrary"

    def test_all_classes_are_distinct(self):
        """Verify the five classes don't overlap."""
        classes = {
            classify_phase(Fraction(0)),
            classify_phase(Fraction(1, 1)),
            classify_phase(Fraction(1, 2)),
            classify_phase(Fraction(1, 4)),
            classify_phase(Fraction(1, 3)),
        }
        assert classes == {"zero", "pauli", "clifford", "t_like", "arbitrary"}


class TestPyzxToNetworkx:
    def test_basic_conversion(self, bell_zx_graph):
        nxg = pyzx_to_networkx(bell_zx_graph)
        assert isinstance(nxg, nx.Graph)
        assert nxg.number_of_nodes() == bell_zx_graph.num_vertices()
        assert nxg.number_of_edges() == bell_zx_graph.num_edges()

    def test_node_attributes_present(self, bell_zx_graph):
        nxg = pyzx_to_networkx(bell_zx_graph)
        for _, data in nxg.nodes(data=True):
            assert "vertex_type" in data
            assert "phase" in data
            assert "phase_class" in data
            assert "is_boundary" in data
            assert "label" in data

    def test_edge_attributes_present(self, bell_zx_graph):
        nxg = pyzx_to_networkx(bell_zx_graph)
        for _, _, data in nxg.edges(data=True):
            assert "edge_type" in data
            assert data["edge_type"] in ("SIMPLE", "HADAMARD")

    def test_boundary_nodes_marked(self, bell_zx_graph):
        nxg = pyzx_to_networkx(bell_zx_graph)
        boundary_count = sum(
            1 for _, d in nxg.nodes(data=True) if d["is_boundary"]
        )
        # Bell circuit: 2 inputs + 2 outputs = 4 boundary nodes
        assert boundary_count == 4

    def test_coarsen_phases(self, bell_zx_graph):
        nxg_exact = pyzx_to_networkx(bell_zx_graph, coarsen_phases=False)
        nxg_coarse = pyzx_to_networkx(bell_zx_graph, coarsen_phases=True)
        # Both should have the same number of nodes
        assert nxg_exact.number_of_nodes() == nxg_coarse.number_of_nodes()


class TestComputeGraphFeatures:
    def test_empty_graph(self):
        nxg = nx.Graph()
        feats = compute_graph_features(nxg)
        assert feats["n_nodes"] == 0

    def test_basic_features(self, bell_zx_graph):
        nxg = pyzx_to_networkx(bell_zx_graph)
        feats = compute_graph_features(nxg)
        assert feats["n_nodes"] > 0
        assert feats["n_edges"] > 0
        assert "hadamard_ratio" in feats
        assert "avg_degree" in feats
        assert "type_counts" in feats
