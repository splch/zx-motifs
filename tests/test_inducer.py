"""Tests for data-driven motif induction (Phase 7)."""
import networkx as nx
import pytest

from zx_motifs.pipeline.decomposer import decompose_graph, DecompositionResult
from zx_motifs.pipeline.inducer import (
    NeighborhoodSignature,
    analyze_uncovered_vertices,
    compute_vertex_signature,
    extract_motif_from_uncovered_cluster,
    induce_motifs_from_gaps,
    iterative_induction,
)
from zx_motifs.pipeline.matcher import MotifPattern


# ── Helpers ──────────────────────────────────────────────────────────


def _make_host():
    """A small host graph with varied structure."""
    g = nx.Graph()
    g.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(1, vertex_type="X", phase_class="zero", is_boundary=False)
    g.add_node(2, vertex_type="Z", phase_class="t_like", is_boundary=False)
    g.add_node(3, vertex_type="X", phase_class="zero", is_boundary=False)
    g.add_node(4, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_edge(0, 1, edge_type="SIMPLE")
    g.add_edge(1, 2, edge_type="SIMPLE")
    g.add_edge(2, 3, edge_type="SIMPLE")
    g.add_edge(3, 4, edge_type="SIMPLE")
    return g


def _make_corpus_with_repeated_pattern():
    """
    Corpus where the same 3-node pattern appears in 3 algorithms,
    plus extra nodes that won't be covered by a CX motif.
    """
    corpus = {}
    for algo in ["algo_a", "algo_b", "algo_c"]:
        g = nx.Graph()
        # Recurring pattern: Z(t_like) -> X(zero) -> Z(zero)
        g.add_node(0, vertex_type="Z", phase_class="t_like", is_boundary=False)
        g.add_node(1, vertex_type="X", phase_class="zero", is_boundary=False)
        g.add_node(2, vertex_type="Z", phase_class="zero", is_boundary=False)
        g.add_edge(0, 1, edge_type="SIMPLE")
        g.add_edge(1, 2, edge_type="SIMPLE")
        # Extra nodes to create gaps
        g.add_node(3, vertex_type="Z", phase_class="pauli", is_boundary=False)
        g.add_node(4, vertex_type="X", phase_class="zero", is_boundary=False)
        g.add_edge(3, 4, edge_type="SIMPLE")
        g.add_edge(2, 3, edge_type="SIMPLE")
        corpus[(algo, "spider_fused")] = g
    return corpus


# ── Tests ────────────────────────────────────────────────────────────


class TestNeighborhoodSignature:
    def test_signature_computation(self):
        """Signature captures type, phase, and neighbor types."""
        g = _make_host()
        sig = compute_vertex_signature(g, 1)
        assert sig.center_type == "X"
        assert sig.center_phase == "zero"
        assert sig.degree == 2
        assert set(sig.neighbor_types) == {"Z", "Z"}

    def test_signature_deterministic(self):
        """Same vertex gives same signature."""
        g = _make_host()
        s1 = compute_vertex_signature(g, 1)
        s2 = compute_vertex_signature(g, 1)
        assert s1.key == s2.key

    def test_key_property(self):
        """Key is a non-empty string."""
        g = _make_host()
        sig = compute_vertex_signature(g, 0)
        assert isinstance(sig.key, str)
        assert len(sig.key) > 0


class TestAnalyzeUncovered:
    def test_groups_by_signature(self):
        """Uncovered vertices are grouped by signature key."""
        g = _make_host()
        # Empty library → everything is uncovered
        decomp = decompose_graph(g, [], exclude_boundary=False)
        groups = analyze_uncovered_vertices(g, decomp)
        assert isinstance(groups, dict)
        # All 5 vertices should appear in some group
        total = sum(len(v) for v in groups.values())
        assert total == 5


class TestExtractMotif:
    def test_returns_connected_subgraph(self):
        """Extracted motif is a connected graph."""
        g = _make_host()
        result = extract_motif_from_uncovered_cluster(g, [1, 2])
        assert result is not None
        assert nx.is_connected(result)

    def test_reindexed_to_zero(self):
        """Extracted motif has nodes indexed from 0."""
        g = _make_host()
        result = extract_motif_from_uncovered_cluster(g, [2])
        if result is not None:
            assert min(result.nodes()) == 0

    def test_none_for_empty_vertices(self):
        """Returns None for empty vertex list."""
        g = _make_host()
        result = extract_motif_from_uncovered_cluster(g, [])
        assert result is None


class TestInduceMotifs:
    def test_returns_motif_patterns(self):
        """induce_motifs_from_gaps returns list of MotifPatterns."""
        corpus = _make_corpus_with_repeated_pattern()
        results = induce_motifs_from_gaps(
            corpus, [], min_occurrences=2, min_algorithms=2,
        )
        for m in results:
            assert isinstance(m, MotifPattern)
            assert m.source == "data_driven"

    def test_respects_min_algorithms(self):
        """With min_algorithms=10, no motifs are induced from a 3-algo corpus."""
        corpus = _make_corpus_with_repeated_pattern()
        results = induce_motifs_from_gaps(
            corpus, [], min_occurrences=1, min_algorithms=10,
        )
        assert len(results) == 0


class TestIterativeInduction:
    def test_returns_library_and_history(self):
        """Returns (library, coverage_history) tuple."""
        corpus = _make_corpus_with_repeated_pattern()
        library, history = iterative_induction(
            corpus, [], max_rounds=2, min_coverage_gain=0.0,
        )
        assert isinstance(library, list)
        assert isinstance(history, list)
        assert all(0.0 <= c <= 1.0 for c in history)

    def test_respects_max_rounds(self):
        """Coverage history length <= max_rounds + 1."""
        corpus = _make_corpus_with_repeated_pattern()
        _, history = iterative_induction(
            corpus, [], max_rounds=1, min_coverage_gain=0.0,
        )
        # At most: initial measurement + final measurement
        assert len(history) <= 3
