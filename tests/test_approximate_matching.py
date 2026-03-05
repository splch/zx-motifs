"""Tests for feature vectors, pre-filtering, and approximate matching."""
import numpy as np
import networkx as nx
import pytest

from zx_motifs.pipeline.featurizer import (
    compute_motif_feature_vector,
    motif_similarity,
)
from zx_motifs.pipeline.matcher import (
    ApproximateMatch,
    can_possibly_match,
    find_approximate_matches,
    find_motif_in_graph,
)
from zx_motifs.pipeline.motif_generators import (
    make_cx_spider_motif,
    make_zz_interaction_motif,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_simple_host():
    """A small host graph with Z/X spiders and both edge types."""
    g = nx.Graph()
    g.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(1, vertex_type="X", phase_class="zero", is_boundary=False)
    g.add_node(2, vertex_type="Z", phase_class="t_like", is_boundary=False)
    g.add_node(3, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(10, vertex_type="BOUNDARY", phase_class="zero", is_boundary=True)
    g.add_edge(0, 1, edge_type="SIMPLE")
    g.add_edge(1, 2, edge_type="SIMPLE")
    g.add_edge(2, 3, edge_type="HADAMARD")
    g.add_edge(10, 0, edge_type="SIMPLE")
    return g


def _make_zz_like_host():
    """Host containing both exact and near-miss ZZ interaction patterns."""
    g = nx.Graph()
    # Exact ZZ: Z-zero → Z-arbitrary → Z-zero (simple edges)
    g.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(1, vertex_type="Z", phase_class="arbitrary", is_boundary=False)
    g.add_node(2, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_edge(0, 1, edge_type="SIMPLE")
    g.add_edge(1, 2, edge_type="SIMPLE")
    # Near-miss: Z-zero → Z-t_like → Z-zero (single phase mismatch)
    g.add_node(3, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(4, vertex_type="Z", phase_class="t_like", is_boundary=False)
    g.add_node(5, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_edge(3, 4, edge_type="SIMPLE")
    g.add_edge(4, 5, edge_type="SIMPLE")
    return g


# ── TestCanPossiblyMatch ────────────────────────────────────────────


class TestCanPossiblyMatch:
    def test_true_when_containable(self):
        """Pattern is a subset of host's capabilities."""
        host = _make_simple_host()
        pattern = make_cx_spider_motif().graph
        assert can_possibly_match(pattern, host) is True

    def test_false_on_type_count_violation(self):
        """Pattern needs more X-spiders than host has."""
        host = nx.Graph()
        host.add_node(0, vertex_type="Z", phase_class="zero")
        host.add_node(1, vertex_type="Z", phase_class="zero")
        host.add_edge(0, 1, edge_type="SIMPLE")

        pattern = nx.Graph()
        pattern.add_node(0, vertex_type="X", phase_class="zero")
        pattern.add_node(1, vertex_type="X", phase_class="zero")
        pattern.add_edge(0, 1, edge_type="SIMPLE")
        assert can_possibly_match(pattern, host) is False

    def test_false_on_edge_type_violation(self):
        """Pattern needs HADAMARD edges but host has none."""
        host = nx.Graph()
        host.add_node(0, vertex_type="Z", phase_class="zero")
        host.add_node(1, vertex_type="Z", phase_class="zero")
        host.add_edge(0, 1, edge_type="SIMPLE")

        pattern = nx.Graph()
        pattern.add_node(0, vertex_type="Z", phase_class="zero")
        pattern.add_node(1, vertex_type="Z", phase_class="zero")
        pattern.add_edge(0, 1, edge_type="HADAMARD")
        assert can_possibly_match(pattern, host) is False

    def test_false_on_degree_violation(self):
        """Pattern has higher max degree than host."""
        host = nx.Graph()
        host.add_node(0, vertex_type="Z", phase_class="zero")
        host.add_node(1, vertex_type="Z", phase_class="zero")
        host.add_edge(0, 1, edge_type="SIMPLE")

        # Star pattern: center has degree 3
        pattern = nx.Graph()
        pattern.add_node(0, vertex_type="Z", phase_class="zero")
        pattern.add_node(1, vertex_type="Z", phase_class="zero")
        pattern.add_node(2, vertex_type="Z", phase_class="zero")
        pattern.add_node(3, vertex_type="Z", phase_class="zero")
        pattern.add_edge(0, 1, edge_type="SIMPLE")
        pattern.add_edge(0, 2, edge_type="SIMPLE")
        pattern.add_edge(0, 3, edge_type="SIMPLE")
        assert can_possibly_match(pattern, host) is False

    def test_empty_pattern_always_matches(self):
        """An empty pattern can match anything."""
        host = _make_simple_host()
        pattern = nx.Graph()
        assert can_possibly_match(pattern, host) is True


# ── TestMotifFeatureVector ──────────────────────────────────────────


class TestMotifFeatureVector:
    def test_length_12(self):
        """Feature vector has exactly 12 elements."""
        g = make_cx_spider_motif().graph
        vec = compute_motif_feature_vector(g)
        assert len(vec) == 12

    def test_identical_graphs_identical_vectors(self):
        """Same graph produces same vector."""
        g = make_cx_spider_motif().graph
        v1 = compute_motif_feature_vector(g)
        v2 = compute_motif_feature_vector(g)
        np.testing.assert_array_equal(v1, v2)

    def test_different_graphs_different_vectors(self):
        """Different motifs produce different vectors."""
        v1 = compute_motif_feature_vector(make_cx_spider_motif().graph)
        v2 = compute_motif_feature_vector(make_zz_interaction_motif().graph)
        assert not np.array_equal(v1, v2)

    def test_values_are_nonnegative(self):
        """All feature values are non-negative."""
        g = make_zz_interaction_motif().graph
        vec = compute_motif_feature_vector(g)
        assert np.all(vec >= 0)

    def test_empty_graph(self):
        """Empty graph has mostly-zero vector."""
        g = nx.Graph()
        vec = compute_motif_feature_vector(g)
        assert len(vec) == 12
        assert vec[0] == 0  # n_nodes
        assert vec[1] == 0  # n_edges


# ── TestMotifSimilarity ─────────────────────────────────────────────


class TestMotifSimilarity:
    def test_identical_is_one(self):
        """Identical vectors have similarity 1.0."""
        v = compute_motif_feature_vector(make_cx_spider_motif().graph)
        assert motif_similarity(v, v) == pytest.approx(1.0)

    def test_empty_is_zero(self):
        """Zero vector has similarity 0.0."""
        v = compute_motif_feature_vector(make_cx_spider_motif().graph)
        z = np.zeros(12)
        assert motif_similarity(v, z) == 0.0
        assert motif_similarity(z, z) == 0.0

    def test_symmetric(self):
        """Similarity is symmetric."""
        v1 = compute_motif_feature_vector(make_cx_spider_motif().graph)
        v2 = compute_motif_feature_vector(make_zz_interaction_motif().graph)
        assert motif_similarity(v1, v2) == pytest.approx(motif_similarity(v2, v1))

    def test_in_unit_range(self):
        """Cosine similarity of non-negative vectors is in [0, 1]."""
        v1 = compute_motif_feature_vector(make_cx_spider_motif().graph)
        v2 = compute_motif_feature_vector(make_zz_interaction_motif().graph)
        sim = motif_similarity(v1, v2)
        assert 0.0 <= sim <= 1.0


# ── TestApproximateMatching ─────────────────────────────────────────


class TestApproximateMatching:
    def test_exact_match_is_distance_zero(self):
        """Exact subgraph match has edit_distance=0."""
        host = _make_zz_like_host()
        pattern = make_zz_interaction_motif().graph
        approx = find_approximate_matches(
            pattern, host, max_edit_distance=0, exclude_boundary=False,
        )
        assert len(approx) >= 1
        assert all(a.edit_distance == 0 for a in approx)

    def test_single_phase_mismatch(self):
        """Near-miss with one phase difference has distance 1."""
        host = _make_zz_like_host()
        pattern = make_zz_interaction_motif().graph
        approx = find_approximate_matches(
            pattern, host, max_edit_distance=1, exclude_boundary=False,
        )
        # Should find both exact (dist=0) and near-miss (dist=1)
        distances = {a.edit_distance for a in approx}
        assert 0 in distances
        assert 1 in distances

    def test_respects_max_edit_distance(self):
        """Matches beyond max_edit_distance are excluded."""
        host = _make_zz_like_host()
        pattern = make_zz_interaction_motif().graph
        exact = find_approximate_matches(
            pattern, host, max_edit_distance=0, exclude_boundary=False,
        )
        relaxed = find_approximate_matches(
            pattern, host, max_edit_distance=2, exclude_boundary=False,
        )
        assert len(relaxed) >= len(exact)

    def test_similarity_score_range(self):
        """Similarity scores are in [0, 1]."""
        host = _make_zz_like_host()
        pattern = make_zz_interaction_motif().graph
        approx = find_approximate_matches(
            pattern, host, max_edit_distance=2, exclude_boundary=False,
        )
        for a in approx:
            assert 0.0 <= a.similarity_score <= 1.0

    def test_exact_match_has_perfect_similarity(self):
        """Exact match has similarity_score=1.0."""
        host = _make_zz_like_host()
        pattern = make_zz_interaction_motif().graph
        approx = find_approximate_matches(
            pattern, host, max_edit_distance=0, exclude_boundary=False,
        )
        for a in approx:
            assert a.similarity_score == pytest.approx(1.0)

    def test_prefilter_integration(self):
        """Pre-filter doesn't break existing exact matching."""
        host = _make_simple_host()
        pattern = make_cx_spider_motif().graph
        matches = find_motif_in_graph(pattern, host)
        assert len(matches) >= 1
