"""Tests for motif library optimization (Phase 8)."""
import networkx as nx
import pytest

from zx_motifs.pipeline.matcher import MotifPattern
from zx_motifs.pipeline.motif_generators import (
    make_cx_spider_motif,
    make_zz_interaction_motif,
)
from zx_motifs.pipeline.optimizer import (
    MotifScore,
    OptimizedLibrary,
    compute_motif_coverage_map,
    optimize_library,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_corpus():
    """Corpus with 2 algorithms containing cx_pair and zz_interaction patterns."""
    corpus = {}
    for algo in ["algo_a", "algo_b"]:
        g = nx.Graph()
        # CX pair pattern: Z-X
        g.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
        g.add_node(1, vertex_type="X", phase_class="zero", is_boundary=False)
        g.add_edge(0, 1, edge_type="SIMPLE")
        # ZZ interaction pattern: Z-Z(arb)-Z
        g.add_node(2, vertex_type="Z", phase_class="zero", is_boundary=False)
        g.add_node(3, vertex_type="Z", phase_class="arbitrary", is_boundary=False)
        g.add_node(4, vertex_type="Z", phase_class="zero", is_boundary=False)
        g.add_edge(2, 3, edge_type="SIMPLE")
        g.add_edge(3, 4, edge_type="SIMPLE")
        # Extra node
        g.add_node(5, vertex_type="Z", phase_class="t_like", is_boundary=False)
        g.add_edge(4, 5, edge_type="SIMPLE")
        corpus[(algo, "spider_fused")] = g
    return corpus


# ── Tests ────────────────────────────────────────────────────────────


class TestCoverageMap:
    def test_finds_matches(self):
        """Coverage map finds matches for known motifs."""
        corpus = _make_corpus()
        cx = make_cx_spider_motif()
        cmap = compute_motif_coverage_map(cx, corpus)
        assert len(cmap) >= 1
        for algo, verts in cmap.items():
            assert isinstance(verts, set)
            assert len(verts) >= 2

    def test_empty_corpus_returns_empty(self):
        """Empty corpus returns empty coverage map."""
        cx = make_cx_spider_motif()
        cmap = compute_motif_coverage_map(cx, {})
        assert cmap == {}

    def test_no_match_returns_empty(self):
        """Motif with no matches returns empty coverage map."""
        g = nx.Graph()
        g.add_node(0, vertex_type="H_BOX", phase_class="zero", is_boundary=False)
        g.add_node(1, vertex_type="H_BOX", phase_class="zero", is_boundary=False)
        g.add_edge(0, 1, edge_type="HADAMARD")
        motif = MotifPattern(motif_id="nonexistent", graph=g, source="test")
        corpus = _make_corpus()
        cmap = compute_motif_coverage_map(motif, corpus)
        assert cmap == {}


class TestOptimizeLibrary:
    def test_selects_useful_motifs(self):
        """Optimization selects motifs that contribute coverage."""
        corpus = _make_corpus()
        candidates = [make_cx_spider_motif(), make_zz_interaction_motif()]
        result = optimize_library(candidates, corpus)
        assert isinstance(result, OptimizedLibrary)
        assert len(result.selected_motifs) >= 1

    def test_respects_max_library_size(self):
        """Selected motifs don't exceed max_library_size."""
        corpus = _make_corpus()
        candidates = [make_cx_spider_motif(), make_zz_interaction_motif()]
        result = optimize_library(candidates, corpus, max_library_size=1)
        assert len(result.selected_motifs) <= 1

    def test_final_coverage_in_range(self):
        """Final coverage is between 0 and 1."""
        corpus = _make_corpus()
        candidates = [make_cx_spider_motif(), make_zz_interaction_motif()]
        result = optimize_library(candidates, corpus)
        assert 0.0 <= result.final_coverage <= 1.0

    def test_empty_candidates_returns_empty(self):
        """Empty candidates list returns empty library."""
        corpus = _make_corpus()
        result = optimize_library([], corpus)
        assert len(result.selected_motifs) == 0
        assert result.final_coverage == 0.0

    def test_summary_returns_string(self):
        """summary() returns a non-empty string."""
        corpus = _make_corpus()
        candidates = [make_cx_spider_motif(), make_zz_interaction_motif()]
        result = optimize_library(candidates, corpus)
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_dropped_motifs_tracked(self):
        """Motifs not selected are tracked in dropped_motifs."""
        corpus = _make_corpus()
        candidates = [make_cx_spider_motif(), make_zz_interaction_motif()]
        result = optimize_library(
            candidates, corpus, min_marginal_coverage=0.99,
        )
        # With very high threshold, most/all motifs should be dropped
        total = len(result.selected_motifs) + len(result.dropped_motifs)
        assert total == len(candidates)

    def test_scores_populated(self):
        """Each selected motif has a corresponding score."""
        corpus = _make_corpus()
        candidates = [make_cx_spider_motif(), make_zz_interaction_motif()]
        result = optimize_library(candidates, corpus)
        assert len(result.scores) == len(result.selected_motifs)
        for score in result.scores:
            assert isinstance(score, MotifScore)
            assert score.weight > 0
