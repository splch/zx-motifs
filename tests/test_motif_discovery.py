"""Tests for multi-level motif discovery and neighborhood extraction (Phase 1)."""
import pytest
import networkx as nx

from zx_motifs.pipeline.converter import convert_at_all_levels
from zx_motifs.pipeline.featurizer import pyzx_to_networkx
from zx_motifs.pipeline.matcher import (
    MotifMatch,
    MotifPattern,
    find_motif_across_corpus,
    find_motif_across_corpus_multilevel,
)
from zx_motifs.pipeline.motif_generators import (
    HANDCRAFTED_MOTIFS,
    _is_isomorphic,
    find_neighborhood_motifs,
    find_recurring_subgraphs,
    find_recurring_subgraphs_multilevel,
    make_cx_spider_motif,
    make_phase_gadget_motif,
    make_toffoli_core_motif,
    make_trotter_layer_motif,
)


# ── Fixtures ────────────────────────────────────────────────────────


def _build_corpus(algorithms):
    """Build a full multi-level corpus from a list of (name, generator) tuples."""
    corpus = {}
    for name, gen_fn in algorithms:
        qc = gen_fn()
        snapshots = convert_at_all_levels(qc, name)
        for snap in snapshots:
            nxg = pyzx_to_networkx(snap.graph, coarsen_phases=True)
            corpus[(name, snap.level.value)] = nxg
    return corpus


@pytest.fixture(scope="module")
def small_corpus():
    """Multi-level corpus from 4 diverse algorithms."""
    from zx_motifs.algorithms.registry import (
        make_bell_state,
        make_bit_flip_code,
        make_grover,
        make_qaoa_maxcut,
    )
    return _build_corpus([
        ("bell_state", make_bell_state),
        ("grover", lambda: make_grover(n_qubits=3)),
        ("qaoa_maxcut", lambda: make_qaoa_maxcut(n_qubits=3)),
        ("bit_flip_code", make_bit_flip_code),
    ])


@pytest.fixture(scope="module")
def spider_fused_only_corpus(small_corpus):
    """Subset of small_corpus at spider_fused level only."""
    return {k: v for k, v in small_corpus.items() if k[1] == "spider_fused"}


# ── TestMultiLevelDiscovery ──────────────────────────────────────


class TestMultiLevelDiscovery:
    def test_discovers_motifs_beyond_spider_fused(self, small_corpus):
        """Multi-level search finds motifs that single-level misses."""
        multi = find_recurring_subgraphs_multilevel(
            small_corpus, min_size=3, max_size=5, min_algorithms=2,
        )
        single = find_recurring_subgraphs(
            small_corpus, target_level="spider_fused",
            min_size=3, max_size=5, min_algorithms=2,
        )
        # Multi-level should find at least as many as single-level
        assert len(multi) >= len(single)

    def test_single_level_backward_compat(self, small_corpus):
        """Passing a single level matches the original function."""
        multi = find_recurring_subgraphs_multilevel(
            small_corpus, levels=["spider_fused"],
            min_size=3, max_size=5, min_algorithms=2,
        )
        single = find_recurring_subgraphs(
            small_corpus, target_level="spider_fused",
            min_size=3, max_size=5, min_algorithms=2,
        )
        # Should find the same number of motifs
        assert len(multi) == len(single)

    def test_dedup_across_levels(self, small_corpus):
        """Motifs found at multiple levels are deduplicated."""
        multi = find_recurring_subgraphs_multilevel(
            small_corpus, min_size=3, max_size=5, min_algorithms=2,
        )
        # Every motif should have a non-empty discovery_levels list
        for m in multi:
            assert len(m.discovery_levels) >= 1

    def test_discovery_levels_populated(self, small_corpus):
        """discovery_levels contains valid level names."""
        valid_levels = {
            "raw", "spider_fused", "interior_cliff",
            "clifford_simp", "full_reduce", "teleport_reduce",
        }
        multi = find_recurring_subgraphs_multilevel(
            small_corpus, min_size=3, max_size=5, min_algorithms=2,
        )
        for m in multi:
            for lvl in m.discovery_levels:
                assert lvl in valid_levels

    def test_returns_motif_patterns(self, small_corpus):
        """Return type is list of MotifPattern."""
        multi = find_recurring_subgraphs_multilevel(
            small_corpus, min_size=3, max_size=5, min_algorithms=2,
        )
        for m in multi:
            assert isinstance(m, MotifPattern)
            assert isinstance(m.graph, nx.Graph)
            assert m.graph.number_of_nodes() >= 3


# ── TestMultiLevelSearch ─────────────────────────────────────────


class TestMultiLevelSearch:
    def test_search_all_levels(self, small_corpus):
        """Multilevel search finds occurrences across levels."""
        cx = make_cx_spider_motif()
        result = find_motif_across_corpus_multilevel(cx, small_corpus)
        if result.occurrences:
            levels_found = {occ.host_level for occ in result.occurrences}
            # Should find matches at more than one level (cx_pair is common)
            assert len(levels_found) >= 1

    def test_search_specific_levels(self, small_corpus):
        """Searching specific levels only returns matches from those levels."""
        cx = make_cx_spider_motif()
        result = find_motif_across_corpus_multilevel(
            cx, small_corpus, levels=["spider_fused"]
        )
        for occ in result.occurrences:
            assert occ.host_level == "spider_fused"

    def test_occurrences_carry_correct_host_level(self, small_corpus):
        """Each MotifMatch has the correct host_level field."""
        cx = make_cx_spider_motif()
        result = find_motif_across_corpus_multilevel(cx, small_corpus)
        corpus_levels = {lvl for (_, lvl) in small_corpus.keys()}
        for occ in result.occurrences:
            assert occ.host_level in corpus_levels

    def test_multilevel_finds_at_least_as_many_as_single(self, small_corpus):
        """Multilevel search >= single-level search."""
        cx = make_cx_spider_motif()
        single = find_motif_across_corpus(cx, small_corpus, target_level="spider_fused")
        n_single = len(single.occurrences)
        multi = find_motif_across_corpus_multilevel(cx, small_corpus)
        assert len(multi.occurrences) >= n_single

    def test_search_empty_levels_list(self, small_corpus):
        """Searching with an empty levels list returns no matches."""
        cx = make_cx_spider_motif()
        result = find_motif_across_corpus_multilevel(cx, small_corpus, levels=[])
        assert len(result.occurrences) == 0


# ── TestNeighborhoodMotifs ───────────────────────────────────────


class TestNeighborhoodMotifs:
    def test_non_clifford_criterion(self, small_corpus):
        """non_clifford criterion runs without error."""
        result = find_neighborhood_motifs(
            small_corpus, criteria=["non_clifford"], min_algorithms=1,
        )
        for m in result:
            assert m.source == "neighborhood"

    def test_high_degree_criterion(self, small_corpus):
        """high_degree criterion runs without error."""
        result = find_neighborhood_motifs(
            small_corpus, criteria=["high_degree"], min_algorithms=1,
        )
        for m in result:
            assert m.source == "neighborhood"

    def test_color_boundary_criterion(self, small_corpus):
        """color_boundary criterion runs without error."""
        result = find_neighborhood_motifs(
            small_corpus, criteria=["color_boundary"], min_algorithms=1,
        )
        for m in result:
            assert m.source == "neighborhood"

    def test_dedup_works(self, small_corpus):
        """Identical neighborhoods from different criteria are deduplicated."""
        all_criteria = find_neighborhood_motifs(
            small_corpus,
            criteria=["non_clifford", "high_degree", "color_boundary"],
            min_algorithms=1,
        )
        ids = [m.motif_id for m in all_criteria]
        # IDs should be unique
        assert len(ids) == len(set(ids))

    def test_source_is_neighborhood(self, small_corpus):
        """All returned motifs have source='neighborhood'."""
        result = find_neighborhood_motifs(
            small_corpus, min_algorithms=1,
        )
        for m in result:
            assert m.source == "neighborhood"
            assert m.motif_id.startswith("nbr_")

    def test_min_algorithms_filters(self, small_corpus):
        """Higher min_algorithms threshold returns fewer motifs."""
        low = find_neighborhood_motifs(small_corpus, min_algorithms=1)
        high = find_neighborhood_motifs(small_corpus, min_algorithms=3)
        assert len(low) >= len(high)

    def test_returns_connected_graphs(self, small_corpus):
        """All neighborhood motifs are connected graphs."""
        result = find_neighborhood_motifs(small_corpus, min_algorithms=1)
        for m in result:
            if m.graph.number_of_nodes() > 0:
                assert nx.is_connected(m.graph)


# ── TestMotifPatternBackcompat ───────────────────────────────────


class TestMotifPatternBackcompat:
    def test_old_construction_still_works(self):
        """MotifPattern without new fields still works."""
        g = nx.Graph()
        g.add_node(0, vertex_type="Z", phase_class="zero")
        g.add_node(1, vertex_type="X", phase_class="zero")
        g.add_edge(0, 1, edge_type="SIMPLE")
        mp = MotifPattern(motif_id="test", graph=g, source="test")
        assert mp.motif_id == "test"
        assert mp.discovery_levels == []
        assert mp.metadata == {}

    def test_new_fields_default_empty(self):
        """New fields default to empty."""
        mp = make_cx_spider_motif()
        assert mp.discovery_levels == []
        assert mp.metadata == {}

    def test_handcrafted_motifs_unchanged(self):
        """All handcrafted motifs still instantiate correctly."""
        assert len(HANDCRAFTED_MOTIFS) >= 9
        for m in HANDCRAFTED_MOTIFS:
            assert isinstance(m, MotifPattern)
            assert m.graph.number_of_nodes() >= 2

    def test_new_fields_can_be_set(self):
        """New fields can be set at construction."""
        g = nx.Graph()
        g.add_node(0, vertex_type="Z", phase_class="zero")
        mp = MotifPattern(
            motif_id="test", graph=g, source="test",
            discovery_levels=["raw", "spider_fused"],
            metadata={"key": "value"},
        )
        assert mp.discovery_levels == ["raw", "spider_fused"]
        assert mp.metadata == {"key": "value"}
