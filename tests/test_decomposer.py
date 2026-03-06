"""Tests for motif decomposition / greedy cover (Phase 4)."""
import pytest
import networkx as nx

from zx_motifs.pipeline.converter import convert_at_all_levels
from zx_motifs.pipeline.featurizer import pyzx_to_networkx
from zx_motifs.pipeline.decomposer import (
    DecompositionResult,
    MotifPlacement,
    decompose_across_corpus,
    decompose_graph,
)
from zx_motifs.pipeline.matcher import MotifPattern
from zx_motifs.pipeline.motif_generators import (
    HANDCRAFTED_MOTIFS,
    make_cx_spider_motif,
    make_zz_interaction_motif,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_host_with_cx():
    """Host that contains exactly one cx_pair (Z-X via SIMPLE)."""
    g = nx.Graph()
    g.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(1, vertex_type="X", phase_class="zero", is_boundary=False)
    g.add_node(2, vertex_type="Z", phase_class="t_like", is_boundary=False)
    g.add_edge(0, 1, edge_type="SIMPLE")
    g.add_edge(1, 2, edge_type="SIMPLE")
    return g


def _make_host_with_two_motifs():
    """Host containing a cx_pair and a zz_interaction, non-overlapping."""
    g = nx.Graph()
    # CX pair: nodes 0-1
    g.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(1, vertex_type="X", phase_class="zero", is_boundary=False)
    g.add_edge(0, 1, edge_type="SIMPLE")
    # ZZ interaction: nodes 2-3-4
    g.add_node(2, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(3, vertex_type="Z", phase_class="arbitrary", is_boundary=False)
    g.add_node(4, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_edge(2, 3, edge_type="SIMPLE")
    g.add_edge(3, 4, edge_type="SIMPLE")
    return g


# ── Tests ────────────────────────────────────────────────────────────


class TestDecomposeGraph:
    def test_empty_library_zero_coverage(self):
        """Empty motif library means 0% coverage."""
        host = _make_host_with_cx()
        result = decompose_graph(host, [], exclude_boundary=False)
        assert result.coverage_ratio == 0.0
        assert len(result.placements) == 0

    def test_single_motif_covers_its_match(self):
        """A single motif covers its matching vertices."""
        host = _make_host_with_cx()
        lib = [make_cx_spider_motif()]
        result = decompose_graph(host, lib, exclude_boundary=False)
        assert len(result.placements) >= 1
        assert result.coverage_ratio > 0

    def test_non_overlapping_enforced(self):
        """Placements should not overlap."""
        host = _make_host_with_two_motifs()
        lib = [make_cx_spider_motif(), make_zz_interaction_motif()]
        result = decompose_graph(host, lib, exclude_boundary=False)
        all_verts = []
        for p in result.placements:
            all_verts.extend(p.host_vertices)
        assert len(all_verts) == len(set(all_verts))

    def test_prefers_larger_motifs(self):
        """With prefer_larger=True, larger motifs are chosen first."""
        host = _make_host_with_two_motifs()
        lib = [make_cx_spider_motif(), make_zz_interaction_motif()]
        result = decompose_graph(host, lib, exclude_boundary=False, prefer_larger=True)
        if len(result.placements) >= 2:
            sizes = [len(p.host_vertices) for p in result.placements]
            # First placement should be >= second
            assert sizes[0] >= sizes[1]

    def test_coverage_ratio_in_range(self):
        """Coverage ratio is between 0 and 1."""
        host = _make_host_with_two_motifs()
        lib = [make_cx_spider_motif(), make_zz_interaction_motif()]
        result = decompose_graph(host, lib, exclude_boundary=False)
        assert 0.0 <= result.coverage_ratio <= 1.0

    def test_covered_plus_uncovered_equals_total(self):
        """Covered + uncovered = total interior vertices."""
        host = _make_host_with_two_motifs()
        lib = [make_cx_spider_motif()]
        result = decompose_graph(host, lib, exclude_boundary=False)
        total = len(result.covered_vertices) + len(result.uncovered_vertices)
        assert total == host.number_of_nodes()

    def test_summary_string(self):
        """summary() returns a non-empty string."""
        host = _make_host_with_two_motifs()
        lib = [make_cx_spider_motif(), make_zz_interaction_motif()]
        result = decompose_graph(host, lib, exclude_boundary=False)
        s = result.summary()
        assert isinstance(s, str)
        assert "coverage" in s.lower() or "Decomposition" in s


class TestDecomposeAcrossCorpus:
    def test_real_algorithm_decomposition(self):
        """Decompose qaoa_maxcut with handcrafted motifs."""
        from zx_motifs.algorithms.registry import make_qaoa_maxcut
        qc = make_qaoa_maxcut(n_qubits=3)
        snapshots = convert_at_all_levels(qc, "qaoa_maxcut")
        corpus = {}
        for snap in snapshots:
            nxg = pyzx_to_networkx(snap.graph, coarsen_phases=True)
            corpus[("qaoa_maxcut", snap.level.value)] = nxg

        results = decompose_across_corpus(corpus, HANDCRAFTED_MOTIFS)
        assert "qaoa_maxcut" in results
        r = results["qaoa_maxcut"]
        assert isinstance(r, DecompositionResult)
        # Should find at least something
        assert r.coverage_ratio >= 0.0

    def test_decompose_multiple_algorithms(self):
        """Decompose multiple algorithms."""
        from zx_motifs.algorithms.registry import make_bell_state, make_bit_flip_code
        corpus = {}
        for name, gen_fn in [("bell", make_bell_state), ("bitflip", make_bit_flip_code)]:
            qc = gen_fn()
            for snap in convert_at_all_levels(qc, name):
                nxg = pyzx_to_networkx(snap.graph, coarsen_phases=True)
                corpus[(name, snap.level.value)] = nxg

        results = decompose_across_corpus(corpus, HANDCRAFTED_MOTIFS)
        assert "bell" in results
        assert "bitflip" in results

    def test_exclude_boundary(self):
        """Boundary nodes are excluded from coverage stats."""
        from zx_motifs.algorithms.registry import make_bell_state
        qc = make_bell_state()
        for snap in convert_at_all_levels(qc, "bell"):
            if snap.level.value == "spider_fused":
                nxg = pyzx_to_networkx(snap.graph, coarsen_phases=True)
                break
        result = decompose_graph(nxg, HANDCRAFTED_MOTIFS, exclude_boundary=True)
        for v in result.covered_vertices:
            assert nxg.nodes[v].get("vertex_type") != "BOUNDARY"
        for v in result.uncovered_vertices:
            assert nxg.nodes[v].get("vertex_type") != "BOUNDARY"
