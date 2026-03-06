"""Tests for phase-parametric motifs (Phase 6)."""
import networkx as nx

from zx_motifs.pipeline.matcher import (
    PHASE_ANY,
    PHASE_ANY_NONCLIFFORD,
    PHASE_ANY_NONZERO,
    MotifPattern,
    find_motif_in_graph,
    is_parametric_motif,
    phase_class_matches,
)
from zx_motifs.pipeline.motif_generators import (
    EXTENDED_MOTIFS,
    HANDCRAFTED_MOTIFS,
    PARAMETRIC_MOTIFS,
    make_syndrome_extraction_motif,
    make_syndrome_extraction_param_motif,
)


# ── phase_class_matches ──────────────────────────────────────────


class TestPhaseClassMatches:
    def test_any_matches_all(self):
        """PHASE_ANY matches every phase class."""
        for pc in ("zero", "pauli", "clifford", "t_like", "arbitrary"):
            assert phase_class_matches(PHASE_ANY, pc)

    def test_any_nonzero_rejects_zero(self):
        """PHASE_ANY_NONZERO rejects 'zero'."""
        assert not phase_class_matches(PHASE_ANY_NONZERO, "zero")
        assert phase_class_matches(PHASE_ANY_NONZERO, "pauli")
        assert phase_class_matches(PHASE_ANY_NONZERO, "t_like")
        assert phase_class_matches(PHASE_ANY_NONZERO, "arbitrary")

    def test_any_nonclifford_rejects_zero_clifford_pauli(self):
        """PHASE_ANY_NONCLIFFORD rejects zero, clifford, and pauli."""
        assert not phase_class_matches(PHASE_ANY_NONCLIFFORD, "zero")
        assert not phase_class_matches(PHASE_ANY_NONCLIFFORD, "clifford")
        assert not phase_class_matches(PHASE_ANY_NONCLIFFORD, "pauli")
        assert phase_class_matches(PHASE_ANY_NONCLIFFORD, "t_like")
        assert phase_class_matches(PHASE_ANY_NONCLIFFORD, "arbitrary")

    def test_exact_match(self):
        """Exact phase classes match only themselves."""
        assert phase_class_matches("zero", "zero")
        assert not phase_class_matches("zero", "pauli")
        assert phase_class_matches("t_like", "t_like")


# ── Parametric motif matching ────────────────────────────────────


class TestParametricMatching:
    def test_parametric_syndrome_matches_t_like(self):
        """Parametric syndrome matches Z(t_like) → 2×X(zero)."""
        host = nx.Graph()
        host.add_node(0, vertex_type="Z", phase_class="t_like", is_boundary=False)
        host.add_node(1, vertex_type="X", phase_class="zero", is_boundary=False)
        host.add_node(2, vertex_type="X", phase_class="zero", is_boundary=False)
        host.add_edge(0, 1, edge_type="SIMPLE")
        host.add_edge(0, 2, edge_type="SIMPLE")

        param = make_syndrome_extraction_param_motif()
        matches = find_motif_in_graph(param.graph, host, exclude_boundary=False)
        assert len(matches) >= 1

    def test_original_syndrome_misses_t_like(self):
        """Original (exact) syndrome does NOT match Z(t_like) center."""
        host = nx.Graph()
        host.add_node(0, vertex_type="Z", phase_class="t_like", is_boundary=False)
        host.add_node(1, vertex_type="X", phase_class="zero", is_boundary=False)
        host.add_node(2, vertex_type="X", phase_class="zero", is_boundary=False)
        host.add_edge(0, 1, edge_type="SIMPLE")
        host.add_edge(0, 2, edge_type="SIMPLE")

        exact = make_syndrome_extraction_motif()
        matches = find_motif_in_graph(exact.graph, host, exclude_boundary=False)
        assert len(matches) == 0

    def test_parametric_syndrome_also_matches_zero(self):
        """Parametric syndrome still matches the Z(zero) case."""
        host = nx.Graph()
        host.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
        host.add_node(1, vertex_type="X", phase_class="zero", is_boundary=False)
        host.add_node(2, vertex_type="X", phase_class="zero", is_boundary=False)
        host.add_edge(0, 1, edge_type="SIMPLE")
        host.add_edge(0, 2, edge_type="SIMPLE")

        param = make_syndrome_extraction_param_motif()
        matches = find_motif_in_graph(param.graph, host, exclude_boundary=False)
        assert len(matches) >= 1

    def test_any_nonzero_rejects_zero_host(self):
        """ZZ interaction param (any_nonzero center) rejects Z(zero) center."""
        from zx_motifs.pipeline.motif_generators import make_zz_interaction_param_motif

        host = nx.Graph()
        host.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
        host.add_node(1, vertex_type="Z", phase_class="zero", is_boundary=False)
        host.add_node(2, vertex_type="Z", phase_class="zero", is_boundary=False)
        host.add_edge(0, 1, edge_type="SIMPLE")
        host.add_edge(1, 2, edge_type="SIMPLE")

        param = make_zz_interaction_param_motif()
        matches = find_motif_in_graph(param.graph, host, exclude_boundary=False)
        assert len(matches) == 0

    def test_any_nonclifford_accepts_t_like(self):
        """Toffoli core param accepts t_like outer phases."""
        from zx_motifs.pipeline.motif_generators import make_toffoli_core_param_motif

        host = nx.Graph()
        host.add_node(0, vertex_type="Z", phase_class="t_like", is_boundary=False)
        host.add_node(1, vertex_type="Z", phase_class="zero", is_boundary=False)
        host.add_node(2, vertex_type="Z", phase_class="t_like", is_boundary=False)
        host.add_edge(0, 1, edge_type="SIMPLE")
        host.add_edge(1, 2, edge_type="SIMPLE")

        param = make_toffoli_core_param_motif()
        matches = find_motif_in_graph(param.graph, host, exclude_boundary=False)
        assert len(matches) >= 1


# ── is_parametric_motif ──────────────────────────────────────────


class TestIsParametricMotif:
    def test_parametric_detected(self):
        """Parametric motifs are detected as parametric."""
        param = make_syndrome_extraction_param_motif()
        assert is_parametric_motif(param.graph)

    def test_exact_not_parametric(self):
        """Exact motifs are not parametric."""
        exact = make_syndrome_extraction_motif()
        assert not is_parametric_motif(exact.graph)


# ── Library composition ──────────────────────────────────────────


class TestExtendedMotifs:
    def test_extended_motifs_count(self):
        """EXTENDED_MOTIFS = handcrafted + parametric."""
        assert len(HANDCRAFTED_MOTIFS) >= 9
        assert len(PARAMETRIC_MOTIFS) >= 6
        assert len(EXTENDED_MOTIFS) == len(HANDCRAFTED_MOTIFS) + len(PARAMETRIC_MOTIFS)

    def test_all_have_unique_ids(self):
        """All motif IDs in EXTENDED_MOTIFS are unique."""
        ids = [m.motif_id for m in EXTENDED_MOTIFS]
        assert len(ids) == len(set(ids))
