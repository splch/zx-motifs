"""Tests for cross-level motif tracking and structural catalog similarity (Phase 3)."""
import pytest
import networkx as nx
import numpy as np

from zx_motifs.pipeline.converter import convert_at_all_levels
from zx_motifs.pipeline.featurizer import pyzx_to_networkx, compute_motif_feature_vector
from zx_motifs.pipeline.cross_level import (
    MotifEvolution,
    track_motif_evolution,
    track_all_motifs_evolution,
)
from zx_motifs.pipeline.catalog import CatalogEntry, MotifCatalog
from zx_motifs.pipeline.matcher import MotifPattern
from zx_motifs.pipeline.motif_generators import (
    make_cx_spider_motif,
    make_zz_interaction_motif,
    make_hadamard_sandwich_motif,
)


# ── Fixtures ────────────────────────────────────────────────────────


def _build_corpus(algorithms):
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


# ── TestMotifEvolution ───────────────────────────────────────────


class TestMotifEvolution:
    def test_cx_pair_survives_raw_to_spider_fused(self, small_corpus):
        """cx_pair should survive at least raw and spider_fused."""
        cx = make_cx_spider_motif()
        evo = track_motif_evolution(cx, small_corpus, "bit_flip_code")
        # cx_pair is so common it should survive at early levels
        assert evo.motif_id == "cx_pair"
        early_levels = {"raw", "spider_fused"}
        survived = set(evo.survives_to)
        assert survived & early_levels  # at least one early level

    def test_all_six_levels_covered(self, small_corpus):
        """Evolution should check all 6 levels."""
        cx = make_cx_spider_motif()
        evo = track_motif_evolution(cx, small_corpus, "bit_flip_code")
        total = (
            len(evo.survives_to) + len(evo.transforms_at) + len(evo.vanishes_at)
        )
        assert total == 6

    def test_level_occurrences_populated(self, small_corpus):
        """level_occurrences maps levels to match counts."""
        cx = make_cx_spider_motif()
        evo = track_motif_evolution(cx, small_corpus, "bit_flip_code")
        # Survived levels should have positive occurrence counts
        for lvl in evo.survives_to:
            assert evo.level_occurrences[lvl] > 0

    def test_vanished_has_zero_occurrences(self, small_corpus):
        """Vanished levels should have 0 occurrences."""
        cx = make_cx_spider_motif()
        evo = track_motif_evolution(cx, small_corpus, "bit_flip_code")
        for lvl in evo.vanishes_at:
            assert evo.level_occurrences.get(lvl, 0) == 0

    def test_track_all_motifs(self, small_corpus):
        """Batch tracking returns results for all motifs and algorithms."""
        motifs = [make_cx_spider_motif(), make_zz_interaction_motif()]
        result = track_all_motifs_evolution(motifs, small_corpus)
        assert "cx_pair" in result
        assert "zz_interaction" in result
        # Should have one evolution per algorithm
        n_algos = len({a for (a, _) in small_corpus.keys()})
        assert len(result["cx_pair"]) == n_algos

    def test_specific_levels_only(self, small_corpus):
        """Can restrict to specific levels."""
        cx = make_cx_spider_motif()
        evo = track_motif_evolution(
            cx, small_corpus, "bit_flip_code",
            levels=["raw", "spider_fused"],
        )
        total = len(evo.survives_to) + len(evo.transforms_at) + len(evo.vanishes_at)
        assert total == 2


# ── TestCatalogStructuralSimilarity ──────────────────────────────


class TestCatalogStructuralSimilarity:
    def test_feature_vector_stored(self, tmp_path):
        """add_motif stores feature_vector in CatalogEntry."""
        cat = MotifCatalog(path=str(tmp_path / "cat.json"))
        cx = make_cx_spider_motif()
        cat.add_motif(cx, {})
        entry = cat.entries["cx_pair"]
        assert len(entry.feature_vector) == 12
        assert all(isinstance(v, float) for v in entry.feature_vector)

    def test_find_related_returns_structurally_similar(self, tmp_path):
        """find_related uses structural similarity."""
        cat = MotifCatalog(path=str(tmp_path / "cat.json"))
        # Add two structurally similar motifs (both are 3-node Z chains)
        zz = make_zz_interaction_motif()
        cat.add_motif(zz, {})
        hsand = make_hadamard_sandwich_motif()
        cat.add_motif(hsand, {})
        # They should be related structurally (same topology, different labels)
        related = cat.find_related("zz_interaction", similarity_threshold=0.3)
        related_ids = [r[0] for r in related]
        assert "hadamard_sandwich" in related_ids

    def test_combined_score_in_unit_range(self, tmp_path):
        """Combined scores are in [0, 1]."""
        cat = MotifCatalog(path=str(tmp_path / "cat.json"))
        cat.add_motif(make_cx_spider_motif(), {})
        cat.add_motif(make_zz_interaction_motif(), {})
        cat.add_motif(make_hadamard_sandwich_motif(), {})
        related = cat.find_related("cx_pair", similarity_threshold=0.0)
        for _, score in related:
            assert 0.0 <= score <= 1.0

    def test_backward_compatible_default_weights(self, tmp_path):
        """Default weights work without error."""
        cat = MotifCatalog(path=str(tmp_path / "cat.json"))
        cat.add_motif(make_cx_spider_motif(), {})
        cat.add_motif(make_zz_interaction_motif(), {})
        # Should work with default signature
        related = cat.find_related("cx_pair")
        assert isinstance(related, list)

    def test_cross_level_info_field(self, tmp_path):
        """CatalogEntry has cross_level_info field."""
        cat = MotifCatalog(path=str(tmp_path / "cat.json"))
        cx = make_cx_spider_motif()
        cat.add_motif(cx, {})
        entry = cat.entries["cx_pair"]
        assert hasattr(entry, "cross_level_info")
        assert entry.cross_level_info == {}
