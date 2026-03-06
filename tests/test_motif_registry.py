"""Tests for the declarative motif registry (JSON-backed)."""
import pytest
import networkx as nx

from zx_motifs.motifs import MOTIF_REGISTRY, get_motif


# ── Valid values from the JSON schema ────────────────────────────────

VALID_VERTEX_TYPES = {"Z", "X", "H_BOX"}
VALID_PHASE_CLASSES = {
    "zero", "pauli", "clifford", "t_like", "arbitrary",
    "any", "any_nonzero", "any_nonclifford",
}
VALID_EDGE_TYPES = {"SIMPLE", "HADAMARD"}


# ── Parametrized fixture over all motifs ─────────────────────────────


def _motif_ids():
    return [m.motif_id for m in MOTIF_REGISTRY]


@pytest.fixture(params=MOTIF_REGISTRY, ids=_motif_ids())
def motif(request):
    return request.param


# ── Per-motif tests ──────────────────────────────────────────────────


def test_motif_is_connected(motif):
    """Every motif graph must be connected."""
    assert nx.is_connected(motif.graph), (
        f"Motif {motif.motif_id} graph is not connected"
    )


def test_motif_has_valid_node_attrs(motif):
    """Each node must have a valid vertex_type and phase_class."""
    for node, data in motif.graph.nodes(data=True):
        vtype = data.get("vertex_type")
        assert vtype in VALID_VERTEX_TYPES, (
            f"Motif {motif.motif_id}, node {node}: "
            f"invalid vertex_type={vtype!r}"
        )
        phase = data.get("phase_class")
        assert phase in VALID_PHASE_CLASSES, (
            f"Motif {motif.motif_id}, node {node}: "
            f"invalid phase_class={phase!r}"
        )


def test_motif_has_valid_edge_attrs(motif):
    """Each edge must have a valid edge_type."""
    for u, v, data in motif.graph.edges(data=True):
        etype = data.get("edge_type")
        assert etype in VALID_EDGE_TYPES, (
            f"Motif {motif.motif_id}, edge {u}--{v}: "
            f"invalid edge_type={etype!r}"
        )


def test_motif_has_description(motif):
    """Every motif must have a non-empty description."""
    assert len(motif.description) > 0, (
        f"Motif {motif.motif_id} has an empty description"
    )


def test_motif_min_size(motif):
    """Every motif graph must have at least 2 nodes."""
    assert motif.graph.number_of_nodes() >= 2, (
        f"Motif {motif.motif_id} has fewer than 2 nodes"
    )


def test_get_motif_lookup(motif):
    """get_motif() returns the correct motif for each registered ID."""
    result = get_motif(motif.motif_id)
    assert result is not None, f"get_motif({motif.motif_id!r}) returned None"
    assert result.motif_id == motif.motif_id


# ── Registry-wide tests (not parametrized) ──────────────────────────


def test_motif_has_unique_id():
    """No duplicate motif_ids across the registry."""
    ids = [m.motif_id for m in MOTIF_REGISTRY]
    assert len(ids) == len(set(ids)), (
        f"Duplicate motif IDs found: "
        f"{[mid for mid in ids if ids.count(mid) > 1]}"
    )


def test_get_motif_missing():
    """get_motif() returns None for a nonexistent motif ID."""
    assert get_motif("nonexistent") is None


def test_registry_count():
    """Registry must contain at least 15 motifs."""
    assert len(MOTIF_REGISTRY) >= 15, (
        f"Expected >= 15 motifs, found {len(MOTIF_REGISTRY)}"
    )
