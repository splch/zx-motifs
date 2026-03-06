"""Tests for Weisfeiler-Leman hashing (Phase 5)."""
import networkx as nx

from zx_motifs.pipeline.motif_generators import (
    _ensure_wl_label,
    canonical_hash,
    get_hash_fn,
    set_hash_fn,
    wl_hash,
)


def _make_zx_pair():
    """Z-X pair via SIMPLE edge."""
    g = nx.Graph()
    g.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(1, vertex_type="X", phase_class="zero", is_boundary=False)
    g.add_edge(0, 1, edge_type="SIMPLE")
    return g


def _make_zx_pair_relabeled():
    """Same topology as _make_zx_pair but with different node IDs."""
    g = nx.Graph()
    g.add_node(10, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(20, vertex_type="X", phase_class="zero", is_boundary=False)
    g.add_edge(10, 20, edge_type="SIMPLE")
    return g


class TestWLHash:
    def test_isomorphic_same_hash(self):
        """Isomorphic graphs produce the same WL hash."""
        g1 = _make_zx_pair()
        g2 = _make_zx_pair_relabeled()
        assert wl_hash(g1) == wl_hash(g2)

    def test_different_labels_different_hash(self):
        """Different node labels produce different hashes."""
        g1 = _make_zx_pair()
        g2 = nx.Graph()
        g2.add_node(0, vertex_type="Z", phase_class="t_like", is_boundary=False)
        g2.add_node(1, vertex_type="X", phase_class="zero", is_boundary=False)
        g2.add_edge(0, 1, edge_type="SIMPLE")
        assert wl_hash(g1) != wl_hash(g2)

    def test_different_topology_different_hash(self):
        """Different topologies produce different hashes."""
        g_pair = _make_zx_pair()
        g_chain = nx.Graph()
        g_chain.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
        g_chain.add_node(1, vertex_type="X", phase_class="zero", is_boundary=False)
        g_chain.add_node(2, vertex_type="Z", phase_class="zero", is_boundary=False)
        g_chain.add_edge(0, 1, edge_type="SIMPLE")
        g_chain.add_edge(1, 2, edge_type="SIMPLE")
        assert wl_hash(g_pair) != wl_hash(g_chain)

    def test_edge_type_matters(self):
        """SIMPLE vs HADAMARD edges produce different hashes."""
        g1 = _make_zx_pair()
        g2 = nx.Graph()
        g2.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
        g2.add_node(1, vertex_type="X", phase_class="zero", is_boundary=False)
        g2.add_edge(0, 1, edge_type="HADAMARD")
        assert wl_hash(g1) != wl_hash(g2)

    def test_wl_label_attribute_set(self):
        """_ensure_wl_label sets the wl_label attribute correctly."""
        g = _make_zx_pair()
        _ensure_wl_label(g)
        assert g.nodes[0]["wl_label"] == "Z_zero"
        assert g.nodes[1]["wl_label"] == "X_zero"

    def test_returns_32_char_hex(self):
        """wl_hash returns a 32-character hex string (digest_size=16)."""
        g = _make_zx_pair()
        h = wl_hash(g)
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_fn_switchable(self):
        """get/set_hash_fn allows switching hash functions."""
        original = get_hash_fn()
        assert original is wl_hash

        set_hash_fn(canonical_hash)
        assert get_hash_fn() is canonical_hash

        # Restore
        set_hash_fn(wl_hash)
        assert get_hash_fn() is wl_hash
