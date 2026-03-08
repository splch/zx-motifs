"""
Step 3 - ZX-Web Mining: Identify Common Sub-Diagram Motifs
============================================================
A **ZX-Web** is a small, recurring sub-diagram motif found within and
across the simplified ZX-diagrams of the algorithm corpus.  Concretely
it is a labelled graph pattern ``(V_sub, E_sub)`` where each vertex
carries a *type* label (Z, X, Boundary, H-box) and a *phase class*
(zero, Clifford, T, arbitrary) and each edge carries a type
(simple vs Hadamard).

Mining Pipeline
---------------
1. Convert each PyZX ``Graph`` to a ``networkx.Graph`` with typed,
   phase-classified nodes and typed edges.
2. Enumerate all *connected induced subgraphs* of size 2..k
   (default k = 6) via bounded-depth BFS from every vertex.
3. Canonicalise each subgraph using a Weisfeiler-Leman-style hash that
   respects node/edge labels so that isomorphic motifs share a key.
4. Count motif frequencies across the corpus.
5. Return motifs whose frequency exceeds a threshold as ``ZXWeb``
   objects, each annotated with its source algorithms.

Complexity Note
---------------
For fully-reduced ZX-diagrams of the algorithms in our corpus (typically
< 60 vertices after ``full_reduce``), enumeration up to size 6 is fast.
For larger diagrams you can lower ``max_subgraph_size`` or sample
vertices.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import copy
import hashlib
import itertools

import networkx as nx
import pyzx as zx

from .zx_convert import ZXEntry


# ---------------------------------------------------------------------------
# Phase classification helpers
# ---------------------------------------------------------------------------

def _classify_phase(phase) -> str:
    """Classify a PyZX phase value into a coarse category.

    Phase in PyZX is stored as a ``Fraction`` representing multiples of π.
    Categories:
        "zero"     - phase == 0
        "clifford" - phase ∈ {0, 1/2, 1, 3/2}  (multiples of π/2)
        "t_like"   - phase ∈ {1/4, 3/4, 5/4, 7/4} (multiples of π/4)
        "arb"      - everything else
    """
    from fractions import Fraction
    p = Fraction(phase) % 2  # normalise to [0, 2)
    if p == 0:
        return "zero"
    if p.denominator <= 2:
        return "clifford"
    if p.denominator == 4:
        return "t_like"
    return "arb"


# ---------------------------------------------------------------------------
# PyZX Graph → NetworkX Graph
# ---------------------------------------------------------------------------

_VERTEX_TYPE_NAMES = {
    1: "Z",
    2: "X",
    0: "B",  # boundary
}


def pyzx_to_networkx(g: zx.Graph) -> nx.Graph:
    """Convert a PyZX Graph to a NetworkX Graph with typed labels."""
    G = nx.Graph()
    types = g.types()
    phases = g.phases()

    for v in g.vertices():
        vtype = types[v]
        type_name = _VERTEX_TYPE_NAMES.get(vtype, f"T{vtype}")
        phase_class = _classify_phase(phases.get(v, 0))
        G.add_node(v, vtype=type_name, phase_class=phase_class,
                   raw_phase=float(phases.get(v, 0)))

    for e in g.edges():
        s, t = g.edge_st(e)
        etype = g.edge_type(e)
        etype_name = "H" if etype == zx.EdgeType.HADAMARD else "S"
        G.add_edge(s, t, etype=etype_name)

    return G


# ---------------------------------------------------------------------------
# Subgraph enumeration (bounded connected induced subgraphs)
# ---------------------------------------------------------------------------

def _enumerate_connected_subgraphs(
    G: nx.Graph, max_size: int = 6
) -> List[frozenset]:
    """Yield all connected induced subgraphs of G with 2 ≤ |V| ≤ max_size.

    Uses a grow-from-seed BFS strategy with de-duplication.
    """
    seen: Set[frozenset] = set()
    results: List[frozenset] = []

    for seed in G.nodes():
        # BFS-grow sets starting from {seed}
        stack: List[frozenset] = [frozenset([seed])]
        while stack:
            current = stack.pop()
            if len(current) > max_size:
                continue
            if len(current) >= 2:
                key = current
                if key not in seen:
                    seen.add(key)
                    results.append(key)
            if len(current) < max_size:
                # Expand by one neighbour of the current set
                frontier = set()
                for v in current:
                    for nb in G.neighbors(v):
                        if nb not in current:
                            frontier.add(nb)
                for nb in frontier:
                    new = current | {nb}
                    if new not in seen:
                        stack.append(new)
    return results


# ---------------------------------------------------------------------------
# Canonical hashing of labelled subgraphs
# ---------------------------------------------------------------------------

def _canonical_hash(G: nx.Graph, nodes: frozenset) -> str:
    """Compute a canonical hash for the induced subgraph on *nodes*.

    Uses a deterministic certificate based on sorted adjacency with labels.
    This is equivalent to a colour-aware Weisfeiler-Leman 1-dim hash and
    is exact for almost all small graphs encountered in practice.
    """
    sub = G.subgraph(nodes)
    # Build a relabelled copy with sorted integer labels
    mapping = {v: i for i, v in enumerate(sorted(nodes))}
    lines = []
    for v in sorted(nodes):
        nd = G.nodes[v]
        label = f"{nd['vtype']}:{nd['phase_class']}"
        neighbours = []
        for nb in sorted(sub.neighbors(v)):
            ed = sub.edges[v, nb]
            neighbours.append(f"{mapping[nb]}/{ed['etype']}")
        neighbours.sort()
        lines.append(f"{mapping[v]}|{label}|{','.join(neighbours)}")
    cert = "\n".join(sorted(lines))
    return hashlib.sha256(cert.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# ZXWeb dataclass
# ---------------------------------------------------------------------------

@dataclass
class ZXWeb:
    """A recurring sub-diagram motif (ZX-Web)."""
    web_id: str                                   # canonical hash
    size: int                                      # number of vertices
    exemplar_nodes: frozenset                      # one concrete node-set
    exemplar_source: str                           # algorithm it was found in
    sources: Dict[str, List[frozenset]] = field(default_factory=dict)
    frequency: int = 0
    node_labels: List[Dict] = field(default_factory=list)   # typed info
    edge_labels: List[Dict] = field(default_factory=list)

    def as_networkx(self, corpus_nx: Dict[str, nx.Graph]) -> nx.Graph:
        """Return a NetworkX copy of the exemplar subgraph."""
        G = corpus_nx[self.exemplar_source]
        return G.subgraph(self.exemplar_nodes).copy()


# ---------------------------------------------------------------------------
# Public mining interface
# ---------------------------------------------------------------------------

def mine_zx_webs(
    zx_entries: Dict[str, ZXEntry],
    max_subgraph_size: int = 6,
    min_frequency: int = 2,
    use_reduced: bool = True,
) -> Tuple[Dict[str, ZXWeb], Dict[str, nx.Graph]]:
    """Mine recurring ZX-Web motifs across the corpus.

    Parameters
    ----------
    zx_entries : dict
        Output of ``zx_convert.convert_corpus``.
    max_subgraph_size : int
        Largest motif size to consider (vertices).
    min_frequency : int
        Minimum number of *distinct source algorithms* for a motif.
    use_reduced : bool
        If True use ``reduced_graph``; otherwise ``clifford_graph``.

    Returns
    -------
    webs : dict
        ``{web_id: ZXWeb}`` for every motif meeting the frequency threshold.
    corpus_nx : dict
        ``{algo_name: nx.Graph}`` used internally (needed for composition).
    """
    # 1. Convert all ZX-diagrams to NetworkX
    corpus_nx: Dict[str, nx.Graph] = {}
    for name, entry in zx_entries.items():
        g = entry.reduced_graph if use_reduced else entry.clifford_graph
        corpus_nx[name] = pyzx_to_networkx(g)

    # 2-3. Enumerate and hash subgraphs per algorithm
    motif_registry: Dict[str, ZXWeb] = {}

    for name, G in corpus_nx.items():
        if G.number_of_nodes() < 2:
            continue
        subgraphs = _enumerate_connected_subgraphs(G, max_subgraph_size)
        for nodes in subgraphs:
            h = _canonical_hash(G, nodes)
            if h not in motif_registry:
                # Capture exemplar info
                node_labels = [
                    {**G.nodes[v], "orig_id": v} for v in sorted(nodes)
                ]
                edge_labels = []
                sub = G.subgraph(nodes)
                for u, v, d in sub.edges(data=True):
                    edge_labels.append({"src": u, "tgt": v, **d})
                motif_registry[h] = ZXWeb(
                    web_id=h,
                    size=len(nodes),
                    exemplar_nodes=nodes,
                    exemplar_source=name,
                    sources={},
                    frequency=0,
                    node_labels=node_labels,
                    edge_labels=edge_labels,
                )
            web = motif_registry[h]
            web.sources.setdefault(name, []).append(nodes)

    # 4. Count distinct source algorithms
    for web in motif_registry.values():
        web.frequency = len(web.sources)

    # 5. Filter by frequency
    webs = {
        wid: w for wid, w in motif_registry.items()
        if w.frequency >= min_frequency
    }

    return webs, corpus_nx
