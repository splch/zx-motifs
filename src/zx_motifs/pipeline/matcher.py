"""
Subgraph isomorphism engine for ZX diagrams.

Uses NetworkX's VF2 implementation with semantic node/edge matching
to find occurrences of small ZX patterns in larger diagrams.

Key design decisions:
  1. Node matching respects vertex_type and phase_class (not exact phase).
  2. Edge matching respects edge_type (SIMPLE vs HADAMARD).
  3. Boundary nodes are excluded from interior matching — we only match
     the "guts" of a pattern, not its I/O connections.
  4. Phase wildcards allow parametric motifs (any, any_nonzero, any_nonclifford).

This gives you commutation-aware pattern detection "for free" because
ZX diagrams have already absorbed commutation relations into their topology.
"""
from collections import Counter
from dataclasses import dataclass, field

import networkx as nx
from networkx.algorithms import isomorphism


# ── Phase Wildcard Constants ──────────────────────────────────────

PHASE_ANY = "any"
PHASE_ANY_NONZERO = "any_nonzero"
PHASE_ANY_NONCLIFFORD = "any_nonclifford"

_PHASE_WILDCARDS = {PHASE_ANY, PHASE_ANY_NONZERO, PHASE_ANY_NONCLIFFORD}

# Phases rejected by each wildcard
_NONZERO_REJECT = {"zero"}
_NONCLIFFORD_REJECT = {"zero", "clifford", "pauli"}


@dataclass
class MotifMatch:
    """A single match of a motif pattern in a host graph."""

    motif_id: str
    host_algorithm: str
    host_level: str
    mapping: dict[int, int]  # {pattern_node: host_node}
    matched_node_types: list[str]


@dataclass
class MotifPattern:
    """A candidate motif pattern to search for."""

    motif_id: str
    graph: nx.Graph
    source: str  # Which algorithm / strategy it came from
    description: str = ""
    occurrences: list[MotifMatch] = field(default_factory=list)
    discovery_levels: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ── Phase Wildcard Matching ────────────────────────────────────────


def phase_class_matches(pattern_phase: str, host_phase: str) -> bool:
    """
    Check if a host's phase_class satisfies a pattern's phase_class.

    Supports wildcards in the pattern:
      - "any": matches any host phase
      - "any_nonzero": matches everything except "zero"
      - "any_nonclifford": matches only "t_like" and "arbitrary"
      - exact string: must match exactly
    """
    if pattern_phase == PHASE_ANY:
        return True
    if pattern_phase == PHASE_ANY_NONZERO:
        return host_phase not in _NONZERO_REJECT
    if pattern_phase == PHASE_ANY_NONCLIFFORD:
        return host_phase not in _NONCLIFFORD_REJECT
    return pattern_phase == host_phase


def is_parametric_motif(pattern: nx.Graph) -> bool:
    """Check if any node in the pattern uses a phase wildcard."""
    for _, d in pattern.nodes(data=True):
        if d.get("phase_class") in _PHASE_WILDCARDS:
            return True
    return False


# ── Semantic Matching Functions ──────────────────────────────────────


def node_match_fn(n1_attrs: dict, n2_attrs: dict) -> bool:
    """
    Node compatibility for VF2 subgraph isomorphism.

    In VF2's GraphMatcher(host, pattern), n1=host, n2=pattern.
    Matches on vertex_type exactly and phase_class with wildcard support.
    """
    if n1_attrs.get("vertex_type") != n2_attrs.get("vertex_type"):
        return False
    pattern_phase = n2_attrs.get("phase_class", "")
    host_phase = n1_attrs.get("phase_class", "")
    return phase_class_matches(pattern_phase, host_phase)


def edge_match_fn(e1_attrs: dict, e2_attrs: dict) -> bool:
    """Edge compatibility: SIMPLE and HADAMARD edges are distinct."""
    return e1_attrs.get("edge_type") == e2_attrs.get("edge_type")


# ── Pre-Filtering ───────────────────────────────────────────────────


def can_possibly_match(pattern: nx.Graph, host: nx.Graph) -> bool:
    """
    O(V+E) necessary-condition check: can `pattern` possibly be a
    subgraph of `host`?  Returns False only when it's provably impossible.

    Checks:
      - Host has >= pattern's count of each vertex_type
      - Host has >= pattern's count of each edge_type
      - Pattern's max degree <= host's max degree
    """
    # Vertex type counts
    p_types: Counter = Counter(
        d.get("vertex_type") for _, d in pattern.nodes(data=True)
    )
    h_types: Counter = Counter(
        d.get("vertex_type") for _, d in host.nodes(data=True)
    )
    for vtype, count in p_types.items():
        if h_types.get(vtype, 0) < count:
            return False

    # Edge type counts
    p_etypes: Counter = Counter(
        d.get("edge_type") for _, _, d in pattern.edges(data=True)
    )
    h_etypes: Counter = Counter(
        d.get("edge_type") for _, _, d in host.edges(data=True)
    )
    for etype, count in p_etypes.items():
        if h_etypes.get(etype, 0) < count:
            return False

    # Max degree
    if pattern.number_of_nodes() > 0 and host.number_of_nodes() > 0:
        p_max_deg = max(d for _, d in pattern.degree())
        h_max_deg = max(d for _, d in host.degree())
        if p_max_deg > h_max_deg:
            return False

    return True


# ── Core Matching ────────────────────────────────────────────────────


def find_motif_in_graph(
    pattern: nx.Graph,
    host: nx.Graph,
    max_matches: int = 100,
    exclude_boundary: bool = True,
) -> list[dict]:
    """
    Find all subgraph isomorphism matches of `pattern` in `host`.

    Args:
        pattern: The motif to search for (small graph, ~3-10 nodes).
        host: The algorithm's ZX graph to search in.
        max_matches: Cap on matches to avoid combinatorial explosion.
        exclude_boundary: If True, remove BOUNDARY nodes from the host
            before matching (we want interior structure).

    Returns:
        List of node mappings {pattern_node: host_node}.
    """
    if exclude_boundary:
        interior_nodes = [
            n for n, d in host.nodes(data=True)
            if d.get("vertex_type") != "BOUNDARY"
        ]
        host_interior = host.subgraph(interior_nodes)
    else:
        host_interior = host

    if host_interior.number_of_nodes() < pattern.number_of_nodes():
        return []

    if not can_possibly_match(pattern, host_interior):
        return []

    gm = isomorphism.GraphMatcher(
        host_interior,
        pattern,
        node_match=node_match_fn,
        edge_match=edge_match_fn,
    )

    matches = []
    for mapping in gm.subgraph_isomorphisms_iter():
        # mapping is {host_node: pattern_node}; invert for our convention
        inv_mapping = {v: k for k, v in mapping.items()}
        matches.append(inv_mapping)
        if len(matches) >= max_matches:
            break

    return matches


def find_motif_across_corpus(
    pattern: MotifPattern,
    corpus: dict,  # {(algo_name, level): nx.Graph}
    target_level: str = "spider_fused",
    max_matches_per_graph: int = 50,
) -> MotifPattern:
    """
    Search for a motif across all algorithm graphs at a given level.
    Populates pattern.occurrences with MotifMatch objects.
    """
    pattern.occurrences = []

    for (algo_name, level), host_graph in corpus.items():
        if level != target_level:
            continue

        matches = find_motif_in_graph(
            pattern.graph,
            host_graph,
            max_matches=max_matches_per_graph,
        )

        for m in matches:
            pattern.occurrences.append(
                MotifMatch(
                    motif_id=pattern.motif_id,
                    host_algorithm=algo_name,
                    host_level=level,
                    mapping=m,
                    matched_node_types=[
                        host_graph.nodes[v]["vertex_type"] for v in m.values()
                    ],
                )
            )

    return pattern


def find_motif_across_corpus_multilevel(
    pattern: MotifPattern,
    corpus: dict,
    levels: list[str] | None = None,
    max_matches_per_graph: int = 50,
) -> MotifPattern:
    """
    Search for a motif across all algorithm graphs at multiple levels.
    If levels is None, searches all levels present in the corpus.
    Populates pattern.occurrences with MotifMatch objects from all levels.
    """
    if levels is None:
        levels = sorted({lvl for (_, lvl) in corpus.keys()})

    pattern.occurrences = []

    for (algo_name, level), host_graph in corpus.items():
        if level not in levels:
            continue

        matches = find_motif_in_graph(
            pattern.graph,
            host_graph,
            max_matches=max_matches_per_graph,
        )

        for m in matches:
            pattern.occurrences.append(
                MotifMatch(
                    motif_id=pattern.motif_id,
                    host_algorithm=algo_name,
                    host_level=level,
                    mapping=m,
                    matched_node_types=[
                        host_graph.nodes[v]["vertex_type"] for v in m.values()
                    ],
                )
            )

    return pattern


# ── Approximate Matching ────────────────────────────────────────────


@dataclass
class ApproximateMatch:
    """A near-miss match where some labels differ."""

    motif_id: str
    host_algorithm: str
    host_level: str
    host_subgraph_nodes: set[int]
    edit_distance: int
    similarity_score: float


def _count_label_mismatches(
    pattern: nx.Graph, host: nx.Graph, mapping: dict[int, int]
) -> int:
    """
    Count label mismatches between pattern and host under a given mapping.
    Checks vertex_type, phase_class (wildcard-aware) for nodes, and edge_type for edges.
    mapping: {pattern_node: host_node}
    """
    mismatches = 0
    for pn, hn in mapping.items():
        pd = pattern.nodes[pn]
        hd = host.nodes[hn]
        if pd.get("vertex_type") != hd.get("vertex_type"):
            mismatches += 1
        elif not phase_class_matches(pd.get("phase_class", ""), hd.get("phase_class", "")):
            mismatches += 1

    # Check edges
    for pu, pv, pd in pattern.edges(data=True):
        hu, hv = mapping[pu], mapping[pv]
        if host.has_edge(hu, hv):
            hd = host.edges[hu, hv]
            if pd.get("edge_type") != hd.get("edge_type"):
                mismatches += 1
        else:
            mismatches += 1  # missing edge

    return mismatches


def find_approximate_matches(
    pattern: nx.Graph,
    host: nx.Graph,
    max_edit_distance: int = 2,
    max_matches: int = 50,
    exclude_boundary: bool = True,
) -> list[ApproximateMatch]:
    """
    Find approximate subgraph matches by relaxing label constraints.

    Runs VF2 with topology-only matching (no label constraints), then
    counts label mismatches for each match and filters by edit distance.
    """
    if exclude_boundary:
        interior_nodes = [
            n for n, d in host.nodes(data=True)
            if d.get("vertex_type") != "BOUNDARY"
        ]
        host_interior = host.subgraph(interior_nodes)
    else:
        host_interior = host

    if host_interior.number_of_nodes() < pattern.number_of_nodes():
        return []

    # Topology-only VF2: always-True node/edge match
    gm = isomorphism.GraphMatcher(host_interior, pattern)

    results = []
    seen_node_sets: set[frozenset] = set()

    for mapping in gm.subgraph_isomorphisms_iter():
        # mapping is {host_node: pattern_node}
        inv_mapping = {v: k for k, v in mapping.items()}
        host_nodes = frozenset(inv_mapping.values())

        if host_nodes in seen_node_sets:
            continue
        seen_node_sets.add(host_nodes)

        dist = _count_label_mismatches(pattern, host_interior, inv_mapping)
        if dist <= max_edit_distance:
            # Similarity: 1 - (mismatches / total_labels)
            total = pattern.number_of_nodes() + pattern.number_of_edges()
            sim = 1.0 - (dist / max(total, 1))
            results.append(
                ApproximateMatch(
                    motif_id="",
                    host_algorithm="",
                    host_level="",
                    host_subgraph_nodes=set(host_nodes),
                    edit_distance=dist,
                    similarity_score=sim,
                )
            )
            if len(results) >= max_matches:
                break

    return results
