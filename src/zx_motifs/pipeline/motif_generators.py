"""
Three strategies for generating candidate motifs to search for:

1. Top-down: Hand-craft motifs from known algorithmic primitives.
2. Bottom-up: Enumerate small connected subgraphs and keep recurring ones.
3. Hybrid: Extract neighborhoods around "interesting" vertices and cluster.
"""
import hashlib
from typing import Callable

import networkx as nx
from networkx.algorithms import isomorphism

from .featurizer import extract_local_neighborhood
from .matcher import (
    PHASE_ANY,
    PHASE_ANY_NONCLIFFORD,
    PHASE_ANY_NONZERO,
    MotifPattern,
    node_match_fn,
    edge_match_fn,
)


# ════════════════════════════════════════════════════════════════════
# Hashing: WL hash for better deduplication quality
# ════════════════════════════════════════════════════════════════════


def _ensure_wl_label(subg: nx.Graph) -> nx.Graph:
    """Set ``wl_label`` on each node from vertex_type + phase_class."""
    for n, d in subg.nodes(data=True):
        d["wl_label"] = f"{d.get('vertex_type', '?')}_{d.get('phase_class', '?')}"
    return subg


def wl_hash(subg: nx.Graph, iterations: int = 3) -> str:
    """Weisfeiler-Leman graph hash using node and edge labels."""
    _ensure_wl_label(subg)
    return nx.weisfeiler_lehman_graph_hash(
        subg,
        node_attr="wl_label",
        edge_attr="edge_type",
        iterations=iterations,
        digest_size=16,
    )


_HASH_FN: Callable[[nx.Graph], str] = wl_hash


def get_hash_fn() -> Callable[[nx.Graph], str]:
    """Return the current hash function used for motif deduplication."""
    return _HASH_FN


def set_hash_fn(fn: Callable[[nx.Graph], str]) -> None:
    """Override the hash function used for motif deduplication."""
    global _HASH_FN
    _HASH_FN = fn


# ════════════════════════════════════════════════════════════════════
# Strategy 1: Hand-crafted motifs from known primitives
# ════════════════════════════════════════════════════════════════════


def make_phase_gadget_motif(n_targets: int = 2) -> MotifPattern:
    """
    Phase gadget: a Z-spider with a non-Clifford phase connected via
    Hadamard edges to n_targets phaseless Z-spiders.
    """
    g = nx.Graph()
    g.add_node(0, vertex_type="Z", phase_class="t_like", is_boundary=False)
    g.add_node(1, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_edge(0, 1, edge_type="HADAMARD")

    for i in range(n_targets):
        node_id = i + 2
        g.add_node(node_id, vertex_type="Z", phase_class="zero", is_boundary=False)
        g.add_edge(1, node_id, edge_type="HADAMARD")

    return MotifPattern(
        motif_id=f"phase_gadget_{n_targets}t",
        graph=g,
        source="hand_crafted",
        description=f"Phase gadget with {n_targets} targets",
    )


def make_cx_spider_motif() -> MotifPattern:
    """CNOT in ZX: a Z-spider connected to an X-spider via a simple edge."""
    g = nx.Graph()
    g.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(1, vertex_type="X", phase_class="zero", is_boundary=False)
    g.add_edge(0, 1, edge_type="SIMPLE")
    return MotifPattern(
        motif_id="cx_pair",
        graph=g,
        source="hand_crafted",
        description="CNOT as Z-X spider pair",
    )


def make_hadamard_sandwich_motif() -> MotifPattern:
    """
    Clifford Z-spider between two phaseless Z-spiders via H-edges.
    Appears in controlled-phase gate ZX representations.
    """
    g = nx.Graph()
    g.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(1, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(2, vertex_type="Z", phase_class="clifford", is_boundary=False)
    g.add_edge(0, 2, edge_type="HADAMARD")
    g.add_edge(2, 1, edge_type="HADAMARD")
    return MotifPattern(
        motif_id="hadamard_sandwich",
        graph=g,
        source="hand_crafted",
        description="Clifford Z-spider between two phaseless Z-spiders via H-edges",
    )


def make_zz_interaction_motif() -> MotifPattern:
    """ZZ interaction: core of QAOA problem unitaries (CX-Rz-CX pattern)."""
    g = nx.Graph()
    g.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(1, vertex_type="Z", phase_class="arbitrary", is_boundary=False)
    g.add_node(2, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_edge(0, 1, edge_type="SIMPLE")
    g.add_edge(1, 2, edge_type="SIMPLE")
    return MotifPattern(
        motif_id="zz_interaction",
        graph=g,
        source="hand_crafted",
        description="ZZ interaction pattern from QAOA/VQE",
    )


def make_syndrome_extraction_motif() -> MotifPattern:
    """Syndrome extraction fan-out: a Z-spider connected via simple edges
    to two X-spiders (data-to-ancilla CX pattern in error correction)."""
    g = nx.Graph()
    g.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(1, vertex_type="X", phase_class="zero", is_boundary=False)
    g.add_node(2, vertex_type="X", phase_class="zero", is_boundary=False)
    g.add_edge(0, 1, edge_type="SIMPLE")
    g.add_edge(0, 2, edge_type="SIMPLE")
    return MotifPattern(
        motif_id="syndrome_extraction",
        graph=g,
        source="hand_crafted",
        description="Syndrome extraction fan-out from error correction codes",
    )


def make_toffoli_core_motif() -> MotifPattern:
    """Core of decomposed Toffoli gate: chain of T-like Z-spiders
    connected by simple edges (from Clifford+T decomposition)."""
    g = nx.Graph()
    g.add_node(0, vertex_type="Z", phase_class="t_like", is_boundary=False)
    g.add_node(1, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(2, vertex_type="Z", phase_class="t_like", is_boundary=False)
    g.add_edge(0, 1, edge_type="SIMPLE")
    g.add_edge(1, 2, edge_type="SIMPLE")
    return MotifPattern(
        motif_id="toffoli_core",
        graph=g,
        source="hand_crafted",
        description="T-gate chain from decomposed Toffoli gate",
    )


def make_cluster_chain_motif() -> MotifPattern:
    """Cluster state chain: three Z-spiders connected via Hadamard edges.
    Characteristic of 1D cluster/graph states (MBQC building block)."""
    g = nx.Graph()
    g.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(1, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(2, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_edge(0, 1, edge_type="HADAMARD")
    g.add_edge(1, 2, edge_type="HADAMARD")
    return MotifPattern(
        motif_id="cluster_chain",
        graph=g,
        source="hand_crafted",
        description="Hadamard-edge chain from cluster/graph states",
    )


def make_trotter_layer_motif() -> MotifPattern:
    """Trotter layer: ZZ interaction (Z-Z-Z chain via simple edges) adjacent
    to a single-qubit rotation spider. Core of Trotterized simulation."""
    g = nx.Graph()
    g.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(1, vertex_type="Z", phase_class="arbitrary", is_boundary=False)
    g.add_node(2, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(3, vertex_type="Z", phase_class="arbitrary", is_boundary=False)
    g.add_edge(0, 1, edge_type="SIMPLE")
    g.add_edge(1, 2, edge_type="SIMPLE")
    g.add_edge(2, 3, edge_type="SIMPLE")
    return MotifPattern(
        motif_id="trotter_layer",
        graph=g,
        source="hand_crafted",
        description="Alternating ZZ-interaction + rotation from Trotter simulation",
    )


_HANDCRAFTED_MOTIFS_INLINE = [
    make_phase_gadget_motif(2),
    make_phase_gadget_motif(3),
    make_cx_spider_motif(),
    make_hadamard_sandwich_motif(),
    make_zz_interaction_motif(),
    make_syndrome_extraction_motif(),
    make_toffoli_core_motif(),
    make_cluster_chain_motif(),
    make_trotter_layer_motif(),
]


# ════════════════════════════════════════════════════════════════════
# Phase-parametric motifs (wildcard phase classes)
# ════════════════════════════════════════════════════════════════════


def make_syndrome_extraction_param_motif() -> MotifPattern:
    """Syndrome extraction with any-phase center Z-spider."""
    g = nx.Graph()
    g.add_node(0, vertex_type="Z", phase_class=PHASE_ANY, is_boundary=False)
    g.add_node(1, vertex_type="X", phase_class="zero", is_boundary=False)
    g.add_node(2, vertex_type="X", phase_class="zero", is_boundary=False)
    g.add_edge(0, 1, edge_type="SIMPLE")
    g.add_edge(0, 2, edge_type="SIMPLE")
    return MotifPattern(
        motif_id="syndrome_extraction_param",
        graph=g,
        source="hand_crafted",
        description="Syndrome extraction fan-out with any-phase center",
    )


def make_zz_interaction_param_motif() -> MotifPattern:
    """ZZ interaction with any-nonzero center phase."""
    g = nx.Graph()
    g.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(1, vertex_type="Z", phase_class=PHASE_ANY_NONZERO, is_boundary=False)
    g.add_node(2, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_edge(0, 1, edge_type="SIMPLE")
    g.add_edge(1, 2, edge_type="SIMPLE")
    return MotifPattern(
        motif_id="zz_interaction_param",
        graph=g,
        source="hand_crafted",
        description="ZZ interaction with any-nonzero center phase",
    )


def make_toffoli_core_param_motif() -> MotifPattern:
    """Toffoli core with any-nonclifford outer phases."""
    g = nx.Graph()
    g.add_node(0, vertex_type="Z", phase_class=PHASE_ANY_NONCLIFFORD, is_boundary=False)
    g.add_node(1, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(2, vertex_type="Z", phase_class=PHASE_ANY_NONCLIFFORD, is_boundary=False)
    g.add_edge(0, 1, edge_type="SIMPLE")
    g.add_edge(1, 2, edge_type="SIMPLE")
    return MotifPattern(
        motif_id="toffoli_core_param",
        graph=g,
        source="hand_crafted",
        description="Toffoli core with any-nonclifford outer phases",
    )


def make_x_hub_3z_param_motif() -> MotifPattern:
    """X-spider hub connected to 3 Z-spiders with any phase."""
    g = nx.Graph()
    g.add_node(0, vertex_type="X", phase_class="zero", is_boundary=False)
    g.add_node(1, vertex_type="Z", phase_class=PHASE_ANY, is_boundary=False)
    g.add_node(2, vertex_type="Z", phase_class=PHASE_ANY, is_boundary=False)
    g.add_node(3, vertex_type="Z", phase_class=PHASE_ANY, is_boundary=False)
    g.add_edge(0, 1, edge_type="SIMPLE")
    g.add_edge(0, 2, edge_type="SIMPLE")
    g.add_edge(0, 3, edge_type="SIMPLE")
    return MotifPattern(
        motif_id="x_hub_3z_param",
        graph=g,
        source="hand_crafted",
        description="X-spider hub connected to 3 Z-spiders (any phase)",
    )


def make_hadamard_pauli_pair_motif() -> MotifPattern:
    """Z(zero) connected to X(pauli) via HADAMARD edge."""
    g = nx.Graph()
    g.add_node(0, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(1, vertex_type="X", phase_class="pauli", is_boundary=False)
    g.add_edge(0, 1, edge_type="HADAMARD")
    return MotifPattern(
        motif_id="hadamard_pauli_pair",
        graph=g,
        source="hand_crafted",
        description="Z(zero)-X(pauli) pair via Hadamard edge",
    )


def make_pauli_x_hub_3z_motif() -> MotifPattern:
    """X(pauli) hub connected to 3 Z(zero) spiders."""
    g = nx.Graph()
    g.add_node(0, vertex_type="X", phase_class="pauli", is_boundary=False)
    g.add_node(1, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(2, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_node(3, vertex_type="Z", phase_class="zero", is_boundary=False)
    g.add_edge(0, 1, edge_type="SIMPLE")
    g.add_edge(0, 2, edge_type="SIMPLE")
    g.add_edge(0, 3, edge_type="SIMPLE")
    return MotifPattern(
        motif_id="pauli_x_hub_3z",
        graph=g,
        source="hand_crafted",
        description="X(pauli) hub connected to 3 Z(zero) spiders",
    )


_PARAMETRIC_MOTIFS_INLINE = [
    make_syndrome_extraction_param_motif(),
    make_zz_interaction_param_motif(),
    make_toffoli_core_param_motif(),
    make_x_hub_3z_param_motif(),
    make_hadamard_pauli_pair_motif(),
    make_pauli_x_hub_3z_motif(),
]

HANDCRAFTED_MOTIFS = [
    make_phase_gadget_motif(2),
    make_phase_gadget_motif(3),
    make_cx_spider_motif(),
    make_hadamard_sandwich_motif(),
    make_zz_interaction_motif(),
    make_syndrome_extraction_motif(),
    make_toffoli_core_motif(),
    make_cluster_chain_motif(),
    make_trotter_layer_motif(),
]

PARAMETRIC_MOTIFS = [
    make_syndrome_extraction_param_motif(),
    make_zz_interaction_param_motif(),
    make_toffoli_core_param_motif(),
    make_x_hub_3z_param_motif(),
    make_hadamard_pauli_pair_motif(),
    make_pauli_x_hub_3z_motif(),
]

EXTENDED_MOTIFS = HANDCRAFTED_MOTIFS + PARAMETRIC_MOTIFS


# ════════════════════════════════════════════════════════════════════
# Strategy 2: Bottom-up enumeration of small connected subgraphs
# ════════════════════════════════════════════════════════════════════


def canonical_hash(subg: nx.Graph) -> str:
    """
    Compute a fast (approximate) canonical hash of a labeled subgraph.
    Two subgraphs with the same hash are *likely* isomorphic, but
    collisions are possible. Always confirm with VF2 before trusting.
    """
    node_labels = sorted(
        (d.get("vertex_type", "?"), d.get("phase_class", "?"))
        for _, d in subg.nodes(data=True)
    )
    edge_labels = sorted(
        (
            min(
                subg.nodes[u].get("vertex_type", "?"),
                subg.nodes[v].get("vertex_type", "?"),
            ),
            max(
                subg.nodes[u].get("vertex_type", "?"),
                subg.nodes[v].get("vertex_type", "?"),
            ),
            d.get("edge_type", "?"),
        )
        for u, v, d in subg.edges(data=True)
    )
    degree_seq = sorted(subg.degree(n) for n in subg.nodes())
    raw = str((node_labels, edge_labels, degree_seq))
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _is_isomorphic(g1: nx.Graph, g2: nx.Graph) -> bool:
    """Check labeled isomorphism between two small graphs."""
    gm = isomorphism.GraphMatcher(
        g1, g2, node_match=node_match_fn, edge_match=edge_match_fn
    )
    return gm.is_isomorphic()


def enumerate_connected_subgraphs(
    host: nx.Graph,
    min_size: int = 3,
    max_size: int = 6,
    max_subgraphs: int = 500,
    exclude_boundary: bool = True,
) -> list[nx.Graph]:
    """
    Enumerate small connected induced subgraphs via recursive expansion.

    Grows subgraphs one node at a time from each starting vertex,
    collecting unique subgraphs of each size along the way.
    """
    interior_nodes = sorted(
        n for n, d in host.nodes(data=True)
        if not (exclude_boundary and d.get("is_boundary"))
    )
    interior_set = set(interior_nodes)

    subgraphs: list[nx.Graph] = []
    seen_hashes: dict[str, nx.Graph] = {}  # hash → representative graph

    def _expand(node_set: frozenset, candidates: frozenset) -> None:
        """Recursively expand node_set by adding neighbors from candidates."""
        if len(subgraphs) >= max_subgraphs:
            return

        size = len(node_set)
        if size >= min_size:
            subg = host.subgraph(node_set).copy()
            h = _HASH_FN(subg)
            if h not in seen_hashes:
                seen_hashes[h] = subg
                subgraphs.append(subg)
            elif not _is_isomorphic(seen_hashes[h], subg):
                # Hash collision — store under disambiguated key
                alt_h = h + f"_{len(seen_hashes)}"
                if alt_h not in seen_hashes:
                    seen_hashes[alt_h] = subg
                    subgraphs.append(subg)

        if size >= max_size:
            return

        # Find all neighbors of current set that are valid candidates
        new_candidates = set()
        for n in node_set:
            for nbr in host.neighbors(n):
                if nbr in interior_set and nbr not in node_set and nbr in candidates:
                    new_candidates.add(nbr)

        for nbr in sorted(new_candidates):
            # Only consider candidates > nbr to avoid duplicates
            remaining = frozenset(c for c in new_candidates if c > nbr)
            _expand(node_set | {nbr}, remaining)

    for start in interior_nodes:
        if len(subgraphs) >= max_subgraphs:
            break
        # Candidates: all interior nodes > start (canonical ordering avoids dupes)
        initial_candidates = frozenset(n for n in interior_nodes if n > start)
        _expand(frozenset({start}), initial_candidates)

    return [s for s in subgraphs if s.number_of_nodes() >= min_size]


def find_recurring_subgraphs(
    corpus: dict,
    target_level: str = "spider_fused",
    min_size: int = 3,
    max_size: int = 6,
    min_algorithms: int = 2,
) -> list[MotifPattern]:
    """
    Bottom-up motif discovery: enumerate subgraphs, hash them,
    keep the ones that appear across multiple algorithms.
    """
    # hash → (representative graph, set of algorithm names)
    hash_registry: dict[str, tuple[nx.Graph, set[str]]] = {}

    for (algo_name, level), host_graph in corpus.items():
        if level != target_level:
            continue

        subgraphs = enumerate_connected_subgraphs(
            host_graph, min_size=min_size, max_size=max_size,
        )

        for sg in subgraphs:
            h = _HASH_FN(sg)

            if h in hash_registry:
                existing, algo_set = hash_registry[h]
                # Confirm isomorphism to guard against hash collisions
                if _is_isomorphic(existing, sg):
                    algo_set.add(algo_name)
                else:
                    alt_h = h + f"_{algo_name}"
                    if alt_h not in hash_registry:
                        hash_registry[alt_h] = (sg, {algo_name})
                    else:
                        hash_registry[alt_h][1].add(algo_name)
            else:
                hash_registry[h] = (sg, {algo_name})

    # Filter to motifs appearing in enough distinct algorithms
    motifs = []
    for h, (sg, algo_set) in sorted(
        hash_registry.items(), key=lambda x: -len(x[1][1])
    ):
        if len(algo_set) < min_algorithms:
            continue

        motifs.append(
            MotifPattern(
                motif_id=f"auto_{h}",
                graph=sg,
                source="bottom_up",
                description=(
                    f"Auto-discovered {sg.number_of_nodes()}-node motif, "
                    f"found in {len(algo_set)} algorithms: "
                    f"{', '.join(sorted(algo_set))}"
                ),
            )
        )

    return motifs


def find_recurring_subgraphs_multilevel(
    corpus: dict,
    levels: list[str] | None = None,
    min_size: int = 3,
    max_size: int = 6,
    min_algorithms: int = 2,
) -> list[MotifPattern]:
    """
    Bottom-up motif discovery across multiple simplification levels.

    Runs find_recurring_subgraphs at each level, then deduplicates across
    levels using isomorphism. Motifs found at multiple levels get a
    discovery_levels list tracking where they appeared.
    """
    if levels is None:
        levels = sorted({lvl for (_, lvl) in corpus.keys()})

    # Collect motifs per level
    all_motifs: list[tuple[MotifPattern, str]] = []  # (motif, level)
    for level in levels:
        level_motifs = find_recurring_subgraphs(
            corpus,
            target_level=level,
            min_size=min_size,
            max_size=max_size,
            min_algorithms=min_algorithms,
        )
        for m in level_motifs:
            all_motifs.append((m, level))

    # Cross-level deduplication
    # Each entry: (representative MotifPattern, set of levels)
    deduped: list[tuple[MotifPattern, set[str]]] = []

    for motif, level in all_motifs:
        merged = False
        for existing_motif, level_set in deduped:
            if _is_isomorphic(existing_motif.graph, motif.graph):
                level_set.add(level)
                # Merge algorithm info into description
                merged = True
                break
        if not merged:
            deduped.append((motif, {level}))

    # Build final list with discovery_levels
    result = []
    for motif, level_set in deduped:
        motif.discovery_levels = sorted(level_set)
        result.append(motif)

    return result


# ════════════════════════════════════════════════════════════════════
# Strategy 3: Neighborhood extraction around interesting vertices
# ════════════════════════════════════════════════════════════════════


def extract_interesting_neighborhoods(
    host: nx.Graph,
    radius: int = 2,
    interest_criteria: str = "non_clifford",
) -> list[nx.Graph]:
    """
    Extract local neighborhoods around 'interesting' vertices.

    Criteria:
      - "non_clifford": vertices with T-like or arbitrary phases
      - "high_degree": vertices with degree > 4
      - "color_boundary": vertices adjacent to both Z and X spiders
    """
    interesting = []
    for n, data in host.nodes(data=True):
        if data.get("is_boundary"):
            continue

        if interest_criteria == "non_clifford":
            if data.get("phase_class") in ("t_like", "arbitrary"):
                interesting.append(n)
        elif interest_criteria == "high_degree":
            if host.degree(n) > 4:
                interesting.append(n)
        elif interest_criteria == "color_boundary":
            my_type = data.get("vertex_type")
            neighbor_types = {
                host.nodes[nbr].get("vertex_type") for nbr in host.neighbors(n)
            }
            if neighbor_types - {my_type, "BOUNDARY"}:
                interesting.append(n)

    neighborhoods = []
    seen_hashes: set[str] = set()

    for center in interesting:
        neighborhood = extract_local_neighborhood(host, center, radius)
        # Remove boundary nodes
        interior = {n for n in neighborhood.nodes() if not host.nodes[n].get("is_boundary")}
        if len(interior) >= 3:
            subg = host.subgraph(interior).copy()
            if nx.is_connected(subg):
                h = _HASH_FN(subg)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    neighborhoods.append(subg)

    return neighborhoods


def find_neighborhood_motifs(
    corpus: dict,
    target_level: str = "spider_fused",
    radius: int = 2,
    criteria: list[str] | None = None,
    min_algorithms: int = 2,
) -> list[MotifPattern]:
    """
    Strategy 3: Extract neighborhoods around interesting vertices across
    the corpus and return those that recur in multiple algorithms.

    Args:
        corpus: {(algo_name, level): nx.Graph}
        target_level: Which simplification level to search.
        radius: BFS radius for neighborhood extraction.
        criteria: Interest criteria to apply. Defaults to all three.
        min_algorithms: Minimum algorithms a neighborhood must appear in.

    Returns:
        List of MotifPattern with source="neighborhood".
    """
    if criteria is None:
        criteria = ["non_clifford", "high_degree", "color_boundary"]

    # hash → (representative graph, set of algorithm names)
    hash_registry: dict[str, tuple[nx.Graph, set[str]]] = {}

    for (algo_name, level), host_graph in corpus.items():
        if level != target_level:
            continue

        for criterion in criteria:
            neighborhoods = extract_interesting_neighborhoods(
                host_graph, radius=radius, interest_criteria=criterion
            )
            for subg in neighborhoods:
                h = _HASH_FN(subg)
                if h in hash_registry:
                    existing, algo_set = hash_registry[h]
                    if _is_isomorphic(existing, subg):
                        algo_set.add(algo_name)
                    else:
                        alt_h = h + f"_{algo_name}"
                        if alt_h not in hash_registry:
                            hash_registry[alt_h] = (subg, {algo_name})
                        else:
                            hash_registry[alt_h][1].add(algo_name)
                else:
                    hash_registry[h] = (subg, {algo_name})

    motifs = []
    for h, (sg, algo_set) in sorted(
        hash_registry.items(), key=lambda x: -len(x[1][1])
    ):
        if len(algo_set) < min_algorithms:
            continue

        motifs.append(
            MotifPattern(
                motif_id=f"nbr_{h}",
                graph=sg,
                source="neighborhood",
                description=(
                    f"Neighborhood-extracted {sg.number_of_nodes()}-node motif, "
                    f"found in {len(algo_set)} algorithms: "
                    f"{', '.join(sorted(algo_set))}"
                ),
            )
        )

    return motifs
