"""
Data-driven motif induction: analyze decomposition gaps, extract common
neighborhood patterns from uncovered vertices, and promote them to motifs.
"""
from dataclasses import dataclass

import networkx as nx

from .decomposer import decompose_graph, DecompositionResult
from .featurizer import extract_local_neighborhood
from .matcher import MotifPattern, find_motif_in_graph
from .motif_generators import _HASH_FN, _is_isomorphic


@dataclass
class NeighborhoodSignature:
    """1-hop structural signature of a vertex in a host graph."""

    center_type: str
    center_phase: str
    neighbor_types: tuple[str, ...]
    degree: int

    @property
    def key(self) -> str:
        nbrs = ",".join(self.neighbor_types)
        return f"{self.center_type}({self.center_phase})_d{self.degree}[{nbrs}]"


def compute_vertex_signature(host: nx.Graph, vertex: int) -> NeighborhoodSignature:
    """Compute the 1-hop neighborhood signature of a vertex."""
    d = host.nodes[vertex]
    neighbors = sorted(
        host.nodes[nbr].get("vertex_type", "?")
        for nbr in host.neighbors(vertex)
        if not host.nodes[nbr].get("is_boundary", False)
    )
    return NeighborhoodSignature(
        center_type=d.get("vertex_type", "?"),
        center_phase=d.get("phase_class", "?"),
        neighbor_types=tuple(neighbors),
        degree=len(neighbors),
    )


def analyze_uncovered_vertices(
    host: nx.Graph, decomposition: DecompositionResult
) -> dict[str, list[int]]:
    """Group uncovered vertices by their neighborhood signature key."""
    groups: dict[str, list[int]] = {}
    for v in decomposition.uncovered_vertices:
        sig = compute_vertex_signature(host, v)
        groups.setdefault(sig.key, []).append(v)
    return groups


def extract_motif_from_uncovered_cluster(
    host: nx.Graph,
    vertices: list[int],
    radius: int = 1,
    min_subgraph_size: int = 2,
    max_subgraph_size: int = 6,
) -> nx.Graph | None:
    """
    Extract a representative motif from a cluster of uncovered vertices.

    Uses the first vertex as exemplar, extracts its BFS neighborhood,
    filters to interior nodes, and re-indexes to 0..n-1.
    """
    if not vertices:
        return None

    exemplar = vertices[0]
    neighborhood = extract_local_neighborhood(host, exemplar, radius)

    # Filter to interior (non-boundary) nodes
    interior = [
        n for n in neighborhood.nodes()
        if not host.nodes[n].get("is_boundary", False)
    ]

    if len(interior) < min_subgraph_size:
        return None

    subg = host.subgraph(interior).copy()
    if not nx.is_connected(subg):
        # Take the largest connected component
        components = sorted(nx.connected_components(subg), key=len, reverse=True)
        subg = subg.subgraph(components[0]).copy()

    if subg.number_of_nodes() < min_subgraph_size:
        return None
    if subg.number_of_nodes() > max_subgraph_size:
        # Trim to max size by BFS from exemplar
        bfs_nodes = []
        for layer in nx.bfs_layers(subg, exemplar):
            for n in layer:
                bfs_nodes.append(n)
                if len(bfs_nodes) >= max_subgraph_size:
                    break
            if len(bfs_nodes) >= max_subgraph_size:
                break
        subg = subg.subgraph(bfs_nodes).copy()
        if not nx.is_connected(subg):
            return None

    # Re-index nodes to 0..n-1
    mapping = {old: new for new, old in enumerate(sorted(subg.nodes()))}
    return nx.relabel_nodes(subg, mapping)


def induce_motifs_from_gaps(
    corpus: dict,
    motif_library: list[MotifPattern],
    target_level: str = "spider_fused",
    min_occurrences: int = 5,
    min_algorithms: int = 2,
    radius: int = 1,
    max_new_motifs: int = 20,
) -> list[MotifPattern]:
    """
    Discover new motifs from decomposition gaps across a corpus.

    Pipeline:
      1. Decompose each algorithm with current library
      2. Collect uncovered vertex signatures across corpus
      3. Rank signatures by frequency × algorithm spread
      4. Extract representative motifs from top signatures
      5. Deduplicate via WL hash + VF2 isomorphism
      6. Validate each candidate gets real matches
    """
    # Step 1-2: Decompose and collect signatures
    sig_registry: dict[str, dict] = {}  # key → {vertices_by_algo, total_count}

    for (algo_name, level), host_graph in corpus.items():
        if level != target_level:
            continue
        decomp = decompose_graph(host_graph, motif_library)
        groups = analyze_uncovered_vertices(host_graph, decomp)
        for sig_key, verts in groups.items():
            if sig_key not in sig_registry:
                sig_registry[sig_key] = {
                    "vertices_by_algo": {},
                    "total_count": 0,
                    "exemplar_algo": None,
                    "exemplar_vertices": None,
                }
            entry = sig_registry[sig_key]
            entry["vertices_by_algo"][algo_name] = verts
            entry["total_count"] += len(verts)
            if entry["exemplar_algo"] is None:
                entry["exemplar_algo"] = algo_name
                entry["exemplar_vertices"] = verts

    # Step 3: Rank by frequency × algorithm spread
    ranked = sorted(
        sig_registry.items(),
        key=lambda x: x[1]["total_count"] * len(x[1]["vertices_by_algo"]),
        reverse=True,
    )

    # Step 4-6: Extract, deduplicate, validate
    new_motifs: list[MotifPattern] = []
    seen_hashes: dict[str, nx.Graph] = {}

    for sig_key, info in ranked:
        if len(new_motifs) >= max_new_motifs:
            break
        if info["total_count"] < min_occurrences:
            continue
        if len(info["vertices_by_algo"]) < min_algorithms:
            continue

        algo = info["exemplar_algo"]
        verts = info["exemplar_vertices"]
        host = corpus.get((algo, target_level))
        if host is None:
            continue

        candidate = extract_motif_from_uncovered_cluster(
            host, verts, radius=radius,
        )
        if candidate is None:
            continue

        # Deduplicate
        h = _HASH_FN(candidate)
        if h in seen_hashes:
            if _is_isomorphic(seen_hashes[h], candidate):
                continue
        seen_hashes[h] = candidate

        # Validate: check it actually matches somewhere
        total_matches = 0
        algos_matched = set()
        for (a, lvl), hg in corpus.items():
            if lvl != target_level:
                continue
            matches = find_motif_in_graph(candidate, hg, max_matches=10)
            if matches:
                total_matches += len(matches)
                algos_matched.add(a)

        if total_matches < min_occurrences or len(algos_matched) < min_algorithms:
            continue

        motif = MotifPattern(
            motif_id=f"induced_{h[:12]}",
            graph=candidate,
            source="data_driven",
            description=(
                f"Data-driven {candidate.number_of_nodes()}-node motif from sig {sig_key}, "
                f"{total_matches} matches in {len(algos_matched)} algorithms"
            ),
        )
        new_motifs.append(motif)

    return new_motifs


def iterative_induction(
    corpus: dict,
    initial_library: list[MotifPattern],
    target_level: str = "spider_fused",
    max_rounds: int = 3,
    min_coverage_gain: float = 0.02,
) -> tuple[list[MotifPattern], list[float]]:
    """
    Multi-round motif induction with coverage tracking.

    Each round discovers new motifs from gaps, adds them to the library,
    and measures coverage improvement. Stops when gains plateau.

    Returns:
        (final_library, coverage_history) where coverage_history[i] is
        the average coverage ratio after round i.
    """
    library = list(initial_library)
    coverage_history: list[float] = []

    for _round in range(max_rounds):
        # Measure current coverage
        coverages = []
        for (algo_name, level), host_graph in corpus.items():
            if level != target_level:
                continue
            decomp = decompose_graph(host_graph, library)
            coverages.append(decomp.coverage_ratio)

        avg_coverage = sum(coverages) / max(len(coverages), 1)
        coverage_history.append(avg_coverage)

        # Check stopping criterion
        if len(coverage_history) >= 2:
            gain = coverage_history[-1] - coverage_history[-2]
            if gain < min_coverage_gain:
                break

        # Induce new motifs
        new_motifs = induce_motifs_from_gaps(
            corpus, library, target_level=target_level,
        )
        if not new_motifs:
            break

        library.extend(new_motifs)

    # Final measurement if we added motifs in the last round
    if coverage_history:
        coverages = []
        for (algo_name, level), host_graph in corpus.items():
            if level != target_level:
                continue
            decomp = decompose_graph(host_graph, library)
            coverages.append(decomp.coverage_ratio)
        final = sum(coverages) / max(len(coverages), 1)
        if final != coverage_history[-1]:
            coverage_history.append(final)

    return library, coverage_history
