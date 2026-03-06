"""
Motif decomposition: invert the pipeline by decomposing a ZX graph
into its constituent motifs using a greedy set-cover approach.
"""
from dataclasses import dataclass, field

import networkx as nx

from .matcher import MotifPattern, find_motif_in_graph


@dataclass
class MotifPlacement:
    """A single motif placed onto a host graph."""

    motif_id: str
    host_vertices: set[int]
    mapping: dict[int, int]  # {pattern_node: host_node}


@dataclass
class DecompositionResult:
    """Result of decomposing a graph into motifs."""

    placements: list[MotifPlacement]
    covered_vertices: set[int]
    uncovered_vertices: set[int]
    coverage_ratio: float

    def summary(self) -> str:
        lines = [
            f"Decomposition: {len(self.placements)} placements, "
            f"{self.coverage_ratio:.1%} coverage",
        ]
        from collections import Counter
        motif_counts = Counter(p.motif_id for p in self.placements)
        for mid, count in motif_counts.most_common():
            lines.append(f"  {mid}: {count}x")
        lines.append(
            f"  Covered: {len(self.covered_vertices)} / "
            f"{len(self.covered_vertices) + len(self.uncovered_vertices)} vertices"
        )
        return "\n".join(lines)


def decompose_graph(
    host: nx.Graph,
    motif_library: list[MotifPattern],
    max_matches_per_motif: int = 100,
    exclude_boundary: bool = True,
    prefer_larger: bool = True,
) -> DecompositionResult:
    """
    Decompose a host graph into non-overlapping motif placements.

    Uses a greedy set-cover: find all matches of all motifs, then
    greedily select non-overlapping placements (largest motifs first).

    Args:
        host: The ZX graph to decompose.
        motif_library: Motifs to search for.
        max_matches_per_motif: Cap on matches per motif.
        exclude_boundary: Exclude BOUNDARY nodes from matching.
        prefer_larger: Sort candidates by motif size descending.

    Returns:
        DecompositionResult with coverage statistics.
    """
    # Determine interior vertices (the universe to cover)
    if exclude_boundary:
        interior = {
            n for n, d in host.nodes(data=True)
            if d.get("vertex_type") != "BOUNDARY"
        }
    else:
        interior = set(host.nodes())

    # Find all matches of all motifs
    candidates: list[MotifPlacement] = []
    for motif in motif_library:
        matches = find_motif_in_graph(
            motif.graph, host,
            max_matches=max_matches_per_motif,
            exclude_boundary=exclude_boundary,
        )
        for mapping in matches:
            host_verts = set(mapping.values())
            candidates.append(
                MotifPlacement(
                    motif_id=motif.motif_id,
                    host_vertices=host_verts,
                    mapping=mapping,
                )
            )

    # Sort: prefer larger motifs (greedy set cover heuristic)
    if prefer_larger:
        candidates.sort(key=lambda p: -len(p.host_vertices))

    # Greedy non-overlapping selection
    used_vertices: set[int] = set()
    placements: list[MotifPlacement] = []

    for cand in candidates:
        if cand.host_vertices & used_vertices:
            continue  # overlaps with already-placed motif
        placements.append(cand)
        used_vertices.update(cand.host_vertices)

    covered = used_vertices & interior
    uncovered = interior - covered
    ratio = len(covered) / max(len(interior), 1)

    return DecompositionResult(
        placements=placements,
        covered_vertices=covered,
        uncovered_vertices=uncovered,
        coverage_ratio=ratio,
    )


def decompose_across_corpus(
    corpus: dict,
    motif_library: list[MotifPattern],
    target_level: str = "spider_fused",
    max_matches_per_motif: int = 100,
) -> dict[str, DecompositionResult]:
    """
    Decompose all algorithms in a corpus at a given level.

    Returns: {algorithm_name: DecompositionResult}
    """
    results = {}
    for (algo_name, level), host_graph in corpus.items():
        if level != target_level:
            continue
        results[algo_name] = decompose_graph(
            host_graph, motif_library,
            max_matches_per_motif=max_matches_per_motif,
        )
    return results
