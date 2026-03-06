"""
Motif library optimization: rank motifs by marginal coverage contribution,
prune redundant ones via weighted greedy set cover.
"""
from dataclasses import dataclass, field

import networkx as nx

from .decomposer import decompose_graph
from .matcher import MotifPattern, find_motif_in_graph


@dataclass
class MotifScore:
    """Scoring data for a single motif candidate."""

    motif_id: str
    total_matches: int = 0
    algorithms_covered: int = 0
    unique_vertices_covered: int = 0
    marginal_coverage: float = 0.0
    weight: float = 0.0


@dataclass
class OptimizedLibrary:
    """Result of library optimization."""

    selected_motifs: list[MotifPattern]
    scores: list[MotifScore]
    coverage_by_motif: dict[str, int]  # motif_id → vertices covered
    final_coverage: float
    dropped_motifs: list[str]

    def summary(self) -> str:
        lines = [
            f"Optimized library: {len(self.selected_motifs)} motifs, "
            f"{self.final_coverage:.1%} coverage",
        ]
        for score in sorted(self.scores, key=lambda s: -s.weight):
            lines.append(
                f"  {score.motif_id}: {score.total_matches} matches, "
                f"{score.algorithms_covered} algos, "
                f"marginal={score.marginal_coverage:.3f}"
            )
        if self.dropped_motifs:
            lines.append(f"  Dropped: {', '.join(self.dropped_motifs)}")
        return "\n".join(lines)


def compute_motif_coverage_map(
    motif: MotifPattern,
    corpus: dict,
    target_level: str = "spider_fused",
) -> dict[str, set[int]]:
    """
    Compute which host vertices a motif covers per algorithm.

    Returns: {algorithm_name: set of covered vertex IDs}
    """
    coverage: dict[str, set[int]] = {}
    for (algo_name, level), host_graph in corpus.items():
        if level != target_level:
            continue
        matches = find_motif_in_graph(
            motif.graph, host_graph, max_matches=100,
        )
        if matches:
            verts = set()
            for mapping in matches:
                verts.update(mapping.values())
            coverage[algo_name] = verts
    return coverage


def optimize_library(
    candidates: list[MotifPattern],
    corpus: dict,
    target_level: str = "spider_fused",
    max_library_size: int = 30,
    min_marginal_coverage: float = 0.005,
    algorithm_diversity_weight: float = 0.3,
    size_preference_weight: float = 0.1,
) -> OptimizedLibrary:
    """
    Select the best motifs via weighted greedy set cover.

    Score = marginal_coverage + diversity_weight * (new_algos/total_algos)
            + size_weight * (motif_size/max_size)

    Iteratively selects the highest-scoring motif, updates covered sets,
    and stops when marginal contribution falls below threshold.
    """
    if not candidates or not corpus:
        return OptimizedLibrary(
            selected_motifs=[],
            scores=[],
            coverage_by_motif={},
            final_coverage=0.0,
            dropped_motifs=[m.motif_id for m in candidates],
        )

    # Compute total interior vertices per algorithm
    algo_interiors: dict[str, set[int]] = {}
    total_interior = 0
    all_algos = set()
    for (algo_name, level), host_graph in corpus.items():
        if level != target_level:
            continue
        all_algos.add(algo_name)
        interior = {
            n for n, d in host_graph.nodes(data=True)
            if d.get("vertex_type") != "BOUNDARY"
        }
        algo_interiors[algo_name] = interior
        total_interior += len(interior)

    if total_interior == 0:
        return OptimizedLibrary(
            selected_motifs=[],
            scores=[],
            coverage_by_motif={},
            final_coverage=0.0,
            dropped_motifs=[m.motif_id for m in candidates],
        )

    total_algos = len(all_algos)

    # Pre-compute coverage maps for all candidates
    coverage_maps: dict[str, dict[str, set[int]]] = {}
    for motif in candidates:
        coverage_maps[motif.motif_id] = compute_motif_coverage_map(
            motif, corpus, target_level,
        )

    # Find max motif size for normalization
    max_motif_size = max(
        (m.graph.number_of_nodes() for m in candidates), default=1
    )

    # Greedy selection
    selected: list[MotifPattern] = []
    scores: list[MotifScore] = []
    coverage_by_motif: dict[str, int] = {}
    globally_covered: dict[str, set[int]] = {a: set() for a in all_algos}
    algos_with_coverage: set[str] = set()
    remaining = list(candidates)

    while remaining and len(selected) < max_library_size:
        best_motif = None
        best_score_val = -1.0
        best_score_obj = None

        for motif in remaining:
            cmap = coverage_maps[motif.motif_id]

            # Marginal coverage: new vertices covered
            new_verts = 0
            new_algos = 0
            for algo, verts in cmap.items():
                new_verts += len(verts - globally_covered.get(algo, set()))
                if algo not in algos_with_coverage and verts:
                    new_algos += 1

            marginal = new_verts / total_interior

            if marginal < min_marginal_coverage:
                continue

            diversity = (new_algos / max(total_algos, 1)) if total_algos > 0 else 0
            size_pref = motif.graph.number_of_nodes() / max_motif_size

            weight = (
                marginal
                + algorithm_diversity_weight * diversity
                + size_preference_weight * size_pref
            )

            if weight > best_score_val:
                best_score_val = weight
                best_motif = motif
                best_score_obj = MotifScore(
                    motif_id=motif.motif_id,
                    total_matches=sum(len(v) for v in cmap.values()),
                    algorithms_covered=len(cmap),
                    unique_vertices_covered=new_verts,
                    marginal_coverage=marginal,
                    weight=weight,
                )

        if best_motif is None:
            break

        selected.append(best_motif)
        scores.append(best_score_obj)
        coverage_by_motif[best_motif.motif_id] = best_score_obj.unique_vertices_covered

        # Update global coverage
        cmap = coverage_maps[best_motif.motif_id]
        for algo, verts in cmap.items():
            globally_covered.setdefault(algo, set()).update(verts)
            if verts:
                algos_with_coverage.add(algo)

        remaining.remove(best_motif)

    # Compute final coverage
    total_covered = sum(len(v) for v in globally_covered.values())
    final_coverage = total_covered / max(total_interior, 1)

    dropped = [m.motif_id for m in remaining]

    return OptimizedLibrary(
        selected_motifs=selected,
        scores=scores,
        coverage_by_motif=coverage_by_motif,
        final_coverage=final_coverage,
        dropped_motifs=dropped,
    )
