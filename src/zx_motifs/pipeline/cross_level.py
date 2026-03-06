"""
Cross-level motif tracking: how motifs evolve through simplification levels.
"""
from dataclasses import dataclass, field

import networkx as nx

from .featurizer import compute_motif_feature_vector, motif_similarity
from .matcher import MotifPattern, find_motif_in_graph
from .motif_generators import enumerate_connected_subgraphs


@dataclass
class MotifEvolution:
    """Tracks how a motif appears/disappears across simplification levels."""

    motif_id: str
    level_occurrences: dict[str, int] = field(default_factory=dict)
    survives_to: list[str] = field(default_factory=list)
    transforms_at: list[str] = field(default_factory=list)
    vanishes_at: list[str] = field(default_factory=list)


_DEFAULT_LEVELS = [
    "raw", "spider_fused", "interior_cliff",
    "clifford_simp", "full_reduce", "teleport_reduce",
]


def track_motif_evolution(
    motif: MotifPattern,
    corpus: dict,
    algorithm: str,
    levels: list[str] | None = None,
    transform_threshold: float = 0.7,
) -> MotifEvolution:
    """
    Track how a motif evolves across simplification levels for one algorithm.

    For each level:
      - If exact match found → survives
      - If no exact match but feature-similar subgraph exists → transforms
      - Otherwise → vanishes

    Args:
        motif: The motif to track.
        corpus: {(algo_name, level): nx.Graph}
        algorithm: Which algorithm to track in.
        levels: Levels to check (default: all 6).
        transform_threshold: Feature similarity threshold for "transforms".
    """
    if levels is None:
        levels = _DEFAULT_LEVELS

    evo = MotifEvolution(motif_id=motif.motif_id)
    motif_vec = compute_motif_feature_vector(motif.graph)
    motif_size = motif.graph.number_of_nodes()

    for level in levels:
        key = (algorithm, level)
        if key not in corpus:
            continue

        host = corpus[key]

        # Try exact match
        matches = find_motif_in_graph(motif.graph, host)
        n_matches = len(matches)
        evo.level_occurrences[level] = n_matches

        if n_matches > 0:
            evo.survives_to.append(level)
            continue

        # No exact match — check for structural similarity
        best_sim = 0.0
        subgraphs = enumerate_connected_subgraphs(
            host,
            min_size=max(motif_size - 1, 2),
            max_size=motif_size + 1,
            max_subgraphs=100,
        )
        for sg in subgraphs:
            sg_vec = compute_motif_feature_vector(sg)
            sim = motif_similarity(motif_vec, sg_vec)
            best_sim = max(best_sim, sim)

        if best_sim >= transform_threshold:
            evo.transforms_at.append(level)
        else:
            evo.vanishes_at.append(level)

    return evo


def track_all_motifs_evolution(
    motifs: list[MotifPattern],
    corpus: dict,
    algorithms: list[str] | None = None,
    levels: list[str] | None = None,
) -> dict[str, list[MotifEvolution]]:
    """
    Batch version: track all motifs across all algorithms.

    Returns: {motif_id: [MotifEvolution per algorithm]}
    """
    if algorithms is None:
        algorithms = sorted({algo for (algo, _) in corpus.keys()})

    result: dict[str, list[MotifEvolution]] = {}

    for motif in motifs:
        evolutions = []
        for algo in algorithms:
            evo = track_motif_evolution(motif, corpus, algo, levels=levels)
            evolutions.append(evo)
        result[motif.motif_id] = evolutions

    return result
