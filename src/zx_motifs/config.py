"""Centralized configuration for tuneable pipeline parameters.

Each field controls a limit or threshold in the pipeline. Defaults are
chosen to balance thoroughness against runtime on a typical desktop
machine with ~20 algorithms and ~50 motifs.
"""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """All tuneable limits for corpus building, motif discovery, and matching."""

    # -- Corpus building --

    # Upper bound on qubit count when generating algorithm instances for the
    # corpus. Larger values yield richer graphs but increase conversion and
    # matching time roughly quadratically.
    max_qubits: int = 10

    # -- Bottom-up motif discovery --

    # Minimum number of nodes a candidate motif must have. Motifs with fewer
    # nodes (e.g. a single spider) are too generic to be informative.
    min_motif_size: int = 3

    # Maximum number of nodes in a candidate motif. Larger subgraphs are
    # exponentially more expensive to enumerate and rarely recur across
    # different algorithms.
    max_motif_size: int = 7

    # Safety cap on the number of connected subgraphs enumerated per host
    # graph during bottom-up discovery, preventing combinatorial blow-up on
    # dense graphs.
    max_subgraphs: int = 2000

    # A candidate motif must appear in at least this many distinct algorithms
    # to be promoted to the library; filters out algorithm-specific patterns.
    min_algorithms: int = 2

    # -- Neighbourhood extraction --

    # BFS hop radius when extracting local neighbourhoods around each node.
    # Radius 2 captures the two-hop context that typically spans a single
    # gate gadget in ZX form.
    neighbourhood_radius: int = 2

    # -- Fingerprinting --

    # Maximum motif matches counted per (algorithm, simplification_level)
    # pair when building fingerprint vectors. Capping this avoids
    # disproportionate weight from highly repetitive structures.
    fingerprint_max_matches: int = 50

    # -- Motif matching --

    # Hard limit on VF2 subgraph isomorphism matches returned per single
    # match_motif() call, bounding worst-case runtime.
    match_max_matches: int = 100

    # Per-graph cap used when matching a motif against every graph in a
    # corpus batch.
    batch_max_matches_per_graph: int = 50

    # Maximum graph-edit distance for approximate (fuzzy) motif matching.
    # Allows minor structural deviations (e.g. an extra spider or changed
    # edge type).
    approximate_max_edit_distance: int = 2

    # Cap on approximate match results, since fuzzy search can produce many
    # near-miss hits.
    approximate_max_matches: int = 50

    # -- Inducer --

    # BFS radius for the inducer's neighbourhood extraction around each
    # unmatched node.
    inducer_radius: int = 1

    # Minimum / maximum node count for subgraphs proposed by the inducer.
    inducer_min_subgraph_size: int = 2
    inducer_max_subgraph_size: int = 7

    # A candidate induced motif must occur at least this many times across
    # the corpus to be worth adding.
    inducer_min_occurrences: int = 5

    # Must appear in at least this many distinct algorithms.
    inducer_min_algorithms: int = 2

    # Maximum new motifs the inducer may add in a single run.
    inducer_max_new_motifs: int = 50

    # Number of induce-then-match refinement rounds.
    inducer_max_rounds: int = 3

    # Per-motif match cap during inducer evaluation.
    inducer_max_matches: int = 10

    # -- Cross-level --

    # Subgraph enumeration cap when comparing motifs across different
    # simplification levels.
    cross_level_max_subgraphs: int = 500

    # -- Optimizer --

    # Target upper bound on library size after redundancy pruning. Keeps
    # the fingerprint matrix manageable for downstream classifiers.
    optimizer_max_library_size: int = 30

    # -- Decomposer --

    # Per-motif match cap during greedy set-cover decomposition of a graph.
    decomposer_max_matches_per_motif: int = 100


CONFIG = PipelineConfig()
