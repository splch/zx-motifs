"""Centralized configuration for tuneable pipeline parameters."""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """All tuneable limits for corpus building, motif discovery, and matching."""

    # -- Corpus building --
    max_qubits: int = 10

    # -- Bottom-up motif discovery --
    min_motif_size: int = 3
    max_motif_size: int = 7
    max_subgraphs: int = 2000
    min_algorithms: int = 2

    # -- Neighbourhood extraction --
    neighbourhood_radius: int = 2

    # -- Fingerprinting --
    fingerprint_max_matches: int = 50

    # -- Motif matching --
    match_max_matches: int = 100
    batch_max_matches_per_graph: int = 50
    approximate_max_edit_distance: int = 2
    approximate_max_matches: int = 50

    # -- Inducer --
    inducer_radius: int = 1
    inducer_min_subgraph_size: int = 2
    inducer_max_subgraph_size: int = 7
    inducer_min_occurrences: int = 5
    inducer_min_algorithms: int = 2
    inducer_max_new_motifs: int = 50
    inducer_max_rounds: int = 3
    inducer_max_matches: int = 10

    # -- Cross-level --
    cross_level_max_subgraphs: int = 500

    # -- Optimizer --
    optimizer_max_library_size: int = 30

    # -- Decomposer --
    decomposer_max_matches_per_motif: int = 100


CONFIG = PipelineConfig()
