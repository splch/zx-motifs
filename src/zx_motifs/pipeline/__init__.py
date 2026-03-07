"""Core pipeline modules for ZX motif discovery."""

from .ansatz import (
    build_hamiltonian,
    cx_chain_entangler,
    hea_entangler,
    irr_pair11_entangler,
    irr_pair11_original_6q,
)
from .evaluation import (
    compute_entangling_power,
    count_2q,
    run_benchmark,
    vqe_test,
)
from .fingerprint import (
    build_corpus,
    build_fingerprint_matrix,
    discover_motifs,
)

# MAGIC framework imports (lazy-friendly: imported here for discoverability)
from .assembler import MAGICConfig, assemble_for_problem, assemble_magic
from .baselines import BASELINE_FACTORIES, build_all_baselines, build_baseline
from .benchmark_suite import BenchmarkProblem, build_suite, get_problem, list_problems
from .classical_hardness import evaluate_classical_hardness
from .evolution import EvolutionConfig, Recipe, run_ablation, run_evolution
from .interaction import build_interaction_graph
from .noise_analysis import evaluate_t0_ideal, gate_budget_analysis
from .phase_poly import PhasePolynomial, extract_from_circuit
from .problem_aware import decompose_hamiltonian
from .scorecard import Scorecard, evaluate_scorecard
from .statistics import compare_results, fit_scaling
from .trainability import compute_dla_dimension

__all__ = [
    # fingerprint
    "build_corpus",
    "discover_motifs",
    "build_fingerprint_matrix",
    # ansatz
    "irr_pair11_entangler",
    "irr_pair11_original_6q",
    "cx_chain_entangler",
    "hea_entangler",
    "build_hamiltonian",
    # evaluation
    "vqe_test",
    "run_benchmark",
    "count_2q",
    "compute_entangling_power",
    # MAGIC framework
    "MAGICConfig",
    "assemble_magic",
    "assemble_for_problem",
    "PhasePolynomial",
    "extract_from_circuit",
    "decompose_hamiltonian",
    "build_interaction_graph",
    "compute_dla_dimension",
    "BenchmarkProblem",
    "build_suite",
    "get_problem",
    "list_problems",
    "Scorecard",
    "evaluate_scorecard",
    "BASELINE_FACTORIES",
    "build_baseline",
    "build_all_baselines",
    "compare_results",
    "fit_scaling",
    "evaluate_classical_hardness",
    "evaluate_t0_ideal",
    "gate_budget_analysis",
    "Recipe",
    "EvolutionConfig",
    "run_evolution",
    "run_ablation",
]
