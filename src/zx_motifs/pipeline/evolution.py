"""
Recipe DSL and evolutionary search for MAGIC ansatz optimization.

A Recipe encodes a compact genotype for constructing quantum ansatze.
The evolutionary loop applies mutation/crossover operators, evaluates
fitness via the scorecard, and grows the library with emergent motifs.
"""
from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit

from .assembler import MAGICConfig, PatternType, assemble_magic
from .benchmark_suite import BenchmarkProblem, PauliTerm
from .evaluation import count_2q, vqe_test
from .scorecard import Scorecard, evaluate_scorecard

logger = logging.getLogger(__name__)


# ── Recipe DSL ────────────────────────────────────────────────────


@dataclass
class MotifSlot:
    """A slot in a recipe that references a motif or pattern."""

    motif_id: str | None = None  # None means "any from pool"
    weight: float = 1.0  # Relative importance in composition
    frozen: bool = False  # If True, mutation skips this slot


@dataclass
class Recipe:
    """Compact genotype for a MAGIC ansatz.

    A recipe specifies:
      - pattern: Circuit topology (layered/brick/star/adaptive)
      - n_layers: Repetition depth
      - motif_slots: Optional motif references
      - gate_budget: Max 2-qubit gates
      - symmetry_constraints: Required symmetries
      - metadata: Provenance and fitness history
    """

    pattern: PatternType = "layered"
    n_layers: int = 2
    motif_slots: list[MotifSlot] = field(default_factory=list)
    gate_budget: int | None = None
    symmetry_constraints: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_config(self) -> MAGICConfig:
        """Convert recipe to a MAGICConfig for assembly."""
        return MAGICConfig(
            pattern=self.pattern,
            n_layers=self.n_layers,
            symmetry_filter=bool(self.symmetry_constraints),
            required_symmetries=self.symmetry_constraints,
            max_2q_gates=self.gate_budget,
        )

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "n_layers": self.n_layers,
            "motif_slots": [
                {"motif_id": s.motif_id, "weight": s.weight, "frozen": s.frozen}
                for s in self.motif_slots
            ],
            "gate_budget": self.gate_budget,
            "symmetry_constraints": self.symmetry_constraints,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Recipe:
        slots = [
            MotifSlot(motif_id=s.get("motif_id"), weight=s.get("weight", 1.0),
                      frozen=s.get("frozen", False))
            for s in d.get("motif_slots", [])
        ]
        return cls(
            pattern=d.get("pattern", "layered"),
            n_layers=d.get("n_layers", 2),
            motif_slots=slots,
            gate_budget=d.get("gate_budget"),
            symmetry_constraints=d.get("symmetry_constraints", []),
            metadata=d.get("metadata", {}),
        )


# ── Fitness Evaluation ────────────────────────────────────────────


@dataclass
class FitnessResult:
    """Multi-objective fitness for evolutionary selection."""

    energy: float  # Best VQE energy (lower = better)
    relative_error: float  # |E - E_exact| / |E_exact|
    gate_count: int  # 2-qubit gate count
    scorecard_layers_passed: int  # Layers passed in scorecard
    killed: bool  # Whether scorecard killed the candidate
    raw_scores: dict = field(default_factory=dict)

    @property
    def fitness(self) -> float:
        """Combined fitness score (higher = better).

        Balances energy accuracy with circuit efficiency.
        Killed candidates get -inf.
        """
        if self.killed:
            return float("-inf")
        # Negative relative error (want to minimize) + gate efficiency bonus
        gate_penalty = 0.001 * self.gate_count
        layer_bonus = 0.1 * self.scorecard_layers_passed
        return -(self.relative_error + gate_penalty) + layer_bonus


def evaluate_recipe(
    recipe: Recipe,
    problem: BenchmarkProblem,
    scorecard_layers: list[int] | None = None,
    n_restarts: int = 5,
    maxiter: int = 200,
) -> FitnessResult:
    """Evaluate a recipe on a benchmark problem.

    Assembles the circuit, runs VQE, and computes fitness metrics.
    """
    if scorecard_layers is None:
        scorecard_layers = [1]

    try:
        config = recipe.to_config()
        qc = assemble_magic(problem=problem, config=config)
    except Exception as e:
        logger.debug("Recipe assembly failed: %s", e)
        return FitnessResult(
            energy=float("inf"), relative_error=float("inf"),
            gate_count=0, scorecard_layers_passed=0, killed=True,
        )

    n_qubits = problem.n_qubits
    n_2q = count_2q(qc)

    # Gate budget check
    if recipe.gate_budget is not None and n_2q > recipe.gate_budget:
        return FitnessResult(
            energy=float("inf"), relative_error=float("inf"),
            gate_count=n_2q, scorecard_layers_passed=0, killed=True,
            raw_scores={"reason": "gate_budget_exceeded"},
        )

    # Scorecard evaluation (fast layers only)
    card = evaluate_scorecard(
        qc, f"recipe_{id(recipe)}", n_qubits,
        layers=scorecard_layers, fast_fail=True,
    )

    if card.n_killed > 0:
        return FitnessResult(
            energy=float("inf"), relative_error=float("inf"),
            gate_count=n_2q, scorecard_layers_passed=card.n_passed,
            killed=True,
        )

    # VQE evaluation
    H = problem.hamiltonian_matrix()
    result = vqe_test(qc, n_qubits, H, n_restarts=n_restarts, maxiter=maxiter)
    exact = problem.exact_energy
    rel_error = abs(result["best_energy"] - exact) / abs(exact) if exact != 0 else float("inf")

    return FitnessResult(
        energy=result["best_energy"],
        relative_error=rel_error,
        gate_count=n_2q,
        scorecard_layers_passed=card.n_passed,
        killed=False,
        raw_scores={
            "mean_energy": result["mean_energy"],
            "std_energy": result["std_energy"],
        },
    )


# ── Mutation Operators ────────────────────────────────────────────

PATTERNS: list[PatternType] = ["layered", "brick", "star", "adaptive"]


def mutate_pattern(recipe: Recipe, rng: random.Random) -> Recipe:
    """Mutate the circuit pattern."""
    r = copy.deepcopy(recipe)
    r.pattern = rng.choice(PATTERNS)
    return r


def mutate_layers(recipe: Recipe, rng: random.Random) -> Recipe:
    """Mutate the number of layers (±1 or ±2)."""
    r = copy.deepcopy(recipe)
    delta = rng.choice([-2, -1, 1, 2])
    r.n_layers = max(1, r.n_layers + delta)
    return r


def mutate_symmetry(
    recipe: Recipe,
    rng: random.Random,
    all_symmetries: list[str] | None = None,
) -> Recipe:
    """Add or remove a symmetry constraint."""
    if all_symmetries is None:
        all_symmetries = ["particle_number", "z2_parity", "real_valued"]

    r = copy.deepcopy(recipe)
    if r.symmetry_constraints and rng.random() < 0.5:
        # Remove a random constraint
        r.symmetry_constraints.pop(rng.randrange(len(r.symmetry_constraints)))
    else:
        # Add a random constraint not already present
        available = [s for s in all_symmetries if s not in r.symmetry_constraints]
        if available:
            r.symmetry_constraints.append(rng.choice(available))
    return r


def mutate_gate_budget(recipe: Recipe, rng: random.Random) -> Recipe:
    """Adjust the gate budget."""
    r = copy.deepcopy(recipe)
    if r.gate_budget is None:
        r.gate_budget = rng.choice([10, 20, 50, 100])
    else:
        factor = rng.choice([0.5, 0.75, 1.25, 1.5, 2.0])
        r.gate_budget = max(4, int(r.gate_budget * factor))
    return r


MUTATORS = [mutate_pattern, mutate_layers, mutate_symmetry, mutate_gate_budget]


def mutate(recipe: Recipe, rng: random.Random) -> Recipe:
    """Apply a random mutation to a recipe."""
    mutator = rng.choice(MUTATORS)
    child = mutator(recipe, rng)
    child.metadata["parent"] = recipe.metadata.get("id", "unknown")
    child.metadata["mutation"] = mutator.__name__
    return child


def crossover(a: Recipe, b: Recipe, rng: random.Random) -> Recipe:
    """Crossover two recipes, taking traits from each parent."""
    child = Recipe(
        pattern=rng.choice([a.pattern, b.pattern]),
        n_layers=rng.choice([a.n_layers, b.n_layers]),
        gate_budget=rng.choice([a.gate_budget, b.gate_budget]),
        symmetry_constraints=list(
            set(a.symmetry_constraints) | set(b.symmetry_constraints)
        ) if rng.random() < 0.5 else list(
            set(a.symmetry_constraints) & set(b.symmetry_constraints)
        ),
        metadata={
            "parents": [
                a.metadata.get("id", "unknown"),
                b.metadata.get("id", "unknown"),
            ],
            "mutation": "crossover",
        },
    )
    return child


# ── Tournament Selection ──────────────────────────────────────────


def tournament_select(
    population: list[tuple[Recipe, FitnessResult]],
    k: int = 3,
    rng: random.Random | None = None,
) -> Recipe:
    """Select a recipe via tournament selection.

    Picks k random individuals and returns the one with highest fitness.
    """
    if rng is None:
        rng = random.Random()
    contestants = rng.sample(population, min(k, len(population)))
    best = max(contestants, key=lambda x: x[1].fitness)
    return copy.deepcopy(best[0])


# ── Seed Population ───────────────────────────────────────────────


def seed_population(
    size: int = 20,
    rng: random.Random | None = None,
) -> list[Recipe]:
    """Generate an initial population of diverse recipes."""
    if rng is None:
        rng = random.Random(42)

    population = []
    for i in range(size):
        recipe = Recipe(
            pattern=rng.choice(PATTERNS),
            n_layers=rng.randint(1, 4),
            gate_budget=rng.choice([None, 10, 20, 50]),
            symmetry_constraints=rng.sample(
                ["particle_number", "z2_parity", "real_valued"],
                k=rng.randint(0, 2),
            ),
            metadata={"id": f"seed_{i}", "generation": 0},
        )
        population.append(recipe)
    return population


# ── Evolutionary Loop ─────────────────────────────────────────────


@dataclass
class EvolutionConfig:
    """Configuration for the evolutionary search."""

    population_size: int = 20
    n_generations: int = 10
    mutation_rate: float = 0.7
    crossover_rate: float = 0.3
    tournament_k: int = 3
    elitism: int = 2  # Top-k survive unchanged
    scorecard_layers: list[int] = field(default_factory=lambda: [1])
    n_restarts: int = 5
    maxiter: int = 200
    seed: int = 42


@dataclass
class EvolutionResult:
    """Result of an evolutionary search run."""

    best_recipe: Recipe
    best_fitness: FitnessResult
    history: list[dict]  # Per-generation stats
    final_population: list[tuple[Recipe, FitnessResult]]

    def summary(self) -> dict:
        return {
            "best_fitness": self.best_fitness.fitness,
            "best_energy": self.best_fitness.energy,
            "best_relative_error": self.best_fitness.relative_error,
            "best_gate_count": self.best_fitness.gate_count,
            "n_generations": len(self.history),
            "best_recipe": self.best_recipe.to_dict(),
        }


def run_evolution(
    problem: BenchmarkProblem,
    config: EvolutionConfig | None = None,
) -> EvolutionResult:
    """Run evolutionary search for optimal MAGIC recipes.

    Args:
        problem: The benchmark problem to optimize for.
        config: Evolution configuration. Defaults to reasonable settings.

    Returns:
        EvolutionResult with the best recipe, fitness, and history.
    """
    if config is None:
        config = EvolutionConfig()

    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)

    # Initialize population
    population = seed_population(config.population_size, rng)
    history: list[dict] = []

    # Evaluate initial population
    evaluated: list[tuple[Recipe, FitnessResult]] = []
    for recipe in population:
        fitness = evaluate_recipe(
            recipe, problem,
            scorecard_layers=config.scorecard_layers,
            n_restarts=config.n_restarts,
            maxiter=config.maxiter,
        )
        evaluated.append((recipe, fitness))

    for gen in range(config.n_generations):
        # Sort by fitness (descending)
        evaluated.sort(key=lambda x: x[1].fitness, reverse=True)

        # Record stats
        fitnesses = [f.fitness for _, f in evaluated if f.fitness > float("-inf")]
        gen_stats = {
            "generation": gen,
            "best_fitness": evaluated[0][1].fitness,
            "best_energy": evaluated[0][1].energy,
            "best_rel_error": evaluated[0][1].relative_error,
            "mean_fitness": float(np.mean(fitnesses)) if fitnesses else float("-inf"),
            "population_size": len(evaluated),
            "alive": sum(1 for _, f in evaluated if not f.killed),
        }
        history.append(gen_stats)
        logger.info(
            "Gen %d: best_fitness=%.4f, best_energy=%.4f, alive=%d/%d",
            gen, gen_stats["best_fitness"], gen_stats["best_energy"],
            gen_stats["alive"], len(evaluated),
        )

        # Create next generation
        next_gen: list[tuple[Recipe, FitnessResult]] = []

        # Elitism: keep top-k unchanged
        for i in range(min(config.elitism, len(evaluated))):
            next_gen.append(evaluated[i])

        # Fill rest with offspring
        while len(next_gen) < config.population_size:
            r = rng.random()
            if r < config.crossover_rate and len(evaluated) >= 2:
                # Crossover
                parent_a = tournament_select(evaluated, config.tournament_k, rng)
                parent_b = tournament_select(evaluated, config.tournament_k, rng)
                child = crossover(parent_a, parent_b, rng)
            else:
                # Mutation
                parent = tournament_select(evaluated, config.tournament_k, rng)
                child = mutate(parent, rng)

            child.metadata["id"] = f"gen{gen + 1}_{len(next_gen)}"
            child.metadata["generation"] = gen + 1

            # Evaluate child
            fitness = evaluate_recipe(
                child, problem,
                scorecard_layers=config.scorecard_layers,
                n_restarts=config.n_restarts,
                maxiter=config.maxiter,
            )
            next_gen.append((child, fitness))

        evaluated = next_gen

    # Final sort
    evaluated.sort(key=lambda x: x[1].fitness, reverse=True)
    best_recipe, best_fitness = evaluated[0]

    return EvolutionResult(
        best_recipe=best_recipe,
        best_fitness=best_fitness,
        history=history,
        final_population=evaluated,
    )


# ── Ablation Runner ──────────────────────────────────────────────


@dataclass
class AblationResult:
    """Result of an ablation study."""

    baseline_fitness: FitnessResult
    ablations: dict[str, FitnessResult]

    def summary(self) -> dict:
        result = {
            "baseline": {
                "fitness": self.baseline_fitness.fitness,
                "relative_error": self.baseline_fitness.relative_error,
                "gate_count": self.baseline_fitness.gate_count,
            },
            "ablations": {},
        }
        for name, f in self.ablations.items():
            result["ablations"][name] = {
                "fitness": f.fitness,
                "relative_error": f.relative_error,
                "delta_fitness": f.fitness - self.baseline_fitness.fitness,
            }
        return result


def run_ablation(
    recipe: Recipe,
    problem: BenchmarkProblem,
    n_restarts: int = 5,
    maxiter: int = 200,
) -> AblationResult:
    """Run ablation study on a recipe.

    Tests the recipe with each feature removed or modified
    to quantify the contribution of each component.
    """
    # Baseline
    baseline = evaluate_recipe(recipe, problem, n_restarts=n_restarts, maxiter=maxiter)

    ablations: dict[str, FitnessResult] = {}

    # Ablation 1: Random pattern
    for pattern in PATTERNS:
        if pattern != recipe.pattern:
            ablated = copy.deepcopy(recipe)
            ablated.pattern = pattern
            ablations[f"pattern_{pattern}"] = evaluate_recipe(
                ablated, problem, n_restarts=n_restarts, maxiter=maxiter,
            )

    # Ablation 2: Layer count variations
    for delta in [-1, 1, 2]:
        n = max(1, recipe.n_layers + delta)
        if n != recipe.n_layers:
            ablated = copy.deepcopy(recipe)
            ablated.n_layers = n
            ablations[f"layers_{n}"] = evaluate_recipe(
                ablated, problem, n_restarts=n_restarts, maxiter=maxiter,
            )

    # Ablation 3: No symmetry constraints
    if recipe.symmetry_constraints:
        ablated = copy.deepcopy(recipe)
        ablated.symmetry_constraints = []
        ablations["no_symmetry"] = evaluate_recipe(
            ablated, problem, n_restarts=n_restarts, maxiter=maxiter,
        )

    # Ablation 4: No gate budget
    if recipe.gate_budget is not None:
        ablated = copy.deepcopy(recipe)
        ablated.gate_budget = None
        ablations["no_gate_budget"] = evaluate_recipe(
            ablated, problem, n_restarts=n_restarts, maxiter=maxiter,
        )

    return AblationResult(baseline_fitness=baseline, ablations=ablations)
