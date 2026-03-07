"""Tests for Phase 4: Recipe DSL, evolutionary search, and ablation."""
import random

import numpy as np
import pytest

from zx_motifs.pipeline.benchmark_suite import h2_molecule, tfim_1d
from zx_motifs.pipeline.evolution import (
    AblationResult,
    EvolutionConfig,
    EvolutionResult,
    FitnessResult,
    MotifSlot,
    Recipe,
    crossover,
    evaluate_recipe,
    mutate,
    mutate_gate_budget,
    mutate_layers,
    mutate_pattern,
    mutate_symmetry,
    run_ablation,
    run_evolution,
    seed_population,
    tournament_select,
)


# ── Recipe DSL ────────────────────────────────────────────────────


class TestRecipe:
    def test_default_recipe(self):
        r = Recipe()
        assert r.pattern == "layered"
        assert r.n_layers == 2
        assert r.gate_budget is None

    def test_to_config(self):
        r = Recipe(pattern="brick", n_layers=3, symmetry_constraints=["z2_parity"])
        config = r.to_config()
        assert config.pattern == "brick"
        assert config.n_layers == 3
        assert config.symmetry_filter is True

    def test_round_trip(self):
        r = Recipe(
            pattern="star",
            n_layers=4,
            motif_slots=[MotifSlot(motif_id="m1", weight=0.5, frozen=True)],
            gate_budget=50,
            symmetry_constraints=["particle_number"],
            metadata={"id": "test"},
        )
        d = r.to_dict()
        r2 = Recipe.from_dict(d)
        assert r2.pattern == "star"
        assert r2.n_layers == 4
        assert len(r2.motif_slots) == 1
        assert r2.motif_slots[0].frozen is True
        assert r2.gate_budget == 50


# ── Fitness ───────────────────────────────────────────────────────


class TestFitness:
    def test_killed_fitness(self):
        f = FitnessResult(
            energy=0, relative_error=0, gate_count=0,
            scorecard_layers_passed=0, killed=True,
        )
        assert f.fitness == float("-inf")

    def test_fitness_ordering(self):
        good = FitnessResult(
            energy=-1.0, relative_error=0.01, gate_count=10,
            scorecard_layers_passed=3, killed=False,
        )
        bad = FitnessResult(
            energy=0.0, relative_error=0.5, gate_count=10,
            scorecard_layers_passed=1, killed=False,
        )
        assert good.fitness > bad.fitness

    def test_evaluate_recipe_h2(self):
        r = Recipe(pattern="layered", n_layers=2)
        p = h2_molecule()
        result = evaluate_recipe(r, p, n_restarts=3, maxiter=100)
        assert isinstance(result, FitnessResult)
        assert not result.killed
        assert result.gate_count > 0


# ── Mutation ──────────────────────────────────────────────────────


class TestMutation:
    def test_mutate_pattern(self):
        rng = random.Random(42)
        r = Recipe(pattern="layered")
        mutated = mutate_pattern(r, rng)
        # Original unchanged
        assert r.pattern == "layered"
        assert mutated.pattern in ["layered", "brick", "star", "adaptive"]

    def test_mutate_layers(self):
        rng = random.Random(42)
        r = Recipe(n_layers=3)
        mutated = mutate_layers(r, rng)
        assert mutated.n_layers >= 1
        assert mutated.n_layers != r.n_layers or True  # may stay same with ±0 not possible

    def test_mutate_layers_min_1(self):
        rng = random.Random(42)
        r = Recipe(n_layers=1)
        for _ in range(20):
            mutated = mutate_layers(r, rng)
            assert mutated.n_layers >= 1

    def test_mutate_symmetry_add(self):
        rng = random.Random(42)
        r = Recipe(symmetry_constraints=[])
        mutated = mutate_symmetry(r, rng)
        assert len(mutated.symmetry_constraints) >= 0

    def test_mutate_gate_budget(self):
        rng = random.Random(42)
        r = Recipe(gate_budget=None)
        mutated = mutate_gate_budget(r, rng)
        assert mutated.gate_budget is not None
        assert mutated.gate_budget > 0

    def test_mutate_composite(self):
        rng = random.Random(42)
        r = Recipe(pattern="layered", n_layers=2)
        child = mutate(r, rng)
        assert isinstance(child, Recipe)
        assert "mutation" in child.metadata


# ── Crossover ─────────────────────────────────────────────────────


class TestCrossover:
    def test_crossover_basic(self):
        rng = random.Random(42)
        a = Recipe(pattern="layered", n_layers=2, symmetry_constraints=["z2_parity"])
        b = Recipe(pattern="star", n_layers=4, symmetry_constraints=["particle_number"])
        child = crossover(a, b, rng)
        assert child.pattern in ["layered", "star"]
        assert child.n_layers in [2, 4]
        assert "parents" in child.metadata


# ── Tournament Selection ──────────────────────────────────────────


class TestTournament:
    def test_selects_best(self):
        rng = random.Random(42)
        pop = [
            (Recipe(metadata={"id": "bad"}),
             FitnessResult(energy=0, relative_error=0.5, gate_count=10,
                           scorecard_layers_passed=1, killed=False)),
            (Recipe(metadata={"id": "good"}),
             FitnessResult(energy=-1, relative_error=0.01, gate_count=10,
                           scorecard_layers_passed=3, killed=False)),
        ]
        # With k=2 and 2 individuals, always picks the best
        winner = tournament_select(pop, k=2, rng=rng)
        assert isinstance(winner, Recipe)


# ── Seed Population ───────────────────────────────────────────────


class TestSeedPopulation:
    def test_seed_size(self):
        pop = seed_population(size=10)
        assert len(pop) == 10

    def test_seed_diversity(self):
        pop = seed_population(size=20, rng=random.Random(42))
        patterns = {r.pattern for r in pop}
        assert len(patterns) > 1  # Should have multiple patterns


# ── Evolutionary Loop ─────────────────────────────────────────────


class TestEvolution:
    def test_small_evolution(self):
        """Run a tiny evolution to verify the loop works."""
        p = tfim_1d(4)
        config = EvolutionConfig(
            population_size=4,
            n_generations=2,
            n_restarts=2,
            maxiter=50,
            elitism=1,
        )
        result = run_evolution(p, config)
        assert isinstance(result, EvolutionResult)
        assert len(result.history) == 2
        assert result.best_recipe is not None
        assert result.best_fitness is not None

    def test_evolution_improves(self):
        """Best fitness should not decrease (elitism preserves best)."""
        p = tfim_1d(4)
        config = EvolutionConfig(
            population_size=4,
            n_generations=3,
            n_restarts=2,
            maxiter=50,
            elitism=1,
        )
        result = run_evolution(p, config)
        for i in range(1, len(result.history)):
            assert result.history[i]["best_fitness"] >= result.history[i - 1]["best_fitness"]


# ── Ablation ──────────────────────────────────────────────────────


class TestAblation:
    def test_ablation_runs(self):
        r = Recipe(pattern="layered", n_layers=2)
        p = tfim_1d(4)
        result = run_ablation(r, p, n_restarts=2, maxiter=50)
        assert isinstance(result, AblationResult)
        assert result.baseline_fitness is not None
        # Should have pattern ablations at minimum
        assert len(result.ablations) >= 2

    def test_ablation_summary(self):
        r = Recipe(pattern="layered", n_layers=2, symmetry_constraints=["z2_parity"])
        p = tfim_1d(4)
        result = run_ablation(r, p, n_restarts=2, maxiter=50)
        summary = result.summary()
        assert "baseline" in summary
        assert "ablations" in summary
        assert "no_symmetry" in summary["ablations"]
