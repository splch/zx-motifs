#!/usr/bin/env python3
"""
ZX-MOTIFS MAGIC: End-to-end demonstration.

Runs every major pipeline component from algebraic foundations through
evolutionary search, producing console output at each stage.

Usage:
    python scripts/run_end_to_end.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ── Helpers ───────────────────────────────────────────────────────

_t0 = time.perf_counter()


def banner(title: str) -> None:
    elapsed = time.perf_counter() - _t0
    print(f"\n{'═' * 68}")
    print(f"  {title}  [{elapsed:.1f}s]")
    print(f"{'═' * 68}")


def section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 54 - len(title))}")


# ══════════════════════════════════════════════════════════════════
#  PHASE 1: Algebraic & Structural Foundations
# ══════════════════════════════════════════════════════════════════

banner("PHASE 1: Algebraic & Structural Foundations")

# ── 1a. Benchmark problems ────────────────────────────────────────
section("1a. Benchmark Problem Suite")

from zx_motifs.pipeline.benchmark_suite import (
    BenchmarkProblem,
    build_suite,
    h2_molecule,
    heisenberg_xxx_1d,
    list_problems,
    tfim_1d,
)

print(f"Available problems: {list_problems()}")

problems = {
    "TFIM-4":       tfim_1d(4),
    "Heisenberg-4": heisenberg_xxx_1d(4),
    "H2":           h2_molecule(),
}

for name, p in problems.items():
    print(f"  {name:16s}  n={p.n_qubits}  terms={len(p.pauli_terms):3d}  "
          f"E_exact={p.exact_energy:+.6f}  domain={p.domain}")

# ── 1b. Phase polynomial extraction ──────────────────────────────
section("1b. Phase Polynomial Extraction")

from qiskit import QuantumCircuit
from zx_motifs.pipeline.phase_poly import (
    PhasePolynomial,
    compute_symmetry_tags,
    extract_from_circuit,
)

# Build a small test circuit
qc = QuantumCircuit(4)
qc.h(range(4))
qc.cx(0, 1); qc.rz(0.3, 1); qc.cx(0, 1)
qc.cx(1, 2); qc.rz(0.5, 2); qc.cx(1, 2)
qc.cx(2, 3); qc.rz(0.7, 3); qc.cx(2, 3)

poly = extract_from_circuit(qc)
print(f"  Circuit: {qc.num_qubits} qubits, {qc.depth()} depth, {len(qc.data)} gates")
print(f"  Phase polynomial: {len(poly.gadgets)} gadgets, T-count={poly.t_count}")
for ps, phase in poly.pauli_strings():
    print(f"    {ps}  phase={float(phase):.4f}π")

tags = compute_symmetry_tags(poly)
print(f"  Symmetry tags: {tags}")

# ── 1c. Interaction graph ─────────────────────────────────────────
section("1c. Interaction Graph")

from zx_motifs.pipeline.interaction import (
    build_interaction_graph,
    interaction_degree_sequence,
    interaction_density,
    max_interaction_weight,
)

ig = build_interaction_graph(poly)
print(f"  Nodes: {ig.number_of_nodes()}, Edges: {ig.number_of_edges()}")
print(f"  Density: {interaction_density(ig):.3f}")
print(f"  Max weight: {max_interaction_weight(ig)}")
print(f"  Degree sequence: {interaction_degree_sequence(ig)}")

# ── 1d. Motif registry ────────────────────────────────────────────
section("1d. Motif Registry")

from zx_motifs.motifs.registry import MOTIF_REGISTRY

print(f"  Library size: {len(MOTIF_REGISTRY)} motifs")
for m in MOTIF_REGISTRY[:5]:
    print(f"    {m.motif_id:30s}  nodes={m.graph.number_of_nodes()}")
print(f"    ... and {len(MOTIF_REGISTRY) - 5} more")

# ── 1e. Fingerprint matrix ────────────────────────────────────────
section("1e. Fingerprint Matrix (sample)")

from zx_motifs.pipeline.fingerprint import build_corpus, build_fingerprint_matrix

# Build corpus for a subset of algorithms at one simplification level
corpus = build_corpus(max_qubits=4, max_workers=1)
print(f"  Corpus: {len(corpus)} (algorithm, level) pairs")

counts_df, freq_df = build_fingerprint_matrix(corpus, MOTIF_REGISTRY, max_workers=1)
print(f"  Fingerprint matrix: {counts_df.shape[0]} algorithms × {counts_df.shape[1]} motifs")
total_hits = int(counts_df.values.sum())
nonzero_pct = (counts_df.values > 0).sum() / max(counts_df.size, 1) * 100
print(f"  Total motif hits: {total_hits}, non-zero entries: {nonzero_pct:.1f}%")


# ══════════════════════════════════════════════════════════════════
#  PHASE 2: MAGIC Composition Engine
# ══════════════════════════════════════════════════════════════════

banner("PHASE 2: MAGIC Composition Engine")

# ── 2a. Hamiltonian decomposition ─────────────────────────────────
section("2a. Hamiltonian Decomposition")

from zx_motifs.pipeline.problem_aware import decompose_hamiltonian

h2 = h2_molecule()
decomp = decompose_hamiltonian(h2.pauli_terms)
print(f"  Problem: {h2.name}, {h2.n_qubits} qubits, {len(h2.pauli_terms)} Pauli terms")
print(f"  Commuting groups: {decomp.n_groups}")
print(f"  Identity offset: {decomp.identity_offset:+.6f}")
for g in decomp.groups:
    terms_str = ", ".join(t.pauli_string for t in g.terms[:3])
    extra = f" + {len(g.terms) - 3} more" if len(g.terms) > 3 else ""
    print(f"    Group {g.group_id}: [{terms_str}{extra}]")

# ── 2b. MAGIC assembly (all 4 patterns) ──────────────────────────
section("2b. MAGIC Ansatz Assembly")

from zx_motifs.pipeline.assembler import MAGICConfig, assemble_magic
from zx_motifs.pipeline.evaluation import count_2q

pattern_circuits = {}
for pattern in ["layered", "brick", "star", "adaptive"]:
    config = MAGICConfig(pattern=pattern, n_layers=2)
    qc = assemble_magic(problem=h2, config=config)
    n2q = count_2q(qc)
    pattern_circuits[pattern] = qc
    print(f"  Pattern {pattern:10s}: {qc.num_qubits}q, depth={qc.depth():3d}, "
          f"2q-gates={n2q:3d}, params≈{sum(1 for i in qc.data if i.operation.params)}")

# ── 2c. VQE convergence on H2 ────────────────────────────────────
section("2c. VQE on H2 (layered pattern)")

from zx_motifs.pipeline.evaluation import vqe_test

qc_magic = pattern_circuits["layered"]
H_h2 = h2.hamiltonian_matrix()
vqe = vqe_test(qc_magic, h2.n_qubits, H_h2, n_restarts=10, maxiter=300)
rel_err = abs(vqe["best_energy"] - h2.exact_energy) / abs(h2.exact_energy)
print(f"  Best energy:  {vqe['best_energy']:+.6f}")
print(f"  Exact energy: {h2.exact_energy:+.6f}")
print(f"  Rel. error:   {rel_err:.6f}  ({rel_err*100:.2f}%)")
print(f"  Mean ± std:   {vqe['mean_energy']:.4f} ± {vqe['std_energy']:.4f}")

# ── 2d. Trainability (DLA) ────────────────────────────────────────
section("2d. Trainability Analysis")

from zx_motifs.pipeline.trainability import compute_dla_dimension

generators = [t.pauli_string for t in h2.pauli_terms
              if any(c != "I" for c in t.pauli_string)][:6]
dla = compute_dla_dimension(generators, n_qubits=h2.n_qubits)
print(f"  Generators: {len(generators)}")
print(f"  DLA dimension: {dla.dimension} / {dla.max_possible} ({dla.fraction_of_max:.2%})")
print(f"  Exponential risk: {dla.is_exponential}")

# ── 2e. Phase optimization ────────────────────────────────────────
section("2e. Phase Optimization")

from zx_motifs.pipeline.phase_opt import optimize_phase_polynomial, optimization_report

qc_opt = optimize_phase_polynomial(qc_magic)
report = optimization_report(qc_magic, qc_opt)
print(f"  Original CX:  {report['original_cx']}, depth={report['original_depth']}")
print(f"  Optimized CX: {report['optimized_cx']}, depth={report['optimized_depth']}")
print(f"  Reduction:    {report['reduction']} ({report['reduction_pct']:.1f}%)")


# ══════════════════════════════════════════════════════════════════
#  PHASE 3: Evaluation Stack & Baselines
# ══════════════════════════════════════════════════════════════════

banner("PHASE 3: Evaluation Stack & Baselines")

# ── 3a. Baselines ─────────────────────────────────────────────────
section("3a. Baseline Ansatze")

from zx_motifs.pipeline.baselines import BASELINE_FACTORIES, build_all_baselines

baselines = build_all_baselines(n_qubits=4)
print(f"  Built {len(baselines)} baselines:")
for name, bqc in baselines.items():
    print(f"    {name:20s}: depth={bqc.depth():3d}, 2q={count_2q(bqc):3d}")

# ── 3b. Scorecard ─────────────────────────────────────────────────
section("3b. Scorecard (Layer 1 structural)")

from zx_motifs.pipeline.scorecard import evaluate_scorecard

card = evaluate_scorecard(qc_magic, "MAGIC-layered", h2.n_qubits, layers=[1])
print(f"  Ansatz: {card.ansatz_name}")
print(f"  Classification: {card.classification}")
print(f"  Layers passed: {card.n_passed}, killed: {card.n_killed}")
for lr in card.layers:
    print(f"    L{lr.layer} ({lr.name}): passed={lr.passed}, killed={lr.killed}")
    for k, v in lr.metrics.items():
        print(f"      {k}: {v}")

# ── 3c. VQE comparison: MAGIC vs best baseline ───────────────────
section("3c. MAGIC vs Baselines on H2")

n_restarts, maxiter = 5, 200
magic_result = vqe_test(qc_magic, 4, H_h2, n_restarts=n_restarts, maxiter=maxiter)
magic_err = abs(magic_result["best_energy"] - h2.exact_energy) / abs(h2.exact_energy)

baseline_results = {}
for name, bqc in list(baselines.items())[:4]:
    res = vqe_test(bqc, 4, H_h2, n_restarts=n_restarts, maxiter=maxiter)
    berr = abs(res["best_energy"] - h2.exact_energy) / abs(h2.exact_energy)
    baseline_results[name] = (res["best_energy"], berr)

print(f"  {'Ansatz':20s}  {'Energy':>10s}  {'Rel.Err':>8s}")
print(f"  {'─'*20}  {'─'*10}  {'─'*8}")
print(f"  {'MAGIC-layered':20s}  {magic_result['best_energy']:+10.6f}  {magic_err:8.4f}")
for name, (en, err) in baseline_results.items():
    print(f"  {name:20s}  {en:+10.6f}  {err:8.4f}")
print(f"  {'Exact':20s}  {h2.exact_energy:+10.6f}")

# ── 3d. Statistical comparison ────────────────────────────────────
section("3d. Statistical Comparison")

from zx_motifs.pipeline.statistics import bootstrap_ci, cohens_d, compare_results

# Multi-seed for a fair comparison
magic_energies = np.array(magic_result["all_energies"])
# Pick the best baseline
best_bl_name = min(baseline_results, key=lambda k: baseline_results[k][1])
best_bl_qc = baselines[best_bl_name]
bl_result = vqe_test(best_bl_qc, 4, H_h2, n_restarts=10, maxiter=maxiter)
bl_energies = np.array(bl_result["all_energies"])

comp = compare_results(magic_energies, bl_energies, "MAGIC", best_bl_name, "H2")
print(f"  MAGIC mean: {comp.mean_a:.4f} ± {comp.std_a:.4f}")
print(f"  {best_bl_name} mean: {comp.mean_b:.4f} ± {comp.std_b:.4f}")
print(f"  Cohen's d: {comp.cohens_d:.3f}")
print(f"  Cliff's delta: {comp.cliffs_delta:.3f}")
print(f"  Wilcoxon p: {comp.wilcoxon_p:.4f}")
print(f"  Significant: {comp.significant}")

# ── 3e. Classical hardness ────────────────────────────────────────
section("3e. Classical Hardness Certificates")

from zx_motifs.pipeline.classical_hardness import evaluate_classical_hardness

hardness = evaluate_classical_hardness(qc_magic)
for h in hardness:
    print(f"  {h.method:15s}: tractable={h.classically_tractable}, "
          f"error={h.error:.4g}, details={h.details}")

# ── 3f. Noise analysis ────────────────────────────────────────────
section("3f. Noise Analysis")

from zx_motifs.pipeline.noise_analysis import (
    evaluate_t0_ideal,
    fault_tolerant_estimate,
    gate_budget_analysis,
)

t0 = evaluate_t0_ideal(qc_magic, H_h2)
print(f"  T0 ideal energy: {t0.energy:+.6f}")

budget = gate_budget_analysis(qc_magic, error_rate=1e-3)
print(f"  Gate budget: 2q={budget['n_2q_gates']}, total_err={budget['total_error_estimate']:.4f}, "
      f"fidelity={budget['effective_fidelity']:.4f}")
print(f"  Exceeds budget: {budget['exceeds_budget']}")

ft = fault_tolerant_estimate(qc_magic)
print(f"  Fault-tolerant: logical_qubits={ft['logical_qubits']}, T-count={ft['t_count']}, "
      f"physical_qubits≈{ft['total_physical_qubits']}")


# ══════════════════════════════════════════════════════════════════
#  PHASE 4: Evolutionary Search
# ══════════════════════════════════════════════════════════════════

banner("PHASE 4: Evolutionary Search")

# ── 4a. Recipe DSL ────────────────────────────────────────────────
section("4a. Recipe DSL")

from zx_motifs.pipeline.evolution import (
    EvolutionConfig,
    Recipe,
    evaluate_recipe,
    run_ablation,
    run_evolution,
    seed_population,
)

r = Recipe(pattern="layered", n_layers=2, symmetry_constraints=["z2_parity"])
print(f"  Recipe: pattern={r.pattern}, layers={r.n_layers}, "
      f"symmetries={r.symmetry_constraints}")
print(f"  Config: {r.to_config()}")

# ── 4b. Evaluate a single recipe ─────────────────────────────────
section("4b. Single Recipe Fitness")

tfim = tfim_1d(4)
fitness = evaluate_recipe(r, tfim, n_restarts=5, maxiter=150)
print(f"  Energy: {fitness.energy:+.6f}")
print(f"  Rel. error: {fitness.relative_error:.4f}")
print(f"  Gate count: {fitness.gate_count}")
print(f"  Fitness score: {fitness.fitness:.4f}")
print(f"  Killed: {fitness.killed}")

# ── 4c. Evolutionary search ──────────────────────────────────────
section("4c. Evolutionary Search (3 generations × 6 individuals)")

evo_config = EvolutionConfig(
    population_size=6,
    n_generations=3,
    n_restarts=3,
    maxiter=100,
    elitism=1,
    seed=42,
)
evo_result = run_evolution(tfim, evo_config)

print(f"  Generations: {len(evo_result.history)}")
for h in evo_result.history:
    print(f"    Gen {h['generation']}: best_fit={h['best_fitness']:.4f}, "
          f"best_energy={h['best_energy']:+.4f}, alive={h['alive']}/{h['population_size']}")

best = evo_result.best_recipe
print(f"  Best recipe: pattern={best.pattern}, layers={best.n_layers}, "
      f"budget={best.gate_budget}")
print(f"  Best fitness: {evo_result.best_fitness.fitness:.4f}")
print(f"  Best energy: {evo_result.best_fitness.energy:+.6f} "
      f"(exact: {tfim.exact_energy:+.6f})")

# ── 4d. Ablation study ───────────────────────────────────────────
section("4d. Ablation Study on Best Recipe")

ablation = run_ablation(best, tfim, n_restarts=3, maxiter=100)
summary = ablation.summary()
print(f"  Baseline fitness: {summary['baseline']['fitness']:.4f}")
print(f"  Ablations:")
for name, info in summary["ablations"].items():
    delta = info["delta_fitness"]
    direction = "↑" if delta > 0 else "↓" if delta < 0 else "="
    print(f"    {name:20s}: fitness={info['fitness']:.4f}  "
          f"Δ={delta:+.4f} {direction}")


# ══════════════════════════════════════════════════════════════════
#  PHASE 5: Visualization
# ══════════════════════════════════════════════════════════════════

banner("PHASE 5: Visualization")

import matplotlib
matplotlib.use("Agg")

from zx_motifs.pipeline.viz import (
    plot_ablation,
    plot_energy_comparison,
    plot_evolution_history,
    plot_scaling,
    plot_scorecard_comparison,
)
from zx_motifs.pipeline.statistics import fit_scaling

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# ── 5a. Evolution history plot ────────────────────────────────────
section("5a. Evolution History")
fig, ax = plot_evolution_history(evo_result.history, metric="best_fitness")
fig.savefig(output_dir / "evolution_history.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {output_dir / 'evolution_history.png'}")

# ── 5b. Scorecard comparison ─────────────────────────────────────
section("5b. Scorecard Comparison")

scorecards_data = [
    card.summary(),
    evaluate_scorecard(
        baselines["B1_hea_cz"], "B1_hea_cz", 4, layers=[1],
    ).summary(),
    evaluate_scorecard(
        baselines["B3_uccsd"], "B3_uccsd", 4, layers=[1],
    ).summary(),
]
fig, ax = plot_scorecard_comparison(scorecards_data)
fig.savefig(output_dir / "scorecard_comparison.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {output_dir / 'scorecard_comparison.png'}")

# ── 5c. Ablation plot ────────────────────────────────────────────
section("5c. Ablation Impact")
fig, ax = plot_ablation(summary, title="Ablation: Best Evolved Recipe")
fig.savefig(output_dir / "ablation.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {output_dir / 'ablation.png'}")

# ── 5d. Scaling analysis ─────────────────────────────────────────
section("5d. Scaling Analysis (gate count vs system size)")

ns = np.array([4, 6, 8])
gate_counts = []
for n in ns:
    p = tfim_1d(n)
    config = MAGICConfig(pattern="layered", n_layers=2)
    qc = assemble_magic(problem=p, config=config)
    gate_counts.append(count_2q(qc))
gate_counts = np.array(gate_counts, dtype=float)

fits = fit_scaling(ns, gate_counts)
fig, ax = plot_scaling(ns, gate_counts, fits=fits, ylabel="2-qubit gates",
                       title="MAGIC Gate Count Scaling")
fig.savefig(output_dir / "scaling.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {output_dir / 'scaling.png'}")
for f in fits:
    print(f"    {f.model}: R²={f.r_squared:.4f}, params={f.params}")

# ── 5e. Energy distribution box plot ─────────────────────────────
section("5e. Energy Distributions")

energy_data = {
    "MAGIC": np.array(magic_result["all_energies"]),
    best_bl_name: np.array(bl_result["all_energies"]),
}
fig, ax = plot_energy_comparison(energy_data, exact_energy=h2.exact_energy,
                                 title="H2 Energy Distributions")
fig.savefig(output_dir / "energy_comparison.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {output_dir / 'energy_comparison.png'}")


# ══════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════

banner("SUMMARY")

elapsed = time.perf_counter() - _t0
print(f"""
  Pipeline components exercised:
    Phase 1: Phase polynomials, interaction graphs, benchmark suite,
             motif registry, fingerprint matrix
    Phase 2: Hamiltonian decomposition, 4 assembly patterns, VQE,
             DLA trainability, phase optimization
    Phase 3: 8 baselines, scorecard, statistical comparison,
             classical hardness, noise analysis
    Phase 4: Recipe DSL, fitness evaluation, evolutionary search (3 gen),
             ablation study
    Phase 5: 5 publication-ready plots saved to output/

  Key results:
    H2 MAGIC rel. error:   {magic_err:.4f}
    TFIM evolved fitness:  {evo_result.best_fitness.fitness:.4f}
    Motif library:         {len(MOTIF_REGISTRY)} motifs
    Fingerprint matrix:    {counts_df.shape}
    Total time:            {elapsed:.1f}s
""")
