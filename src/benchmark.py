"""
Metrics, simulation, and baseline comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from qiskit.qasm2 import loads as qasm2_loads
from qiskit.quantum_info import Statevector

logger = logging.getLogger(__name__)

TWO_QUBIT_GATES = {
    "cx", "cz", "swap", "ecr", "rzz", "rxx", "ryy", "cp", "crz", "cry", "crx",
}


# ── Metrics ─────────────────────────────────────────────────────────


@dataclass
class CircuitMetrics:
    """Static analysis metrics for a quantum circuit."""

    n_qubits: int = 0
    gate_count: int = 0
    two_qubit_count: int = 0
    t_count: int = 0
    depth: int = 0
    gate_density: float = 0.0
    entanglement_ratio: float = 0.0


def compute_metrics_from_qasm(qasm: str) -> CircuitMetrics:
    """Compute metrics by parsing an OpenQASM 2.0 string."""
    qc = qasm2_loads(qasm)
    ops = qc.count_ops()

    n_qubits = qc.num_qubits
    gate_count = qc.size()
    depth = qc.depth()
    two_qubit_count = sum(ops.get(g, 0) for g in TWO_QUBIT_GATES)
    t_count = ops.get("t", 0) + ops.get("tdg", 0)
    gate_density = gate_count / max(n_qubits * depth, 1)
    entanglement_ratio = two_qubit_count / max(gate_count, 1)

    return CircuitMetrics(
        n_qubits=n_qubits,
        gate_count=gate_count,
        two_qubit_count=two_qubit_count,
        t_count=t_count,
        depth=depth,
        gate_density=gate_density,
        entanglement_ratio=entanglement_ratio,
    )


# ── Simulation ──────────────────────────────────────────────────────


@dataclass
class SimulationResult:
    """Output of simulating a quantum circuit."""

    statevector: np.ndarray | None = None
    fidelity: float | None = None


def simulate_statevector(qasm: str) -> SimulationResult:
    """Run a circuit on an ideal statevector simulator."""
    try:
        qc = qasm2_loads(qasm)
        qc.remove_final_measurements(inplace=True)
        sv = Statevector.from_instruction(qc)
        return SimulationResult(statevector=sv.data)
    except Exception:
        logger.debug("Statevector simulation failed", exc_info=True)
        return SimulationResult()


def compute_fidelity(
    sv_candidate: np.ndarray,
    sv_reference: np.ndarray,
) -> float:
    """State fidelity between two statevectors: F = |<ref|cand>|^2."""
    if sv_candidate.shape != sv_reference.shape:
        return 0.0
    return float(abs(np.vdot(sv_reference, sv_candidate)) ** 2)


# ── Comparison ──────────────────────────────────────────────────────


@dataclass
class ComparisonResult:
    """Head-to-head comparison of a candidate vs a baseline."""

    candidate_id: str
    baseline_id: str
    candidate_metrics: CircuitMetrics
    baseline_metrics: CircuitMetrics
    improvements: dict[str, float] = field(default_factory=dict)
    overall_better: bool = False


def compute_improvement(
    candidate: CircuitMetrics,
    baseline: CircuitMetrics,
) -> dict[str, float]:
    """Compute per-metric relative improvement.

    improvement = (baseline - candidate) / baseline
    Positive values mean the candidate is better.
    """
    improvements: dict[str, float] = {}
    for metric in ("gate_count", "two_qubit_count", "t_count", "depth"):
        baseline_val = getattr(baseline, metric)
        candidate_val = getattr(candidate, metric)
        if baseline_val == 0:
            improvements[metric] = 0.0 if candidate_val == 0 else -1.0
        else:
            improvements[metric] = (baseline_val - candidate_val) / baseline_val
    return improvements


def compare_against_baselines(
    candidate_qasm: str,
    candidate_id: str,
    baseline_dir: str,
) -> list[ComparisonResult]:
    """Compare a candidate circuit against all relevant baselines."""
    baseline_path = Path(baseline_dir)
    results: list[ComparisonResult] = []

    candidate_metrics = compute_metrics_from_qasm(candidate_qasm)
    candidate_sim = simulate_statevector(candidate_qasm)

    for qasm_file in sorted(baseline_path.glob("*.qasm")):
        try:
            baseline_qasm = qasm_file.read_text()
            baseline_metrics = compute_metrics_from_qasm(baseline_qasm)

            # Skip if qubit count differs
            if baseline_metrics.n_qubits != candidate_metrics.n_qubits:
                continue

            improvements = compute_improvement(candidate_metrics, baseline_metrics)

            # Compute fidelity if both statevectors available
            baseline_sim = simulate_statevector(baseline_qasm)
            if candidate_sim.statevector is not None and baseline_sim.statevector is not None:
                fidelity = compute_fidelity(candidate_sim.statevector, baseline_sim.statevector)
                improvements["simulated_fidelity"] = fidelity

            # Overall better = majority of improvement values > 0
            improvement_vals = [v for k, v in improvements.items() if k != "simulated_fidelity"]
            overall_better = sum(1 for v in improvement_vals if v > 0) > len(improvement_vals) / 2

            results.append(ComparisonResult(
                candidate_id=candidate_id,
                baseline_id=qasm_file.stem,
                candidate_metrics=candidate_metrics,
                baseline_metrics=baseline_metrics,
                improvements=improvements,
                overall_better=overall_better,
            ))
        except Exception:
            logger.warning("Failed to compare against %s", qasm_file.name, exc_info=True)

    return results


def compare_against_source_algorithms(
    candidate_qasm: str,
    candidate_id: str,
    source_keys: list[str],
    corpus_dir: str,
) -> list[ComparisonResult]:
    """Compare a candidate against the algorithms that contributed its webs."""
    corpus_path = Path(corpus_dir)
    results: list[ComparisonResult] = []

    candidate_metrics = compute_metrics_from_qasm(candidate_qasm)
    candidate_sim = simulate_statevector(candidate_qasm)

    for source_key in source_keys:
        # Match naming convention: {source_key}_{n}q.qasm
        for qasm_file in sorted(corpus_path.glob(f"{source_key}_*q.qasm")):
            try:
                baseline_qasm = qasm_file.read_text()
                baseline_metrics = compute_metrics_from_qasm(baseline_qasm)

                if baseline_metrics.n_qubits != candidate_metrics.n_qubits:
                    continue

                improvements = compute_improvement(candidate_metrics, baseline_metrics)

                baseline_sim = simulate_statevector(baseline_qasm)
                if candidate_sim.statevector is not None and baseline_sim.statevector is not None:
                    fidelity = compute_fidelity(candidate_sim.statevector, baseline_sim.statevector)
                    improvements["simulated_fidelity"] = fidelity

                improvement_vals = [v for k, v in improvements.items() if k != "simulated_fidelity"]
                overall_better = sum(1 for v in improvement_vals if v > 0) > len(improvement_vals) / 2

                results.append(ComparisonResult(
                    candidate_id=candidate_id,
                    baseline_id=qasm_file.stem,
                    candidate_metrics=candidate_metrics,
                    baseline_metrics=baseline_metrics,
                    improvements=improvements,
                    overall_better=overall_better,
                ))
            except Exception:
                logger.warning("Failed to compare against %s", qasm_file.name, exc_info=True)

    return results
