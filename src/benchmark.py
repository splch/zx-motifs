"""
benchmark.py — Metrics, simulation, and baseline comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
    raise NotImplementedError


# ── Simulation ──────────────────────────────────────────────────────


@dataclass
class SimulationResult:
    """Output of simulating a quantum circuit."""

    statevector: np.ndarray | None = None
    fidelity: float | None = None


def simulate_statevector(qasm: str) -> SimulationResult:
    """Run a circuit on an ideal statevector simulator."""
    raise NotImplementedError


def compute_fidelity(
    sv_candidate: np.ndarray,
    sv_reference: np.ndarray,
) -> float:
    """State fidelity between two statevectors: F = |<ref|cand>|^2."""
    raise NotImplementedError


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
    raise NotImplementedError


def compare_against_baselines(
    candidate_qasm: str,
    candidate_id: str,
    baseline_dir: str,
) -> list[ComparisonResult]:
    """Compare a candidate circuit against all relevant baselines."""
    raise NotImplementedError


def compare_against_source_algorithms(
    candidate_qasm: str,
    candidate_id: str,
    source_keys: list[str],
    corpus_dir: str,
) -> list[ComparisonResult]:
    """Compare a candidate against the algorithms that contributed its webs."""
    raise NotImplementedError
