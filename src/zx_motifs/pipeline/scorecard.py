"""
Seven-layer adversarial evaluation scorecard for quantum ansatze.

Each layer tests a progressively harder criterion. The layers are ordered
by computational cost (milliseconds to hours) to enable fast-fail in
evolutionary search.

Layer 1: Structural validation (unitarity, non-Clifford, extractability)
Layer 2: Expressibility & entanglement
Layer 3: Classical hardness certificates
Layer 4: Task performance (VQE)
Layer 5: Classical comparison (DMRG/CCSD)
Layer 6: Noise resilience
Layer 7: Scaling projection
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from .converter import count_t_gates, qiskit_to_zx
from .evaluation import compute_entangling_power, count_2q, vqe_test

logger = logging.getLogger(__name__)


# ── Scorecard Data Structures ──────────────────────────────────────


@dataclass
class LayerResult:
    """Result of evaluating one scorecard layer."""

    layer: int
    name: str
    passed: bool
    killed: bool
    metrics: dict = field(default_factory=dict)
    details: str = ""


@dataclass
class Scorecard:
    """Complete seven-layer evaluation result."""

    ansatz_name: str
    n_qubits: int
    layers: list[LayerResult] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return sum(1 for l in self.layers if l.passed)

    @property
    def n_killed(self) -> int:
        return sum(1 for l in self.layers if l.killed)

    @property
    def classification(self) -> str:
        if self.n_passed >= 7:
            return "credible_advantage_candidate"
        elif self.n_passed >= 6:
            return "publishable"
        elif self.n_passed >= 4:
            return "interesting"
        else:
            return "insufficient"

    def summary(self) -> dict:
        return {
            "ansatz": self.ansatz_name,
            "n_qubits": self.n_qubits,
            "classification": self.classification,
            "layers_passed": self.n_passed,
            "layers_killed": self.n_killed,
            "layer_details": {
                l.name: {"passed": l.passed, "killed": l.killed, **l.metrics}
                for l in self.layers
            },
        }


# ── Layer 1: Structural Validation ────────────────────────────────


def evaluate_layer1_structural(qc: QuantumCircuit, n_qubits: int) -> LayerResult:
    """Check unitarity, non-Clifford content, and circuit extractability."""
    metrics = {}

    # Unitarity check
    try:
        U = Operator(qc).data
        is_unitary = np.allclose(U @ U.conj().T, np.eye(2**n_qubits), atol=1e-8)
        metrics["is_unitary"] = is_unitary
    except Exception:
        is_unitary = False
        metrics["is_unitary"] = False

    # Non-Clifford content
    try:
        import pyzx as zx
        zx_circ = qiskit_to_zx(qc)
        g = zx_circ.to_graph()
        g2 = copy.deepcopy(g)
        zx.simplify.full_reduce(g2)
        t_count = count_t_gates(g2)
        metrics["t_count_after_reduce"] = t_count
        non_clifford = t_count > 0
    except Exception:
        non_clifford = True  # assume non-Clifford if we can't check
        metrics["t_count_after_reduce"] = -1

    # Circuit extractability
    try:
        zx_circ = qiskit_to_zx(qc)
        g = zx_circ.to_graph()
        extractable = True
        metrics["extractable"] = True
    except Exception:
        extractable = False
        metrics["extractable"] = False

    passed = is_unitary and extractable
    killed = not is_unitary  # Non-unitary is a hard kill

    return LayerResult(
        layer=1, name="structural", passed=passed, killed=killed,
        metrics=metrics,
        details=f"unitary={is_unitary}, non_clifford={non_clifford}, extractable={extractable}",
    )


# ── Layer 2: Expressibility & Entanglement ─────────────────────────


def _expressibility_kl(qc: QuantumCircuit, n_samples: int = 500) -> float:
    """Estimate KL divergence of fidelity distribution from Haar random.

    Lower = more expressible. Returns approximate KL divergence.
    """
    n = qc.num_qubits
    rng = np.random.default_rng(42)

    try:
        U = Operator(qc).data
    except Exception:
        return float("inf")

    fidelities = []
    for _ in range(n_samples):
        params1 = rng.uniform(-np.pi, np.pi, 4 * n)
        params2 = rng.uniform(-np.pi, np.pi, 4 * n)

        # Build two random instances
        qc1 = QuantumCircuit(n)
        qc2 = QuantumCircuit(n)
        for i in range(n):
            qc1.ry(params1[2 * i], i)
            qc1.rz(params1[2 * i + 1], i)
            qc2.ry(params2[2 * i], i)
            qc2.rz(params2[2 * i + 1], i)
        qc1.compose(qc, inplace=True)
        qc2.compose(qc, inplace=True)
        for i in range(n):
            qc1.ry(params1[2 * n + 2 * i], i)
            qc1.rz(params1[2 * n + 2 * i + 1], i)
            qc2.ry(params2[2 * n + 2 * i], i)
            qc2.rz(params2[2 * n + 2 * i + 1], i)

        sv1 = Statevector.from_instruction(qc1)
        sv2 = Statevector.from_instruction(qc2)
        fid = abs(np.vdot(sv1.data, sv2.data)) ** 2
        fidelities.append(fid)

    fidelities = np.array(fidelities)
    # Compare to Haar distribution: P(F) = (2^n - 1)(1-F)^(2^n - 2)
    dim = 2**n
    bins = np.linspace(0, 1, 50)
    hist, _ = np.histogram(fidelities, bins=bins, density=True)

    # Haar reference
    bin_centers = (bins[:-1] + bins[1:]) / 2
    haar = (dim - 1) * (1 - bin_centers) ** max(dim - 2, 0)
    haar = haar / (np.sum(haar) * (bins[1] - bins[0]))  # normalize

    # KL divergence (with smoothing)
    epsilon = 1e-10
    hist_safe = hist + epsilon
    haar_safe = haar + epsilon
    kl = float(np.sum(hist_safe * np.log(hist_safe / haar_safe) * (bins[1] - bins[0])))
    return max(0, kl)


def evaluate_layer2_expressibility(
    qc: QuantumCircuit, n_qubits: int, n_samples: int = 200,
) -> LayerResult:
    """Evaluate expressibility (KL divergence) and entangling power."""
    metrics = {}

    # Expressibility
    kl = _expressibility_kl(qc, n_samples=n_samples)
    metrics["kl_divergence"] = kl

    # Entangling power
    ep_result = compute_entangling_power(qc, n_samples=min(n_samples, 100))
    metrics["entangling_power"] = ep_result["entangling_power"]
    metrics["ep_std"] = ep_result["epd"]
    max_entropy = n_qubits / 2  # maximum bipartite entropy
    ep_ratio = ep_result["entangling_power"] / max_entropy if max_entropy > 0 else 0

    passed = kl < 0.1 and ep_ratio > 0.1
    killed = kl > 1.0 or ep_ratio < 0.01

    return LayerResult(
        layer=2, name="expressibility", passed=passed, killed=killed,
        metrics=metrics,
        details=f"KL={kl:.4f}, EP={ep_result['entangling_power']:.4f}, ratio={ep_ratio:.4f}",
    )


# ── Layer 4: Task Performance ──────────────────────────────────────


def evaluate_layer4_performance(
    qc: QuantumCircuit,
    n_qubits: int,
    hamiltonian: np.ndarray,
    exact_energy: float,
    n_restarts: int = 10,
    maxiter: int = 400,
) -> LayerResult:
    """Evaluate VQE task performance."""
    result = vqe_test(qc, n_qubits, hamiltonian, n_restarts=n_restarts, maxiter=maxiter)
    rel_error = abs(result["best_energy"] - exact_energy) / abs(exact_energy) if exact_energy != 0 else float("inf")

    metrics = {
        "best_energy": result["best_energy"],
        "exact_energy": exact_energy,
        "relative_error": rel_error,
        "mean_energy": result["mean_energy"],
        "std_energy": result["std_energy"],
        "n_params": result["n_params"],
    }

    passed = rel_error < 0.01
    killed = rel_error > 0.5  # Worse than 50% error

    return LayerResult(
        layer=4, name="performance", passed=passed, killed=killed,
        metrics=metrics,
        details=f"rel_error={rel_error:.6f}, best={result['best_energy']:.6f}",
    )


# ── Scorecard Runner ───────────────────────────────────────────────


def evaluate_scorecard(
    qc: QuantumCircuit,
    ansatz_name: str,
    n_qubits: int,
    hamiltonian: np.ndarray | None = None,
    exact_energy: float | None = None,
    layers: list[int] | None = None,
    fast_fail: bool = True,
) -> Scorecard:
    """Run the evaluation scorecard on a circuit.

    Args:
        qc: The ansatz circuit to evaluate.
        ansatz_name: Name for reporting.
        n_qubits: Number of qubits.
        hamiltonian: Hamiltonian matrix (required for Layer 4).
        exact_energy: Exact ground state energy (required for Layer 4).
        layers: Which layers to evaluate (default: [1, 2, 4]).
        fast_fail: If True, stop after first kill.

    Returns:
        Scorecard with all evaluated layers.
    """
    if layers is None:
        layers = [1, 2, 4]

    card = Scorecard(ansatz_name=ansatz_name, n_qubits=n_qubits)

    for layer_num in sorted(layers):
        if layer_num == 1:
            result = evaluate_layer1_structural(qc, n_qubits)
        elif layer_num == 2:
            result = evaluate_layer2_expressibility(qc, n_qubits)
        elif layer_num == 4:
            if hamiltonian is not None and exact_energy is not None:
                result = evaluate_layer4_performance(
                    qc, n_qubits, hamiltonian, exact_energy,
                )
            else:
                result = LayerResult(
                    layer=4, name="performance", passed=False, killed=False,
                    details="Skipped: no Hamiltonian provided",
                )
        else:
            result = LayerResult(
                layer=layer_num, name=f"layer_{layer_num}", passed=False,
                killed=False, details="Not yet implemented",
            )

        card.layers.append(result)
        if fast_fail and result.killed:
            logger.info("Fast-fail: Layer %d killed %s", layer_num, ansatz_name)
            break

    return card
