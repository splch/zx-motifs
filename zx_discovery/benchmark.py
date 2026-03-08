"""
Step 6 - Benchmark Candidates Against an Application Suite
============================================================
Each surviving candidate circuit is evaluated on a suite of
application-level tasks.  Because the candidates are *new* circuits
whose functionality is not known a priori, the benchmarks are
**structure-oriented** and **resource-oriented** rather than
task-specific.

Benchmark Metrics
-----------------
1. **Gate efficiency** - total gate count, 2-qubit gate count, T-count
   normalised by qubit width.
2. **Entanglement capacity** - average bipartite entanglement entropy
   across all cuts of the output state (computed via statevector
   simulation for small circuits).
3. **Expressibility** - how uniformly the circuit's output states
   cover the Hilbert space when its free parameters are varied
   (approximated by sampling random input states).
4. **Circuit depth** - total depth and critical-path 2-qubit depth.
5. **Equivalence novelty** - check that the unitary is *not* equivalent
   (up to global phase) to any circuit already in the corpus.

Comparison Baselines
--------------------
Every metric is also computed for the *original corpus algorithms* so
that candidates can be directly compared.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

import pyzx as zx
from qiskit.circuit import QuantumCircuit
from qiskit import qasm2

from .extractor import ExtractedCandidate


@dataclass
class BenchmarkResult:
    """Benchmark scores for a single circuit."""
    name: str
    n_qubits: int
    gate_count: int
    two_qubit_count: int
    t_count: int
    depth: int
    two_qubit_depth: int
    gates_per_qubit: float
    entanglement_entropy: float       # avg bipartite entropy
    expressibility_score: float       # Hilbert-space coverage
    is_novel: bool                    # not equivalent to any corpus circuit
    source_info: str = ""


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _pyzx_to_qiskit(qasm_str: str) -> Optional[QuantumCircuit]:
    """Convert a PyZX QASM string to a Qiskit QuantumCircuit."""
    try:
        qc = qasm2.loads(
            qasm_str,
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS
        )
        return qc
    except Exception:
        return None


def _statevector(qc: QuantumCircuit) -> Optional[np.ndarray]:
    """Compute the output statevector of a circuit applied to |0…0⟩.

    Uses Qiskit's built-in Operator for small circuits.
    """
    from qiskit.quantum_info import Statevector
    try:
        sv = Statevector.from_instruction(qc)
        return sv.data
    except Exception:
        return None


def _unitary_matrix(qc: QuantumCircuit) -> Optional[np.ndarray]:
    """Compute the unitary matrix of a circuit."""
    from qiskit.quantum_info import Operator
    try:
        op = Operator(qc)
        return op.data
    except Exception:
        return None


def _bipartite_entropy(sv: np.ndarray, n_qubits: int) -> float:
    """Average bipartite entanglement entropy across all bipartitions.

    For each cut (A, B) where |A| = k for k = 1 .. n//2, compute the
    von Neumann entropy of the reduced density matrix on A.  Return
    the average.
    """
    if n_qubits > 10:
        return -1.0  # too expensive
    dim = 2 ** n_qubits
    state = sv.reshape((dim, 1))
    entropies = []
    for k in range(1, n_qubits // 2 + 1):
        dim_a = 2 ** k
        dim_b = 2 ** (n_qubits - k)
        rho = state @ state.conj().T
        rho_reshaped = rho.reshape((dim_a, dim_b, dim_a, dim_b))
        rho_a = np.trace(rho_reshaped, axis1=1, axis2=3)
        # Von Neumann entropy
        eigvals = np.linalg.eigvalsh(rho_a)
        eigvals = eigvals[eigvals > 1e-12]
        S = -np.sum(eigvals * np.log2(eigvals + 1e-15))
        entropies.append(float(S))
    return float(np.mean(entropies)) if entropies else 0.0


def _expressibility(qc: QuantumCircuit, n_samples: int = 50) -> float:
    """Estimate expressibility by measuring spread over Hilbert space.

    Applies the circuit to ``n_samples`` random input product states
    and measures pairwise fidelity variance.  Lower variance ≈ more
    expressive.  Returns a score in [0, 1] where higher is better.
    """
    n = qc.num_qubits
    if n > 8:
        return -1.0

    from qiskit.quantum_info import Statevector, random_statevector
    outputs = []
    for _ in range(n_samples):
        # Random product state: tensor product of n random 1-qubit states
        angles = np.random.uniform([0, 0], [np.pi, 2 * np.pi], (n, 2))
        init_vec = np.array([1.0 + 0j])
        for theta, phi in angles:
            qubit = np.array([np.cos(theta / 2),
                              np.exp(1j * phi) * np.sin(theta / 2)])
            init_vec = np.kron(init_vec, qubit)
        init_sv = Statevector(init_vec)
        try:
            out = init_sv.evolve(qc)
            outputs.append(out.data)
        except Exception:
            continue

    if len(outputs) < 2:
        return 0.0

    # Pairwise fidelities
    fidelities = []
    for i in range(len(outputs)):
        for j in range(i + 1, min(i + 20, len(outputs))):
            f = abs(np.dot(outputs[i].conj(), outputs[j])) ** 2
            fidelities.append(f)

    if not fidelities:
        return 0.0

    # Expressibility ∝ how close the fidelity distribution is to Haar-random
    # For Haar-random on d-dim space, E[F] = 1/d
    d = 2 ** n
    mean_f = np.mean(fidelities)
    # Score: higher = more expressive (closer to Haar)
    ideal_mean = 1.0 / d
    score = 1.0 - min(abs(mean_f - ideal_mean) / max(ideal_mean, 1e-10), 1.0)
    return float(score)


def _circuit_depth(qc: QuantumCircuit) -> Tuple[int, int]:
    """Return (total_depth, two_qubit_depth)."""
    total = qc.depth()
    # 2-qubit depth
    two_q_ops = [
        inst for inst in qc.data
        if inst.operation.num_qubits == 2
    ]
    if two_q_ops:
        sub = QuantumCircuit(qc.num_qubits)
        for inst in two_q_ops:
            sub.append(inst.operation, inst.qubits)
        tq_depth = sub.depth()
    else:
        tq_depth = 0
    return total, tq_depth


# ---------------------------------------------------------------------------
# Novelty checking
# ---------------------------------------------------------------------------

def _are_equivalent(u1: np.ndarray, u2: np.ndarray) -> bool:
    """Check if two unitaries are equivalent up to global phase."""
    if u1.shape != u2.shape:
        return False
    product = u1 @ u2.conj().T
    # If equivalent, product = e^{iθ} I
    diag = np.diag(product)
    if not np.allclose(np.abs(diag), 1.0, atol=1e-6):
        return False
    phase = diag[0]
    return np.allclose(product, phase * np.eye(u1.shape[0]), atol=1e-6)


# ---------------------------------------------------------------------------
# Public benchmarking interface
# ---------------------------------------------------------------------------

def benchmark_candidates(
    extracted: List[ExtractedCandidate],
    corpus_qasm: Dict[str, str],
) -> Tuple[List[BenchmarkResult], List[BenchmarkResult]]:
    """Benchmark candidates and corpus algorithms.

    Parameters
    ----------
    extracted : list
        Output of ``extractor.filter_candidates``.
    corpus_qasm : dict
        ``{name: qasm_string}`` for the original algorithm corpus.

    Returns
    -------
    candidate_results : list of BenchmarkResult
    baseline_results : list of BenchmarkResult
    """
    # 1. Precompute corpus unitaries for novelty checking
    corpus_unitaries: Dict[str, np.ndarray] = {}
    for name, qasm_str in corpus_qasm.items():
        qc = _pyzx_to_qiskit(qasm_str)
        if qc is not None and qc.num_qubits <= 8:
            u = _unitary_matrix(qc)
            if u is not None:
                corpus_unitaries[name] = u

    # 2. Benchmark corpus (baselines)
    baseline_results: List[BenchmarkResult] = []
    for name, qasm_str in corpus_qasm.items():
        qc = _pyzx_to_qiskit(qasm_str)
        if qc is None:
            continue
        result = _benchmark_single(qc, name, corpus_unitaries, is_corpus=True)
        if result is not None:
            result.source_info = "corpus"
            baseline_results.append(result)

    # 3. Benchmark candidates
    candidate_results: List[BenchmarkResult] = []
    for ec in extracted:
        qc = _pyzx_to_qiskit(ec.qasm)
        if qc is None:
            continue
        result = _benchmark_single(
            qc, ec.candidate.candidate_id, corpus_unitaries, is_corpus=False
        )
        if result is not None:
            result.source_info = (
                f"composed from webs: "
                f"{', '.join(w[:8] for w in ec.candidate.source_webs)}"
            )
            candidate_results.append(result)

    return candidate_results, baseline_results


def _benchmark_single(
    qc: QuantumCircuit,
    name: str,
    corpus_unitaries: Dict[str, np.ndarray],
    is_corpus: bool,
) -> Optional[BenchmarkResult]:
    """Benchmark a single circuit."""
    n = qc.num_qubits
    if n < 1 or n > 10:
        return None

    # Gate counts
    gate_count = qc.size()
    two_q = sum(1 for inst in qc.data if inst.operation.num_qubits == 2)
    t_count = sum(1 for inst in qc.data
                  if inst.operation.name.lower() in ('t', 'tdg'))

    # Depth
    total_depth, tq_depth = _circuit_depth(qc)

    # Entanglement
    sv = _statevector(qc)
    ent = _bipartite_entropy(sv, n) if sv is not None else -1.0

    # Expressibility
    expr = _expressibility(qc, n_samples=30)

    # Novelty
    is_novel = True
    if not is_corpus and n <= 8:
        u = _unitary_matrix(qc)
        if u is not None:
            for cname, cu in corpus_unitaries.items():
                if _are_equivalent(u, cu):
                    is_novel = False
                    break

    return BenchmarkResult(
        name=name,
        n_qubits=n,
        gate_count=gate_count,
        two_qubit_count=two_q,
        t_count=t_count,
        depth=total_depth,
        two_qubit_depth=tq_depth,
        gates_per_qubit=gate_count / max(n, 1),
        entanglement_entropy=ent,
        expressibility_score=expr,
        is_novel=is_novel,
    )
