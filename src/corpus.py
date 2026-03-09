"""
corpus.py — Algorithm corpus: builders, registry, and QASM export.

Merges stage1_corpus/{algorithms,registry,export}.py into a single module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from qiskit import QuantumCircuit, transpile as qiskit_transpile
from qiskit.circuit.library import (
    ExactReciprocalGate,
    IntegerComparatorGate,
    ModularAdderGate,
    PauliEvolutionGate,
    QFTGate,
    ZGate,
    grover_operator,
    phase_estimation,
    qaoa_ansatz,
    real_amplitudes,
)
from qiskit.qasm2 import dumps as qasm2_dumps
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter

logger = logging.getLogger(__name__)


# ── Data ────────────────────────────────────────────────────────────


@dataclass
class AlgorithmEntry:
    """Describes one algorithm in the corpus."""

    key: str
    category: str
    builder: Callable[..., Any]  # -> QuantumCircuit
    description: str = ""
    min_qubits: int = 2
    max_qubits: int = 20


class AlgorithmRegistry:
    """In-memory catalog of available algorithms."""

    def __init__(self) -> None:
        self._entries: dict[str, AlgorithmEntry] = {}

    def register(self, entry: AlgorithmEntry) -> None:
        """Add an algorithm. Raises ValueError on duplicate key."""
        if entry.key in self._entries:
            raise ValueError(f"Duplicate key: {entry.key!r}")
        self._entries[entry.key] = entry

    def get(self, key: str) -> AlgorithmEntry:
        """Retrieve an entry by key. Raises KeyError if not found."""
        return self._entries[key]

    def by_category(self, category: str) -> list[AlgorithmEntry]:
        """Return all entries matching the given category."""
        return [e for e in self._entries.values() if e.category == category]

    def all_keys(self) -> list[str]:
        """Return a sorted list of all registered keys."""
        return sorted(self._entries)


def build_default_registry() -> AlgorithmRegistry:
    """Construct a registry pre-populated with all implemented algorithms."""
    reg = AlgorithmRegistry()
    entries = [
        AlgorithmEntry("qft", "fourier", build_qft, "Quantum Fourier Transform"),
        AlgorithmEntry(
            "qpe", "fourier", build_qpe,
            "Quantum Phase Estimation", min_qubits=3,
        ),
        AlgorithmEntry(
            "grover", "search", build_grover,
            "Grover's search", min_qubits=2,
        ),
        AlgorithmEntry(
            "amplitude_estimation", "search", build_amplitude_estimation,
            "Canonical amplitude estimation", min_qubits=3,
        ),
        AlgorithmEntry(
            "hhl", "search", build_hhl,
            "HHL linear solver", min_qubits=4,
        ),
        AlgorithmEntry(
            "vqe_ansatz", "variational", build_vqe_ansatz,
            "RealAmplitudes VQE ansatz",
        ),
        AlgorithmEntry(
            "qaoa", "variational", build_qaoa,
            "QAOA for MaxCut",
        ),
        AlgorithmEntry(
            "trotter", "simulation", build_trotter,
            "Suzuki-Trotter for Heisenberg chain",
        ),
        AlgorithmEntry(
            "steane_encoder", "error_correction", build_steane_encoder,
            "[[7,1,3]] Steane code encoder", min_qubits=7, max_qubits=7,
        ),
        AlgorithmEntry(
            "adder", "arithmetic", build_adder,
            "Modular adder", min_qubits=4,
        ),
        AlgorithmEntry(
            "comparator", "arithmetic", build_comparator,
            "Integer comparator", min_qubits=3,
        ),
    ]
    for entry in entries:
        reg.register(entry)
    return reg


# ── Algorithm builders ──────────────────────────────────────────────


def build_qft(n_qubits: int) -> QuantumCircuit:
    """Quantum Fourier Transform."""
    qc = QuantumCircuit(n_qubits, name="qft")
    qc.append(QFTGate(n_qubits), range(n_qubits))
    return qc


def build_qpe(n_qubits: int) -> QuantumCircuit:
    """Quantum Phase Estimation for a fixed unitary (Z gate)."""
    n_eval = n_qubits - 1
    qc = phase_estimation(n_eval, ZGate())
    qc.name = "qpe"
    return qc


def build_grover(n_qubits: int, n_iterations: int = 1) -> QuantumCircuit:
    """Grover's search with an oracle marking |11...1⟩."""
    # Oracle: multi-controlled Z marking the all-ones state
    oracle = QuantumCircuit(n_qubits, name="oracle")
    if n_qubits == 1:
        oracle.z(0)
    elif n_qubits == 2:
        oracle.cz(0, 1)
    else:
        oracle.h(n_qubits - 1)
        oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        oracle.h(n_qubits - 1)

    go = grover_operator(oracle)

    qc = QuantumCircuit(n_qubits, name="grover")
    qc.h(range(n_qubits))
    for _ in range(n_iterations):
        qc.compose(go, inplace=True)
    return qc


def build_amplitude_estimation(n_qubits: int) -> QuantumCircuit:
    """Canonical amplitude estimation circuit (QPE on Grover operator)."""
    # 1-qubit search space, n_qubits-1 evaluation qubits
    oracle = QuantumCircuit(1, name="oracle")
    oracle.z(0)

    a_op = QuantumCircuit(1, name="A")
    a_op.ry(np.pi / 3, 0)

    go = grover_operator(oracle, state_preparation=a_op)
    n_eval = n_qubits - 1
    qc = phase_estimation(n_eval, go.to_gate())
    qc.name = "amplitude_estimation"
    return qc


def build_hhl(n_qubits: int) -> QuantumCircuit:
    """HHL algorithm for a toy linear system."""
    n_clock = n_qubits - 2  # clock/evaluation qubits
    # Layout: [0..n_clock-1] clock, [n_clock] system, [n_clock+1] ancilla

    qpe_circ = phase_estimation(n_clock, ZGate())
    reciprocal = ExactReciprocalGate(n_clock, 0.5)

    qc = QuantumCircuit(n_qubits, name="hhl")
    qc.x(n_clock)  # prepare system qubit in |1⟩
    qc.append(qpe_circ.to_gate(label="QPE"), range(n_clock + 1))
    qc.append(reciprocal, list(range(n_clock)) + [n_clock + 1])
    qc.append(qpe_circ.to_gate(label="QPE_inv").inverse(), range(n_clock + 1))
    return qc


def build_vqe_ansatz(n_qubits: int, depth: int = 2) -> QuantumCircuit:
    """Hardware-efficient VQE ansatz (RealAmplitudes-style)."""
    ansatz = real_amplitudes(n_qubits, reps=depth)
    rng = np.random.default_rng(42)
    param_values = rng.uniform(-np.pi, np.pi, size=len(ansatz.parameters))
    qc = ansatz.assign_parameters(dict(zip(ansatz.parameters, param_values)))
    qc.name = "vqe_ansatz"
    return qc


def build_qaoa(n_qubits: int, p: int = 1) -> QuantumCircuit:
    """QAOA circuit for a linear-chain MaxCut instance."""
    # Cost operator: ZZ on adjacent pairs
    terms = [(f"{'I' * i}ZZ{'I' * (n_qubits - i - 2)}", 1.0) for i in range(n_qubits - 1)]
    cost_op = SparsePauliOp.from_list(terms)

    ansatz = qaoa_ansatz(cost_op, reps=p)
    rng = np.random.default_rng(42)
    param_values = rng.uniform(-np.pi, np.pi, size=len(ansatz.parameters))
    qc = ansatz.assign_parameters(dict(zip(ansatz.parameters, param_values)))
    qc.name = "qaoa"
    return qc


def build_trotter(n_qubits: int, steps: int = 2) -> QuantumCircuit:
    """Suzuki-Trotter decomposition for a Heisenberg chain."""
    # Heisenberg Hamiltonian: sum of XX + YY + ZZ on adjacent pairs
    terms = []
    for i in range(n_qubits - 1):
        for pauli in ["XX", "YY", "ZZ"]:
            label = "I" * i + pauli + "I" * (n_qubits - i - 2)
            terms.append((label, 1.0))
    hamiltonian = SparsePauliOp.from_list(terms)

    evo_gate = PauliEvolutionGate(
        hamiltonian, time=1.0, synthesis=SuzukiTrotter(order=2, reps=steps),
    )
    qc = QuantumCircuit(n_qubits, name="trotter")
    qc.append(evo_gate, range(n_qubits))
    return qc


def build_steane_encoder(n_qubits: int = 7) -> QuantumCircuit:
    """Encoding circuit for the [[7,1,3]] Steane code."""
    if n_qubits != 7:
        raise ValueError("Steane code requires exactly 7 qubits")
    qc = QuantumCircuit(7, name="steane_encoder")
    # Logical qubit on q0; ancillas q1–q6
    # Hadamard on generator qubits
    qc.h(3)
    qc.h(4)
    qc.h(5)
    # CNOT pattern for [7,1,3] code
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(3, 0)
    qc.cx(3, 1)
    qc.cx(3, 6)
    qc.cx(4, 0)
    qc.cx(4, 2)
    qc.cx(4, 6)
    qc.cx(5, 1)
    qc.cx(5, 2)
    qc.cx(5, 6)
    return qc


def build_adder(n_qubits: int) -> QuantumCircuit:
    """Quantum modular adder."""
    num_state = n_qubits // 2
    gate = ModularAdderGate(num_state)
    qc = QuantumCircuit(n_qubits, name="adder")
    qc.append(gate, range(gate.num_qubits))
    return qc


def build_comparator(n_qubits: int) -> QuantumCircuit:
    """Quantum integer comparator."""
    num_state = n_qubits - 1
    value = 2 ** (num_state - 1)  # compare against midpoint
    gate = IntegerComparatorGate(num_state, value)
    qc = QuantumCircuit(n_qubits, name="comparator")
    qc.append(gate, range(gate.num_qubits))
    return qc


# ── Export ──────────────────────────────────────────────────────────


def transpile_to_gate_set(
    circuit: QuantumCircuit,
    gate_set: list[str],
) -> QuantumCircuit:
    """Transpile a circuit to a specific basis gate set."""
    return qiskit_transpile(circuit, basis_gates=gate_set, optimization_level=2)


def circuit_to_qasm(circuit: QuantumCircuit) -> str:
    """Convert a QuantumCircuit to an OpenQASM 2.0 string."""
    return qasm2_dumps(circuit)


def export_corpus(
    registry: AlgorithmRegistry,
    output_dir: str | Path,
    gate_set: list[str],
    qubit_sizes: dict[str, list[int]],
) -> list[Path]:
    """Build and export every algorithm at every requested size."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    all_sizes = sorted({n for sizes in qubit_sizes.values() for n in sizes})

    for key in registry.all_keys():
        entry = registry.get(key)
        for n in all_sizes:
            if n < entry.min_qubits or n > entry.max_qubits:
                continue
            try:
                qc = entry.builder(n)
                qc_transpiled = transpile_to_gate_set(qc, gate_set)
                qasm_str = circuit_to_qasm(qc_transpiled)
                path = output_dir / f"{key}_{n}q.qasm"
                path.write_text(qasm_str)
                written.append(path)
                logger.info("Exported %s (%d qubits)", key, n)
            except Exception:
                logger.warning("Failed to build %s at %d qubits", key, n, exc_info=True)

    return written
