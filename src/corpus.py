"""
Algorithm corpus: builders, registry, and QASM export.
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
    GraphStateGate,
    InnerProductGate,
    IntegerComparatorGate,
    LinearFunction,
    ModularAdderGate,
    PauliEvolutionGate,
    PermutationGate,
    QFTGate,
    ZGate,
    efficient_su2,
    excitation_preserving,
    fourier_checking,
    grover_operator,
    hidden_linear_function,
    iqp,
    phase_estimation,
    qaoa_ansatz,
    quantum_volume,
    real_amplitudes,
    zz_feature_map,
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
        # Fourier
        AlgorithmEntry("qft", "fourier", build_qft, "Quantum Fourier Transform"),
        AlgorithmEntry(
            "qpe", "fourier", build_qpe,
            "Quantum Phase Estimation", min_qubits=3,
        ),
        # Search
        AlgorithmEntry("grover", "search", build_grover, "Grover's search"),
        AlgorithmEntry(
            "amplitude_estimation", "search", build_amplitude_estimation,
            "Canonical amplitude estimation", min_qubits=3,
        ),
        AlgorithmEntry(
            "hhl", "search", build_hhl,
            "HHL linear solver", min_qubits=4,
        ),
        # Oracular
        AlgorithmEntry(
            "deutsch_jozsa", "oracular", build_deutsch_jozsa,
            "Deutsch-Jozsa with balanced oracle",
        ),
        AlgorithmEntry(
            "bernstein_vazirani", "oracular", build_bernstein_vazirani,
            "Bernstein-Vazirani with alternating secret",
        ),
        # Variational
        AlgorithmEntry(
            "vqe_ansatz", "variational", build_vqe_ansatz,
            "RealAmplitudes VQE ansatz",
        ),
        AlgorithmEntry("qaoa", "variational", build_qaoa, "QAOA for MaxCut"),
        AlgorithmEntry(
            "efficient_su2", "variational", build_efficient_su2,
            "EfficientSU2 ansatz",
        ),
        AlgorithmEntry(
            "excitation_preserving", "variational", build_excitation_preserving,
            "Excitation-preserving ansatz",
        ),
        # Simulation
        AlgorithmEntry(
            "trotter", "simulation", build_trotter,
            "Suzuki-Trotter for Heisenberg chain",
        ),
        AlgorithmEntry(
            "ising_trotter", "simulation", build_ising_trotter,
            "Suzuki-Trotter for transverse-field Ising",
        ),
        # Error correction
        AlgorithmEntry(
            "steane_encoder", "error_correction", build_steane_encoder,
            "[[7,1,3]] Steane code encoder", min_qubits=7, max_qubits=7,
        ),
        AlgorithmEntry(
            "shor_encoder", "error_correction", build_shor_encoder,
            "[[9,1,3]] Shor code encoder", min_qubits=9, max_qubits=9,
        ),
        AlgorithmEntry(
            "bitflip_encoder", "error_correction", build_bitflip_encoder,
            "3-qubit bit-flip code", min_qubits=3, max_qubits=3,
        ),
        # Arithmetic
        AlgorithmEntry(
            "adder", "arithmetic", build_adder, "Modular adder", min_qubits=4,
        ),
        AlgorithmEntry(
            "comparator", "arithmetic", build_comparator,
            "Integer comparator", min_qubits=3,
        ),
        # State preparation
        AlgorithmEntry("ghz", "state_preparation", build_ghz, "GHZ state"),
        AlgorithmEntry(
            "w_state", "state_preparation", build_w_state,
            "W state", min_qubits=3,
        ),
        AlgorithmEntry(
            "graph_state", "state_preparation", build_graph_state,
            "Linear-chain graph state",
        ),
        # Data encoding
        AlgorithmEntry(
            "zz_feature_map", "data_encoding", build_zz_feature_map,
            "ZZ feature map for data encoding",
        ),
        # Benchmarking
        AlgorithmEntry(
            "quantum_volume", "benchmarking", build_quantum_volume,
            "Quantum volume circuit",
        ),
        AlgorithmEntry(
            "iqp_circuit", "benchmarking", build_iqp,
            "IQP circuit", max_qubits=12,
        ),
        AlgorithmEntry(
            "hidden_linear_function", "benchmarking",
            build_hidden_linear_function, "Hidden linear function",
        ),
        AlgorithmEntry(
            "fourier_checking", "benchmarking", build_fourier_checking,
            "Fourier checking circuit", max_qubits=10,
        ),
        # Structural
        AlgorithmEntry(
            "inner_product", "structural", build_inner_product,
            "Inner product mod 2", min_qubits=4,
        ),
        AlgorithmEntry(
            "permutation", "structural", build_permutation,
            "Cyclic permutation",
        ),
        AlgorithmEntry(
            "linear_reversible", "structural", build_linear_function,
            "Random linear reversible circuit",
        ),
    ]
    for entry in entries:
        reg.register(entry)
    return reg


# ── Algorithm builders: Fourier ─────────────────────────────────────


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


# ── Algorithm builders: Search ──────────────────────────────────────


def build_grover(n_qubits: int, n_iterations: int = 1) -> QuantumCircuit:
    """Grover's search with an oracle marking |11...1>."""
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
    n_clock = n_qubits - 2

    qpe_circ = phase_estimation(n_clock, ZGate())
    reciprocal = ExactReciprocalGate(n_clock, 0.5)

    qc = QuantumCircuit(n_qubits, name="hhl")
    qc.x(n_clock)
    qc.append(qpe_circ.to_gate(label="QPE"), range(n_clock + 1))
    qc.append(reciprocal, list(range(n_clock)) + [n_clock + 1])
    qc.append(qpe_circ.to_gate(label="QPE_inv").inverse(), range(n_clock + 1))
    return qc


# ── Algorithm builders: Oracular ────────────────────────────────────


def build_deutsch_jozsa(n_qubits: int) -> QuantumCircuit:
    """Deutsch-Jozsa algorithm with a balanced oracle."""
    qc = QuantumCircuit(n_qubits, name="deutsch_jozsa")
    qc.x(n_qubits - 1)
    qc.h(range(n_qubits))
    for i in range(min(n_qubits - 1, max(1, (n_qubits - 1) // 2))):
        qc.cx(i, n_qubits - 1)
    qc.h(range(n_qubits - 1))
    return qc


def build_bernstein_vazirani(n_qubits: int) -> QuantumCircuit:
    """Bernstein-Vazirani with an alternating secret string."""
    qc = QuantumCircuit(n_qubits, name="bernstein_vazirani")
    qc.x(n_qubits - 1)
    qc.h(range(n_qubits))
    for i in range(0, n_qubits - 1, 2):
        qc.cx(i, n_qubits - 1)
    qc.h(range(n_qubits - 1))
    return qc


# ── Algorithm builders: Variational ────────────────────────────────


def _bind_with_seed(ansatz: QuantumCircuit, seed: int = 42) -> QuantumCircuit:
    """Bind all free parameters with seeded random values."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(-np.pi, np.pi, size=len(ansatz.parameters))
    return ansatz.assign_parameters(dict(zip(ansatz.parameters, values)))


def build_vqe_ansatz(n_qubits: int, depth: int = 2) -> QuantumCircuit:
    """Hardware-efficient VQE ansatz (RealAmplitudes-style)."""
    qc = _bind_with_seed(real_amplitudes(n_qubits, reps=depth))
    qc.name = "vqe_ansatz"
    return qc


def build_qaoa(n_qubits: int, p: int = 1) -> QuantumCircuit:
    """QAOA circuit for a linear-chain MaxCut instance."""
    terms = [
        ("I" * i + "ZZ" + "I" * (n_qubits - i - 2), 1.0)
        for i in range(n_qubits - 1)
    ]
    cost_op = SparsePauliOp.from_list(terms)
    qc = _bind_with_seed(qaoa_ansatz(cost_op, reps=p))
    qc.name = "qaoa"
    return qc


def build_efficient_su2(n_qubits: int, depth: int = 2) -> QuantumCircuit:
    """EfficientSU2 ansatz with full single-qubit rotations."""
    qc = _bind_with_seed(efficient_su2(n_qubits, reps=depth))
    qc.name = "efficient_su2"
    return qc


def build_excitation_preserving(n_qubits: int, depth: int = 2) -> QuantumCircuit:
    """Excitation-preserving ansatz with RXX+RYY entanglement."""
    qc = _bind_with_seed(excitation_preserving(n_qubits, reps=depth))
    qc.name = "excitation_preserving"
    return qc


# ── Algorithm builders: Simulation ──────────────────────────────────


def build_trotter(n_qubits: int, steps: int = 2) -> QuantumCircuit:
    """Suzuki-Trotter decomposition for a Heisenberg chain."""
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


def build_ising_trotter(n_qubits: int, steps: int = 2) -> QuantumCircuit:
    """Suzuki-Trotter for a transverse-field Ising model."""
    terms = []
    for i in range(n_qubits - 1):
        label = "I" * i + "ZZ" + "I" * (n_qubits - i - 2)
        terms.append((label, 1.0))
    for i in range(n_qubits):
        label = "I" * i + "X" + "I" * (n_qubits - i - 1)
        terms.append((label, 0.5))
    hamiltonian = SparsePauliOp.from_list(terms)
    evo_gate = PauliEvolutionGate(
        hamiltonian, time=1.0, synthesis=SuzukiTrotter(order=2, reps=steps),
    )
    qc = QuantumCircuit(n_qubits, name="ising_trotter")
    qc.append(evo_gate, range(n_qubits))
    return qc


# ── Algorithm builders: Error correction ────────────────────────────


def build_steane_encoder(n_qubits: int = 7) -> QuantumCircuit:
    """Encoding circuit for the [[7,1,3]] Steane code."""
    if n_qubits != 7:
        raise ValueError("Steane code requires exactly 7 qubits")
    qc = QuantumCircuit(7, name="steane_encoder")
    qc.h(3)
    qc.h(4)
    qc.h(5)
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


def build_shor_encoder(n_qubits: int = 9) -> QuantumCircuit:
    """Encoding circuit for the [[9,1,3]] Shor code."""
    if n_qubits != 9:
        raise ValueError("Shor code requires exactly 9 qubits")
    qc = QuantumCircuit(9, name="shor_encoder")
    qc.cx(0, 3)
    qc.cx(0, 6)
    qc.h(0)
    qc.h(3)
    qc.h(6)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(3, 4)
    qc.cx(3, 5)
    qc.cx(6, 7)
    qc.cx(6, 8)
    return qc


def build_bitflip_encoder(n_qubits: int = 3) -> QuantumCircuit:
    """Encoding circuit for the 3-qubit bit-flip code."""
    if n_qubits != 3:
        raise ValueError("Bit-flip code requires exactly 3 qubits")
    qc = QuantumCircuit(3, name="bitflip_encoder")
    qc.cx(0, 1)
    qc.cx(0, 2)
    return qc


# ── Algorithm builders: Arithmetic ──────────────────────────────────


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
    value = 2 ** (num_state - 1)
    gate = IntegerComparatorGate(num_state, value)
    qc = QuantumCircuit(n_qubits, name="comparator")
    qc.append(gate, range(gate.num_qubits))
    return qc


# ── Algorithm builders: State preparation ───────────────────────────


def build_ghz(n_qubits: int) -> QuantumCircuit:
    """GHZ state preparation."""
    qc = QuantumCircuit(n_qubits, name="ghz")
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def build_w_state(n_qubits: int) -> QuantumCircuit:
    """W state preparation via controlled-rotation cascade."""
    qc = QuantumCircuit(n_qubits, name="w_state")
    qc.x(n_qubits - 1)
    for i in range(n_qubits - 1, 0, -1):
        angle = 2 * np.arccos(np.sqrt(1 / (i + 1)))
        qc.cry(angle, i, i - 1)
        qc.cx(i - 1, i)
    return qc


def build_graph_state(n_qubits: int) -> QuantumCircuit:
    """Linear-chain graph state."""
    adj = np.zeros((n_qubits, n_qubits), dtype=int)
    for i in range(n_qubits - 1):
        adj[i][i + 1] = 1
        adj[i + 1][i] = 1
    gate = GraphStateGate(adj)
    qc = QuantumCircuit(n_qubits, name="graph_state")
    qc.append(gate, range(n_qubits))
    return qc


# ── Algorithm builders: Data encoding ───────────────────────────────


def build_zz_feature_map(n_qubits: int) -> QuantumCircuit:
    """ZZ feature map for quantum data encoding."""
    qc = _bind_with_seed(zz_feature_map(n_qubits, reps=2))
    qc.name = "zz_feature_map"
    return qc


# ── Algorithm builders: Benchmarking ────────────────────────────────


def build_quantum_volume(n_qubits: int) -> QuantumCircuit:
    """Random quantum volume circuit."""
    qc = quantum_volume(n_qubits, seed=42)
    qc.name = "quantum_volume"
    return qc


def build_iqp(n_qubits: int) -> QuantumCircuit:
    """Instantaneous Quantum Polynomial circuit."""
    rng = np.random.default_rng(42)
    interactions = rng.integers(0, 8, size=(n_qubits, n_qubits))
    interactions = (interactions + interactions.T) // 2
    qc = iqp(interactions)
    qc.name = "iqp_circuit"
    return qc


def build_hidden_linear_function(n_qubits: int) -> QuantumCircuit:
    """Hidden linear function circuit from a linear-chain graph."""
    adj = np.zeros((n_qubits, n_qubits), dtype=int)
    for i in range(n_qubits - 1):
        adj[i][i + 1] = 1
        adj[i + 1][i] = 1
    qc = hidden_linear_function(adj)
    qc.name = "hidden_linear_function"
    return qc


def build_fourier_checking(n_qubits: int) -> QuantumCircuit:
    """Fourier checking circuit with random Boolean functions."""
    rng = np.random.default_rng(42)
    size = 2**n_qubits
    f = rng.choice([-1, 1], size=size).tolist()
    g = rng.choice([-1, 1], size=size).tolist()
    qc = fourier_checking(f, g)
    qc.name = "fourier_checking"
    return qc


# ── Algorithm builders: Structural ──────────────────────────────────


def build_inner_product(n_qubits: int) -> QuantumCircuit:
    """Inner product mod 2."""
    half = n_qubits // 2
    gate = InnerProductGate(half)
    qc = QuantumCircuit(n_qubits, name="inner_product")
    qc.append(gate, range(gate.num_qubits))
    return qc


def build_permutation(n_qubits: int) -> QuantumCircuit:
    """Cyclic permutation gate."""
    pattern = list(range(1, n_qubits)) + [0]
    gate = PermutationGate(pattern)
    qc = QuantumCircuit(n_qubits, name="permutation")
    qc.append(gate, range(n_qubits))
    return qc


def build_linear_function(n_qubits: int) -> QuantumCircuit:
    """Random linear reversible (CNOT-only) circuit."""
    rng = np.random.default_rng(42)
    mat = np.eye(n_qubits, dtype=int)
    for _ in range(n_qubits * 2):
        i, j = rng.integers(0, n_qubits, size=2)
        if i != j:
            mat[i] = (mat[i] + mat[j]) % 2
    gate = LinearFunction(mat)
    qc = QuantumCircuit(n_qubits, name="linear_reversible")
    qc.append(gate, range(n_qubits))
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
                logger.warning(
                    "Failed to build %s at %d qubits", key, n, exc_info=True,
                )

    return written
