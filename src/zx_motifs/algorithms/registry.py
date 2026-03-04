"""
Central registry of quantum algorithm circuit generators.
Each generator returns a Qiskit QuantumCircuit.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit


@dataclass
class AlgorithmEntry:
    name: str
    family: str  # "oracle", "variational", "transform", "entanglement", "protocol"
    generator: Callable  # (n_qubits, **kwargs) -> QuantumCircuit
    qubit_range: tuple  # (min_qubits, max_qubits)
    tags: list = field(default_factory=list)
    description: str = ""


# ── Circuit Generators ──────────────────────────────────────────────


def make_bell_state(n_qubits=2, **kwargs) -> QuantumCircuit:
    """Simplest entangling circuit — useful as a baseline."""
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)
    return qc


def make_ghz_state(n_qubits=3, **kwargs) -> QuantumCircuit:
    """GHZ state: linear chain of CNOTs after initial Hadamard."""
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def make_qft(n_qubits=4, **kwargs) -> QuantumCircuit:
    """Quantum Fourier Transform."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
        for j in range(i + 1, n_qubits):
            angle = np.pi / (2 ** (j - i))
            qc.cp(angle, j, i)
    for i in range(n_qubits // 2):
        qc.swap(i, n_qubits - i - 1)
    return qc


def make_grover(n_qubits=3, marked_state=0, n_iterations=1, **kwargs) -> QuantumCircuit:
    """Grover's algorithm with a simple oracle for |marked_state>."""
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))

    for _ in range(n_iterations):
        # Oracle: flip phase of |marked_state>
        binary = format(marked_state, f"0{n_qubits}b")
        for i, bit in enumerate(binary):
            if bit == "0":
                qc.x(i)
        if n_qubits == 2:
            qc.cz(0, 1)
        elif n_qubits >= 3:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i, bit in enumerate(binary):
            if bit == "0":
                qc.x(i)

        # Diffusion operator
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    return qc


def make_bernstein_vazirani(n_qubits=4, secret=None, **kwargs) -> QuantumCircuit:
    """Bernstein-Vazirani with a secret bitstring oracle."""
    if secret is None:
        secret = 2 ** (n_qubits - 1) - 1

    qc = QuantumCircuit(n_qubits + 1)
    qc.x(n_qubits)
    qc.h(range(n_qubits + 1))

    for i in range(n_qubits):
        if secret & (1 << i):
            qc.cx(i, n_qubits)

    qc.h(range(n_qubits))
    return qc


def make_deutsch_jozsa(n_qubits=3, balanced=True, **kwargs) -> QuantumCircuit:
    """Deutsch-Jozsa algorithm."""
    qc = QuantumCircuit(n_qubits + 1)
    qc.x(n_qubits)
    qc.h(range(n_qubits + 1))

    if balanced:
        for i in range(n_qubits):
            qc.cx(i, n_qubits)

    qc.h(range(n_qubits))
    return qc


def make_phase_estimation(
    n_qubits=4, n_counting=3, angle=np.pi / 4, **kwargs
) -> QuantumCircuit:
    """Quantum Phase Estimation for a single-qubit Z-rotation."""
    total = n_counting + 1
    qc = QuantumCircuit(total)
    qc.x(n_counting)
    qc.h(range(n_counting))

    for i in range(n_counting):
        repetitions = 2**i
        for _ in range(repetitions):
            qc.cp(angle, i, n_counting)

    # Inverse QFT on counting register
    for i in range(n_counting // 2):
        qc.swap(i, n_counting - 1 - i)
    for i in range(n_counting):
        for j in range(i):
            qc.cp(-np.pi / (2 ** (i - j)), j, i)
        qc.h(i)

    return qc


def make_qaoa_maxcut(n_qubits=4, p=1, gamma=0.5, beta=0.3, **kwargs) -> QuantumCircuit:
    """QAOA for MaxCut on a ring graph."""
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))

    edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]

    for _layer in range(p):
        for i, j in edges:
            qc.cx(i, j)
            qc.rz(gamma, j)
            qc.cx(i, j)
        for i in range(n_qubits):
            qc.rx(2 * beta, i)

    return qc


def make_vqe_uccsd_fragment(n_qubits=4, theta=0.5, **kwargs) -> QuantumCircuit:
    """A single UCCSD excitation operator (simplified double excitation)."""
    qc = QuantumCircuit(n_qubits)
    qc.x(0)
    qc.x(1)

    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.rz(theta, 3)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)

    return qc


def make_hardware_efficient_ansatz(n_qubits=4, layers=2, **kwargs) -> QuantumCircuit:
    """Hardware-efficient variational ansatz with Ry + CZ layers."""
    qc = QuantumCircuit(n_qubits)
    rng = np.random.default_rng(42)
    params = rng.uniform(0, 2 * np.pi, (layers, n_qubits))

    for layer in range(layers):
        for i in range(n_qubits):
            qc.ry(params[layer, i], i)
        for i in range(0, n_qubits - 1, 2):
            qc.cz(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            qc.cz(i, i + 1)

    return qc


def make_teleportation(n_qubits=3, **kwargs) -> QuantumCircuit:
    """Quantum teleportation circuit (canonical ZX-calculus example)."""
    qc = QuantumCircuit(3)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    return qc


# ── Registry ────────────────────────────────────────────────────────

REGISTRY = [
    AlgorithmEntry(
        "bell_state", "entanglement", make_bell_state, (2, 8),
        tags=["entanglement", "baseline"],
    ),
    AlgorithmEntry(
        "ghz_state", "entanglement", make_ghz_state, (3, 8),
        tags=["entanglement", "multipartite"],
    ),
    AlgorithmEntry(
        "teleportation", "protocol", make_teleportation, (3, 3),
        tags=["communication", "bell_measurement"],
    ),
    AlgorithmEntry(
        "qft", "transform", make_qft, (2, 7),
        tags=["phase_rotation", "butterfly"],
    ),
    AlgorithmEntry(
        "grover", "oracle", make_grover, (3, 6),
        tags=["amplitude_amplification", "diffusion", "oracle"],
    ),
    AlgorithmEntry(
        "bernstein_vazirani", "oracle", make_bernstein_vazirani, (3, 6),
        tags=["hidden_structure", "oracle"],
    ),
    AlgorithmEntry(
        "deutsch_jozsa", "oracle", make_deutsch_jozsa, (2, 6),
        tags=["decision_problem", "oracle"],
    ),
    AlgorithmEntry(
        "phase_estimation", "transform", make_phase_estimation, (4, 8),
        tags=["phase_kickback", "inverse_qft"],
    ),
    AlgorithmEntry(
        "qaoa_maxcut", "variational", make_qaoa_maxcut, (3, 8),
        tags=["combinatorial", "zz_interaction", "mixer"],
    ),
    AlgorithmEntry(
        "vqe_uccsd", "variational", make_vqe_uccsd_fragment, (4, 4),
        tags=["chemistry", "excitation"],
    ),
    AlgorithmEntry(
        "hw_efficient_ansatz", "variational", make_hardware_efficient_ansatz, (3, 8),
        tags=["hardware_efficient", "brick_layer"],
    ),
]

ALGORITHM_FAMILY_MAP = {entry.name: entry.family for entry in REGISTRY}
