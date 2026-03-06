"""Linear algebra family: hhl, vqls."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm


@register_algorithm(
    "hhl", "linear_algebra", (4, 7),
    tags=["linear_systems", "phase_estimation"],
)
def make_hhl(n_qubits=5, **kwargs) -> QuantumCircuit:
    """HHL algorithm core circuit for solving linear systems Ax = b.

    Qubit layout (n_qubits total):
        - qubit 0: ancilla (eigenvalue inversion via controlled rotation)
        - qubits 1 .. n_counting: counting register (QPE)
        - qubit n_qubits-1: system qubit (holds |b>)
    """
    n = max(4, n_qubits)
    n_counting = n - 2  # counting register size
    ancilla = 0
    counting = list(range(1, n_counting + 1))
    system = n - 1

    qc = QuantumCircuit(n)

    # (a) Prepare system qubit in eigenstate |1>
    qc.x(system)

    # (b) QPE: Hadamard on counting register
    for c in counting:
        qc.h(c)

    # Controlled unitary: e^{i A t} approximated by controlled-phase gates
    for idx, c in enumerate(counting):
        angle = np.pi / (2 ** idx)
        for _ in range(2 ** idx):
            qc.cp(angle, c, system)

    # Inverse QFT on counting register
    for i in range(n_counting // 2):
        qc.swap(counting[i], counting[n_counting - 1 - i])
    for i in range(n_counting):
        for j in range(i):
            qc.cp(-np.pi / (2 ** (i - j)), counting[j], counting[i])
        qc.h(counting[i])

    # (c) Controlled rotations on ancilla (eigenvalue inversion)
    C = 0.5  # scaling constant
    for idx, c in enumerate(counting):
        theta = 2 * np.arcsin(C / (2 ** (idx + 1)))
        # Decompose controlled-RY: RY(theta/2) - CX - RY(-theta/2) - CX
        qc.ry(theta / 2, ancilla)
        qc.cx(c, ancilla)
        qc.ry(-theta / 2, ancilla)
        qc.cx(c, ancilla)

    # (d) Inverse QPE: undo phase estimation
    # Forward QFT on counting register
    for i in range(n_counting - 1, -1, -1):
        qc.h(counting[i])
        for j in range(i - 1, -1, -1):
            qc.cp(np.pi / (2 ** (i - j)), counting[j], counting[i])
    for i in range(n_counting // 2):
        qc.swap(counting[i], counting[n_counting - 1 - i])

    # Inverse controlled unitaries
    for idx in range(len(counting) - 1, -1, -1):
        c = counting[idx]
        angle = -np.pi / (2 ** idx)
        for _ in range(2 ** idx):
            qc.cp(angle, c, system)

    # Hadamard on counting register to complete uncomputation
    for c in counting:
        qc.h(c)

    return qc


@register_algorithm(
    "vqls", "linear_algebra", (2, 8),
    tags=["linear_systems", "variational"],
)
def make_vqls(n_qubits=4, layers=2, **kwargs) -> QuantumCircuit:
    """Variational Quantum Linear Solver (VQLS) ansatz circuit.

    Prepares a parameterized trial state |x(theta)> using a hardware-efficient
    ansatz with RY + RZ single-qubit rotations and CX entangling layers.
    """
    layers = kwargs.get("layers", layers)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(789)

    for _layer in range(layers):
        # Single-qubit variational rotations
        for i in range(n):
            qc.ry(rng.uniform(0, 2 * np.pi), i)
            qc.rz(rng.uniform(0, 2 * np.pi), i)
        # Entangling layer: CX on adjacent pairs
        for i in range(n - 1):
            qc.cx(i, i + 1)

    # Final rotation layer for expressibility
    for i in range(n):
        qc.ry(rng.uniform(0, 2 * np.pi), i)

    return qc
