"""Differential equations family: poisson_solver."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm


@register_algorithm(
    "poisson_solver", "differential_equations", (4, 7),
    tags=["linear_systems", "pde"],
)
def make_poisson_solver(n_qubits=5, **kwargs) -> QuantumCircuit:
    """HHL-based Poisson equation solver circuit.

    Implements the core of the HHL algorithm specialised for the 1-D
    Poisson equation  -d^2u/dx^2 = f.

    Layout:
        qubit 0          -- ancilla (eigenvalue inversion via controlled-RY)
        qubits 1..n_c    -- counting register (QPE)
        qubit n-1         -- system qubit (holds |b>)

    Args:
        n_qubits: Total qubit count (minimum 4, default 5).

    Tags: linear_systems, pde
    """
    n = max(4, n_qubits)
    n_counting = n - 2
    ancilla = 0
    counting = list(range(1, n_counting + 1))
    system = n - 1

    qc = QuantumCircuit(n)

    # (a) Prepare system qubit
    qc.x(system)

    # (b) QPE: Hadamard on counting register
    for c in counting:
        qc.h(c)

    # Controlled-phase rotations representing Laplacian eigenvalues
    for idx, c in enumerate(counting):
        k = idx + 1
        angle = np.pi * k / (n_counting + 1)
        for _ in range(2 ** idx):
            qc.cp(angle, c, system)

    # (c) Inverse QFT on counting register
    for i in range(n_counting // 2):
        qc.swap(counting[i], counting[n_counting - 1 - i])
    for i in range(n_counting):
        for j in range(i):
            qc.cp(-np.pi / (2 ** (i - j)), counting[j], counting[i])
        qc.h(counting[i])

    # (d) Controlled-RY on ancilla (eigenvalue inversion)
    C = 0.5
    for idx, c in enumerate(counting):
        lam_approx = 2 ** (idx + 1)  # approximate eigenvalue from counting
        theta = 2 * np.arcsin(min(C / lam_approx, 1.0))
        # Controlled-RY decomposition: RY(theta/2) - CX - RY(-theta/2) - CX
        qc.ry(theta / 2, ancilla)
        qc.cx(c, ancilla)
        qc.ry(-theta / 2, ancilla)
        qc.cx(c, ancilla)

    # (e) Inverse QPE: forward QFT then undo controlled-phases
    for i in range(n_counting - 1, -1, -1):
        qc.h(counting[i])
        for j in range(i - 1, -1, -1):
            qc.cp(np.pi / (2 ** (i - j)), counting[j], counting[i])
    for i in range(n_counting // 2):
        qc.swap(counting[i], counting[n_counting - 1 - i])

    # Undo controlled unitaries
    for idx in range(len(counting) - 1, -1, -1):
        c = counting[idx]
        k = idx + 1
        angle = -np.pi * k / (n_counting + 1)
        for _ in range(2 ** idx):
            qc.cp(angle, c, system)

    # Hadamard to uncompute counting register
    for c in counting:
        qc.h(c)

    return qc
