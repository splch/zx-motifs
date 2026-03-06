"""Sampling family: iqp_sampling, random_circuit_sampling."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm


@register_algorithm(
    "iqp_sampling", "sampling", (2, 8),
    tags=["commuting_gates", "supremacy"],
)
def make_iqp_sampling(n_qubits=5, seed=77, **kwargs) -> QuantumCircuit:
    """IQP (Instantaneous Quantum Polynomial) sampling circuit.

    Structure: H^n -> D(diagonal) -> H^n

    The diagonal layer consists of commuting gates (all diagonal in the
    Z-basis): CZ gates between selected pairs and T gates on individual
    qubits.

    Tags: commuting_gates, supremacy
    """
    seed = kwargs.get("seed", seed)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(seed)

    # Layer 1: Hadamard on all qubits
    qc.h(range(n))

    # Layer 2: Diagonal gates (all commute, all diagonal in Z-basis)
    # CZ gates on random pairs
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                qc.cz(i, j)
    # T gates on each qubit (with random inclusion)
    for i in range(n):
        if rng.random() < 0.6:
            qc.t(i)

    # Layer 3: Hadamard on all qubits
    qc.h(range(n))

    return qc


@register_algorithm(
    "random_circuit_sampling", "sampling", (2, 8),
    tags=["random_circuit", "supremacy"],
)
def make_random_circuit_sampling(n_qubits=5, depth=6, seed=55,
                                 **kwargs) -> QuantumCircuit:
    """Random circuit sampling with Clifford+T gates.

    Alternating layers of:
        - Random single-qubit gates chosen from {H, S, T} on each qubit
        - CX gates on adjacent qubit pairs (even-odd / odd-even alternating)

    Tags: random_circuit, supremacy
    """
    depth = kwargs.get("depth", depth)
    seed = kwargs.get("seed", seed)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(seed)

    single_qubit_gates = ["h", "s", "t"]

    for layer in range(depth):
        # Single-qubit Clifford+T layer
        for i in range(n):
            gate = rng.choice(single_qubit_gates)
            if gate == "h":
                qc.h(i)
            elif gate == "s":
                qc.s(i)
            else:
                qc.t(i)

        # Entangling layer: CX on adjacent pairs
        start = layer % 2
        for i in range(start, n - 1, 2):
            qc.cx(i, i + 1)

    return qc
