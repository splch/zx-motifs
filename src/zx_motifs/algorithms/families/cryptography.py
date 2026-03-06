"""Cryptography family: bb84_encode, e91_protocol."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm
from zx_motifs.algorithms._helpers import bell_pair


@register_algorithm(
    "bb84_encode", "cryptography", (2, 16),
    tags=["qkd", "prepare_measure"],
)
def make_bb84_encode(n_qubits=8, seed=42, **kwargs) -> QuantumCircuit:
    """BB84 QKD state preparation circuit.

    For each qubit, Alice randomly chooses:
        - A bit value (0 or 1)
        - A basis (Z-basis or X-basis)

    Encoding:
        - bit=1: apply X gate (flip to |1>)
        - X-basis: apply H gate (rotate to |+>/|-> basis)

    Uses a fixed-seed RNG for reproducibility.
    """
    seed = kwargs.get("seed", seed)
    n = max(1, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(seed)

    bits = rng.integers(0, 2, size=n)    # random bit values
    bases = rng.integers(0, 2, size=n)   # 0 = Z-basis, 1 = X-basis

    for i in range(n):
        if bits[i] == 1:
            qc.x(i)          # encode bit value
        if bases[i] == 1:
            qc.h(i)          # switch to X-basis

    return qc


@register_algorithm(
    "e91_protocol", "cryptography", (2, 16),
    tags=["qkd", "entanglement_based"],
)
def make_e91_protocol(n_qubits=8, seed=99, **kwargs) -> QuantumCircuit:
    """E91 entanglement-based QKD protocol circuit.

    Creates n_qubits//2 Bell pairs, then applies random measurement basis
    rotations (RY) on each qubit to simulate Alice and Bob independently
    choosing measurement bases.
    """
    seed = kwargs.get("seed", seed)
    n = max(2, n_qubits)
    n_pairs = n // 2
    total = 2 * n_pairs  # ensure even number of qubits
    qc = QuantumCircuit(total)
    rng = np.random.default_rng(seed)

    # Create Bell pairs: (0,1), (2,3), (4,5), ...
    for p in range(n_pairs):
        q_a = 2 * p
        q_b = 2 * p + 1
        bell_pair(qc, q_a, q_b)

    # Random measurement basis rotations
    alice_angles = [0.0, np.pi / 4, np.pi / 2]
    bob_angles = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    for p in range(n_pairs):
        q_a = 2 * p
        q_b = 2 * p + 1
        qc.ry(rng.choice(alice_angles), q_a)
        qc.ry(rng.choice(bob_angles), q_b)

    return qc
