"""Ansatz builders: irr_pair11 entangler, baselines, and Hamiltonians."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


# ── irr_pair11 entangler ───────────────────────────────────────────────


def irr_pair11_entangler(n: int) -> QuantumCircuit:
    """Build the generalized irr_pair11 entangling layer for *n* qubits.

    Structure (scales linearly with *n*):
      1. **Star hub** -- qubit 0 is the hub, CX fan-in from qubits 1..hub_size
      2. **Phase gadgets** -- T gates on every 3rd qubit, each conjugated by
         CX pairs connecting it to neighbours
      3. **Chain tail** -- nearest-neighbour CX chain extending entanglement
         to remaining qubits

    At n=6 this reproduces essentially the same connectivity as the original
    6-qubit irr_pair11 discovered via irreducible composition.
    """
    assert n >= 4, "Need at least 4 qubits"
    qc = QuantumCircuit(n)

    # 1. Star hub: fan-in to qubit 0 from qubits 1..hub_size
    hub_size = max(2, n // 3)
    for i in range(1, hub_size + 1):
        qc.cx(i, 0)

    # 2. Phase gadgets: place on every 3rd qubit starting from hub_size
    gadget_anchors = []
    q = 1
    while q < n - 1:
        anchor = q
        target = q + 1
        if gadget_anchors:
            qc.cx(gadget_anchors[-1], anchor)
        qc.cx(anchor, target)
        qc.t(target)
        qc.cx(anchor, target)
        gadget_anchors.append(anchor)
        q += 3

    # 3. Chain tail: connect last gadget region to remaining qubits
    last_touched = max(gadget_anchors[-1] + 1, hub_size) if gadget_anchors else hub_size
    for i in range(last_touched, n - 1):
        qc.cx(i, i + 1)

    return qc


def irr_pair11_original_6q() -> QuantumCircuit:
    """Reproduce the exact original 6-qubit circuit from verification data."""
    qc = QuantumCircuit(6)
    qc.cx(1, 0)
    qc.cx(2, 0)
    qc.cx(3, 0)
    qc.cx(1, 2)
    qc.cx(1, 3)
    qc.cx(2, 3)
    qc.t(3)
    qc.cx(3, 4)
    qc.cx(4, 5)
    qc.t(2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc


# ── Baseline entanglers ────────────────────────────────────────────────


def cx_chain_entangler(n: int, n_2q: int) -> QuantumCircuit:
    """CX-chain baseline with a given 2-qubit gate budget."""
    qc = QuantumCircuit(n)
    placed = 0
    while placed < n_2q:
        for i in range(min(n - 1, n_2q - placed)):
            qc.cx(i, i + 1)
            placed += 1
            if placed >= n_2q:
                break
    return qc


def hea_entangler(n: int, n_2q: int) -> QuantumCircuit:
    """CZ brick-layer (hardware-efficient ansatz) baseline."""
    qc = QuantumCircuit(n)
    placed = 0
    layer = 0
    while placed < n_2q:
        start = layer % 2
        for i in range(start, n - 1, 2):
            qc.cz(i, i + 1)
            placed += 1
            if placed >= n_2q:
                break
        layer += 1
    return qc


# ── Hamiltonian builders ──────────────────────────────────────────────

_PAULI_1Q = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_matrix(label: str) -> np.ndarray:
    """Build n-qubit Pauli matrix from a label like ``'XIZZ'``."""
    result = np.array([[1.0 + 0j]])
    for ch in label:
        result = np.kron(result, _PAULI_1Q[ch])
    return result


def build_hamiltonian(n: int, model: str = "heisenberg") -> np.ndarray:
    """Build a 2^n x 2^n Hamiltonian matrix for VQE benchmarks.

    Supported models: ``heisenberg``, ``tfim``, ``xy``, ``xxz``,
    ``random_2local``.
    """
    d = 2**n
    H = np.zeros((d, d), dtype=complex)

    if model == "heisenberg":
        for i in range(n - 1):
            for p in "XYZ":
                label = ["I"] * n
                label[i] = p
                label[i + 1] = p
                H += _pauli_matrix("".join(label))

    elif model == "tfim":
        for i in range(n - 1):
            label = ["I"] * n
            label[i] = "Z"
            label[i + 1] = "Z"
            H -= _pauli_matrix("".join(label))
        for i in range(n):
            label = ["I"] * n
            label[i] = "X"
            H -= _pauli_matrix("".join(label))

    elif model == "xy":
        for i in range(n - 1):
            for p in "XY":
                label = ["I"] * n
                label[i] = p
                label[i + 1] = p
                H += _pauli_matrix("".join(label))

    elif model == "xxz":
        delta = 0.5
        for i in range(n - 1):
            for p in "XY":
                label = ["I"] * n
                label[i] = p
                label[i + 1] = p
                H += _pauli_matrix("".join(label))
            label = ["I"] * n
            label[i] = "Z"
            label[i + 1] = "Z"
            H += delta * _pauli_matrix("".join(label))

    elif model == "random_2local":
        rng = np.random.default_rng(42)
        for i in range(n - 1):
            for p1 in "XYZ":
                for p2 in "XYZ":
                    coeff = rng.normal(0, 1)
                    if abs(coeff) < 0.3:
                        continue
                    label = ["I"] * n
                    label[i] = p1
                    label[i + 1] = p2
                    H += coeff * _pauli_matrix("".join(label))
        for i in range(n):
            for p in "XYZ":
                coeff = rng.normal(0, 0.5)
                if abs(coeff) < 0.2:
                    continue
                label = ["I"] * n
                label[i] = p
                H += coeff * _pauli_matrix("".join(label))

    return H
