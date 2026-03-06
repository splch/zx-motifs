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


# ── Novel motif-composed entanglers ────────────────────────────────────


def irr_triangle_t(n: int) -> QuantumCircuit:
    """Triangle-clique entangler inspired by iqp_triangle_2t + cluster_chain.

    Groups qubits into overlapping triangles (CX cliques) with T-gates on
    two vertices of each triangle, creating denser connectivity than the
    star hub of irr_pair11.  Adjacent triangles share one qubit, forming
    a chain of fused triangles.

    ZX motif basis: iqp_triangle_2t (3-node H-edge triangle, 2 T-like phases)
                    + cluster_chain (H-edge path for inter-triangle links).
    """
    assert n >= 4, "Need at least 4 qubits"
    qc = QuantumCircuit(n)

    # Build overlapping triangles: (i, i+1, i+2) stepping by 2
    i = 0
    while i + 2 < n:
        a, b, c = i, i + 1, i + 2
        # Triangle clique (3 CX edges)
        qc.cx(a, b)
        qc.cx(b, c)
        qc.cx(a, c)
        # T-gates on two vertices (irreducible non-Clifford)
        qc.t(a)
        qc.t(b)
        i += 2  # overlap: next triangle shares vertex c

    # If n is even, connect the last pair
    if n % 2 == 0 and n >= 4:
        qc.cx(n - 2, n - 1)
        qc.t(n - 1)

    return qc


def irr_double_star(n: int) -> QuantumCircuit:
    """Double-star entangler: two graph_state_star hubs linked by T-gate bridge.

    Creates two star-topology entanglement hubs (qubits 0 and n//2), each
    receiving CX fan-in from their local neighbourhood.  A phase gadget
    bridge (CX-T-CX) connects the two hubs, creating long-range
    entanglement with irreducible non-Clifford structure.

    ZX motif basis: graph_state_star_3 x 2 + iqp_bridge_1t (T-gate bridge).
    """
    assert n >= 4, "Need at least 4 qubits"
    qc = QuantumCircuit(n)
    mid = n // 2

    # Star 1: fan-in to qubit 0
    for i in range(1, mid):
        qc.cx(i, 0)

    # Star 2: fan-in to qubit mid
    for i in range(mid + 1, n):
        qc.cx(i, mid)

    # T-gate bridge between the two hubs
    qc.cx(0, mid)
    qc.t(mid)
    qc.cx(0, mid)

    # Extra T on hub 0 for irreducibility
    qc.t(0)

    return qc


def irr_gadget_ladder(n: int) -> QuantumCircuit:
    """Ladder of parallel phase gadgets with backbone chain.

    Places phase gadgets (CX-T-CX triplets) on alternating qubit pairs,
    connected by a nearest-neighbour backbone.  This maximises the density
    of non-Clifford (T-gate) resources while maintaining linear connectivity.

    ZX motif basis: phase_gadget_2t x parallel + cluster_chain backbone.
    """
    assert n >= 4, "Need at least 4 qubits"
    qc = QuantumCircuit(n)

    # Phase gadgets on pairs (0,1), (2,3), (4,5), ...
    for i in range(0, n - 1, 2):
        qc.cx(i, i + 1)
        qc.t(i + 1)
        qc.cx(i, i + 1)

    # Backbone chain connecting gadget pairs
    for i in range(1, n - 1, 2):
        qc.cx(i, i + 1)

    # Second layer: offset gadgets on (1,2), (3,4), ...
    for i in range(1, n - 1, 2):
        qc.cx(i, i + 1)
        qc.t(i)
        qc.cx(i, i + 1)

    return qc


def irr_iqp_dense(n: int) -> QuantumCircuit:
    """Dense IQP-inspired entangler with CZ pairs and interleaved T-gates.

    Inspired by iqp_triangle_2t and iqp_bridge_1t motifs.  Uses CZ gates
    between all pairs within distance 2, interleaved with T-gates, creating
    much denser connectivity than irr_pair11's linear structure.

    Gate count scales as ~n*2 CZ + ~n T, keeping the circuit shallow.
    """
    assert n >= 4, "Need at least 4 qubits"
    qc = QuantumCircuit(n)

    # Layer 1: nearest-neighbour CZ + T on each qubit
    for i in range(n - 1):
        qc.cz(i, i + 1)
    for i in range(n):
        qc.t(i)

    # Layer 2: next-nearest-neighbour CZ (distance-2 pairs)
    for i in range(n - 2):
        qc.cz(i, i + 2)

    return qc


def irr_sandwich_star(n: int) -> QuantumCircuit:
    """Hadamard-sandwich star with Toffoli-core spokes.

    Central qubit acts as a Clifford hub (S-gate), connected to peripheral
    qubits via CX spokes.  Each spoke terminates in a T-gate, creating
    toffoli_core-like chains.  Balances Clifford and non-Clifford resources.

    ZX motif basis: hadamard_sandwich (central clifford node)
                    + toffoli_core (T-gate chain on spokes).
    """
    assert n >= 4, "Need at least 4 qubits"
    qc = QuantumCircuit(n)

    # Central hub with S-gate (Clifford phase)
    qc.s(0)

    # Spokes: CX from hub to each peripheral qubit, T on target
    for i in range(1, n):
        qc.cx(0, i)
        qc.t(i)

    # Chain the peripheral qubits
    for i in range(1, n - 1):
        qc.cx(i, i + 1)

    # Complete the sandwich
    qc.sdg(0)

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
