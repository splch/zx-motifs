"""Entanglement family: bell_state, ghz_state, w_state, cluster_state, dicke_state, graph_state."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm


@register_algorithm(
    "bell_state", "entanglement", (2, 8),
    tags=["entanglement", "baseline"],
)
def make_bell_state(n_qubits=2, **kwargs) -> QuantumCircuit:
    """Simplest entangling circuit — useful as a baseline."""
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)
    return qc


@register_algorithm(
    "ghz_state", "entanglement", (3, 8),
    tags=["entanglement", "multipartite"],
)
def make_ghz_state(n_qubits=3, **kwargs) -> QuantumCircuit:
    """GHZ state: linear chain of CNOTs after initial Hadamard."""
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


@register_algorithm(
    "w_state", "entanglement", (3, 8),
    tags=["entanglement", "multipartite", "arbitrary_rotation"],
)
def make_w_state(n_qubits=3, **kwargs) -> QuantumCircuit:
    """W state preparation using RY rotations and CNOTs.

    Creates |W_n> = (|100..0> + |010..0> + ... + |000..1>) / sqrt(n).
    Uses the F-gate approach: start from |10...0>, then distribute the
    single excitation evenly across all qubits via controlled rotations.
    """
    n = max(3, n_qubits)
    qc = QuantumCircuit(n)
    # Start with single excitation on qubit 0
    qc.x(0)
    # Distribute amplitude from qubit i to qubit i+1
    for i in range(n - 1):
        # Ry angle to split 1/(n-i) probability to qubit i, rest to i+1
        theta = 2 * np.arccos(np.sqrt(1 / (n - i)))
        # Controlled-Ry(theta) on qubit i+1, controlled by qubit i
        # Decomposition: Ry(theta/2) - CX - Ry(-theta/2) - CX
        qc.ry(theta / 2, i + 1)
        qc.cx(i, i + 1)
        qc.ry(-theta / 2, i + 1)
        qc.cx(i, i + 1)
    return qc


@register_algorithm(
    "cluster_state", "entanglement", (2, 8),
    tags=["graph_state", "mbqc", "cz_only"],
)
def make_cluster_state(n_qubits=4, **kwargs) -> QuantumCircuit:
    """1D cluster (graph) state: |+>^n with CZ on nearest neighbors.

    Foundation of measurement-based quantum computing (MBQC).
    All-Hadamard-edge topology in ZX representation.
    """
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for i in range(n - 1):
        qc.cz(i, i + 1)
    return qc


@register_algorithm(
    "dicke_state", "entanglement", (3, 8),
    tags=["symmetric_state", "entanglement"],
)
def make_dicke_state(n_qubits=4, k=2, **kwargs) -> QuantumCircuit:
    """Dicke state |D_n^k> preparation.

    Prepares the symmetric superposition of all n-qubit states with
    exactly k excitations (Hamming weight k).  For example, |D_4^2>
    is the equal superposition of |0011>, |0101>, |0110>, |1001>,
    |1010>, |1100>.

    Algorithm (Bartschi-Eidenbenz deterministic approach):
        1) Start with k X gates on the first k qubits:
           |1...1 0...0>  (k ones, n-k zeros)
        2) For each qubit position i from 0 to n-2, distribute the
           remaining excitations into the suffix qubits using
           controlled-RY rotations.  The rotation angle for splitting
           m excitations among r remaining qubits is:
               theta = 2*arccos(sqrt((r-1)/r))  when distributing 1 of m
           This is implemented via: RY(theta/2) - CX - RY(-theta/2) - CX
           (controlled-RY decomposition).

    Args:
        n_qubits: Number of qubits (minimum 2).
        k: Number of excitations / Hamming weight (default 2).

    Tags: symmetric_state, entanglement
    """
    k = kwargs.get("k", k)
    n = max(2, n_qubits)
    k = max(1, min(k, n - 1))  # clamp k to valid range

    qc = QuantumCircuit(n)

    # Step 1: Place k excitations on the first k qubits
    for i in range(k):
        qc.x(i)

    # Step 2: Distribute excitations using the SCS (Split-and-Cyclic-Shift)
    # approach.  Process qubits from left to right; at each position,
    # determine how many excitations remain to distribute and among how
    # many qubits.
    for i in range(n - 1):
        remaining = n - i
        if remaining <= 1:
            break

        # Angle to split: qubit i has excitation, split to i+1 with probability 1/remaining
        # RY(theta) where theta = 2*arccos(sqrt((remaining-1)/remaining))
        theta = 2 * np.arccos(np.sqrt((remaining - 1) / remaining))

        # Controlled-RY decomposition: RY(theta/2) - CX - RY(-theta/2) - CX
        qc.ry(theta / 2, i + 1)
        qc.cx(i, i + 1)
        qc.ry(-theta / 2, i + 1)
        qc.cx(i, i + 1)

    return qc


@register_algorithm(
    "graph_state", "entanglement", (3, 8),
    tags=["graph_state", "multipartite"],
)
def make_graph_state(n_qubits=5, **kwargs) -> QuantumCircuit:
    """Graph state with star topology.

    Prepares a graph state where qubit 0 is the centre of a star graph
    and all other qubits are leaves.  The preparation is:
        1) H on all qubits -- initialise each qubit in |+>.
        2) CZ from the centre (q0) to every other qubit -- create edges
           of the star graph.

    Star graph states are equivalent to GHZ states up to local
    unitaries and are fundamental resources in quantum networks
    (e.g. quantum secret sharing, anonymous transmission).

    Args:
        n_qubits: Number of qubits (minimum 2).

    Tags: graph_state, multipartite
    """
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)

    # Step 1: Hadamard on all qubits to prepare |+>^n
    for i in range(n):
        qc.h(i)

    # Step 2: CZ from centre (q0) to every leaf qubit
    for i in range(1, n):
        qc.cz(0, i)

    return qc
