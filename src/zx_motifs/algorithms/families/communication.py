"""Communication family: quantum_fingerprinting."""
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm
from zx_motifs.algorithms._helpers import decompose_toffoli


@register_algorithm(
    "quantum_fingerprinting", "communication", (4, 8),
    tags=["communication", "exponential_saving"],
)
def make_quantum_fingerprinting(n_qubits=6, **kwargs) -> QuantumCircuit:
    """Quantum fingerprinting for equality testing.

    Two parties (Alice and Bob) each encode their classical input into
    quantum fingerprint states.  A referee then performs a controlled-SWAP
    test to determine whether the two inputs are equal.

    Layout:
        qubit 0                       -- ancilla (referee's control qubit)
        qubits 1 .. n_a              -- Alice's register
        qubits n_a+1 .. n-1          -- Bob's register

    Args:
        n_qubits: Total qubit count (minimum 5, default 6).

    Tags: communication, exponential_saving
    """
    n = max(5, n_qubits)
    qc = QuantumCircuit(n)

    ancilla = 0
    # Split remaining qubits between Alice and Bob
    remaining = n - 1
    n_alice = remaining // 2
    n_bob = remaining - n_alice

    alice_qubits = list(range(1, 1 + n_alice))
    bob_qubits = list(range(1 + n_alice, n))

    # Step 1: Alice's fingerprint encoding
    for q in alice_qubits:
        qc.h(q)
    # CX chain within Alice's register (error-correcting structure)
    for i in range(len(alice_qubits) - 1):
        qc.cx(alice_qubits[i], alice_qubits[i + 1])

    # Step 2: Bob's fingerprint encoding
    for q in bob_qubits:
        qc.h(q)
    # CX chain within Bob's register
    for i in range(len(bob_qubits) - 1):
        qc.cx(bob_qubits[i], bob_qubits[i + 1])

    # Step 3: Referee's controlled-SWAP test
    qc.h(ancilla)

    # Controlled-SWAP between paired Alice/Bob qubits
    # Fredkin decomposition: CX(b, a), Toffoli(ancilla, a, b), CX(b, a)
    n_pairs = min(n_alice, n_bob)
    for i in range(n_pairs):
        a_q = alice_qubits[i]
        b_q = bob_qubits[i]
        qc.cx(b_q, a_q)
        decompose_toffoli(qc, ancilla, a_q, b_q)
        qc.cx(b_q, a_q)

    qc.h(ancilla)

    return qc
