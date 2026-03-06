"""Protocol family: teleportation, superdense_coding, entanglement_swapping, swap_test."""
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm
from zx_motifs.algorithms._helpers import decompose_toffoli


@register_algorithm(
    "teleportation", "protocol", (3, 3),
    tags=["communication", "bell_measurement"],
)
def make_teleportation(n_qubits=3, **kwargs) -> QuantumCircuit:
    """Quantum teleportation circuit (canonical ZX-calculus example)."""
    qc = QuantumCircuit(3)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    return qc


@register_algorithm(
    "superdense_coding", "protocol", (2, 2),
    tags=["communication", "bell_pair", "dense_coding"],
)
def make_superdense_coding(n_qubits=2, **kwargs) -> QuantumCircuit:
    """Superdense coding: send 2 classical bits via 1 qubit."""
    qc = QuantumCircuit(2)
    # Create Bell pair
    qc.h(0)
    qc.cx(0, 1)
    # Encode: apply X and Z to send bits "11"
    qc.x(0)
    qc.z(0)
    # Decode: Bell measurement
    qc.cx(0, 1)
    qc.h(0)
    return qc


@register_algorithm(
    "entanglement_swapping", "protocol", (4, 4),
    tags=["bell_measurement", "relay", "teleportation_variant"],
)
def make_entanglement_swapping(n_qubits=4, **kwargs) -> QuantumCircuit:
    """Entanglement swapping: extend entanglement via Bell measurement.

    4 qubits: (0,1) form Bell pair, (2,3) form Bell pair.
    Bell measurement on (1,2) entangles (0,3).
    """
    qc = QuantumCircuit(4)
    # Create two Bell pairs: (0,1) and (2,3)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(2)
    qc.cx(2, 3)
    # Bell measurement on (1,2)
    qc.cx(1, 2)
    qc.h(1)
    return qc


@register_algorithm(
    "swap_test", "protocol", (3, 7),
    tags=["fidelity", "comparison"],
)
def make_swap_test(n_qubits=3, **kwargs) -> QuantumCircuit:
    """SWAP test for quantum state fidelity estimation.

    Estimates |<psi|phi>|^2 by interfering two states through a
    controlled-SWAP operation.  Measuring the ancilla qubit in the
    |0> state occurs with probability (1 + |<psi|phi>|^2) / 2.

    Qubit layout:
        q0             -- ancilla (control)
        q1 .. q_mid    -- first state register  |psi>
        q_{mid+1} .. q_{n-1} -- second state register |phi>

    For odd n_qubits the registers are split as evenly as possible
    (the first register gets the extra qubit and the controlled-SWAP
    pairs as many qubits as the smaller register allows).

    Circuit structure:
        1. H on ancilla
        2. Controlled-SWAP between paired qubits of the two registers
           (decomposed as CX-Toffoli-CX, the standard Fredkin
            decomposition)
        3. H on ancilla

    Tags: fidelity, comparison
    """
    n = max(3, n_qubits)
    qc = QuantumCircuit(n)

    ancilla = 0
    # Split remaining qubits into two registers
    remaining = n - 1
    reg_size = remaining // 2  # size of the smaller register

    first_start = 1
    second_start = 1 + (remaining - reg_size)

    # Step 1: Hadamard on ancilla
    qc.h(ancilla)

    # Step 2: Controlled-SWAP for each pair
    for i in range(reg_size):
        t1 = first_start + i
        t2 = second_start + i
        # Fredkin decomposition: CX(t2, t1), Toffoli(ancilla, t1, t2), CX(t2, t1)
        qc.cx(t2, t1)
        decompose_toffoli(qc, ancilla, t1, t2)
        qc.cx(t2, t1)

    # Step 3: Hadamard on ancilla
    qc.h(ancilla)

    return qc
