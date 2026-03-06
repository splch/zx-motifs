"""Error mitigation family: zne_folding, pauli_twirling."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm


@register_algorithm(
    "zne_folding", "error_mitigation", (2, 8),
    tags=["noise_mitigation", "folding"],
)
def make_zne_folding(n_qubits=4, **kwargs) -> QuantumCircuit:
    """Zero-noise extrapolation via global unitary folding.

    Builds a simple base circuit (H + CX chain), appends its inverse
    (adjoint in reverse order), then appends the original again.

    Tags: noise_mitigation, folding
    """
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)

    # Base circuit: H on q0, then CX chain
    def _append_base(circuit):
        circuit.h(0)
        for i in range(n - 1):
            circuit.cx(i, i + 1)

    # Inverse of base circuit (adjoint, reversed gate order)
    def _append_base_inverse(circuit):
        for i in range(n - 2, -1, -1):
            circuit.cx(i, i + 1)       # CX is self-adjoint
        circuit.h(0)                    # H  is self-adjoint

    # fold-1: U
    _append_base(qc)
    # fold-2: U-dagger
    _append_base_inverse(qc)
    # fold-3: U
    _append_base(qc)

    return qc


@register_algorithm(
    "pauli_twirling", "error_mitigation", (2, 8),
    tags=["noise_mitigation", "randomized_compiling"],
)
def make_pauli_twirling(n_qubits=4, seed=33, **kwargs) -> QuantumCircuit:
    """Pauli twirling of a CNOT layer.

    For each adjacent CNOT, randomly chosen Pauli gates are inserted
    before and the appropriate conjugate Paulis after, so that the
    net unitary is identical to the bare CNOT layer.

    Tags: noise_mitigation, randomized_compiling
    """
    seed = kwargs.get("seed", seed)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(seed)

    # Pauli twirl table for CNOT
    _twirl_table = {
        (0, 0): (0, 0), (0, 1): (0, 1), (0, 2): (3, 2), (0, 3): (3, 3),
        (1, 0): (1, 1), (1, 1): (1, 0), (1, 2): (2, 3), (1, 3): (2, 2),
        (2, 0): (2, 1), (2, 1): (2, 0), (2, 2): (1, 3), (2, 3): (1, 2),
        (3, 0): (3, 0), (3, 1): (3, 1), (3, 2): (0, 2), (3, 3): (0, 3),
    }

    def _apply_pauli(circuit, qubit, pauli_id):
        if pauli_id == 1:
            circuit.x(qubit)
        elif pauli_id == 2:
            circuit.y(qubit)
        elif pauli_id == 3:
            circuit.z(qubit)
        # 0 = identity, do nothing

    for i in range(n - 1):
        # Random before-Paulis
        bc = int(rng.integers(4))
        bt = int(rng.integers(4))
        ac, at = _twirl_table[(bc, bt)]

        _apply_pauli(qc, i, bc)
        _apply_pauli(qc, i + 1, bt)
        qc.cx(i, i + 1)
        _apply_pauli(qc, i, ac)
        _apply_pauli(qc, i + 1, at)

    return qc
