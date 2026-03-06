"""Arithmetic family: ripple_carry_adder, qft_adder, quantum_multiplier, quantum_comparator."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm
from zx_motifs.algorithms._helpers import decompose_toffoli


@register_algorithm(
    "ripple_carry_adder", "arithmetic", (5, None),
    tags=["toffoli", "addition", "classical_reversible"],
)
def make_ripple_carry_adder(n_qubits=5, **kwargs) -> QuantumCircuit:
    """Cuccaro-style ripple-carry adder for n-bit addition.

    Uses 2n+1 qubits: 1 carry qubit (c), n a-qubits, n b-qubits,
    where n = (n_qubits - 1) // 2.  n_qubits must be odd and >= 5.
    MAJ(c,a,b) = CX(c,b), CX(c,a), Toffoli(a,b,c).
    UMA(c,a,b) = Toffoli(a,b,c), CX(c,a), CX(a,b).
    Toffoli gates decomposed into Clifford+T.
    """
    n_qubits = max(5, n_qubits)
    # Force odd qubit count (2n+1 layout)
    if n_qubits % 2 == 0:
        n_qubits -= 1
    n_bits = (n_qubits - 1) // 2  # number of bits per operand

    qc = QuantumCircuit(n_qubits)
    c = 0  # carry qubit
    a = list(range(1, n_bits + 1))              # a register
    b = list(range(n_bits + 1, 2 * n_bits + 1)) # b register

    # Forward pass: MAJ chain to propagate carries
    for i in range(n_bits):
        qc.cx(c, b[i])
        qc.cx(c, a[i])
        decompose_toffoli(qc, a[i], b[i], c)

    # Reverse pass: UMA chain to uncompute carries and produce sums
    for i in range(n_bits - 1, -1, -1):
        decompose_toffoli(qc, a[i], b[i], c)
        qc.cx(c, a[i])
        qc.cx(a[i], b[i])

    return qc


@register_algorithm(
    "qft_adder", "arithmetic", (4, None),
    tags=["addition", "controlled_phase", "qft_based"],
)
def make_qft_adder(n_qubits=4, **kwargs) -> QuantumCircuit:
    """Draper's QFT-based addition: QFT -> controlled-phase -> QFT-dagger.

    Uses n_qubits split into two n/2-qubit registers.
    """
    n = max(4, n_qubits)
    half = n // 2
    qc = QuantumCircuit(n)

    a_reg = list(range(half))
    b_reg = list(range(half, n))

    # QFT on b register
    for i in range(half):
        qc.h(b_reg[i])
        for j in range(i + 1, half):
            angle = np.pi / (2 ** (j - i))
            qc.cp(angle, b_reg[j], b_reg[i])

    # Controlled-phase addition
    for i in range(half):
        for j in range(half):
            if i + j < half:
                angle = np.pi / (2 ** (j))
                qc.cp(angle, a_reg[i], b_reg[i + j])

    # Inverse QFT on b register
    for i in range(half - 1, -1, -1):
        for j in range(half - 1, i, -1):
            angle = -np.pi / (2 ** (j - i))
            qc.cp(angle, b_reg[j], b_reg[i])
        qc.h(b_reg[i])

    return qc


@register_algorithm(
    "quantum_multiplier", "arithmetic", (8, None),
    tags=["multiplication", "classical_reversible"],
)
def make_quantum_multiplier(n_qubits=8, **kwargs) -> QuantumCircuit:
    """Schoolbook quantum multiplier for k-bit x k-bit unsigned integers.

    Uses 4k qubits: k a-qubits (multiplicand), k b-qubits (multiplier),
    2k p-qubits (product accumulator), where k = n_qubits // 4.
    Partial products are accumulated via Toffoli gates (decomposed into
    Clifford+T), followed by CX carry propagation on the product register.
    """
    n_qubits = max(8, n_qubits)
    k = n_qubits // 4
    total = 4 * k

    qc = QuantumCircuit(total)
    a = list(range(k))                     # multiplicand A
    b = list(range(k, 2 * k))              # multiplier B
    p = list(range(2 * k, 4 * k))          # product / accumulator P

    # Accumulate partial products: a[i] * b[j] -> p[i+j]
    for i in range(k):
        for j in range(k):
            if i + j < 2 * k:
                decompose_toffoli(qc, a[i], b[j], p[i + j])

    # Carry propagation on product register
    for i in range(2 * k - 1):
        qc.cx(p[i], p[i + 1])
    for i in range(2 * k - 2, -1, -1):
        qc.cx(p[i + 1], p[i])

    return qc


@register_algorithm(
    "quantum_comparator", "arithmetic", (5, None),
    tags=["comparison", "classical_reversible"],
)
def make_quantum_comparator(n_qubits=5, **kwargs) -> QuantumCircuit:
    """Quantum less-than comparator for two n-bit unsigned integers.

    Uses 2n+1 qubits: n a-qubits, n b-qubits, 1 result qubit,
    where n = (n_qubits - 1) // 2.  n_qubits must be odd and >= 5.
    Result qubit is set to |1> iff A < B.
    Toffoli gates decomposed into Clifford+T.
    """
    n_qubits = max(5, n_qubits)
    # Force odd qubit count (2n+1 layout)
    if n_qubits % 2 == 0:
        n_qubits -= 1
    n_bits = (n_qubits - 1) // 2  # number of bits per operand

    qc = QuantumCircuit(n_qubits)
    a = list(range(n_bits))                     # A register
    b = list(range(n_bits, 2 * n_bits))         # B register
    result = 2 * n_bits                         # output qubit

    for i in range(n_bits):
        # Borrow at bit i: borrow if a[i]=0 AND b[i]=1
        qc.x(a[i])
        decompose_toffoli(qc, a[i], b[i], result)
        qc.x(a[i])

        # Propagate borrow into next bit's comparison (except last)
        if i < n_bits - 1:
            qc.cx(result, a[i + 1])
            qc.cx(result, b[i + 1])

    # Uncompute borrow propagation in reverse (leave result intact)
    for i in range(n_bits - 2, -1, -1):
        qc.cx(result, b[i + 1])
        qc.cx(result, a[i + 1])

    return qc
