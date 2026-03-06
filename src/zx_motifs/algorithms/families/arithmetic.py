"""Arithmetic family: ripple_carry_adder, qft_adder, quantum_multiplier, quantum_comparator."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm
from zx_motifs.algorithms._helpers import decompose_toffoli


@register_algorithm(
    "ripple_carry_adder", "arithmetic", (5, 5),
    tags=["toffoli", "addition", "classical_reversible"],
)
def make_ripple_carry_adder(n_qubits=5, **kwargs) -> QuantumCircuit:
    """Cuccaro-style ripple-carry adder for 2-bit addition.

    Uses 5 qubits: c0 (carry_in=0), a0, b0, a1, b1.
    After execution: b0 = sum bit 0, b1 = sum bit 1, a1 = carry_out.
    MAJ(c,a,b) = CX(c,b), CX(c,a), Toffoli(a,b,c).
    UMA(c,a,b) = Toffoli(a,b,c), CX(c,a), CX(a,b).
    Toffoli gates decomposed into Clifford+T.
    """
    qc = QuantumCircuit(5)
    c0, a0, b0, a1, b1 = 0, 1, 2, 3, 4
    # MAJ(c0, a0, b0): propagate carry through bit 0
    qc.cx(c0, b0)
    qc.cx(c0, a0)
    decompose_toffoli(qc, a0, b0, c0)
    # MAJ(c0, a1, b1): propagate carry through bit 1
    qc.cx(c0, b1)
    qc.cx(c0, a1)
    decompose_toffoli(qc, a1, b1, c0)
    # UMA(c0, a1, b1): uncompute and add bit 1
    decompose_toffoli(qc, a1, b1, c0)
    qc.cx(c0, a1)
    qc.cx(a1, b1)
    # UMA(c0, a0, b0): uncompute and add bit 0
    decompose_toffoli(qc, a0, b0, c0)
    qc.cx(c0, a0)
    qc.cx(a0, b0)
    return qc


@register_algorithm(
    "qft_adder", "arithmetic", (4, 8),
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
    "quantum_multiplier", "arithmetic", (6, 6),
    tags=["multiplication", "classical_reversible"],
)
def make_quantum_multiplier(n_qubits=6, **kwargs) -> QuantumCircuit:
    """Schoolbook quantum multiplier for 2-bit x 2-bit unsigned integers.

    Qubit layout (minimum 6):
        q0, q1 -- input register A  (2-bit multiplicand)
        q2, q3 -- input register B  (2-bit multiplier)
        q4, q5 -- output / accumulator P

    Tags: multiplication, classical_reversible
    """
    qc = QuantumCircuit(max(6, n_qubits))
    a0, a1 = 0, 1        # multiplicand A
    b0, b1 = 2, 3        # multiplier B
    p0, p1 = 4, 5        # product / accumulator

    # Partial product a0*b0 -> p0
    decompose_toffoli(qc, a0, b0, p0)

    # Partial product a0*b1 -> p1
    decompose_toffoli(qc, a0, b1, p1)

    # Partial product a1*b0 -> p1 (shifted by 1)
    decompose_toffoli(qc, a1, b0, p1)

    # Cross term a1*b1 carry propagation
    decompose_toffoli(qc, a1, b1, p0)

    # Carry propagation
    qc.cx(p0, p1)
    qc.cx(p1, p0)

    return qc


@register_algorithm(
    "quantum_comparator", "arithmetic", (5, 5),
    tags=["comparison", "classical_reversible"],
)
def make_quantum_comparator(n_qubits=5, **kwargs) -> QuantumCircuit:
    """Quantum less-than comparator for two 2-bit unsigned integers.

    Qubit layout (minimum 5):
        q0, q1 -- input register A  (2-bit, A = 2*a1 + a0)
        q2, q3 -- input register B  (2-bit, B = 2*b1 + b0)
        q4     -- result qubit (set to |1> iff A < B)

    Tags: comparison, classical_reversible
    """
    qc = QuantumCircuit(max(5, n_qubits))
    a0, a1 = 0, 1        # A register
    b0, b1 = 2, 3        # B register
    result = 4            # output: 1 iff A < B

    # Bit-0 borrow: borrow if a0=0 AND b0=1
    qc.x(a0)
    decompose_toffoli(qc, a0, b0, result)
    qc.x(a0)

    # Propagate borrow into bit-1 comparison
    qc.cx(result, a1)
    qc.cx(result, b1)

    # Bit-1 borrow: borrow if (adjusted) a1=0 AND b1=1
    qc.x(a1)
    decompose_toffoli(qc, a1, b1, result)
    qc.x(a1)

    # Uncompute the borrow propagation into a1, b1
    qc.cx(result, b1)
    qc.cx(result, a1)

    return qc
