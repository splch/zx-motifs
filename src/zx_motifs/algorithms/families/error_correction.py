"""Error correction family: bit_flip_code, phase_flip_code, steane_code, shor_code,
five_qubit_code, surface_code_patch, color_code, bacon_shor, reed_muller_code."""
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm
from zx_motifs.algorithms._helpers import decompose_toffoli


@register_algorithm(
    "bit_flip_code", "error_correction", (5, 5),
    tags=["repetition_code", "syndrome_extraction", "toffoli"],
)
def make_bit_flip_code(n_qubits=5, **kwargs) -> QuantumCircuit:
    """3-qubit bit-flip code: encode + syndrome extraction + correction.

    Qubits: 0-2 = data (logical encoded), 3-4 = syndrome ancillas.
    """
    qc = QuantumCircuit(5)
    # Encode: |psi> -> |psi psi psi>
    qc.cx(0, 1)
    qc.cx(0, 2)
    # Syndrome extraction
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.cx(1, 4)
    qc.cx(2, 4)
    # Correction using Toffoli (decomposed): syndrome (1,1) = error on q1
    decompose_toffoli(qc, 3, 4, 1)
    return qc


@register_algorithm(
    "phase_flip_code", "error_correction", (5, 5),
    tags=["repetition_code", "hadamard_basis", "toffoli"],
)
def make_phase_flip_code(n_qubits=5, **kwargs) -> QuantumCircuit:
    """3-qubit phase-flip code: Hadamard-wrapped repetition code.

    Same structure as bit-flip but with Hadamard layers to detect phase errors.
    """
    qc = QuantumCircuit(5)
    # Encode in Hadamard basis
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    # Syndrome extraction
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.cx(1, 4)
    qc.cx(2, 4)
    # Correction: syndrome (1,1) = error on q1
    decompose_toffoli(qc, 3, 4, 1)
    # Decode back
    qc.h(0)
    qc.h(1)
    qc.h(2)
    return qc


@register_algorithm(
    "steane_code", "error_correction", (7, 7),
    tags=["css_code", "hamming", "stabilizer"],
)
def make_steane_code(n_qubits=7, **kwargs) -> QuantumCircuit:
    """[[7,1,3]] Steane code encoder.

    Encodes logical |0> into 7 physical qubits using the CSS construction
    from the classical [7,4,3] Hamming code.
    """
    qc = QuantumCircuit(7)
    # Prepare logical |0> encoded state
    qc.h(0)
    qc.h(1)
    qc.h(2)
    # Steane code parity checks (CX connectivity from Hamming code)
    qc.cx(0, 3)
    qc.cx(0, 4)
    qc.cx(0, 5)
    qc.cx(1, 3)
    qc.cx(1, 4)
    qc.cx(1, 6)
    qc.cx(2, 3)
    qc.cx(2, 5)
    qc.cx(2, 6)
    return qc


@register_algorithm(
    "shor_code", "error_correction", (9, 9),
    tags=["concatenated_code", "nine_qubit"],
)
def make_shor_code(n_qubits=9, **kwargs) -> QuantumCircuit:
    """Shor's [[9,1,3]] code: concatenation of phase-flip inside bit-flip.

    9 physical qubits encode 1 logical qubit.
    """
    qc = QuantumCircuit(9)
    # Phase-flip encoding: spread across 3 blocks
    qc.cx(0, 3)
    qc.cx(0, 6)
    qc.h(0)
    qc.h(3)
    qc.h(6)
    # Bit-flip encoding within each block
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(3, 4)
    qc.cx(3, 5)
    qc.cx(6, 7)
    qc.cx(6, 8)
    return qc


@register_algorithm(
    "five_qubit_code", "error_correction", (5, 5),
    tags=["perfect_code", "stabilizer"],
)
def make_five_qubit_code(n_qubits=5, **kwargs) -> QuantumCircuit:
    """[[5,1,3]] perfect code encoder.

    The smallest quantum error-correcting code that corrects arbitrary
    single-qubit errors.  5 physical qubits encode 1 logical qubit.
    """
    qc = QuantumCircuit(5)
    # Qubit 4 holds the logical input; qubits 0-3 are ancillas.
    # Step 1: Hadamard on ancilla qubits to create superposition
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    # Step 2: CZ gates encoding Z-components of stabilizer generators
    qc.cz(0, 1)
    qc.cz(0, 4)
    qc.cz(1, 2)
    qc.cz(2, 3)
    qc.cz(3, 4)
    # Step 3: CX gates from ancillas to logical qubit encode X-components
    qc.cx(0, 4)
    qc.cx(1, 4)
    qc.cx(2, 4)
    qc.cx(3, 4)
    return qc


@register_algorithm(
    "surface_code_patch", "error_correction", (9, 9),
    tags=["topological_code", "surface_code", "stabilizer"],
)
def make_surface_code_patch(n_qubits=9, **kwargs) -> QuantumCircuit:
    """Minimal rotated surface code patch (distance-2).

    Layout (rotated surface code d=2):
        4 data qubits (0-3) arranged in a 2x2 grid
        4 stabilizer ancilla qubits (4-7)
        1 extra ancilla (qubit 8) for additional boundary stabilizer
    """
    qc = QuantumCircuit(9)

    # X-stabilizer measurement
    # X-stab 1 (ancilla 4): measures X0*X1*X2*X3 (weight-4 bulk plaquette)
    qc.h(4)
    qc.cx(4, 0)
    qc.cx(4, 1)
    qc.cx(4, 2)
    qc.cx(4, 3)
    qc.h(4)

    # X-stab 2 (ancilla 5): measures X0*X1 (weight-2 boundary)
    qc.h(5)
    qc.cx(5, 0)
    qc.cx(5, 1)
    qc.h(5)

    # Z-stabilizer measurement
    # Z-stab 1 (ancilla 6): measures Z0*Z2 (weight-2 boundary)
    qc.cx(0, 6)
    qc.cx(2, 6)

    # Z-stab 2 (ancilla 7): measures Z1*Z3 (weight-2 boundary)
    qc.cx(1, 7)
    qc.cx(3, 7)

    # Additional boundary check (ancilla 8)
    # Z-check on bottom boundary: Z2*Z3
    qc.cx(2, 8)
    qc.cx(3, 8)

    return qc


@register_algorithm(
    "color_code", "error_correction", (7, 7),
    tags=["topological_code", "color_code"],
)
def make_color_code(n_qubits=7, **kwargs) -> QuantumCircuit:
    """[[7,1,3]] color code encoder (triangular / Steane-like).

    The 7-qubit color code is defined on a triangular lattice with
    three-colorable faces.

    Args:
        n_qubits: Ignored (always 7 physical qubits).

    Tags: topological_code, color_code
    """
    qc = QuantumCircuit(7)

    # Step 1: Hadamard on the first 3 qubits (generators of the code)
    qc.h(0)
    qc.h(1)
    qc.h(2)

    # Step 2: CX pattern from Hamming code parity-check structure
    qc.cx(0, 3)
    qc.cx(0, 4)
    qc.cx(0, 5)
    qc.cx(1, 3)
    qc.cx(1, 4)
    qc.cx(1, 6)
    qc.cx(2, 3)
    qc.cx(2, 5)
    qc.cx(2, 6)

    # Step 3: CZ gates emphasising the 3-colorable lattice structure
    qc.cz(0, 3)
    qc.cz(1, 3)
    qc.cz(2, 6)
    qc.cz(0, 4)
    qc.cz(1, 5)
    qc.cz(4, 5)

    return qc


@register_algorithm(
    "bacon_shor", "error_correction", (9, 9),
    tags=["subsystem_code", "stabilizer"],
)
def make_bacon_shor(n_qubits=9, **kwargs) -> QuantumCircuit:
    """[[9,1,3]] Bacon-Shor subsystem code encoder.

    9 qubits arranged in a 3x3 grid.

    Grid layout:
        q0  q1  q2
        q3  q4  q5
        q6  q7  q8

    Args:
        n_qubits: Ignored (always 9 physical qubits).

    Tags: subsystem_code, stabilizer
    """
    qc = QuantumCircuit(9)

    # Row encoding: CX from row leader to row members
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(3, 4)
    qc.cx(3, 5)
    qc.cx(6, 7)
    qc.cx(6, 8)

    # Column encoding: spread phase information down columns
    qc.h(0)
    qc.h(1)
    qc.h(2)

    # CX from column leaders to column members
    qc.cx(0, 3)
    qc.cx(0, 6)
    qc.cx(1, 4)
    qc.cx(1, 7)
    qc.cx(2, 5)
    qc.cx(2, 8)

    return qc


@register_algorithm(
    "reed_muller_code", "error_correction", (15, 15),
    tags=["reed_muller", "transversal_t"],
)
def make_reed_muller_code(n_qubits=15, **kwargs) -> QuantumCircuit:
    """[[15,1,3]] quantum Reed-Muller code encoder.

    15 physical qubits encoding 1 logical qubit.  This code is a CSS
    code constructed from the classical punctured Reed-Muller codes
    RM(1,4) and RM(2,4).  Its key property is supporting a transversal
    T gate, making it valuable for fault-tolerant quantum computation.

    Args:
        n_qubits: Ignored (always 15 physical qubits).

    Tags: reed_muller, transversal_t
    """
    qc = QuantumCircuit(15)

    # Step 1: H on the first 4 qubits (RM(1,4) generator qubits)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)

    # Step 2: CX pattern from RM(1,4) generator matrix
    for data_qubit in range(4, 15):
        pos = data_qubit + 1  # 1-indexed position
        for gen in range(4):
            if pos & (1 << gen):
                qc.cx(gen, data_qubit)

    return qc
