"""Transform family: qft, phase_estimation, iterative_qpe, amplitude_estimation."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm


@register_algorithm(
    "qft", "transform", (2, 7),
    tags=["phase_rotation", "butterfly"],
)
def make_qft(n_qubits=4, **kwargs) -> QuantumCircuit:
    """Quantum Fourier Transform."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
        for j in range(i + 1, n_qubits):
            angle = np.pi / (2 ** (j - i))
            qc.cp(angle, j, i)
    for i in range(n_qubits // 2):
        qc.swap(i, n_qubits - i - 1)
    return qc


@register_algorithm(
    "phase_estimation", "transform", (4, 8),
    tags=["phase_kickback", "inverse_qft"],
)
def make_phase_estimation(
    n_qubits=4, n_counting=3, angle=np.pi / 4, **kwargs
) -> QuantumCircuit:
    """Quantum Phase Estimation for a single-qubit Z-rotation."""
    total = n_counting + 1
    qc = QuantumCircuit(total)
    qc.x(n_counting)
    qc.h(range(n_counting))

    for i in range(n_counting):
        repetitions = 2**i
        for _ in range(repetitions):
            qc.cp(angle, i, n_counting)

    # Inverse QFT on counting register
    for i in range(n_counting // 2):
        qc.swap(i, n_counting - 1 - i)
    for i in range(n_counting):
        for j in range(i):
            qc.cp(-np.pi / (2 ** (i - j)), j, i)
        qc.h(i)

    return qc


@register_algorithm(
    "iterative_qpe", "transform", (2, 2),
    tags=["phase_estimation", "iterative"],
)
def make_iterative_qpe(n_qubits=3, n_bits=3, angle=np.pi / 4,
                       **kwargs) -> QuantumCircuit:
    """Single-ancilla iterative phase estimation.

    Estimates the phase of a unitary (Z-rotation by `angle`) one bit
    at a time, using only 1 ancilla qubit + 1 system qubit.

    Args:
        n_qubits: Accepted but circuit always uses 2 qubits
                  (1 ancilla + 1 system).
        n_bits: Number of phase bits to extract (default 3).
        angle: Phase angle of the unitary to estimate (default pi/4).

    Tags: phase_estimation, iterative
    """
    n_bits = kwargs.get("n_bits", n_bits)
    angle = kwargs.get("angle", angle)
    qc = QuantumCircuit(2)
    ancilla = 0
    system = 1

    # Prepare system qubit in eigenstate |1> of the Z-rotation
    qc.x(system)

    # Track extracted phase bits for correction
    extracted_phases = []

    for k in range(n_bits - 1, -1, -1):
        # Step 1: H on ancilla
        qc.h(ancilla)

        # Step 2: Controlled-U^{2^k}
        reps = 2 ** k
        for _ in range(reps):
            qc.cp(angle, ancilla, system)

        # Step 3: Phase correction from previously extracted bits
        correction = 0.0
        for idx, phase_bit in enumerate(extracted_phases):
            correction += phase_bit / (2 ** (idx + 1))
        if correction != 0.0:
            qc.rz(-2 * np.pi * correction, ancilla)

        # Step 4: H on ancilla
        qc.h(ancilla)

        # Step 5: "Reset" ancilla
        extracted_phases.append(0.5)  # assume average for reproducibility
        qc.x(ancilla)

    return qc


@register_algorithm(
    "amplitude_estimation", "transform", (4, 8),
    tags=["amplitude_estimation", "grover_qpe"],
)
def make_amplitude_estimation(n_qubits=5, **kwargs) -> QuantumCircuit:
    """Canonical quantum amplitude estimation (Grover operator + QPE).

    Combines Grover's search operator Q with quantum phase estimation
    to estimate the amplitude a = sin^2(theta) of a marked state.

    Args:
        n_qubits: Total qubit count (minimum 4).

    Tags: amplitude_estimation, grover_qpe
    """
    n = max(4, n_qubits)
    n_counting = max(2, n // 2)
    n_state = n - n_counting

    if n_state < 2:
        n_state = 2
        n_counting = n - n_state

    total = n_counting + n_state
    qc = QuantumCircuit(total)

    counting = list(range(n_counting))
    state = list(range(n_counting, total))

    # (a) Initialize state register in uniform superposition
    for s in state:
        qc.h(s)

    # (b) H on counting register
    for c in counting:
        qc.h(c)

    # (c) Controlled-Grover iterations: Q^{2^k} controlled by counting[k]
    for k in range(n_counting):
        reps = 2 ** k
        for _ in range(reps):
            # Oracle S_chi: flip phase of "marked" state
            qc.cz(counting[k], state[-1])

            # Diffusion S_0 (controlled): H-X-CZ-X-H on state register
            for s in state:
                qc.h(s)
                qc.x(s)
            # Multi-controlled Z approximated by pairwise CZ chain
            qc.cz(counting[k], state[0])
            for i in range(len(state) - 1):
                qc.cz(state[i], state[i + 1])
            for s in state:
                qc.x(s)
                qc.h(s)

    # (d) Inverse QFT on counting register
    for i in range(n_counting // 2):
        qc.swap(counting[i], counting[n_counting - 1 - i])
    for i in range(n_counting):
        for j in range(i):
            qc.cp(-np.pi / (2 ** (i - j)), counting[j], counting[i])
        qc.h(counting[i])

    return qc
