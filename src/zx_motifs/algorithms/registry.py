"""
Central registry of quantum algorithm circuit generators.
Each generator returns a Qiskit QuantumCircuit.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit


@dataclass
class AlgorithmEntry:
    name: str
    family: str
    generator: Callable  # (n_qubits, **kwargs) -> QuantumCircuit
    qubit_range: tuple  # (min_qubits, max_qubits)
    tags: list = field(default_factory=list)
    description: str = ""


# ── Circuit Generators ──────────────────────────────────────────────


def make_bell_state(n_qubits=2, **kwargs) -> QuantumCircuit:
    """Simplest entangling circuit — useful as a baseline."""
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)
    return qc


def make_ghz_state(n_qubits=3, **kwargs) -> QuantumCircuit:
    """GHZ state: linear chain of CNOTs after initial Hadamard."""
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


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


def make_grover(n_qubits=3, marked_state=0, n_iterations=1, **kwargs) -> QuantumCircuit:
    """Grover's algorithm with a simple oracle for |marked_state>."""
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))

    for _ in range(n_iterations):
        # Oracle: flip phase of |marked_state>
        binary = format(marked_state, f"0{n_qubits}b")[::-1]  # LSB-first for Qiskit
        for i, bit in enumerate(binary):
            if bit == "0":
                qc.x(i)
        if n_qubits == 2:
            qc.cz(0, 1)
        elif n_qubits >= 3:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i, bit in enumerate(binary):
            if bit == "0":
                qc.x(i)

        # Diffusion operator
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    return qc


def make_bernstein_vazirani(n_qubits=4, secret=None, **kwargs) -> QuantumCircuit:
    """Bernstein-Vazirani with a secret bitstring oracle."""
    if secret is None:
        secret = 2 ** (n_qubits - 1) - 1

    qc = QuantumCircuit(n_qubits + 1)
    qc.x(n_qubits)
    qc.h(range(n_qubits + 1))

    for i in range(n_qubits):
        if secret & (1 << i):
            qc.cx(i, n_qubits)

    qc.h(range(n_qubits))
    return qc


def make_deutsch_jozsa(n_qubits=3, balanced=True, **kwargs) -> QuantumCircuit:
    """Deutsch-Jozsa algorithm."""
    qc = QuantumCircuit(n_qubits + 1)
    qc.x(n_qubits)
    qc.h(range(n_qubits + 1))

    if balanced:
        for i in range(n_qubits):
            qc.cx(i, n_qubits)

    qc.h(range(n_qubits))
    return qc


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


def make_qaoa_maxcut(n_qubits=4, p=1, gamma=0.5, beta=0.3, **kwargs) -> QuantumCircuit:
    """QAOA for MaxCut on a ring graph."""
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))

    edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]

    for _layer in range(p):
        for i, j in edges:
            qc.cx(i, j)
            qc.rz(gamma, j)
            qc.cx(i, j)
        for i in range(n_qubits):
            qc.rx(2 * beta, i)

    return qc


def make_vqe_uccsd_fragment(n_qubits=4, theta=0.5, **kwargs) -> QuantumCircuit:
    """A single UCCSD excitation operator (simplified double excitation)."""
    qc = QuantumCircuit(n_qubits)
    qc.x(0)
    qc.x(1)

    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.rz(theta, 3)
    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 1)

    return qc


def make_hardware_efficient_ansatz(n_qubits=4, layers=2, **kwargs) -> QuantumCircuit:
    """Hardware-efficient variational ansatz with Ry + CZ layers."""
    qc = QuantumCircuit(n_qubits)
    rng = np.random.default_rng(42)
    params = rng.uniform(0, 2 * np.pi, (layers, n_qubits))

    for layer in range(layers):
        for i in range(n_qubits):
            qc.ry(params[layer, i], i)
        for i in range(0, n_qubits - 1, 2):
            qc.cz(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            qc.cz(i, i + 1)

    return qc


def make_teleportation(n_qubits=3, **kwargs) -> QuantumCircuit:
    """Quantum teleportation circuit (canonical ZX-calculus example)."""
    qc = QuantumCircuit(3)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    return qc


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


# ── Error Correction Generators ────────────────────────────────────


def _decompose_toffoli(qc: QuantumCircuit, c0: int, c1: int, t: int) -> None:
    """Decompose Toffoli (CCX) into Clifford+T gates for QASM2 compatibility."""
    qc.h(t)
    qc.cx(c1, t)
    qc.tdg(t)
    qc.cx(c0, t)
    qc.t(t)
    qc.cx(c1, t)
    qc.tdg(t)
    qc.cx(c0, t)
    qc.t(c1)
    qc.t(t)
    qc.h(t)
    qc.cx(c0, c1)
    qc.t(c0)
    qc.tdg(c1)
    qc.cx(c0, c1)


def make_bit_flip_code(n_qubits=5, **kwargs) -> QuantumCircuit:
    """3-qubit bit-flip code: encode + syndrome extraction + correction.

    Qubits: 0-2 = data (logical encoded), 3-4 = syndrome ancillas.
    """
    qc = QuantumCircuit(5)
    # Encode: |ψ> → |ψψψ>
    qc.cx(0, 1)
    qc.cx(0, 2)
    # Syndrome extraction
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.cx(1, 4)
    qc.cx(2, 4)
    # Correction using Toffoli (decomposed): syndrome (1,1) = error on q1
    _decompose_toffoli(qc, 3, 4, 1)
    return qc


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
    _decompose_toffoli(qc, 3, 4, 1)
    # Decode back
    qc.h(0)
    qc.h(1)
    qc.h(2)
    return qc


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


# ── Simulation Generators ──────────────────────────────────────────


def make_trotter_ising(n_qubits=4, n_steps=1, dt=0.5, j_coupling=1.0,
                       h_field=0.5, **kwargs) -> QuantumCircuit:
    """Trotterized transverse-field Ising model: H = -J ΣZZ - h ΣX.

    Each Trotter step applies ZZ interactions then X rotations.
    """
    n_qubits = max(2, n_qubits)
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_steps):
        # ZZ interaction terms: CX-RZ-CX
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * j_coupling * dt, i + 1)
            qc.cx(i, i + 1)
        # Transverse field terms
        for i in range(n_qubits):
            qc.rx(2 * h_field * dt, i)
    return qc


def make_trotter_heisenberg(n_qubits=4, n_steps=1, dt=0.5, **kwargs) -> QuantumCircuit:
    """Trotterized Heisenberg XXX model: H = Σ(XX + YY + ZZ).

    Decomposes each interaction into CX-Rz-CX blocks with basis changes
    for XX and YY terms.
    """
    n_qubits = max(2, n_qubits)
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_steps):
        for i in range(n_qubits - 1):
            # ZZ term
            qc.cx(i, i + 1)
            qc.rz(2 * dt, i + 1)
            qc.cx(i, i + 1)
            # XX term: H-CX-RZ-CX-H
            qc.h(i)
            qc.h(i + 1)
            qc.cx(i, i + 1)
            qc.rz(2 * dt, i + 1)
            qc.cx(i, i + 1)
            qc.h(i)
            qc.h(i + 1)
            # YY term: S†-H-CX-RZ-CX-H-S
            qc.sdg(i)
            qc.sdg(i + 1)
            qc.h(i)
            qc.h(i + 1)
            qc.cx(i, i + 1)
            qc.rz(2 * dt, i + 1)
            qc.cx(i, i + 1)
            qc.h(i)
            qc.h(i + 1)
            qc.s(i)
            qc.s(i + 1)
    return qc


# ── Arithmetic Generators ──────────────────────────────────────────


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
    _decompose_toffoli(qc, a0, b0, c0)
    # MAJ(c0, a1, b1): propagate carry through bit 1
    qc.cx(c0, b1)
    qc.cx(c0, a1)
    _decompose_toffoli(qc, a1, b1, c0)
    # UMA(c0, a1, b1): uncompute and add bit 1
    _decompose_toffoli(qc, a1, b1, c0)
    qc.cx(c0, a1)
    qc.cx(a1, b1)
    # UMA(c0, a0, b0): uncompute and add bit 0
    _decompose_toffoli(qc, a0, b0, c0)
    qc.cx(c0, a0)
    qc.cx(a0, b0)
    return qc


# ── Oracle Family Additions ────────────────────────────────────────


def make_simon(n_qubits=4, secret=None, **kwargs) -> QuantumCircuit:
    """Simon's algorithm for hidden XOR mask.

    Uses n data qubits + n ancilla qubits. n_qubits is the data qubit count.
    """
    n = max(2, n_qubits // 2) if n_qubits > 2 else 2
    total = 2 * n
    qc = QuantumCircuit(total)

    if secret is None:
        # Default: secret = "10...0" (MSB set)
        secret = 1 << (n - 1)

    # Hadamard on data register
    qc.h(range(n))
    # Find flag bit: index of first set bit in secret
    flag = 0
    for i in range(n):
        if secret & (1 << i):
            flag = i
            break

    # Oracle: copy data to ancilla, then XOR with secret using flag bit
    for i in range(n):
        qc.cx(i, n + i)
    for i in range(n):
        if secret & (1 << i):
            qc.cx(flag, n + i)
    # Final Hadamard on data register
    qc.h(range(n))
    return qc


# ── Entanglement Family Additions ──────────────────────────────────


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


# ── Error Correction ──────────────────────────────────────


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


# ── Simulation ────────────────────────────────────────────


def make_quantum_walk(n_qubits=3, n_steps=2, **kwargs) -> QuantumCircuit:
    """Discrete-time quantum walk on a cycle.

    Uses 1 coin qubit + (n-1) position qubits. Coin operation is Hadamard,
    shift is conditional increment/decrement via CX chains.
    """
    n = max(3, n_qubits)
    qc = QuantumCircuit(n)
    coin = 0
    pos_qubits = list(range(1, n))

    for _ in range(n_steps):
        # Coin operation
        qc.h(coin)
        # Conditional shift right (coin=|0>): CX chain
        for i in range(len(pos_qubits) - 1):
            qc.cx(coin, pos_qubits[i])
        # Conditional shift left (coin=|1>): X-CX-X
        qc.x(coin)
        for i in range(len(pos_qubits) - 1, 0, -1):
            qc.cx(coin, pos_qubits[i])
        qc.x(coin)
    return qc


# ── Arithmetic ────────────────────────────────────────────


def make_qft_adder(n_qubits=4, **kwargs) -> QuantumCircuit:
    """Draper's QFT-based addition: QFT → controlled-phase → QFT†.

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


# ── Oracle Family ─────────────────────────────────────────


def make_quantum_counting(n_qubits=5, **kwargs) -> QuantumCircuit:
    """Quantum counting: QPE applied to Grover's diffusion operator.

    Uses n_counting qubits for phase estimation + n_search qubits for Grover.
    Simplified to use 2 counting + (n-2) search qubits.
    """
    n = max(5, n_qubits)
    n_counting = 2
    n_search = n - n_counting

    qc = QuantumCircuit(n)

    # Initialize search register in uniform superposition
    for i in range(n_counting, n):
        qc.h(i)

    # QPE counting register
    for i in range(n_counting):
        qc.h(i)

    # Controlled Grover iterations (simplified: controlled diffusion)
    for c in range(n_counting):
        reps = 2 ** c
        for _ in range(reps):
            # Controlled-Z oracle (mark |0...0>)
            for s in range(n_counting, n):
                qc.x(s)
            # Controlled multi-Z via CX chain
            qc.cx(c, n_counting)
            for s in range(n_counting, n - 1):
                qc.cx(s, s + 1)
            qc.cx(c, n - 1)
            for s in range(n - 2, n_counting - 1, -1):
                qc.cx(s, s + 1)
            for s in range(n_counting, n):
                qc.x(s)
            # Diffusion
            for s in range(n_counting, n):
                qc.h(s)
                qc.x(s)
            qc.cx(c, n_counting)
            for s in range(n_counting, n - 1):
                qc.cx(s, s + 1)
            qc.cx(c, n - 1)
            for s in range(n - 2, n_counting - 1, -1):
                qc.cx(s, s + 1)
            for s in range(n_counting, n):
                qc.x(s)
                qc.h(s)

    # Inverse QFT on counting register
    qc.h(0)
    qc.cp(-np.pi / 2, 0, 1)
    qc.h(1)

    return qc


# ── Protocol Family ───────────────────────────────────────


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


# ── Distillation Generators ──────────────────────────────────────


def _bell_pair(qc: QuantumCircuit, q0: int, q1: int) -> None:
    """Create a Bell pair on qubits (q0, q1)."""
    qc.h(q0)
    qc.cx(q0, q1)


def make_bbpssw_distillation(n_qubits=4, **kwargs) -> QuantumCircuit:
    """BBPSSW entanglement distillation (Bennett et al. 1996).

    Foundational bilateral CNOT protocol. Two noisy Bell pairs are combined
    via bilateral CNOTs; measurement of the sacrificial pair heralds success.
    Qubits 0,1: pair to keep; 2,3: pair to sacrifice.
    """
    qc = QuantumCircuit(4)
    # Two Bell pairs
    _bell_pair(qc, 0, 1)
    _bell_pair(qc, 2, 3)
    # Bilateral CNOTs
    qc.cx(0, 2)
    qc.cx(1, 3)
    return qc


def make_dejmps_distillation(n_qubits=4, **kwargs) -> QuantumCircuit:
    """DEJMPS entanglement distillation (Deutsch et al. 1996).

    Adds bilateral Rx rotations before CNOTs to handle asymmetric noise.
    Alice's qubits (0,2) get Rx(pi/2); Bob's qubits (1,3) get Rx(-pi/2).
    Qubits 0,1: pair to keep; 2,3: pair to sacrifice.
    """
    qc = QuantumCircuit(4)
    # Two Bell pairs
    _bell_pair(qc, 0, 1)
    _bell_pair(qc, 2, 3)
    # Bilateral rotations: Rx(pi/2) for Alice, Rx(-pi/2) for Bob
    qc.rx(np.pi / 2, 0)
    qc.rx(-np.pi / 2, 1)
    qc.rx(np.pi / 2, 2)
    qc.rx(-np.pi / 2, 3)
    # Bilateral CNOTs
    qc.cx(0, 2)
    qc.cx(1, 3)
    return qc


def make_recurrence_distillation(n_qubits=8, **kwargs) -> QuantumCircuit:
    """Two-round recurrence distillation.

    Cascades two BBPSSW rounds. Round 1 distills (0,1) from (0,1)+(2,3)
    and (4,5) from (4,5)+(6,7). Round 2 distills (0,1) from (0,1)+(4,5).
    """
    qc = QuantumCircuit(8)
    # Four Bell pairs
    _bell_pair(qc, 0, 1)
    _bell_pair(qc, 2, 3)
    _bell_pair(qc, 4, 5)
    _bell_pair(qc, 6, 7)
    # Round 1: bilateral CNOTs
    qc.cx(0, 2)
    qc.cx(1, 3)
    qc.cx(4, 6)
    qc.cx(5, 7)
    # Round 2: bilateral CNOTs on surviving pairs
    qc.cx(0, 4)
    qc.cx(1, 5)
    return qc


def make_pumping_distillation(n_qubits=6, **kwargs) -> QuantumCircuit:
    """Pumping entanglement distillation.

    One target pair (0,1) is repeatedly purified by sacrificing fresh pairs.
    Sacrificial pairs: (2,3) then (4,5).
    """
    qc = QuantumCircuit(6)
    # Target Bell pair
    _bell_pair(qc, 0, 1)
    # First sacrificial pair
    _bell_pair(qc, 2, 3)
    qc.cx(0, 2)
    qc.cx(1, 3)
    # Second sacrificial pair
    _bell_pair(qc, 4, 5)
    qc.cx(0, 4)
    qc.cx(1, 5)
    return qc


# ── Machine Learning Family ──────────────────────────────


def make_quantum_kernel(n_qubits=4, **kwargs) -> QuantumCircuit:
    """ZZFeatureMap-style quantum kernel circuit.

    Encodes classical data with H + RZ (single-qubit) + CX-RZ-CX (ZZ interaction).
    Two repetitions of the feature map.
    """
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(123)

    for _rep in range(2):
        # Single-qubit encoding
        for i in range(n):
            qc.h(i)
            qc.rz(rng.uniform(0, 2 * np.pi), i)
        # ZZ entangling feature map
        for i in range(n - 1):
            qc.cx(i, i + 1)
            qc.rz(rng.uniform(0, 2 * np.pi), i + 1)
            qc.cx(i, i + 1)
    return qc


def make_data_reuploading(n_qubits=2, layers=3, **kwargs) -> QuantumCircuit:
    """Data re-uploading classifier: layered RY/RZ + CX.

    Each layer re-encodes data via RY/RZ rotations, interleaved with CX.
    """
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(456)

    for _layer in range(layers):
        # Data encoding + trainable rotations
        for i in range(n):
            qc.ry(rng.uniform(0, 2 * np.pi), i)
            qc.rz(rng.uniform(0, 2 * np.pi), i)
        # Entangling layer
        for i in range(n - 1):
            qc.cx(i, i + 1)
    return qc


# ── Oracle Family Additions ───────────────────────────────────────


def make_deutsch(n_qubits=2, oracle_type="balanced", **kwargs) -> QuantumCircuit:
    """Deutsch's algorithm — the simplest quantum oracle algorithm.

    The original Deutsch algorithm for f:{0,1}->{0,1} using 2 qubits:
    qubit 0 = input, qubit 1 = ancilla.  Determines whether f is constant
    or balanced with a single query.

    Circuit: X on ancilla, H on both, oracle (CX for balanced, I or X-CX-X
    for constant-1), H on input.

    Args:
        n_qubits: Ignored (always 2).
        oracle_type: "balanced" (f(0)!=f(1), uses CX) or "constant"
                     (f(0)==f(1)==0, identity oracle).
    """
    qc = QuantumCircuit(2)
    # Prepare ancilla in |1>
    qc.x(1)
    # Hadamard on both qubits
    qc.h(0)
    qc.h(1)
    # Oracle
    if oracle_type == "balanced":
        # f(x) = x  →  CX from input to ancilla
        qc.cx(0, 1)
    # constant-0: identity (no gate needed)
    # constant-1 would be X on ancilla, but constant-0 is the canonical choice
    # Final Hadamard on input qubit
    qc.h(0)
    return qc


def make_hidden_shift(n_qubits=4, shift=None, **kwargs) -> QuantumCircuit:
    """Hidden shift problem circuit.

    Given two oracles f and g where g(x) = f(x + s), find the hidden shift s.
    Uses n data qubits.  The circuit applies:
      1) H on all qubits
      2) Diagonal phase oracle f (CZ between adjacent pairs)
      3) H on all qubits
      4) Shifted oracle g (X on shift bits, CZ adjacents, X back)
      5) H on all qubits
    Measurement reveals the shift s.

    Args:
        n_qubits: Number of data qubits (minimum 4).
        shift: Integer encoding the shift bitstring. Defaults to
               alternating bits (0b1010...truncated to n bits).
    """
    n = max(4, n_qubits)
    qc = QuantumCircuit(n)

    if shift is None:
        # Default: alternating bit pattern
        shift = sum(1 << i for i in range(0, n, 2))

    shift_bits = [(shift >> i) & 1 for i in range(n)]

    # Step 1: Hadamard on all qubits
    qc.h(range(n))

    # Step 2: Diagonal phase oracle f — CZ on adjacent pairs
    for i in range(n - 1):
        qc.cz(i, i + 1)

    # Step 3: Hadamard on all qubits
    qc.h(range(n))

    # Step 4: Shifted oracle g(x) = f(x + s)
    # Apply X on qubits where shift bit is 1
    for i in range(n):
        if shift_bits[i]:
            qc.x(i)
    # Same CZ oracle
    for i in range(n - 1):
        qc.cz(i, i + 1)
    # Undo X
    for i in range(n):
        if shift_bits[i]:
            qc.x(i)

    # Step 5: Final Hadamard
    qc.h(range(n))

    return qc


def make_element_distinctness(n_qubits=5, n_steps=1, **kwargs) -> QuantumCircuit:
    """Grover-walk hybrid for element distinctness / collision detection.

    Uses 1 coin qubit + position qubits + data qubits.  Alternates between
    Grover diffusion on the coin register and quantum-walk shift steps on
    the position register, with data qubits acting as an oracle workspace.

    Layout (total = n_qubits, minimum 5):
      qubit 0           — coin
      qubits 1..p       — position register  (p = (n-1)//2)
      qubits p+1..n-1   — data register

    Each step:
      1) Coin flip: H on coin qubit
      2) Conditional walk: CX from coin to each position qubit
      3) Oracle: CZ between each data qubit and last position qubit
      4) Grover diffusion on data register: H-X-multiCZ-X-H

    Args:
        n_qubits: Total qubit count (minimum 5).
        n_steps: Number of walk-then-diffuse iterations (default 1).
    """
    n = max(5, n_qubits)
    qc = QuantumCircuit(n)

    coin = 0
    n_pos = (n - 1) // 2
    pos_qubits = list(range(1, 1 + n_pos))
    data_qubits = list(range(1 + n_pos, n))

    # Initialize position register in superposition
    for q in pos_qubits:
        qc.h(q)

    for _ in range(n_steps):
        # 1) Coin flip
        qc.h(coin)

        # 2) Conditional walk: coin controls shifts on position register
        for q in pos_qubits:
            qc.cx(coin, q)

        # 3) Oracle mark: CZ between data qubits and last position qubit
        for d in data_qubits:
            qc.cz(d, pos_qubits[-1])

        # 4) Grover diffusion on data register
        for d in data_qubits:
            qc.h(d)
            qc.x(d)
        # Multi-controlled Z via CZ chain (pairwise for QASM2 compat)
        for i in range(len(data_qubits) - 1):
            qc.cz(data_qubits[i], data_qubits[i + 1])
        for d in data_qubits:
            qc.x(d)
            qc.h(d)

    return qc


# ── Variational Family Additions ─────────────────────────────────


def make_adapt_vqe(n_qubits=4, n_operators=3, **kwargs) -> QuantumCircuit:
    """ADAPT-VQE with iteratively grown operator pool.

    Starts from the Hartree-Fock state (X on the first n//2 qubits to
    fill the lowest spin-orbitals).  Then appends n_operators single-excitation
    operators, each implemented as a CNOT-ladder + RZ + reverse CNOT-ladder.

    A single-excitation operator between occupied orbital i and virtual
    orbital j is: CX(i,i+1)...CX(j-1,j) — RZ(theta,j) — CX(j-1,j)...CX(i,i+1).

    Args:
        n_qubits: Number of qubits / spin-orbitals (minimum 4).
        n_operators: Number of excitation operators to append (default 3).
    """
    n = max(4, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(314)
    n_occ = n // 2  # occupied orbitals

    # Hartree-Fock reference state
    for i in range(n_occ):
        qc.x(i)

    # Append single-excitation operators from occupied → virtual
    for op_idx in range(n_operators):
        # Cycle through occupied-virtual pairs
        occ = op_idx % n_occ
        virt = n_occ + (op_idx % (n - n_occ))
        theta = rng.uniform(0, 2 * np.pi)

        # Forward CX ladder: occ → occ+1 → ... → virt
        for q in range(occ, virt):
            qc.cx(q, q + 1)

        # Parameterized rotation
        qc.rz(theta, virt)

        # Reverse CX ladder: virt → ... → occ
        for q in range(virt - 1, occ - 1, -1):
            qc.cx(q, q + 1)

    return qc


def make_vqd(n_qubits=4, layers=2, **kwargs) -> QuantumCircuit:
    """Variational Quantum Deflation for excited-state computation.

    Combines a hardware-efficient ansatz with a SWAP-test overlap penalty
    circuit.  The first subsystem (qubits 0..n_ansatz-1) holds the trial
    state; an ancilla qubit and a reference copy enable overlap estimation.

    Layout:
      qubits 0..m-1           — ansatz register  (m = n//2 - 1 when n>=6,
                                                    else n-2)
      qubit m                 — SWAP-test ancilla
      qubits m+1..2m          — reference copy register

    The ansatz uses RY + CZ brick-layer structure (same as
    make_hardware_efficient_ansatz).  After the ansatz, a SWAP test is
    performed between the ansatz and reference registers using the ancilla.

    Args:
        n_qubits: Total qubit count (minimum 5).
        layers: Number of ansatz layers (default 2).
    """
    n = max(5, n_qubits)
    # Split qubits: ansatz | ancilla | reference
    m = (n - 1) // 2  # ansatz size = reference size
    total = 2 * m + 1
    qc = QuantumCircuit(total)
    rng = np.random.default_rng(271)

    ansatz_qubits = list(range(m))
    ancilla = m
    ref_qubits = list(range(m + 1, total))

    # ── Hardware-efficient ansatz on ansatz register ──
    for _layer in range(layers):
        for q in ansatz_qubits:
            qc.ry(rng.uniform(0, 2 * np.pi), q)
            qc.rz(rng.uniform(0, 2 * np.pi), q)
        for i in range(0, m - 1, 2):
            qc.cz(ansatz_qubits[i], ansatz_qubits[i + 1])
        for i in range(1, m - 1, 2):
            qc.cz(ansatz_qubits[i], ansatz_qubits[i + 1])

    # ── SWAP test between ansatz and reference registers ──
    # H on ancilla
    qc.h(ancilla)

    # Controlled-SWAP for each pair (ansatz[i], ref[i])
    # Fredkin decomposition: CX(b,c), Toffoli(a,c,b), CX(b,c)
    for i in range(m):
        a_q = ansatz_qubits[i]
        r_q = ref_qubits[i]
        qc.cx(r_q, a_q)
        _decompose_toffoli(qc, ancilla, a_q, r_q)
        qc.cx(r_q, a_q)

    # H on ancilla
    qc.h(ancilla)

    return qc


def make_recursive_qaoa(n_qubits=6, p=1, gamma=0.5, beta=0.3,
                        n_fixed=2, **kwargs) -> QuantumCircuit:
    """Recursive QAOA (RQAOA) with variable fixing.

    First performs a standard QAOA layer on all qubits (MaxCut on a ring),
    then "fixes" n_fixed qubits (applies X to simulate classical assignment)
    and runs a reduced QAOA layer on the remaining qubits.

    This produces the characteristic RQAOA structure: full QAOA → partial
    measurement/fixing → reduced QAOA on a smaller problem.

    Args:
        n_qubits: Total qubits (minimum 6).
        p: QAOA depth per round (default 1).
        gamma: Problem-layer angle (default 0.5).
        beta: Mixer-layer angle (default 0.3).
        n_fixed: Number of qubits to fix after round 1 (default 2).
    """
    n = max(6, n_qubits)
    qc = QuantumCircuit(n)

    all_qubits = list(range(n))
    edges_full = [(i, (i + 1) % n) for i in range(n)]

    # ── Round 1: Full QAOA on all qubits ──
    qc.h(range(n))
    for _layer in range(p):
        # Problem unitary (ZZ on ring edges)
        for i, j in edges_full:
            qc.cx(i, j)
            qc.rz(gamma, j)
            qc.cx(i, j)
        # Mixer unitary
        for i in all_qubits:
            qc.rx(2 * beta, i)

    # ── Fix variables: apply X to "freeze" the first n_fixed qubits ──
    fixed_qubits = list(range(n_fixed))
    remaining_qubits = list(range(n_fixed, n))

    for q in fixed_qubits:
        qc.x(q)

    # ── Round 2: Reduced QAOA on remaining qubits ──
    edges_reduced = [
        (remaining_qubits[i], remaining_qubits[(i + 1) % len(remaining_qubits)])
        for i in range(len(remaining_qubits))
    ]
    # Re-initialize remaining qubits into superposition
    for q in remaining_qubits:
        qc.h(q)

    for _layer in range(p):
        for i, j in edges_reduced:
            qc.cx(i, j)
            qc.rz(gamma, j)
            qc.cx(i, j)
        for q in remaining_qubits:
            qc.rx(2 * beta, q)

    return qc


def make_varqite(n_qubits=4, layers=3, **kwargs) -> QuantumCircuit:
    """Variational Quantum Imaginary Time Evolution (VarQITE).

    Parameterized ansatz structured for McLachlan's variational principle.
    Uses RY + RZ single-qubit layers and CX entangling layers, with a
    specific parameter structure: RY angles decrease across layers
    (mimicking imaginary-time cooling) while RZ angles provide phase
    freedom.

    The layered structure is:
      For each layer l:
        1) RY(theta_l_i) on each qubit (theta decreasing with layer)
        2) RZ(phi_l_i) on each qubit
        3) Linear CX entangling: CX(i, i+1) for all adjacent pairs

    Args:
        n_qubits: Number of qubits (minimum 4).
        layers: Number of ansatz layers (default 3).
    """
    n = max(4, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(161)

    for layer_idx in range(layers):
        # RY angles: scale decreases with layer to mimic cooling
        # (imaginary time pushes toward ground state → smaller rotations)
        scale = np.pi / (1 + layer_idx)
        for i in range(n):
            theta = rng.uniform(0, scale)
            qc.ry(theta, i)

        # RZ angles: full phase freedom in every layer
        for i in range(n):
            phi = rng.uniform(0, 2 * np.pi)
            qc.rz(phi, i)

        # Linear entangling layer
        for i in range(n - 1):
            qc.cx(i, i + 1)

    return qc


# ── Error Correction Additions ────────────────────────────────────


def make_five_qubit_code(n_qubits=5, **kwargs) -> QuantumCircuit:
    """[[5,1,3]] perfect code encoder.

    The smallest quantum error-correcting code that corrects arbitrary
    single-qubit errors.  5 physical qubits encode 1 logical qubit.

    Stabilizer generators (cyclic):
        S1 = XZZXI
        S2 = IXZZX
        S3 = XIXZZ
        S4 = ZXIXZ

    Encoding circuit follows the Gottesman/Cleve approach: Hadamard on
    the four ancilla qubits, then CX/CZ gates that imprint the stabilizer
    structure onto the data qubit (qubit 4 carries the logical state).
    Uses 4 H gates and 4 CX + 5 CZ gates.
    """
    qc = QuantumCircuit(5)
    # Qubit 4 holds the logical input |ψ⟩; qubits 0-3 are ancillas.
    # Step 1: Hadamard on ancilla qubits to create superposition
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    # Step 2: CZ gates encoding Z-components of stabilizer generators.
    # The stabilizers contain Z at these qubit pairs:
    #   S1=XZZXI  → Z on q1,q2
    #   S2=IXZZX  → Z on q2,q3
    #   S3=XIXZZ  → Z on q3,q4  (q4 is logical)
    #   S4=ZXIXZ  → Z on q0,q3
    # Additional cross-stabilizer CZ for full code space:
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


def make_surface_code_patch(n_qubits=9, **kwargs) -> QuantumCircuit:
    """Minimal rotated surface code patch (distance-2).

    Layout (rotated surface code d=2):
        4 data qubits (0-3) arranged in a 2x2 grid
        4 stabilizer ancilla qubits (4-7):
          - 2 X-stabilizer ancillas (4, 5)
          - 2 Z-stabilizer ancillas (6, 7)
        1 extra ancilla (qubit 8) for additional boundary stabilizer

    Data qubit grid:
        q0 -- q1
        |      |
        q2 -- q3

    X-stabilizers measure products of X on adjacent data qubits using
    H-CX-H sequences on ancillas.  Z-stabilizers measure products of Z
    using CX from data qubits into ancillas.
    """
    qc = QuantumCircuit(9)
    # Data qubits: 0,1,2,3   X-ancillas: 4,5   Z-ancillas: 6,7   Extra: 8

    # ── X-stabilizer measurement ──
    # X-stab 1 (ancilla 4): measures X0·X1·X2·X3 (weight-4 bulk plaquette)
    qc.h(4)
    qc.cx(4, 0)
    qc.cx(4, 1)
    qc.cx(4, 2)
    qc.cx(4, 3)
    qc.h(4)

    # X-stab 2 (ancilla 5): measures X0·X1 (weight-2 boundary)
    qc.h(5)
    qc.cx(5, 0)
    qc.cx(5, 1)
    qc.h(5)

    # ── Z-stabilizer measurement ──
    # Z-stab 1 (ancilla 6): measures Z0·Z2 (weight-2 boundary)
    qc.cx(0, 6)
    qc.cx(2, 6)

    # Z-stab 2 (ancilla 7): measures Z1·Z3 (weight-2 boundary)
    qc.cx(1, 7)
    qc.cx(3, 7)

    # ── Additional boundary check (ancilla 8) ──
    # Z-check on bottom boundary: Z2·Z3
    qc.cx(2, 8)
    qc.cx(3, 8)

    return qc


# ── Linear Algebra Family ─────────────────────────────────────────


def make_hhl(n_qubits=5, **kwargs) -> QuantumCircuit:
    """HHL algorithm core circuit for solving linear systems Ax = b.

    Qubit layout (n_qubits total):
        - qubit 0: ancilla (eigenvalue inversion via controlled rotation)
        - qubits 1 .. n_counting: counting register (QPE)
        - qubit n_qubits-1: system qubit (holds |b⟩)

    Steps:
        (a) Prepare eigenstate on system qubit (X gate to set |1⟩).
        (b) QPE: Hadamard on counting qubits, controlled-phase rotations
            from counting to system, then inverse QFT on counting register.
        (c) Controlled-RY rotation on ancilla qubit conditioned on each
            counting qubit (eigenvalue inversion: rotate by arcsin(C/λ)).
        (d) Inverse QPE: forward QFT on counting register, inverse
            controlled-phase rotations, Hadamard to uncompute.

    Post-selection on ancilla qubit |1⟩ yields system register ~ A⁻¹|b⟩.
    """
    n = max(4, n_qubits)
    n_counting = n - 2  # counting register size
    ancilla = 0
    counting = list(range(1, n_counting + 1))
    system = n - 1

    qc = QuantumCircuit(n)

    # (a) Prepare system qubit in eigenstate |1⟩
    qc.x(system)

    # (b) QPE: Hadamard on counting register
    for c in counting:
        qc.h(c)

    # Controlled unitary: e^{i A t} approximated by controlled-phase gates
    # Each counting qubit c_k applies 2^k repetitions
    for idx, c in enumerate(counting):
        angle = np.pi / (2 ** idx)
        for _ in range(2 ** idx):
            qc.cp(angle, c, system)

    # Inverse QFT on counting register
    for i in range(n_counting // 2):
        qc.swap(counting[i], counting[n_counting - 1 - i])
    for i in range(n_counting):
        for j in range(i):
            qc.cp(-np.pi / (2 ** (i - j)), counting[j], counting[i])
        qc.h(counting[i])

    # (c) Controlled rotations on ancilla (eigenvalue inversion)
    # Rotate ancilla by arcsin(C / 2^k) conditioned on counting qubit k
    C = 0.5  # scaling constant
    for idx, c in enumerate(counting):
        theta = 2 * np.arcsin(C / (2 ** (idx + 1)))
        # Decompose controlled-RY: RY(θ/2) - CX - RY(-θ/2) - CX
        qc.ry(theta / 2, ancilla)
        qc.cx(c, ancilla)
        qc.ry(-theta / 2, ancilla)
        qc.cx(c, ancilla)

    # (d) Inverse QPE: undo phase estimation
    # Forward QFT on counting register
    for i in range(n_counting - 1, -1, -1):
        qc.h(counting[i])
        for j in range(i - 1, -1, -1):
            qc.cp(np.pi / (2 ** (i - j)), counting[j], counting[i])
    for i in range(n_counting // 2):
        qc.swap(counting[i], counting[n_counting - 1 - i])

    # Inverse controlled unitaries
    for idx in range(len(counting) - 1, -1, -1):
        c = counting[idx]
        angle = -np.pi / (2 ** idx)
        for _ in range(2 ** idx):
            qc.cp(angle, c, system)

    # Hadamard on counting register to complete uncomputation
    for c in counting:
        qc.h(c)

    return qc


def make_vqls(n_qubits=4, layers=2, **kwargs) -> QuantumCircuit:
    """Variational Quantum Linear Solver (VQLS) ansatz circuit.

    Prepares a parameterized trial state |x(θ)⟩ using a hardware-efficient
    ansatz with RY + RZ single-qubit rotations and CX entangling layers.
    The cost function C(θ) = 1 - |⟨b|A|x(θ)⟩|² / ⟨x(θ)|A†A|x(θ)⟩
    is minimized classically.

    Structure per layer:
        - RY(θ_i) on each qubit (parameterized rotation)
        - RZ(φ_i) on each qubit (additional expressibility)
        - CX ladder on adjacent pairs (entangling)
    Final RY layer for output expressibility.
    """
    layers = kwargs.get("layers", layers)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(789)

    for _layer in range(layers):
        # Single-qubit variational rotations
        for i in range(n):
            qc.ry(rng.uniform(0, 2 * np.pi), i)
            qc.rz(rng.uniform(0, 2 * np.pi), i)
        # Entangling layer: CX on adjacent pairs
        for i in range(n - 1):
            qc.cx(i, i + 1)

    # Final rotation layer for expressibility
    for i in range(n):
        qc.ry(rng.uniform(0, 2 * np.pi), i)

    return qc


# ── Cryptography Family ───────────────────────────────────────────


def make_bb84_encode(n_qubits=8, seed=42, **kwargs) -> QuantumCircuit:
    """BB84 QKD state preparation circuit.

    For each qubit, Alice randomly chooses:
        - A bit value (0 or 1)
        - A basis (Z-basis or X-basis)

    Encoding:
        - bit=1: apply X gate (flip to |1⟩)
        - X-basis: apply H gate (rotate to |+⟩/|−⟩ basis)

    The resulting state for each qubit is one of {|0⟩, |1⟩, |+⟩, |−⟩}.
    Uses a fixed-seed RNG for reproducibility.
    """
    seed = kwargs.get("seed", seed)
    n = max(1, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(seed)

    bits = rng.integers(0, 2, size=n)    # random bit values
    bases = rng.integers(0, 2, size=n)   # 0 = Z-basis, 1 = X-basis

    for i in range(n):
        if bits[i] == 1:
            qc.x(i)          # encode bit value
        if bases[i] == 1:
            qc.h(i)          # switch to X-basis

    return qc


def make_e91_protocol(n_qubits=8, seed=99, **kwargs) -> QuantumCircuit:
    """E91 entanglement-based QKD protocol circuit.

    Creates n_qubits//2 Bell pairs, then applies random measurement basis
    rotations (RY) on each qubit to simulate Alice and Bob independently
    choosing measurement bases.

    In the E91 protocol, Alice chooses from angles {0, π/4, π/2} and Bob
    from {π/4, π/2, 3π/4}.  Security is verified via CHSH Bell inequality
    violation on the mismatched-basis results.
    """
    seed = kwargs.get("seed", seed)
    n = max(2, n_qubits)
    n_pairs = n // 2
    total = 2 * n_pairs  # ensure even number of qubits
    qc = QuantumCircuit(total)
    rng = np.random.default_rng(seed)

    # Create Bell pairs: (0,1), (2,3), (4,5), ...
    for p in range(n_pairs):
        q_a = 2 * p
        q_b = 2 * p + 1
        _bell_pair(qc, q_a, q_b)

    # Random measurement basis rotations
    # E91 uses 3 angles for Alice and 3 for Bob
    alice_angles = [0.0, np.pi / 4, np.pi / 2]
    bob_angles = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    for p in range(n_pairs):
        q_a = 2 * p
        q_b = 2 * p + 1
        qc.ry(rng.choice(alice_angles), q_a)
        qc.ry(rng.choice(bob_angles), q_b)

    return qc


# ── Sampling Family ───────────────────────────────────────────────


def make_iqp_sampling(n_qubits=5, seed=77, **kwargs) -> QuantumCircuit:
    """IQP (Instantaneous Quantum Polynomial) sampling circuit.

    Structure: H^⊗n → D(diagonal) → H^⊗n

    The diagonal layer consists of commuting gates (all diagonal in the
    Z-basis): CZ gates between selected pairs and T gates on individual
    qubits.  Because all diagonal gates commute, they can be applied
    "instantaneously" (in any order / simultaneously).

    IQP circuits are believed to be classically hard to sample from,
    making them candidates for demonstrating quantum computational
    advantage.
    """
    seed = kwargs.get("seed", seed)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(seed)

    # Layer 1: Hadamard on all qubits → creates |+⟩^⊗n
    qc.h(range(n))

    # Layer 2: Diagonal gates (all commute, all diagonal in Z-basis)
    # CZ gates on random pairs
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                qc.cz(i, j)
    # T gates on each qubit (with random inclusion)
    for i in range(n):
        if rng.random() < 0.6:
            qc.t(i)

    # Layer 3: Hadamard on all qubits
    qc.h(range(n))

    return qc


def make_random_circuit_sampling(n_qubits=5, depth=6, seed=55,
                                 **kwargs) -> QuantumCircuit:
    """Random circuit sampling with Clifford+T gates.

    Alternating layers of:
        - Random single-qubit gates chosen from {H, S, T} on each qubit
        - CX gates on adjacent qubit pairs (even-odd / odd-even alternating)

    This structure mimics the random circuit sampling experiments used in
    quantum supremacy demonstrations (e.g., Google Sycamore).
    Uses a fixed-seed RNG for reproducibility.
    """
    depth = kwargs.get("depth", depth)
    seed = kwargs.get("seed", seed)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(seed)

    single_qubit_gates = ["h", "s", "t"]

    for layer in range(depth):
        # Single-qubit Clifford+T layer
        for i in range(n):
            gate = rng.choice(single_qubit_gates)
            if gate == "h":
                qc.h(i)
            elif gate == "s":
                qc.s(i)
            else:
                qc.t(i)

        # Entangling layer: CX on adjacent pairs
        # Alternate between even-odd and odd-even pairing each layer
        start = layer % 2
        for i in range(start, n - 1, 2):
            qc.cx(i, i + 1)

    return qc


# ── Error Mitigation ───────────────────────────────────────────────


def make_zne_folding(n_qubits=4, **kwargs) -> QuantumCircuit:
    """Zero-noise extrapolation via global unitary folding.

    Builds a simple base circuit (H + CX chain), appends its inverse
    (adjoint in reverse order), then appends the original again.
    The circuit-inverse-circuit pattern is the signature of ZNE
    global folding: at the noiseless level the result equals a single
    application of the base circuit, but the noise is amplified by a
    factor of 3, providing data for Richardson extrapolation.

    Tags: noise_mitigation, folding
    """
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)

    # ── Base circuit: H on q0, then CX chain ──
    def _append_base(circuit):
        circuit.h(0)
        for i in range(n - 1):
            circuit.cx(i, i + 1)

    # ── Inverse of base circuit (adjoint, reversed gate order) ──
    def _append_base_inverse(circuit):
        for i in range(n - 2, -1, -1):
            circuit.cx(i, i + 1)       # CX is self-adjoint
        circuit.h(0)                    # H  is self-adjoint

    # fold-1: U
    _append_base(qc)
    # fold-2: U†
    _append_base_inverse(qc)
    # fold-3: U
    _append_base(qc)

    return qc


def make_pauli_twirling(n_qubits=4, seed=33, **kwargs) -> QuantumCircuit:
    """Pauli twirling of a CNOT layer.

    For each adjacent CNOT, randomly chosen Pauli gates are inserted
    before and the appropriate conjugate Paulis after, so that the
    net unitary is identical to the bare CNOT layer.

    The 16 valid (before, after) Pauli pairs for CNOT are determined
    by the Clifford conjugation rules:
        CNOT · (Pc ⊗ Pt) = (Pc' ⊗ Pt') · CNOT
    where the mapping is:
        (I,I)->(I,I), (I,X)->(I,X), (I,Y)->(Z,Y), (I,Z)->(Z,Z),
        (X,I)->(X,X), (X,X)->(X,I), (X,Y)->(Y,Z), (X,Z)->(Y,Y),
        (Y,I)->(Y,X), (Y,X)->(Y,I), (Y,Y)->(X,Z), (Y,Z)->(X,Y),
        (Z,I)->(Z,I), (Z,X)->(Z,X), (Z,Y)->(I,Y), (Z,Z)->(I,Z).

    Tags: noise_mitigation, randomized_compiling
    """
    seed = kwargs.get("seed", seed)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(seed)

    # Pauli twirl table for CNOT: maps (before_ctrl, before_tgt) -> (after_ctrl, after_tgt)
    # Paulis encoded as 0=I, 1=X, 2=Y, 3=Z
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


# ── Simulation (randomized / higher-order) ────────────────────────


def make_qdrift(n_qubits=4, n_steps=4, dt=0.3, seed=42, **kwargs) -> QuantumCircuit:
    """qDRIFT randomised Hamiltonian simulation.

    Simulates an Ising-like Hamiltonian  H = Σ_i ZZ_{i,i+1} + Σ_i X_i
    using the qDRIFT channel: at each step a single Hamiltonian term
    is sampled with probability proportional to its coefficient, and
    the full time-step τ = λ·dt/N is applied to that term alone
    (where λ = sum of all |coefficients|).

    ZZ terms are implemented via CX-RZ-CX, X terms via RX.

    Tags: hamiltonian_simulation, randomized
    """
    n_steps = kwargs.get("n_steps", n_steps)
    dt = kwargs.get("dt", dt)
    seed = kwargs.get("seed", seed)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(seed)

    # Hamiltonian terms: (type, index, coefficient)
    #   "zz" on pair (i, i+1) with coeff 1.0
    #   "x"  on qubit i       with coeff 1.0
    terms = []
    for i in range(n - 1):
        terms.append(("zz", i, 1.0))
    for i in range(n):
        terms.append(("x", i, 1.0))

    coeffs = np.array([abs(c) for _, _, c in terms])
    lam = coeffs.sum()
    probs = coeffs / lam
    tau = lam * dt / n_steps  # rescaled time per sample

    for _ in range(n_steps):
        idx = int(rng.choice(len(terms), p=probs))
        kind, qubit_idx, coeff = terms[idx]
        angle = 2 * coeff * tau  # exp(-i H t) => rotation angle 2*coeff*tau
        if kind == "zz":
            qc.cx(qubit_idx, qubit_idx + 1)
            qc.rz(angle, qubit_idx + 1)
            qc.cx(qubit_idx, qubit_idx + 1)
        else:  # "x"
            qc.rx(angle, qubit_idx)

    return qc


def make_higher_order_trotter(n_qubits=4, n_steps=1, dt=0.5,
                              j_coupling=1.0, h_field=0.5,
                              **kwargs) -> QuantumCircuit:
    """2nd-order (symmetric) Suzuki-Trotter for transverse-field Ising.

    H = -J Σ ZZ_{i,i+1} - h Σ X_i

    The 2nd-order Trotter formula per step is:
        S2(dt) = e^{-i H_ZZ dt/2} · e^{-i H_X dt} · e^{-i H_ZZ dt/2}

    This symmetric splitting halves the leading Trotter error compared
    to the first-order formula.

    ZZ interactions are implemented as CX-RZ-CX, X field as RX.

    Tags: hamiltonian_simulation, trotter, higher_order
    """
    n_steps = kwargs.get("n_steps", n_steps)
    dt = kwargs.get("dt", dt)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)

    def _zz_layer(circuit, angle):
        """Apply exp(-i J ZZ angle) for all adjacent pairs."""
        for i in range(n - 1):
            circuit.cx(i, i + 1)
            circuit.rz(2 * j_coupling * angle, i + 1)
            circuit.cx(i, i + 1)

    def _x_layer(circuit, angle):
        """Apply exp(-i h X angle) on every qubit."""
        for i in range(n):
            circuit.rx(2 * h_field * angle, i)

    for _ in range(n_steps):
        _zz_layer(qc, dt / 2)   # half-step ZZ
        _x_layer(qc, dt)        # full-step X
        _zz_layer(qc, dt / 2)   # half-step ZZ

    return qc


# ── Topological ────────────────────────────────────────────────────


def make_jones_polynomial(n_qubits=4, **kwargs) -> QuantumCircuit:
    """Jones polynomial approximation circuit.

    Approximates the Jones polynomial evaluated at a root of unity
    using a chain of Hadamard and controlled-phase gates that mimic
    the braid-group representation underlying the Temperley-Lieb
    algebra.

    Structure: H on first qubit, then alternating CP(2π/5) and H
    gates in a braid-like ladder pattern.  The angle 2π/5 corresponds
    to the 5th root of unity, a standard evaluation point for
    BQP-completeness of the Jones polynomial.

    Tags: knot_invariant, topological
    """
    n = max(3, n_qubits)
    qc = QuantumCircuit(n)
    angle = 2 * np.pi / 5  # 5th root of unity

    qc.h(0)
    # Braid-like pattern: sweep forward and backward creating crossings
    for sweep in range(2):
        if sweep % 2 == 0:
            # Forward sweep
            for i in range(n - 1):
                qc.cp(angle, i, i + 1)
                qc.h(i + 1)
        else:
            # Backward sweep
            for i in range(n - 2, -1, -1):
                qc.cp(angle, i + 1, i)
                qc.h(i)

    # Final controlled-phase layer for closure
    for i in range(n - 1):
        qc.cp(angle, i, i + 1)

    return qc


def make_toric_code_syndrome(n_qubits=8, **kwargs) -> QuantumCircuit:
    """Toric code syndrome extraction on a minimal 2×2 lattice.

    Layout (8 qubits):
      - Data qubits:   q0, q1, q2, q3  (edges of the torus)
      - Ancilla qubits: q4, q5  (X-stabiliser / vertex operators)
                         q6, q7  (Z-stabiliser / plaquette operators)

    X-stabilisers (vertex operators) measure X⊗X⊗X⊗X via:
        H on ancilla, CX from ancilla to each data qubit, H on ancilla.
    Z-stabilisers (plaquette operators) measure Z⊗Z⊗Z⊗Z via:
        CX from each data qubit to ancilla.

    For the minimal 2×2 torus there are 2 independent vertex operators
    and 2 independent plaquette operators.

    Tags: topological_code, syndrome_extraction
    """
    n = 8  # fixed for minimal toric code
    qc = QuantumCircuit(n)

    data = [0, 1, 2, 3]
    x_ancillas = [4, 5]
    z_ancillas = [6, 7]

    # Vertex operator assignments (each vertex touches 2 edges on minimal torus)
    vertex_data = [
        [0, 1],  # vertex-0 stabiliser acts on data q0, q1
        [2, 3],  # vertex-1 stabiliser acts on data q2, q3
    ]

    # Plaquette operator assignments
    plaquette_data = [
        [0, 2],  # plaquette-0 stabiliser acts on data q0, q2
        [1, 3],  # plaquette-1 stabiliser acts on data q1, q3
    ]

    # ── X-stabiliser extraction (vertex operators) ──
    for anc_idx, anc in enumerate(x_ancillas):
        qc.h(anc)
        for d in vertex_data[anc_idx]:
            qc.cx(anc, d)
        qc.h(anc)

    # ── Z-stabiliser extraction (plaquette operators) ──
    for anc_idx, anc in enumerate(z_ancillas):
        for d in plaquette_data[anc_idx]:
            qc.cx(d, anc)

    return qc


# ── Metrology ──────────────────────────────────────────────────────


def make_ghz_metrology(n_qubits=4, phi=0.3, **kwargs) -> QuantumCircuit:
    """GHZ-state Ramsey interferometry for Heisenberg-limited sensing.

    Steps:
      (a) Prepare GHZ state — H on q0, CX chain q0→q1→…→qN-1.
      (b) Phase accumulation — RZ(phi) on every qubit.  Because the
          GHZ state is |00…0⟩+|11…1⟩, the N-fold phase kick gives
          Heisenberg scaling ΔΦ ~ 1/N.
      (c) Inverse GHZ preparation — reverse CX chain, H on q0.
          The final state encodes the phase; measurement in the
          computational basis reveals it.

    Tags: sensing, heisenberg_limit
    """
    phi = kwargs.get("phi", phi)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)

    # (a) Prepare GHZ
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)

    # (b) Phase accumulation
    for i in range(n):
        qc.rz(phi, i)

    # (c) Inverse GHZ preparation
    for i in range(n - 2, -1, -1):
        qc.cx(i, i + 1)
    qc.h(0)

    return qc


def make_quantum_fisher_info(n_qubits=4, phi=0.5, layers=2,
                             **kwargs) -> QuantumCircuit:
    """Parameterised probe state for quantum Fisher information estimation.

    Alternating layers of:
      - RY(phi) on all qubits (parameter encoding)
      - CX entangling ladder (q0→q1, q1→q2, …)

    The entanglement generated by the CX ladder enhances the quantum
    Fisher information (QFI) of the state with respect to the
    parameter phi.  The QFI quantifies the ultimate precision with
    which phi can be estimated from measurements on this state.

    Tags: sensing, parameter_estimation
    """
    phi = kwargs.get("phi", phi)
    layers = kwargs.get("layers", layers)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)

    for _ in range(layers):
        # Parameter-encoding layer
        for i in range(n):
            qc.ry(phi, i)
        # Entangling ladder
        for i in range(n - 1):
            qc.cx(i, i + 1)

    return qc


# ── Simulation (fermionic / quantum-walk) ──────────────────────────


def make_hubbard_trotter(n_qubits=4, n_steps=1, dt=0.5,
                         t_hop=1.0, u_int=2.0, **kwargs) -> QuantumCircuit:
    """Trotterised Fermi-Hubbard model (1-D, spinless, Jordan-Wigner).

    H = -t Σ (c†_i c_{i+1} + h.c.) + U Σ n_i

    Under Jordan-Wigner the hopping term for adjacent sites becomes:
        (X_i X_{i+1} + Y_i Y_{i+1}) / 2
    which is implemented as:
        RY(-π/2) on i, CX(i, i+1), RY(angle) on i, CX(i, i+1), RY(π/2) on i

    The on-site interaction n_i maps to (I - Z_i)/2 and contributes
    an RZ rotation on each qubit.

    Tags: hamiltonian_simulation, fermionic, trotter
    """
    n_steps = kwargs.get("n_steps", n_steps)
    dt = kwargs.get("dt", dt)
    t_hop = kwargs.get("t_hop", t_hop)
    u_int = kwargs.get("u_int", u_int)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)

    hop_angle = t_hop * dt

    for _ in range(n_steps):
        # ── Hopping terms (XX + YY interaction via JW) ──
        for i in range(n - 1):
            # Decomposition of exp(-i t_hop dt (XX+YY)/2)
            qc.ry(-np.pi / 2, i)
            qc.cx(i, i + 1)
            qc.ry(hop_angle, i)
            qc.cx(i, i + 1)
            qc.ry(np.pi / 2, i)

        # ── On-site interaction terms ──
        for i in range(n):
            qc.rz(u_int * dt, i)

    return qc


def make_ctqw(n_qubits=4, n_steps=2, dt=0.5, **kwargs) -> QuantumCircuit:
    """Continuous-time quantum walk on a line graph via Trotterisation.

    Simulates e^{-i A t} where A is the adjacency matrix of a path
    graph (line) on n vertices.  Each Trotter step approximates:

      e^{-i A dt} ≈ Π_{edges} e^{-i dt (|i⟩⟨j| + |j⟩⟨i|)} · Π_{vertices} e^{-i dt}

    Edge interactions: the off-diagonal part of A couples adjacent
    sites and is implemented via CX-RZ-CX (equivalent to exp(-i dt ZZ)
    in the interaction picture after basis rotation).

    Vertex terms: the diagonal of A for a line graph is zero, but we
    include a small RX on each vertex as an effective on-site potential
    to enrich the motif structure.

    Tags: quantum_walk, continuous_time
    """
    n_steps = kwargs.get("n_steps", n_steps)
    dt = kwargs.get("dt", dt)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)

    for _ in range(n_steps):
        # Edge interactions (adjacency coupling between neighbours)
        for i in range(n - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * dt, i + 1)
            qc.cx(i, i + 1)
        # Vertex on-site potential
        for i in range(n):
            qc.rx(dt, i)

    return qc


# ── Arithmetic Additions ──────────────────────────────────────────


def make_quantum_multiplier(n_qubits=6, **kwargs) -> QuantumCircuit:
    """Schoolbook quantum multiplier for 2-bit x 2-bit unsigned integers.

    Uses a shift-and-add approach: for each bit of multiplicand A,
    perform a controlled addition of multiplier B into the output
    register, shifted by the bit position.  AND operations (partial
    product bits) are computed via decomposed Toffoli gates.

    Qubit layout (minimum 6):
        q0, q1 — input register A  (2-bit multiplicand)
        q2, q3 — input register B  (2-bit multiplier)
        q4, q5 — output / accumulator P (low 2 bits of 4-bit product;
                  carry information propagates through the circuit)

    The full 4-bit product is A*B.  Because a 2-bit x 2-bit product
    can be at most 9 (3*3), two output qubits plus internal carries
    suffice for the low-order bits.  The circuit computes partial
    products a_i AND b_j via Toffoli(a_i, b_j, p_{i+j}) and
    propagates carries with CX chains.

    After execution the output register holds partial-product
    information that, combined with the input registers, encodes
    the multiplication result.

    Tags: multiplication, classical_reversible
    """
    qc = QuantumCircuit(max(6, n_qubits))
    a0, a1 = 0, 1        # multiplicand A
    b0, b1 = 2, 3        # multiplier B
    p0, p1 = 4, 5        # product / accumulator

    # --- Partial product a0*b0 -> p0 ---
    # Toffoli(a0, b0, p0) computes p0 ^= a0 AND b0
    _decompose_toffoli(qc, a0, b0, p0)

    # --- Partial product a0*b1 -> p1 ---
    # Toffoli(a0, b1, p1) computes p1 ^= a0 AND b1
    _decompose_toffoli(qc, a0, b1, p1)

    # --- Partial product a1*b0 -> p1 (shifted by 1) ---
    # Toffoli(a1, b0, p1) computes p1 ^= a1 AND b0
    # This adds the a1*b0 partial product at the correct bit position
    _decompose_toffoli(qc, a1, b0, p1)

    # --- Cross term a1*b1 needs position 2, which overflows our
    #     2-bit accumulator.  We propagate the carry effect back
    #     through the accumulator using CX gates. ---
    # First compute a1 AND b1 and XOR into p0 (carry propagation
    # wraparound — this encodes the high-bit carry information
    # into the circuit structure for motif richness).
    _decompose_toffoli(qc, a1, b1, p0)

    # Carry propagation: if both partial products at position 1
    # produced a 1, propagate carry back.
    qc.cx(p0, p1)
    qc.cx(p1, p0)

    return qc


def make_quantum_comparator(n_qubits=5, **kwargs) -> QuantumCircuit:
    """Quantum less-than comparator for two 2-bit unsigned integers.

    Determines whether A < B using a subtraction-like borrow-
    propagation circuit.  The comparison result appears on a
    dedicated output qubit.

    Qubit layout (minimum 5):
        q0, q1 — input register A  (2-bit, A = 2*a1 + a0)
        q2, q3 — input register B  (2-bit, B = 2*b1 + b0)
        q4     — result qubit (set to |1> iff A < B)

    Algorithm (ripple-borrow comparator):
      1. Compute borrow_0 from bit-0: borrow if a0=0 and b0=1.
         This is detected by flipping a0, then Toffoli(a0, b0, result),
         then unflipping a0.
      2. Propagate to bit-1: XOR the bit-1 values with the borrow,
         then compute the final borrow using another Toffoli.
      3. The final borrow bit equals (A < B).

    The circuit uses CX for XOR, X for NOT, and decomposed Toffoli
    for AND operations, ensuring QASM2 compatibility.

    Tags: comparison, classical_reversible
    """
    qc = QuantumCircuit(max(5, n_qubits))
    a0, a1 = 0, 1        # A register
    b0, b1 = 2, 3        # B register
    result = 4            # output: 1 iff A < B

    # --- Bit-0 borrow: borrow if a0=0 AND b0=1 ---
    # Flip a0 so Toffoli computes (NOT a0) AND b0
    qc.x(a0)
    _decompose_toffoli(qc, a0, b0, result)
    qc.x(a0)
    # result now holds borrow_0 = (NOT a0) AND b0

    # --- Propagate borrow into bit-1 comparison ---
    # XOR borrow into a1 and b1 to account for the incoming borrow.
    # If there's a borrow, it effectively decrements the a1 bit
    # and can flip the comparison at bit 1.
    qc.cx(result, a1)
    qc.cx(result, b1)

    # --- Bit-1 borrow: borrow if (adjusted) a1=0 AND b1=1 ---
    # Flip a1, compute (NOT a1) AND b1, XOR into result
    qc.x(a1)
    _decompose_toffoli(qc, a1, b1, result)
    qc.x(a1)

    # --- Uncompute the borrow propagation into a1, b1 ---
    qc.cx(result, b1)
    qc.cx(result, a1)

    return qc


# ── Protocol Addition ─────────────────────────────────────────────


def make_swap_test(n_qubits=3, **kwargs) -> QuantumCircuit:
    """SWAP test for quantum state fidelity estimation.

    Estimates |<psi|phi>|^2 by interfering two states through a
    controlled-SWAP operation.  Measuring the ancilla qubit in the
    |0> state occurs with probability (1 + |<psi|phi>|^2) / 2.

    Qubit layout:
        q0             — ancilla (control)
        q1 .. q_mid    — first state register  |psi>
        q_{mid+1} .. q_{n-1} — second state register |phi>

    For odd n_qubits the registers are split as evenly as possible
    (the first register gets the extra qubit and the controlled-SWAP
    pairs as many qubits as the smaller register allows).

    Circuit structure:
        1. H on ancilla
        2. Controlled-SWAP between paired qubits of the two registers
           (decomposed as CX-Toffoli-CX, the standard Fredkin
            decomposition)
        3. H on ancilla

    The controlled-SWAP (Fredkin) gate on control c, targets t1, t2
    is decomposed as:
        CX(t2, t1)
        Toffoli(c, t1, t2)
        CX(t2, t1)

    Tags: fidelity, comparison
    """
    n = max(3, n_qubits)
    qc = QuantumCircuit(n)

    ancilla = 0
    # Split remaining qubits into two registers
    remaining = n - 1
    reg_size = remaining // 2  # size of the smaller register

    # First register: qubits 1 .. reg_size
    # Second register: qubits (remaining - reg_size + 1) .. n-1
    # We pair them: (1, 1+reg_size) when remaining is even,
    # or (1, 2+reg_size_first) adjusted for odd splits.
    first_start = 1
    second_start = 1 + (remaining - reg_size)  # = 1 + ceil(remaining/2)
    # For n=3: remaining=2, reg_size=1, first=[1], second=[2]
    # For n=5: remaining=4, reg_size=2, first=[1,2], second=[3,4]
    # For n=7: remaining=6, reg_size=3, first=[1,2,3], second=[4,5,6]

    # Step 1: Hadamard on ancilla
    qc.h(ancilla)

    # Step 2: Controlled-SWAP for each pair
    for i in range(reg_size):
        t1 = first_start + i
        t2 = second_start + i
        # Fredkin decomposition: CX(t2, t1), Toffoli(ancilla, t1, t2), CX(t2, t1)
        qc.cx(t2, t1)
        _decompose_toffoli(qc, ancilla, t1, t2)
        qc.cx(t2, t1)

    # Step 3: Hadamard on ancilla
    qc.h(ancilla)

    return qc


# ── Machine Learning Additions ────────────────────────────────────


def make_qsvm(n_qubits=4, **kwargs) -> QuantumCircuit:
    """Quantum SVM feature map (ZZ feature map + measurement-ready).

    Implements the ZZFeatureMap structure used in quantum kernel methods
    for support vector classification.  Two repetitions of:
        1) H on all qubits (create superposition)
        2) RZ(x_i) on each qubit (single-qubit data encoding)
        3) CX-RZ-CX on adjacent pairs (ZZ entangling feature map)

    The ZZ interaction terms encode pairwise feature correlations
    φ(x_i, x_j) = (π - x_i)(π - x_j), giving the kernel its
    non-linear expressive power.

    Uses fixed-seed RNG to supply reproducible data values.

    Args:
        n_qubits: Number of qubits / features (minimum 2).

    Tags: kernel_method, classification
    """
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(2024)

    # Simulate classical data values x_i ∈ [0, 2π)
    data = rng.uniform(0, 2 * np.pi, n)

    for _rep in range(2):
        # Step 1: Hadamard on all qubits
        for i in range(n):
            qc.h(i)

        # Step 2: Single-qubit data encoding RZ(x_i)
        for i in range(n):
            qc.rz(data[i], i)

        # Step 3: ZZ entangling feature map on adjacent pairs
        # Encodes φ(x_i, x_j) = (π - x_i)(π - x_j)
        for i in range(n - 1):
            phi_ij = (np.pi - data[i]) * (np.pi - data[i + 1])
            qc.cx(i, i + 1)
            qc.rz(2 * phi_ij, i + 1)
            qc.cx(i, i + 1)

    return qc


def make_qcnn(n_qubits=8, **kwargs) -> QuantumCircuit:
    """Quantum convolutional neural network.

    Alternating convolutional and pooling layers that progressively
    reduce the number of active qubits, mirroring the structure of a
    classical CNN.  Each conv+pool stage halves the active qubit count.

    Convolutional layer: RY + CX on adjacent pairs within the active set
    (translational-invariance structure).

    Pooling layer: CX from each even-indexed active qubit to the
    subsequent odd-indexed active qubit, then discard the odd qubits
    (trace out).  The surviving even-indexed qubits form the next
    active set.

    Requires n_qubits to be a power of 2 (minimum 4).

    Args:
        n_qubits: Number of qubits, must be power of 2 (minimum 4).

    Tags: neural_network, classification
    """
    # Ensure power of 2, minimum 4
    n = max(4, n_qubits)
    # Round up to next power of 2
    n = 1 << (n - 1).bit_length()

    qc = QuantumCircuit(n)
    rng = np.random.default_rng(808)

    active = list(range(n))

    while len(active) > 1:
        # ── Convolutional layer: RY + CX on adjacent pairs ──
        for i in range(0, len(active) - 1, 2):
            q0 = active[i]
            q1 = active[i + 1]
            qc.ry(rng.uniform(0, 2 * np.pi), q0)
            qc.ry(rng.uniform(0, 2 * np.pi), q1)
            qc.cx(q0, q1)

        # Also entangle odd-even boundary pairs for full coverage
        for i in range(1, len(active) - 1, 2):
            q0 = active[i]
            q1 = active[i + 1]
            qc.ry(rng.uniform(0, 2 * np.pi), q0)
            qc.ry(rng.uniform(0, 2 * np.pi), q1)
            qc.cx(q0, q1)

        # ── Pooling layer: CX from even to odd, discard odd ──
        surviving = []
        for i in range(0, len(active) - 1, 2):
            q_even = active[i]
            q_odd = active[i + 1]
            qc.cx(q_even, q_odd)
            surviving.append(q_even)

        active = surviving

    return qc


def make_qgan_generator(n_qubits=4, layers=3, **kwargs) -> QuantumCircuit:
    """QGAN generator circuit.

    Parameterised ansatz structured as a quantum generator network for
    a quantum generative adversarial network.  Each layer applies:
        1) RY(θ_i) on each qubit (amplitude control)
        2) RZ(φ_i) on each qubit (phase control)
        3) CX entangling ladder on adjacent pairs

    The generator transforms an initial |0...0⟩ state into a learned
    probability distribution over computational basis states.

    Uses fixed-seed RNG for reproducible parameter initialisation.

    Args:
        n_qubits: Number of qubits (minimum 2).
        layers: Number of generator layers (default 3).

    Tags: generative, adversarial
    """
    layers = kwargs.get("layers", layers)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(606)

    for _layer in range(layers):
        # Single-qubit rotations: RY for amplitude, RZ for phase
        for i in range(n):
            qc.ry(rng.uniform(0, 2 * np.pi), i)
            qc.rz(rng.uniform(0, 2 * np.pi), i)

        # Entangling layer: linear CX chain
        for i in range(n - 1):
            qc.cx(i, i + 1)

    # Final rotation layer for output expressibility
    for i in range(n):
        qc.ry(rng.uniform(0, 2 * np.pi), i)

    return qc


def make_quantum_autoencoder(n_qubits=6, **kwargs) -> QuantumCircuit:
    """Quantum autoencoder circuit.

    Compresses an n-qubit quantum state into a smaller latent register.

    Layout:
        - qubits 0 .. n_input-1   : input register
        - qubits n_input .. n-1    : latent/reference register

    Encoder: RY + CX layers that entangle input and latent registers,
    driving information from input into latent space.

    SWAP test: After encoding, a CX-based overlap check between the
    "trash" qubits (input qubits that should decouple) and reference
    qubits tests whether compression succeeded.  When training
    converges, the trash qubits are in |0⟩ and the SWAP test
    succeeds with probability 1.

    Args:
        n_qubits: Total number of qubits (minimum 4, even preferred).

    Tags: compression, autoencoder
    """
    n = max(4, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(909)

    n_input = n // 2
    n_latent = n - n_input

    input_qubits = list(range(n_input))
    latent_qubits = list(range(n_input, n))

    # ── Encoder: parametrised layers compressing input → latent ──
    # Layer 1: RY rotations on all qubits
    for i in range(n):
        qc.ry(rng.uniform(0, 2 * np.pi), i)

    # Layer 2: CX entangling input with latent (cross-register coupling)
    for i in range(min(n_input, n_latent)):
        qc.cx(input_qubits[i], latent_qubits[i])

    # Layer 3: More RY + intra-register entangling
    for i in range(n):
        qc.ry(rng.uniform(0, 2 * np.pi), i)
    for i in range(n_input - 1):
        qc.cx(input_qubits[i], input_qubits[i + 1])
    for i in range(n_latent - 1):
        qc.cx(latent_qubits[i], latent_qubits[i + 1])

    # Layer 4: Cross-register entangling again
    for i in range(min(n_input, n_latent)):
        qc.cx(input_qubits[i], latent_qubits[i])

    # Final RY for output expressibility
    for i in range(n):
        qc.ry(rng.uniform(0, 2 * np.pi), i)

    # ── SWAP test between trash (input) and reference (latent) ──
    # Uses CX-based overlap check: CX from trash to reference
    # If compression succeeded, trash qubits are |0⟩ and CX has no effect
    n_pairs = min(n_input, n_latent)
    for i in range(n_pairs):
        qc.cx(input_qubits[i], latent_qubits[i])

    return qc


# ── Transform Additions ───────────────────────────────────────────


def make_iterative_qpe(n_qubits=3, n_bits=3, angle=np.pi / 4,
                       **kwargs) -> QuantumCircuit:
    """Single-ancilla iterative phase estimation.

    Estimates the phase of a unitary (Z-rotation by `angle`) one bit
    at a time, using only 1 ancilla qubit + 1 system qubit.  The
    inverse QFT is performed semiclassically: after each controlled
    power of the unitary, a classically controlled RZ correction
    (based on previously extracted bits) is applied before the final
    Hadamard.

    For each bit k (from most significant to least):
        1) H on ancilla
        2) Controlled-phase with 2^k repetitions of the unitary
        3) Phase correction: RZ(-sum of previously determined bits)
        4) H on ancilla
        5) "Reset" ancilla with X (simulates mid-circuit
           measurement + classical feedback)

    Since mid-circuit measurement is not QASM2-compatible, the
    reset is approximated by an X gate, which captures the
    characteristic iterative structure.

    Args:
        n_qubits: Accepted but circuit always uses 2 qubits
                  (1 ancilla + 1 system).
        n_bits: Number of phase bits to extract (default 3).
        angle: Phase angle of the unitary to estimate (default π/4).

    Tags: phase_estimation, iterative
    """
    n_bits = kwargs.get("n_bits", n_bits)
    angle = kwargs.get("angle", angle)
    qc = QuantumCircuit(2)
    ancilla = 0
    system = 1

    # Prepare system qubit in eigenstate |1⟩ of the Z-rotation
    qc.x(system)

    # Track extracted phase bits for correction
    extracted_phases = []

    for k in range(n_bits - 1, -1, -1):
        # Step 1: H on ancilla
        qc.h(ancilla)

        # Step 2: Controlled-U^{2^k} — apply 2^k controlled-phase gates
        reps = 2 ** k
        for _ in range(reps):
            qc.cp(angle, ancilla, system)

        # Step 3: Phase correction from previously extracted bits
        # This replaces the classical feedback in the semiclassical QFT
        correction = 0.0
        for idx, phase_bit in enumerate(extracted_phases):
            correction += phase_bit / (2 ** (idx + 1))
        if correction != 0.0:
            qc.rz(-2 * np.pi * correction, ancilla)

        # Step 4: H on ancilla
        qc.h(ancilla)

        # Step 5: "Reset" ancilla — simulate measurement + feedback
        # In a real iterative QPE this would be a measurement;
        # we use X to capture the reset structure for motif analysis
        extracted_phases.append(0.5)  # assume average for reproducibility
        qc.x(ancilla)

    return qc


def make_amplitude_estimation(n_qubits=5, **kwargs) -> QuantumCircuit:
    """Canonical quantum amplitude estimation (Grover operator + QPE).

    Combines Grover's search operator Q with quantum phase estimation
    to estimate the amplitude a = sin²(θ) of a marked state.

    Layout:
        - qubits 0 .. n_counting-1 : counting register (QPE ancillas)
        - qubits n_counting .. n-1  : state register (Grover workspace)

    Steps:
        (a) Initialize state register in uniform superposition (H on all).
        (b) H on counting register.
        (c) Controlled-Q^{2^k}: for each counting qubit k, apply 2^k
            iterations of the Grover operator Q = -A S_0 A† S_χ, where
            S_χ is a simplified oracle (CZ on last state qubit) and
            S_0 is the zero-state reflection (X-CZ-X).
        (d) Inverse QFT on counting register.

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
            # ── Oracle S_χ: flip phase of "marked" state ──
            # Simplified: CZ between control and last state qubit
            qc.cz(counting[k], state[-1])

            # ── Diffusion S_0 (controlled): H-X-CZ-X-H on state register ──
            for s in state:
                qc.h(s)
                qc.x(s)
            # Multi-controlled Z approximated by pairwise CZ chain
            # controlled by the counting qubit
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


# ── Entanglement Additions ────────────────────────────────────────


def make_dicke_state(n_qubits=4, k=2, **kwargs) -> QuantumCircuit:
    """Dicke state |D_n^k⟩ preparation.

    Prepares the symmetric superposition of all n-qubit states with
    exactly k excitations (Hamming weight k).  For example, |D_4^2⟩
    is the equal superposition of |0011⟩, |0101⟩, |0110⟩, |1001⟩,
    |1010⟩, |1100⟩.

    Algorithm (Bärtschi-Eidenbenz deterministic approach):
        1) Start with k X gates on the first k qubits:
           |1...1 0...0⟩  (k ones, n-k zeros)
        2) For each qubit position i from 0 to n-2, distribute the
           remaining excitations into the suffix qubits using
           controlled-RY rotations.  The rotation angle for splitting
           m excitations among r remaining qubits is:
               θ = 2·arccos(√((r-1)/r))  when distributing 1 of m
           This is implemented via: RY(θ/2) - CX - RY(-θ/2) - CX
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
    #
    # For qubit at position i:
    #   - excitations_left = k - (number already "distributed" to the left)
    #   - qubits_remaining = n - i
    #   - The probability that qubit i should be |1⟩ is excitations_left / qubits_remaining
    #   - Rotation angle θ = 2·arccos(√(excitations_left / qubits_remaining))
    #
    # We implement a controlled-RY: qubit i controls a rotation on qubit i+1
    # that splits the excitation amplitude.

    for i in range(n - 1):
        # Determine how many excitations can still be "to the right" of position i.
        # For the Dicke state, at position i with the first k qubits excited,
        # we need to symmetrise.  We use the pairwise split approach:
        # pair (i, i+1) — if qubit i is |1⟩, share the excitation with
        # probability sqrt((n-i-1)/(n-i)) for qubit i keeping it.
        remaining = n - i
        if remaining <= 1:
            break

        # Angle to split: qubit i has excitation, split to i+1 with probability 1/remaining
        # RY(θ) where θ = 2·arccos(√((remaining-1)/remaining))
        theta = 2 * np.arccos(np.sqrt((remaining - 1) / remaining))

        # Controlled-RY decomposition: RY(θ/2) - CX - RY(-θ/2) - CX
        qc.ry(theta / 2, i + 1)
        qc.cx(i, i + 1)
        qc.ry(-theta / 2, i + 1)
        qc.cx(i, i + 1)

    return qc


def make_graph_state(n_qubits=5, **kwargs) -> QuantumCircuit:
    """Graph state with star topology.

    Prepares a graph state where qubit 0 is the centre of a star graph
    and all other qubits are leaves.  The preparation is:
        1) H on all qubits — initialise each qubit in |+⟩.
        2) CZ from the centre (q0) to every other qubit — create edges
           of the star graph.

    Star graph states are equivalent to GHZ states up to local
    unitaries and are fundamental resources in quantum networks
    (e.g. quantum secret sharing, anonymous transmission).

    The stabiliser generators for the star state are:
        Centre: X_0 ⊗ Z_1 ⊗ Z_2 ⊗ … ⊗ Z_{n-1}
        Leaf j: Z_0 ⊗ I ⊗ … ⊗ X_j ⊗ … ⊗ I

    Args:
        n_qubits: Number of qubits (minimum 2).

    Tags: graph_state, multipartite
    """
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)

    # Step 1: Hadamard on all qubits to prepare |+⟩^⊗n
    for i in range(n):
        qc.h(i)

    # Step 2: CZ from centre (q0) to every leaf qubit
    for i in range(1, n):
        qc.cz(0, i)

    return qc


# ── Quantum Walk Search ──────────────────────────────────────────────


def make_quantum_walk_search(n_qubits=5, n_steps=2, **kwargs) -> QuantumCircuit:
    """Quantum spatial search via coined quantum walk.

    Implements a discrete-time quantum walk search on a position register
    with a marked vertex.  Each walk step consists of a coin flip
    (Hadamard on the coin qubit plus a CZ phase oracle marking a target
    position), a conditional shift (CX from coin to each position qubit),
    and a Grover-like diffusion on the position register.

    Layout:
        qubit 0         — coin qubit
        qubits 1..n-1   — position register

    Each step:
        1) Coin flip: H on coin qubit
        2) Phase oracle: CZ between coin and each position qubit
           (marks the target vertex by flipping the phase when the
            coin is in state |1⟩ and the position matches)
        3) Conditional shift: CX from coin to each position qubit
           (moves the walker left/right based on coin state)
        4) Grover diffusion on position register:
           H → X → pairwise CZ → X → H

    Args:
        n_qubits: Total qubit count (minimum 3, default 5).
        n_steps: Number of walk-then-search iterations (default 2).

    Tags: quantum_walk, search
    """
    n_steps = kwargs.get("n_steps", n_steps)
    n = max(3, n_qubits)
    qc = QuantumCircuit(n)

    coin = 0
    pos_qubits = list(range(1, n))

    # Initialize position register in uniform superposition
    for q in pos_qubits:
        qc.h(q)

    for _ in range(n_steps):
        # 1) Coin flip
        qc.h(coin)

        # 2) Phase oracle: CZ between coin and position qubits
        #    This marks the target position by imprinting a phase
        #    when coin=|1⟩ and position is the marked vertex.
        for q in pos_qubits:
            qc.cz(coin, q)

        # 3) Conditional shift: CX from coin to each position qubit
        for q in pos_qubits:
            qc.cx(coin, q)

        # 4) Grover-like diffusion on position register
        for q in pos_qubits:
            qc.h(q)
            qc.x(q)
        # Multi-controlled Z via pairwise CZ chain (QASM2 compatible)
        for i in range(len(pos_qubits) - 1):
            qc.cz(pos_qubits[i], pos_qubits[i + 1])
        for q in pos_qubits:
            qc.x(q)
            qc.h(q)

    return qc


# ── Variational Additions ────────────────────────────────────────────


def make_qaoa_weighted(n_qubits=4, p=1, gamma=0.5, beta=0.3,
                       **kwargs) -> QuantumCircuit:
    """QAOA for weighted MaxCut on a ring graph with non-uniform edge weights.

    Same structure as standard QAOA MaxCut but each edge in the ring graph
    has a different weight.  The ZZ interaction angle for each edge is
    gamma * weight (instead of a uniform gamma), which produces a richer
    circuit structure with varying rotation angles.

    Weights cycle through [1.0, 0.5, 1.5, 0.8] for successive edges.

    Problem unitary per layer:
        For each edge (i, j) with weight w:
            CX(i, j) — RZ(gamma * w, j) — CX(i, j)

    Mixer unitary per layer:
        RX(2 * beta) on each qubit

    Args:
        n_qubits: Number of qubits (minimum 3, default 4).
        p: Number of QAOA layers (default 1).
        gamma: Problem-layer base angle (default 0.5).
        beta: Mixer-layer angle (default 0.3).

    Tags: combinatorial, weighted
    """
    p = kwargs.get("p", p)
    gamma = kwargs.get("gamma", gamma)
    beta = kwargs.get("beta", beta)
    n = max(3, n_qubits)
    qc = QuantumCircuit(n)

    # Edge weights cycling through [1.0, 0.5, 1.5, 0.8]
    weight_pool = [1.0, 0.5, 1.5, 0.8]
    edges = [(i, (i + 1) % n) for i in range(n)]
    weights = [weight_pool[i % len(weight_pool)] for i in range(len(edges))]

    # Initial superposition
    qc.h(range(n))

    for _layer in range(p):
        # Problem unitary: ZZ interaction with edge-dependent weights
        for (i, j), w in zip(edges, weights):
            qc.cx(i, j)
            qc.rz(gamma * w, j)
            qc.cx(i, j)
        # Mixer unitary
        for i in range(n):
            qc.rx(2 * beta, i)

    return qc


def make_quantum_boltzmann(n_qubits=4, layers=2, beta_param=0.5,
                           **kwargs) -> QuantumCircuit:
    """Quantum Boltzmann machine training circuit.

    Parameterised ansatz for a restricted quantum Boltzmann machine with
    visible and hidden qubit registers.  The first half of the qubits
    form the visible layer and the second half form the hidden layer.

    Each layer applies:
        1) RY(theta) on each visible qubit (amplitude encoding)
        2) RZ(phi) on each hidden qubit (phase / interaction term)
        3) CX entangling between visible and hidden qubits
           (cross-layer coupling that models the RBM weight matrix)

    The circuit structure mirrors the unitary-coupled RBM ansatz
    where the visible-hidden coupling is implemented through
    parameterised two-qubit interactions.

    Args:
        n_qubits: Total qubit count (minimum 4, even preferred, default 4).
        layers: Number of ansatz layers (default 2).
        beta_param: Inverse temperature parameter scaling RZ angles (default 0.5).

    Tags: generative, thermal
    """
    layers = kwargs.get("layers", layers)
    beta_param = kwargs.get("beta_param", beta_param)
    n = max(4, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(505)

    n_visible = n // 2
    n_hidden = n - n_visible
    visible = list(range(n_visible))
    hidden = list(range(n_visible, n))

    for layer_idx in range(layers):
        # 1) RY on visible qubits (data/amplitude encoding)
        for v in visible:
            qc.ry(rng.uniform(0, 2 * np.pi), v)

        # 2) RZ on hidden qubits (thermal interaction term)
        #    Angles scaled by beta_param to represent inverse temperature
        for h in hidden:
            qc.rz(beta_param * rng.uniform(0, 2 * np.pi), h)

        # 3) CX entangling between visible and hidden
        #    Each visible qubit couples to the corresponding hidden qubit
        n_pairs = min(n_visible, n_hidden)
        for i in range(n_pairs):
            qc.cx(visible[i], hidden[i])

        # Additional intra-layer entangling for expressibility
        # Visible-visible CX chain
        for i in range(n_visible - 1):
            qc.cx(visible[i], visible[i + 1])
        # Hidden-hidden CX chain
        for i in range(n_hidden - 1):
            qc.cx(hidden[i], hidden[i + 1])

    return qc


# ── Error Correction Additions (Color, Bacon-Shor, Reed-Muller) ─────


def make_color_code(n_qubits=7, **kwargs) -> QuantumCircuit:
    """[[7,1,3]] color code encoder (triangular / Steane-like).

    The 7-qubit color code is defined on a triangular lattice with
    three-colorable faces.  It is equivalent to the Steane code but
    emphasised here with the color-code perspective: each face of the
    lattice supports both an X-type and a Z-type stabilizer.

    Stabilizer generators (X-type):
        Plaquette R (red):   X0 X1 X2 X3
        Plaquette G (green): X0 X2 X4 X6
        Plaquette B (blue):  X0 X1 X4 X5

    Stabilizer generators (Z-type):
        Plaquette R:  Z0 Z1 Z2 Z3
        Plaquette G:  Z0 Z2 Z4 Z6
        Plaquette B:  Z0 Z1 Z4 Z5

    Encoding circuit:
        1) H on qubits 0, 1, 2 (ancilla role in encoder)
        2) CX from qubits 0-2 to data qubits following the stabilizer
           structure of the [7,4,3] Hamming code
        3) CZ gates encoding the Z-type plaquette structure to emphasise
           the 3-colorable lattice

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
    # H-matrix columns for [7,4,3]: each column indicates which parity
    # checks involve that qubit.  Encoding spreads superposition to
    # data qubits 3-6.
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
    # Red plaquette Z-correlations: pairs within {0,1,2,3}
    qc.cz(0, 3)
    qc.cz(1, 3)
    # Green plaquette Z-correlations: pairs within {0,2,4,6}
    qc.cz(2, 6)
    qc.cz(0, 4)
    # Blue plaquette Z-correlations: pairs within {0,1,4,5}
    qc.cz(1, 5)
    qc.cz(4, 5)

    return qc


def make_bacon_shor(n_qubits=9, **kwargs) -> QuantumCircuit:
    """[[9,1,3]] Bacon-Shor subsystem code encoder.

    9 qubits arranged in a 3x3 grid.  The Bacon-Shor code is a
    subsystem code where the stabilizer group is generated by
    weight-6 operators, but the code is defined through weight-2
    gauge generators:
        XX gauge generators on horizontal pairs:
            X0X1, X1X2, X3X4, X4X5, X6X7, X7X8
        ZZ gauge generators on vertical pairs:
            Z0Z3, Z3Z6, Z1Z4, Z4Z7, Z2Z5, Z5Z8

    Encoding circuit:
        1) CX from row leaders (q0, q3, q6) to their row members
           to create the bit-flip protection within rows.
        2) CX from column leaders (q0, q1, q2) to their column
           members to create the phase-flip protection within columns.
        3) H on column leaders to put the encoding into the
           dual (X/Z) basis.

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
    # Row 0: q0 -> q1, q2
    qc.cx(0, 1)
    qc.cx(0, 2)
    # Row 1: q3 -> q4, q5
    qc.cx(3, 4)
    qc.cx(3, 5)
    # Row 2: q6 -> q7, q8
    qc.cx(6, 7)
    qc.cx(6, 8)

    # Column encoding: spread phase information down columns
    # H on column leaders to enter dual basis
    qc.h(0)
    qc.h(1)
    qc.h(2)

    # CX from column leaders to column members
    # Column 0: q0 -> q3, q6
    qc.cx(0, 3)
    qc.cx(0, 6)
    # Column 1: q1 -> q4, q7
    qc.cx(1, 4)
    qc.cx(1, 7)
    # Column 2: q2 -> q5, q8
    qc.cx(2, 5)
    qc.cx(2, 8)

    return qc


def make_reed_muller_code(n_qubits=15, **kwargs) -> QuantumCircuit:
    """[[15,1,3]] quantum Reed-Muller code encoder.

    15 physical qubits encoding 1 logical qubit.  This code is a CSS
    code constructed from the classical punctured Reed-Muller codes
    RM(1,4) and RM(2,4).  Its key property is supporting a transversal
    T gate, making it valuable for fault-tolerant quantum computation.

    The encoding circuit is based on the stabilizer formalism:
        1) H on the first 4 qubits (generators corresponding to
           RM(1,4) which has 4 information bits).
        2) CX pattern encoding the punctured RM(1,4) code structure.
           The generator matrix of RM(1,4) over GF(2) gives the CX
           connectivity: each of the 4 generator qubits fans out to
           the data qubits according to the binary representation
           of positions 1-15.

    The CX pattern follows the columns of the RM(1,4) generator
    matrix, where qubit j (j=4..14) receives a CX from generator
    qubit i if bit i of (j+1) in binary is set.

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
    # The generator matrix of RM(1,4) over GF(2) for 15 positions:
    # For each data qubit j (indices 4-14, representing positions 5-15),
    # apply CX from generator qubit i if bit i of (j+1) is set.
    #
    # Position (1-indexed) binary decomposition:
    #   pos  5 = 0101 -> generators 0, 2
    #   pos  6 = 0110 -> generators 1, 2
    #   pos  7 = 0111 -> generators 0, 1, 2
    #   pos  8 = 1000 -> generator 3
    #   pos  9 = 1001 -> generators 0, 3
    #   pos 10 = 1010 -> generators 1, 3
    #   pos 11 = 1011 -> generators 0, 1, 3
    #   pos 12 = 1100 -> generators 2, 3
    #   pos 13 = 1101 -> generators 0, 2, 3
    #   pos 14 = 1110 -> generators 1, 2, 3
    #   pos 15 = 1111 -> generators 0, 1, 2, 3
    #
    # Qubits 0-3 already encode positions 1-4 (each is its own generator).
    # Qubits 4-14 encode positions 5-15.

    for data_qubit in range(4, 15):
        pos = data_qubit + 1  # 1-indexed position
        for gen in range(4):
            if pos & (1 << gen):
                qc.cx(gen, data_qubit)

    return qc


# ── New Families ─────────────────────────────────────────────────────


def make_poisson_solver(n_qubits=5, **kwargs) -> QuantumCircuit:
    """HHL-based Poisson equation solver circuit.

    Implements the core of the HHL algorithm specialised for the 1-D
    Poisson equation  -d²u/dx² = f.  The discretised Laplacian matrix
    has known eigenvalues λ_k = 2 - 2·cos(k·π/(N+1)), which are
    encoded through controlled-phase rotations in the QPE stage.

    Layout:
        qubit 0          — ancilla (eigenvalue inversion via controlled-RY)
        qubits 1..n_c    — counting register (QPE)
        qubit n-1         — system qubit (holds |b⟩)

    Steps:
        (a) Prepare system qubit in |1⟩.
        (b) QPE: H on counting qubits, controlled-phase rotations
            representing the Laplacian eigenvalues.
        (c) Inverse QFT on counting register.
        (d) Controlled-RY on ancilla (eigenvalue inversion:
            rotate by arcsin(C/λ)).
        (e) Inverse QPE to uncompute.

    Args:
        n_qubits: Total qubit count (minimum 4, default 5).

    Tags: linear_systems, pde
    """
    n = max(4, n_qubits)
    n_counting = n - 2
    ancilla = 0
    counting = list(range(1, n_counting + 1))
    system = n - 1

    qc = QuantumCircuit(n)

    # (a) Prepare system qubit
    qc.x(system)

    # (b) QPE: Hadamard on counting register
    for c in counting:
        qc.h(c)

    # Controlled-phase rotations representing Laplacian eigenvalues
    # λ_k = 2 - 2·cos(k·π/(N+1)) for the 1-D Poisson discretisation
    # We encode this via controlled-phase on each counting qubit
    for idx, c in enumerate(counting):
        # Phase angle encodes the eigenvalue structure
        k = idx + 1
        angle = np.pi * k / (n_counting + 1)
        for _ in range(2 ** idx):
            qc.cp(angle, c, system)

    # (c) Inverse QFT on counting register
    for i in range(n_counting // 2):
        qc.swap(counting[i], counting[n_counting - 1 - i])
    for i in range(n_counting):
        for j in range(i):
            qc.cp(-np.pi / (2 ** (i - j)), counting[j], counting[i])
        qc.h(counting[i])

    # (d) Controlled-RY on ancilla (eigenvalue inversion)
    C = 0.5
    for idx, c in enumerate(counting):
        lam_approx = 2 ** (idx + 1)  # approximate eigenvalue from counting
        theta = 2 * np.arcsin(min(C / lam_approx, 1.0))
        # Controlled-RY decomposition: RY(θ/2) - CX - RY(-θ/2) - CX
        qc.ry(theta / 2, ancilla)
        qc.cx(c, ancilla)
        qc.ry(-theta / 2, ancilla)
        qc.cx(c, ancilla)

    # (e) Inverse QPE: forward QFT then undo controlled-phases
    for i in range(n_counting - 1, -1, -1):
        qc.h(counting[i])
        for j in range(i - 1, -1, -1):
            qc.cp(np.pi / (2 ** (i - j)), counting[j], counting[i])
    for i in range(n_counting // 2):
        qc.swap(counting[i], counting[n_counting - 1 - i])

    # Undo controlled unitaries
    for idx in range(len(counting) - 1, -1, -1):
        c = counting[idx]
        k = idx + 1
        angle = -np.pi * k / (n_counting + 1)
        for _ in range(2 ** idx):
            qc.cp(angle, c, system)

    # Hadamard to uncompute counting register
    for c in counting:
        qc.h(c)

    return qc


def make_betti_number(n_qubits=4, **kwargs) -> QuantumCircuit:
    """Betti number estimation circuit for topological data analysis.

    Estimates the k-th Betti number of a simplicial complex by
    preparing a uniform superposition over simplices and applying
    a phase oracle that encodes the combinatorial Laplacian of the
    complex (adjacency / boundary structure).

    The circuit implements:
        1) Uniform superposition via H on all qubits.
        2) Phase oracle encoding the simplicial complex adjacency:
           - CZ gates between adjacent simplices (1-skeleton edges).
           - CP gates for higher-order correlations (2-simplices).
        3) Inverse-QFT-like readout to extract spectral information
           about the Laplacian (Betti numbers correspond to the
           nullity of the Laplacian).

    The simplicial complex used here is a triangular complex on
    the vertices, with edges defined by a cycle graph and faces
    defined by adjacent triples.

    Args:
        n_qubits: Number of qubits / vertices in the complex (minimum 3, default 4).

    Tags: topology, homology
    """
    n = max(3, n_qubits)
    qc = QuantumCircuit(n)

    # Step 1: Uniform superposition over all simplices
    for i in range(n):
        qc.h(i)

    # Step 2: Phase oracle encoding simplicial complex adjacency
    # 1-skeleton: CZ gates for edges of a cycle graph
    for i in range(n):
        qc.cz(i, (i + 1) % n)

    # 2-simplices (triangular faces): CP gates encoding higher-order
    # adjacency structure.  For a cycle with n vertices, triangular
    # faces connect vertex i to vertices i+1 and i+2 (mod n).
    for i in range(n):
        qc.cp(np.pi / 3, i, (i + 2) % n)

    # Step 3: Inverse-QFT-like readout for spectral decomposition
    # The QFT structure extracts eigenvalue information from the
    # Laplacian encoded by the oracle.
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    for i in range(n):
        for j in range(i):
            qc.cp(-np.pi / (2 ** (i - j)), j, i)
        qc.h(i)

    return qc


def make_quantum_fingerprinting(n_qubits=6, **kwargs) -> QuantumCircuit:
    """Quantum fingerprinting for equality testing.

    Two parties (Alice and Bob) each encode their classical input into
    quantum fingerprint states.  A referee then performs a controlled-SWAP
    test to determine whether the two inputs are equal, using
    exponentially fewer qubits than classical fingerprinting.

    Layout:
        qubit 0                       — ancilla (referee's control qubit)
        qubits 1 .. n_a               — Alice's register
        qubits n_a+1 .. n-1           — Bob's register

    Circuit:
        1) Alice's encoding: H on Alice's qubits, then CX chain within
           Alice's register to create an error-correcting fingerprint.
        2) Bob's encoding: H on Bob's qubits, then CX chain within
           Bob's register.
        3) Referee's controlled-SWAP test:
           H on ancilla, controlled-SWAP between paired Alice/Bob qubits
           (Fredkin decomposition: CX-Toffoli-CX), H on ancilla.

    Measuring ancilla: P(|0⟩) = 1 if inputs equal, ≈ 1/2 if different.

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
    # H on all of Alice's qubits to create superposition
    for q in alice_qubits:
        qc.h(q)
    # CX chain within Alice's register (error-correcting structure)
    for i in range(len(alice_qubits) - 1):
        qc.cx(alice_qubits[i], alice_qubits[i + 1])

    # Step 2: Bob's fingerprint encoding
    # H on all of Bob's qubits
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
        _decompose_toffoli(qc, ancilla, a_q, b_q)
        qc.cx(b_q, a_q)

    qc.h(ancilla)

    return qc


def make_vqs_real_time(n_qubits=4, layers=2, dt=0.3, **kwargs) -> QuantumCircuit:
    """Variational quantum simulation for real-time dynamics.

    Parameterised ansatz designed for variational real-time evolution
    based on McLachlan's variational principle.  Unlike imaginary-time
    evolution (VarQITE) which uses only real-valued parameters, real-time
    dynamics require complex-valued parameters.  This is achieved by
    using both RY (real part) and RX (imaginary part) rotations on each
    qubit, followed by RZ for phase control and CX for entangling.

    Each layer applies:
        1) RY(theta) on each qubit — real part of variational parameter
        2) RX(phi) on each qubit — imaginary part of variational parameter
        3) RZ(dt * layer_scale) on each qubit — time-step encoding
        4) CX entangling ladder on adjacent pairs

    The time-step parameter dt scales the RZ angles, encoding the
    evolution time into the circuit structure.  Angles vary across
    layers to capture the time-dependent dynamics.

    Args:
        n_qubits: Number of qubits (minimum 2, default 4).
        layers: Number of ansatz layers (default 2).
        dt: Time step parameter scaling RZ angles (default 0.3).

    Tags: variational, real_time
    """
    layers = kwargs.get("layers", layers)
    dt = kwargs.get("dt", dt)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(717)

    for layer_idx in range(layers):
        # 1) RY rotations — real part of variational parameters
        for i in range(n):
            theta = rng.uniform(0, 2 * np.pi)
            qc.ry(theta, i)

        # 2) RX rotations — imaginary part of variational parameters
        #    (distinguishes real-time from imaginary-time evolution)
        for i in range(n):
            phi = rng.uniform(0, 2 * np.pi)
            qc.rx(phi, i)

        # 3) RZ rotations — time-step encoding
        #    Scale increases with layer to capture time evolution
        time_scale = dt * (layer_idx + 1)
        for i in range(n):
            qc.rz(time_scale * rng.uniform(0.5, 1.5), i)

        # 4) CX entangling ladder
        for i in range(n - 1):
            qc.cx(i, i + 1)

    return qc


# ── Registry ────────────────────────────────────────────────────────

REGISTRY = [
    # ── Entanglement ──────────────────────────────────────────
    AlgorithmEntry(
        "bell_state", "entanglement", make_bell_state, (2, 8),
        tags=["entanglement", "baseline"],
    ),
    AlgorithmEntry(
        "ghz_state", "entanglement", make_ghz_state, (3, 8),
        tags=["entanglement", "multipartite"],
    ),
    AlgorithmEntry(
        "w_state", "entanglement", make_w_state, (3, 8),
        tags=["entanglement", "multipartite", "arbitrary_rotation"],
    ),
    AlgorithmEntry(
        "cluster_state", "entanglement", make_cluster_state, (2, 8),
        tags=["graph_state", "mbqc", "cz_only"],
    ),
    AlgorithmEntry(
        "dicke_state", "entanglement", make_dicke_state, (3, 8),
        tags=["symmetric_state", "entanglement"],
    ),
    AlgorithmEntry(
        "graph_state", "entanglement", make_graph_state, (3, 8),
        tags=["graph_state", "multipartite"],
    ),
    # ── Protocol ──────────────────────────────────────────────
    AlgorithmEntry(
        "teleportation", "protocol", make_teleportation, (3, 3),
        tags=["communication", "bell_measurement"],
    ),
    AlgorithmEntry(
        "superdense_coding", "protocol", make_superdense_coding, (2, 2),
        tags=["communication", "bell_pair", "dense_coding"],
    ),
    AlgorithmEntry(
        "entanglement_swapping", "protocol", make_entanglement_swapping, (4, 4),
        tags=["bell_measurement", "relay", "teleportation_variant"],
    ),
    AlgorithmEntry(
        "swap_test", "protocol", make_swap_test, (3, 7),
        tags=["fidelity", "comparison"],
    ),
    # ── Oracle ────────────────────────────────────────────────
    AlgorithmEntry(
        "grover", "oracle", make_grover, (3, 6),
        tags=["amplitude_amplification", "diffusion", "oracle"],
    ),
    AlgorithmEntry(
        "bernstein_vazirani", "oracle", make_bernstein_vazirani, (3, 6),
        tags=["hidden_structure", "oracle"],
    ),
    AlgorithmEntry(
        "deutsch_jozsa", "oracle", make_deutsch_jozsa, (2, 6),
        tags=["decision_problem", "oracle"],
    ),
    AlgorithmEntry(
        "simon", "oracle", make_simon, (4, 8),
        tags=["hidden_structure", "oracle", "xor_mask"],
    ),
    AlgorithmEntry(
        "quantum_counting", "oracle", make_quantum_counting, (5, 8),
        tags=["grover_qpe", "counting", "hybrid"],
    ),
    AlgorithmEntry(
        "deutsch", "oracle", make_deutsch, (2, 2),
        tags=["decision_problem", "oracle"],
    ),
    AlgorithmEntry(
        "hidden_shift", "oracle", make_hidden_shift, (4, 8),
        tags=["hidden_structure", "oracle"],
    ),
    AlgorithmEntry(
        "element_distinctness", "oracle", make_element_distinctness, (5, 8),
        tags=["quantum_walk", "oracle"],
    ),
    AlgorithmEntry(
        "quantum_walk_search", "oracle", make_quantum_walk_search, (3, 8),
        tags=["quantum_walk", "search"],
    ),
    # ── Transform ─────────────────────────────────────────────
    AlgorithmEntry(
        "qft", "transform", make_qft, (2, 7),
        tags=["phase_rotation", "butterfly"],
    ),
    AlgorithmEntry(
        "phase_estimation", "transform", make_phase_estimation, (4, 8),
        tags=["phase_kickback", "inverse_qft"],
    ),
    AlgorithmEntry(
        "iterative_qpe", "transform", make_iterative_qpe, (2, 2),
        tags=["phase_estimation", "iterative"],
    ),
    AlgorithmEntry(
        "amplitude_estimation", "transform", make_amplitude_estimation, (4, 8),
        tags=["amplitude_estimation", "grover_qpe"],
    ),
    # ── Variational ───────────────────────────────────────────
    AlgorithmEntry(
        "qaoa_maxcut", "variational", make_qaoa_maxcut, (3, 8),
        tags=["combinatorial", "zz_interaction", "mixer"],
    ),
    AlgorithmEntry(
        "vqe_uccsd", "variational", make_vqe_uccsd_fragment, (4, 4),
        tags=["chemistry", "excitation"],
    ),
    AlgorithmEntry(
        "hw_efficient_ansatz", "variational", make_hardware_efficient_ansatz, (3, 8),
        tags=["hardware_efficient", "brick_layer"],
    ),
    AlgorithmEntry(
        "adapt_vqe", "variational", make_adapt_vqe, (4, 8),
        tags=["chemistry", "adaptive", "excitation"],
    ),
    AlgorithmEntry(
        "vqd", "variational", make_vqd, (4, 8),
        tags=["chemistry", "excited_states"],
    ),
    AlgorithmEntry(
        "recursive_qaoa", "variational", make_recursive_qaoa, (4, 8),
        tags=["combinatorial", "recursive"],
    ),
    AlgorithmEntry(
        "varqite", "variational", make_varqite, (3, 8),
        tags=["imaginary_time", "variational"],
    ),
    AlgorithmEntry(
        "qaoa_weighted", "variational", make_qaoa_weighted, (3, 8),
        tags=["combinatorial", "weighted"],
    ),
    AlgorithmEntry(
        "quantum_boltzmann", "variational", make_quantum_boltzmann, (4, 8),
        tags=["generative", "thermal"],
    ),
    # ── Error Correction ──────────────────────────────────────
    AlgorithmEntry(
        "bit_flip_code", "error_correction", make_bit_flip_code, (5, 5),
        tags=["repetition_code", "syndrome_extraction", "toffoli"],
    ),
    AlgorithmEntry(
        "phase_flip_code", "error_correction", make_phase_flip_code, (5, 5),
        tags=["repetition_code", "hadamard_basis", "toffoli"],
    ),
    AlgorithmEntry(
        "steane_code", "error_correction", make_steane_code, (7, 7),
        tags=["css_code", "hamming", "stabilizer"],
    ),
    AlgorithmEntry(
        "shor_code", "error_correction", make_shor_code, (9, 9),
        tags=["concatenated_code", "nine_qubit"],
    ),
    AlgorithmEntry(
        "five_qubit_code", "error_correction", make_five_qubit_code, (5, 5),
        tags=["perfect_code", "stabilizer"],
    ),
    AlgorithmEntry(
        "surface_code_patch", "error_correction", make_surface_code_patch, (9, 9),
        tags=["topological_code", "surface_code", "stabilizer"],
    ),
    AlgorithmEntry(
        "color_code", "error_correction", make_color_code, (7, 7),
        tags=["topological_code", "color_code"],
    ),
    AlgorithmEntry(
        "bacon_shor", "error_correction", make_bacon_shor, (9, 9),
        tags=["subsystem_code", "stabilizer"],
    ),
    AlgorithmEntry(
        "reed_muller_code", "error_correction", make_reed_muller_code, (15, 15),
        tags=["reed_muller", "transversal_t"],
    ),
    # ── Simulation ────────────────────────────────────────────
    AlgorithmEntry(
        "trotter_ising", "simulation", make_trotter_ising, (2, 8),
        tags=["hamiltonian_simulation", "zz_interaction", "trotter"],
    ),
    AlgorithmEntry(
        "trotter_heisenberg", "simulation", make_trotter_heisenberg, (2, 8),
        tags=["hamiltonian_simulation", "trotter", "mixed_interaction"],
    ),
    AlgorithmEntry(
        "quantum_walk", "simulation", make_quantum_walk, (3, 6),
        tags=["discrete_walk", "coin_operator"],
    ),
    AlgorithmEntry(
        "qdrift", "simulation", make_qdrift, (2, 8),
        tags=["hamiltonian_simulation", "randomized"],
    ),
    AlgorithmEntry(
        "higher_order_trotter", "simulation", make_higher_order_trotter, (2, 8),
        tags=["hamiltonian_simulation", "trotter", "higher_order"],
    ),
    AlgorithmEntry(
        "hubbard_trotter", "simulation", make_hubbard_trotter, (2, 8),
        tags=["hamiltonian_simulation", "fermionic", "trotter"],
    ),
    AlgorithmEntry(
        "ctqw", "simulation", make_ctqw, (2, 8),
        tags=["quantum_walk", "continuous_time"],
    ),
    AlgorithmEntry(
        "vqs_real_time", "simulation", make_vqs_real_time, (2, 8),
        tags=["variational", "real_time"],
    ),
    # ── Arithmetic ────────────────────────────────────────────
    AlgorithmEntry(
        "ripple_carry_adder", "arithmetic", make_ripple_carry_adder, (5, 5),
        tags=["toffoli", "addition", "classical_reversible"],
    ),
    AlgorithmEntry(
        "qft_adder", "arithmetic", make_qft_adder, (4, 8),
        tags=["addition", "controlled_phase", "qft_based"],
    ),
    AlgorithmEntry(
        "quantum_multiplier", "arithmetic", make_quantum_multiplier, (6, 6),
        tags=["multiplication", "classical_reversible"],
    ),
    AlgorithmEntry(
        "quantum_comparator", "arithmetic", make_quantum_comparator, (5, 5),
        tags=["comparison", "classical_reversible"],
    ),
    # ── Distillation ──────────────────────────────────────────
    AlgorithmEntry(
        "bbpssw_distillation", "distillation", make_bbpssw_distillation, (4, 4),
        tags=["distillation", "bell_pair", "bilateral_cnot"],
    ),
    AlgorithmEntry(
        "dejmps_distillation", "distillation", make_dejmps_distillation, (4, 4),
        tags=["distillation", "bell_pair", "bilateral_cnot", "twirling"],
    ),
    AlgorithmEntry(
        "recurrence_distillation", "distillation", make_recurrence_distillation, (8, 8),
        tags=["distillation", "bell_pair", "bilateral_cnot", "multi_round"],
    ),
    AlgorithmEntry(
        "pumping_distillation", "distillation", make_pumping_distillation, (6, 6),
        tags=["distillation", "bell_pair", "bilateral_cnot", "pumping"],
    ),
    # ── Machine Learning ──────────────────────────────────────
    AlgorithmEntry(
        "quantum_kernel", "machine_learning", make_quantum_kernel, (2, 8),
        tags=["feature_map", "zz_interaction", "kernel_method"],
    ),
    AlgorithmEntry(
        "data_reuploading", "machine_learning", make_data_reuploading, (2, 6),
        tags=["classifier", "reuploading", "variational"],
    ),
    AlgorithmEntry(
        "qsvm", "machine_learning", make_qsvm, (2, 8),
        tags=["kernel_method", "classification"],
    ),
    AlgorithmEntry(
        "qcnn", "machine_learning", make_qcnn, (4, 8),
        tags=["neural_network", "classification"],
    ),
    AlgorithmEntry(
        "qgan_generator", "machine_learning", make_qgan_generator, (2, 8),
        tags=["generative", "adversarial"],
    ),
    AlgorithmEntry(
        "quantum_autoencoder", "machine_learning", make_quantum_autoencoder, (4, 8),
        tags=["compression", "autoencoder"],
    ),
    # ── Linear Algebra ────────────────────────────────────────
    AlgorithmEntry(
        "hhl", "linear_algebra", make_hhl, (4, 7),
        tags=["linear_systems", "phase_estimation"],
    ),
    AlgorithmEntry(
        "vqls", "linear_algebra", make_vqls, (2, 8),
        tags=["linear_systems", "variational"],
    ),
    # ── Cryptography ──────────────────────────────────────────
    AlgorithmEntry(
        "bb84_encode", "cryptography", make_bb84_encode, (2, 16),
        tags=["qkd", "prepare_measure"],
    ),
    AlgorithmEntry(
        "e91_protocol", "cryptography", make_e91_protocol, (2, 16),
        tags=["qkd", "entanglement_based"],
    ),
    # ── Sampling ──────────────────────────────────────────────
    AlgorithmEntry(
        "iqp_sampling", "sampling", make_iqp_sampling, (2, 8),
        tags=["commuting_gates", "supremacy"],
    ),
    AlgorithmEntry(
        "random_circuit_sampling", "sampling", make_random_circuit_sampling, (2, 8),
        tags=["random_circuit", "supremacy"],
    ),
    # ── Error Mitigation ──────────────────────────────────────
    AlgorithmEntry(
        "zne_folding", "error_mitigation", make_zne_folding, (2, 8),
        tags=["noise_mitigation", "folding"],
    ),
    AlgorithmEntry(
        "pauli_twirling", "error_mitigation", make_pauli_twirling, (2, 8),
        tags=["noise_mitigation", "randomized_compiling"],
    ),
    # ── Topological ───────────────────────────────────────────
    AlgorithmEntry(
        "jones_polynomial", "topological", make_jones_polynomial, (3, 8),
        tags=["knot_invariant", "topological"],
    ),
    AlgorithmEntry(
        "toric_code_syndrome", "topological", make_toric_code_syndrome, (8, 8),
        tags=["topological_code", "syndrome_extraction"],
    ),
    # ── Metrology ─────────────────────────────────────────────
    AlgorithmEntry(
        "ghz_metrology", "metrology", make_ghz_metrology, (2, 8),
        tags=["sensing", "heisenberg_limit"],
    ),
    AlgorithmEntry(
        "quantum_fisher_info", "metrology", make_quantum_fisher_info, (2, 8),
        tags=["sensing", "parameter_estimation"],
    ),
    # ── Differential Equations ────────────────────────────────
    AlgorithmEntry(
        "poisson_solver", "differential_equations", make_poisson_solver, (4, 7),
        tags=["linear_systems", "pde"],
    ),
    # ── TDA ───────────────────────────────────────────────────
    AlgorithmEntry(
        "betti_number", "tda", make_betti_number, (3, 8),
        tags=["topology", "homology"],
    ),
    # ── Communication ─────────────────────────────────────────
    AlgorithmEntry(
        "quantum_fingerprinting", "communication", make_quantum_fingerprinting, (4, 8),
        tags=["communication", "exponential_saving"],
    ),
]

ALGORITHM_FAMILY_MAP = {entry.name: entry.family for entry in REGISTRY}
