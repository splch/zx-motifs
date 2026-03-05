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


# ── Registry ────────────────────────────────────────────────────────

REGISTRY = [
    AlgorithmEntry(
        "bell_state", "entanglement", make_bell_state, (2, 8),
        tags=["entanglement", "baseline"],
    ),
    AlgorithmEntry(
        "ghz_state", "entanglement", make_ghz_state, (3, 8),
        tags=["entanglement", "multipartite"],
    ),
    AlgorithmEntry(
        "teleportation", "protocol", make_teleportation, (3, 3),
        tags=["communication", "bell_measurement"],
    ),
    AlgorithmEntry(
        "qft", "transform", make_qft, (2, 7),
        tags=["phase_rotation", "butterfly"],
    ),
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
        "phase_estimation", "transform", make_phase_estimation, (4, 8),
        tags=["phase_kickback", "inverse_qft"],
    ),
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
    # ── Error Correction ───────────────────────────────────────────
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
    # ── Simulation ─────────────────────────────────────────────────
    AlgorithmEntry(
        "trotter_ising", "simulation", make_trotter_ising, (2, 8),
        tags=["hamiltonian_simulation", "zz_interaction", "trotter"],
    ),
    AlgorithmEntry(
        "trotter_heisenberg", "simulation", make_trotter_heisenberg, (2, 8),
        tags=["hamiltonian_simulation", "trotter", "mixed_interaction"],
    ),
    # ── Arithmetic ─────────────────────────────────────────────────
    AlgorithmEntry(
        "ripple_carry_adder", "arithmetic", make_ripple_carry_adder, (5, 5),
        tags=["toffoli", "addition", "classical_reversible"],
    ),
    # ── Additional Oracle ──────────────────────────────────────────
    AlgorithmEntry(
        "simon", "oracle", make_simon, (4, 8),
        tags=["hidden_structure", "oracle", "xor_mask"],
    ),
    # ── Additional Entanglement ────────────────────────────────────
    AlgorithmEntry(
        "w_state", "entanglement", make_w_state, (3, 8),
        tags=["entanglement", "multipartite", "arbitrary_rotation"],
    ),
    AlgorithmEntry(
        "cluster_state", "entanglement", make_cluster_state, (2, 8),
        tags=["graph_state", "mbqc", "cz_only"],
    ),
    # ── Additional Protocol ────────────────────────────────────────
    AlgorithmEntry(
        "superdense_coding", "protocol", make_superdense_coding, (2, 2),
        tags=["communication", "bell_pair", "dense_coding"],
    ),
    # ── Error Correction ──────────────────────────────────
    AlgorithmEntry(
        "shor_code", "error_correction", make_shor_code, (9, 9),
        tags=["concatenated_code", "nine_qubit"],
    ),
    # ── Simulation ────────────────────────────────────────
    AlgorithmEntry(
        "quantum_walk", "simulation", make_quantum_walk, (3, 6),
        tags=["discrete_walk", "coin_operator"],
    ),
    # ── Arithmetic ────────────────────────────────────────
    AlgorithmEntry(
        "qft_adder", "arithmetic", make_qft_adder, (4, 8),
        tags=["addition", "controlled_phase", "qft_based"],
    ),
    # ── Oracle ────────────────────────────────────────────
    AlgorithmEntry(
        "quantum_counting", "oracle", make_quantum_counting, (5, 8),
        tags=["grover_qpe", "counting", "hybrid"],
    ),
    # ── Protocol ──────────────────────────────────────────
    AlgorithmEntry(
        "entanglement_swapping", "protocol", make_entanglement_swapping, (4, 4),
        tags=["bell_measurement", "relay", "teleportation_variant"],
    ),
    # ── Distillation ──────────────────────────────────────────────
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
    # ── Machine Learning ──────────────────────────────────
    AlgorithmEntry(
        "quantum_kernel", "machine_learning", make_quantum_kernel, (2, 8),
        tags=["feature_map", "zz_interaction", "kernel_method"],
    ),
    AlgorithmEntry(
        "data_reuploading", "machine_learning", make_data_reuploading, (2, 6),
        tags=["classifier", "reuploading", "variational"],
    ),
]

ALGORITHM_FAMILY_MAP = {entry.name: entry.family for entry in REGISTRY}
