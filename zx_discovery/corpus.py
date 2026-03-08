"""
Step 1 - Quantum Algorithm Corpus
==================================
Implements a diverse set of well-known quantum algorithms in Qiskit,
then exports each as an OpenQASM 2.0 string for downstream conversion
to ZX-diagrams via PyZX.

Every builder returns a *unitary-only* QuantumCircuit (no measurements
or classical registers) so that the circuit maps cleanly to a
ZX-diagram.  Measurements are stripped because PyZX's QASM importer
ignores classical operations anyway.

Gate Palette
------------
All circuits are built from the gates that both Qiskit's QASM exporter
and PyZX's QASM importer handle reliably:
    h, x, z, s, t, sdg, tdg, cx, cz, ccx, rz, rx, ry, swap
"""

from __future__ import annotations
import math
from typing import Dict, Callable
from qiskit.circuit import QuantumCircuit


# ---------------------------------------------------------------------------
# Individual algorithm builders
# ---------------------------------------------------------------------------

def build_bell_state(n_pairs: int = 2) -> QuantumCircuit:
    """Create n entangled Bell pairs: |Φ+⟩^{⊗n}."""
    qc = QuantumCircuit(2 * n_pairs, name="bell_state")
    for i in range(n_pairs):
        qc.h(2 * i)
        qc.cx(2 * i, 2 * i + 1)
    return qc


def build_ghz(n: int = 4) -> QuantumCircuit:
    """n-qubit GHZ state: (|0…0⟩ + |1…1⟩)/√2."""
    qc = QuantumCircuit(n, name="ghz")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def build_deutsch_jozsa(n: int = 3) -> QuantumCircuit:
    """Deutsch-Jozsa for a balanced oracle on n input qubits + 1 ancilla."""
    total = n + 1
    qc = QuantumCircuit(total, name="deutsch_jozsa")
    # Prepare ancilla in |−⟩
    qc.x(n)
    qc.h(n)
    # Hadamard on inputs
    for i in range(n):
        qc.h(i)
    # Balanced oracle: CNOT from each input to ancilla
    for i in range(n):
        qc.cx(i, n)
    # Hadamard on inputs
    for i in range(n):
        qc.h(i)
    return qc


def build_bernstein_vazirani(secret: int = 0b1011, n: int = 4) -> QuantumCircuit:
    """Bernstein-Vazirani with a given secret string."""
    total = n + 1
    qc = QuantumCircuit(total, name="bernstein_vazirani")
    qc.x(n)
    qc.h(n)
    for i in range(n):
        qc.h(i)
    # Oracle
    for i in range(n):
        if (secret >> i) & 1:
            qc.cx(i, n)
    for i in range(n):
        qc.h(i)
    return qc


def _qft_rotations(qc: QuantumCircuit, n: int, start: int = 0):
    """Apply QFT rotations to qubits [start, start+n)."""
    if n == 0:
        return
    n -= 1
    qc.h(start + n)
    for qubit in range(n):
        angle = math.pi / (2 ** (n - qubit))
        # Controlled-Rz decomposed as CX-Rz-CX
        qc.cx(start + qubit, start + n)
        qc.rz(-angle / 2, start + n)
        qc.cx(start + qubit, start + n)
        qc.rz(angle / 2, start + n)
    _qft_rotations(qc, n, start)


def build_qft(n: int = 4) -> QuantumCircuit:
    """Quantum Fourier Transform on n qubits (unitary only)."""
    qc = QuantumCircuit(n, name="qft")
    _qft_rotations(qc, n)
    # Swap to match standard QFT ordering
    for i in range(n // 2):
        qc.swap(i, n - i - 1)
    return qc


def build_inverse_qft(n: int = 4) -> QuantumCircuit:
    """Inverse QFT on n qubits."""
    qc = build_qft(n).inverse()
    qc.name = "iqft"
    return qc


def build_grover_diffusion(n: int = 3) -> QuantumCircuit:
    """Grover diffusion operator on n qubits."""
    qc = QuantumCircuit(n, name="grover_diffusion")
    for i in range(n):
        qc.h(i)
        qc.x(i)
    # Multi-controlled Z
    if n == 2:
        qc.cz(0, 1)
    elif n == 3:
        qc.h(2)
        qc.ccx(0, 1, 2)
        qc.h(2)
    else:
        # General: phase kickback via ancilla-free multi-CZ
        qc.h(n - 1)
        # Build a ladder of Toffolis
        for i in range(n - 2):
            qc.ccx(i, i + 1, min(i + 2, n - 1))
        qc.h(n - 1)
        for i in reversed(range(n - 2)):
            qc.ccx(i, i + 1, min(i + 2, n - 1))
    for i in range(n):
        qc.x(i)
        qc.h(i)
    return qc


def build_grover_single_oracle(n: int = 3, target: int = 5) -> QuantumCircuit:
    """Grover's algorithm (single iteration) marking |target⟩."""
    qc = QuantumCircuit(n, name="grover_oracle")
    # Superposition
    for i in range(n):
        qc.h(i)
    # Oracle: flip |target⟩
    for i in range(n):
        if not ((target >> i) & 1):
            qc.x(i)
    # Multi-controlled Z
    if n == 3:
        qc.h(2)
        qc.ccx(0, 1, 2)
        qc.h(2)
    else:
        qc.cz(0, n - 1)
    for i in range(n):
        if not ((target >> i) & 1):
            qc.x(i)
    # Diffusion
    qc = qc.compose(build_grover_diffusion(n))
    qc.name = "grover_oracle"
    return qc


def build_teleportation_unitary() -> QuantumCircuit:
    """Unitary part of quantum teleportation (Bell basis measurement prep)."""
    qc = QuantumCircuit(3, name="teleport_unitary")
    # Prepare Bell pair on qubits 1,2
    qc.h(1)
    qc.cx(1, 2)
    # Bell-basis rotation on qubits 0,1
    qc.cx(0, 1)
    qc.h(0)
    return qc


def build_quantum_walk_step(n: int = 4) -> QuantumCircuit:
    """One step of a discrete quantum walk on a cycle of 2^n nodes.
    Uses a coin qubit (qubit 0) and n position qubits."""
    total = n + 1
    qc = QuantumCircuit(total, name="quantum_walk")
    # Coin flip
    qc.h(0)
    # Conditional increment (coin=|1⟩ → shift right)
    for i in range(1, total):
        qc.cx(0, i)
    # Coin flip again
    qc.h(0)
    # Conditional decrement (coin=|0⟩ → shift left)
    qc.x(0)
    for i in range(1, total):
        qc.cx(0, i)
    qc.x(0)
    return qc


def build_phase_estimation_core(n_counting: int = 3) -> QuantumCircuit:
    """Core of phase estimation: Hadamards + controlled rotations + inverse QFT.
    The unitary U is a simple T-gate applied to one target qubit."""
    total = n_counting + 1  # counting qubits + 1 target
    qc = QuantumCircuit(total, name="phase_estimation")
    # Hadamard on counting register
    for i in range(n_counting):
        qc.h(i)
    # Controlled-U^{2^k}
    target = n_counting
    for k in range(n_counting):
        repetitions = 2 ** k
        for _ in range(repetitions):
            # Controlled-T on target, controlled by qubit k
            qc.cx(k, target)
            qc.rz(math.pi / 4, target)
            qc.cx(k, target)
            qc.rz(-math.pi / 4, target)
            qc.rz(math.pi / 4, k)  # phase correction on control
    # Inverse QFT on counting register
    for i in range(n_counting // 2):
        qc.swap(i, n_counting - i - 1)
    for j in range(n_counting):
        for k in range(j):
            angle = -math.pi / (2 ** (j - k))
            qc.cx(k, j)
            qc.rz(-angle / 2, j)
            qc.cx(k, j)
            qc.rz(angle / 2, j)
        qc.h(j)
    return qc


def build_swap_test() -> QuantumCircuit:
    """SWAP test circuit for comparing two single-qubit states."""
    qc = QuantumCircuit(3, name="swap_test")
    qc.h(0)
    # Controlled-SWAP (Fredkin) via Toffoli decomposition
    qc.cx(2, 1)
    qc.ccx(0, 1, 2)
    qc.cx(2, 1)
    qc.h(0)
    return qc


def build_variational_ansatz(n: int = 4, depth: int = 2) -> QuantumCircuit:
    """Hardware-efficient variational ansatz (RY-CX layers)."""
    qc = QuantumCircuit(n, name="variational_ansatz")
    param_idx = 0
    for d in range(depth):
        for i in range(n):
            angle = 0.1 * (param_idx + 1)  # fixed proxy for parameters
            qc.ry(angle, i)
            param_idx += 1
        for i in range(0, n - 1, 2):
            qc.cx(i, i + 1)
        for i in range(1, n - 1, 2):
            qc.cx(i, i + 1)
    # Final rotation layer
    for i in range(n):
        qc.ry(0.1 * (param_idx + 1), i)
        param_idx += 1
    return qc


def build_quantum_adder() -> QuantumCircuit:
    """2-bit quantum ripple-carry adder (Cuccaro style, simplified)."""
    # a0,a1 = qubits 0,1; b0,b1 = qubits 2,3; carry_out = qubit 4
    qc = QuantumCircuit(5, name="quantum_adder")
    # Half-add bit 0
    qc.ccx(0, 2, 4)  # carry
    qc.cx(0, 2)      # sum in b0
    # Full-add bit 1
    qc.ccx(1, 3, 4)
    qc.cx(1, 3)
    return qc


def build_w_state(n: int = 3) -> QuantumCircuit:
    """Prepare the W state |W_n⟩ = (|100..0⟩ + |010..0⟩ + … + |00..01⟩)/√n."""
    qc = QuantumCircuit(n, name="w_state")
    qc.x(0)
    for i in range(n - 1):
        theta = 2 * math.acos(math.sqrt(1 / (n - i)))
        qc.ry(theta, i + 1)
        qc.cx(i + 1, i)
        qc.cx(i, i + 1)
    return qc


def build_toffoli_decomposed() -> QuantumCircuit:
    """Standard Toffoli decomposition into 1- and 2-qubit gates."""
    qc = QuantumCircuit(3, name="toffoli_decomp")
    qc.h(2)
    qc.cx(1, 2)
    qc.tdg(2)
    qc.cx(0, 2)
    qc.t(2)
    qc.cx(1, 2)
    qc.tdg(2)
    qc.cx(0, 2)
    qc.t(1)
    qc.t(2)
    qc.h(2)
    qc.cx(0, 1)
    qc.t(0)
    qc.tdg(1)
    qc.cx(0, 1)
    return qc


# ---------------------------------------------------------------------------
# Registry & export helpers
# ---------------------------------------------------------------------------

ALGORITHM_BUILDERS: Dict[str, Callable[[], QuantumCircuit]] = {
    "bell_2pair":           lambda: build_bell_state(2),
    "bell_3pair":           lambda: build_bell_state(3),
    "ghz_4":               lambda: build_ghz(4),
    "ghz_5":               lambda: build_ghz(5),
    "deutsch_jozsa_3":     lambda: build_deutsch_jozsa(3),
    "deutsch_jozsa_4":     lambda: build_deutsch_jozsa(4),
    "bernstein_vazirani":  lambda: build_bernstein_vazirani(0b1011, 4),
    "qft_3":               lambda: build_qft(3),
    "qft_4":               lambda: build_qft(4),
    "iqft_4":              lambda: build_inverse_qft(4),
    "grover_3q":           lambda: build_grover_single_oracle(3, 5),
    "grover_diffusion_3":  lambda: build_grover_diffusion(3),
    "grover_diffusion_4":  lambda: build_grover_diffusion(4),
    "teleport_unitary":    build_teleportation_unitary,
    "swap_test":           build_swap_test,
    "phase_est_3":         lambda: build_phase_estimation_core(3),
    "quantum_walk_3":      lambda: build_quantum_walk_step(3),
    "variational_4x2":     lambda: build_variational_ansatz(4, 2),
    "quantum_adder":       build_quantum_adder,
    "w_state_3":           lambda: build_w_state(3),
    "w_state_4":           lambda: build_w_state(4),
    "toffoli_decomp":      build_toffoli_decomposed,
}


def build_corpus() -> Dict[str, QuantumCircuit]:
    """Build every algorithm and return {name: QuantumCircuit}."""
    return {name: builder() for name, builder in ALGORITHM_BUILDERS.items()}


def circuit_to_qasm(qc: QuantumCircuit) -> str:
    """Export a Qiskit QuantumCircuit to OpenQASM 2.0 string."""
    from qiskit import qasm2
    return qasm2.dumps(qc)


def corpus_to_qasm(corpus: Dict[str, QuantumCircuit]) -> Dict[str, str]:
    """Convert entire corpus to QASM strings."""
    out = {}
    for name, qc in corpus.items():
        try:
            out[name] = circuit_to_qasm(qc)
        except Exception as e:
            print(f"  [warn] Could not export '{name}' to QASM: {e}")
    return out
