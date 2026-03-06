"""Oracle family: grover, bernstein_vazirani, deutsch_jozsa, simon, quantum_counting,
deutsch, hidden_shift, element_distinctness, quantum_walk_search."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm


@register_algorithm(
    "grover", "oracle", (3, 6),
    tags=["amplitude_amplification", "diffusion", "oracle"],
)
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


@register_algorithm(
    "bernstein_vazirani", "oracle", (3, 6),
    tags=["hidden_structure", "oracle"],
)
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


@register_algorithm(
    "deutsch_jozsa", "oracle", (2, 6),
    tags=["decision_problem", "oracle"],
)
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


@register_algorithm(
    "simon", "oracle", (4, 8),
    tags=["hidden_structure", "oracle", "xor_mask"],
)
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


@register_algorithm(
    "quantum_counting", "oracle", (5, 8),
    tags=["grover_qpe", "counting", "hybrid"],
)
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


@register_algorithm(
    "deutsch", "oracle", (2, 2),
    tags=["decision_problem", "oracle"],
)
def make_deutsch(n_qubits=2, oracle_type="balanced", **kwargs) -> QuantumCircuit:
    """Deutsch's algorithm -- the simplest quantum oracle algorithm.

    The original Deutsch algorithm for f:{0,1}->{0,1} using 2 qubits:
    qubit 0 = input, qubit 1 = ancilla.  Determines whether f is constant
    or balanced with a single query.

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
        # f(x) = x  ->  CX from input to ancilla
        qc.cx(0, 1)
    # constant-0: identity (no gate needed)
    # Final Hadamard on input qubit
    qc.h(0)
    return qc


@register_algorithm(
    "hidden_shift", "oracle", (4, 8),
    tags=["hidden_structure", "oracle"],
)
def make_hidden_shift(n_qubits=4, shift=None, **kwargs) -> QuantumCircuit:
    """Hidden shift problem circuit.

    Given two oracles f and g where g(x) = f(x + s), find the hidden shift s.
    Uses n data qubits.

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

    # Step 2: Diagonal phase oracle f -- CZ on adjacent pairs
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


@register_algorithm(
    "element_distinctness", "oracle", (5, 8),
    tags=["quantum_walk", "oracle"],
)
def make_element_distinctness(n_qubits=5, n_steps=1, **kwargs) -> QuantumCircuit:
    """Grover-walk hybrid for element distinctness / collision detection.

    Uses 1 coin qubit + position qubits + data qubits.  Alternates between
    Grover diffusion on the coin register and quantum-walk shift steps on
    the position register, with data qubits acting as an oracle workspace.

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


@register_algorithm(
    "quantum_walk_search", "oracle", (3, 8),
    tags=["quantum_walk", "search"],
)
def make_quantum_walk_search(n_qubits=5, n_steps=2, **kwargs) -> QuantumCircuit:
    """Quantum spatial search via coined quantum walk.

    Implements a discrete-time quantum walk search on a position register
    with a marked vertex.

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
