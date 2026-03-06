"""Variational family: qaoa_maxcut, vqe_uccsd_fragment, hardware_efficient_ansatz,
adapt_vqe, vqd, recursive_qaoa, varqite, qaoa_weighted, quantum_boltzmann."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm
from zx_motifs.algorithms._helpers import decompose_toffoli


@register_algorithm(
    "qaoa_maxcut", "variational", (3, 8),
    tags=["combinatorial", "zz_interaction", "mixer"],
)
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


@register_algorithm(
    "vqe_uccsd", "variational", (4, 4),
    tags=["chemistry", "excitation"],
)
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


@register_algorithm(
    "hw_efficient_ansatz", "variational", (3, 8),
    tags=["hardware_efficient", "brick_layer"],
)
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


@register_algorithm(
    "adapt_vqe", "variational", (4, 8),
    tags=["chemistry", "adaptive", "excitation"],
)
def make_adapt_vqe(n_qubits=4, n_operators=3, **kwargs) -> QuantumCircuit:
    """ADAPT-VQE with iteratively grown operator pool.

    Starts from the Hartree-Fock state (X on the first n//2 qubits to
    fill the lowest spin-orbitals).  Then appends n_operators single-excitation
    operators, each implemented as a CNOT-ladder + RZ + reverse CNOT-ladder.

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

    # Append single-excitation operators from occupied -> virtual
    for op_idx in range(n_operators):
        # Cycle through occupied-virtual pairs
        occ = op_idx % n_occ
        virt = n_occ + (op_idx % (n - n_occ))
        theta = rng.uniform(0, 2 * np.pi)

        # Forward CX ladder: occ -> occ+1 -> ... -> virt
        for q in range(occ, virt):
            qc.cx(q, q + 1)

        # Parameterized rotation
        qc.rz(theta, virt)

        # Reverse CX ladder: virt -> ... -> occ
        for q in range(virt - 1, occ - 1, -1):
            qc.cx(q, q + 1)

    return qc


@register_algorithm(
    "vqd", "variational", (4, 8),
    tags=["chemistry", "excited_states"],
)
def make_vqd(n_qubits=4, layers=2, **kwargs) -> QuantumCircuit:
    """Variational Quantum Deflation for excited-state computation.

    Combines a hardware-efficient ansatz with a SWAP-test overlap penalty
    circuit.

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

    # Hardware-efficient ansatz on ansatz register
    for _layer in range(layers):
        for q in ansatz_qubits:
            qc.ry(rng.uniform(0, 2 * np.pi), q)
            qc.rz(rng.uniform(0, 2 * np.pi), q)
        for i in range(0, m - 1, 2):
            qc.cz(ansatz_qubits[i], ansatz_qubits[i + 1])
        for i in range(1, m - 1, 2):
            qc.cz(ansatz_qubits[i], ansatz_qubits[i + 1])

    # SWAP test between ansatz and reference registers
    # H on ancilla
    qc.h(ancilla)

    # Controlled-SWAP for each pair (ansatz[i], ref[i])
    # Fredkin decomposition: CX(b,c), Toffoli(a,c,b), CX(b,c)
    for i in range(m):
        a_q = ansatz_qubits[i]
        r_q = ref_qubits[i]
        qc.cx(r_q, a_q)
        decompose_toffoli(qc, ancilla, a_q, r_q)
        qc.cx(r_q, a_q)

    # H on ancilla
    qc.h(ancilla)

    return qc


@register_algorithm(
    "recursive_qaoa", "variational", (4, 8),
    tags=["combinatorial", "recursive"],
)
def make_recursive_qaoa(n_qubits=6, p=1, gamma=0.5, beta=0.3,
                        n_fixed=2, **kwargs) -> QuantumCircuit:
    """Recursive QAOA (RQAOA) with variable fixing.

    First performs a standard QAOA layer on all qubits (MaxCut on a ring),
    then "fixes" n_fixed qubits (applies X to simulate classical assignment)
    and runs a reduced QAOA layer on the remaining qubits.

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

    # Round 1: Full QAOA on all qubits
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

    # Fix variables: apply X to "freeze" the first n_fixed qubits
    fixed_qubits = list(range(n_fixed))
    remaining_qubits = list(range(n_fixed, n))

    for q in fixed_qubits:
        qc.x(q)

    # Round 2: Reduced QAOA on remaining qubits
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


@register_algorithm(
    "varqite", "variational", (3, 8),
    tags=["imaginary_time", "variational"],
)
def make_varqite(n_qubits=4, layers=3, **kwargs) -> QuantumCircuit:
    """Variational Quantum Imaginary Time Evolution (VarQITE).

    Parameterized ansatz structured for McLachlan's variational principle.
    Uses RY + RZ single-qubit layers and CX entangling layers, with a
    specific parameter structure: RY angles decrease across layers
    (mimicking imaginary-time cooling) while RZ angles provide phase
    freedom.

    Args:
        n_qubits: Number of qubits (minimum 4).
        layers: Number of ansatz layers (default 3).
    """
    n = max(4, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(161)

    for layer_idx in range(layers):
        # RY angles: scale decreases with layer to mimic cooling
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


@register_algorithm(
    "qaoa_weighted", "variational", (3, 8),
    tags=["combinatorial", "weighted"],
)
def make_qaoa_weighted(n_qubits=4, p=1, gamma=0.5, beta=0.3,
                       **kwargs) -> QuantumCircuit:
    """QAOA for weighted MaxCut on a ring graph with non-uniform edge weights.

    Same structure as standard QAOA MaxCut but each edge in the ring graph
    has a different weight.  The ZZ interaction angle for each edge is
    gamma * weight (instead of a uniform gamma).

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


@register_algorithm(
    "quantum_boltzmann", "variational", (4, 8),
    tags=["generative", "thermal"],
)
def make_quantum_boltzmann(n_qubits=4, layers=2, beta_param=0.5,
                           **kwargs) -> QuantumCircuit:
    """Quantum Boltzmann machine training circuit.

    Parameterised ansatz for a restricted quantum Boltzmann machine with
    visible and hidden qubit registers.

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
        for h in hidden:
            qc.rz(beta_param * rng.uniform(0, 2 * np.pi), h)

        # 3) CX entangling between visible and hidden
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
