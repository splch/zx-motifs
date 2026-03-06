"""Simulation family: trotter_ising, trotter_heisenberg, quantum_walk, qdrift,
higher_order_trotter, hubbard_trotter, ctqw, vqs_real_time."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm


@register_algorithm(
    "trotter_ising", "simulation", (2, 8),
    tags=["hamiltonian_simulation", "zz_interaction", "trotter"],
)
def make_trotter_ising(n_qubits=4, n_steps=1, dt=0.5, j_coupling=1.0,
                       h_field=0.5, **kwargs) -> QuantumCircuit:
    """Trotterized transverse-field Ising model: H = -J SumZZ - h SumX.

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


@register_algorithm(
    "trotter_heisenberg", "simulation", (2, 8),
    tags=["hamiltonian_simulation", "trotter", "mixed_interaction"],
)
def make_trotter_heisenberg(n_qubits=4, n_steps=1, dt=0.5, **kwargs) -> QuantumCircuit:
    """Trotterized Heisenberg XXX model: H = Sum(XX + YY + ZZ).

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
            # YY term: Sdg-H-CX-RZ-CX-H-S
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


@register_algorithm(
    "quantum_walk", "simulation", (3, 6),
    tags=["discrete_walk", "coin_operator"],
)
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


@register_algorithm(
    "qdrift", "simulation", (2, 8),
    tags=["hamiltonian_simulation", "randomized"],
)
def make_qdrift(n_qubits=4, n_steps=4, dt=0.3, seed=42, **kwargs) -> QuantumCircuit:
    """qDRIFT randomised Hamiltonian simulation.

    Simulates an Ising-like Hamiltonian  H = Sum_i ZZ_{i,i+1} + Sum_i X_i
    using the qDRIFT channel.

    Tags: hamiltonian_simulation, randomized
    """
    n_steps = kwargs.get("n_steps", n_steps)
    dt = kwargs.get("dt", dt)
    seed = kwargs.get("seed", seed)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)
    rng = np.random.default_rng(seed)

    # Hamiltonian terms: (type, index, coefficient)
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
        angle = 2 * coeff * tau
        if kind == "zz":
            qc.cx(qubit_idx, qubit_idx + 1)
            qc.rz(angle, qubit_idx + 1)
            qc.cx(qubit_idx, qubit_idx + 1)
        else:  # "x"
            qc.rx(angle, qubit_idx)

    return qc


@register_algorithm(
    "higher_order_trotter", "simulation", (2, 8),
    tags=["hamiltonian_simulation", "trotter", "higher_order"],
)
def make_higher_order_trotter(n_qubits=4, n_steps=1, dt=0.5,
                              j_coupling=1.0, h_field=0.5,
                              **kwargs) -> QuantumCircuit:
    """2nd-order (symmetric) Suzuki-Trotter for transverse-field Ising.

    H = -J Sum ZZ_{i,i+1} - h Sum X_i

    The 2nd-order Trotter formula per step is:
        S2(dt) = e^{-i H_ZZ dt/2} * e^{-i H_X dt} * e^{-i H_ZZ dt/2}

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


@register_algorithm(
    "hubbard_trotter", "simulation", (2, 8),
    tags=["hamiltonian_simulation", "fermionic", "trotter"],
)
def make_hubbard_trotter(n_qubits=4, n_steps=1, dt=0.5,
                         t_hop=1.0, u_int=2.0, **kwargs) -> QuantumCircuit:
    """Trotterised Fermi-Hubbard model (1-D, spinless, Jordan-Wigner).

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
        # Hopping terms (XX + YY interaction via JW)
        for i in range(n - 1):
            # Decomposition of exp(-i t_hop dt (XX+YY)/2)
            qc.ry(-np.pi / 2, i)
            qc.cx(i, i + 1)
            qc.ry(hop_angle, i)
            qc.cx(i, i + 1)
            qc.ry(np.pi / 2, i)

        # On-site interaction terms
        for i in range(n):
            qc.rz(u_int * dt, i)

    return qc


@register_algorithm(
    "ctqw", "simulation", (2, 8),
    tags=["quantum_walk", "continuous_time"],
)
def make_ctqw(n_qubits=4, n_steps=2, dt=0.5, **kwargs) -> QuantumCircuit:
    """Continuous-time quantum walk on a line graph via Trotterisation.

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


@register_algorithm(
    "vqs_real_time", "simulation", (2, 8),
    tags=["variational", "real_time"],
)
def make_vqs_real_time(n_qubits=4, layers=2, dt=0.3, **kwargs) -> QuantumCircuit:
    """Variational quantum simulation for real-time dynamics.

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
        # 1) RY rotations -- real part of variational parameters
        for i in range(n):
            theta = rng.uniform(0, 2 * np.pi)
            qc.ry(theta, i)

        # 2) RX rotations -- imaginary part of variational parameters
        for i in range(n):
            phi = rng.uniform(0, 2 * np.pi)
            qc.rx(phi, i)

        # 3) RZ rotations -- time-step encoding
        time_scale = dt * (layer_idx + 1)
        for i in range(n):
            qc.rz(time_scale * rng.uniform(0.5, 1.5), i)

        # 4) CX entangling ladder
        for i in range(n - 1):
            qc.cx(i, i + 1)

    return qc
