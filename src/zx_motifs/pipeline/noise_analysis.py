"""
Noise analysis for quantum ansatz evaluation.

Three noise tiers:
  T0: Ideal statevector simulation
  T1: Abstract depolarizing noise at configurable rates
  T2: Device-calibrated noise (requires qiskit-aer)

Evaluates how circuit performance degrades under noise and estimates
fault-tolerant resource requirements.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .evaluation import count_2q

logger = logging.getLogger(__name__)


@dataclass
class NoiseResult:
    """Result of noise analysis at a specific tier."""

    tier: str
    error_rate: float
    energy: float
    ideal_energy: float
    degradation: float  # |noisy - ideal| / |ideal|
    gate_budget_exceeded: bool


# ── T0: Ideal ──────────────────────────────────────────────────────


def evaluate_t0_ideal(
    qc: QuantumCircuit,
    hamiltonian: np.ndarray,
    params: np.ndarray | None = None,
) -> NoiseResult:
    """T0: Ideal statevector simulation."""
    n = qc.num_qubits
    if params is not None:
        bound_qc = qc.assign_parameters(dict(zip(qc.parameters, params)))
    else:
        bound_qc = qc

    sv = Statevector.from_instruction(bound_qc)
    energy = float(np.real(np.array(sv.data).conj() @ hamiltonian @ np.array(sv.data)))

    return NoiseResult(
        tier="T0",
        error_rate=0.0,
        energy=energy,
        ideal_energy=energy,
        degradation=0.0,
        gate_budget_exceeded=False,
    )


# ── T1: Abstract Depolarizing ─────────────────────────────────────


def _apply_depolarizing_channel(
    density_matrix: np.ndarray,
    n_qubits: int,
    target_qubit: int,
    p: float,
) -> np.ndarray:
    """Apply single-qubit depolarizing channel to a density matrix."""
    dim = 2**n_qubits
    d_q = 2
    rho = density_matrix.copy()

    # Partial trace machinery
    I = np.eye(d_q)
    paulis = [
        np.eye(d_q),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.array([[1, 0], [0, -1]]),
    ]

    # Depolarizing: rho -> (1-p)*rho + p/3 * sum_i sigma_i rho sigma_i
    result = (1 - p) * rho
    for sigma in paulis[1:]:  # X, Y, Z
        # Build full operator
        ops = [np.eye(2)] * n_qubits
        ops[target_qubit] = sigma
        full_op = ops[0]
        for op in ops[1:]:
            full_op = np.kron(full_op, op)
        result += (p / 3) * full_op @ rho @ full_op.conj().T

    return result


def evaluate_t1_depolarizing(
    qc: QuantumCircuit,
    hamiltonian: np.ndarray,
    ideal_energy: float,
    error_rates: list[float] | None = None,
    params: np.ndarray | None = None,
) -> list[NoiseResult]:
    """T1: Evaluate under abstract depolarizing noise.

    Simulates depolarizing noise after each 2-qubit gate.
    Uses density matrix simulation for small circuits.
    """
    if error_rates is None:
        error_rates = [1e-4, 1e-3, 1e-2]

    n = qc.num_qubits
    if n > 12:
        logger.warning("T1 density matrix simulation limited to 12 qubits, skipping")
        return []

    results = []
    for p in error_rates:
        if params is not None:
            bound_qc = qc.assign_parameters(dict(zip(qc.parameters, params)))
        else:
            bound_qc = qc

        # Start with |0...0>
        dim = 2**n
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 1.0

        # Apply circuit gates with noise
        for instruction in bound_qc.data:
            op_name = instruction.operation.name.lower()
            qubit_indices = [bound_qc.find_bit(q).index for q in instruction.qubits]

            # Get unitary
            try:
                from qiskit.quantum_info import Operator
                gate_unitary = Operator(instruction.operation).data
            except Exception:
                continue

            # Embed in full space
            if len(qubit_indices) == 1:
                full_u = _embed_single(gate_unitary, qubit_indices[0], n)
            elif len(qubit_indices) == 2:
                full_u = _embed_two(gate_unitary, qubit_indices[0], qubit_indices[1], n)
            else:
                continue

            rho = full_u @ rho @ full_u.conj().T

            # Apply depolarizing noise after 2-qubit gates
            if len(qubit_indices) >= 2:
                for qi in qubit_indices:
                    rho = _apply_depolarizing_channel(rho, n, qi, p)

        energy = float(np.real(np.trace(hamiltonian @ rho)))
        degradation = abs(energy - ideal_energy) / abs(ideal_energy) if ideal_energy != 0 else 0.0

        results.append(NoiseResult(
            tier="T1",
            error_rate=p,
            energy=energy,
            ideal_energy=ideal_energy,
            degradation=degradation,
            gate_budget_exceeded=False,
        ))

    return results


def _embed_single(gate: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Embed a single-qubit gate into the full Hilbert space."""
    ops = [np.eye(2)] * n_qubits
    ops[qubit] = gate
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def _embed_two(gate: np.ndarray, q0: int, q1: int, n_qubits: int) -> np.ndarray:
    """Embed a two-qubit gate into the full Hilbert space.

    Handles non-adjacent qubits via SWAP routing.
    """
    if abs(q0 - q1) == 1 and q0 < q1:
        # Adjacent, in order
        ops = [np.eye(2)] * n_qubits
        # Replace qubits q0, q0+1 with the 2-qubit gate
        result = np.eye(1)
        for i in range(n_qubits):
            if i == q0:
                continue
            elif i == q0 + 1:
                result = np.kron(result, gate) if i == q0 + 1 and q0 == 0 else result
            else:
                pass

    # General case: use permutation approach
    dim = 2**n_qubits
    result = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            bits_i = [(i >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits)]
            bits_j = [(j >> (n_qubits - 1 - k)) & 1 for k in range(n_qubits)]
            # Gate acts on qubits q0, q1
            gate_row = bits_i[q0] * 2 + bits_i[q1]
            gate_col = bits_j[q0] * 2 + bits_j[q1]
            # Other qubits must match
            other_match = all(bits_i[k] == bits_j[k] for k in range(n_qubits) if k not in (q0, q1))
            if other_match:
                result[i, j] = gate[gate_row, gate_col]

    return result


# ── Gate Budget Analysis ───────────────────────────────────────────


def gate_budget_analysis(
    qc: QuantumCircuit,
    error_rate: float = 1e-3,
) -> dict:
    """Estimate whether the circuit exceeds the noise budget.

    A circuit with N 2-qubit gates at error rate p produces
    approximately N*p total error. If N*p > 1, output is ~random.
    """
    n_2q = count_2q(qc)
    total_error = n_2q * error_rate
    depth = qc.depth()

    return {
        "n_2q_gates": n_2q,
        "error_rate": error_rate,
        "total_error_estimate": total_error,
        "depth": depth,
        "exceeds_budget": total_error > 1.0,
        "effective_fidelity": max(0, (1 - error_rate) ** n_2q),
    }


# ── T-Count Resource Estimation ───────────────────────────────────


def fault_tolerant_estimate(qc: QuantumCircuit) -> dict:
    """Estimate fault-tolerant resource requirements.

    Counts T-gates (which dominate distillation cost) and
    estimates logical qubit and time overhead.
    """
    from .converter import qiskit_to_zx, count_t_gates
    import copy
    import pyzx as zx

    try:
        zx_circ = qiskit_to_zx(qc)
        g = zx_circ.to_graph()
        g2 = copy.deepcopy(g)
        zx.simplify.full_reduce(g2)
        t_count = count_t_gates(g2)
    except Exception:
        t_count = -1

    n = qc.num_qubits
    n_2q = count_2q(qc)

    return {
        "logical_qubits": n,
        "t_count": t_count,
        "cx_count": n_2q,
        "depth": qc.depth(),
        # Rough estimates for surface code at d=17
        "physical_qubits_per_logical": 2 * 17**2,  # ~578
        "total_physical_qubits": n * 2 * 17**2,
        "distillation_factories": max(1, t_count // 10) if t_count > 0 else 0,
    }
