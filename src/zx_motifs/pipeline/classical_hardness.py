"""
Classical hardness certificates for quantum ansatz evaluation.

Wraps tensor network simulation (via quimb, optional), stabilizer
simulation, and light-cone analysis to actively attempt classical
simulation of quantum circuits. If any method succeeds cheaply,
the circuit provides no quantum advantage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HardnessResult:
    """Result of classical hardness evaluation."""

    method: str
    classically_tractable: bool
    details: dict
    error: float  # Approximation error achieved


# ── MPS Simulation (quimb) ─────────────────────────────────────────


def mps_simulation(
    hamiltonian: np.ndarray,
    n_qubits: int,
    bond_dims: list[int] | None = None,
    tol: float = 1e-3,
) -> HardnessResult:
    """Attempt ground state via MPS/DMRG using quimb.

    If a small bond dimension suffices for good accuracy,
    the problem is classically tractable.

    Requires: pip install quimb
    """
    if bond_dims is None:
        bond_dims = [16, 32, 64, 128, 256]

    try:
        import quimb.tensor as qtn
        import quimb as qu
    except ImportError:
        return HardnessResult(
            method="mps_dmrg",
            classically_tractable=False,
            details={"error": "quimb not installed"},
            error=float("inf"),
        )

    # Exact reference
    exact_energy = float(np.linalg.eigvalsh(hamiltonian.real)[0])

    results_by_chi = {}
    for chi in bond_dims:
        try:
            # Build MPO from Hamiltonian
            H_mpo = qtn.MatrixProductOperator.from_dense(
                hamiltonian, [2] * n_qubits
            )
            dmrg = qtn.DMRG2(H_mpo, bond_dims=chi)
            dmrg.solve(tol=1e-8, max_sweeps=20, verbosity=0)
            energy = float(dmrg.energy)
            error = abs(energy - exact_energy)
            results_by_chi[chi] = {"energy": energy, "error": error}

            if error < tol:
                return HardnessResult(
                    method="mps_dmrg",
                    classically_tractable=True,
                    details={
                        "bond_dim": chi,
                        "energy": energy,
                        "exact_energy": exact_energy,
                        "all_results": results_by_chi,
                    },
                    error=error,
                )
        except Exception as e:
            results_by_chi[chi] = {"error": str(e)}
            continue

    best_error = min(
        (r.get("error", float("inf")) for r in results_by_chi.values()
         if isinstance(r.get("error"), (int, float))),
        default=float("inf"),
    )

    return HardnessResult(
        method="mps_dmrg",
        classically_tractable=False,
        details={"all_results": results_by_chi, "exact_energy": exact_energy},
        error=float(best_error),
    )


# ── Light Cone Analysis ────────────────────────────────────────────


def light_cone_size(
    qc,
    target_qubit: int,
) -> int:
    """Compute the backward light cone size for a target qubit.

    Returns the number of qubits in the backward light cone.
    If this is small, the circuit is efficiently simulable for
    that qubit's measurement.
    """
    n = qc.num_qubits
    active = {target_qubit}

    # Walk backward through the circuit
    for instruction in reversed(qc.data):
        qubit_indices = [qc.find_bit(q).index for q in instruction.qubits]
        if any(q in active for q in qubit_indices):
            active.update(qubit_indices)

    return len(active)


def light_cone_analysis(qc) -> HardnessResult:
    """Analyze light cones for all qubits.

    If the maximum light cone is small relative to total qubits,
    the circuit is classically tractable.
    """
    n = qc.num_qubits
    sizes = [light_cone_size(qc, q) for q in range(n)]
    max_size = max(sizes)
    avg_size = sum(sizes) / n

    tractable = max_size <= n // 2 + 1  # More than half unused

    return HardnessResult(
        method="light_cone",
        classically_tractable=tractable,
        details={
            "max_light_cone": max_size,
            "avg_light_cone": avg_size,
            "n_qubits": n,
            "sizes": sizes,
        },
        error=0.0 if tractable else float("inf"),
    )


# ── Stabilizer Analysis ───────────────────────────────────────────


def stabilizer_check(qc) -> HardnessResult:
    """Check if a circuit is a Clifford circuit (efficiently simulable).

    Counts non-Clifford gates. If zero, the circuit is in the
    stabilizer formalism and classically simulable.
    """
    non_clifford_gates = {"t", "tdg", "rz", "ry", "rx", "u1", "u2", "u3", "p"}
    non_cliff_count = 0
    for inst in qc.data:
        name = inst.operation.name.lower()
        if name in non_clifford_gates:
            # Check if it's actually Clifford (S, Z, etc. are Clifford)
            if name in ("rz", "ry", "rx", "p", "u1"):
                angle = float(inst.operation.params[0]) if inst.operation.params else 0
                # Clifford angles: multiples of pi/2
                if abs(angle % (np.pi / 2)) > 1e-8:
                    non_cliff_count += 1
            else:
                non_cliff_count += 1

    return HardnessResult(
        method="stabilizer",
        classically_tractable=non_cliff_count == 0,
        details={"non_clifford_gates": non_cliff_count},
        error=0.0 if non_cliff_count == 0 else float("inf"),
    )


# ── Combined Hardness Evaluation ───────────────────────────────────


def evaluate_classical_hardness(
    qc,
    hamiltonian: np.ndarray | None = None,
    n_qubits: int | None = None,
) -> list[HardnessResult]:
    """Run all classical hardness checks.

    Returns a list of HardnessResult objects, one per method.
    """
    results = []
    nq = n_qubits or qc.num_qubits

    # Fast checks first
    results.append(stabilizer_check(qc))
    results.append(light_cone_analysis(qc))

    # Expensive check (MPS) only if fast checks don't already show tractability
    if hamiltonian is not None and not any(r.classically_tractable for r in results):
        results.append(mps_simulation(hamiltonian, nq))

    return results
