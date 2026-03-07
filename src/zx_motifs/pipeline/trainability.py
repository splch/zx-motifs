"""
Trainability analysis: Dynamical Lie Algebra (DLA) dimension estimation
and gradient variance computation for barren plateau detection.

Given a set of Pauli string generators (the gadgets of an ansatz), computes
the dimension of their Lie closure under commutation. If DLA dimension grows
exponentially with n, the ansatz is at risk of barren plateaus.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .ansatz import _pauli_matrix

logger = logging.getLogger(__name__)


# ── Pauli Algebra ──────────────────────────────────────────────────


def _commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute [A, B] = AB - BA."""
    return A @ B - B @ A


def _is_linearly_independent(
    vectors: list[np.ndarray], candidate: np.ndarray, tol: float = 1e-10
) -> bool:
    """Check if candidate is linearly independent of existing vectors.

    Uses the vectorized (flattened) form and checks if adding the candidate
    increases the rank.
    """
    if not vectors:
        return np.linalg.norm(candidate) > tol

    mat = np.array([v.ravel() for v in vectors])
    candidate_flat = candidate.ravel()
    extended = np.vstack([mat, candidate_flat[np.newaxis, :]])
    rank_before = np.linalg.matrix_rank(mat, tol=tol)
    rank_after = np.linalg.matrix_rank(extended, tol=tol)
    return rank_after > rank_before


# ── DLA Computation ────────────────────────────────────────────────


@dataclass
class DLAResult:
    """Result of DLA dimension estimation."""

    dimension: int
    n_qubits: int
    n_generators: int
    max_possible: int  # 4^n - 1 for n qubits (full su(2^n))
    is_exponential: bool  # dimension > 2^n threshold
    warning: str = ""

    @property
    def fraction_of_max(self) -> float:
        return self.dimension / self.max_possible if self.max_possible > 0 else 0.0


def compute_dla_dimension(
    generators: list[str],
    n_qubits: int | None = None,
    max_iterations: int = 50,
    max_dimension: int = 500,
) -> DLAResult:
    """Compute the DLA dimension from Pauli string generators.

    Iteratively computes commutators until closure. For n <= 10 qubits,
    this is exact. For larger systems, uses a sampling estimator.

    Args:
        generators: List of Pauli strings, e.g. ["ZZII", "IZZI"].
        n_qubits: Number of qubits (inferred from string length if None).
        max_iterations: Maximum commutator iterations.
        max_dimension: Stop if DLA dimension exceeds this (likely exponential).

    Returns:
        DLAResult with dimension and analysis.
    """
    if not generators:
        return DLAResult(
            dimension=0, n_qubits=0, n_generators=0, max_possible=0,
            is_exponential=False, warning="No generators provided",
        )

    if n_qubits is None:
        n_qubits = len(generators[0])

    max_possible = 4**n_qubits - 1  # dim(su(2^n))

    # For large systems, use sampling
    if n_qubits > 10:
        return _dla_sampling_estimate(generators, n_qubits, max_iterations)

    # Build initial basis from generators (as matrices times -i)
    basis: list[np.ndarray] = []
    for g in generators:
        mat = -1j * _pauli_matrix(g)
        if _is_linearly_independent(basis, mat):
            basis.append(mat)

    # Iterate: compute all commutators of basis elements
    for iteration in range(max_iterations):
        new_elements = []
        n = len(basis)
        for i in range(n):
            for j in range(i + 1, n):
                comm = _commutator(basis[i], basis[j])
                if np.linalg.norm(comm) < 1e-10:
                    continue
                if _is_linearly_independent(basis + new_elements, comm):
                    new_elements.append(comm)
                    if len(basis) + len(new_elements) >= max_dimension:
                        dim = len(basis) + len(new_elements)
                        return DLAResult(
                            dimension=dim,
                            n_qubits=n_qubits,
                            n_generators=len(generators),
                            max_possible=max_possible,
                            is_exponential=dim > 2**n_qubits,
                            warning=f"Stopped at max_dimension={max_dimension}; likely exponential",
                        )

        if not new_elements:
            break
        basis.extend(new_elements)

    dim = len(basis)
    return DLAResult(
        dimension=dim,
        n_qubits=n_qubits,
        n_generators=len(generators),
        max_possible=max_possible,
        is_exponential=dim > 2**n_qubits,
    )


def _dla_sampling_estimate(
    generators: list[str],
    n_qubits: int,
    max_iterations: int,
) -> DLAResult:
    """Sampling-based DLA dimension estimate for large systems.

    Tracks the number of linearly independent Pauli strings generated
    by nested commutators, without building full matrices.
    """
    max_possible = 4**n_qubits - 1

    # Track Pauli strings as a set (commutator of Pauli strings gives Pauli strings)
    known_strings: set[str] = set()
    for g in generators:
        if any(c != "I" for c in g):
            known_strings.add(g)

    for _ in range(max_iterations):
        new_strings: set[str] = set()
        gen_list = list(known_strings)
        for i in range(len(gen_list)):
            for j in range(i + 1, len(gen_list)):
                comm_str = _pauli_string_commutator(gen_list[i], gen_list[j])
                if comm_str and comm_str not in known_strings:
                    new_strings.add(comm_str)
        if not new_strings:
            break
        known_strings |= new_strings

    dim = len(known_strings)
    return DLAResult(
        dimension=dim,
        n_qubits=n_qubits,
        n_generators=len(generators),
        max_possible=max_possible,
        is_exponential=dim > 2**n_qubits,
        warning="Sampling estimate (Pauli string algebra)",
    )


def _pauli_string_commutator(p1: str, p2: str) -> str | None:
    """Compute the commutator of two Pauli strings at the string level.

    [P1, P2] = 2i * P1*P2 if they anticommute, 0 if they commute.
    Returns the product Pauli string if they anticommute, None if they commute.
    """
    # Count anticommuting positions
    n_anti = 0
    result_chars = []
    for a, b in zip(p1, p2):
        product = _single_pauli_product(a, b)
        result_chars.append(product)
        if a != "I" and b != "I" and a != b:
            n_anti += 1

    if n_anti % 2 == 0:
        return None  # Commuting
    result = "".join(result_chars)
    if all(c == "I" for c in result):
        return None
    return result


_PAULI_PRODUCT = {
    ("I", "I"): "I", ("I", "X"): "X", ("I", "Y"): "Y", ("I", "Z"): "Z",
    ("X", "I"): "X", ("X", "X"): "I", ("X", "Y"): "Z", ("X", "Z"): "Y",
    ("Y", "I"): "Y", ("Y", "X"): "Z", ("Y", "Y"): "I", ("Y", "Z"): "X",
    ("Z", "I"): "Z", ("Z", "X"): "Y", ("Z", "Y"): "X", ("Z", "Z"): "I",
}


def _single_pauli_product(a: str, b: str) -> str:
    """Product of two single-qubit Paulis (ignoring phase)."""
    return _PAULI_PRODUCT[(a, b)]


# ── Gradient Variance ──────────────────────────────────────────────


@dataclass
class GradientVarianceResult:
    """Result of gradient variance estimation."""

    variance: float
    n_qubits: int
    n_samples: int
    is_barren: bool  # variance < 1e-4 threshold


def estimate_gradient_variance(
    ansatz_fn,
    n_qubits: int,
    hamiltonian: np.ndarray,
    param_index: int = 0,
    n_samples: int = 100,
    seed: int = 42,
) -> GradientVarianceResult:
    """Estimate the variance of the gradient of a VQE cost function.

    Uses finite-difference parameter-shift rule to estimate the gradient
    at random parameter points.

    Args:
        ansatz_fn: Callable(params) -> QuantumCircuit.
        n_qubits: Number of qubits.
        hamiltonian: Hamiltonian matrix.
        param_index: Which parameter to compute gradient for.
        n_samples: Number of random parameter samples.
        seed: Random seed.
    """
    from qiskit.quantum_info import Statevector

    rng = np.random.default_rng(seed)
    gradients = []

    for _ in range(n_samples):
        params = rng.uniform(-np.pi, np.pi, ansatz_fn.__code__.co_varnames.__len__()
                             if hasattr(ansatz_fn, '__code__') else 4 * n_qubits)
        # Use as many params as the function needs
        try:
            shift = np.pi / 2
            params_plus = params.copy()
            params_plus[param_index] += shift
            params_minus = params.copy()
            params_minus[param_index] -= shift

            qc_plus = ansatz_fn(params_plus)
            qc_minus = ansatz_fn(params_minus)

            sv_plus = Statevector.from_instruction(qc_plus)
            sv_minus = Statevector.from_instruction(qc_minus)

            e_plus = float(np.real(np.array(sv_plus.data).conj() @ hamiltonian @ np.array(sv_plus.data)))
            e_minus = float(np.real(np.array(sv_minus.data).conj() @ hamiltonian @ np.array(sv_minus.data)))

            grad = (e_plus - e_minus) / 2.0
            gradients.append(grad)
        except Exception:
            continue

    if not gradients:
        return GradientVarianceResult(
            variance=0.0, n_qubits=n_qubits, n_samples=0, is_barren=True,
        )

    var = float(np.var(gradients))
    return GradientVarianceResult(
        variance=var,
        n_qubits=n_qubits,
        n_samples=len(gradients),
        is_barren=var < 1e-4,
    )
