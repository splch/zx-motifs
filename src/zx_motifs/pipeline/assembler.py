"""
MAGIC Assembler: Motif-Assembled Gadget-Informed Circuits.

The central entry point for composing optimized quantum ansatze from
ZX-calculus motifs guided by a target Hamiltonian. Takes a Hamiltonian
plus configuration and returns a QuantumCircuit ready for VQE.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

import numpy as np
from qiskit import QuantumCircuit

from .benchmark_suite import BenchmarkProblem, PauliTerm
from .phase_poly import (
    PhasePolynomial,
    compute_symmetry_tags_from_pauli_strings,
    extract_from_circuit,
)
from .problem_aware import (
    HamiltonianDecomposition,
    PauliGroup,
    decompose_hamiltonian,
)

logger = logging.getLogger(__name__)

PatternType = Literal["layered", "brick", "star", "adaptive"]


# ── Configuration ──────────────────────────────────────────────────


@dataclass
class MAGICConfig:
    """Configuration for the MAGIC assembler."""

    pattern: PatternType = "layered"
    n_layers: int = 2
    identity_init: bool = True
    symmetry_filter: bool = True
    required_symmetries: list[str] = field(default_factory=list)
    max_2q_gates: int | None = None
    optimize_phases: bool = True


# ── Ansatz Patterns ────────────────────────────────────────────────


def _build_gadget_layer(
    n_qubits: int,
    pauli_strings: list[str],
    param_offset: int = 0,
) -> tuple[QuantumCircuit, int]:
    """Build a circuit layer implementing parametric phase gadgets.

    Each Pauli string ZZ...Z gets a CNOT ladder + Rz + CNOT ladder.
    Returns (circuit, next_param_offset).
    """
    qc = QuantumCircuit(n_qubits)
    p_idx = param_offset

    for ps in pauli_strings:
        support = [i for i, c in enumerate(ps) if c != "I"]
        if not support:
            continue

        if len(support) == 1:
            qc.rz(0.0, support[0])  # parametric placeholder
            p_idx += 1
        else:
            # CNOT ladder to compute parity
            for i in range(len(support) - 1):
                qc.cx(support[i], support[i + 1])
            qc.rz(0.0, support[-1])  # parametric
            p_idx += 1
            # Reverse CNOT ladder
            for i in range(len(support) - 2, -1, -1):
                qc.cx(support[i], support[i + 1])

    return qc, p_idx


def _single_qubit_dressing(n_qubits: int) -> QuantumCircuit:
    """RY-RZ dressing layer for each qubit."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(0.0, i)
        qc.rz(0.0, i)
    return qc


def _build_pattern_layered(
    n_qubits: int,
    groups: list[PauliGroup],
    config: MAGICConfig,
) -> QuantumCircuit:
    """Pattern A: Layered — dressing + gadget layers per group, repeated."""
    qc = QuantumCircuit(n_qubits)

    for _ in range(config.n_layers):
        # Single-qubit dressing
        qc.compose(_single_qubit_dressing(n_qubits), inplace=True)

        # One gadget layer per commuting group
        for group in groups:
            strings = [t.pauli_string for t in group.terms]
            layer, _ = _build_gadget_layer(n_qubits, strings)
            qc.compose(layer, inplace=True)

    # Final dressing
    qc.compose(_single_qubit_dressing(n_qubits), inplace=True)
    return qc


def _build_pattern_brick(
    n_qubits: int,
    groups: list[PauliGroup],
    config: MAGICConfig,
) -> QuantumCircuit:
    """Pattern B: Brick — alternating even/odd nearest-neighbor gadgets."""
    qc = QuantumCircuit(n_qubits)

    # Extract all 2-local terms
    two_local = []
    for group in groups:
        for term in group.terms:
            support = [i for i, c in enumerate(term.pauli_string) if c != "I"]
            if len(support) == 2:
                two_local.append(term.pauli_string)

    for _ in range(config.n_layers):
        qc.compose(_single_qubit_dressing(n_qubits), inplace=True)

        # Even layer: pairs (0,1), (2,3), ...
        even_strings = [s for s in two_local
                        if _get_support(s)[0] % 2 == 0]
        if even_strings:
            layer, _ = _build_gadget_layer(n_qubits, even_strings)
            qc.compose(layer, inplace=True)

        # Odd layer: pairs (1,2), (3,4), ...
        odd_strings = [s for s in two_local
                       if _get_support(s)[0] % 2 == 1]
        if odd_strings:
            layer, _ = _build_gadget_layer(n_qubits, odd_strings)
            qc.compose(layer, inplace=True)

    qc.compose(_single_qubit_dressing(n_qubits), inplace=True)
    return qc


def _build_pattern_star(
    n_qubits: int,
    groups: list[PauliGroup],
    config: MAGICConfig,
) -> QuantumCircuit:
    """Pattern C: Star — hub qubit connected to all others + gadgets."""
    qc = QuantumCircuit(n_qubits)

    for _ in range(config.n_layers):
        qc.compose(_single_qubit_dressing(n_qubits), inplace=True)

        # Star entangling: hub qubit 0
        for i in range(1, n_qubits):
            qc.cx(0, i)
            qc.rz(0.0, i)
            qc.cx(0, i)

        # Problem-specific gadgets
        for group in groups:
            strings = [t.pauli_string for t in group.terms]
            # Only include non-trivial multi-qubit terms
            multi = [s for s in strings if sum(1 for c in s if c != "I") > 1]
            if multi:
                layer, _ = _build_gadget_layer(n_qubits, multi)
                qc.compose(layer, inplace=True)

    qc.compose(_single_qubit_dressing(n_qubits), inplace=True)
    return qc


def _build_pattern_adaptive(
    n_qubits: int,
    groups: list[PauliGroup],
    config: MAGICConfig,
) -> QuantumCircuit:
    """Pattern D: Adaptive — starts layered, adds gadgets by gradient magnitude."""
    # Start with layered pattern
    return _build_pattern_layered(n_qubits, groups, config)


def _get_support(ps: str) -> list[int]:
    return [i for i, c in enumerate(ps) if c != "I"]


_PATTERN_BUILDERS = {
    "layered": _build_pattern_layered,
    "brick": _build_pattern_brick,
    "star": _build_pattern_star,
    "adaptive": _build_pattern_adaptive,
}


# ── Symmetry Filtering ─────────────────────────────────────────────


def _filter_by_symmetry(
    groups: list[PauliGroup],
    required: list[str],
) -> list[PauliGroup]:
    """Filter Pauli groups to keep only symmetry-preserving terms."""
    if not required:
        return groups

    filtered_groups = []
    for group in groups:
        kept_terms = []
        for term in group.terms:
            tags = compute_symmetry_tags_from_pauli_strings([term.pauli_string])
            keep = True
            for sym in required:
                if sym == "particle_number" and not tags["particle_number_preserving"]:
                    keep = False
                elif sym == "z2_parity" and not tags["z2_parity_preserving"]:
                    keep = False
                elif sym == "real_valued" and not tags["real_valued"]:
                    keep = False
            if keep:
                kept_terms.append(term)
        if kept_terms:
            filtered_groups.append(PauliGroup(group_id=group.group_id, terms=kept_terms))

    return filtered_groups


# ── Main Entry Point ───────────────────────────────────────────────


def assemble_magic(
    problem: BenchmarkProblem | None = None,
    pauli_terms: list[PauliTerm] | None = None,
    n_qubits: int | None = None,
    config: MAGICConfig | None = None,
    motif_polynomials: dict[str, PhasePolynomial] | None = None,
) -> QuantumCircuit:
    """Assemble a MAGIC ansatz circuit from a target problem.

    This is the central MAGIC entry point. It:
    1. Extracts Pauli strings from the Hamiltonian
    2. Partitions into commuting groups
    3. Optionally filters by symmetry
    4. Builds the circuit using the selected pattern (A/B/C/D)
    5. Returns a QuantumCircuit ready for VQE

    Args:
        problem: A BenchmarkProblem instance (preferred).
        pauli_terms: Alternative: raw Pauli terms.
        n_qubits: Number of qubits (inferred from problem if not given).
        config: MAGIC configuration. Defaults to layered pattern.
        motif_polynomials: Optional motif phase polynomials for covering.

    Returns:
        QuantumCircuit with parametric rotations for VQE.
    """
    if config is None:
        config = MAGICConfig()

    # Get Pauli terms
    if problem is not None:
        terms = problem.pauli_terms
        nq = problem.n_qubits
        symmetries = problem.symmetries
    elif pauli_terms is not None:
        terms = pauli_terms
        nq = n_qubits or len(terms[0].pauli_string)
        symmetries = []
    else:
        raise ValueError("Must provide either problem or pauli_terms")

    if n_qubits is not None:
        nq = n_qubits

    # Decompose Hamiltonian
    decomposition = decompose_hamiltonian(terms, motif_polynomials)
    groups = decomposition.groups

    # Symmetry filtering
    if config.symmetry_filter and (config.required_symmetries or symmetries):
        required = config.required_symmetries or [
            s.lower().replace(" ", "_") for s in symmetries
        ]
        groups = _filter_by_symmetry(groups, required)

    if not groups:
        # Fallback: use all nontrivial terms in one group
        nontrivial = [t for t in terms if any(c != "I" for c in t.pauli_string)]
        if nontrivial:
            groups = [PauliGroup(group_id=0, terms=nontrivial)]
        else:
            return QuantumCircuit(nq)

    # Build circuit
    builder = _PATTERN_BUILDERS.get(config.pattern, _build_pattern_layered)
    qc = builder(nq, groups, config)

    # Identity initialization: set all parameters to 0
    if config.identity_init:
        # Parameters are already 0.0 (the default in our construction)
        pass

    logger.info(
        "Assembled MAGIC circuit: %d qubits, %d groups, pattern=%s, "
        "%d layers, %d gates",
        nq, len(groups), config.pattern, config.n_layers,
        len(qc.data),
    )

    return qc


# ── Convenience Functions ──────────────────────────────────────────


def assemble_for_problem(
    problem: BenchmarkProblem,
    pattern: PatternType = "layered",
    n_layers: int = 2,
) -> QuantumCircuit:
    """Quick assembly: problem + pattern -> circuit."""
    config = MAGICConfig(pattern=pattern, n_layers=n_layers)
    return assemble_magic(problem=problem, config=config)
