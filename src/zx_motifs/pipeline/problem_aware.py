"""
Problem-aware composition: from Hamiltonian to motif-covered circuit layers.

Decomposes a Hamiltonian into Pauli terms, builds a commutation graph,
partitions into simultaneously-diagonalizable groups via graph coloring,
and covers each group with motifs from the registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import networkx as nx
import numpy as np

from .benchmark_suite import PauliTerm
from .phase_poly import PhaseGadget, PhasePolynomial


# ── Pauli Commutation ──────────────────────────────────────────────


def _paulis_commute(p1: str, p2: str) -> bool:
    """Check if two Pauli strings commute.

    Two Pauli strings commute iff they differ on an even number of positions
    (excluding positions where either is identity).
    """
    n_anticommuting = 0
    for a, b in zip(p1, p2):
        if a == "I" or b == "I":
            continue
        if a != b:
            n_anticommuting += 1
    return n_anticommuting % 2 == 0


def build_commutation_graph(terms: list[PauliTerm]) -> nx.Graph:
    """Build a graph where vertices are Pauli terms and edges connect commuting pairs.

    This is the complement of the conflict graph: edges mean terms CAN be
    measured simultaneously.
    """
    g = nx.Graph()
    for i, t in enumerate(terms):
        g.add_node(i, term=t)
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            if _paulis_commute(terms[i].pauli_string, terms[j].pauli_string):
                g.add_edge(i, j)
    return g


def build_anticommutation_graph(terms: list[PauliTerm]) -> nx.Graph:
    """Build a graph where edges connect non-commuting (conflicting) pairs.

    Graph coloring on this gives simultaneously-diagonalizable groups.
    """
    g = nx.Graph()
    for i, t in enumerate(terms):
        g.add_node(i, term=t)
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            if not _paulis_commute(terms[i].pauli_string, terms[j].pauli_string):
                g.add_edge(i, j)
    return g


# ── Grouping ───────────────────────────────────────────────────────


@dataclass
class PauliGroup:
    """A group of simultaneously-diagonalizable Pauli terms."""

    group_id: int
    terms: list[PauliTerm]

    @property
    def n_qubits(self) -> int:
        return len(self.terms[0].pauli_string) if self.terms else 0

    def to_phase_polynomial(self) -> PhasePolynomial:
        """Convert group to a phase polynomial (one gadget per term)."""
        n = self.n_qubits
        poly = PhasePolynomial(n_qubits=n)
        for term in self.terms:
            support = frozenset(
                i for i, p in enumerate(term.pauli_string) if p != "I"
            )
            if support:
                # Use coefficient magnitude as a proxy phase
                # Real VQE will use parametric phases
                phase = Fraction(1, 4)  # Default parametric placeholder
                poly.add_gadget(support, phase)
        return poly


def partition_into_groups(terms: list[PauliTerm]) -> list[PauliGroup]:
    """Partition Pauli terms into simultaneously-diagonalizable groups.

    Uses greedy graph coloring on the anticommutation graph.
    """
    # Filter out pure identity terms
    nontrivial = [t for t in terms if any(c != "I" for c in t.pauli_string)]
    if not nontrivial:
        return []

    anti_graph = build_anticommutation_graph(nontrivial)
    coloring = nx.coloring.greedy_color(anti_graph, strategy="largest_first")

    groups: dict[int, list[PauliTerm]] = {}
    for node_idx, color in coloring.items():
        if color not in groups:
            groups[color] = []
        groups[color].append(nontrivial[node_idx])

    return [
        PauliGroup(group_id=gid, terms=gterms)
        for gid, gterms in sorted(groups.items())
    ]


# ── Motif Covering ─────────────────────────────────────────────────


@dataclass
class MotifCover:
    """Result of covering a Pauli group with motifs."""

    group_id: int
    matched_motifs: list[str]  # motif IDs
    uncovered_terms: list[PauliTerm]
    coverage_ratio: float


def cover_group_with_motifs(
    group: PauliGroup,
    motif_polynomials: dict[str, PhasePolynomial],
) -> MotifCover:
    """Cover a Pauli group's terms using motifs from the registry.

    For each term in the group, finds motifs whose phase polynomial
    signatures overlap with the term's support. Uses a greedy approach:
    pick the motif covering the most uncovered supports.
    """
    n = group.n_qubits
    # Build target supports from the group
    target_supports: list[frozenset[int]] = []
    for term in group.terms:
        support = frozenset(i for i, p in enumerate(term.pauli_string) if p != "I")
        if support:
            target_supports.append(support)

    if not target_supports:
        return MotifCover(group.group_id, [], [], 1.0)

    uncovered = set(range(len(target_supports)))
    matched_motifs: list[str] = []

    # Build coverage map: which motif covers which targets
    motif_coverage: dict[str, set[int]] = {}
    for mid, mpoly in motif_polynomials.items():
        motif_supports = {g.support for g in mpoly.gadgets}
        covered = set()
        for idx in range(len(target_supports)):
            if target_supports[idx] in motif_supports:
                covered.add(idx)
            # Also check subset coverage
            for ms in motif_supports:
                if ms.issubset(target_supports[idx]) or target_supports[idx].issubset(ms):
                    covered.add(idx)
        if covered:
            motif_coverage[mid] = covered

    # Greedy set cover
    while uncovered and motif_coverage:
        best_motif = max(
            motif_coverage,
            key=lambda m: len(motif_coverage[m] & uncovered),
        )
        newly_covered = motif_coverage[best_motif] & uncovered
        if not newly_covered:
            break
        uncovered -= newly_covered
        matched_motifs.append(best_motif)

    uncovered_terms = [group.terms[i] for i in sorted(uncovered)]
    total = len(target_supports)
    covered_count = total - len(uncovered)

    return MotifCover(
        group_id=group.group_id,
        matched_motifs=matched_motifs,
        uncovered_terms=uncovered_terms,
        coverage_ratio=covered_count / total if total > 0 else 1.0,
    )


# ── Top-Level Decomposition ───────────────────────────────────────


@dataclass
class HamiltonianDecomposition:
    """Complete decomposition of a Hamiltonian into Pauli groups and motif covers."""

    n_qubits: int
    n_terms: int
    groups: list[PauliGroup]
    covers: list[MotifCover]
    identity_offset: float  # Coefficient of the identity term

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def total_coverage(self) -> float:
        if not self.covers:
            return 0.0
        return sum(c.coverage_ratio for c in self.covers) / len(self.covers)


def decompose_hamiltonian(
    pauli_terms: list[PauliTerm],
    motif_polynomials: dict[str, PhasePolynomial] | None = None,
) -> HamiltonianDecomposition:
    """Decompose a Hamiltonian into grouped, motif-covered layers.

    Args:
        pauli_terms: The Hamiltonian as a list of PauliTerm objects.
        motif_polynomials: Optional dict mapping motif_id to PhasePolynomial.
            If None, no motif covering is attempted.

    Returns:
        HamiltonianDecomposition with groups and covers.
    """
    n_qubits = len(pauli_terms[0].pauli_string) if pauli_terms else 0

    # Separate identity term
    identity_offset = 0.0
    nontrivial = []
    for t in pauli_terms:
        if all(c == "I" for c in t.pauli_string):
            identity_offset += t.coefficient
        else:
            nontrivial.append(t)

    groups = partition_into_groups(nontrivial)
    covers = []
    if motif_polynomials:
        for group in groups:
            covers.append(cover_group_with_motifs(group, motif_polynomials))

    return HamiltonianDecomposition(
        n_qubits=n_qubits,
        n_terms=len(pauli_terms),
        groups=groups,
        covers=covers,
        identity_offset=identity_offset,
    )
