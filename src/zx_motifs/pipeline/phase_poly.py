"""
Phase polynomial extraction and arithmetic for ZX-calculus motifs.

A phase polynomial represents the non-Clifford content of a ZX diagram as a
set of (support_vector, phase) pairs. Each pair corresponds to a phase gadget
e^{i*phase*Z_1*Z_2*...*Z_k} acting on the qubits in the support.

This converts motif composition from subgraph isomorphism (VF2, scales poorly)
into linear algebra over F_2, enabling algebraic queries and fast composition.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from fractions import Fraction

import networkx as nx
import pyzx as zx
from pyzx.graph import Graph
from pyzx.utils import EdgeType, VertexType

from .featurizer import classify_phase


# ── Data Structures ─────────────────────────────────────────────────


@dataclass(frozen=True)
class PhaseGadget:
    """A single phase gadget: e^{i*phase*P} where P is a Pauli Z string."""

    support: frozenset[int]
    phase: Fraction
    n_qubits: int

    @property
    def phase_class(self) -> str:
        return classify_phase(self.phase)

    @property
    def weight(self) -> int:
        return len(self.support)

    def pauli_string(self) -> str:
        """Return Pauli string representation, e.g. 'ZZII' for support {0,1} on 4 qubits."""
        return "".join("Z" if i in self.support else "I" for i in range(self.n_qubits))

    def support_vector(self) -> list[int]:
        """Return binary support vector over F_2."""
        return [1 if i in self.support else 0 for i in range(self.n_qubits)]

    def is_particle_number_preserving(self) -> bool:
        """A ZZ...Z gadget always commutes with total Z, so always True."""
        return True

    def is_z2_parity_preserving(self) -> bool:
        """Even weight Pauli string preserves Z_2 parity."""
        return self.weight % 2 == 0

    def is_real_valued(self) -> bool:
        """Pure Z strings contain no Y Paulis, so always real."""
        return True


@dataclass
class PhasePolynomial:
    """A phase polynomial: a collection of phase gadgets.

    Represents the non-Clifford content of a ZX diagram or circuit as a
    sum of phase terms, each with a Boolean support vector and a phase angle.
    """

    n_qubits: int
    gadgets: list[PhaseGadget] = field(default_factory=list)

    def add_gadget(self, support: frozenset[int], phase: Fraction) -> None:
        """Add a phase gadget to the polynomial."""
        self.gadgets.append(PhaseGadget(support=support, phase=phase, n_qubits=self.n_qubits))

    def simplify(self) -> PhasePolynomial:
        """Merge gadgets with identical support by adding their phases.

        Returns a new PhasePolynomial with one gadget per unique support.
        Gadgets whose phases cancel to zero are removed.
        """
        merged: dict[frozenset[int], Fraction] = {}
        for g in self.gadgets:
            merged[g.support] = (merged.get(g.support, Fraction(0)) + g.phase) % 2
        result = PhasePolynomial(n_qubits=self.n_qubits)
        for support, phase in sorted(merged.items(), key=lambda x: sorted(x[0])):
            if phase != 0:
                result.gadgets.append(
                    PhaseGadget(support=support, phase=phase, n_qubits=self.n_qubits)
                )
        return result

    def compose(self, other: PhasePolynomial) -> PhasePolynomial:
        """Compose two phase polynomials (concatenation).

        The result has all gadgets from both polynomials. Call simplify()
        to merge matching support vectors.
        """
        n = max(self.n_qubits, other.n_qubits)
        result = PhasePolynomial(n_qubits=n)
        for g in self.gadgets:
            result.gadgets.append(
                PhaseGadget(support=g.support, phase=g.phase, n_qubits=n)
            )
        for g in other.gadgets:
            result.gadgets.append(
                PhaseGadget(support=g.support, phase=g.phase, n_qubits=n)
            )
        return result

    def cnot_propagate(self, control: int, target: int) -> PhasePolynomial:
        """Apply a CNOT gate to the polynomial (F_2 linear transformation).

        CNOT(c,t) transforms Z supports: if target is in support, toggle control.
        This is because CNOT^dag Z_t CNOT = Z_c Z_t.
        """
        result = PhasePolynomial(n_qubits=self.n_qubits)
        for g in self.gadgets:
            if target in g.support:
                new_support = g.support ^ frozenset([control])
            else:
                new_support = g.support
            result.gadgets.append(
                PhaseGadget(support=new_support, phase=g.phase, n_qubits=self.n_qubits)
            )
        return result

    @property
    def t_count(self) -> int:
        """Count T-like gadgets (phase denominator = 4)."""
        return sum(1 for g in self.gadgets if g.phase.denominator == 4)

    @property
    def non_clifford_count(self) -> int:
        """Count non-Clifford gadgets."""
        return sum(1 for g in self.gadgets if g.phase_class not in ("zero", "pauli", "clifford"))

    def pauli_strings(self) -> list[tuple[str, Fraction]]:
        """Return list of (pauli_string, phase) pairs."""
        return [(g.pauli_string(), g.phase) for g in self.gadgets]

    def support_matrix(self) -> list[list[int]]:
        """Return the support matrix over F_2 (rows = gadgets, cols = qubits)."""
        return [g.support_vector() for g in self.gadgets]

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "n_qubits": self.n_qubits,
            "gadgets": [
                {"support": sorted(g.support), "phase": str(g.phase)}
                for g in self.gadgets
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> PhasePolynomial:
        """Deserialize from a dict."""
        poly = cls(n_qubits=data["n_qubits"])
        for gd in data["gadgets"]:
            poly.add_gadget(
                support=frozenset(gd["support"]),
                phase=Fraction(gd["phase"]),
            )
        return poly


# ── Extraction from ZX graphs ──────────────────────────────────────


def extract_from_zx_graph(g: Graph) -> PhasePolynomial:
    """Extract a phase polynomial from a PyZX graph.

    Works on graphs at any simplification level. For each non-zero-phase
    non-boundary spider, determines qubit support from its Hadamard-connected
    neighbors' qubit assignments (via PyZX's qubit attribute).

    For spiders with only SIMPLE-edge connections, uses the spider's own
    qubit attribute as a single-qubit gadget.
    """
    boundary_qubits: set[int] = set()
    for v in g.vertices():
        if g.type(v) == VertexType.BOUNDARY:
            boundary_qubits.add(int(g.qubit(v)))
    n_qubits = max(boundary_qubits, default=-1) + 1
    if n_qubits == 0:
        return PhasePolynomial(n_qubits=0)

    poly = PhasePolynomial(n_qubits=n_qubits)

    for v in g.vertices():
        phase = g.phase(v)
        if phase == 0:
            continue
        if g.type(v) == VertexType.BOUNDARY:
            continue

        # Collect qubit indices from Hadamard-connected neighbors
        had_qubits: set[int] = set()
        for nbr in g.neighbors(v):
            etype = g.edge_type(g.edge(v, nbr))
            if etype == EdgeType.HADAMARD:
                q = int(g.qubit(nbr))
                if 0 <= q < n_qubits:
                    had_qubits.add(q)

        if had_qubits:
            poly.add_gadget(frozenset(had_qubits), phase)
        else:
            # Single-qubit phase: use the spider's own qubit
            q = int(g.qubit(v))
            if 0 <= q < n_qubits:
                poly.add_gadget(frozenset([q]), phase)

    return poly


def extract_from_circuit(qc) -> PhasePolynomial:
    """Extract phase polynomial from a Qiskit QuantumCircuit.

    Parses the circuit gate-by-gate, tracking CNOT operations as F_2
    basis changes and Rz/T/S gates as phase additions. This gives the
    exact phase polynomial without ZX simplification artifacts.
    """
    from math import pi

    n = qc.num_qubits
    # Current basis: basis[q] is the set of original qubits that qubit q
    # currently represents (starts as {q} for each q, CNOT XORs them).
    basis: list[set[int]] = [frozenset([i]) for i in range(n)]
    poly = PhasePolynomial(n_qubits=n)

    for instruction in qc.data:
        op = instruction.operation
        qubit_indices = [qc.find_bit(q).index for q in instruction.qubits]
        name = op.name.lower()

        if name == "cx" or name == "cnot":
            ctrl, tgt = qubit_indices
            basis[tgt] = basis[tgt] ^ basis[ctrl]

        elif name == "rz":
            q = qubit_indices[0]
            angle = float(op.params[0])
            # Convert angle to fraction of pi
            frac = Fraction(angle / pi).limit_denominator(256)
            if frac != 0:
                poly.add_gadget(frozenset(basis[q]), frac)

        elif name == "t":
            q = qubit_indices[0]
            poly.add_gadget(frozenset(basis[q]), Fraction(1, 4))

        elif name == "tdg":
            q = qubit_indices[0]
            poly.add_gadget(frozenset(basis[q]), Fraction(7, 4))

        elif name == "s":
            q = qubit_indices[0]
            poly.add_gadget(frozenset(basis[q]), Fraction(1, 2))

        elif name == "sdg":
            q = qubit_indices[0]
            poly.add_gadget(frozenset(basis[q]), Fraction(3, 2))

        elif name == "z":
            q = qubit_indices[0]
            poly.add_gadget(frozenset(basis[q]), Fraction(1, 1))

        elif name == "p" or name == "u1":
            q = qubit_indices[0]
            angle = float(op.params[0])
            frac = Fraction(angle / pi).limit_denominator(256)
            if frac != 0:
                poly.add_gadget(frozenset(basis[q]), frac)

        # Other gates (H, X, Y, etc.) are not phase-polynomial gates
        # and are ignored. For full accuracy, the circuit should be
        # decomposed into {CNOT, Rz} before extraction.

    return poly


def extract_from_networkx(nxg: nx.Graph) -> list[PhaseGadget]:
    """Extract gadget-like structures from a labeled NetworkX graph.

    Identifies nodes with non-zero phase connected via Hadamard edges
    to other nodes, which is the characteristic pattern of phase gadgets
    in featurized ZX graphs. Returns gadgets with relative node indices
    (not qubit indices, since NetworkX graphs from featurizer don't have
    boundary-to-qubit mapping).
    """
    gadgets = []
    for node, data in nxg.nodes(data=True):
        phase_class = data.get("phase_class", "zero")
        if phase_class == "zero":
            continue
        if data.get("is_boundary", False):
            continue

        # Find Hadamard-connected neighbors (potential qubit-line spiders)
        had_neighbors = set()
        for nbr in nxg.neighbors(node):
            edata = nxg.edges[node, nbr]
            if edata.get("edge_type") == "HADAMARD":
                had_neighbors.add(nbr)

        # Use the node's Hadamard neighbors as the "support"
        # (these are spider IDs, not qubit IDs, but capture the structure)
        support = frozenset(had_neighbors) if had_neighbors else frozenset([node])
        n = max(nxg.nodes()) + 1

        # Map phase class to representative fraction
        phase_map = {
            "pauli": Fraction(1, 1),
            "clifford": Fraction(1, 2),
            "t_like": Fraction(1, 4),
            "arbitrary": Fraction(1, 3),  # sentinel for non-standard
        }
        phase = phase_map.get(phase_class, Fraction(0))
        if phase != 0:
            gadgets.append(PhaseGadget(support=support, phase=phase, n_qubits=n))

    return gadgets


# ── Symmetry Analysis ───────────────────────────────────────────────


def compute_symmetry_tags(poly: PhasePolynomial) -> dict[str, bool]:
    """Compute symmetry tags for a phase polynomial.

    Returns:
        particle_number_preserving: All gadgets commute with total Z (always True for ZZ gadgets).
        z2_parity_preserving: All gadgets have even weight.
        real_valued: All gadgets use only Z Paulis (no Y), always True for phase polynomials.
    """
    return {
        "particle_number_preserving": all(g.is_particle_number_preserving() for g in poly.gadgets),
        "z2_parity_preserving": all(g.is_z2_parity_preserving() for g in poly.gadgets),
        "real_valued": all(g.is_real_valued() for g in poly.gadgets),
    }


def compute_symmetry_tags_from_pauli_strings(
    pauli_strings: list[str],
) -> dict[str, bool]:
    """Compute symmetry tags from explicit Pauli strings (e.g. 'XYZZ').

    Handles full Pauli algebra (X, Y, Z, I), not just Z-type gadgets.

    Returns:
        particle_number_preserving: Each string has equal X and Y count.
        z2_parity_preserving: Each string has even total Pauli weight.
        real_valued: No string contains an odd number of Y Paulis.
    """
    particle_preserving = True
    z2_preserving = True
    real_valued = True

    for ps in pauli_strings:
        x_count = ps.count("X")
        y_count = ps.count("Y")
        z_count = ps.count("Z")
        weight = x_count + y_count + z_count

        if x_count != y_count:
            particle_preserving = False
        if weight % 2 != 0:
            z2_preserving = False
        if y_count % 2 != 0:
            real_valued = False

    return {
        "particle_number_preserving": particle_preserving,
        "z2_parity_preserving": z2_preserving,
        "real_valued": real_valued,
    }
