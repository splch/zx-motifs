"""
Convert Qiskit circuits → ZX diagrams at multiple simplification levels.
Each level reveals different structural properties.
"""
import copy
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

import pyzx as zx
from pyzx.graph import Graph
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm2_dumps


class SimplificationLevel(Enum):
    RAW = "raw"
    SPIDER_FUSED = "spider_fused"
    INTERIOR_CLIFFORD = "interior_cliff"
    CLIFFORD_SIMP = "clifford_simp"
    FULL_REDUCE = "full_reduce"
    TELEPORT_REDUCE = "teleport_reduce"


@dataclass
class ZXSnapshot:
    """A ZX diagram at a specific simplification level, with metadata."""

    graph: Graph
    level: SimplificationLevel
    algorithm_name: str
    n_qubits: int
    num_vertices: int
    num_edges: int
    num_t_gates: int
    has_circuit_structure: bool

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm_name,
            "level": self.level.value,
            "n_qubits": self.n_qubits,
            "vertices": self.num_vertices,
            "edges": self.num_edges,
            "t_gates": self.num_t_gates,
            "circuit_like": self.has_circuit_structure,
        }


def count_t_gates(g: Graph) -> int:
    """Count spiders with phase π/4 or 7π/4 (T and T†)."""
    count = 0
    for v in g.vertices():
        phase = g.phase(v)
        if phase == Fraction(1, 4) or phase == Fraction(7, 4):
            count += 1
    return count


def has_circuit_like_structure(g: Graph) -> bool:
    """Heuristic: circuit-like if most vertices have low degree."""
    degrees = [len(list(g.neighbors(v))) for v in g.vertices()]
    if not degrees:
        return False
    avg_degree = sum(degrees) / len(degrees)
    max_degree = max(degrees)
    return avg_degree < 4.0 and max_degree < 10


def qiskit_to_zx(qc: QuantumCircuit) -> zx.Circuit:
    """Convert a Qiskit QuantumCircuit to a PyZX Circuit via QASM2."""
    qasm_str = qasm2_dumps(qc)
    return zx.Circuit.from_qasm(qasm_str)


def _make_snapshot(
    g: Graph,
    level: SimplificationLevel,
    algorithm_name: str,
    n_qubits: int,
) -> ZXSnapshot:
    return ZXSnapshot(
        graph=g,
        level=level,
        algorithm_name=algorithm_name,
        n_qubits=n_qubits,
        num_vertices=g.num_vertices(),
        num_edges=g.num_edges(),
        num_t_gates=count_t_gates(g),
        has_circuit_structure=has_circuit_like_structure(g),
    )


_SIMPLIFIERS = [
    (SimplificationLevel.RAW, None, False),
    (SimplificationLevel.SPIDER_FUSED, zx.simplify.spider_simp, False),
    (SimplificationLevel.INTERIOR_CLIFFORD, zx.simplify.interior_clifford_simp, False),
    (SimplificationLevel.CLIFFORD_SIMP, zx.simplify.clifford_simp, False),
    (SimplificationLevel.FULL_REDUCE, zx.simplify.full_reduce, False),
    # teleport_reduce returns a new graph unlike the others
    (SimplificationLevel.TELEPORT_REDUCE, zx.simplify.teleport_reduce, True),
]


def convert_at_all_levels(
    qc: QuantumCircuit, algorithm_name: str
) -> list[ZXSnapshot]:
    """
    Convert a Qiskit circuit to ZX diagrams at every simplification level.
    Returns a list of ZXSnapshot objects for comparison.
    """
    zx_circ = qiskit_to_zx(qc)
    g_raw = zx_circ.to_graph()
    snapshots = []

    for level, simplifier, returns_graph in _SIMPLIFIERS:
        g = copy.deepcopy(g_raw)
        if simplifier is not None:
            result = simplifier(g)
            if returns_graph:
                g = result
        snapshots.append(_make_snapshot(g, level, algorithm_name, qc.num_qubits))

    return snapshots
