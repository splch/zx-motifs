"""
Composable ZX box library.

A 'box' is a ZX subdiagram with explicitly typed boundaries:
  - Left boundary (inputs): ordered list of boundary spiders
  - Right boundary (outputs): ordered list of boundary spiders
  - Interior: the motif subgraph

Composition connects the outputs of one box to the inputs of another.
Two composition modes:
  - Sequential (;):  box_a outputs → box_b inputs
  - Parallel   (⊗): stack side-by-side

The boundary contract specifies:
  - Number of input/output wires
  - Type of each boundary spider (Z or X)
  - Phase constraints on boundary spiders (usually phaseless)
"""
import copy
from dataclasses import dataclass, field

import pyzx as zx
from pyzx.graph import Graph
from pyzx.utils import EdgeType, VertexType
from qiskit import QuantumCircuit

from .converter import qiskit_to_zx


class BoundaryDestroyedError(Exception):
    """Raised when simplification destroys boundary vertices."""


@dataclass
class BoundarySpec:
    """Specification of one side of a box's boundary."""

    n_wires: int
    wire_types: list  # [VertexType.Z, ...] for each wire
    phases: list  # Phase of each boundary spider (usually 0)


@dataclass
class ZXBox:
    """
    A composable ZX diagram fragment with explicit boundaries.

    This is a morphism in the category of ZX-diagrams:
    domain (left boundary) → codomain (right boundary).
    """

    name: str
    graph: Graph
    left_boundary: list[int]  # Vertex IDs of input boundary spiders
    right_boundary: list[int]  # Vertex IDs of output boundary spiders
    left_spec: BoundarySpec
    right_spec: BoundarySpec
    description: str = ""
    tags: list = field(default_factory=list)
    semantic_role: str = ""

    @property
    def n_inputs(self) -> int:
        return len(self.left_boundary)

    @property
    def n_outputs(self) -> int:
        return len(self.right_boundary)


def make_box_from_circuit(
    name: str, qc: QuantumCircuit, semantic_role: str = ""
) -> ZXBox:
    """
    Create a ZXBox from a Qiskit QuantumCircuit.
    The circuit's input/output qubits become the box boundaries.
    """
    zx_circ = qiskit_to_zx(qc)
    g = zx_circ.to_graph()

    inputs = list(g.inputs())
    outputs = list(g.outputs())

    left_spec = BoundarySpec(
        n_wires=len(inputs),
        wire_types=[g.type(v) for v in inputs],
        phases=[g.phase(v) for v in inputs],
    )

    right_spec = BoundarySpec(
        n_wires=len(outputs),
        wire_types=[g.type(v) for v in outputs],
        phases=[g.phase(v) for v in outputs],
    )

    return ZXBox(
        name=name,
        graph=g,
        left_boundary=inputs,
        right_boundary=outputs,
        left_spec=left_spec,
        right_spec=right_spec,
        semantic_role=semantic_role,
    )


def compose_sequential(box_a: ZXBox, box_b: ZXBox) -> ZXBox:
    """
    Compose two boxes sequentially: box_a ; box_b
    (connect box_a's outputs to box_b's inputs).

    Uses PyZX's built-in Graph.compose() when possible (fast path),
    falls back to manual vertex remapping when boundary structure
    doesn't meet compose()'s requirements.
    """
    if box_a.n_outputs != box_b.n_inputs:
        raise ValueError(
            f"Boundary mismatch: {box_a.name} has {box_a.n_outputs} outputs, "
            f"{box_b.name} has {box_b.n_inputs} inputs"
        )

    # Try fast path: PyZX built-in composition
    # Requires each boundary vertex to have exactly one neighbor.
    try:
        return _compose_via_pyzx(box_a, box_b)
    except Exception:
        return _compose_manual(box_a, box_b)


def _compose_via_pyzx(box_a: ZXBox, box_b: ZXBox) -> ZXBox:
    """
    Fast path using PyZX's Graph.compose() (the + operator).
    This connects outputs of g_a to inputs of g_b by matching them in order.
    """
    g_a = copy.deepcopy(box_a.graph)
    g_b = copy.deepcopy(box_b.graph)

    # Graph.compose() mutates g_a in place and remaps g_b's vertices.
    # We need to track what g_b's boundary vertices become after remapping.
    # The compose method adds g_b's vertices with new IDs.
    old_b_outputs = list(g_b.outputs())

    g_a.compose(g_b)

    # After compose, g_a's outputs are now what g_b's outputs were (remapped).
    # PyZX sets the composed graph's outputs to g_b's (remapped) outputs.
    new_outputs = list(g_a.outputs())
    new_inputs = list(g_a.inputs())

    left_spec = box_a.left_spec
    right_spec = box_b.right_spec

    return ZXBox(
        name=f"{box_a.name}_then_{box_b.name}",
        graph=g_a,
        left_boundary=new_inputs,
        right_boundary=new_outputs,
        left_spec=left_spec,
        right_spec=right_spec,
        semantic_role=f"({box_a.semantic_role} ; {box_b.semantic_role})",
    )


def _compose_manual(box_a: ZXBox, box_b: ZXBox) -> ZXBox:
    """
    Manual composition: remap box_b's vertices into box_a's graph,
    then fuse boundary vertices.
    """
    g = copy.deepcopy(box_a.graph)
    g_b = copy.deepcopy(box_b.graph)

    # Find the next safe vertex ID
    max_id = max(g.vertices()) + 1 if list(g.vertices()) else 0

    # Remap box_b's vertex IDs
    id_map: dict[int, int] = {}
    for v in g_b.vertices():
        new_id = v + max_id
        id_map[v] = new_id
        g.add_vertex(
            ty=g_b.type(v),
            phase=g_b.phase(v),
            qubit=g_b.qubit(v),
            row=g_b.row(v),
            index=new_id,
        )

    # Add box_b's edges
    for e in g_b.edges():
        src, tgt = g_b.edge_st(e)
        g.add_edge((id_map[src], id_map[tgt]), g_b.edge_type(e))

    # Fuse: connect box_a's outputs to box_b's inputs
    a_outputs = list(box_a.right_boundary)
    b_inputs = [id_map[v] for v in box_b.left_boundary]

    for a_out, b_in in zip(a_outputs, b_inputs):
        # Reconnect all of b_in's neighbors to a_out
        for nbr in list(g.neighbors(b_in)):
            if nbr != a_out and not g.connected(a_out, nbr):
                etype = g.edge_type(g.edge(b_in, nbr))
                g.add_edge((a_out, nbr), etype)
        g.remove_vertex(b_in)

    new_right = [id_map[v] for v in box_b.right_boundary]

    # Update inputs/outputs tuples on the graph
    g.set_inputs(tuple(box_a.left_boundary))
    g.set_outputs(tuple(new_right))

    return ZXBox(
        name=f"{box_a.name}_then_{box_b.name}",
        graph=g,
        left_boundary=list(box_a.left_boundary),
        right_boundary=new_right,
        left_spec=box_a.left_spec,
        right_spec=box_b.right_spec,
        semantic_role=f"({box_a.semantic_role} ; {box_b.semantic_role})",
    )


def compose_parallel(box_a: ZXBox, box_b: ZXBox) -> ZXBox:
    """
    Compose two boxes in parallel (tensor product): box_a ⊗ box_b.
    """
    g = copy.deepcopy(box_a.graph)
    g_b = copy.deepcopy(box_b.graph)

    max_id = max(g.vertices()) + 1 if list(g.vertices()) else 0
    id_map: dict[int, int] = {}

    for v in g_b.vertices():
        new_id = v + max_id
        id_map[v] = new_id
        g.add_vertex(
            ty=g_b.type(v),
            phase=g_b.phase(v),
            qubit=g_b.qubit(v),
            row=g_b.row(v),
            index=new_id,
        )

    for e in g_b.edges():
        src, tgt = g_b.edge_st(e)
        g.add_edge((id_map[src], id_map[tgt]), g_b.edge_type(e))

    new_left = list(box_a.left_boundary) + [id_map[v] for v in box_b.left_boundary]
    new_right = list(box_a.right_boundary) + [id_map[v] for v in box_b.right_boundary]

    g.set_inputs(tuple(new_left))
    g.set_outputs(tuple(new_right))

    return ZXBox(
        name=f"{box_a.name}_par_{box_b.name}",
        graph=g,
        left_boundary=new_left,
        right_boundary=new_right,
        left_spec=BoundarySpec(
            n_wires=box_a.left_spec.n_wires + box_b.left_spec.n_wires,
            wire_types=box_a.left_spec.wire_types + box_b.left_spec.wire_types,
            phases=box_a.left_spec.phases + box_b.left_spec.phases,
        ),
        right_spec=BoundarySpec(
            n_wires=box_a.right_spec.n_wires + box_b.right_spec.n_wires,
            wire_types=box_a.right_spec.wire_types + box_b.right_spec.wire_types,
            phases=box_a.right_spec.phases + box_b.right_spec.phases,
        ),
        semantic_role=f"({box_a.semantic_role} ⊗ {box_b.semantic_role})",
    )


def simplify_box(box: ZXBox, level: str = "interior_clifford") -> ZXBox:
    """
    Simplify the interior of a box.

    Only 'spider_fusion' and 'interior_clifford' are guaranteed to
    preserve boundary vertices. Higher levels will be validated and
    raise BoundaryDestroyedError if boundaries are lost.
    """
    g = copy.deepcopy(box.graph)
    boundary_ids = set(box.left_boundary) | set(box.right_boundary)

    if level == "spider_fusion":
        zx.simplify.spider_simp(g)
    elif level == "interior_clifford":
        zx.simplify.interior_clifford_simp(g)
    elif level == "clifford":
        zx.simplify.clifford_simp(g)
    elif level == "full":
        zx.simplify.full_reduce(g)
    else:
        raise ValueError(f"Unknown simplification level: {level}")

    # Validate boundaries survived
    surviving = set(g.vertices())
    lost = boundary_ids - surviving
    if lost:
        raise BoundaryDestroyedError(
            f"Simplification '{level}' destroyed {len(lost)} boundary vertices: {lost}. "
            f"Use 'interior_clifford' or 'spider_fusion' for boundary-safe simplification."
        )

    return ZXBox(
        name=f"{box.name}_simplified",
        graph=g,
        left_boundary=box.left_boundary,
        right_boundary=box.right_boundary,
        left_spec=box.left_spec,
        right_spec=box.right_spec,
        semantic_role=box.semantic_role,
    )
