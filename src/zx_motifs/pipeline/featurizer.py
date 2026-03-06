"""
Convert PyZX graphs into labeled NetworkX graphs with rich node/edge attributes.
These attributes are critical for meaningful subgraph isomorphism —
without them, you're just doing graph isomorphism on the topology,
which misses the entire point of ZX.
"""
from fractions import Fraction

import networkx as nx
import numpy as np
from pyzx.graph import Graph
from pyzx.utils import EdgeType, VertexType


# ── Phase Classification ────────────────────────────────────────────


def classify_phase(phase: Fraction) -> str:
    """
    Classify a ZX phase into a coarsened equivalence class.

    In ZX-calculus, phases are multiples of π:
      0        → "zero"      (identity spider)
      π        → "pauli"     (X or Z gate)
      π/2,3π/2 → "clifford"  (S, S† gates)
      π/4 etc  → "t_like"    (T, T† gates)
      other    → "arbitrary"  (continuous rotations)
    """
    if phase == 0:
        return "zero"
    if phase == Fraction(1, 1):  # π
        return "pauli"
    if phase.denominator <= 2:  # π/2, 3π/2
        return "clifford"
    if phase.denominator == 4:  # π/4, 3π/4, 5π/4, 7π/4
        return "t_like"
    return "arbitrary"


# ── ZX → NetworkX Conversion ────────────────────────────────────────

_TYPE_MAP = {
    VertexType.Z: "Z",
    VertexType.X: "X",
    VertexType.H_BOX: "H_BOX",
    VertexType.BOUNDARY: "BOUNDARY",
}

_EDGE_TYPE_MAP = {
    EdgeType.SIMPLE: "SIMPLE",
    EdgeType.HADAMARD: "HADAMARD",
}


def pyzx_to_networkx(g: Graph, coarsen_phases: bool = False) -> nx.Graph:
    """
    Convert a PyZX graph to a NetworkX graph with labeled nodes and edges.

    Args:
        g: PyZX Graph object.
        coarsen_phases: If True, use phase_class instead of exact phase
            for the composite label. This lets you find motifs that
            share structure but differ in specific angles.

    Returns:
        NetworkX Graph with node attributes (vertex_type, phase,
        phase_class, degree, is_boundary, label) and edge attributes
        (edge_type).
    """
    nxg = nx.Graph()

    for v in g.vertices():
        vtype = _TYPE_MAP.get(g.type(v), "UNKNOWN")
        phase = g.phase(v)
        phase_cls = classify_phase(phase)
        # Use list() because PyZX neighbors() returns dict_keys
        degree = len(list(g.neighbors(v)))

        if coarsen_phases:
            label = f"{vtype}_{phase_cls}_d{degree}"
        else:
            label = f"{vtype}_{phase}_d{degree}"

        nxg.add_node(
            v,
            vertex_type=vtype,
            phase=str(phase),
            phase_class=phase_cls,
            degree=degree,
            is_boundary=(vtype == "BOUNDARY"),
            label=label,
        )

    for e in g.edges():
        src, tgt = g.edge_st(e)
        etype = _EDGE_TYPE_MAP.get(g.edge_type(e), "UNKNOWN")
        nxg.add_edge(src, tgt, edge_type=etype)

    return nxg


# ── Neighborhood Extraction ─────────────────────────────────────────


def extract_local_neighborhood(
    nxg: nx.Graph, center: int, radius: int = 2
) -> nx.Graph:
    """
    Extract the subgraph within `radius` hops of `center`.
    Useful for building candidate motif templates from interesting vertices.
    """
    nodes = set()
    frontier = {center}
    for _ in range(radius):
        next_frontier = set()
        for n in frontier:
            for nbr in nxg.neighbors(n):
                if nbr not in nodes and nbr not in frontier:
                    next_frontier.add(nbr)
        nodes.update(frontier)
        frontier = next_frontier
    nodes.update(frontier)
    return nxg.subgraph(nodes).copy()


# ── Graph-Level Features ────────────────────────────────────────────


def compute_graph_features(nxg: nx.Graph) -> dict:
    """
    Compute summary features of a ZX graph for clustering/comparison.
    These are NOT used for subgraph matching — they're for quickly
    comparing overall graph structure across algorithms.
    """
    if nxg.number_of_nodes() == 0:
        return {"n_nodes": 0}

    type_counts: dict[str, int] = {}
    phase_class_counts: dict[str, int] = {}
    for _, data in nxg.nodes(data=True):
        vt = data.get("vertex_type", "UNK")
        pc = data.get("phase_class", "UNK")
        type_counts[vt] = type_counts.get(vt, 0) + 1
        phase_class_counts[pc] = phase_class_counts.get(pc, 0) + 1

    degrees = [d for _, d in nxg.degree()]

    had_edges = sum(
        1 for _, _, d in nxg.edges(data=True) if d.get("edge_type") == "HADAMARD"
    )
    simple_edges = nxg.number_of_edges() - had_edges

    return {
        "n_nodes": nxg.number_of_nodes(),
        "n_edges": nxg.number_of_edges(),
        "n_hadamard_edges": had_edges,
        "n_simple_edges": simple_edges,
        "hadamard_ratio": had_edges / max(nxg.number_of_edges(), 1),
        "avg_degree": sum(degrees) / len(degrees),
        "max_degree": max(degrees),
        "type_counts": type_counts,
        "phase_class_counts": phase_class_counts,
        "density": nx.density(nxg),
        "n_connected_components": nx.number_connected_components(nxg),
    }


# ── Motif Feature Vectors ─────────────────────────────────────────


def compute_motif_feature_vector(nxg: nx.Graph) -> np.ndarray:
    """
    Compute a fixed-length 12-element feature vector for a motif graph.

    Elements: [n_nodes, n_edges, n_Z, n_X, n_H_BOX, n_zero, n_clifford,
               n_t_like, n_arbitrary, n_hadamard_edges, n_simple_edges, density]
    """
    n_nodes = nxg.number_of_nodes()
    n_edges = nxg.number_of_edges()

    type_counts = {"Z": 0, "X": 0, "H_BOX": 0}
    phase_counts = {"zero": 0, "clifford": 0, "t_like": 0, "arbitrary": 0}

    for _, data in nxg.nodes(data=True):
        vt = data.get("vertex_type", "")
        pc = data.get("phase_class", "")
        if vt in type_counts:
            type_counts[vt] += 1
        if pc in phase_counts:
            phase_counts[pc] += 1

    n_hadamard = sum(
        1 for _, _, d in nxg.edges(data=True) if d.get("edge_type") == "HADAMARD"
    )
    n_simple = n_edges - n_hadamard

    density = nx.density(nxg) if n_nodes > 0 else 0.0

    return np.array([
        n_nodes, n_edges,
        type_counts["Z"], type_counts["X"], type_counts["H_BOX"],
        phase_counts["zero"], phase_counts["clifford"],
        phase_counts["t_like"], phase_counts["arbitrary"],
        n_hadamard, n_simple, density,
    ], dtype=float)


def motif_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity between two motif feature vectors, in [0, 1].
    Returns 0.0 if either vector is all-zero.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
