"""
Interaction graph construction from phase gadgets.

Given a list of phase gadgets for an algorithm, builds a weighted graph where
vertices are qubits and edge weight is the number of gadgets acting jointly
on each qubit pair. This drives problem-aware motif selection in MAGIC.
"""
from __future__ import annotations

from itertools import combinations

import networkx as nx

from .phase_poly import PhaseGadget, PhasePolynomial


def build_interaction_graph(poly: PhasePolynomial) -> nx.Graph:
    """Build a weighted qubit interaction graph from a phase polynomial.

    Vertices are qubit indices (0..n_qubits-1). An edge (i, j) exists
    if at least one gadget acts on both qubits i and j. Edge weight
    is the number of such gadgets.

    Single-qubit gadgets (weight 1) add self-loops with their count.
    """
    g = nx.Graph()
    g.add_nodes_from(range(poly.n_qubits))

    edge_counts: dict[tuple[int, int], int] = {}
    for gadget in poly.gadgets:
        if gadget.weight < 2:
            continue
        for i, j in combinations(sorted(gadget.support), 2):
            key = (i, j)
            edge_counts[key] = edge_counts.get(key, 0) + 1

    for (i, j), w in edge_counts.items():
        g.add_edge(i, j, weight=w)

    return g


def interaction_density(g: nx.Graph) -> float:
    """Fraction of possible edges that have non-zero interaction weight."""
    n = g.number_of_nodes()
    if n < 2:
        return 0.0
    return g.number_of_edges() / (n * (n - 1) / 2)


def max_interaction_weight(g: nx.Graph) -> int:
    """Maximum edge weight in the interaction graph."""
    if g.number_of_edges() == 0:
        return 0
    return max(d.get("weight", 1) for _, _, d in g.edges(data=True))


def interaction_degree_sequence(g: nx.Graph) -> list[int]:
    """Weighted degree of each qubit (sum of incident edge weights)."""
    degrees = []
    for node in sorted(g.nodes()):
        wd = sum(d.get("weight", 1) for _, _, d in g.edges(node, data=True))
        degrees.append(wd)
    return degrees


def compare_interaction_graphs(g1: nx.Graph, g2: nx.Graph) -> float:
    """Jaccard similarity between edge sets of two interaction graphs.

    Returns 1.0 if both graphs have identical edge sets, 0.0 if disjoint.
    Only considers edge existence, not weights.
    """
    e1 = set(g1.edges())
    e2 = set(g2.edges())
    if not e1 and not e2:
        return 1.0
    union = e1 | e2
    if not union:
        return 1.0
    return len(e1 & e2) / len(union)
