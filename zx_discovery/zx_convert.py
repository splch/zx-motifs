"""
Step 2 – Convert Quantum Circuits to Simplified ZX-Diagrams
============================================================
Uses PyZX to parse OpenQASM 2.0 strings into ZX-diagrams, then applies
a cascade of simplification strategies:

    1. ``interior_clifford_simp`` – fuse adjacent Clifford spiders.
    2. ``clifford_simp``          – full Clifford simplification.
    3. ``full_reduce``            – strongest built-in reduction (may
       destroy circuit structure but minimises spider count).

Each algorithm receives *two* diagram snapshots:
    • **clifford** – after ``clifford_simp`` (circuit-like structure preserved).
    • **reduced**  – after ``full_reduce`` (maximally compressed).

The reduced diagrams are the primary input for sub-diagram mining
because they expose the core computational "skeleton" of each algorithm.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import copy

import pyzx as zx


@dataclass
class ZXEntry:
    """Holds both simplification levels of a single algorithm's ZX-diagram."""
    name: str
    original_graph: zx.Graph       # straight from QASM, no simplification
    clifford_graph: zx.Graph       # after clifford_simp
    reduced_graph: zx.Graph        # after full_reduce
    original_stats: Dict[str, int] = field(default_factory=dict)
    reduced_stats: Dict[str, int]  = field(default_factory=dict)


def _graph_stats(g: zx.Graph) -> Dict[str, int]:
    """Collect basic statistics about a ZX-diagram."""
    types = g.types()
    n_z = sum(1 for v in g.vertices() if types[v] == zx.VertexType.Z)
    n_x = sum(1 for v in g.vertices() if types[v] == zx.VertexType.X)
    n_b = sum(1 for v in g.vertices() if types[v] == zx.VertexType.BOUNDARY)
    n_h = 0
    try:
        n_h = sum(1 for v in g.vertices() if types[v] == zx.VertexType.H_BOX)
    except AttributeError:
        pass
    return {
        "vertices": g.num_vertices(),
        "edges":    g.num_edges(),
        "z_spiders": n_z,
        "x_spiders": n_x,
        "boundaries": n_b,
        "h_boxes": n_h,
    }


def qasm_to_zx(qasm_str: str) -> Optional[zx.Graph]:
    """Parse an OpenQASM 2.0 string into a PyZX Graph.

    Returns None if parsing fails (e.g. unsupported gate).
    """
    try:
        circuit = zx.Circuit.from_qasm(qasm_str)
        return circuit.to_graph()
    except Exception:
        return None


def simplify_clifford(g: zx.Graph) -> zx.Graph:
    """Apply Clifford simplification (preserves circuit-like structure)."""
    gc = copy.deepcopy(g)
    zx.simplify.clifford_simp(gc, quiet=True)
    gc.normalize()
    return gc


def simplify_full(g: zx.Graph) -> zx.Graph:
    """Apply full_reduce (maximum compression, may lose circuit form)."""
    gr = copy.deepcopy(g)
    zx.simplify.full_reduce(gr, quiet=True)
    gr.normalize()
    return gr


def convert_corpus(qasm_corpus: Dict[str, str]) -> Dict[str, ZXEntry]:
    """Convert an entire QASM corpus to ZXEntry objects.

    Parameters
    ----------
    qasm_corpus : dict
        Mapping of algorithm name → OpenQASM 2.0 string.

    Returns
    -------
    dict
        Mapping of algorithm name → ZXEntry.
    """
    entries: Dict[str, ZXEntry] = {}
    for name, qasm_str in qasm_corpus.items():
        g = qasm_to_zx(qasm_str)
        if g is None:
            print(f"  [skip] '{name}' – QASM parse failed")
            continue

        gc = simplify_clifford(g)
        gr = simplify_full(g)

        entry = ZXEntry(
            name=name,
            original_graph=copy.deepcopy(g),
            clifford_graph=gc,
            reduced_graph=gr,
            original_stats=_graph_stats(g),
            reduced_stats=_graph_stats(gr),
        )
        entries[name] = entry
    return entries
