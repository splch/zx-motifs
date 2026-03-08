"""
Step 4 - Compose New Candidate Algorithms from ZX-Webs
========================================================
Given a library of ZX-Web motifs, this module constructs new
ZX-diagrams by *gluing* compatible webs together.

Composition Strategy
--------------------
Two webs can be **glued** if they share a compatible *port*:
a vertex on the boundary of the sub-diagram whose type (Z/X) and
phase class match.  Gluing identifies (merges) the two matching
ports into a single spider, fusing their phases (addition mod 2π),
in accordance with the ZX-calculus spider fusion rule.

The composer:
    1. Selects 2-4 webs from the library (combinatorial sampling).
    2. For each pair of adjacent webs in the sequence, finds compatible
       port pairs and fuses them.
    3. Adds fresh boundary (input/output) vertices to any remaining
       "dangling" ports so the diagram has a well-defined I/O signature.
    4. Returns the result as a PyZX ``Graph``.

Diversity Controls
------------------
* ``max_candidates`` caps total output.
* ``max_web_reuse`` prevents the same web from dominating.
* ``require_distinct_sources`` ensures webs come from ≥ 2 algorithms.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import copy
import itertools
import random

import pyzx as zx
import networkx as nx

from .zx_webs import ZXWeb


@dataclass
class CandidateAlgorithm:
    """A candidate quantum algorithm composed from ZX-Webs."""
    candidate_id: str
    graph: zx.Graph
    source_webs: List[str]          # web_ids used
    source_algorithms: Set[str]     # originating algorithm names
    n_qubits: int = 0
    composition_log: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Port extraction
# ---------------------------------------------------------------------------

def _find_ports(G: nx.Graph, nodes: frozenset) -> List[int]:
    """Return vertices in *nodes* that have at least one neighbour outside.

    These are the "boundary" of the subgraph and serve as glue points.
    Internal boundaries (VertexType BOUNDARY) are also included.
    """
    ports = []
    for v in nodes:
        is_boundary = G.nodes[v].get("vtype") == "B"
        has_external_nb = any(nb not in nodes for nb in G.neighbors(v))
        if is_boundary or has_external_nb:
            ports.append(v)
    return ports


def _compatible_ports(
    web_a_labels: Dict, web_b_labels: Dict
) -> bool:
    """Check if two port vertices are compatible for gluing.

    Compatibility requires matching spider type (Z-Z or X-X) so that
    the spider-fusion rule applies.  Boundary vertices are universally
    compatible because they carry no type.
    """
    ta, tb = web_a_labels.get("vtype"), web_b_labels.get("vtype")
    if ta == "B" or tb == "B":
        return True  # boundaries glue with anything
    return ta == tb   # same-colour spider fusion


# ---------------------------------------------------------------------------
# Core graph gluing
# ---------------------------------------------------------------------------

def _networkx_to_pyzx(G: nx.Graph, n_inputs: int = 0) -> zx.Graph:
    """Convert a labelled NetworkX graph back to a PyZX Graph.

    Assigns qubit indices round-robin and uses row indices to create
    a left-to-right layout.
    """
    g = zx.Graph()
    vtype_map = {"Z": zx.VertexType.Z, "X": zx.VertexType.X,
                 "B": zx.VertexType.BOUNDARY}
    etype_map = {"S": zx.EdgeType.SIMPLE, "H": zx.EdgeType.HADAMARD}

    nx_to_pyzx: Dict[int, int] = {}
    qubit = 0
    row = 0
    for v in sorted(G.nodes()):
        nd = G.nodes[v]
        vt = vtype_map.get(nd.get("vtype", "Z"), zx.VertexType.Z)
        phase = nd.get("raw_phase", 0)
        pv = g.add_vertex(vt, qubit=qubit % 8, row=row, phase=phase)
        nx_to_pyzx[v] = pv
        qubit += 1
        row += 1

    for u, v, d in G.edges(data=True):
        et = etype_map.get(d.get("etype", "S"), zx.EdgeType.SIMPLE)
        g.add_edge(g.edge(nx_to_pyzx[u], nx_to_pyzx[v]), et)

    return g


def _glue_webs(
    web_graphs: List[nx.Graph],
    rng: random.Random,
) -> Optional[nx.Graph]:
    """Glue a sequence of web subgraphs into a single NetworkX graph.

    Merges compatible ports between consecutive webs using the
    spider-fusion rule (identify vertices, add phases).
    """
    if not web_graphs:
        return None

    # Start with a copy of the first web, relabelled to avoid ID clashes
    counter = 0

    def relabel(G: nx.Graph) -> Tuple[nx.Graph, Dict[int, int]]:
        nonlocal counter
        mapping = {}
        for v in sorted(G.nodes()):
            mapping[v] = counter
            counter += 1
        return nx.relabel_nodes(G, mapping), mapping

    composite, _ = relabel(web_graphs[0].copy())

    for i in range(1, len(web_graphs)):
        next_web, new_map = relabel(web_graphs[i].copy())

        # Find ports on composite and next_web
        composite_ports = [
            v for v in composite.nodes()
            if composite.degree(v) <= 2
        ]
        next_ports = [
            v for v in next_web.nodes()
            if next_web.degree(v) <= 2
        ]

        if not composite_ports or not next_ports:
            # No glue point → just merge disjointly
            composite = nx.compose(composite, next_web)
            continue

        # Pick a compatible pair at random
        rng.shuffle(composite_ports)
        rng.shuffle(next_ports)
        glued = False
        for cp in composite_ports:
            for np_ in next_ports:
                if _compatible_ports(
                    composite.nodes[cp], next_web.nodes[np_]
                ):
                    # Spider fusion: merge np_ into cp
                    # 1. Add all edges of np_ to cp
                    merged = nx.compose(composite, next_web)
                    for nb in list(next_web.neighbors(np_)):
                        if nb != np_ and merged.has_node(nb):
                            edata = next_web.edges[np_, nb]
                            if not merged.has_edge(cp, nb):
                                merged.add_edge(cp, nb, **edata)
                    # 2. Fuse phases
                    old_phase = composite.nodes[cp].get("raw_phase", 0)
                    new_phase = next_web.nodes[np_].get("raw_phase", 0)
                    merged.nodes[cp]["raw_phase"] = (old_phase + new_phase) % 2
                    # 3. Remove the redundant vertex
                    if merged.has_node(np_):
                        merged.remove_node(np_)
                    composite = merged
                    glued = True
                    break
            if glued:
                break
        if not glued:
            composite = nx.compose(composite, next_web)

    return composite


def _add_io_boundaries(G: nx.Graph) -> nx.Graph:
    """Add BOUNDARY vertices as inputs/outputs to dangling ports."""
    G = G.copy()
    dangling = [v for v in G.nodes() if G.degree(v) <= 1
                and G.nodes[v].get("vtype") != "B"]
    counter = max(G.nodes(), default=0) + 1
    for v in dangling:
        G.add_node(counter, vtype="B", phase_class="zero", raw_phase=0)
        G.add_edge(v, counter, etype="S")
        counter += 1
    return G


# ---------------------------------------------------------------------------
# Public composition interface
# ---------------------------------------------------------------------------

def compose_candidates(
    webs: Dict[str, ZXWeb],
    corpus_nx: Dict[str, nx.Graph],
    max_candidates: int = 200,
    web_combo_size: int = 3,
    max_web_reuse: int = 2,
    require_distinct_sources: bool = True,
    seed: int = 42,
) -> List[CandidateAlgorithm]:
    """Compose new candidate ZX-diagrams from ZX-Web motifs.

    Parameters
    ----------
    webs : dict
        ``{web_id: ZXWeb}`` from the mining step.
    corpus_nx : dict
        NetworkX representations of the corpus diagrams.
    max_candidates : int
        Maximum number of candidates to generate.
    web_combo_size : int
        Number of webs to combine per candidate (2-4).
    max_web_reuse : int
        Max times a single web can appear in one candidate.
    require_distinct_sources : bool
        If True, the webs in a candidate must span ≥ 2 source algorithms.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    list of CandidateAlgorithm
    """
    rng = random.Random(seed)
    web_list = list(webs.values())
    if len(web_list) < 2:
        return []

    candidates: List[CandidateAlgorithm] = []
    seen_hashes: Set[str] = set()
    attempts = 0
    max_attempts = max_candidates * 20

    while len(candidates) < max_candidates and attempts < max_attempts:
        attempts += 1
        # Sample webs
        combo_size = min(web_combo_size, len(web_list))
        chosen = rng.choices(web_list, k=combo_size)

        # Enforce max_web_reuse
        from collections import Counter
        wid_counts = Counter(w.web_id for w in chosen)
        if any(c > max_web_reuse for c in wid_counts.values()):
            continue

        # Enforce distinct sources
        if require_distinct_sources:
            all_sources = set()
            for w in chosen:
                all_sources.update(w.sources.keys())
            if len(all_sources) < 2:
                continue

        # Build sub-graphs from exemplars
        web_nx_copies = []
        for w in chosen:
            src_g = corpus_nx.get(w.exemplar_source)
            if src_g is None:
                break
            sub = src_g.subgraph(w.exemplar_nodes).copy()
            web_nx_copies.append(sub)
        else:
            # Glue them
            composite_nx = _glue_webs(web_nx_copies, rng)
            if composite_nx is None or composite_nx.number_of_nodes() < 3:
                continue

            composite_nx = _add_io_boundaries(composite_nx)

            # De-duplicate by simple hash
            sig = str(sorted(composite_nx.degree()))
            if sig in seen_hashes:
                continue
            seen_hashes.add(sig)

            # Convert to PyZX
            pyzx_g = _networkx_to_pyzx(composite_nx)
            # Try to set inputs/outputs
            boundaries = [v for v in pyzx_g.vertices()
                          if pyzx_g.type(v) == zx.VertexType.BOUNDARY]
            n_bounds = len(boundaries)
            half = n_bounds // 2
            if half > 0:
                pyzx_g.set_inputs(tuple(boundaries[:half]))
                pyzx_g.set_outputs(tuple(boundaries[half:2*half]))

            n_q = half if half > 0 else 0
            source_algos = set()
            for w in chosen:
                source_algos.update(w.sources.keys())

            cand = CandidateAlgorithm(
                candidate_id=f"cand_{len(candidates):04d}",
                graph=pyzx_g,
                source_webs=[w.web_id for w in chosen],
                source_algorithms=source_algos,
                n_qubits=n_q,
                composition_log=[
                    f"Glued {combo_size} webs: "
                    + ", ".join(w.web_id[:8] for w in chosen)
                ],
            )
            candidates.append(cand)

    return candidates
