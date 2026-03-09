"""
Template-based candidate algorithm composition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pyzx as zx
from pyzx.utils import EdgeType, VertexType

from src.mining import Boundary, ZXWeb, WebLibrary

logger = logging.getLogger(__name__)

# ── Role fallbacks and limits ────────────────────────────────────────

_ROLE_FALLBACKS: dict[str, list[str]] = {
    "amplify": ["phase", "entangle"],
    "readout": ["state_prep", "entangle"],
    "decode": ["encode"],
}

_MAX_WEBS_PER_SLOT = 15


# ── Templates ───────────────────────────────────────────────────────


@dataclass
class Slot:
    """One position in an algorithm template."""

    role: str
    min_qubits: int | None = None
    max_qubits: int | None = None


@dataclass
class AlgorithmTemplate:
    """Ordered sequence of slots defining an algorithm's high-level shape."""

    name: str
    slots: list[Slot]
    description: str = ""


def load_templates_from_config(
    template_specs: list[list[str]],
) -> list[AlgorithmTemplate]:
    """Convert config.yaml template specs to AlgorithmTemplate objects."""
    templates: list[AlgorithmTemplate] = []
    for i, spec in enumerate(template_specs):
        slots = [Slot(role=r) for r in spec]
        templates.append(AlgorithmTemplate(name=f"custom_{i}", slots=slots))
    return templates


SEARCH_TEMPLATE = AlgorithmTemplate(
    name="search",
    slots=[
        Slot(role="state_prep"),
        Slot(role="oracle"),
        Slot(role="amplify"),
        Slot(role="readout"),
    ],
    description="Grover-like search: prepare, oracle + amplification, measure.",
)

SIMULATION_TEMPLATE = AlgorithmTemplate(
    name="simulation",
    slots=[
        Slot(role="state_prep"),
        Slot(role="evolve"),
        Slot(role="evolve"),
        Slot(role="readout"),
    ],
    description="Hamiltonian simulation: prepare, evolve steps, measure.",
)

QEC_TEMPLATE = AlgorithmTemplate(
    name="error_correction",
    slots=[
        Slot(role="encode"),
        Slot(role="entangle"),
        Slot(role="decode"),
    ],
    description="Error correction: encode, syndrome extraction, decode.",
)

BUILTIN_TEMPLATES = [SEARCH_TEMPLATE, SIMULATION_TEMPLATE, QEC_TEMPLATE]


# ── Composition ─────────────────────────────────────────────────────


@dataclass
class CompositionRecipe:
    """Records exactly how a candidate was assembled."""

    candidate_id: str
    template_name: str | None
    web_sequence: list[str]
    connections: list[tuple[int, int]] = field(default_factory=list)


def connect_webs(
    web_a: ZXWeb,
    web_b: ZXWeb,
    boundary_map: list[tuple[int, int]],
) -> Any:  # pyzx.Graph
    """Connect two ZXWebs at specified boundary pairs.

    For each pair (out_idx, in_idx) in boundary_map:
      - If both boundary spiders are the same type (Z-Z or X-X) and connected
        by simple edges, fuse them (sum phases, merge neighbors).
      - Otherwise, add an edge between them.
    """
    g = zx.Graph()

    # -- Copy web_a's graph --
    map_a: dict[int, int] = {}
    for v in web_a.graph.vertices():
        new_v = g.add_vertex(
            ty=web_a.graph.type(v),
            qubit=web_a.graph.qubit(v),
            row=web_a.graph.row(v),
            phase=web_a.graph.phase(v),
        )
        map_a[v] = new_v

    for e in web_a.graph.edges():
        src, tgt = web_a.graph.edge_st(e)
        g.add_edge((map_a[src], map_a[tgt]), edgetype=web_a.graph.edge_type(e))

    # -- Compute row offset --
    rows_a = [web_a.graph.row(v) for v in web_a.graph.vertices()]
    row_offset = (max(rows_a) + 1) if rows_a else 1

    # -- Copy web_b's graph with row offset --
    map_b: dict[int, int] = {}
    for v in web_b.graph.vertices():
        new_v = g.add_vertex(
            ty=web_b.graph.type(v),
            qubit=web_b.graph.qubit(v),
            row=web_b.graph.row(v) + row_offset,
            phase=web_b.graph.phase(v),
        )
        map_b[v] = new_v

    for e in web_b.graph.edges():
        src, tgt = web_b.graph.edge_st(e)
        g.add_edge((map_b[src], map_b[tgt]), edgetype=web_b.graph.edge_type(e))

    # -- Connect boundaries --
    out_boundaries = web_a.boundaries[web_a.n_input_boundaries:]
    in_boundaries = web_b.boundaries[:web_b.n_input_boundaries]

    for out_idx, in_idx in boundary_map:
        out_b = out_boundaries[out_idx]
        in_b = in_boundaries[in_idx]

        va = map_a[out_b.vertex_id] if out_b.vertex_id is not None else None
        vb = map_b[in_b.vertex_id] if in_b.vertex_id is not None else None

        if va is None or vb is None:
            continue

        same_type = (g.type(va) == g.type(vb))
        simple_edges = (out_b.edge_type == "simple" and in_b.edge_type == "simple")

        if same_type and simple_edges:
            # Fuse: sum phases, redirect vb's neighbors to va, remove vb
            new_phase = g.phase(va) + g.phase(vb)
            g.set_phase(va, new_phase)

            for nb in list(g.neighbors(vb)):
                if nb == va:
                    continue
                # Get the edge type between vb and nb
                et = g.edge_type(g.edge(vb, nb))
                g.add_edge((va, nb), edgetype=et)

            g.remove_vertex(vb)
        else:
            # Add edge (hadamard if either boundary is hadamard)
            et = EdgeType.HADAMARD if not simple_edges else EdgeType.SIMPLE
            g.add_edge((va, vb), edgetype=et)

    # -- Set inputs from web_a's input boundaries, outputs from web_b's output boundaries --
    input_boundaries = web_a.boundaries[:web_a.n_input_boundaries]
    output_boundaries = web_b.boundaries[web_b.n_input_boundaries:]

    inputs = []
    for b in input_boundaries:
        if b.vertex_id is not None and b.vertex_id in map_a:
            inputs.append(map_a[b.vertex_id])

    outputs = []
    for b in output_boundaries:
        if b.vertex_id is not None and b.vertex_id in map_b:
            outputs.append(map_b[b.vertex_id])

    if inputs:
        g.set_inputs(tuple(inputs))
    if outputs:
        g.set_outputs(tuple(outputs))

    return g


def validate_candidate(graph: Any, expected_qubits: int) -> bool:
    """Quick sanity check on a composed candidate."""
    if graph.num_vertices() == 0:
        return False

    n_in = len(graph.inputs())
    n_out = len(graph.outputs())

    if n_in == 0 or n_out == 0:
        return False

    if n_in != n_out:
        return False

    if expected_qubits > 0 and n_in != expected_qubits:
        return False

    return True


# ── Helpers ──────────────────────────────────────────────────────────


def _find_webs_for_slot(slot: Slot, library: WebLibrary) -> list[ZXWeb]:
    """Find candidate webs for a slot, trying role fallbacks."""
    # Try exact role first
    webs = library.search(role=slot.role)

    # Fall back through alternatives
    if not webs:
        fallbacks = _ROLE_FALLBACKS.get(slot.role, [])
        for fb_role in fallbacks:
            webs = library.search(role=fb_role)
            if webs:
                break

    # If still empty, return all webs as a last resort
    if not webs:
        webs = library.all_webs()

    # Sort by support (descending) and cap
    webs.sort(key=lambda w: w.support, reverse=True)
    return webs[:_MAX_WEBS_PER_SLOT]


def _generate_combinations(
    slot_candidates: list[list[ZXWeb]],
    max_qubits: int,
) -> list[list[ZXWeb]]:
    """Generate valid web combinations using backtracking.

    Prunes incompatible sequences early via is_compatible().
    """
    results: list[list[ZXWeb]] = []

    def _backtrack(idx: int, current: list[ZXWeb]) -> None:
        if idx == len(slot_candidates):
            results.append(list(current))
            return

        for web in slot_candidates[idx]:
            # Check compatibility with previous web
            if current and not current[-1].is_compatible(web):
                continue

            # Quick qubit bound check
            if max_qubits > 0 and web.n_inputs() > max_qubits:
                continue

            current.append(web)
            _backtrack(idx + 1, current)
            current.pop()

            # Cap total results to avoid combinatorial explosion
            if len(results) >= 50:
                return

    _backtrack(0, [])
    return results


def _compose_sequence(webs: list[ZXWeb]) -> Any:
    """Sequentially connect a list of webs into a single graph."""
    if not webs:
        return zx.Graph()
    if len(webs) == 1:
        return webs[0].graph.copy()

    # Start with first pair
    current_a = webs[0]
    result = None

    for i in range(1, len(webs)):
        web_b = webs[i]

        if result is None:
            web_a = current_a
        else:
            # Wrap the intermediate result as a ZXWeb for the next connection
            # Carry forward output boundaries from previous composition
            web_a = ZXWeb(
                web_id="_intermediate",
                graph=result,
                boundaries=_intermediate_boundaries(result, current_a, web_b),
                spider_count=result.num_vertices(),
                n_input_boundaries=current_a.n_input_boundaries,
            )

        n_outputs = web_a.n_outputs()
        n_inputs = web_b.n_inputs()
        n_connect = min(n_outputs, n_inputs)
        boundary_map = [(j, j) for j in range(n_connect)]

        result = connect_webs(web_a, web_b, boundary_map)

        # Update current_a to track input boundaries through composition
        # The composed graph's inputs come from the first web's inputs
        current_a = ZXWeb(
            web_id="_composed",
            graph=result,
            boundaries=(
                current_a.boundaries[:current_a.n_input_boundaries]
                + web_b.boundaries[web_b.n_input_boundaries:]
            ),
            spider_count=result.num_vertices(),
            n_input_boundaries=current_a.n_input_boundaries,
        )

    return result


def _intermediate_boundaries(
    graph: Any,
    prev_web: ZXWeb,
    next_web: ZXWeb,
) -> list[Boundary]:
    """Build boundary list for an intermediate composed graph.

    Inputs come from prev_web's inputs, outputs come from the graph's outputs.
    """
    # Keep prev_web's input boundaries
    in_bounds = prev_web.boundaries[:prev_web.n_input_boundaries]

    # Build output boundaries from graph's outputs
    out_bounds = []
    outputs = list(graph.outputs()) if hasattr(graph, 'outputs') else []
    for idx, v in enumerate(outputs):
        sp_type = "Z"  # default
        vtype = graph.type(v)
        if vtype == VertexType.Z:
            sp_type = "Z"
        elif vtype == VertexType.X:
            sp_type = "X"
        out_bounds.append(Boundary(
            index=len(in_bounds) + idx,
            spider_type=sp_type,
            phase=float(graph.phase(v)),
            edge_type="simple",
            vertex_id=v,
        ))

    return in_bounds + out_bounds


def compose_from_template(
    template: AlgorithmTemplate,
    library: WebLibrary,
    max_qubits: int,
    enforce_flow: bool = True,
) -> list[tuple[Any, CompositionRecipe]]:
    """Generate candidate diagrams by filling a template's slots."""
    # Find candidate webs per slot
    slot_candidates: list[list[ZXWeb]] = []
    for slot in template.slots:
        webs = _find_webs_for_slot(slot, library)
        if not webs:
            return []
        slot_candidates.append(webs)

    # Generate valid combinations via backtracking
    combinations = _generate_combinations(slot_candidates, max_qubits)

    results: list[tuple[Any, CompositionRecipe]] = []
    for combo_idx, combo in enumerate(combinations):
        try:
            graph = _compose_sequence(combo)

            if not validate_candidate(graph, expected_qubits=0):
                continue

            recipe = CompositionRecipe(
                candidate_id=f"{template.name}_{combo_idx:04d}",
                template_name=template.name,
                web_sequence=[w.web_id for w in combo],
            )
            results.append((graph, recipe))
        except Exception:
            logger.debug(
                "Failed to compose combination %d for template %s",
                combo_idx,
                template.name,
                exc_info=True,
            )

    return results
