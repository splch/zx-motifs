"""
Sub-diagram discovery: webs, fingerprinting, mining, and library.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import pyzx as zx
from pyzx.utils import EdgeType, VertexType

from src.zx import _VERTEX_TYPE_LABELS, classify_phase, extract_subgraph

logger = logging.getLogger(__name__)


# ── Role mapping ───────────────────────────────────────────────────

_CATEGORY_TO_ROLE = {
    "search": "oracle",
    "fourier": "phase",
    "oracular": "oracle",
    "state_preparation": "state_prep",
    "simulation": "evolve",
    "error_correction": "encode",
    "variational": "phase",
    "data_encoding": "encode",
    "arithmetic": "phase",
    "structural": "entangle",
    "benchmarking": "entangle",
}


# ── ZXWeb ───────────────────────────────────────────────────────────


@dataclass
class Boundary:
    """Describes one open leg of a ZXWeb."""

    index: int
    spider_type: str
    phase: float | None
    edge_type: str
    vertex_id: int | None = None


@dataclass
class ZXWeb:
    """A reusable sub-diagram fragment from one or more quantum algorithms."""

    web_id: str
    graph: Any  # pyzx.Graph
    boundaries: list[Boundary]
    spider_count: int
    sources: list[str] = field(default_factory=list)
    support: int = 0
    role: str | None = None
    phase_class: str = "mixed"
    n_input_boundaries: int = 0

    def n_inputs(self) -> int:
        """Number of input-side boundaries."""
        return self.n_input_boundaries

    def n_outputs(self) -> int:
        """Number of output-side boundaries."""
        return len(self.boundaries) - self.n_input_boundaries

    def is_compatible(self, other: "ZXWeb") -> bool:
        """Check whether this web's outputs can connect to other's inputs."""
        if self.n_outputs() != other.n_inputs():
            return False
        my_outputs = self.boundaries[self.n_input_boundaries:]
        other_inputs = other.boundaries[: other.n_input_boundaries]
        for out_b, in_b in zip(my_outputs, other_inputs):
            if out_b.edge_type != in_b.edge_type:
                return False
        return True

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "web_id": self.web_id,
            "graph": self.graph.to_dict(),
            "boundaries": [
                {
                    "index": b.index,
                    "spider_type": b.spider_type,
                    "phase": b.phase,
                    "edge_type": b.edge_type,
                    "vertex_id": b.vertex_id,
                }
                for b in self.boundaries
            ],
            "spider_count": self.spider_count,
            "sources": self.sources,
            "support": self.support,
            "role": self.role,
            "phase_class": self.phase_class,
            "n_input_boundaries": self.n_input_boundaries,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ZXWeb":
        """Reconstruct a ZXWeb from a serialized dictionary."""
        graph_data = data["graph"]
        if isinstance(graph_data, dict):
            graph_data = json.dumps(graph_data)
        from src.zx import _sanitize_phase_tildes

        graph = zx.Graph.from_json(_sanitize_phase_tildes(graph_data))

        boundaries = [
            Boundary(
                index=b["index"],
                spider_type=b["spider_type"],
                phase=b["phase"],
                edge_type=b["edge_type"],
                vertex_id=b.get("vertex_id"),
            )
            for b in data["boundaries"]
        ]

        return cls(
            web_id=data["web_id"],
            graph=graph,
            boundaries=boundaries,
            spider_count=data["spider_count"],
            sources=data.get("sources", []),
            support=data.get("support", 0),
            role=data.get("role"),
            phase_class=data.get("phase_class", "mixed"),
            n_input_boundaries=data.get("n_input_boundaries", 0),
        )


# ── Fingerprinting ──────────────────────────────────────────────────


@dataclass
class DiagramFingerprint:
    """Compact feature vector summarizing a ZX-diagram's structure."""

    n_z_spiders: int
    n_x_spiders: int
    n_hadamard_edges: int
    n_simple_edges: int
    degree_histogram: dict[int, int]
    phase_histogram: dict[str, int]


def compute_fingerprint(graph: Any) -> DiagramFingerprint:
    """Compute the fingerprint of a ZX-diagram."""
    n_z = 0
    n_x = 0
    degree_counts: dict[int, int] = {}
    phase_counts: dict[str, int] = {}

    for v in graph.vertices():
        vtype = graph.type(v)
        if vtype == VertexType.BOUNDARY:
            continue
        if vtype == VertexType.Z:
            n_z += 1
        elif vtype == VertexType.X:
            n_x += 1

        deg = len(list(graph.neighbors(v)))
        degree_counts[deg] = degree_counts.get(deg, 0) + 1

        pc = classify_phase(graph.phase(v))
        phase_counts[pc] = phase_counts.get(pc, 0) + 1

    n_had = 0
    n_simple = 0
    for e in graph.edges():
        if graph.edge_type(e) == EdgeType.HADAMARD:
            n_had += 1
        else:
            n_simple += 1

    return DiagramFingerprint(
        n_z_spiders=n_z,
        n_x_spiders=n_x,
        n_hadamard_edges=n_had,
        n_simple_edges=n_simple,
        degree_histogram=degree_counts,
        phase_histogram=phase_counts,
    )


def fingerprints_compatible(
    parent: DiagramFingerprint,
    sub: DiagramFingerprint,
) -> bool:
    """Quick check whether sub could be a sub-diagram of parent."""
    if sub.n_z_spiders > parent.n_z_spiders:
        return False
    if sub.n_x_spiders > parent.n_x_spiders:
        return False
    if sub.n_hadamard_edges > parent.n_hadamard_edges:
        return False
    if sub.n_simple_edges > parent.n_simple_edges:
        return False
    # Note: degree histogram is NOT checked because subgraph extraction
    # reduces degrees at boundary vertices, making it an invalid necessary condition.
    for pc, count in sub.phase_histogram.items():
        if parent.phase_histogram.get(pc, 0) < count:
            return False
    return True


# ── NetworkX conversion ─────────────────────────────────────────────


def _pyzx_to_nx_internal(graph: Any, phase_abstraction: str) -> nx.Graph:
    """Convert a pyzx graph to NetworkX, excluding BOUNDARY vertices."""
    G = nx.Graph()
    for v in graph.vertices():
        vtype = graph.type(v)
        if vtype == VertexType.BOUNDARY:
            continue
        type_label = _VERTEX_TYPE_LABELS.get(vtype, "?")
        if phase_abstraction == "class":
            pc = classify_phase(graph.phase(v))
            label = f"{type_label}_{pc}"
        else:
            label = type_label
        G.add_node(v, label=label)
    for e in graph.edges():
        src, tgt = graph.edge_st(e)
        if src not in G or tgt not in G:
            continue
        etype = graph.edge_type(e)
        elabel = "hadamard" if etype == EdgeType.HADAMARD else "simple"
        G.add_edge(src, tgt, label=elabel)
    return G


# ── Mining ──────────────────────────────────────────────────────────


def _determine_role(sources: list[str], category_map: dict[str, str]) -> str | None:
    """Determine web role from source algorithm categories."""
    roles: list[str] = []
    for src in sources:
        # source might be like "ghz_3q" — extract the algo name
        cat = category_map.get(src)
        if cat:
            role = _CATEGORY_TO_ROLE.get(cat)
            if role:
                roles.append(role)
    if not roles:
        return None
    counter = Counter(roles)
    return counter.most_common(1)[0][0]


def _determine_phase_class(graph: Any) -> str:
    """Determine dominant phase class of a subgraph."""
    classes: list[str] = []
    for v in graph.vertices():
        if graph.type(v) == VertexType.BOUNDARY:
            continue
        classes.append(classify_phase(graph.phase(v)))
    if not classes:
        return "mixed"
    counter = Counter(classes)
    dominant, count = counter.most_common(1)[0]
    if count == len(classes):
        return dominant
    return "mixed"


def mine_webs(
    diagrams: list[tuple[str, Any]],  # (algorithm_key, pyzx.Graph)
    min_support: int,
    min_spiders: int,
    max_spiders: int,
    phase_abstraction: str,
    max_diagram_vertices: int | None = 800,
    wl_iterations: int = 4,
) -> list[ZXWeb]:
    """Discover frequent sub-diagrams using Weisfeiler-Lehman neighborhood hashing.

    For each diagram, computes WL subgraph hashes at each vertex and depth.
    Hashes appearing in >= min_support diagrams identify frequent local patterns.
    The ego graph around an exemplar vertex is extracted as a ZXWeb.

    Runs in O(V * k * d) time per graph (linear), where V = vertices,
    k = wl_iterations, d = average degree.
    """
    if not diagrams:
        return []

    # Step 1: Pre-filter oversized diagrams
    if max_diagram_vertices is not None:
        filtered = []
        for algo_key, graph in diagrams:
            n_internal = sum(
                1 for v in graph.vertices() if graph.type(v) != VertexType.BOUNDARY
            )
            if n_internal <= max_diagram_vertices:
                filtered.append((algo_key, graph))
            else:
                logger.warning(
                    "Skipping %s: %d internal vertices > limit %d",
                    algo_key,
                    n_internal,
                    max_diagram_vertices,
                )
        if len(filtered) < len(diagrams):
            logger.info(
                "Filtered diagrams: %d -> %d (dropped %d oversized)",
                len(diagrams),
                len(filtered),
                len(diagrams) - len(filtered),
            )
        diagrams = filtered

    if not diagrams:
        logger.warning("No diagrams remain after filtering")
        return []

    # Step 2: Convert all diagrams to NetworkX
    nx_graphs: list[nx.Graph] = []
    for _algo_key, graph in diagrams:
        nx_graphs.append(_pyzx_to_nx_internal(graph, phase_abstraction))

    # Step 3: Compute WL subgraph hashes and collect support info
    # Key: (hash_str, depth_index) -> info dict
    pattern_info: dict[tuple[str, int], dict] = {}

    for diag_idx, (algo_key, _graph) in enumerate(diagrams):
        nx_g = nx_graphs[diag_idx]
        if nx_g.number_of_nodes() == 0:
            continue

        subgraph_hashes = nx.weisfeiler_lehman_subgraph_hashes(
            nx_g,
            iterations=wl_iterations,
            node_attr="label",
            edge_attr="label",
        )

        # Deduplicate per diagram: only count each (hash, depth) once
        seen_in_diagram: set[tuple[str, int]] = set()

        for vertex, hash_list in subgraph_hashes.items():
            for depth_idx, h in enumerate(hash_list):
                key = (h, depth_idx)
                if key in seen_in_diagram:
                    continue
                seen_in_diagram.add(key)

                if key not in pattern_info:
                    pattern_info[key] = {
                        "diagram_indices": set(),
                        "algo_keys": set(),
                        "exemplar": (diag_idx, vertex),
                    }
                pattern_info[key]["diagram_indices"].add(diag_idx)
                pattern_info[key]["algo_keys"].add(algo_key)

    # Step 4: Filter by minimum support
    frequent = {
        key: info
        for key, info in pattern_info.items()
        if len(info["diagram_indices"]) >= min_support
    }

    # Sort by (support descending, depth descending) to prioritize robust, larger patterns
    sorted_patterns = sorted(
        frequent.items(),
        key=lambda item: (len(item[1]["diagram_indices"]), item[0][1]),
        reverse=True,
    )

    logger.info(
        "WL hashing found %d frequent patterns (from %d total hashes)",
        len(sorted_patterns),
        len(pattern_info),
    )

    # Step 5: Build ZXWeb objects from frequent patterns
    webs: list[ZXWeb] = []

    for (hash_str, depth_idx), info in sorted_patterns:
        # Skip depth 0 — just the node's own label, too trivial
        if depth_idx == 0:
            continue

        diag_idx, vertex = info["exemplar"]
        nx_g = nx_graphs[diag_idx]

        # Extract ego graph at the WL depth radius
        ego = nx.ego_graph(nx_g, vertex, radius=depth_idx)
        ego_vertices = set(ego.nodes())
        n_ego = len(ego_vertices)

        if n_ego < min_spiders or n_ego > max_spiders:
            continue

        # Extract pyzx subgraph (ego vertex IDs are original pyzx vertex IDs)
        pyzx_graph = diagrams[diag_idx][1]
        pyzx_subgraph, boundary_verts = extract_subgraph(pyzx_graph, ego_vertices)

        if pyzx_subgraph.num_vertices() == 0:
            continue

        # Compute average row for input/output boundary classification
        internal_verts = [
            v
            for v in pyzx_subgraph.vertices()
            if pyzx_subgraph.type(v) != VertexType.BOUNDARY
        ]
        if internal_verts:
            avg_row = sum(pyzx_subgraph.row(v) for v in internal_verts) / len(
                internal_verts
            )
        else:
            avg_row = 0

        input_boundaries: list[Boundary] = []
        output_boundaries: list[Boundary] = []

        for idx, bv in enumerate(boundary_verts):
            sp_type = _VERTEX_TYPE_LABELS.get(pyzx_subgraph.type(bv), "?")
            phase_val = float(pyzx_subgraph.phase(bv))

            # Determine edge type at boundary (use simple as default)
            b_edge_type = "simple"
            for e in pyzx_subgraph.edges():
                src, tgt = pyzx_subgraph.edge_st(e)
                if src == bv or tgt == bv:
                    if pyzx_subgraph.edge_type(e) == EdgeType.HADAMARD:
                        b_edge_type = "hadamard"
                    break

            b = Boundary(
                index=idx,
                spider_type=sp_type,
                phase=phase_val,
                edge_type=b_edge_type,
                vertex_id=bv,
            )

            # Classify as input (low row) or output (high row)
            row = pyzx_subgraph.row(bv)
            if row <= avg_row:
                input_boundaries.append(b)
            else:
                output_boundaries.append(b)

        # Sort and assign indices
        all_boundaries = input_boundaries + output_boundaries
        for idx, b in enumerate(all_boundaries):
            b.index = idx

        # Collect sources
        source_algos = sorted(info["algo_keys"])
        support_count = len(info["diagram_indices"])
        # Determine phase class and role
        pc = _determine_phase_class(pyzx_subgraph)

        web = ZXWeb(
            web_id=f"web_{len(webs):04d}",
            graph=pyzx_subgraph,
            boundaries=all_boundaries,
            spider_count=pyzx_subgraph.num_vertices(),
            sources=source_algos,
            support=support_count,
            role=None,
            phase_class=pc,
            n_input_boundaries=len(input_boundaries),
        )
        webs.append(web)

    return webs


# ── Library ─────────────────────────────────────────────────────────


class WebLibrary:
    """Manages a collection of ZXWebs on disk."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._index: dict[str, dict] = {}

    def add(self, web: ZXWeb) -> None:
        """Add a ZXWeb to the library."""
        self._dir.mkdir(parents=True, exist_ok=True)
        web_path = self._dir / f"{web.web_id}.json"
        from src.zx import _sanitize_phase_tildes

        web_path.write_text(_sanitize_phase_tildes(json.dumps(web.to_dict(), default=str)))
        self._index[web.web_id] = {
            "web_id": web.web_id,
            "spider_count": web.spider_count,
            "n_boundaries": len(web.boundaries),
            "n_input_boundaries": web.n_input_boundaries,
            "sources": web.sources,
            "support": web.support,
            "role": web.role,
            "phase_class": web.phase_class,
        }

    def get(self, web_id: str) -> ZXWeb:
        """Load a single ZXWeb by its ID."""
        web_path = self._dir / f"{web_id}.json"
        data = json.loads(web_path.read_text())
        return ZXWeb.from_dict(data)

    def search(
        self,
        min_boundaries: int | None = None,
        max_boundaries: int | None = None,
        role: str | None = None,
        phase_class: str | None = None,
        min_support: int | None = None,
    ) -> list[ZXWeb]:
        """Search the library with optional filters."""
        results: list[ZXWeb] = []
        for web_id, meta in self._index.items():
            if min_boundaries is not None and meta["n_boundaries"] < min_boundaries:
                continue
            if max_boundaries is not None and meta["n_boundaries"] > max_boundaries:
                continue
            if role is not None and meta["role"] != role:
                continue
            if phase_class is not None and meta["phase_class"] != phase_class:
                continue
            if min_support is not None and meta["support"] < min_support:
                continue
            results.append(self.get(web_id))
        return results

    def all_webs(self) -> list[ZXWeb]:
        """Load and return every web in the library."""
        return [self.get(web_id) for web_id in self._index]

    def save_index(self) -> None:
        """Write the index to library.json."""
        self._dir.mkdir(parents=True, exist_ok=True)
        index_path = self._dir / "library.json"
        index_path.write_text(json.dumps(self._index, default=str))

    def load_index(self) -> None:
        """Read the index from library.json."""
        index_path = self._dir / "library.json"
        if index_path.exists():
            self._index = json.loads(index_path.read_text())
        else:
            self._index = {}
