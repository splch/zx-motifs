"""
Sub-diagram discovery: webs, fingerprinting, mining, and library.
"""

from __future__ import annotations

import copy
import json
import logging
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import pyzx as zx
from pyzx.utils import EdgeType, VertexType

from src.zx import _VERTEX_TYPE_LABELS, classify_phase, extract_subgraph

logger = logging.getLogger(__name__)


# ── Label encoding ─────────────────────────────────────────────────

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


def _build_label_encoder(phase_abstraction: str = "class"):
    """Build vertex/edge label encoding dicts for gSpan.

    Returns (vlabel_encode, vlabel_decode, elabel_encode, elabel_decode).
    """
    type_keys = ["Z", "X", "H"]
    phase_keys = ["pauli", "clifford", "t", "arbitrary"]

    vlabel_encode: dict[str, str] = {}
    vlabel_decode: dict[str, str] = {}
    code = 0
    for tk in type_keys:
        if phase_abstraction == "class":
            for pk in phase_keys:
                label = f"{tk}_{pk}"
                encoded = f"{code:02d}"
                vlabel_encode[label] = encoded
                vlabel_decode[encoded] = label
                code += 1
        else:
            encoded = f"{code:02d}"
            vlabel_encode[tk] = encoded
            vlabel_decode[encoded] = tk
            code += 1

    elabel_encode = {"simple": "0", "hadamard": "1"}
    elabel_decode = {"0": "simple", "1": "hadamard"}

    return vlabel_encode, vlabel_decode, elabel_encode, elabel_decode


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
        graph = zx.Graph.from_json(graph_data)

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


# ── gSpan helpers ──────────────────────────────────────────────────


def _pyzx_to_gspan_lines(
    graph: Any,
    graph_id: int,
    vlabel_encode: dict[str, str],
    elabel_encode: dict[str, str],
    phase_abstraction: str,
) -> tuple[list[str], dict[int, int]]:
    """Convert a pyzx graph to gSpan text lines.

    Returns (lines, id_remap) where id_remap maps new contiguous IDs
    to original pyzx vertex IDs.
    """
    # Filter out BOUNDARY vertices
    internal_verts = [
        v for v in graph.vertices() if graph.type(v) != VertexType.BOUNDARY
    ]

    if not internal_verts:
        return [], {}

    # Remap to contiguous 0-based IDs
    old_to_new: dict[int, int] = {}
    new_to_old: dict[int, int] = {}
    for i, v in enumerate(sorted(internal_verts)):
        old_to_new[v] = i
        new_to_old[i] = v

    lines = [f"t # {graph_id}"]

    for v in sorted(internal_verts):
        vtype = graph.type(v)
        type_label = _VERTEX_TYPE_LABELS.get(vtype, "?")
        if phase_abstraction == "class":
            pc = classify_phase(graph.phase(v))
            key = f"{type_label}_{pc}"
        else:
            key = type_label
        vlabel = vlabel_encode.get(key, "99")
        lines.append(f"v {old_to_new[v]} {vlabel}")

    for e in graph.edges():
        src, tgt = graph.edge_st(e)
        if src not in old_to_new or tgt not in old_to_new:
            continue
        etype = graph.edge_type(e)
        elabel_key = "hadamard" if etype == EdgeType.HADAMARD else "simple"
        elabel = elabel_encode.get(elabel_key, "0")
        lines.append(f"e {old_to_new[src]} {old_to_new[tgt]} {elabel}")

    return lines, new_to_old


def _write_gspan_database(
    diagrams: list[tuple[str, Any]],
    phase_abstraction: str,
) -> tuple[str, list[dict], dict[str, str], dict[str, str]]:
    """Write all diagrams to a gSpan database file.

    Returns (file_path, per_graph_metadata, vlabel_decode, elabel_decode).
    per_graph_metadata[i] = {"algo_key": ..., "id_remap": ...}
    """
    vlabel_encode, vlabel_decode, elabel_encode, elabel_decode = _build_label_encoder(
        phase_abstraction
    )

    all_lines: list[str] = []
    per_graph_meta: list[dict] = []

    for gid, (algo_key, graph) in enumerate(diagrams):
        lines, id_remap = _pyzx_to_gspan_lines(
            graph, gid, vlabel_encode, elabel_encode, phase_abstraction
        )
        if not lines:
            per_graph_meta.append(
                {"algo_key": algo_key, "id_remap": {}, "graph": graph}
            )
            continue
        all_lines.extend(lines)
        per_graph_meta.append(
            {"algo_key": algo_key, "id_remap": id_remap, "graph": graph}
        )

    all_lines.append("t # -1")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".gspan", delete=False
    )
    tmp.write("\n".join(all_lines))
    tmp.close()

    return tmp.name, per_graph_meta, vlabel_decode, elabel_decode


def _gspan_graph_to_networkx(
    gspan_graph: Any,
    vlabel_decode: dict[str, str],
    elabel_decode: dict[str, str],
) -> nx.Graph:
    """Convert a gspan Graph object to a NetworkX graph with decoded labels."""
    G = nx.Graph()
    for vid, vertex in gspan_graph.vertices.items():
        decoded = vlabel_decode.get(str(vertex.vlb), str(vertex.vlb))
        G.add_node(vid, label=decoded)
    for vid, vertex in gspan_graph.vertices.items():
        for to_vid, edge in vertex.edges.items():
            if not G.has_edge(vid, to_vid):
                decoded = elabel_decode.get(str(edge.elb), str(edge.elb))
                G.add_edge(vid, to_vid, label=decoded)
    return G


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


class _QuietGSpan:
    """Wrapper around gSpan that captures results without pandas issues."""

    def __init__(
        self,
        database_file_name: str,
        min_support: int = 2,
        min_num_vertices: int = 1,
        max_num_vertices: int = float("inf"),
        is_undirected: bool = True,
    ):
        from gspan_mining.gspan import gSpan as _GSpan

        self._gs = _GSpan(
            database_file_name=database_file_name,
            min_support=min_support,
            min_num_vertices=min_num_vertices,
            max_num_vertices=max_num_vertices,
            is_undirected=is_undirected,
            verbose=False,
            visualize=False,
            where=False,
        )
        self._frequent_subgraphs: list = []
        self._pattern_supports: list[dict] = []

        # Monkey-patch _report to avoid pandas issues
        original_report = self._gs._report

        def patched_report(projected):
            self._frequent_subgraphs.append(copy.copy(self._gs._DFScode))
            gids = set(p.gid for p in projected)
            self._pattern_supports.append(
                {
                    "support": len(gids),
                    "num_vertices": self._gs._DFScode.get_num_vertices(),
                    "graph_ids": gids,
                }
            )

        self._gs._report = patched_report

    def run(self):
        self._gs.run()

    @property
    def graphs(self):
        return self._gs.graphs


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
) -> list[ZXWeb]:
    """Discover frequent sub-diagrams using gSpan.

    Converts ZX-diagrams to gSpan's input format, runs gSpan, and
    converts results back to ZXWebs with boundaries.
    """
    import os

    if not diagrams:
        return []

    # Write gSpan database
    db_path, per_graph_meta, vlabel_decode, elabel_decode = _write_gspan_database(
        diagrams, phase_abstraction
    )

    try:
        # Run gSpan
        gs = _QuietGSpan(
            database_file_name=db_path,
            min_support=min_support,
            min_num_vertices=min_spiders,
            max_num_vertices=max_spiders,
            is_undirected=True,
        )
        gs.run()

        # Pre-convert pyzx graphs to NetworkX for VF2 matching
        nx_cache: dict[int, nx.Graph] = {}
        for gid in range(len(diagrams)):
            meta = per_graph_meta[gid]
            if meta["id_remap"]:
                nx_cache[gid] = _pyzx_to_nx_internal(
                    meta["graph"], phase_abstraction
                )

        webs: list[ZXWeb] = []

        for i, dfscode in enumerate(gs._frequent_subgraphs):
            support_info = gs._pattern_supports[i]

            if support_info["num_vertices"] < min_spiders:
                continue
            if support_info["num_vertices"] > max_spiders:
                continue

            # Convert pattern to NetworkX for VF2 matching
            pattern_gspan_graph = dfscode.to_graph(gid=0, is_undirected=True)
            pattern_nx = _gspan_graph_to_networkx(
                pattern_gspan_graph, vlabel_decode, elabel_decode
            )

            # Find one embedding in a source graph
            source_gids = support_info["graph_ids"]
            embedding_found = False
            pyzx_subgraph = None
            boundary_verts = None
            first_source_gid = None

            for gid in sorted(source_gids):
                if gid not in nx_cache:
                    continue
                target_nx = nx_cache[gid]

                def node_match(n1, n2):
                    return n1.get("label") == n2.get("label")

                def edge_match(e1, e2):
                    return e1.get("label") == e2.get("label")

                matcher = nx.algorithms.isomorphism.GraphMatcher(
                    target_nx,
                    pattern_nx,
                    node_match=node_match,
                    edge_match=edge_match,
                )

                mapping = None
                for m in matcher.subgraph_isomorphisms_iter():
                    mapping = m
                    break

                if mapping is not None:
                    # mapping: target_vertex -> pattern_vertex
                    matched_pyzx_verts = set(mapping.keys())

                    source_graph = per_graph_meta[gid]["graph"]
                    pyzx_subgraph, boundary_verts = extract_subgraph(
                        source_graph, matched_pyzx_verts
                    )
                    first_source_gid = gid
                    embedding_found = True
                    break

            if not embedding_found or pyzx_subgraph is None:
                continue

            # Build Boundary objects
            # Classify boundaries as input/output based on row position
            source_graph = per_graph_meta[first_source_gid]["graph"]
            matched_pyzx_verts_list = list(
                v
                for v in source_graph.vertices()
                if source_graph.type(v) != VertexType.BOUNDARY
                and v
                in set(
                    mapping.keys()  # type: ignore[union-attr]
                )
            )
            if matched_pyzx_verts_list:
                avg_row = sum(
                    source_graph.row(v) for v in matched_pyzx_verts_list
                ) / len(matched_pyzx_verts_list)
            else:
                avg_row = 0

            # Map boundary verts in the subgraph back to original
            # boundary_verts are IDs in the subgraph
            boundaries: list[Boundary] = []
            input_boundaries: list[Boundary] = []
            output_boundaries: list[Boundary] = []

            for idx, bv in enumerate(boundary_verts):
                sp_type = _VERTEX_TYPE_LABELS.get(
                    pyzx_subgraph.type(bv), "?"
                )
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
            source_algos = list(
                {per_graph_meta[gid]["algo_key"] for gid in source_gids}
            )

            # Determine phase class and role
            pc = _determine_phase_class(pyzx_subgraph)

            web = ZXWeb(
                web_id=f"web_{i:04d}",
                graph=pyzx_subgraph,
                boundaries=all_boundaries,
                spider_count=pyzx_subgraph.num_vertices(),
                sources=sorted(source_algos),
                support=support_info["support"],
                role=None,
                phase_class=pc,
                n_input_boundaries=len(input_boundaries),
            )
            webs.append(web)

        return webs
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


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
        web_path.write_text(json.dumps(web.to_dict(), default=str))
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
