"""
ZX-diagram conversion, simplification, storage, and graph helpers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import networkx as nx
import pyzx as zx
from pyzx.utils import EdgeType, VertexType

logger = logging.getLogger(__name__)


def _sanitize_phase_tildes(json_str: str) -> str:
    """Strip leading '~' from phase values in a PyZX JSON string.

    PyZX serializes approximate phases as e.g. ``"~263π/240"`` but its own
    parser cannot read the ``~`` back.  Removing it keeps the rational
    approximation, which is the best we can do for a floating-point phase.
    """
    import re

    return re.sub(r'"~([^"]*)"', r'"\1"', json_str)


# ── Conversion ──────────────────────────────────────────────────────


def load_qasm_file(path: str | Path) -> str:
    """Read a .qasm file and return its contents. Validates OPENQASM 2.0 header."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"QASM file not found: {path}")
    text = path.read_text()
    if not text.strip().startswith("OPENQASM 2.0"):
        raise ValueError(f"File does not start with OPENQASM 2.0 header: {path}")
    return text


def qasm_to_pyzx_circuit(qasm: str) -> Any:  # pyzx.Circuit
    """Parse an OpenQASM 2.0 string into a PyZX Circuit."""
    try:
        return zx.Circuit.from_qasm(qasm)
    except TypeError as e:
        raise ValueError(f"Unsupported gate in QASM: {e}") from e


def pyzx_circuit_to_graph(circuit: Any) -> Any:  # pyzx.Graph
    """Convert a PyZX Circuit to a ZX-diagram (Graph)."""
    return circuit.to_graph()


# ── Simplification ──────────────────────────────────────────────────


@dataclass
class SimplificationResult:
    """Holds a diagram at each simplification level."""

    raw: Any  # pyzx.Graph
    clifford: Any  # pyzx.Graph
    full: Any  # pyzx.Graph
    spider_counts: dict[str, int] | None = None


def simplify_graph(graph: Any) -> SimplificationResult:
    """Apply all three simplification levels to a ZX-diagram."""
    raw = graph.copy()

    clifford_g = graph.copy()
    zx.simplify.clifford_simp(clifford_g, quiet=True)

    full_g = graph.copy()
    zx.simplify.full_reduce(full_g, quiet=True)

    spider_counts = {
        "raw": raw.num_vertices(),
        "clifford": clifford_g.num_vertices(),
        "full": full_g.num_vertices(),
    }

    return SimplificationResult(
        raw=raw,
        clifford=clifford_g,
        full=full_g,
        spider_counts=spider_counts,
    )


def _timeout_handler(signum, frame):
    raise TimeoutError("Circuit conversion timed out")


def convert_single_qasm(
    qasm_path_str: str,
    levels: list[str],
    output_dir_str: str,
    timeout: int | None = None,
) -> tuple[str, int] | None:
    """Worker: convert one QASM file to ZX-diagrams at all simplification levels.

    Returns (stem, raw_spider_count) on success, None on failure/timeout.
    """
    import signal

    qasm_path = Path(qasm_path_str)
    output_dir = Path(output_dir_str)
    stem = qasm_path.stem
    parts = stem.rsplit("_", 1)
    algo_name = parts[0] if len(parts) == 2 else stem
    try:
        n_qubits = int(parts[1].rstrip("q")) if len(parts) == 2 else 0
    except ValueError:
        n_qubits = 0

    if timeout is not None:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

    try:
        qasm_text = load_qasm_file(qasm_path)
        circuit = qasm_to_pyzx_circuit(qasm_text)
        graph = pyzx_circuit_to_graph(circuit)
        result = simplify_graph(graph)

        level_graphs = {"raw": result.raw, "clifford": result.clifford, "full": result.full}
        for level_name in levels:
            g = level_graphs.get(level_name)
            if g is None:
                continue
            diagram_id = f"{stem}_{level_name}"
            metadata = {"source_algorithm": algo_name, "n_qubits": n_qubits, "level": level_name}
            save_diagram(g, diagram_id, output_dir, metadata)

        return (stem, result.spider_counts["raw"])
    except TimeoutError:
        logging.getLogger(__name__).warning("Timed out converting %s (limit %ds)", stem, timeout)
        return None
    except Exception:
        return None
    finally:
        if timeout is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


# ── Storage ─────────────────────────────────────────────────────────


@dataclass
class DiagramRecord:
    """Metadata envelope around a stored ZX-diagram."""

    diagram_id: str
    source_algorithm: str
    n_qubits: int
    level: str
    spider_count: int
    edge_count: int
    json_path: Path


def save_diagram(
    graph: Any,
    diagram_id: str,
    output_dir: str | Path,
    metadata: dict,
) -> DiagramRecord:
    """Serialize a ZX-diagram to JSON with metadata sidecar."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    record = DiagramRecord(
        diagram_id=diagram_id,
        source_algorithm=metadata.get("source_algorithm", ""),
        n_qubits=metadata.get("n_qubits", 0),
        level=metadata.get("level", "raw"),
        spider_count=graph.num_vertices(),
        edge_count=graph.num_edges(),
        json_path=output_dir / f"{diagram_id}.json",
    )

    envelope = {
        "metadata": {
            "diagram_id": record.diagram_id,
            "source_algorithm": record.source_algorithm,
            "n_qubits": record.n_qubits,
            "level": record.level,
            "spider_count": record.spider_count,
            "edge_count": record.edge_count,
        },
        "graph": graph.to_dict(),
    }

    record.json_path.write_text(_sanitize_phase_tildes(json.dumps(envelope, default=str)))
    return record


def load_diagram(json_path: str | Path) -> Any:
    """Deserialize a ZX-diagram from a JSON file."""
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())
    graph_data = data["graph"]
    # Graph.from_json accepts either a JSON string or a dict
    if isinstance(graph_data, dict):
        graph_data = json.dumps(graph_data)
    return zx.Graph.from_json(_sanitize_phase_tildes(graph_data))


def load_all_diagrams(
    directory: str | Path,
    level: str | None = None,
) -> list[tuple[DiagramRecord, Any]]:
    """Load all diagrams from a directory, optionally filtered by level."""
    directory = Path(directory)
    results: list[tuple[DiagramRecord, Any]] = []

    for json_path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(json_path.read_text())
            meta = data["metadata"]
            if level is not None and meta.get("level") != level:
                continue
            record = DiagramRecord(
                diagram_id=meta["diagram_id"],
                source_algorithm=meta["source_algorithm"],
                n_qubits=meta["n_qubits"],
                level=meta["level"],
                spider_count=meta["spider_count"],
                edge_count=meta["edge_count"],
                json_path=json_path,
            )
            graph_data = data["graph"]
            if isinstance(graph_data, dict):
                graph_data = json.dumps(graph_data)
            graph = zx.Graph.from_json(_sanitize_phase_tildes(graph_data))
            results.append((record, graph))
        except Exception:
            logger.warning("Failed to load diagram from %s", json_path, exc_info=True)

    return results


# ── Graph helpers ───────────────────────────────────────────────────


_VERTEX_TYPE_LABELS = {
    VertexType.BOUNDARY: "B",
    VertexType.Z: "Z",
    VertexType.X: "X",
    VertexType.H_BOX: "H",
}


def classify_phase(phase: float) -> str:
    """Classify a phase (in units of pi) into pauli/clifford/t/arbitrary."""
    frac = Fraction(phase).limit_denominator(1024)
    frac = frac % 2
    denom = frac.denominator
    if denom == 1:
        return "pauli"
    if denom <= 2:
        return "clifford"
    if denom <= 4:
        return "t"
    return "arbitrary"


def pyzx_to_networkx(graph: Any) -> nx.Graph:
    """Convert a PyZX Graph to a NetworkX graph with labeled nodes/edges."""
    G = nx.Graph()

    for v in graph.vertices():
        vtype = graph.type(v)
        phase = graph.phase(v)
        type_label = _VERTEX_TYPE_LABELS.get(vtype, "?")
        phase_class = classify_phase(phase)
        G.add_node(
            v,
            label=f"{type_label}_{phase_class}",
            type=type_label,
            phase=str(phase),
            phase_class=phase_class,
        )

    for e in graph.edges():
        src, tgt = graph.edge_st(e)
        etype = graph.edge_type(e)
        if etype == EdgeType.HADAMARD:
            elabel = "hadamard"
        else:
            elabel = "simple"
        G.add_edge(src, tgt, label=elabel, edge_type=str(etype))

    return G


def extract_subgraph(
    graph: Any,  # pyzx.Graph
    vertex_set: set[int],
) -> tuple[Any, list[int]]:
    """Extract a sub-diagram from a PyZX Graph.

    Returns the induced subgraph and a list of boundary vertex IDs.
    """
    sub = zx.Graph()
    id_map: dict[int, int] = {}

    for v in vertex_set:
        new_v = sub.add_vertex(
            ty=graph.type(v),
            qubit=graph.qubit(v),
            row=graph.row(v),
            phase=graph.phase(v),
        )
        id_map[v] = new_v

    for e in graph.edges():
        src, tgt = graph.edge_st(e)
        if src in vertex_set and tgt in vertex_set:
            sub.add_edge(
                (id_map[src], id_map[tgt]),
                edgetype=graph.edge_type(e),
            )

    # Boundary: vertices with at least one neighbor outside the set
    boundary = []
    for v in vertex_set:
        neighbors = set(graph.neighbors(v))
        if neighbors - vertex_set:
            boundary.append(id_map[v])

    return sub, boundary
