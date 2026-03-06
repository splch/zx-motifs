"""Command-line interface for zx-motifs.

Usage::

    zx-motifs list algorithms [--family F] [--tags T,T]
    zx-motifs list motifs [--source S]
    zx-motifs list families
    zx-motifs info <name>
    zx-motifs info --motif <id>
    zx-motifs scaffold algorithm --name N --family F
    zx-motifs scaffold motif --name N
    zx-motifs validate [--file PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Template strings for scaffold commands
# ---------------------------------------------------------------------------

_ALGORITHM_TEMPLATE = '''\
"""{family} family algorithm: {name}."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm


@register_algorithm(
    name="{name}",
    family="{family}",
    qubit_range=(2, 8),
    tags=[],
    description="TODO: describe this algorithm",
)
def make_{name}(n_qubits=2, **kwargs) -> QuantumCircuit:
    """TODO: describe the quantum circuit."""
    qc = QuantumCircuit(n_qubits)
    # TODO: implement circuit
    return qc
'''

_MOTIF_TEMPLATE = '''\
{{
  "motif_id": "{name}",
  "description": "TODO: describe this motif",
  "source": "contributed",
  "tags": [],
  "graph": {{
    "nodes": [
      {{"id": 0, "vertex_type": "Z", "phase_class": "zero"}},
      {{"id": 1, "vertex_type": "X", "phase_class": "zero"}}
    ],
    "links": [
      {{"source": 0, "target": 1, "edge_type": "SIMPLE"}}
    ]
  }}
}}
'''

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).resolve().parent
_FAMILIES_DIR = _SRC_ROOT / "algorithms" / "families"
_MOTIF_LIBRARY_DIR = _SRC_ROOT / "motifs" / "library"
_MOTIF_SCHEMA_PATH = _SRC_ROOT / "motifs" / "schema" / "motif.schema.json"


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_list_algorithms(args: argparse.Namespace) -> None:
    from zx_motifs.algorithms import REGISTRY

    entries = list(REGISTRY)

    if args.family:
        entries = [e for e in entries if e.family == args.family]

    if args.tags:
        required = set(args.tags.split(","))
        entries = [e for e in entries if required.issubset(set(e.tags))]

    if not entries:
        print("No algorithms found matching the given filters.")
        return

    # Table output
    name_w = max(len(e.name) for e in entries)
    fam_w = max(len(e.family) for e in entries)
    qr_w = max(len(str(e.qubit_range)) for e in entries)

    header = f"{'NAME':<{name_w}}  {'FAMILY':<{fam_w}}  {'QUBITS':<{qr_w}}  TAGS"
    print(header)
    print("-" * len(header))
    for e in entries:
        tags_str = ", ".join(e.tags) if e.tags else ""
        print(f"{e.name:<{name_w}}  {e.family:<{fam_w}}  {str(e.qubit_range):<{qr_w}}  {tags_str}")


def _cmd_list_motifs(args: argparse.Namespace) -> None:
    from zx_motifs.motifs import MOTIF_REGISTRY

    motifs = list(MOTIF_REGISTRY)

    if args.source:
        motifs = [m for m in motifs if m.source == args.source]

    if not motifs:
        print("No motifs found matching the given filters.")
        return

    id_w = max(len(m.motif_id) for m in motifs)
    src_w = max(len(m.source) for m in motifs)

    header = f"{'MOTIF_ID':<{id_w}}  {'SOURCE':<{src_w}}  DESCRIPTION"
    print(header)
    print("-" * len(header))
    for m in motifs:
        desc = (m.description[:60] + "...") if len(m.description) > 60 else m.description
        print(f"{m.motif_id:<{id_w}}  {m.source:<{src_w}}  {desc}")


def _cmd_list_families(args: argparse.Namespace) -> None:
    from collections import Counter
    from zx_motifs.algorithms import REGISTRY

    counts = Counter(e.family for e in REGISTRY)

    if not counts:
        print("No algorithm families found.")
        return

    fam_w = max(len(f) for f in counts)
    header = f"{'FAMILY':<{fam_w}}  COUNT"
    print(header)
    print("-" * len(header))
    for family, count in sorted(counts.items()):
        print(f"{family:<{fam_w}}  {count}")


def _cmd_info(args: argparse.Namespace) -> None:
    if args.motif:
        _cmd_info_motif(args.name)
    else:
        _cmd_info_algorithm(args.name)


def _cmd_info_algorithm(name: str) -> None:
    from zx_motifs.algorithms import REGISTRY

    entry = None
    for e in REGISTRY:
        if e.name == name:
            entry = e
            break

    if entry is None:
        print(f"Algorithm '{name}' not found in registry.")
        sys.exit(1)

    print(f"Name:        {entry.name}")
    print(f"Family:      {entry.family}")
    print(f"Qubit range: {entry.qubit_range}")
    print(f"Tags:        {', '.join(entry.tags) if entry.tags else '(none)'}")
    print(f"Description: {entry.description or '(none)'}")
    print()

    docstring = entry.generator.__doc__
    if docstring:
        print("Docstring:")
        print(textwrap.indent(textwrap.dedent(docstring).strip(), "  "))
    else:
        print("Docstring:   (none)")


def _cmd_info_motif(motif_id: str) -> None:
    from zx_motifs.motifs import get_motif

    m = get_motif(motif_id)
    if m is None:
        print(f"Motif '{motif_id}' not found in registry.")
        sys.exit(1)

    print(f"Motif ID:    {m.motif_id}")
    print(f"Source:      {m.source}")
    print(f"Description: {m.description}")
    print(f"Nodes:       {m.graph.number_of_nodes()}")
    print(f"Edges:       {m.graph.number_of_edges()}")

    tags = m.metadata.get("tags", [])
    print(f"Tags:        {', '.join(tags) if tags else '(none)'}")
    print()

    # Graph structure summary
    print("Graph structure:")
    for node, data in m.graph.nodes(data=True):
        vtype = data.get("vertex_type", "?")
        phase = data.get("phase_class", "?")
        print(f"  Node {node}: type={vtype}, phase_class={phase}")
    for u, v, data in m.graph.edges(data=True):
        etype = data.get("edge_type", "?")
        print(f"  Edge {u} -- {v}: type={etype}")


def _cmd_scaffold_algorithm(args: argparse.Namespace) -> None:
    name = args.name
    family = args.family
    target = _FAMILIES_DIR / f"{family}.py"

    content = _ALGORITHM_TEMPLATE.format(name=name, family=family)

    if target.exists():
        print(f"Family file already exists: {target}")
        print("Template printed to stdout (append manually to avoid overwriting):\n")
        print(content)
    else:
        target.write_text(content)
        print(f"Created algorithm template: {target}")


def _cmd_scaffold_motif(args: argparse.Namespace) -> None:
    name = args.name
    target = _MOTIF_LIBRARY_DIR / f"{name}.json"

    content = _MOTIF_TEMPLATE.format(name=name)

    if target.exists():
        print(f"Motif file already exists: {target}")
        sys.exit(1)

    target.write_text(content)
    print(f"Created motif template: {target}")


def _cmd_validate(args: argparse.Namespace) -> None:
    if args.file:
        _validate_file(Path(args.file))
    else:
        _validate_all()


def _validate_file(path: Path) -> None:
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    if path.suffix == ".json":
        ok = _validate_motif_json(path)
    elif path.suffix == ".py":
        ok = _validate_algorithm_file(path)
    else:
        print(f"Unknown file type: {path.suffix} (expected .py or .json)")
        sys.exit(1)

    if ok:
        print(f"PASS: {path}")
    else:
        sys.exit(1)


def _validate_motif_json(path: Path) -> bool:
    """Validate a single motif JSON file against the schema and check connectivity."""
    import jsonschema
    import networkx as nx
    from networkx.readwrite import json_graph as _json_graph

    with open(_MOTIF_SCHEMA_PATH) as f:
        schema = json.load(f)

    with open(path) as f:
        data = json.load(f)

    # Schema validation
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as exc:
        print(f"FAIL: {path} -- schema error: {exc.message}")
        return False

    # Connectivity check
    graph_section = data["graph"]
    envelope = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": graph_section["nodes"],
        "links": graph_section["links"],
    }
    g = _json_graph.node_link_graph(envelope, edges="links")

    if not nx.is_connected(g):
        print(f"FAIL: {path} -- graph is not connected")
        return False

    return True


def _validate_algorithm_file(path: Path) -> bool:
    """Validate a Python algorithm file by attempting to import and run it."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_cli_check", path)
    if spec is None or spec.loader is None:
        print(f"FAIL: {path} -- could not load as Python module")
        return False

    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:
        print(f"FAIL: {path} -- import error: {exc}")
        return False

    return True


def _validate_all() -> None:
    """Validate all registered algorithms and all motif JSON files."""
    errors = 0

    # --- Algorithms ---
    print("Validating algorithms...")
    from zx_motifs.algorithms import REGISTRY

    for entry in REGISTRY:
        min_q = entry.qubit_range[0]
        try:
            qc = entry.generator(min_q)
            # Check QASM2 export
            from qiskit.qasm2 import dumps as qasm2_dumps
            qasm2_dumps(qc)
            print(f"  PASS: {entry.name} (n_qubits={min_q})")
        except Exception as exc:
            print(f"  FAIL: {entry.name} -- {exc}")
            errors += 1

    # --- Motif JSON files ---
    print("Validating motif JSON files...")
    if _MOTIF_LIBRARY_DIR.is_dir():
        for json_path in sorted(_MOTIF_LIBRARY_DIR.glob("*.json")):
            ok = _validate_motif_json(json_path)
            if ok:
                print(f"  PASS: {json_path.name}")
            else:
                errors += 1

    if errors:
        print(f"\n{errors} validation error(s) found.")
        sys.exit(1)
    else:
        print("\nAll validations passed.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zx-motifs",
        description="ZX-calculus motif discovery toolkit CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── list ───────────────────────────────────────────────────────────
    list_parser = subparsers.add_parser("list", help="List registered items")
    list_sub = list_parser.add_subparsers(dest="list_type", help="What to list")

    # list algorithms
    list_alg = list_sub.add_parser("algorithms", help="List registered algorithms")
    list_alg.add_argument("--family", default=None, help="Filter by algorithm family")
    list_alg.add_argument("--tags", default=None, help="Filter by comma-separated tags (must have ALL)")
    list_alg.set_defaults(func=_cmd_list_algorithms)

    # list motifs
    list_mot = list_sub.add_parser("motifs", help="List registered motifs")
    list_mot.add_argument("--source", default=None, help="Filter by motif source")
    list_mot.set_defaults(func=_cmd_list_motifs)

    # list families
    list_fam = list_sub.add_parser("families", help="List algorithm families")
    list_fam.set_defaults(func=_cmd_list_families)

    # ── info ──────────────────────────────────────────────────────────
    info_parser = subparsers.add_parser("info", help="Show detailed info about an item")
    info_parser.add_argument("name", help="Algorithm name or motif ID")
    info_parser.add_argument("--motif", action="store_true", help="Look up a motif instead of an algorithm")
    info_parser.set_defaults(func=_cmd_info)

    # ── scaffold ──────────────────────────────────────────────────────
    scaffold_parser = subparsers.add_parser("scaffold", help="Generate a template for a new contribution")
    scaffold_sub = scaffold_parser.add_subparsers(dest="scaffold_type", help="What to scaffold")

    # scaffold algorithm
    scaf_alg = scaffold_sub.add_parser("algorithm", help="Scaffold a new algorithm")
    scaf_alg.add_argument("--name", required=True, help="Algorithm name (snake_case)")
    scaf_alg.add_argument("--family", required=True, help="Algorithm family")
    scaf_alg.set_defaults(func=_cmd_scaffold_algorithm)

    # scaffold motif
    scaf_mot = scaffold_sub.add_parser("motif", help="Scaffold a new motif")
    scaf_mot.add_argument("--name", required=True, help="Motif ID (snake_case)")
    scaf_mot.set_defaults(func=_cmd_scaffold_motif)

    # ── validate ──────────────────────────────────────────────────────
    validate_parser = subparsers.add_parser("validate", help="Validate algorithms and/or motifs")
    validate_parser.add_argument("--file", default=None, help="Validate a specific file (.py or .json)")
    validate_parser.set_defaults(func=_cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the zx-motifs CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)
