"""
Motif registry: auto-discovers JSON motif definitions from the library/
directory, validates them against the schema, and converts them to
MotifPattern objects for use in the pipeline.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from networkx.readwrite import json_graph as _json_graph

import jsonschema

from zx_motifs.pipeline.matcher import MotifPattern

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
_SCHEMA_PATH = _HERE / "schema" / "motif.schema.json"
_LIBRARY_DIR = _HERE / "library"


# ── Schema loading ───────────────────────────────────────────────────

def _load_schema() -> dict:
    """Load and return the JSON schema for motif definitions."""
    with open(_SCHEMA_PATH) as f:
        return json.load(f)


_SCHEMA: dict | None = None


def _get_schema() -> dict:
    """Lazily load the motif JSON schema."""
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = _load_schema()
    return _SCHEMA


# ── Conversion ───────────────────────────────────────────────────────

def _json_to_motif(data: dict) -> MotifPattern:
    """
    Convert a validated JSON motif definition to a MotifPattern.

    Constructs the networkx Graph from the node_link_data-style
    ``graph`` section, wrapping it with the full node_link_data
    envelope expected by networkx.
    """
    graph_section = data["graph"]

    # Build the full node_link_data envelope that networkx expects
    node_link_envelope = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": graph_section["nodes"],
        "links": graph_section["links"],
    }

    g = _json_graph.node_link_graph(node_link_envelope, edges="links")

    metadata = dict(data.get("metadata", {}))
    # Preserve tags in metadata for filtering via list_motifs()
    if "tags" in data:
        metadata["tags"] = data["tags"]
    # Preserve MAGIC schema extensions
    if "symmetry_tags" in data:
        metadata["symmetry_tags"] = data["symmetry_tags"]
    if "polynomial_signature" in data:
        metadata["polynomial_signature"] = data["polynomial_signature"]

    return MotifPattern(
        motif_id=data["motif_id"],
        graph=g,
        source=data.get("source", "contributed"),
        description=data["description"],
        metadata=metadata,
    )


# ── Discovery & loading ─────────────────────────────────────────────

def _discover_motifs() -> list[MotifPattern]:
    """
    Discover all JSON motif files in the library/ directory,
    validate each against the schema, and return as MotifPattern list.

    Files that fail to parse or validate are logged as warnings and
    skipped rather than aborting the entire registry load.
    """
    schema = _get_schema()
    motifs: list[MotifPattern] = []

    if not _LIBRARY_DIR.is_dir():
        logger.warning("Motif library directory not found: %s", _LIBRARY_DIR)
        return motifs

    for json_path in sorted(_LIBRARY_DIR.glob("*.json")):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping %s: invalid JSON — %s", json_path.name, exc)
            continue

        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as exc:
            logger.warning(
                "Skipping %s: schema validation failed — %s",
                json_path.name,
                exc.message,
            )
            continue

        try:
            motifs.append(_json_to_motif(data))
        except Exception as exc:
            logger.warning(
                "Skipping %s: failed to convert to MotifPattern — %s",
                json_path.name,
                exc,
            )
            continue

    return motifs


# ── Public API ───────────────────────────────────────────────────────

MOTIF_REGISTRY: list[MotifPattern] = _discover_motifs()


def get_motif(motif_id: str) -> MotifPattern | None:
    """Look up a motif by its unique ID. Returns None if not found."""
    return next((m for m in MOTIF_REGISTRY if m.motif_id == motif_id), None)


def list_motifs(
    family: str | None = None,
    tags: list[str] | None = None,
) -> list[MotifPattern]:
    """
    List motifs with optional filtering.

    Args:
        family: If given, filter to motifs whose source matches this value.
        tags: If given, filter to motifs whose metadata contains all of
              these tags (looks in the original JSON ``tags`` field stored
              in ``metadata["tags"]``).

    Returns:
        Filtered list of MotifPattern objects.
    """
    result = list(MOTIF_REGISTRY)

    if family is not None:
        result = [m for m in result if m.source == family]

    if tags is not None:
        tag_set = set(tags)
        result = [
            m for m in result
            if tag_set.issubset(set(m.metadata.get("tags", [])))
        ]

    return result
