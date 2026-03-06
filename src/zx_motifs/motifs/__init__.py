"""
Declarative motif registry for ZX-calculus pattern definitions.

Motifs are defined as JSON files in ``library/`` and validated against
``schema/motif.schema.json``.  The registry auto-discovers all JSON
definitions at import time and exposes them as ``MotifPattern`` objects.
"""
from .registry import MOTIF_REGISTRY, get_motif, list_motifs

__all__ = ["MOTIF_REGISTRY", "get_motif", "list_motifs"]
