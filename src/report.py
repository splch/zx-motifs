"""
report.py — Novelty assessment, provenance tracking, and artifact export.

Merges stage7_report/{novelty,provenance,export}.py. WebProvenance replaced
with plain dicts inside CandidateProvenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.extract import ExtractionResult
from src.benchmark import ComparisonResult
from src.compose import CompositionRecipe


# ── Novelty ─────────────────────────────────────────────────────────


@dataclass
class NoveltyVerdict:
    """Assessment of whether a candidate is genuinely novel."""

    candidate_id: str
    is_novel: bool
    best_improvement_metric: str = ""
    best_improvement_value: float = 0.0
    reasons: list[str] | None = None


def assess_novelty(
    comparisons: list[ComparisonResult],
    threshold: float,
) -> NoveltyVerdict:
    """Determine whether a candidate's improvements are significant."""
    raise NotImplementedError


# ── Provenance ──────────────────────────────────────────────────────


@dataclass
class CandidateProvenance:
    """Full provenance trail for a composed candidate.

    web_details is a list of plain dicts with keys:
    web_id, source_algorithms, role_in_composition, spider_count.
    """

    candidate_id: str
    recipe: CompositionRecipe
    web_details: list[dict] = field(default_factory=list)
    all_source_algorithms: list[str] = field(default_factory=list)


def build_provenance(
    recipe: CompositionRecipe,
    web_library: Any,  # WebLibrary
) -> CandidateProvenance:
    """Reconstruct provenance from a composition recipe and the web library."""
    raise NotImplementedError


def provenance_to_dict(prov: CandidateProvenance) -> dict:
    """Serialize provenance to a JSON-compatible dictionary."""
    raise NotImplementedError


def provenance_to_markdown(prov: CandidateProvenance) -> str:
    """Render provenance as a human-readable Markdown snippet."""
    raise NotImplementedError


# ── Export ──────────────────────────────────────────────────────────


def export_novel_algorithm(
    candidate_id: str,
    verdict: NoveltyVerdict,
    extraction: ExtractionResult,
    comparisons: list[ComparisonResult],
    provenance: CandidateProvenance,
    diagram_graph: Any,  # pyzx.Graph
    output_dir: str | Path,
    formats: list[str],
) -> list[Path]:
    """Export all artifacts for a single novel algorithm."""
    raise NotImplementedError


def generate_summary_report(
    all_verdicts: list[NoveltyVerdict],
    output_dir: str | Path,
) -> Path:
    """Generate a top-level summary report across all candidates."""
    raise NotImplementedError
