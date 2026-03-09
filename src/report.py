"""
Novelty assessment, provenance tracking, and artifact export.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.extract import ExtractionResult
from src.benchmark import ComparisonResult
from src.compose import CompositionRecipe

logger = logging.getLogger(__name__)


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
    if not comparisons:
        return NoveltyVerdict(
            candidate_id="",
            is_novel=False,
            reasons=["No comparisons available"],
        )

    candidate_id = comparisons[0].candidate_id
    reasons: list[str] = []
    best_metric = ""
    best_value = 0.0

    for comp in comparisons:
        for metric, value in comp.improvements.items():
            if metric == "simulated_fidelity":
                continue
            if value > threshold:
                reasons.append(
                    f"{metric} improved by {value * 100:.1f}% vs {comp.baseline_id}"
                )
            if value > best_value:
                best_value = value
                best_metric = metric

    is_novel = len(reasons) > 0

    return NoveltyVerdict(
        candidate_id=candidate_id,
        is_novel=is_novel,
        best_improvement_metric=best_metric,
        best_improvement_value=best_value,
        reasons=reasons,
    )


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
    web_details: list[dict] = []
    all_sources: set[str] = set()

    for web_id in recipe.web_sequence:
        try:
            web = web_library.get(web_id)
            sources = web.sources
            detail = {
                "web_id": web_id,
                "source_algorithms": sources,
                "role_in_composition": web.role or "unknown",
                "spider_count": web.spider_count,
            }
            all_sources.update(sources)
        except Exception:
            logger.warning("Web %s not found in library", web_id)
            detail = {
                "web_id": web_id,
                "source_algorithms": [],
                "role_in_composition": "unknown",
                "spider_count": 0,
            }
        web_details.append(detail)

    return CandidateProvenance(
        candidate_id=recipe.candidate_id,
        recipe=recipe,
        web_details=web_details,
        all_source_algorithms=sorted(all_sources),
    )


def provenance_to_dict(prov: CandidateProvenance) -> dict:
    """Serialize provenance to a JSON-compatible dictionary."""
    return {
        "candidate_id": prov.candidate_id,
        "template_name": prov.recipe.template_name,
        "web_sequence": prov.recipe.web_sequence,
        "connections": prov.recipe.connections,
        "web_details": prov.web_details,
        "all_source_algorithms": prov.all_source_algorithms,
    }


def provenance_to_markdown(prov: CandidateProvenance) -> str:
    """Render provenance as a human-readable Markdown snippet."""
    lines: list[str] = []
    lines.append(f"## Provenance: {prov.candidate_id}")
    lines.append("")
    lines.append(f"**Template:** {prov.recipe.template_name or 'N/A'}")
    lines.append("")
    lines.append("### Web Sequence")
    lines.append("")
    lines.append("| Web ID | Role | Spider Count | Source Algorithms |")
    lines.append("|--------|------|-------------|-------------------|")
    for detail in prov.web_details:
        sources = ", ".join(detail["source_algorithms"]) or "—"
        lines.append(
            f"| {detail['web_id']} | {detail['role_in_composition']} "
            f"| {detail['spider_count']} | {sources} |"
        )
    lines.append("")
    lines.append("### All Source Algorithms")
    lines.append("")
    for algo in prov.all_source_algorithms:
        lines.append(f"- {algo}")
    lines.append("")

    return "\n".join(lines)


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
    output_dir = Path(output_dir)
    candidate_dir = output_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []

    if "json" in formats:
        data = {
            "candidate_id": candidate_id,
            "verdict": {
                "is_novel": verdict.is_novel,
                "best_improvement_metric": verdict.best_improvement_metric,
                "best_improvement_value": verdict.best_improvement_value,
                "reasons": verdict.reasons,
            },
            "candidate_metrics": {
                "gate_count": extraction.gate_count,
                "two_qubit_count": extraction.two_qubit_count,
                "t_count": extraction.t_count,
                "depth": extraction.depth,
            },
            "comparisons": [
                {
                    "baseline_id": c.baseline_id,
                    "improvements": c.improvements,
                    "overall_better": c.overall_better,
                }
                for c in comparisons
            ],
            "provenance": provenance_to_dict(provenance),
        }
        json_path = candidate_dir / "result.json"
        json_path.write_text(json.dumps(data, indent=2, default=str))
        exported.append(json_path)

    if "markdown" in formats:
        md_lines: list[str] = []
        md_lines.append(f"# Novel Algorithm: {candidate_id}")
        md_lines.append("")
        md_lines.append("## Verdict")
        md_lines.append("")
        md_lines.append(
            f"- **Novel:** {'Yes' if verdict.is_novel else 'No'}"
        )
        md_lines.append(
            f"- **Best improvement:** {verdict.best_improvement_metric} "
            f"({verdict.best_improvement_value * 100:.1f}%)"
        )
        if verdict.reasons:
            md_lines.append("- **Reasons:**")
            for reason in verdict.reasons:
                md_lines.append(f"  - {reason}")
        md_lines.append("")
        md_lines.append("## Candidate Metrics")
        md_lines.append("")
        md_lines.append("| Metric | Value |")
        md_lines.append("|--------|-------|")
        md_lines.append(f"| Gate count | {extraction.gate_count} |")
        md_lines.append(f"| Two-qubit count | {extraction.two_qubit_count} |")
        md_lines.append(f"| T count | {extraction.t_count} |")
        md_lines.append(f"| Depth | {extraction.depth} |")
        md_lines.append("")
        md_lines.append("## Comparisons")
        md_lines.append("")
        md_lines.append("| Baseline | Overall Better | Key Improvements |")
        md_lines.append("|----------|---------------|-----------------|")
        for c in comparisons:
            imps = ", ".join(
                f"{k}: {v * 100:.1f}%"
                for k, v in c.improvements.items()
                if k != "simulated_fidelity" and v > 0
            )
            md_lines.append(
                f"| {c.baseline_id} | {'Yes' if c.overall_better else 'No'} | {imps or '—'} |"
            )
        md_lines.append("")
        md_lines.append(provenance_to_markdown(provenance))

        md_path = candidate_dir / "report.md"
        md_path.write_text("\n".join(md_lines))
        exported.append(md_path)

    if "qasm" in formats and extraction.qasm:
        qasm_path = candidate_dir / "extraction.qasm"
        qasm_path.write_text(extraction.qasm)
        exported.append(qasm_path)

    return exported


def generate_summary_report(
    all_verdicts: list[NoveltyVerdict],
    output_dir: str | Path,
) -> Path:
    """Generate a top-level summary report across all candidates."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(all_verdicts)
    novel = [v for v in all_verdicts if v.is_novel]
    novel_count = len(novel)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append("# ZX-Webs Discovery Summary")
    lines.append("")
    lines.append(f"**Date:** {date_str}")
    lines.append(f"**Total candidates assessed:** {total}")
    lines.append(f"**Novel candidates found:** {novel_count}")
    lines.append("")

    if novel:
        lines.append("## Novel Candidates")
        lines.append("")
        lines.append("| Candidate | Best Metric | Improvement |")
        lines.append("|-----------|------------|------------|")
        for v in sorted(novel, key=lambda x: x.best_improvement_value, reverse=True):
            lines.append(
                f"| {v.candidate_id} | {v.best_improvement_metric} "
                f"| {v.best_improvement_value * 100:.1f}% |"
            )
        lines.append("")

    lines.append("## All Candidates")
    lines.append("")
    lines.append("| Candidate | Novel | Best Metric | Improvement |")
    lines.append("|-----------|-------|------------|------------|")
    for v in all_verdicts:
        lines.append(
            f"| {v.candidate_id} | {'Yes' if v.is_novel else 'No'} "
            f"| {v.best_improvement_metric or '—'} "
            f"| {v.best_improvement_value * 100:.1f}% |"
        )
    lines.append("")

    report_path = output_dir / "discovery_summary.md"
    report_path.write_text("\n".join(lines))
    return report_path
