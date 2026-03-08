"""
Step 7 – Report Novel Algorithms that Outperform Existing Solutions
=====================================================================
Compares candidate benchmark results against the corpus baselines and
flags any candidate that is Pareto-dominant on at least one meaningful
metric while not being strictly worse on any other.

Dominance Criteria
------------------
A candidate **outperforms** the corpus if it satisfies ALL of:
    • ``is_novel == True``  (not equivalent to a known algorithm)
    • Pareto-dominates ≥ 1 corpus circuit of the same qubit width on
      the metric vector (lower gates_per_qubit, higher entanglement,
      higher expressibility, lower depth).

Output
------
Returns a structured report suitable for logging or display, including:
    • Summary statistics of the pipeline run
    • Per-candidate scorecards
    • Highlighted outperformers
    • QASM of each outperformer for downstream use
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import datetime

from .benchmark import BenchmarkResult
from .extractor import ExtractedCandidate


@dataclass
class PipelineReport:
    """Full report from one run of the ZX-Web discovery pipeline."""
    timestamp: str
    corpus_size: int
    n_webs_mined: int
    n_candidates_composed: int
    n_candidates_extracted: int
    n_candidates_benchmarked: int
    n_outperformers: int
    baseline_summary: Dict[str, float]
    outperformers: List[Dict]
    full_candidate_results: List[Dict]
    full_baseline_results: List[Dict]


def _result_to_dict(r: BenchmarkResult) -> Dict:
    return {
        "name": r.name,
        "n_qubits": r.n_qubits,
        "gate_count": r.gate_count,
        "two_qubit_count": r.two_qubit_count,
        "t_count": r.t_count,
        "depth": r.depth,
        "two_qubit_depth": r.two_qubit_depth,
        "gates_per_qubit": round(r.gates_per_qubit, 3),
        "entanglement_entropy": round(r.entanglement_entropy, 4),
        "expressibility_score": round(r.expressibility_score, 4),
        "is_novel": r.is_novel,
        "source_info": r.source_info,
    }


def _pareto_dominates(
    cand: BenchmarkResult,
    baseline: BenchmarkResult,
) -> bool:
    """Check if *cand* Pareto-dominates *baseline*.

    Metrics (direction):
        gates_per_qubit       – lower is better
        entanglement_entropy  – higher is better
        expressibility_score  – higher is better
        depth                 – lower is better
    """
    metrics = [
        (-cand.gates_per_qubit,       -baseline.gates_per_qubit),
        ( cand.entanglement_entropy,   baseline.entanglement_entropy),
        ( cand.expressibility_score,   baseline.expressibility_score),
        (-cand.depth,                  -baseline.depth),
    ]
    # Filter out invalid measurements (-1)
    valid = [(c, b) for c, b in metrics if c != 1.0 and b != 1.0]
    if not valid:
        return False
    at_least_one_better = False
    for c_val, b_val in valid:
        if c_val < b_val:
            return False    # strictly worse on one dimension
        if c_val > b_val:
            at_least_one_better = True
    return at_least_one_better


def generate_report(
    candidate_results: List[BenchmarkResult],
    baseline_results: List[BenchmarkResult],
    extracted: List[ExtractedCandidate],
    corpus_size: int,
    n_webs: int,
    n_composed: int,
) -> PipelineReport:
    """Generate the final pipeline report.

    Parameters
    ----------
    candidate_results, baseline_results : list of BenchmarkResult
        From ``benchmark.benchmark_candidates``.
    extracted : list of ExtractedCandidate
        The surviving candidates (for QASM output).
    corpus_size, n_webs, n_composed : int
        Pipeline statistics.

    Returns
    -------
    PipelineReport
    """
    # Group baselines by qubit width
    baselines_by_width: Dict[int, List[BenchmarkResult]] = {}
    for br in baseline_results:
        baselines_by_width.setdefault(br.n_qubits, []).append(br)

    # Identify outperformers
    outperformers: List[Dict] = []
    qasm_lookup = {ec.candidate.candidate_id: ec.qasm for ec in extracted}

    for cr in candidate_results:
        if not cr.is_novel:
            continue
        same_width = baselines_by_width.get(cr.n_qubits, [])
        dominates_any = any(_pareto_dominates(cr, bl) for bl in same_width)
        if dominates_any:
            entry = _result_to_dict(cr)
            entry["qasm"] = qasm_lookup.get(cr.name, "")
            entry["dominated_baselines"] = [
                bl.name for bl in same_width
                if _pareto_dominates(cr, bl)
            ]
            outperformers.append(entry)

    # Baseline summary
    if baseline_results:
        avg_gpq = sum(b.gates_per_qubit for b in baseline_results) / len(baseline_results)
        avg_ent = sum(b.entanglement_entropy for b in baseline_results
                      if b.entanglement_entropy >= 0) / max(1, sum(
                          1 for b in baseline_results if b.entanglement_entropy >= 0))
        avg_depth = sum(b.depth for b in baseline_results) / len(baseline_results)
    else:
        avg_gpq = avg_ent = avg_depth = 0.0

    return PipelineReport(
        timestamp=datetime.datetime.now().isoformat(),
        corpus_size=corpus_size,
        n_webs_mined=n_webs,
        n_candidates_composed=n_composed,
        n_candidates_extracted=len(extracted),
        n_candidates_benchmarked=len(candidate_results),
        n_outperformers=len(outperformers),
        baseline_summary={
            "avg_gates_per_qubit": round(avg_gpq, 3),
            "avg_entanglement_entropy": round(avg_ent, 4),
            "avg_depth": round(avg_depth, 1),
        },
        outperformers=outperformers,
        full_candidate_results=[_result_to_dict(r) for r in candidate_results],
        full_baseline_results=[_result_to_dict(r) for r in baseline_results],
    )


def report_to_text(report: PipelineReport) -> str:
    """Render the report as human-readable text."""
    lines = []
    lines.append("=" * 72)
    lines.append("  ZX-Web Discovery Pipeline – Run Report")
    lines.append("=" * 72)
    lines.append(f"  Timestamp:            {report.timestamp}")
    lines.append(f"  Corpus algorithms:    {report.corpus_size}")
    lines.append(f"  ZX-Webs mined:        {report.n_webs_mined}")
    lines.append(f"  Candidates composed:  {report.n_candidates_composed}")
    lines.append(f"  Candidates extracted: {report.n_candidates_extracted}")
    lines.append(f"  Candidates benchmarked: {report.n_candidates_benchmarked}")
    lines.append(f"  OUTPERFORMERS FOUND:  {report.n_outperformers}")
    lines.append("-" * 72)

    lines.append("\n  Corpus Baseline Averages:")
    for k, v in report.baseline_summary.items():
        lines.append(f"    {k:30s} = {v}")

    if report.outperformers:
        lines.append("\n" + "=" * 72)
        lines.append("  OUTPERFORMING CANDIDATES")
        lines.append("=" * 72)
        for i, op in enumerate(report.outperformers, 1):
            lines.append(f"\n  #{i}  {op['name']}")
            lines.append(f"      Qubits: {op['n_qubits']}  |  "
                         f"Gates: {op['gate_count']}  |  "
                         f"Depth: {op['depth']}  |  "
                         f"2Q: {op['two_qubit_count']}")
            lines.append(f"      Gates/qubit: {op['gates_per_qubit']:.3f}  |  "
                         f"Entropy: {op['entanglement_entropy']:.4f}  |  "
                         f"Expressibility: {op['expressibility_score']:.4f}")
            lines.append(f"      Source: {op['source_info']}")
            dom = op.get('dominated_baselines', [])
            if dom:
                lines.append(f"      Dominates: {', '.join(dom)}")
            if op.get('qasm'):
                lines.append(f"      QASM ({len(op['qasm'])} chars) — "
                             f"available in report.outperformers[{i-1}]['qasm']")
    else:
        lines.append("\n  No outperforming candidates found in this run.")
        lines.append("  Consider: expanding the corpus, adjusting composition")
        lines.append("  parameters, or increasing max_candidates.")

    lines.append("\n" + "=" * 72)
    lines.append(f"  Full results: {len(report.full_candidate_results)} candidates, "
                 f"{len(report.full_baseline_results)} baselines")
    lines.append("=" * 72)

    return "\n".join(lines)


def report_to_json(report: PipelineReport) -> str:
    """Serialize the report to JSON."""
    return json.dumps({
        "timestamp": report.timestamp,
        "corpus_size": report.corpus_size,
        "n_webs_mined": report.n_webs_mined,
        "n_candidates_composed": report.n_candidates_composed,
        "n_candidates_extracted": report.n_candidates_extracted,
        "n_candidates_benchmarked": report.n_candidates_benchmarked,
        "n_outperformers": report.n_outperformers,
        "baseline_summary": report.baseline_summary,
        "outperformers": report.outperformers,
        "candidate_results": report.full_candidate_results,
        "baseline_results": report.full_baseline_results,
    }, indent=2)
