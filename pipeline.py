"""
Pipeline Orchestrator
======================
Runs the full 7-step ZX-Web discovery pipeline end-to-end.

Usage
-----
    from zx_discovery.pipeline import run_pipeline
    report = run_pipeline()
    print(report_text)

Or from the command line::

    python -m zx_discovery.pipeline
"""

from __future__ import annotations
from typing import Optional

from . import corpus as step1
from . import zx_convert as step2
from . import zx_webs as step3
from . import composer as step4
from . import extractor as step5
from . import benchmark as step6
from . import reporter as step7


def run_pipeline(
    # Step 3 params
    max_subgraph_size: int = 5,
    min_frequency: int = 2,
    # Step 4 params
    max_candidates: int = 150,
    web_combo_size: int = 3,
    seed: int = 42,
    # Step 5 params
    min_gates: int = 3,
    max_qubits: int = 8,
    # Output
    verbose: bool = True,
) -> step7.PipelineReport:
    """Run the full ZX-Web discovery pipeline.

    Parameters
    ----------
    max_subgraph_size : int
        Largest ZX-Web motif size (vertices).
    min_frequency : int
        Minimum distinct-algorithm frequency for a motif.
    max_candidates : int
        Number of candidate algorithms to compose.
    web_combo_size : int
        Number of webs to glue per candidate (2–4).
    seed : int
        Random seed for composition.
    min_gates : int
        Minimum gate count to keep after extraction.
    max_qubits : int
        Maximum qubit width for benchmarking.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    PipelineReport
    """
    def log(msg: str):
        if verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Step 1: Build algorithm corpus
    # ------------------------------------------------------------------
    log("\n[Step 1/7] Building quantum algorithm corpus …")
    qc_corpus = step1.build_corpus()
    log(f"  Built {len(qc_corpus)} algorithms")
    qasm_corpus = step1.corpus_to_qasm(qc_corpus)
    log(f"  Exported {len(qasm_corpus)} to QASM")

    # ------------------------------------------------------------------
    # Step 2: Convert to ZX-diagrams and simplify
    # ------------------------------------------------------------------
    log("\n[Step 2/7] Converting to ZX-diagrams and simplifying …")
    zx_entries = step2.convert_corpus(qasm_corpus)
    log(f"  Converted {len(zx_entries)} algorithms")
    for name, entry in list(zx_entries.items())[:3]:
        log(f"    {name:25s}  orig={entry.original_stats['vertices']:3d}v  "
            f"reduced={entry.reduced_stats['vertices']:3d}v")
    if len(zx_entries) > 3:
        log(f"    … and {len(zx_entries) - 3} more")

    # ------------------------------------------------------------------
    # Step 3: Mine ZX-Webs
    # ------------------------------------------------------------------
    log(f"\n[Step 3/7] Mining ZX-Web motifs (size ≤ {max_subgraph_size}, "
        f"freq ≥ {min_frequency}) …")
    webs, corpus_nx = step3.mine_zx_webs(
        zx_entries,
        max_subgraph_size=max_subgraph_size,
        min_frequency=min_frequency,
    )
    log(f"  Found {len(webs)} recurring ZX-Web motifs")
    # Show top webs
    top = sorted(webs.values(), key=lambda w: (-w.frequency, w.size))[:5]
    for w in top:
        algos = list(w.sources.keys())[:3]
        log(f"    web {w.web_id[:8]}  size={w.size}  "
            f"freq={w.frequency}  in: {', '.join(algos)}")

    # ------------------------------------------------------------------
    # Step 4: Compose candidate algorithms
    # ------------------------------------------------------------------
    log(f"\n[Step 4/7] Composing up to {max_candidates} candidate "
        f"algorithms …")
    candidates = step4.compose_candidates(
        webs, corpus_nx,
        max_candidates=max_candidates,
        web_combo_size=web_combo_size,
        seed=seed,
    )
    log(f"  Composed {len(candidates)} candidates")

    # ------------------------------------------------------------------
    # Step 5: Filter to extractable circuits
    # ------------------------------------------------------------------
    log("\n[Step 5/7] Extracting quantum circuits from candidates …")
    extracted = step5.filter_candidates(
        candidates,
        min_gates=min_gates,
        max_qubits=max_qubits,
    )
    log(f"  {len(extracted)} / {len(candidates)} survived extraction")
    for ec in extracted[:3]:
        log(f"    {ec.candidate.candidate_id}  "
            f"{ec.n_qubits}q  {ec.gate_count}g  T={ec.t_count}")

    # ------------------------------------------------------------------
    # Step 6: Benchmark
    # ------------------------------------------------------------------
    log("\n[Step 6/7] Benchmarking candidates and corpus …")
    cand_results, base_results = step6.benchmark_candidates(
        extracted, qasm_corpus
    )
    log(f"  Benchmarked {len(cand_results)} candidates, "
        f"{len(base_results)} baselines")

    # ------------------------------------------------------------------
    # Step 7: Report
    # ------------------------------------------------------------------
    log("\n[Step 7/7] Generating report …")
    report = step7.generate_report(
        candidate_results=cand_results,
        baseline_results=base_results,
        extracted=extracted,
        corpus_size=len(qc_corpus),
        n_webs=len(webs),
        n_composed=len(candidates),
    )

    report_text = step7.report_to_text(report)
    log("\n" + report_text)

    return report


# Allow: python -m zx_discovery.pipeline
if __name__ == "__main__":
    run_pipeline()
