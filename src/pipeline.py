"""
End-to-end orchestrator for the seven-stage discovery pipeline.

Usage:
    python -m src.pipeline --config config.yaml           # full run
    python -m src.pipeline --config config.yaml --stage 3  # only Stage 3
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Typed wrapper around the YAML configuration file."""

    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def corpus(self) -> dict:
        return self.raw.get("corpus", {})

    @property
    def zx_conversion(self) -> dict:
        return self.raw.get("zx_conversion", {})

    @property
    def mining(self) -> dict:
        return self.raw.get("mining", {})

    @property
    def composition(self) -> dict:
        return self.raw.get("composition", {})

    @property
    def extraction(self) -> dict:
        return self.raw.get("extraction", {})

    @property
    def benchmark(self) -> dict:
        return self.raw.get("benchmark", {})

    @property
    def reporting(self) -> dict:
        return self.raw.get("reporting", {})

    @property
    def workers(self) -> int | None:
        return self.raw.get("workers", None)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(raw=raw or {})


# ── Stage runners ────────────────────────────────────────────────────


def run_stage_1(cfg: PipelineConfig) -> None:
    """Build the Qiskit algorithm corpus and export QASM files."""
    from src.corpus import AlgorithmRegistry, build_default_registry, export_corpus

    output_dir = Path(cfg.corpus.get("output_dir", "data/corpus"))
    gate_set = cfg.corpus.get("gate_set", ["cx", "rz", "h"])
    qubit_sizes = cfg.corpus.get("qubit_sizes", {"default": [3, 4, 5]})
    categories = cfg.corpus.get("categories", [])

    registry = build_default_registry()

    if categories:
        filtered = AlgorithmRegistry()
        for cat in categories:
            for entry in registry.by_category(cat):
                filtered.register(entry)
        registry = filtered

    written = export_corpus(registry, output_dir, gate_set, qubit_sizes, workers=cfg.workers)
    logger.info("Stage 1: exported %d QASM files to %s", len(written), output_dir)


def run_stage_2(cfg: PipelineConfig) -> None:
    """Convert QASM files to ZX-diagrams at multiple simplification levels."""
    from src.parallel import parallel_map
    from src.zx import convert_single_qasm

    corpus_dir = Path(cfg.corpus.get("output_dir", "data/corpus"))
    output_dir = Path(cfg.zx_conversion.get("output_dir", "data/diagrams"))
    levels = cfg.zx_conversion.get("simplification_levels", ["raw", "clifford", "full"])

    qasm_files = sorted(corpus_dir.glob("*.qasm"))
    if not qasm_files:
        logger.warning("No QASM files found in %s", corpus_dir)
        return

    timeout = cfg.zx_conversion.get("timeout_per_circuit", None)

    logger.info("Converting %d QASM files to ZX-diagrams", len(qasm_files))

    items = [(str(p), levels, str(output_dir), timeout) for p in qasm_files]
    results = parallel_map(convert_single_qasm, items, cfg.workers, desc="ZX conversion")

    for r in results:
        if r is not None:
            stem, spider_count = r
            logger.info("Converted %s (%d spiders raw)", stem, spider_count)


def run_stage_3(cfg: PipelineConfig) -> None:
    """Mine common sub-diagrams (ZX-Webs) from the diagram corpus."""
    from src.mining import WebLibrary, mine_webs
    from src.zx import load_all_diagrams

    diagrams_dir = Path(cfg.zx_conversion.get("output_dir", "data/diagrams"))
    output_dir = Path(cfg.mining.get("output_dir", "data/webs"))
    min_support = cfg.mining.get("min_support", 2)
    min_spiders = cfg.mining.get("min_spiders", 3)
    max_spiders = cfg.mining.get("max_spiders", 30)
    phase_abstraction = cfg.mining.get("phase_abstraction", "class")
    reduction_level = cfg.mining.get("reduction_level", "full")

    records = load_all_diagrams(diagrams_dir, level=reduction_level)
    if not records:
        logger.warning("No diagrams found in %s at level '%s'", diagrams_dir, reduction_level)
        return

    diagram_tuples = [(rec.source_algorithm, graph) for rec, graph in records]
    logger.info("Mining webs from %d diagrams (min_support=%d)", len(diagram_tuples), min_support)

    webs = mine_webs(
        diagrams=diagram_tuples,
        min_support=min_support,
        min_spiders=min_spiders,
        max_spiders=max_spiders,
        phase_abstraction=phase_abstraction,
    )

    library = WebLibrary(output_dir)
    for web in webs:
        library.add(web)
    library.save_index()

    logger.info("Mined %d webs, saved to %s", len(webs), output_dir)


def run_stage_4(cfg: PipelineConfig) -> None:
    """Compose candidate algorithms by combining ZX-Webs."""
    import json

    from src.compose import (
        BUILTIN_TEMPLATES,
        compose_from_template,
        load_templates_from_config,
    )
    from src.mining import WebLibrary

    webs_dir = Path(cfg.mining.get("output_dir", "data/webs"))
    output_dir = Path(cfg.composition.get("output_dir", "data/candidates"))
    max_candidates = cfg.composition.get("max_candidates", 1000)
    max_qubits = cfg.composition.get("max_qubits", 20)
    enforce_flow = cfg.composition.get("enforce_flow", True)
    template_specs = cfg.composition.get("templates", [])

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load web library
    library = WebLibrary(webs_dir)
    library.load_index()

    # Combine built-in and config templates
    templates = list(BUILTIN_TEMPLATES)
    if template_specs:
        templates.extend(load_templates_from_config(template_specs))

    total_saved = 0
    for template in templates:
        if total_saved >= max_candidates:
            break

        candidates = compose_from_template(
            template, library, max_qubits, enforce_flow=enforce_flow
        )

        for graph, recipe in candidates:
            if total_saved >= max_candidates:
                break

            candidate_data = {
                "candidate_id": recipe.candidate_id,
                "template_name": recipe.template_name,
                "web_sequence": recipe.web_sequence,
                "connections": recipe.connections,
                "graph": graph.to_dict(),
            }

            out_path = output_dir / f"{recipe.candidate_id}.json"
            from src.zx import _sanitize_phase_tildes

            out_path.write_text(_sanitize_phase_tildes(json.dumps(candidate_data, default=str)))
            total_saved += 1

    logger.info(
        "Stage 4: composed %d candidates from %d templates, saved to %s",
        total_saved,
        len(templates),
        output_dir,
    )


def run_stage_5(cfg: PipelineConfig) -> None:
    """Filter candidates to those with extractable circuits."""
    import json

    from src.extract import run_extraction_filter

    candidate_dir = Path(cfg.extraction.get("output_dir", "data/candidates"))
    post_optimize = cfg.extraction.get("post_optimize", True)
    cnot_ratio_threshold = cfg.extraction.get("discard_if_cnot_ratio", 5.0)

    survivors, stats = run_extraction_filter(
        candidate_dir, post_optimize, cnot_ratio_threshold, workers=cfg.workers
    )

    summary = {
        "stats": {
            "total_candidates": stats.total_candidates,
            "passed_flow_check": stats.passed_flow_check,
            "passed_extraction": stats.passed_extraction,
            "passed_size_filter": stats.passed_size_filter,
            "final_survivors": stats.final_survivors,
        },
        "survivors": [
            {
                "candidate_id": s.candidate_id,
                "qasm": s.qasm,
                "gate_count": s.gate_count,
                "two_qubit_count": s.two_qubit_count,
                "t_count": s.t_count,
                "depth": s.depth,
                "flow_used": s.flow_used.value,
            }
            for s in survivors
        ],
    }

    summary_path = candidate_dir / "extraction_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info(
        "Stage 5: %d/%d candidates survived extraction filter",
        stats.final_survivors,
        stats.total_candidates,
    )


def _serialize_comparison(comp: Any) -> dict:
    """Convert a ComparisonResult to a plain dict for JSON output."""
    return {
        "candidate_id": comp.candidate_id,
        "baseline_id": comp.baseline_id,
        "candidate_metrics": {
            "n_qubits": comp.candidate_metrics.n_qubits,
            "gate_count": comp.candidate_metrics.gate_count,
            "two_qubit_count": comp.candidate_metrics.two_qubit_count,
            "t_count": comp.candidate_metrics.t_count,
            "depth": comp.candidate_metrics.depth,
            "gate_density": comp.candidate_metrics.gate_density,
            "entanglement_ratio": comp.candidate_metrics.entanglement_ratio,
        },
        "baseline_metrics": {
            "n_qubits": comp.baseline_metrics.n_qubits,
            "gate_count": comp.baseline_metrics.gate_count,
            "two_qubit_count": comp.baseline_metrics.two_qubit_count,
            "t_count": comp.baseline_metrics.t_count,
            "depth": comp.baseline_metrics.depth,
            "gate_density": comp.baseline_metrics.gate_density,
            "entanglement_ratio": comp.baseline_metrics.entanglement_ratio,
        },
        "improvements": comp.improvements,
        "overall_better": comp.overall_better,
    }


def run_stage_6(cfg: PipelineConfig) -> None:
    """Benchmark surviving candidates against application baselines."""
    import json

    from src.benchmark import compare_against_baselines
    from src.parallel import parallel_map

    candidate_dir = Path(cfg.extraction.get("output_dir", "data/candidates"))
    corpus_dir = Path(cfg.corpus.get("output_dir", "data/corpus"))
    output_dir = Path(cfg.benchmark.get("output_dir", "data/results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = candidate_dir / "extraction_summary.json"
    if not summary_path.exists():
        logger.warning("No extraction_summary.json found in %s", candidate_dir)
        return

    summary = json.loads(summary_path.read_text())
    survivors = summary.get("survivors", [])
    if not survivors:
        logger.info("Stage 6: no survivors to benchmark")
        return

    items = [
        (survivor["qasm"], survivor["candidate_id"], str(corpus_dir))
        for survivor in survivors
        if survivor.get("qasm") and survivor.get("candidate_id")
    ]

    all_comparison_lists = parallel_map(
        compare_against_baselines, items, cfg.workers, desc="benchmark"
    )

    all_comparisons: list[dict] = []
    for comparisons in all_comparison_lists:
        for comp in comparisons:
            all_comparisons.append(_serialize_comparison(comp))

    results_path = output_dir / "benchmark_results.json"
    results_path.write_text(json.dumps(all_comparisons, indent=2))

    n_better = sum(1 for c in all_comparisons if c["overall_better"])
    logger.info(
        "Stage 6: %d comparisons, %d show improvement, saved to %s",
        len(all_comparisons),
        n_better,
        results_path,
    )


def run_stage_7(cfg: PipelineConfig) -> None:
    """Report any algorithm that outperforms existing solutions."""
    import json

    from src.benchmark import CircuitMetrics, ComparisonResult
    from src.extract import ExtractionResult, FlowType
    from src.compose import CompositionRecipe
    from src.mining import WebLibrary
    from src.report import (
        assess_novelty,
        build_provenance,
        export_novel_algorithm,
        generate_summary_report,
    )

    output_dir = Path(cfg.reporting.get("output_dir", "data/results"))
    threshold = cfg.reporting.get("improvement_threshold", 0.05)
    export_formats = cfg.reporting.get("export_formats", ["json", "markdown", "qasm"])
    candidate_dir = Path(cfg.extraction.get("output_dir", "data/candidates"))
    webs_dir = Path(cfg.mining.get("output_dir", "data/webs"))

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark results
    results_path = output_dir / "benchmark_results.json"
    if not results_path.exists():
        logger.warning("No benchmark_results.json found in %s", output_dir)
        return

    raw_comparisons = json.loads(results_path.read_text())
    if not raw_comparisons:
        logger.info("Stage 7: no benchmark comparisons to assess")
        return

    # Group comparisons by candidate_id
    comparisons_by_id: dict[str, list[ComparisonResult]] = {}
    for entry in raw_comparisons:
        cid = entry["candidate_id"]

        def _build_metrics(d: dict) -> CircuitMetrics:
            return CircuitMetrics(
                n_qubits=d.get("n_qubits", 0),
                gate_count=d.get("gate_count", 0),
                two_qubit_count=d.get("two_qubit_count", 0),
                t_count=d.get("t_count", 0),
                depth=d.get("depth", 0),
                gate_density=d.get("gate_density", 0.0),
                entanglement_ratio=d.get("entanglement_ratio", 0.0),
            )

        comp = ComparisonResult(
            candidate_id=cid,
            baseline_id=entry["baseline_id"],
            candidate_metrics=_build_metrics(entry["candidate_metrics"]),
            baseline_metrics=_build_metrics(entry["baseline_metrics"]),
            improvements=entry.get("improvements", {}),
            overall_better=entry.get("overall_better", False),
        )
        comparisons_by_id.setdefault(cid, []).append(comp)

    # Load extraction summary for survivor data
    summary_path = candidate_dir / "extraction_summary.json"
    survivors_by_id: dict[str, dict] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for s in summary.get("survivors", []):
            survivors_by_id[s["candidate_id"]] = s

    # Load web library
    library = WebLibrary(webs_dir)
    library.load_index()

    all_verdicts = []
    novel_count = 0

    for cid, comps in comparisons_by_id.items():
        verdict = assess_novelty(comps, threshold)
        verdict.candidate_id = cid
        all_verdicts.append(verdict)

        if not verdict.is_novel:
            continue
        novel_count += 1

        # Build extraction result from survivor data
        survivor = survivors_by_id.get(cid, {})
        extraction = ExtractionResult(
            success=True,
            qasm=survivor.get("qasm"),
            gate_count=survivor.get("gate_count", 0),
            two_qubit_count=survivor.get("two_qubit_count", 0),
            t_count=survivor.get("t_count", 0),
            depth=survivor.get("depth", 0),
            flow_used=FlowType(survivor.get("flow_used", "none")),
            candidate_id=cid,
        )

        # Load candidate JSON for recipe
        candidate_path = candidate_dir / f"{cid}.json"
        if candidate_path.exists():
            cand_data = json.loads(candidate_path.read_text())
            recipe = CompositionRecipe(
                candidate_id=cid,
                template_name=cand_data.get("template_name"),
                web_sequence=cand_data.get("web_sequence", []),
                connections=cand_data.get("connections", []),
            )
            provenance = build_provenance(recipe, library)
        else:
            recipe = CompositionRecipe(
                candidate_id=cid, template_name=None, web_sequence=[]
            )
            provenance = build_provenance(recipe, library)

        export_novel_algorithm(
            candidate_id=cid,
            verdict=verdict,
            extraction=extraction,
            comparisons=comps,
            provenance=provenance,
            diagram_graph=None,
            output_dir=output_dir,
            formats=export_formats,
        )

    # Summary report
    report_path = generate_summary_report(all_verdicts, output_dir)

    logger.info(
        "Stage 7: %d/%d candidates are novel, report at %s",
        novel_count,
        len(all_verdicts),
        report_path,
    )


# ── Orchestration ────────────────────────────────────────────────────

STAGES = {
    1: ("Build corpus", run_stage_1),
    2: ("Convert to ZX", run_stage_2),
    3: ("Mine ZX-Webs", run_stage_3),
    4: ("Compose candidates", run_stage_4),
    5: ("Extract circuits", run_stage_5),
    6: ("Benchmark", run_stage_6),
    7: ("Report", run_stage_7),
}


def run_pipeline(cfg: PipelineConfig, stage: int | None = None) -> None:
    """Execute the pipeline — all stages or a single stage."""
    if stage is not None:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        name, runner = STAGES[stage]
        logger.info("Running Stage %d: %s", stage, name)
        runner(cfg)
        return

    for stage_num, (name, runner) in STAGES.items():
        logger.info("Running Stage %d: %s", stage_num, name)
        runner(cfg)


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for ``python -m src.pipeline``."""
    parser = argparse.ArgumentParser(
        description="ZX Pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        choices=range(1, 8),
        help="Run only this stage (1–7).  Omit to run all.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count; 1 = sequential).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = PipelineConfig.from_yaml(args.config)
    if args.workers is not None:
        cfg.raw["workers"] = args.workers
    run_pipeline(cfg, stage=args.stage)


if __name__ == "__main__":
    main()
