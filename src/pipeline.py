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
    raise NotImplementedError


def run_stage_2(cfg: PipelineConfig) -> None:
    """Convert QASM files to ZX-diagrams at multiple simplification levels."""
    from src.zx import (
        load_qasm_file,
        pyzx_circuit_to_graph,
        qasm_to_pyzx_circuit,
        save_diagram,
        simplify_graph,
    )

    corpus_dir = Path(cfg.corpus.get("output_dir", "data/corpus"))
    output_dir = Path(cfg.zx_conversion.get("output_dir", "data/diagrams"))
    levels = cfg.zx_conversion.get("simplification_levels", ["raw", "clifford", "full"])

    qasm_files = sorted(corpus_dir.glob("*.qasm"))
    if not qasm_files:
        logger.warning("No QASM files found in %s", corpus_dir)
        return

    logger.info("Converting %d QASM files to ZX-diagrams", len(qasm_files))

    for qasm_path in qasm_files:
        stem = qasm_path.stem  # e.g. "ghz_3q"
        # Parse algo name and qubit count from filename convention {key}_{n}q
        parts = stem.rsplit("_", 1)
        algo_name = parts[0] if len(parts) == 2 else stem
        try:
            n_qubits = int(parts[1].rstrip("q")) if len(parts) == 2 else 0
        except ValueError:
            n_qubits = 0

        try:
            qasm_text = load_qasm_file(qasm_path)
            circuit = qasm_to_pyzx_circuit(qasm_text)
            graph = pyzx_circuit_to_graph(circuit)
            result = simplify_graph(graph)

            level_graphs = {
                "raw": result.raw,
                "clifford": result.clifford,
                "full": result.full,
            }

            for level_name in levels:
                g = level_graphs.get(level_name)
                if g is None:
                    continue
                diagram_id = f"{stem}_{level_name}"
                metadata = {
                    "source_algorithm": algo_name,
                    "n_qubits": n_qubits,
                    "level": level_name,
                }
                save_diagram(g, diagram_id, output_dir, metadata)

            logger.info("Converted %s (%d spiders raw)", stem, result.spider_counts["raw"])
        except Exception:
            logger.warning("Failed to convert %s", stem, exc_info=True)


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
    raise NotImplementedError


def run_stage_5(cfg: PipelineConfig) -> None:
    """Filter candidates to those with extractable circuits."""
    raise NotImplementedError


def run_stage_6(cfg: PipelineConfig) -> None:
    """Benchmark surviving candidates against application baselines."""
    raise NotImplementedError


def run_stage_7(cfg: PipelineConfig) -> None:
    """Report any algorithm that outperforms existing solutions."""
    raise NotImplementedError


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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = PipelineConfig.from_yaml(args.config)
    run_pipeline(cfg, stage=args.stage)


if __name__ == "__main__":
    main()
