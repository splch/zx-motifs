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
        raise NotImplementedError


# ── Stage runners ────────────────────────────────────────────────────


def run_stage_1(cfg: PipelineConfig) -> None:
    """Build the Qiskit algorithm corpus and export QASM files."""
    raise NotImplementedError


def run_stage_2(cfg: PipelineConfig) -> None:
    """Convert QASM files to ZX-diagrams at multiple simplification levels."""
    raise NotImplementedError


def run_stage_3(cfg: PipelineConfig) -> None:
    """Mine common sub-diagrams (ZX-Webs) from the diagram corpus."""
    raise NotImplementedError


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
    raise NotImplementedError


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
