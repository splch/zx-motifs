"""
extract.py — Flow detection, circuit extraction, and filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


# ── Flow detection ──────────────────────────────────────────────────


class FlowType(Enum):
    """Classification of the strongest flow found in a diagram."""

    NONE = "none"
    CAUSAL = "causal_flow"
    GFLOW = "gflow"


@dataclass
class FlowResult:
    """Result of a flow detection check."""

    flow_type: FlowType
    exists: bool
    flow_data: Any = None
    check_time_ms: float = 0.0


def check_gflow(graph: Any) -> FlowResult:
    """Check whether a ZX-diagram has generalised flow (gFlow)."""
    raise NotImplementedError


def check_flow(graph: Any) -> FlowResult:
    """Check for gFlow on the diagram."""
    raise NotImplementedError


# ── Extraction ──────────────────────────────────────────────────────


@dataclass
class ExtractionResult:
    """Outcome of attempting to extract a circuit from a ZX-diagram."""

    success: bool
    circuit: Any = None
    qasm: str | None = None
    flow_used: FlowType = FlowType.NONE
    gate_count: int = 0
    two_qubit_count: int = 0
    t_count: int = 0
    depth: int = 0
    error: str | None = None


def extract_circuit_pyzx(graph: Any) -> ExtractionResult:
    """Extract a circuit using PyZX's gFlow-based extraction."""
    raise NotImplementedError


# ── Filter pipeline ─────────────────────────────────────────────────


@dataclass
class FilterStats:
    """Summary statistics for the extraction filtering stage."""

    total_candidates: int = 0
    passed_flow_check: int = 0
    passed_extraction: int = 0
    passed_size_filter: int = 0
    final_survivors: int = 0


def run_extraction_filter(
    candidate_dir: str | Path,
    post_optimize: bool,
    cnot_ratio_threshold: float,
) -> tuple[list[ExtractionResult], FilterStats]:
    """Run the full extraction pipeline on all candidates.

    For each candidate diagram: check gFlow, extract circuit,
    optimize, and filter by CNOT ratio.
    """
    raise NotImplementedError
