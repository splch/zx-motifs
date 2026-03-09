"""
Flow detection, circuit extraction, and filtering.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pyzx as zx

logger = logging.getLogger(__name__)


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
    from pyzx.gflow import gflow

    start = time.perf_counter()
    try:
        result = gflow(graph)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if result is not None:
            return FlowResult(FlowType.GFLOW, exists=True, flow_data=result, check_time_ms=elapsed_ms)
        return FlowResult(FlowType.NONE, exists=False, check_time_ms=elapsed_ms)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return FlowResult(FlowType.NONE, exists=False, check_time_ms=elapsed_ms)


def check_flow(graph: Any) -> FlowResult:
    """Check for gFlow on the diagram."""
    return check_gflow(graph)


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
    flow_result = check_flow(graph)
    if not flow_result.exists:
        return ExtractionResult(success=False, flow_used=FlowType.NONE, error="No flow found")

    try:
        circuit = zx.extract.extract_circuit(graph.copy(), quiet=True)
        stats = circuit.stats_dict(depth=True)
        qasm = circuit.to_qasm()
        return ExtractionResult(
            success=True,
            circuit=circuit,
            qasm=qasm,
            flow_used=flow_result.flow_type,
            gate_count=stats.get("gates", 0),
            two_qubit_count=stats.get("twoqubit", 0),
            t_count=stats.get("tcount", 0),
            depth=stats.get("depth", 0),
        )
    except Exception as e:
        return ExtractionResult(success=False, flow_used=flow_result.flow_type, error=str(e))


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
    candidate_dir = Path(candidate_dir)
    survivors: list[ExtractionResult] = []
    stats = FilterStats()

    json_files = sorted(candidate_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != "extraction_summary.json"]
    stats.total_candidates = len(json_files)

    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text())
            graph_data = data["graph"]
            if isinstance(graph_data, dict):
                graph_json = json.dumps(graph_data)
            else:
                graph_json = graph_data
            graph = zx.Graph.from_json(graph_json)

            # Restore inputs/outputs (from_json does not restore them)
            if isinstance(graph_data, dict):
                inputs = graph_data.get("inputs", [])
                outputs = graph_data.get("outputs", [])
                if inputs:
                    graph.set_inputs(tuple(inputs))
                if outputs:
                    graph.set_outputs(tuple(outputs))

            # Check flow
            flow_result = check_flow(graph)
            if not flow_result.exists:
                continue
            stats.passed_flow_check += 1

            # Extract circuit
            result = extract_circuit_pyzx(graph)
            if not result.success:
                continue
            stats.passed_extraction += 1

            # Post-optimize
            if post_optimize:
                try:
                    result.circuit = zx.optimize.full_optimize(result.circuit, quiet=True)
                    new_stats = result.circuit.stats_dict(depth=True)
                    result.gate_count = new_stats.get("gates", 0)
                    result.two_qubit_count = new_stats.get("twoqubit", 0)
                    result.t_count = new_stats.get("tcount", 0)
                    result.depth = new_stats.get("depth", 0)
                    result.qasm = result.circuit.to_qasm()
                except Exception:
                    logger.warning("Post-optimization failed for %s", json_path.stem, exc_info=True)

            # Filter by CNOT ratio
            n_qubits = max(len(graph.inputs()), 1)
            cnot_ratio = result.two_qubit_count / n_qubits
            if cnot_ratio > cnot_ratio_threshold:
                continue
            stats.passed_size_filter += 1

            survivors.append(result)
            stats.final_survivors += 1

        except Exception:
            logger.warning("Failed to process %s", json_path.name, exc_info=True)

    return survivors, stats
