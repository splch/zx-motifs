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
    candidate_id: str = ""


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


def _filter_single_candidate(
    json_path_str: str,
    post_optimize: bool,
    cnot_ratio_threshold: float,
) -> tuple[dict | None, dict]:
    """Worker: process one candidate through the extraction filter.

    Returns (result_dict | None, stats_increments).
    Uses plain dicts to avoid pickling PyZX Circuit objects across processes.
    """
    stats_inc = {
        "passed_flow_check": 0,
        "passed_extraction": 0,
        "passed_size_filter": 0,
        "final_survivors": 0,
    }
    json_path = Path(json_path_str)

    try:
        data = json.loads(json_path.read_text())
        graph_data = data["graph"]
        if isinstance(graph_data, dict):
            graph_json = json.dumps(graph_data)
        else:
            graph_json = graph_data
        from src.zx import _sanitize_phase_tildes

        graph = zx.Graph.from_json(_sanitize_phase_tildes(graph_json))

        if isinstance(graph_data, dict):
            inputs = graph_data.get("inputs", [])
            outputs = graph_data.get("outputs", [])
            if inputs:
                graph.set_inputs(tuple(inputs))
            if outputs:
                graph.set_outputs(tuple(outputs))

        flow_result = check_flow(graph)
        if not flow_result.exists:
            return (None, stats_inc)
        stats_inc["passed_flow_check"] = 1

        result = extract_circuit_pyzx(graph)
        if not result.success:
            return (None, stats_inc)
        stats_inc["passed_extraction"] = 1

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
                pass

        n_qubits = max(len(graph.inputs()), 1)
        cnot_ratio = result.two_qubit_count / n_qubits
        if cnot_ratio > cnot_ratio_threshold:
            return (None, stats_inc)
        stats_inc["passed_size_filter"] = 1
        stats_inc["final_survivors"] = 1

        result_dict = {
            "candidate_id": json_path.stem,
            "qasm": result.qasm,
            "flow_used": result.flow_used.value,
            "gate_count": result.gate_count,
            "two_qubit_count": result.two_qubit_count,
            "t_count": result.t_count,
            "depth": result.depth,
        }
        return (result_dict, stats_inc)
    except Exception:
        return (None, stats_inc)


def run_extraction_filter(
    candidate_dir: str | Path,
    post_optimize: bool,
    cnot_ratio_threshold: float,
    workers: int | None = None,
) -> tuple[list[ExtractionResult], FilterStats]:
    """Run the full extraction pipeline on all candidates.

    For each candidate diagram: check gFlow, extract circuit,
    optimize, and filter by CNOT ratio.
    """
    from src.parallel import parallel_map

    candidate_dir = Path(candidate_dir)
    json_files = sorted(candidate_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != "extraction_summary.json"]

    stats = FilterStats(total_candidates=len(json_files))

    items = [(str(f), post_optimize, cnot_ratio_threshold) for f in json_files]
    results = parallel_map(_filter_single_candidate, items, workers, desc="extraction filter")

    survivors: list[ExtractionResult] = []
    for result_dict, stats_inc in results:
        stats.passed_flow_check += stats_inc["passed_flow_check"]
        stats.passed_extraction += stats_inc["passed_extraction"]
        stats.passed_size_filter += stats_inc["passed_size_filter"]
        stats.final_survivors += stats_inc["final_survivors"]

        if result_dict is not None:
            survivors.append(ExtractionResult(
                success=True,
                qasm=result_dict["qasm"],
                flow_used=FlowType(result_dict["flow_used"]),
                gate_count=result_dict["gate_count"],
                two_qubit_count=result_dict["two_qubit_count"],
                t_count=result_dict["t_count"],
                depth=result_dict["depth"],
                candidate_id=result_dict["candidate_id"],
            ))

    return survivors, stats
