"""
Phase polynomial optimization for CNOT count reduction.

After ZXBox composition and before circuit extraction, re-synthesizes the
phase polynomial to minimize CNOT count. Uses PyZX's phase_block_optimize
or basic CNOT cancellation.
"""
from __future__ import annotations

import copy
import logging

import pyzx as zx
from pyzx.graph import Graph

from .converter import qiskit_to_zx

logger = logging.getLogger(__name__)


def optimize_phase_polynomial(qc) -> "QuantumCircuit":
    """Optimize a circuit's phase polynomial to reduce CNOT count.

    Converts to PyZX, applies phase_block_optimize (TODD algorithm),
    and extracts back to Qiskit.

    Falls back to basic_optimization if phase_block_optimize fails
    (e.g., for non-Clifford+T circuits).
    """
    from qiskit import QuantumCircuit

    try:
        zx_circ = qiskit_to_zx(qc)
        optimized = zx.phase_block_optimize(zx_circ, quiet=True)
        qasm = optimized.to_qasm()
        return QuantumCircuit.from_qasm_str(qasm)
    except (TypeError, ValueError, KeyError):
        logger.debug("phase_block_optimize failed, trying basic_optimization")

    try:
        zx_circ = qiskit_to_zx(qc)
        optimized = zx.optimize.basic_optimization(zx_circ)
        qasm = optimized.to_qasm()
        return QuantumCircuit.from_qasm_str(qasm)
    except Exception:
        logger.debug("basic_optimization also failed, returning original")
        return qc


def optimize_zx_graph(g: Graph) -> Graph:
    """Apply ZX simplification passes to reduce graph complexity.

    Applies spider fusion, interior Clifford simplification, and
    phase gadget merging in sequence.
    """
    g_opt = copy.deepcopy(g)
    zx.simplify.spider_simp(g_opt)
    zx.simplify.interior_clifford_simp(g_opt)
    try:
        zx.simplify.gadget_simp(g_opt)
    except Exception:
        pass  # gadget_simp may fail on some graph structures
    return g_opt


def count_cnots_pyzx(zx_circ) -> int:
    """Count CNOT gates in a PyZX circuit."""
    return sum(1 for g in zx_circ.gates if g.name == "CNOT")


def count_cnots_qiskit(qc) -> int:
    """Count CX gates in a Qiskit circuit."""
    return sum(1 for inst in qc.data if inst.operation.name in ("cx", "cnot"))


def optimization_report(original_qc, optimized_qc) -> dict:
    """Compare original and optimized circuits."""
    orig_cx = count_cnots_qiskit(original_qc)
    opt_cx = count_cnots_qiskit(optimized_qc)
    return {
        "original_cx": orig_cx,
        "optimized_cx": opt_cx,
        "reduction": orig_cx - opt_cx,
        "reduction_pct": (orig_cx - opt_cx) / orig_cx * 100 if orig_cx > 0 else 0.0,
        "original_depth": original_qc.depth(),
        "optimized_depth": optimized_qc.depth(),
    }
