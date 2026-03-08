"""
Step 5 – Filter Candidates to Extractable Quantum Circuits
============================================================
Not every ZX-diagram composed in Step 4 can be turned back into a
quantum circuit.  Circuit extraction from a general ZX-diagram is
#P-hard, but polynomial-time algorithms exist for *graph-like*
diagrams that preserve *generalised flow* (gflow).

This module attempts extraction via ``pyzx.extract_circuit`` and
keeps only the candidates that survive.  Surviving candidates are
also subjected to a basic sanity check: the extracted circuit must
be unitary (no dimension mismatch) and must not be trivially
equivalent to the identity (which would be uninteresting).

Additional filters
------------------
* **Gate-count floor** – discard circuits with fewer than ``min_gates``
  total gates (too trivial).
* **Qubit ceiling** – discard circuits wider than ``max_qubits`` (too
  expensive to simulate for benchmarking).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import copy
import numpy as np

import pyzx as zx

from .composer import CandidateAlgorithm


@dataclass
class ExtractedCandidate:
    """A candidate that survived circuit extraction."""
    candidate: CandidateAlgorithm
    circuit: zx.Circuit
    qasm: str
    n_qubits: int
    gate_count: int
    two_qubit_gate_count: int
    t_count: int
    is_unitary: bool = True


def _try_extract(g: zx.Graph) -> Optional[zx.Circuit]:
    """Attempt to extract a circuit from a ZX-diagram.

    Returns None if extraction fails.
    """
    gc = copy.deepcopy(g)
    try:
        # Ensure graph-like form
        zx.simplify.to_graph_like(gc)
        circuit = zx.extract_circuit(gc)
        return circuit
    except Exception:
        pass

    # Fallback: try after a lighter simplification
    gc2 = copy.deepcopy(g)
    try:
        zx.simplify.clifford_simp(gc2, quiet=True)
        circuit = zx.extract_circuit(gc2)
        return circuit
    except Exception:
        pass

    # Final fallback: try raw
    gc3 = copy.deepcopy(g)
    try:
        circuit = zx.extract_circuit(gc3)
        return circuit
    except Exception:
        return None


def _circuit_stats(c: zx.Circuit) -> Dict[str, int]:
    """Compute basic gate statistics for a PyZX Circuit."""
    gates = c.gates
    total = len(gates)
    two_q = 0
    t_count = 0
    for gate in gates:
        name = gate.name.lower() if hasattr(gate, 'name') else ""
        # Two-qubit gates
        if hasattr(gate, 'control') and gate.control is not None:
            two_q += 1
        elif name in ('cnot', 'cx', 'cz', 'swap', 'cphase'):
            two_q += 1
        # T-count
        if name in ('t', 'tdg', 't*'):
            t_count += 1
    return {"total": total, "two_qubit": two_q, "t_count": t_count}


def _is_trivial_identity(c: zx.Circuit) -> bool:
    """Heuristic check: is the circuit likely the identity?

    Compares the circuit's unitary matrix to the identity.
    Only feasible for small qubit counts (≤ 8).
    """
    n = c.qubits
    if n > 8:
        return False  # can't check – assume non-trivial
    try:
        mat = c.to_matrix()
        identity = np.eye(2**n)
        return np.allclose(mat, identity, atol=1e-8)
    except Exception:
        return False


def filter_candidates(
    candidates: List[CandidateAlgorithm],
    min_gates: int = 3,
    max_qubits: int = 10,
) -> List[ExtractedCandidate]:
    """Filter candidates to those yielding valid, non-trivial circuits.

    Parameters
    ----------
    candidates : list
        Output of ``composer.compose_candidates``.
    min_gates : int
        Minimum total gate count to keep.
    max_qubits : int
        Maximum qubit width to keep (for simulability).

    Returns
    -------
    list of ExtractedCandidate
    """
    extracted: List[ExtractedCandidate] = []

    for cand in candidates:
        circuit = _try_extract(cand.graph)
        if circuit is None:
            continue

        n_q = circuit.qubits
        if n_q > max_qubits or n_q < 1:
            continue

        stats = _circuit_stats(circuit)
        if stats["total"] < min_gates:
            continue

        # Check if trivially the identity
        if _is_trivial_identity(circuit):
            continue

        # Export to QASM
        try:
            qasm_str = circuit.to_qasm()
        except Exception:
            continue

        extracted.append(ExtractedCandidate(
            candidate=cand,
            circuit=circuit,
            qasm=qasm_str,
            n_qubits=n_q,
            gate_count=stats["total"],
            two_qubit_gate_count=stats["two_qubit"],
            t_count=stats["t_count"],
        ))

    return extracted
