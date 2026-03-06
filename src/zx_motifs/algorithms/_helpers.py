"""Shared helper functions used across multiple algorithm families."""
from qiskit import QuantumCircuit


def decompose_toffoli(qc: QuantumCircuit, c0: int, c1: int, t: int) -> None:
    """Decompose Toffoli (CCX) into Clifford+T gates for QASM2 compatibility."""
    qc.h(t)
    qc.cx(c1, t)
    qc.tdg(t)
    qc.cx(c0, t)
    qc.t(t)
    qc.cx(c1, t)
    qc.tdg(t)
    qc.cx(c0, t)
    qc.t(c1)
    qc.t(t)
    qc.h(t)
    qc.cx(c0, c1)
    qc.t(c0)
    qc.tdg(c1)
    qc.cx(c0, c1)


def bell_pair(qc: QuantumCircuit, q0: int, q1: int) -> None:
    """Create a Bell pair on qubits (q0, q1)."""
    qc.h(q0)
    qc.cx(q0, q1)
