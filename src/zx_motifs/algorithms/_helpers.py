"""Shared helper functions used across multiple algorithm families."""
import functools

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


@functools.lru_cache(maxsize=1)
def _basic_gate_pass_manager():
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    return generate_preset_pass_manager(
        optimization_level=0,
        basis_gates=["cx", "h", "t", "tdg", "s", "sdg", "x", "y", "z",
                      "rz", "ry", "rx", "id"],
    )


def decompose_to_basic_gates(qc: QuantumCircuit) -> QuantumCircuit:
    """Decompose a circuit to basic gates that PyZX can parse via QASM2.

    For large MCX gates (5+ controls), Qiskit emits custom QASM gate definitions
    that PyZX cannot parse. This ensures full decomposition to basic gates.
    """
    return _basic_gate_pass_manager().run(qc)
