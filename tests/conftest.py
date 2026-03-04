"""Shared test fixtures for ZX motif pipeline tests."""
import numpy as np
import pyzx as zx
import pytest
from fractions import Fraction
from qiskit import QuantumCircuit


@pytest.fixture
def bell_circuit() -> QuantumCircuit:
    """Simple Bell state circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


@pytest.fixture
def ghz3_circuit() -> QuantumCircuit:
    """3-qubit GHZ state."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc


@pytest.fixture
def t_gate_circuit() -> QuantumCircuit:
    """Circuit with T gates for phase classification testing."""
    qc = QuantumCircuit(2)
    qc.t(0)
    qc.h(0)
    qc.cx(0, 1)
    qc.tdg(1)
    return qc


@pytest.fixture
def small_qft_circuit() -> QuantumCircuit:
    """2-qubit QFT."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cp(np.pi / 2, 1, 0)
    qc.h(1)
    qc.swap(0, 1)
    return qc


@pytest.fixture
def bell_zx_graph(bell_circuit):
    """Bell circuit as a PyZX graph."""
    from zx_motifs.pipeline.converter import qiskit_to_zx

    circ = qiskit_to_zx(bell_circuit)
    return circ.to_graph()
