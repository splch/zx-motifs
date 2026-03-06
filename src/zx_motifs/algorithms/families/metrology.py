"""Metrology family: ghz_metrology, quantum_fisher_info."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm


@register_algorithm(
    "ghz_metrology", "metrology", (2, 8),
    tags=["sensing", "heisenberg_limit"],
)
def make_ghz_metrology(n_qubits=4, phi=0.3, **kwargs) -> QuantumCircuit:
    """GHZ-state Ramsey interferometry for Heisenberg-limited sensing.

    Steps:
      (a) Prepare GHZ state -- H on q0, CX chain q0->q1->...->qN-1.
      (b) Phase accumulation -- RZ(phi) on every qubit.
      (c) Inverse GHZ preparation -- reverse CX chain, H on q0.

    Tags: sensing, heisenberg_limit
    """
    phi = kwargs.get("phi", phi)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)

    # (a) Prepare GHZ
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)

    # (b) Phase accumulation
    for i in range(n):
        qc.rz(phi, i)

    # (c) Inverse GHZ preparation
    for i in range(n - 2, -1, -1):
        qc.cx(i, i + 1)
    qc.h(0)

    return qc


@register_algorithm(
    "quantum_fisher_info", "metrology", (2, 8),
    tags=["sensing", "parameter_estimation"],
)
def make_quantum_fisher_info(n_qubits=4, phi=0.5, layers=2,
                             **kwargs) -> QuantumCircuit:
    """Parameterised probe state for quantum Fisher information estimation.

    Alternating layers of:
      - RY(phi) on all qubits (parameter encoding)
      - CX entangling ladder (q0->q1, q1->q2, ...)

    Tags: sensing, parameter_estimation
    """
    phi = kwargs.get("phi", phi)
    layers = kwargs.get("layers", layers)
    n = max(2, n_qubits)
    qc = QuantumCircuit(n)

    for _ in range(layers):
        # Parameter-encoding layer
        for i in range(n):
            qc.ry(phi, i)
        # Entangling ladder
        for i in range(n - 1):
            qc.cx(i, i + 1)

    return qc
