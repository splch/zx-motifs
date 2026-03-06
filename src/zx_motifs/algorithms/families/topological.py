"""Topological family: jones_polynomial, toric_code_syndrome."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm


@register_algorithm(
    "jones_polynomial", "topological", (3, 8),
    tags=["knot_invariant", "topological"],
)
def make_jones_polynomial(n_qubits=4, **kwargs) -> QuantumCircuit:
    """Jones polynomial approximation circuit.

    Approximates the Jones polynomial evaluated at a root of unity
    using a chain of Hadamard and controlled-phase gates.

    Tags: knot_invariant, topological
    """
    n = max(3, n_qubits)
    qc = QuantumCircuit(n)
    angle = 2 * np.pi / 5  # 5th root of unity

    qc.h(0)
    # Braid-like pattern: sweep forward and backward creating crossings
    for sweep in range(2):
        if sweep % 2 == 0:
            # Forward sweep
            for i in range(n - 1):
                qc.cp(angle, i, i + 1)
                qc.h(i + 1)
        else:
            # Backward sweep
            for i in range(n - 2, -1, -1):
                qc.cp(angle, i + 1, i)
                qc.h(i)

    # Final controlled-phase layer for closure
    for i in range(n - 1):
        qc.cp(angle, i, i + 1)

    return qc


@register_algorithm(
    "toric_code_syndrome", "topological", (8, 8),
    tags=["topological_code", "syndrome_extraction"],
)
def make_toric_code_syndrome(n_qubits=8, **kwargs) -> QuantumCircuit:
    """Toric code syndrome extraction on a minimal 2x2 lattice.

    Layout (8 qubits):
      - Data qubits:   q0, q1, q2, q3  (edges of the torus)
      - Ancilla qubits: q4, q5  (X-stabiliser / vertex operators)
                         q6, q7  (Z-stabiliser / plaquette operators)

    Tags: topological_code, syndrome_extraction
    """
    n = 8  # fixed for minimal toric code
    qc = QuantumCircuit(n)

    data = [0, 1, 2, 3]
    x_ancillas = [4, 5]
    z_ancillas = [6, 7]

    # Vertex operator assignments
    vertex_data = [
        [0, 1],  # vertex-0 stabiliser acts on data q0, q1
        [2, 3],  # vertex-1 stabiliser acts on data q2, q3
    ]

    # Plaquette operator assignments
    plaquette_data = [
        [0, 2],  # plaquette-0 stabiliser acts on data q0, q2
        [1, 3],  # plaquette-1 stabiliser acts on data q1, q3
    ]

    # X-stabiliser extraction (vertex operators)
    for anc_idx, anc in enumerate(x_ancillas):
        qc.h(anc)
        for d in vertex_data[anc_idx]:
            qc.cx(anc, d)
        qc.h(anc)

    # Z-stabiliser extraction (plaquette operators)
    for anc_idx, anc in enumerate(z_ancillas):
        for d in plaquette_data[anc_idx]:
            qc.cx(d, anc)

    return qc
