"""TDA family: betti_number."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm


@register_algorithm(
    "betti_number", "tda", (3, 8),
    tags=["topology", "homology"],
)
def make_betti_number(n_qubits=4, **kwargs) -> QuantumCircuit:
    """Betti number estimation circuit for topological data analysis.

    Estimates the k-th Betti number of a simplicial complex by
    preparing a uniform superposition over simplices and applying
    a phase oracle that encodes the combinatorial Laplacian.

    Args:
        n_qubits: Number of qubits / vertices in the complex (minimum 3, default 4).

    Tags: topology, homology
    """
    n = max(3, n_qubits)
    qc = QuantumCircuit(n)

    # Step 1: Uniform superposition over all simplices
    for i in range(n):
        qc.h(i)

    # Step 2: Phase oracle encoding simplicial complex adjacency
    # 1-skeleton: CZ gates for edges of a cycle graph
    for i in range(n):
        qc.cz(i, (i + 1) % n)

    # 2-simplices (triangular faces): CP gates encoding higher-order
    # adjacency structure
    for i in range(n):
        qc.cp(np.pi / 3, i, (i + 2) % n)

    # Step 3: Inverse-QFT-like readout for spectral decomposition
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    for i in range(n):
        for j in range(i):
            qc.cp(-np.pi / (2 ** (i - j)), j, i)
        qc.h(i)

    return qc
