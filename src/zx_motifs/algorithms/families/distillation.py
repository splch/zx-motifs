"""Distillation family: bbpssw_distillation, dejmps_distillation,
recurrence_distillation, pumping_distillation."""
import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.algorithms._registry_core import register_algorithm
from zx_motifs.algorithms._helpers import bell_pair


@register_algorithm(
    "bbpssw_distillation", "distillation", (4, 4),
    tags=["distillation", "bell_pair", "bilateral_cnot"],
)
def make_bbpssw_distillation(n_qubits=4, **kwargs) -> QuantumCircuit:
    """BBPSSW entanglement distillation (Bennett et al. 1996).

    Foundational bilateral CNOT protocol. Two noisy Bell pairs are combined
    via bilateral CNOTs; measurement of the sacrificial pair heralds success.
    Qubits 0,1: pair to keep; 2,3: pair to sacrifice.
    """
    qc = QuantumCircuit(4)
    # Two Bell pairs
    bell_pair(qc, 0, 1)
    bell_pair(qc, 2, 3)
    # Bilateral CNOTs
    qc.cx(0, 2)
    qc.cx(1, 3)
    return qc


@register_algorithm(
    "dejmps_distillation", "distillation", (4, 4),
    tags=["distillation", "bell_pair", "bilateral_cnot", "twirling"],
)
def make_dejmps_distillation(n_qubits=4, **kwargs) -> QuantumCircuit:
    """DEJMPS entanglement distillation (Deutsch et al. 1996).

    Adds bilateral Rx rotations before CNOTs to handle asymmetric noise.
    Qubits 0,1: pair to keep; 2,3: pair to sacrifice.
    """
    qc = QuantumCircuit(4)
    # Two Bell pairs
    bell_pair(qc, 0, 1)
    bell_pair(qc, 2, 3)
    # Bilateral rotations: Rx(pi/2) for Alice, Rx(-pi/2) for Bob
    qc.rx(np.pi / 2, 0)
    qc.rx(-np.pi / 2, 1)
    qc.rx(np.pi / 2, 2)
    qc.rx(-np.pi / 2, 3)
    # Bilateral CNOTs
    qc.cx(0, 2)
    qc.cx(1, 3)
    return qc


@register_algorithm(
    "recurrence_distillation", "distillation", (8, 8),
    tags=["distillation", "bell_pair", "bilateral_cnot", "multi_round"],
)
def make_recurrence_distillation(n_qubits=8, **kwargs) -> QuantumCircuit:
    """Two-round recurrence distillation.

    Cascades two BBPSSW rounds. Round 1 distills (0,1) from (0,1)+(2,3)
    and (4,5) from (4,5)+(6,7). Round 2 distills (0,1) from (0,1)+(4,5).
    """
    qc = QuantumCircuit(8)
    # Four Bell pairs
    bell_pair(qc, 0, 1)
    bell_pair(qc, 2, 3)
    bell_pair(qc, 4, 5)
    bell_pair(qc, 6, 7)
    # Round 1: bilateral CNOTs
    qc.cx(0, 2)
    qc.cx(1, 3)
    qc.cx(4, 6)
    qc.cx(5, 7)
    # Round 2: bilateral CNOTs on surviving pairs
    qc.cx(0, 4)
    qc.cx(1, 5)
    return qc


@register_algorithm(
    "pumping_distillation", "distillation", (6, 6),
    tags=["distillation", "bell_pair", "bilateral_cnot", "pumping"],
)
def make_pumping_distillation(n_qubits=6, **kwargs) -> QuantumCircuit:
    """Pumping entanglement distillation.

    One target pair (0,1) is repeatedly purified by sacrificing fresh pairs.
    Sacrificial pairs: (2,3) then (4,5).
    """
    qc = QuantumCircuit(6)
    # Target Bell pair
    bell_pair(qc, 0, 1)
    # First sacrificial pair
    bell_pair(qc, 2, 3)
    qc.cx(0, 2)
    qc.cx(1, 3)
    # Second sacrificial pair
    bell_pair(qc, 4, 5)
    qc.cx(0, 4)
    qc.cx(1, 5)
    return qc
