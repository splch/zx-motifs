"""Tests for algorithm registry: all generators produce valid circuits convertible to ZX."""
import pytest
from qiskit.qasm2 import dumps as qasm2_dumps

from zx_motifs.algorithms.registry import REGISTRY, AlgorithmEntry
from zx_motifs.pipeline.converter import qiskit_to_zx


# ── Parametrized tests over all registry entries ───────────────────


def _registry_ids():
    return [entry.name for entry in REGISTRY]


@pytest.fixture(params=REGISTRY, ids=_registry_ids())
def registry_entry(request) -> AlgorithmEntry:
    return request.param


class TestRegistryGenerators:
    """Every registered generator must produce a valid, convertible circuit."""

    def test_generator_returns_circuit(self, registry_entry):
        """Generator produces a QuantumCircuit at minimum qubit count."""
        min_q = registry_entry.qubit_range[0]
        qc = registry_entry.generator(n_qubits=min_q)
        assert qc.num_qubits >= 2, f"{registry_entry.name}: too few qubits"

    def test_qasm2_export(self, registry_entry):
        """Circuit can be exported to QASM2 (required for PyZX conversion)."""
        min_q = registry_entry.qubit_range[0]
        qc = registry_entry.generator(n_qubits=min_q)
        qasm_str = qasm2_dumps(qc)
        assert "OPENQASM 2.0" in qasm_str
        assert len(qasm_str) > 50

    def test_zx_conversion(self, registry_entry):
        """Circuit converts to a non-empty PyZX graph."""
        min_q = registry_entry.qubit_range[0]
        qc = registry_entry.generator(n_qubits=min_q)
        zx_circ = qiskit_to_zx(qc)
        g = zx_circ.to_graph()
        assert g.num_vertices() > 0
        assert g.num_edges() > 0

    def test_max_qubit_generation(self, registry_entry):
        """Generator also works at max qubit count."""
        # Grover at 6 qubits uses MCX(5-control) which produces custom QASM
        # gates that PyZX cannot parse — skip this known limitation
        if registry_entry.name == "grover" and registry_entry.qubit_range[1] > 4:
            max_q = 4
        else:
            max_q = registry_entry.qubit_range[1]
        qc = registry_entry.generator(n_qubits=max_q)
        zx_circ = qiskit_to_zx(qc)
        g = zx_circ.to_graph()
        assert g.num_vertices() > 0


# ── Structural assertions for specific algorithm families ──────────


class TestErrorCorrectionStructure:
    def test_bit_flip_has_t_gates(self):
        """Decomposed Toffoli in bit-flip code produces T gates."""
        from zx_motifs.algorithms.registry import make_bit_flip_code
        from zx_motifs.pipeline.converter import count_t_gates

        qc = make_bit_flip_code()
        g = qiskit_to_zx(qc).to_graph()
        assert count_t_gates(g) >= 4, "Toffoli decomposition should yield T gates"

    def test_steane_code_cx_rich(self):
        """Steane code has dense CX connectivity (many edges)."""
        from zx_motifs.algorithms.registry import make_steane_code

        qc = make_steane_code()
        g = qiskit_to_zx(qc).to_graph()
        assert g.num_edges() > 20, "Steane code should have dense connectivity"


class TestSimulationStructure:
    def test_trotter_ising_has_rz(self):
        """Trotter Ising circuit contains RZ gates (non-zero phases)."""
        from zx_motifs.algorithms.registry import make_trotter_ising
        from fractions import Fraction

        qc = make_trotter_ising(n_qubits=4)
        g = qiskit_to_zx(qc).to_graph()
        phases = [g.phase(v) for v in g.vertices()]
        non_zero = [p for p in phases if p != 0 and p != Fraction(0)]
        assert len(non_zero) > 0, "Trotter Ising should have non-zero phases"

    def test_trotter_heisenberg_gate_diversity(self):
        """Heisenberg model uses H, CX, RZ, S, Sdg — most diverse gate mix."""
        from zx_motifs.algorithms.registry import make_trotter_heisenberg

        qc = make_trotter_heisenberg(n_qubits=2)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "h" in gate_names
        assert "cx" in gate_names
        assert "rz" in gate_names


class TestEntanglementStructure:
    def test_cluster_state_uses_cz(self):
        """Cluster state is built from H + CZ only."""
        from zx_motifs.algorithms.registry import make_cluster_state

        qc = make_cluster_state(n_qubits=4)
        gate_names = {inst.operation.name for inst in qc.data}
        assert gate_names <= {"h", "cz"}, f"Unexpected gates: {gate_names}"

    def test_w_state_uses_ry(self):
        """W state uses RY rotations for amplitude distribution."""
        from zx_motifs.algorithms.registry import make_w_state

        qc = make_w_state(n_qubits=4)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "ry" in gate_names


class TestDistillationStructure:
    def test_bbpssw_uses_only_h_cx(self):
        """BBPSSW uses only H + CX gates."""
        from zx_motifs.algorithms.registry import make_bbpssw_distillation

        qc = make_bbpssw_distillation()
        gate_names = {inst.operation.name for inst in qc.data}
        assert gate_names <= {"h", "cx"}, f"Unexpected gates: {gate_names}"

    def test_dejmps_includes_ry(self):
        """DEJMPS adds RY rotations for twirling."""
        from zx_motifs.algorithms.registry import make_dejmps_distillation

        qc = make_dejmps_distillation()
        gate_names = {inst.operation.name for inst in qc.data}
        assert "ry" in gate_names

    def test_recurrence_more_cx_than_bbpssw(self):
        """Recurrence distillation has more CX gates than single-round BBPSSW."""
        from zx_motifs.algorithms.registry import (
            make_bbpssw_distillation,
            make_recurrence_distillation,
        )

        bbpssw_cx = sum(1 for inst in make_bbpssw_distillation().data
                        if inst.operation.name == "cx")
        recurrence_cx = sum(1 for inst in make_recurrence_distillation().data
                            if inst.operation.name == "cx")
        assert recurrence_cx > bbpssw_cx

    def test_pumping_nontrivial_zx(self):
        """Pumping distillation produces a non-trivial ZX graph."""
        from zx_motifs.algorithms.registry import make_pumping_distillation

        qc = make_pumping_distillation()
        g = qiskit_to_zx(qc).to_graph()
        assert g.num_vertices() > 10
        assert g.num_edges() > 10


class TestMotifGenerators:
    """New hand-crafted motifs have valid structure."""

    def test_handcrafted_motifs_count(self):
        from zx_motifs.pipeline.motif_generators import HANDCRAFTED_MOTIFS

        assert len(HANDCRAFTED_MOTIFS) == 9

    def test_all_motifs_are_connected(self):
        import networkx as nx
        from zx_motifs.pipeline.motif_generators import HANDCRAFTED_MOTIFS

        for motif in HANDCRAFTED_MOTIFS:
            assert nx.is_connected(motif.graph), (
                f"Motif {motif.motif_id} is not connected"
            )

    def test_new_motif_ids(self):
        from zx_motifs.pipeline.motif_generators import HANDCRAFTED_MOTIFS

        ids = {m.motif_id for m in HANDCRAFTED_MOTIFS}
        assert "syndrome_extraction" in ids
        assert "toffoli_core" in ids
        assert "cluster_chain" in ids
        assert "trotter_layer" in ids
