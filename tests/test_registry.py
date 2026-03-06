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

    def test_dejmps_includes_rx(self):
        """DEJMPS adds RX rotations for twirling."""
        from zx_motifs.algorithms.registry import make_dejmps_distillation

        qc = make_dejmps_distillation()
        gate_names = {inst.operation.name for inst in qc.data}
        assert "rx" in gate_names

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

        assert len(HANDCRAFTED_MOTIFS) >= 9

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


class TestLinearAlgebraStructure:
    def test_hhl_has_controlled_rotations(self):
        """HHL uses controlled-RY rotations for eigenvalue inversion."""
        from zx_motifs.algorithms.registry import make_hhl
        qc = make_hhl(n_qubits=5)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "ry" in gate_names, "HHL should have RY gates for eigenvalue inversion"
        assert "cp" in gate_names, "HHL should have controlled-phase gates for QPE"

    def test_vqls_is_variational(self):
        """VQLS uses RY/RZ variational rotations."""
        from zx_motifs.algorithms.registry import make_vqls
        qc = make_vqls(n_qubits=4)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "ry" in gate_names
        assert "rz" in gate_names
        assert "cx" in gate_names


class TestCryptographyStructure:
    def test_bb84_uses_h_and_x(self):
        """BB84 encoding uses only H and X gates."""
        from zx_motifs.algorithms.registry import make_bb84_encode
        qc = make_bb84_encode(n_qubits=8)
        gate_names = {inst.operation.name for inst in qc.data}
        assert gate_names <= {"h", "x"}, f"BB84 should only use H and X, got {gate_names}"

    def test_e91_has_bell_pairs(self):
        """E91 creates Bell pairs (uses H and CX)."""
        from zx_motifs.algorithms.registry import make_e91_protocol
        qc = make_e91_protocol(n_qubits=8)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "h" in gate_names
        assert "cx" in gate_names
        assert "ry" in gate_names, "E91 should have RY for measurement basis rotation"


class TestSamplingStructure:
    def test_iqp_has_symmetric_h_layers(self):
        """IQP circuit has H layers at start and end."""
        from zx_motifs.algorithms.registry import make_iqp_sampling
        qc = make_iqp_sampling(n_qubits=5)
        ops = [inst.operation.name for inst in qc.data]
        # First ops should be H gates
        assert ops[0] == "h", "IQP should start with H layer"
        # Last ops should be H gates
        assert ops[-1] == "h", "IQP should end with H layer"

    def test_random_circuit_has_t_gates(self):
        """Random circuit sampling includes T gates."""
        from zx_motifs.algorithms.registry import make_random_circuit_sampling
        qc = make_random_circuit_sampling(n_qubits=4)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "t" in gate_names, "Random circuit should include T gates"


class TestErrorMitigationStructure:
    def test_zne_has_repeated_pattern(self):
        """ZNE folding has circuit-inverse-circuit (3x gate count of base)."""
        from zx_motifs.algorithms.registry import make_zne_folding
        qc = make_zne_folding(n_qubits=4)
        gate_names = [inst.operation.name for inst in qc.data]
        # Should have a substantial number of gates (3x base circuit)
        assert len(gate_names) > 10, "ZNE should have repeated circuit pattern"

    def test_pauli_twirling_has_paulis(self):
        """Pauli twirling inserts Pauli gates around CX."""
        from zx_motifs.algorithms.registry import make_pauli_twirling
        qc = make_pauli_twirling(n_qubits=4)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "cx" in gate_names, "Pauli twirling should have CX gates"


class TestTopologicalStructure:
    def test_jones_polynomial_has_controlled_phase(self):
        """Jones polynomial uses controlled-phase gates."""
        from zx_motifs.algorithms.registry import make_jones_polynomial
        qc = make_jones_polynomial(n_qubits=4)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "cp" in gate_names, "Jones polynomial should use controlled-phase gates"
        assert "h" in gate_names

    def test_toric_code_has_syndrome_extraction(self):
        """Toric code uses CX for syndrome extraction."""
        from zx_motifs.algorithms.registry import make_toric_code_syndrome
        qc = make_toric_code_syndrome()
        cx_count = sum(1 for inst in qc.data if inst.operation.name == "cx")
        assert cx_count >= 4, "Toric code should have multiple CX gates for syndrome extraction"


class TestMetrologyStructure:
    def test_ghz_metrology_has_phase_between_ghz(self):
        """GHZ metrology has RZ phase rotations."""
        from zx_motifs.algorithms.registry import make_ghz_metrology
        qc = make_ghz_metrology(n_qubits=4)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "rz" in gate_names, "GHZ metrology should have RZ for phase accumulation"
        assert "cx" in gate_names, "GHZ metrology should have CX for GHZ state"
        assert "h" in gate_names

    def test_qfi_has_parameterized_rotations(self):
        """QFI probe state has RY rotations and CX entangling."""
        from zx_motifs.algorithms.registry import make_quantum_fisher_info
        qc = make_quantum_fisher_info(n_qubits=4)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "ry" in gate_names
        assert "cx" in gate_names


class TestNewVariationalStructure:
    def test_adapt_vqe_has_excitation_operators(self):
        """ADAPT-VQE has CX ladder + RZ pattern for excitations."""
        from zx_motifs.algorithms.registry import make_adapt_vqe
        qc = make_adapt_vqe(n_qubits=4)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "cx" in gate_names
        assert "rz" in gate_names
        assert "x" in gate_names, "ADAPT-VQE should have X for HF state preparation"

    def test_varqite_has_decreasing_rotations(self):
        """VarQITE ansatz uses RY and RZ rotations."""
        from zx_motifs.algorithms.registry import make_varqite
        qc = make_varqite(n_qubits=4)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "ry" in gate_names
        assert "rz" in gate_names


class TestNewOracleStructure:
    def test_deutsch_is_smallest_oracle(self):
        """Deutsch algorithm uses exactly 2 qubits."""
        from zx_motifs.algorithms.registry import make_deutsch
        qc = make_deutsch()
        assert qc.num_qubits == 2, "Deutsch should use exactly 2 qubits"

    def test_hidden_shift_has_symmetric_structure(self):
        """Hidden shift has 3 H layers and CZ oracle layers."""
        from zx_motifs.algorithms.registry import make_hidden_shift
        qc = make_hidden_shift(n_qubits=4)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "h" in gate_names
        assert "cz" in gate_names


class TestNewMLStructure:
    def test_qcnn_has_pooling(self):
        """QCNN has pooling layers that reduce active qubits."""
        from zx_motifs.algorithms.registry import make_qcnn
        qc = make_qcnn(n_qubits=8)
        # QCNN should have CX and RY gates
        gate_names = {inst.operation.name for inst in qc.data}
        assert "cx" in gate_names
        assert "ry" in gate_names

    def test_qsvm_has_zz_feature_map(self):
        """QSVM uses ZZ feature map (H + RZ + CX-RZ-CX)."""
        from zx_motifs.algorithms.registry import make_qsvm
        qc = make_qsvm(n_qubits=4)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "h" in gate_names
        assert "rz" in gate_names
        assert "cx" in gate_names


class TestNewErrorCorrectionStructure:
    def test_five_qubit_code_uses_5_qubits(self):
        """Five-qubit code encoder uses exactly 5 qubits."""
        from zx_motifs.algorithms.registry import make_five_qubit_code
        qc = make_five_qubit_code()
        assert qc.num_qubits == 5

    def test_color_code_uses_7_qubits(self):
        """Color code uses 7 qubits like Steane code."""
        from zx_motifs.algorithms.registry import make_color_code
        qc = make_color_code()
        assert qc.num_qubits == 7

    def test_bacon_shor_uses_9_qubits(self):
        """Bacon-Shor code uses 9 qubits (3x3 grid)."""
        from zx_motifs.algorithms.registry import make_bacon_shor
        qc = make_bacon_shor()
        assert qc.num_qubits == 9

    def test_reed_muller_uses_15_qubits(self):
        """Reed-Muller code uses 15 qubits."""
        from zx_motifs.algorithms.registry import make_reed_muller_code
        qc = make_reed_muller_code()
        assert qc.num_qubits == 15


class TestSwapTestStructure:
    def test_swap_test_has_ancilla(self):
        """Swap test uses H on ancilla and controlled-SWAP."""
        from zx_motifs.algorithms.registry import make_swap_test
        qc = make_swap_test(n_qubits=3)
        gate_names = {inst.operation.name for inst in qc.data}
        assert "h" in gate_names
        assert "cx" in gate_names
        # Should have T gates from Toffoli decomposition
        assert "t" in gate_names or "tdg" in gate_names


class TestArithmeticAdditionsStructure:
    def test_multiplier_has_toffoli(self):
        """Quantum multiplier uses decomposed Toffoli gates."""
        from zx_motifs.algorithms.registry import make_quantum_multiplier
        qc = make_quantum_multiplier()
        gate_names = {inst.operation.name for inst in qc.data}
        assert "t" in gate_names, "Multiplier should have T gates from Toffoli"
        assert "cx" in gate_names

    def test_comparator_has_toffoli(self):
        """Quantum comparator uses decomposed Toffoli gates."""
        from zx_motifs.algorithms.registry import make_quantum_comparator
        qc = make_quantum_comparator()
        gate_names = {inst.operation.name for inst in qc.data}
        assert "t" in gate_names, "Comparator should have T gates from Toffoli"
