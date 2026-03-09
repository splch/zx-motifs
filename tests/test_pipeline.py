"""
Tests for all pipeline modules.

One test class per module plus integration tests.

Run with: pytest tests/
"""

import pytest

from src.corpus import (
    AlgorithmEntry,
    AlgorithmRegistry,
    build_default_registry,
    build_ghz,
    build_grover,
    build_qft,
    circuit_to_qasm,
    transpile_to_gate_set,
)
from src.zx import (
    SimplificationResult,
    load_diagram,
    pyzx_circuit_to_graph,
    qasm_to_pyzx_circuit,
    save_diagram,
    simplify_graph,
)
from pyzx.utils import VertexType


# ── Corpus ──────────────────────────────────────────────────────────


class TestCorpus:
    """Tests for the algorithm registry, builders, and export."""

    def test_register_and_retrieve(self):
        """Registering an entry and retrieving it by key."""
        reg = AlgorithmRegistry()
        entry = AlgorithmEntry("test", "cat", build_qft, "desc")
        reg.register(entry)
        assert reg.get("test") is entry

    def test_duplicate_key_raises(self):
        """Registering two entries with the same key should raise ValueError."""
        reg = AlgorithmRegistry()
        entry = AlgorithmEntry("dup", "cat", build_qft)
        reg.register(entry)
        with pytest.raises(ValueError, match="Duplicate key"):
            reg.register(entry)

    def test_by_category_filters_correctly(self):
        """by_category should return only entries in that category."""
        reg = AlgorithmRegistry()
        reg.register(AlgorithmEntry("a", "foo", build_qft))
        reg.register(AlgorithmEntry("b", "bar", build_qft))
        reg.register(AlgorithmEntry("c", "foo", build_qft))
        assert [e.key for e in reg.by_category("foo")] == ["a", "c"]
        assert len(reg.by_category("bar")) == 1

    def test_default_registry_is_populated(self):
        """build_default_registry should return a populated registry."""
        reg = build_default_registry()
        keys = reg.all_keys()
        assert len(keys) == 29
        assert "qft" in keys
        assert "grover" in keys
        assert "steane_encoder" in keys
        assert "deutsch_jozsa" in keys
        assert "ghz" in keys
        assert "quantum_volume" in keys

    def test_grover_returns_unitary_circuit(self):
        """build_grover should return a circuit with no measurements."""
        qc = build_grover(3)
        assert qc.num_qubits == 3
        op_names = [inst.operation.name for inst in qc]
        assert "measure" not in op_names

    def test_qft_qubit_scaling(self):
        """build_qft(n) should produce a circuit on exactly n qubits."""
        for n in [2, 4, 8]:
            qc = build_qft(n)
            assert qc.num_qubits == n

    def test_all_builders_accept_min_qubits(self):
        """Every builder should work at its declared min_qubits."""
        reg = build_default_registry()
        for key in reg.all_keys():
            entry = reg.get(key)
            qc = entry.builder(entry.min_qubits)
            assert qc.num_qubits >= entry.min_qubits

    def test_circuit_to_qasm_produces_valid_openqasm(self):
        """Exported QASM should start with 'OPENQASM 2.0'."""
        qc = build_qft(3)
        qc_t = transpile_to_gate_set(qc, ["cx", "rz", "h"])
        qasm = circuit_to_qasm(qc_t)
        assert qasm.startswith("OPENQASM 2.0")

    def test_transpile_to_gate_set_restricts_gates(self):
        """After transpilation, only target gates should remain."""
        qc = build_qft(4)
        gate_set = ["cx", "rz", "h"]
        qc_t = transpile_to_gate_set(qc, gate_set)
        allowed = set(gate_set) | {"barrier", "measure"}
        ops = {inst.operation.name for inst in qc_t}
        assert ops <= allowed, f"Unexpected gates: {ops - allowed}"


# ── ZX ──────────────────────────────────────────────────────────────


class TestZX:
    """Tests for ZX conversion, simplification, and storage."""

    def test_qasm_roundtrip(self):
        """A simple circuit should survive QASM export → PyZX import."""
        qc = build_ghz(3)
        qc_t = transpile_to_gate_set(qc, ["cx", "rz", "h"])
        qasm = circuit_to_qasm(qc_t)
        circuit = qasm_to_pyzx_circuit(qasm)
        assert circuit.qubits == 3
        graph = pyzx_circuit_to_graph(circuit)
        assert graph.num_vertices() > 0
        # Should have 3 inputs and 3 outputs (boundary vertices)
        inputs = [v for v in graph.vertices() if graph.type(v) == VertexType.BOUNDARY]
        assert len(inputs) == 6  # 3 inputs + 3 outputs

    def test_unsupported_gate_raises(self):
        """QASM with unsupported gates should raise ValueError."""
        bad_qasm = (
            "OPENQASM 2.0;\n"
            "include \"qelib1.inc\";\n"
            "qreg q[2];\n"
            "foobar q[0],q[1];\n"
        )
        with pytest.raises(ValueError):
            qasm_to_pyzx_circuit(bad_qasm)

    def test_simplification_reduces_spider_count(self):
        """Full reduction should produce fewer spiders than raw."""
        qc = build_ghz(5)
        qc_t = transpile_to_gate_set(qc, ["cx", "rz", "h"])
        qasm = circuit_to_qasm(qc_t)
        circuit = qasm_to_pyzx_circuit(qasm)
        graph = pyzx_circuit_to_graph(circuit)
        result = simplify_graph(graph)
        assert result.spider_counts["full"] <= result.spider_counts["raw"]

    def test_all_three_levels_populated(self):
        """SimplificationResult should have non-None graphs at all levels."""
        qc = build_ghz(3)
        qc_t = transpile_to_gate_set(qc, ["cx", "rz", "h"])
        qasm = circuit_to_qasm(qc_t)
        circuit = qasm_to_pyzx_circuit(qasm)
        graph = pyzx_circuit_to_graph(circuit)
        result = simplify_graph(graph)
        assert result.raw is not None and result.raw.num_vertices() > 0
        assert result.clifford is not None and result.clifford.num_vertices() > 0
        assert result.full is not None and result.full.num_vertices() > 0

    def test_save_and_load_roundtrip(self, tmp_path):
        """save_diagram → load_diagram should reconstruct an equivalent graph."""
        qc = build_ghz(3)
        qc_t = transpile_to_gate_set(qc, ["cx", "rz", "h"])
        qasm = circuit_to_qasm(qc_t)
        circuit = qasm_to_pyzx_circuit(qasm)
        graph = pyzx_circuit_to_graph(circuit)
        metadata = {"source_algorithm": "ghz", "n_qubits": 3, "level": "raw"}
        record = save_diagram(graph, "ghz_3q_raw", tmp_path, metadata)
        loaded = load_diagram(record.json_path)
        assert loaded.num_vertices() == graph.num_vertices()
        assert loaded.num_edges() == graph.num_edges()


# ── Mining ──────────────────────────────────────────────────────────


class TestMining:
    """Tests for fingerprinting, webs, mining, and library."""

    def test_fingerprint_deterministic(self):
        """Same graph should always produce the same fingerprint."""
        pytest.skip("Not yet implemented")

    def test_compatibility_necessary_condition(self):
        """A sub-diagram's fingerprint should be compatible with its parent."""
        pytest.skip("Not yet implemented")

    def test_web_serialization_roundtrip(self):
        """to_dict → from_dict should reconstruct an equivalent web."""
        pytest.skip("Not yet implemented")

    def test_web_compatibility_check(self):
        """Webs with matching boundaries should be compatible."""
        pytest.skip("Not yet implemented")

    def test_library_add_and_get(self, tmp_path):
        """Adding a web and retrieving it by ID should work."""
        pytest.skip("Not yet implemented")

    def test_library_search_by_role(self, tmp_path):
        """search(role='oracle') should only return matching webs."""
        pytest.skip("Not yet implemented")


# ── Compose ─────────────────────────────────────────────────────────


class TestCompose:
    """Tests for templates and composition."""

    def test_load_from_config_format(self):
        """load_templates_from_config should parse the config format."""
        pytest.skip("Not yet implemented")

    def test_builtin_templates_have_slots(self):
        """Built-in templates should have at least 2 slots."""
        pytest.skip("Not yet implemented")

    def test_connect_webs_produces_valid_graph(self):
        """Connecting two compatible webs should produce a valid graph."""
        pytest.skip("Not yet implemented")

    def test_validate_candidate_rejects_unbalanced(self):
        """A graph with mismatched inputs/outputs should fail validation."""
        pytest.skip("Not yet implemented")


# ── Extract ─────────────────────────────────────────────────────────


class TestExtract:
    """Tests for flow detection and circuit extraction."""

    def test_circuit_derived_diagram_has_gflow(self):
        """A diagram from a circuit should always have gFlow."""
        pytest.skip("Not yet implemented")

    def test_flow_result_fields(self):
        """FlowResult should correctly report exists=True and flow type."""
        pytest.skip("Not yet implemented")

    def test_extract_from_simple_circuit(self):
        """Extracting from a small circuit diagram should succeed."""
        pytest.skip("Not yet implemented")

    def test_extracted_circuit_is_equivalent(self):
        """The extracted circuit should be unitarily equivalent."""
        pytest.skip("Not yet implemented")


# ── Benchmark ───────────────────────────────────────────────────────


class TestBenchmark:
    """Tests for metrics, simulation, and comparison."""

    def test_metrics_from_known_circuit(self):
        """Metrics for a hand-crafted circuit should match expected values."""
        pytest.skip("Not yet implemented")

    def test_improvement_positive_when_better(self):
        """compute_improvement should return positive values when better."""
        pytest.skip("Not yet implemented")


# ── Report ──────────────────────────────────────────────────────────


class TestReport:
    """Tests for novelty assessment, provenance, and export."""

    def test_above_threshold_is_novel(self):
        """Candidate improving by more than threshold should be novel."""
        pytest.skip("Not yet implemented")

    def test_below_threshold_is_not_novel(self):
        """Candidate improving by less than threshold should not be novel."""
        pytest.skip("Not yet implemented")

    def test_provenance_to_markdown_format(self):
        """provenance_to_markdown should produce valid Markdown."""
        pytest.skip("Not yet implemented")


# ── Pipeline integration ────────────────────────────────────────────


class TestPipeline:
    """Integration tests for the pipeline orchestrator."""

    def test_config_loading(self, tmp_path):
        """PipelineConfig.from_yaml should load all sections."""
        pytest.skip("Not yet implemented")

    def test_single_stage_execution(self):
        """run_pipeline with a stage argument should only run that stage."""
        pytest.skip("Not yet implemented")
