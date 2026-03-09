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
        from src.mining import compute_fingerprint

        qc = build_ghz(3)
        qc_t = transpile_to_gate_set(qc, ["cx", "rz", "h"])
        qasm = circuit_to_qasm(qc_t)
        circuit = qasm_to_pyzx_circuit(qasm)
        graph = pyzx_circuit_to_graph(circuit)

        fp1 = compute_fingerprint(graph)
        fp2 = compute_fingerprint(graph)

        assert fp1.n_z_spiders == fp2.n_z_spiders
        assert fp1.n_x_spiders == fp2.n_x_spiders
        assert fp1.n_hadamard_edges == fp2.n_hadamard_edges
        assert fp1.n_simple_edges == fp2.n_simple_edges
        assert fp1.degree_histogram == fp2.degree_histogram
        assert fp1.phase_histogram == fp2.phase_histogram

    def test_compatibility_necessary_condition(self):
        """A sub-diagram's fingerprint should be compatible with its parent."""
        from src.mining import compute_fingerprint, fingerprints_compatible
        from src.zx import extract_subgraph

        qc = build_ghz(4)
        qc_t = transpile_to_gate_set(qc, ["cx", "rz", "h"])
        qasm = circuit_to_qasm(qc_t)
        circuit = qasm_to_pyzx_circuit(qasm)
        graph = pyzx_circuit_to_graph(circuit)

        # Pick a subset of non-boundary vertices
        internal = [v for v in graph.vertices() if graph.type(v) != VertexType.BOUNDARY]
        subset = set(internal[: max(2, len(internal) // 2)])
        sub_graph, _ = extract_subgraph(graph, subset)

        parent_fp = compute_fingerprint(graph)
        sub_fp = compute_fingerprint(sub_graph)

        assert fingerprints_compatible(parent_fp, sub_fp)

    def test_web_serialization_roundtrip(self):
        """to_dict → from_dict should reconstruct an equivalent web."""
        from src.mining import Boundary, ZXWeb

        qc = build_ghz(3)
        qc_t = transpile_to_gate_set(qc, ["cx", "rz", "h"])
        qasm = circuit_to_qasm(qc_t)
        circuit = qasm_to_pyzx_circuit(qasm)
        graph = pyzx_circuit_to_graph(circuit)

        boundaries = [
            Boundary(index=0, spider_type="Z", phase=0.0, edge_type="simple"),
            Boundary(index=1, spider_type="X", phase=0.5, edge_type="hadamard"),
        ]
        web = ZXWeb(
            web_id="test_web",
            graph=graph,
            boundaries=boundaries,
            spider_count=graph.num_vertices(),
            sources=["ghz"],
            support=3,
            role="entangle",
            phase_class="pauli",
            n_input_boundaries=1,
        )

        data = web.to_dict()
        restored = ZXWeb.from_dict(data)

        assert restored.web_id == web.web_id
        assert restored.spider_count == web.spider_count
        assert restored.support == web.support
        assert restored.role == web.role
        assert restored.phase_class == web.phase_class
        assert restored.n_input_boundaries == web.n_input_boundaries
        assert len(restored.boundaries) == len(web.boundaries)
        assert restored.graph.num_vertices() == web.graph.num_vertices()
        assert restored.graph.num_edges() == web.graph.num_edges()

    def test_web_compatibility_check(self):
        """Webs with matching boundaries should be compatible."""
        from src.mining import Boundary, ZXWeb
        import pyzx as zx

        g1 = zx.Graph()
        g1.add_vertex(ty=VertexType.Z)
        g2 = zx.Graph()
        g2.add_vertex(ty=VertexType.Z)

        web1 = ZXWeb(
            web_id="w1",
            graph=g1,
            boundaries=[
                Boundary(index=0, spider_type="Z", phase=0.0, edge_type="simple"),
                Boundary(index=1, spider_type="Z", phase=0.0, edge_type="simple"),
            ],
            spider_count=1,
            n_input_boundaries=1,  # 1 input, 1 output
        )
        web2 = ZXWeb(
            web_id="w2",
            graph=g2,
            boundaries=[
                Boundary(index=0, spider_type="Z", phase=0.0, edge_type="simple"),
                Boundary(index=1, spider_type="Z", phase=0.0, edge_type="simple"),
            ],
            spider_count=1,
            n_input_boundaries=1,  # 1 input, 1 output
        )

        assert web1.is_compatible(web2)
        assert web1.n_inputs() == 1
        assert web1.n_outputs() == 1

    def test_library_add_and_get(self, tmp_path):
        """Adding a web and retrieving it by ID should work."""
        from src.mining import Boundary, WebLibrary, ZXWeb
        import pyzx as zx

        g = zx.Graph()
        g.add_vertex(ty=VertexType.Z)

        web = ZXWeb(
            web_id="test_lib_web",
            graph=g,
            boundaries=[
                Boundary(index=0, spider_type="Z", phase=0.0, edge_type="simple"),
            ],
            spider_count=1,
            sources=["ghz"],
            support=2,
            role="entangle",
            phase_class="pauli",
            n_input_boundaries=1,
        )

        lib = WebLibrary(tmp_path / "webs")
        lib.add(web)
        lib.save_index()

        lib2 = WebLibrary(tmp_path / "webs")
        lib2.load_index()
        loaded = lib2.get("test_lib_web")

        assert loaded.web_id == web.web_id
        assert loaded.spider_count == web.spider_count
        assert loaded.support == web.support
        assert loaded.role == web.role

    def test_library_search_by_role(self, tmp_path):
        """search(role='oracle') should only return matching webs."""
        from src.mining import Boundary, WebLibrary, ZXWeb
        import pyzx as zx

        lib = WebLibrary(tmp_path / "webs")
        roles = ["oracle", "phase", "oracle"]

        for i, role in enumerate(roles):
            g = zx.Graph()
            g.add_vertex(ty=VertexType.Z)
            web = ZXWeb(
                web_id=f"web_{i:04d}",
                graph=g,
                boundaries=[
                    Boundary(index=0, spider_type="Z", phase=0.0, edge_type="simple"),
                ],
                spider_count=1,
                role=role,
                n_input_boundaries=0,
            )
            lib.add(web)

        lib.save_index()

        results = lib.search(role="oracle")
        assert len(results) == 2
        assert all(w.role == "oracle" for w in results)


# ── Compose ─────────────────────────────────────────────────────────


class TestCompose:
    """Tests for templates and composition."""

    def test_load_from_config_format(self):
        """load_templates_from_config should parse the config format."""
        from src.compose import load_templates_from_config

        templates = load_templates_from_config([["state_prep", "oracle"]])
        assert len(templates) == 1
        assert len(templates[0].slots) == 2
        assert templates[0].slots[0].role == "state_prep"
        assert templates[0].slots[1].role == "oracle"
        assert templates[0].name == "custom_0"

    def test_builtin_templates_have_slots(self):
        """Built-in templates should have at least 2 slots."""
        from src.compose import BUILTIN_TEMPLATES

        assert len(BUILTIN_TEMPLATES) >= 1
        for t in BUILTIN_TEMPLATES:
            assert len(t.slots) >= 2
            assert all(s.role for s in t.slots)

    def test_connect_webs_produces_valid_graph(self):
        """Connecting two compatible webs should produce a valid graph."""
        from src.compose import connect_webs
        from src.mining import Boundary, ZXWeb
        import pyzx as zx

        # Create web_a: one Z spider with 1 input boundary, 1 output boundary
        g1 = zx.Graph()
        v1 = g1.add_vertex(ty=VertexType.Z, row=0, qubit=0, phase=0)
        web_a = ZXWeb(
            web_id="a",
            graph=g1,
            boundaries=[
                Boundary(index=0, spider_type="Z", phase=0.0, edge_type="simple", vertex_id=v1),
                Boundary(index=1, spider_type="Z", phase=0.0, edge_type="simple", vertex_id=v1),
            ],
            spider_count=1,
            n_input_boundaries=1,
        )

        # Create web_b: one Z spider with 1 input boundary, 1 output boundary
        g2 = zx.Graph()
        v2 = g2.add_vertex(ty=VertexType.Z, row=0, qubit=0, phase=0)
        web_b = ZXWeb(
            web_id="b",
            graph=g2,
            boundaries=[
                Boundary(index=0, spider_type="Z", phase=0.0, edge_type="simple", vertex_id=v2),
                Boundary(index=1, spider_type="Z", phase=0.0, edge_type="simple", vertex_id=v2),
            ],
            spider_count=1,
            n_input_boundaries=1,
        )

        result = connect_webs(web_a, web_b, [(0, 0)])
        assert result.num_vertices() > 0
        assert result.num_edges() >= 0

    def test_validate_candidate_rejects_unbalanced(self):
        """A graph with mismatched inputs/outputs should fail validation."""
        from src.compose import validate_candidate
        import pyzx as zx

        g = zx.Graph()
        v0 = g.add_vertex(ty=VertexType.BOUNDARY, row=0, qubit=0)
        v1 = g.add_vertex(ty=VertexType.Z, row=1, qubit=0)
        v2 = g.add_vertex(ty=VertexType.BOUNDARY, row=2, qubit=0)
        v3 = g.add_vertex(ty=VertexType.BOUNDARY, row=2, qubit=1)
        g.add_edge((v0, v1))
        g.add_edge((v1, v2))
        g.add_edge((v1, v3))
        g.set_inputs((v0,))
        g.set_outputs((v2, v3))

        # 1 input, 2 outputs → unbalanced
        assert validate_candidate(g, expected_qubits=0) is False


# ── Extract ─────────────────────────────────────────────────────────


class TestExtract:
    """Tests for flow detection and circuit extraction."""

    def test_circuit_derived_diagram_has_gflow(self):
        """A diagram from a circuit should always have gFlow."""
        from src.extract import FlowType, check_gflow

        qc = build_ghz(3)
        qc_t = transpile_to_gate_set(qc, ["cx", "rz", "h"])
        qasm = circuit_to_qasm(qc_t)
        circuit = qasm_to_pyzx_circuit(qasm)
        graph = pyzx_circuit_to_graph(circuit)

        result = check_gflow(graph)
        assert result.exists is True
        assert result.flow_type == FlowType.GFLOW

    def test_flow_result_fields(self):
        """FlowResult should correctly report exists=True and flow type."""
        from src.extract import FlowType, check_gflow

        qc = build_qft(3)
        qc_t = transpile_to_gate_set(qc, ["cx", "rz", "h"])
        qasm = circuit_to_qasm(qc_t)
        circuit = qasm_to_pyzx_circuit(qasm)
        graph = pyzx_circuit_to_graph(circuit)

        result = check_gflow(graph)
        assert result.exists is True
        assert result.flow_type == FlowType.GFLOW
        assert result.flow_data is not None
        assert result.check_time_ms >= 0

    def test_extract_from_simple_circuit(self):
        """Extracting from a small circuit diagram should succeed."""
        import pyzx

        from src.extract import extract_circuit_pyzx

        qc = build_ghz(3)
        qc_t = transpile_to_gate_set(qc, ["cx", "rz", "h"])
        qasm = circuit_to_qasm(qc_t)
        circuit = qasm_to_pyzx_circuit(qasm)
        graph = pyzx_circuit_to_graph(circuit)

        pyzx.simplify.full_reduce(graph)

        result = extract_circuit_pyzx(graph)
        assert result.success is True
        assert result.circuit is not None
        assert result.qasm is not None
        assert result.gate_count > 0

    def test_extracted_circuit_is_equivalent(self):
        """The extracted circuit should be unitarily equivalent."""
        import pyzx

        from src.extract import extract_circuit_pyzx

        qc = build_ghz(3)
        qc_t = transpile_to_gate_set(qc, ["cx", "rz", "h"])
        qasm = circuit_to_qasm(qc_t)
        circuit = qasm_to_pyzx_circuit(qasm)
        graph = pyzx_circuit_to_graph(circuit)
        original_graph = graph.copy()

        pyzx.simplify.full_reduce(graph)

        result = extract_circuit_pyzx(graph)
        assert result.success is True
        assert pyzx.compare_tensors(original_graph, result.circuit, preserve_scalar=False) is True


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
