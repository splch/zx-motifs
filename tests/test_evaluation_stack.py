"""Tests for Phase 3: scorecard, baselines, statistics, classical hardness, noise."""
import numpy as np
import pytest
from qiskit import QuantumCircuit

from zx_motifs.pipeline.baselines import (
    BASELINE_FACTORIES,
    build_all_baselines,
    build_baseline,
    hea_cz_brickwork,
    hea_cx_chain,
    hva_ansatz,
    mps_circuit_ansatz,
    symmetry_preserving_hea,
    uccsd_ansatz,
)
from zx_motifs.pipeline.benchmark_suite import h2_molecule, tfim_1d
from zx_motifs.pipeline.classical_hardness import (
    light_cone_analysis,
    stabilizer_check,
)
from zx_motifs.pipeline.noise_analysis import (
    evaluate_t0_ideal,
    gate_budget_analysis,
    fault_tolerant_estimate,
)
from zx_motifs.pipeline.scorecard import (
    Scorecard,
    evaluate_layer1_structural,
    evaluate_layer4_performance,
    evaluate_scorecard,
)
from zx_motifs.pipeline.statistics import (
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    compare_results,
    fit_scaling,
)


# ── Baselines ──────────────────────────────────────────────────────


class TestBaselines:
    def test_all_baselines_build(self):
        baselines = build_all_baselines(n_qubits=4)
        assert len(baselines) == 8

    def test_baseline_qubit_counts(self):
        for name in BASELINE_FACTORIES:
            qc = build_baseline(name, n_qubits=4)
            assert qc.num_qubits == 4, f"{name} has wrong qubit count"

    def test_hea_cz_layers(self):
        qc = hea_cz_brickwork(4, n_layers=2)
        assert qc.num_qubits == 4
        assert len(qc.data) > 0

    def test_uccsd(self):
        qc = uccsd_ansatz(4, n_electrons=2)
        assert qc.num_qubits == 4

    def test_hva(self):
        qc = hva_ansatz(4, n_layers=2)
        assert qc.num_qubits == 4

    def test_mps_circuit(self):
        qc = mps_circuit_ansatz(4, bond_dim=2)
        assert qc.num_qubits == 4

    def test_symmetry_preserving(self):
        qc = symmetry_preserving_hea(4, n_layers=2)
        assert qc.num_qubits == 4

    def test_unknown_baseline_raises(self):
        with pytest.raises(ValueError):
            build_baseline("nonexistent", 4)

    def test_baselines_converge_h2(self):
        """At least one baseline should converge on H2."""
        from zx_motifs.pipeline.evaluation import vqe_test

        p = h2_molecule()
        H = p.hamiltonian_matrix()
        best = float("inf")
        for name, factory in list(BASELINE_FACTORIES.items())[:3]:  # test first 3
            qc = factory(n_qubits=4)
            result = vqe_test(qc, 4, H, n_restarts=5, maxiter=200)
            if result["best_energy"] < best:
                best = result["best_energy"]
        rel_error = abs(best - p.exact_energy) / abs(p.exact_energy)
        assert rel_error < 0.1, f"No baseline converged: best rel_error={rel_error}"


# ── Scorecard ──────────────────────────────────────────────────────


class TestScorecard:
    def test_layer1_valid_circuit(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = evaluate_layer1_structural(qc, 2)
        assert result.passed
        assert not result.killed

    def test_layer4_h2(self):
        from zx_motifs.pipeline.assembler import assemble_for_problem

        p = h2_molecule()
        qc = assemble_for_problem(p, n_layers=2)
        H = p.hamiltonian_matrix()
        result = evaluate_layer4_performance(qc, 4, H, p.exact_energy, n_restarts=5)
        assert result.metrics["relative_error"] < 0.1

    def test_full_scorecard(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.ry(0.5, 0)
        qc.rz(0.3, 1)
        card = evaluate_scorecard(qc, "test", 2, layers=[1])
        assert card.n_qubits == 2
        assert len(card.layers) == 1

    def test_scorecard_classification(self):
        card = Scorecard(ansatz_name="test", n_qubits=4)
        assert card.classification == "insufficient"


# ── Statistics ─────────────────────────────────────────────────────


class TestStatistics:
    def test_cohens_d(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        d = cohens_d(x, y)
        assert abs(d - (-2 / np.sqrt(2.5))) < 0.1

    def test_cliffs_delta(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        delta = cliffs_delta(x, y)
        assert delta == -1.0  # all x < y

    def test_bootstrap_ci(self):
        data = np.random.default_rng(42).normal(0, 1, 100)
        lo, hi = bootstrap_ci(data)
        assert lo < 0 < hi or abs(lo) < 0.5  # 0 should be in CI

    def test_compare_results(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        result = compare_results(a, b, "A", "B", "test")
        assert result.mean_a < result.mean_b
        assert isinstance(result.wilcoxon_p, float)

    def test_fit_scaling(self):
        ns = np.array([4, 6, 8, 10, 12])
        values = 2.0 * ns**1.5  # power law
        fits = fit_scaling(ns, values)
        assert len(fits) > 0
        assert fits[0].r_squared > 0.9


# ── Classical Hardness ─────────────────────────────────────────────


class TestClassicalHardness:
    def test_stabilizer_clifford(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = stabilizer_check(qc)
        assert result.classically_tractable

    def test_stabilizer_non_clifford(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.t(0)
        result = stabilizer_check(qc)
        assert not result.classically_tractable

    def test_light_cone(self):
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(1, 2)
        # Qubit 3 is disconnected
        result = light_cone_analysis(qc)
        assert result.details["sizes"][3] == 1  # qubit 3 has trivial light cone


# ── Noise Analysis ─────────────────────────────────────────────────


class TestNoiseAnalysis:
    def test_t0_ideal(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        H = np.diag([1, -1, -1, 1]).astype(complex)  # ZZ
        result = evaluate_t0_ideal(qc, H)
        assert result.tier == "T0"
        assert abs(result.energy - (-1.0)) < 0.1

    def test_gate_budget(self):
        qc = QuantumCircuit(4)
        for _ in range(10):
            qc.cx(0, 1)
            qc.cx(2, 3)
        result = gate_budget_analysis(qc, error_rate=0.01)
        assert result["n_2q_gates"] == 20
        assert result["total_error_estimate"] == pytest.approx(0.2)

    def test_ft_estimate(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.t(1)
        result = fault_tolerant_estimate(qc)
        assert result["logical_qubits"] == 4
        assert result["t_count"] >= 0
