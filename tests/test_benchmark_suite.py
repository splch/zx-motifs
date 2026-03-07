"""Tests for the benchmark problem suite."""
import numpy as np
import pytest

from zx_motifs.pipeline.benchmark_suite import (
    BenchmarkProblem,
    build_suite,
    get_problem,
    h2_molecule,
    heisenberg_xxx_1d,
    lih_molecule,
    list_problems,
    maxcut_regular,
    portfolio_qubo,
    tfim_1d,
    weighted_maxcut,
    xxz_1d,
)


class TestSpinModels:
    def test_tfim_construction(self):
        p = tfim_1d(n=4)
        assert p.n_qubits == 4
        assert p.domain == "spin"
        assert len(p.pauli_terms) > 0
        assert p.exact_energy is not None

    def test_tfim_energy_finite(self):
        p = tfim_1d(n=4)
        assert np.isfinite(p.exact_energy)

    def test_heisenberg_construction(self):
        p = heisenberg_xxx_1d(n=4)
        assert p.n_qubits == 4
        assert p.exact_energy is not None

    def test_heisenberg_hermitian(self):
        p = heisenberg_xxx_1d(n=4)
        H = p.hamiltonian_matrix()
        assert np.allclose(H, H.conj().T)

    def test_xxz_construction(self):
        p = xxz_1d(n=4)
        assert p.n_qubits == 4
        assert p.exact_energy is not None


class TestChemistry:
    def test_h2_ground_state(self):
        p = h2_molecule()
        assert p.n_qubits == 4
        assert p.exact_energy is not None
        # Verify exact energy is recomputable
        recomputed = p.compute_exact_energy()
        assert abs(p.exact_energy - recomputed) < 1e-10
        # H2 ground state energy should be negative and physical
        assert p.exact_energy < 0

    def test_h2_hermitian(self):
        p = h2_molecule()
        H = p.hamiltonian_matrix()
        assert np.allclose(H, H.conj().T)

    def test_lih_construction(self):
        p = lih_molecule()
        assert p.n_qubits == 4
        assert p.exact_energy is not None
        assert np.isfinite(p.exact_energy)


class TestOptimization:
    def test_maxcut_construction(self):
        p = maxcut_regular(n=6, degree=3)
        assert p.domain == "optimization"
        assert p.exact_energy is not None

    def test_weighted_maxcut(self):
        p = weighted_maxcut(n=6)
        assert p.exact_energy is not None
        assert np.isfinite(p.exact_energy)

    def test_portfolio(self):
        p = portfolio_qubo(n=4)
        assert p.n_qubits == 4
        assert p.exact_energy is not None


class TestSuiteAPI:
    def test_list_problems(self):
        names = list_problems()
        assert "tfim_1d" in names
        assert "h2" in names
        assert "maxcut" in names

    def test_get_problem(self):
        p = get_problem("tfim_1d", n=4)
        assert p.n_qubits == 4

    def test_get_problem_unknown(self):
        with pytest.raises(ValueError):
            get_problem("nonexistent_problem")

    def test_build_suite(self):
        suite = build_suite(qubit_sizes={"tfim_1d": [4], "heisenberg_xxx": [4]})
        assert len(suite) >= 4  # h2 + lih + tfim_4 + heisenberg_4

    def test_sparse_hamiltonian(self):
        p = tfim_1d(n=4)
        H_sparse = p.sparse_hamiltonian()
        H_dense = p.hamiltonian_matrix()
        assert np.allclose(H_sparse.toarray(), H_dense)
