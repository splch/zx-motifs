"""Tests for MAGIC Phase 2 modules: problem_aware, assembler, trainability, phase_opt, emergent."""
from fractions import Fraction

import numpy as np
import pytest
from qiskit import QuantumCircuit

from zx_motifs.pipeline.benchmark_suite import (
    PauliTerm,
    h2_molecule,
    tfim_1d,
    maxcut_regular,
)
from zx_motifs.pipeline.problem_aware import (
    _paulis_commute,
    build_anticommutation_graph,
    decompose_hamiltonian,
    partition_into_groups,
)
from zx_motifs.pipeline.assembler import (
    MAGICConfig,
    assemble_magic,
    assemble_for_problem,
)
from zx_motifs.pipeline.trainability import (
    compute_dla_dimension,
    _pauli_string_commutator,
)
from zx_motifs.pipeline.phase_opt import (
    count_cnots_qiskit,
    optimize_phase_polynomial,
    optimization_report,
)


# ── Problem-Aware Composition ──────────────────────────────────────


class TestPauliCommutation:
    def test_zz_commutes_with_zz(self):
        assert _paulis_commute("ZZII", "IIZZ")

    def test_zz_anticommutes_with_xz(self):
        assert not _paulis_commute("ZZII", "XZII")

    def test_identity_commutes_with_all(self):
        assert _paulis_commute("IIII", "XYZZ")

    def test_same_string_commutes(self):
        assert _paulis_commute("XYZZ", "XYZZ")


class TestGrouping:
    def test_partition_tfim(self):
        p = tfim_1d(n=4)
        groups = partition_into_groups(p.pauli_terms)
        assert len(groups) > 0
        # All terms should be assigned to some group
        total_terms = sum(len(g.terms) for g in groups)
        nontrivial = [t for t in p.pauli_terms if any(c != "I" for c in t.pauli_string)]
        assert total_terms == len(nontrivial)

    def test_commuting_terms_in_same_group(self):
        terms = [
            PauliTerm(1.0, "ZZII"),
            PauliTerm(1.0, "IIZZ"),
        ]
        groups = partition_into_groups(terms)
        # ZZ on different qubits commute, should be in same group
        assert len(groups) == 1

    def test_anticommuting_terms_in_different_groups(self):
        terms = [
            PauliTerm(1.0, "XIII"),
            PauliTerm(1.0, "ZIII"),
        ]
        groups = partition_into_groups(terms)
        assert len(groups) == 2


class TestDecomposition:
    def test_decompose_h2(self):
        p = h2_molecule()
        result = decompose_hamiltonian(p.pauli_terms)
        assert result.n_qubits == 4
        assert result.n_groups > 0
        assert result.identity_offset != 0  # H2 has identity term

    def test_decompose_with_identity(self):
        terms = [
            PauliTerm(-1.0, "IIII"),
            PauliTerm(0.5, "ZZII"),
        ]
        result = decompose_hamiltonian(terms)
        assert abs(result.identity_offset - (-1.0)) < 1e-10
        assert result.n_groups == 1


# ── MAGIC Assembler ────────────────────────────────────────────────


class TestAssembler:
    def test_assemble_h2_layered(self):
        p = h2_molecule()
        qc = assemble_magic(problem=p)
        assert qc.num_qubits == 4
        assert len(qc.data) > 0

    def test_assemble_tfim_brick(self):
        p = tfim_1d(n=4)
        config = MAGICConfig(pattern="brick", n_layers=1)
        qc = assemble_magic(problem=p, config=config)
        assert qc.num_qubits == 4

    def test_assemble_star_pattern(self):
        p = tfim_1d(n=4)
        config = MAGICConfig(pattern="star", n_layers=1)
        qc = assemble_magic(problem=p, config=config)
        assert qc.num_qubits == 4

    def test_assemble_for_problem_convenience(self):
        p = tfim_1d(n=4)
        qc = assemble_for_problem(p, pattern="layered", n_layers=1)
        assert qc.num_qubits == 4

    def test_assemble_from_pauli_terms(self):
        terms = [
            PauliTerm(1.0, "ZZII"),
            PauliTerm(1.0, "IZZI"),
            PauliTerm(0.5, "XIII"),
            PauliTerm(0.5, "IXII"),
        ]
        qc = assemble_magic(pauli_terms=terms, n_qubits=4)
        assert qc.num_qubits == 4

    def test_assemble_no_input_raises(self):
        with pytest.raises(ValueError):
            assemble_magic()

    def test_vqe_convergence_h2(self):
        """MAGIC circuit should converge to within 5% of exact on H2."""
        from zx_motifs.pipeline.evaluation import vqe_test

        p = h2_molecule()
        config = MAGICConfig(pattern="layered", n_layers=2)
        qc = assemble_magic(problem=p, config=config)
        H = p.hamiltonian_matrix()
        result = vqe_test(qc, p.n_qubits, H, n_restarts=10, maxiter=300)
        rel_error = abs(result["best_energy"] - p.exact_energy) / abs(p.exact_energy)
        assert rel_error < 0.05, f"VQE relative error {rel_error:.4f} > 5%"


# ── Trainability (DLA) ─────────────────────────────────────────────


class TestDLA:
    def test_single_generator(self):
        result = compute_dla_dimension(["ZI"])
        assert result.dimension >= 1

    def test_two_commuting_generators(self):
        result = compute_dla_dimension(["ZI", "IZ"])
        assert result.dimension == 2

    def test_two_anticommuting_generators(self):
        result = compute_dla_dimension(["XI", "ZI"])
        assert result.dimension >= 3  # [X,Z] = 2iY, gives at least 3

    def test_full_su4(self):
        """All single-qubit Paulis on 2 qubits should generate su(4)."""
        generators = ["XI", "IX", "ZI", "IZ", "XX", "ZZ"]
        result = compute_dla_dimension(generators)
        assert result.dimension == 15  # dim(su(4)) = 15

    def test_empty_generators(self):
        result = compute_dla_dimension([])
        assert result.dimension == 0


class TestPauliStringCommutator:
    def test_xz_gives_y(self):
        result = _pauli_string_commutator("XI", "ZI")
        assert result == "YI"

    def test_commuting_returns_none(self):
        assert _pauli_string_commutator("ZI", "IZ") is None


# ── Phase Optimization ─────────────────────────────────────────────


class TestPhaseOpt:
    def test_count_cnots(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        assert count_cnots_qiskit(qc) == 2

    def test_optimize_preserves_circuit(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.t(1)
        qc.cx(0, 1)
        optimized = optimize_phase_polynomial(qc)
        assert optimized.num_qubits == 3

    def test_optimization_report(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.t(1)
        qc.cx(0, 1)
        report = optimization_report(qc, qc)
        assert report["original_cx"] == 2
        assert report["reduction"] == 0
