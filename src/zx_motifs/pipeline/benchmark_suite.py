"""
Benchmark problem suite for MAGIC evaluation.

Provides Hamiltonian constructors returning sparse matrices plus Pauli
decompositions, exact ground state energies, and problem metadata for
three domains: quantum chemistry, spin models, and combinatorial optimization.

Chemistry problems require optional PySCF dependency. Spin model and
optimization problems use only numpy/scipy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from .ansatz import _pauli_matrix


# ── Data Structures ─────────────────────────────────────────────────


@dataclass
class PauliTerm:
    """A single Pauli term: coefficient * pauli_string."""

    coefficient: float
    pauli_string: str

    @property
    def n_qubits(self) -> int:
        return len(self.pauli_string)


@dataclass
class BenchmarkProblem:
    """A benchmark problem instance with Hamiltonian and reference data."""

    name: str
    domain: str  # "chemistry", "spin", "optimization"
    n_qubits: int
    pauli_terms: list[PauliTerm]
    exact_energy: float | None = None
    symmetries: list[str] = field(default_factory=list)
    locality: int = 2
    description: str = ""

    def hamiltonian_matrix(self) -> np.ndarray:
        """Build the full 2^n x 2^n Hamiltonian matrix."""
        d = 2**self.n_qubits
        H = np.zeros((d, d), dtype=complex)
        for term in self.pauli_terms:
            H += term.coefficient * _pauli_matrix(term.pauli_string)
        return H

    def sparse_hamiltonian(self) -> csr_matrix:
        """Return the Hamiltonian as a sparse matrix."""
        return csr_matrix(self.hamiltonian_matrix())

    def compute_exact_energy(self) -> float:
        """Compute exact ground state energy via diagonalization."""
        if self.n_qubits <= 16:
            H = self.hamiltonian_matrix()
            eigenvalues = np.linalg.eigvalsh(H.real)
            return float(eigenvalues[0])
        else:
            H_sparse = self.sparse_hamiltonian()
            eigenvalues, _ = eigsh(H_sparse, k=1, which="SA")
            return float(eigenvalues[0])


# ── Spin Model Hamiltonians ────────────────────────────────────────


def _nearest_neighbor_terms(
    n: int, paulis: str, coefficient: float = 1.0, periodic: bool = False
) -> list[PauliTerm]:
    """Generate nearest-neighbor Pauli terms on a 1D chain."""
    terms = []
    limit = n if periodic else n - 1
    for i in range(limit):
        j = (i + 1) % n
        label = ["I"] * n
        label[i] = paulis[0]
        label[j] = paulis[1]
        terms.append(PauliTerm(coefficient, "".join(label)))
    return terms


def _single_site_terms(n: int, pauli: str, coefficient: float = 1.0) -> list[PauliTerm]:
    """Generate single-site Pauli terms."""
    terms = []
    for i in range(n):
        label = ["I"] * n
        label[i] = pauli
        terms.append(PauliTerm(coefficient, "".join(label)))
    return terms


def tfim_1d(n: int, h: float = 1.0, J: float = 1.0) -> BenchmarkProblem:
    """Transverse-field Ising model: H = -J sum ZZ - h sum X."""
    terms = _nearest_neighbor_terms(n, "ZZ", -J)
    terms += _single_site_terms(n, "X", -h)
    problem = BenchmarkProblem(
        name=f"tfim_1d_n{n}",
        domain="spin",
        n_qubits=n,
        pauli_terms=terms,
        symmetries=["Z2_parity"],
        locality=2,
        description=f"1D TFIM with h={h}, J={J}",
    )
    problem.exact_energy = problem.compute_exact_energy()
    return problem


def heisenberg_xxx_1d(n: int, J: float = 1.0) -> BenchmarkProblem:
    """Heisenberg XXX model: H = J sum (XX + YY + ZZ)."""
    terms = []
    for pauli in ["XX", "YY", "ZZ"]:
        terms += _nearest_neighbor_terms(n, pauli, J)
    problem = BenchmarkProblem(
        name=f"heisenberg_xxx_n{n}",
        domain="spin",
        n_qubits=n,
        pauli_terms=terms,
        symmetries=["SU2", "particle_number", "Z2_parity"],
        locality=2,
        description=f"1D Heisenberg XXX with J={J}",
    )
    problem.exact_energy = problem.compute_exact_energy()
    return problem


def xxz_1d(n: int, delta: float = 0.5, J: float = 1.0) -> BenchmarkProblem:
    """XXZ model: H = J sum (XX + YY + delta*ZZ)."""
    terms = []
    for pauli in ["XX", "YY"]:
        terms += _nearest_neighbor_terms(n, pauli, J)
    terms += _nearest_neighbor_terms(n, "ZZ", J * delta)
    problem = BenchmarkProblem(
        name=f"xxz_1d_n{n}_d{delta}",
        domain="spin",
        n_qubits=n,
        pauli_terms=terms,
        symmetries=["U1", "particle_number"],
        locality=2,
        description=f"1D XXZ with delta={delta}, J={J}",
    )
    problem.exact_energy = problem.compute_exact_energy()
    return problem


def j1j2_1d(n: int, j1: float = 1.0, j2: float = 0.5) -> BenchmarkProblem:
    """J1-J2 frustrated Heisenberg: H = J1*NN + J2*NNN."""
    terms = []
    for pauli in ["XX", "YY", "ZZ"]:
        terms += _nearest_neighbor_terms(n, pauli, j1)
    # Next-nearest-neighbor
    for i in range(n - 2):
        for pauli_pair in ["XX", "YY", "ZZ"]:
            label = ["I"] * n
            label[i] = pauli_pair[0]
            label[i + 2] = pauli_pair[1]
            terms.append(PauliTerm(j2, "".join(label)))
    problem = BenchmarkProblem(
        name=f"j1j2_n{n}",
        domain="spin",
        n_qubits=n,
        pauli_terms=terms,
        symmetries=["SU2"],
        locality=2,
        description=f"1D J1-J2 with j1={j1}, j2={j2}",
    )
    problem.exact_energy = problem.compute_exact_energy()
    return problem


# ── Chemistry Hamiltonians ─────────────────────────────────────────


def h2_molecule(bond_length: float = 0.735) -> BenchmarkProblem:
    """Molecular hydrogen in minimal STO-3G basis (4 qubits, JW mapping).

    Uses precomputed Pauli terms for the standard H2 Hamiltonian.
    Reference energy at equilibrium: -1.137275 Ha.
    """
    # H2 STO-3G Hamiltonian in Jordan-Wigner encoding (4 qubits)
    # Standard Pauli decomposition from literature
    terms = [
        PauliTerm(-0.81261, "IIII"),
        PauliTerm(+0.17120, "IIIZ"),
        PauliTerm(-0.22279, "IIZI"),
        PauliTerm(+0.17120, "IZII"),
        PauliTerm(-0.22279, "ZIII"),
        PauliTerm(+0.12055, "IIZZ"),
        PauliTerm(+0.16862, "IZIZ"),
        PauliTerm(+0.04532, "XXYY"),
        PauliTerm(+0.04532, "YYXX"),
        PauliTerm(+0.04532, "XYYX"),
        PauliTerm(+0.04532, "YXXY"),
        PauliTerm(+0.16587, "ZIIZ"),
        PauliTerm(+0.16862, "ZIZI"),
        PauliTerm(+0.17435, "ZZII"),
    ]
    problem = BenchmarkProblem(
        name="h2_sto3g",
        domain="chemistry",
        n_qubits=4,
        pauli_terms=terms,
        exact_energy=-1.137275,
        symmetries=["particle_number", "Z2_parity", "real_valued"],
        locality=4,
        description=f"H2 molecule STO-3G basis, bond length={bond_length} A",
    )
    # Compute exact energy to verify
    computed = problem.compute_exact_energy()
    problem.exact_energy = computed
    return problem


def lih_molecule() -> BenchmarkProblem:
    """Lithium hydride in STO-3G basis (active space, 4 qubits).

    Uses precomputed Pauli terms for LiH active space.
    """
    # LiH active space (2 electrons, 2 orbitals -> 4 qubits JW)
    terms = [
        PauliTerm(-7.49895, "IIII"),
        PauliTerm(+0.18093, "IIIZ"),
        PauliTerm(-0.18093, "IIZI"),
        PauliTerm(-0.01271, "IZII"),
        PauliTerm(+0.01271, "ZIII"),
        PauliTerm(+0.16614, "IIZZ"),
        PauliTerm(+0.12170, "IZIZ"),
        PauliTerm(+0.04530, "XXYY"),
        PauliTerm(+0.04530, "YYXX"),
        PauliTerm(+0.04530, "XYYX"),
        PauliTerm(+0.04530, "YXXY"),
        PauliTerm(+0.17464, "ZIIZ"),
        PauliTerm(+0.12170, "ZIZI"),
        PauliTerm(+0.16892, "ZZII"),
    ]
    problem = BenchmarkProblem(
        name="lih_sto3g",
        domain="chemistry",
        n_qubits=4,
        pauli_terms=terms,
        symmetries=["particle_number", "real_valued"],
        locality=4,
        description="LiH molecule STO-3G active space",
    )
    problem.exact_energy = problem.compute_exact_energy()
    return problem


# ── Optimization Hamiltonians ──────────────────────────────────────


def maxcut_regular(n: int, degree: int = 3, seed: int = 42) -> BenchmarkProblem:
    """MaxCut on a random regular graph.

    H = sum_{(i,j) in E} (1 - Z_i Z_j) / 2 = const - sum ZiZj / 2
    Minimizing H <-> maximizing the cut.
    """
    rng = np.random.default_rng(seed)
    # Generate random regular graph edges
    if degree * n % 2 != 0:
        n += 1  # need even degree*n
    edges = _random_regular_edges(n, degree, rng)
    terms = []
    for i, j in edges:
        label = ["I"] * n
        label[i] = "Z"
        label[j] = "Z"
        terms.append(PauliTerm(-0.5, "".join(label)))
    # Constant offset
    terms.append(PauliTerm(len(edges) / 2, "I" * n))

    problem = BenchmarkProblem(
        name=f"maxcut_reg{degree}_n{n}",
        domain="optimization",
        n_qubits=n,
        pauli_terms=terms,
        symmetries=["Z2_flip"],
        locality=2,
        description=f"MaxCut on {degree}-regular graph, {n} vertices",
    )
    problem.exact_energy = problem.compute_exact_energy()
    return problem


def _random_regular_edges(n: int, degree: int, rng) -> list[tuple[int, int]]:
    """Generate edges for a random regular graph using configuration model."""
    import networkx as nx

    G = nx.random_regular_graph(degree, n, seed=int(rng.integers(0, 2**31)))
    return list(G.edges())


def weighted_maxcut(n: int, p: float = 0.5, seed: int = 42) -> BenchmarkProblem:
    """Weighted MaxCut on Erdos-Renyi graph.

    H = sum_{(i,j) in E} w_{ij} * (1 - Z_i Z_j) / 2
    """
    rng = np.random.default_rng(seed)
    terms = []
    edge_count = 0
    total_weight = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                w = rng.uniform(0.5, 2.0)
                label = ["I"] * n
                label[i] = "Z"
                label[j] = "Z"
                terms.append(PauliTerm(-w / 2, "".join(label)))
                total_weight += w
                edge_count += 1
    # Constant offset
    terms.append(PauliTerm(total_weight / 2, "I" * n))

    problem = BenchmarkProblem(
        name=f"wmaxcut_er_n{n}",
        domain="optimization",
        n_qubits=n,
        pauli_terms=terms,
        symmetries=[],
        locality=2,
        description=f"Weighted MaxCut on ER(p={p}) graph, {n} vertices, {edge_count} edges",
    )
    problem.exact_energy = problem.compute_exact_energy()
    return problem


def portfolio_qubo(n: int, seed: int = 42) -> BenchmarkProblem:
    """Portfolio optimization QUBO as an Ising Hamiltonian.

    Minimize: x^T Sigma x - mu^T x + penalty * (sum x - k)^2
    Mapped to Ising: z_i = 2*x_i - 1.
    """
    rng = np.random.default_rng(seed)
    # Generate random returns and covariance
    mu = rng.uniform(0.01, 0.1, n)
    A = rng.normal(0, 0.02, (n, n))
    sigma = A @ A.T + 0.01 * np.eye(n)
    k = n // 3  # target assets to select
    penalty = 2.0

    # Convert to Ising: x_i = (1 + z_i) / 2
    terms = []
    # Constant term
    c0 = 0.25 * np.sum(sigma) - 0.5 * np.sum(mu) + penalty * (n / 2 - k) ** 2
    terms.append(PauliTerm(float(c0), "I" * n))

    # Linear terms
    for i in range(n):
        hi = 0.5 * np.sum(sigma[i, :]) - 0.5 * mu[i] + penalty * (n / 2 - k)
        hi *= 0.5
        if abs(hi) > 1e-10:
            label = ["I"] * n
            label[i] = "Z"
            terms.append(PauliTerm(float(hi), "".join(label)))

    # Quadratic terms
    for i in range(n):
        for j in range(i + 1, n):
            Jij = 0.25 * sigma[i, j] + 0.25 * penalty
            if abs(Jij) > 1e-10:
                label = ["I"] * n
                label[i] = "Z"
                label[j] = "Z"
                terms.append(PauliTerm(float(Jij), "".join(label)))

    problem = BenchmarkProblem(
        name=f"portfolio_n{n}",
        domain="optimization",
        n_qubits=n,
        pauli_terms=terms,
        symmetries=[],
        locality=2,
        description=f"Portfolio QUBO, {n} assets, target k={k}",
    )
    problem.exact_energy = problem.compute_exact_energy()
    return problem


# ── Problem Suite Registry ─────────────────────────────────────────

_PROBLEM_FACTORIES: dict[str, Callable[..., BenchmarkProblem]] = {
    "tfim_1d": tfim_1d,
    "heisenberg_xxx": heisenberg_xxx_1d,
    "xxz_1d": xxz_1d,
    "j1j2_1d": j1j2_1d,
    "h2": h2_molecule,
    "lih": lih_molecule,
    "maxcut": maxcut_regular,
    "weighted_maxcut": weighted_maxcut,
    "portfolio": portfolio_qubo,
}


def list_problems() -> list[str]:
    """List available benchmark problem names."""
    return list(_PROBLEM_FACTORIES.keys())


def get_problem(name: str, **kwargs) -> BenchmarkProblem:
    """Get a benchmark problem by name with optional parameters."""
    if name not in _PROBLEM_FACTORIES:
        raise ValueError(f"Unknown problem: {name}. Available: {list_problems()}")
    return _PROBLEM_FACTORIES[name](**kwargs)


def build_suite(
    qubit_sizes: dict[str, list[int]] | None = None,
) -> list[BenchmarkProblem]:
    """Build the standard benchmark suite across all domains.

    Args:
        qubit_sizes: Override qubit sizes per problem type.
            Defaults to small instances for testing.
    """
    defaults = {
        "tfim_1d": [4, 6, 8],
        "heisenberg_xxx": [4, 6, 8],
        "xxz_1d": [4, 6, 8],
        "j1j2_1d": [4, 6, 8],
        "maxcut": [4, 6, 8],
        "weighted_maxcut": [4, 6, 8],
        "portfolio": [4, 6, 8],
    }
    sizes = qubit_sizes or defaults
    problems = []

    # Chemistry (fixed qubit count)
    problems.append(h2_molecule())
    problems.append(lih_molecule())

    # Scalable problems
    for pname, ns in sizes.items():
        factory = _PROBLEM_FACTORIES[pname]
        for n in ns:
            problems.append(factory(n=n))

    return problems
