"""VQE evaluation harness, entangling power, and candidate scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize


# ── VQE engine ─────────────────────────────────────────────────────────


def vqe_test(
    entangler_qc: QuantumCircuit,
    n_qubits: int,
    H_matrix: np.ndarray,
    n_restarts: int = 10,
    maxiter: int = 400,
    seed: int = 42,
) -> dict:
    """Run VQE with RY/RZ - entangler - RY/RZ ansatz.

    Returns a dict with ``best_energy``, ``all_energies``, ``mean_energy``,
    ``std_energy``, and ``n_params``.
    """
    n_params = 4 * n_qubits

    def energy(params):
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.ry(params[2 * i], i)
            qc.rz(params[2 * i + 1], i)
        qc.compose(entangler_qc, inplace=True)
        for i in range(n_qubits):
            qc.ry(params[2 * n_qubits + 2 * i], i)
            qc.rz(params[2 * n_qubits + 2 * i + 1], i)
        sv = Statevector.from_instruction(qc)
        return float(np.real(np.array(sv.data).conj() @ H_matrix @ np.array(sv.data)))

    rng = np.random.default_rng(seed)
    best_energy = float("inf")
    all_energies = []

    for _ in range(n_restarts):
        x0 = rng.uniform(-np.pi, np.pi, n_params)
        try:
            res = minimize(energy, x0, method="COBYLA",
                           options={"maxiter": maxiter, "rhobeg": 0.5})
            all_energies.append(float(res.fun))
            if res.fun < best_energy:
                best_energy = float(res.fun)
        except Exception:
            pass

    return {
        "best_energy": best_energy,
        "all_energies": all_energies,
        "mean_energy": float(np.mean(all_energies)) if all_energies else float("inf"),
        "std_energy": float(np.std(all_energies)) if all_energies else 0.0,
        "n_params": n_params,
    }


def count_2q(qc: QuantumCircuit) -> int:
    """Count the number of 2-qubit (or larger) gates in a circuit."""
    return sum(1 for inst in qc.data if inst.operation.num_qubits >= 2)


def run_benchmark(
    configs: list[dict],
    n_seeds: int = 5,
    n_restarts: int = 10,
    maxiter: int = 400,
) -> pd.DataFrame:
    """Comparative VQE across ansatze, qubit sizes, and Hamiltonians.

    Parameters
    ----------
    configs : list[dict]
        Each dict should have keys ``name``, ``entangler_fn`` (callable(n) -> QC),
        ``n_qubits``, ``hamiltonian`` (np.ndarray), ``model`` (str).
    n_seeds : int
        Number of random seeds per configuration.
    n_restarts : int
        COBYLA restarts per VQE run.
    maxiter : int
        Max COBYLA iterations.

    Returns
    -------
    DataFrame with columns: name, n_qubits, model, seed, best_energy, exact_gs,
    relative_error, n_params.
    """
    from zx_motifs.pipeline.ansatz import build_hamiltonian

    rows = []
    for cfg in configs:
        name = cfg["name"]
        n_qubits = cfg["n_qubits"]
        model = cfg["model"]
        entangler_fn = cfg["entangler_fn"]
        H = cfg.get("hamiltonian")
        if H is None:
            H = build_hamiltonian(n_qubits, model)

        exact_gs = float(np.linalg.eigvalsh(H)[0])
        if exact_gs == 0:
            continue

        entangler_qc = entangler_fn(n_qubits)

        for seed in range(42, 42 + n_seeds):
            result = vqe_test(entangler_qc, n_qubits, H, n_restarts, maxiter, seed=seed)
            rel_err = abs(result["best_energy"] - exact_gs) / abs(exact_gs)
            rows.append({
                "name": name,
                "n_qubits": n_qubits,
                "model": model,
                "seed": seed,
                "best_energy": result["best_energy"],
                "exact_gs": exact_gs,
                "relative_error": rel_err,
                "n_params": result["n_params"],
            })

    return pd.DataFrame(rows)


# ── Entangling power ──────────────────────────────────────────────────


def _random_product_state(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random product state on *n* qubits."""
    state = np.array([1.0 + 0j])
    for _ in range(n):
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(0, 2 * np.pi)
        q = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
        state = np.kron(state, q)
    return state


def _half_cut_entropy(sv: np.ndarray, n: int) -> float:
    """Von Neumann entropy across the half-cut bipartition."""
    half = n // 2
    mat = sv.reshape(2**half, 2**(n - half))
    s = np.linalg.svd(mat, compute_uv=False)
    s = s[s > 1e-12]
    probs = s**2
    return float(-np.sum(probs * np.log2(probs)))


def compute_entangling_power(
    qc: QuantumCircuit,
    n_samples: int = 100,
) -> dict:
    """Average bipartite entropy over random product inputs.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit (entangler) to evaluate.
    n_samples : int
        Number of random product-state inputs to average over.

    Returns
    -------
    dict with ``entangling_power``, ``epd`` (std), ``max_entropy``,
    ``min_entropy``.
    """
    from qiskit.quantum_info import Operator

    n = qc.num_qubits
    U = Operator(qc).data
    rng = np.random.default_rng(42)
    entropies = []
    for _ in range(n_samples):
        psi_in = _random_product_state(n, rng)
        psi_out = U @ psi_in
        ent = _half_cut_entropy(psi_out, n)
        entropies.append(ent)
    return {
        "entangling_power": float(np.mean(entropies)),
        "epd": float(np.std(entropies)),
        "max_entropy": float(np.max(entropies)),
        "min_entropy": float(np.min(entropies)),
    }
