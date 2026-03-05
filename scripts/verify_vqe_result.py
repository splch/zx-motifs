#!/usr/bin/env python3
"""
VQE Result Verification Pipeline
=================================

Rigorous 6-phase verification of significant VQE findings from the deep
benchmark in compare_algorithms.py. Auto-selects targets from existing
results — no hardcoded names.

Outputs (scripts/output/verification/):
  verification_report.txt, verification_results.json,
  optimizer_comparison.png, statistical_significance.png,
  entanglement_structure.png, parameter_landscape.png,
  scaling_analysis.png, hamiltonian_generality.png,
  baseline_comparison.png, ablation_study.png,
  candidate_circuit.qasm
"""

from __future__ import annotations

import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector, partial_trace

# ── Project imports ───────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

from compare_algorithms import (
    OUTPUT_DIR as COMPARISON_DIR,
    _build_hamiltonian_matrix,
    _count_1q,
    _count_2q,
    _cx_chain_entangler,
    _half_cut_entropy,
    _hea_entangler,
    _pauli_matrix,
    _random_product_state,
    _scale_candidate_spec,
    compute_zx_compression,
    noise_fidelity,
    pauli_decompose,
    vqe_test,
)
from discover_algorithm import (
    CandidateSpec,
    _HANDCRAFTED_TEMPLATES,
    build_circuit_from_spec,
    build_corpus,
    build_template_registry,
    discover_motifs,
    load_phylogeny_results,
)
from zx_motifs.pipeline.converter import SimplificationLevel, convert_at_all_levels

warnings.filterwarnings("ignore")

OUTPUT_DIR = SCRIPT_DIR / "output" / "verification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE = OUTPUT_DIR / "verification_checkpoint.json"


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═══════════════════════════════════════════════════════════════════════


def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {}


def _save_checkpoint(ckpt: dict):
    CHECKPOINT_FILE.write_text(json.dumps(ckpt, indent=2, default=str))


def _phase_done(ckpt: dict, target_key: str, phase: str) -> bool:
    return ckpt.get(target_key, {}).get(phase) is not None


def _save_phase(ckpt: dict, target_key: str, phase: str, data: dict):
    ckpt.setdefault(target_key, {})[phase] = data
    _save_checkpoint(ckpt)


# ═══════════════════════════════════════════════════════════════════════
# Extended VQE test (multi-optimizer, returns detailed results)
# ═══════════════════════════════════════════════════════════════════════


def _run_optimizer(energy_fn, x0, method, n_params, maxiter, rng):
    """Run a single optimizer restart. Returns (best_fun, best_x, n_evals)."""
    from scipy.optimize import minimize

    if method == "SPSA":
        x = x0.copy()
        a0, c0, A, alpha, gamma = 0.1, 0.1, 10.0, 0.602, 0.101
        best_spsa = float("inf")
        best_x = x.copy()
        n_evals = 0
        for k in range(maxiter):
            ak = a0 / (k + 1 + A) ** alpha
            ck = c0 / (k + 1) ** gamma
            delta = rng.choice([-1, 1], size=n_params)
            fp = energy_fn(x + ck * delta)
            fm = energy_fn(x - ck * delta)
            g = (fp - fm) / (2 * ck * delta)
            x = x - ak * g
            n_evals += 2
            # Evaluate at actual iterate every 50 steps
            if (k + 1) % 50 == 0 or k == maxiter - 1:
                val = energy_fn(x)
                n_evals += 1
                if val < best_spsa:
                    best_spsa = val
                    best_x = x.copy()
        return best_spsa, best_x, n_evals
    elif method == "L-BFGS-B":
        res = minimize(
            energy_fn, x0, method="L-BFGS-B",
            jac="2-point",
            options={"maxiter": maxiter, "maxfun": maxiter * 2},
        )
        return res.fun, res.x, getattr(res, "nfev", maxiter)
    else:
        opts = {"maxiter": maxiter}
        if method == "COBYLA":
            opts["rhobeg"] = 0.5
        res = minimize(energy_fn, x0, method=method, options=opts)
        return res.fun, res.x, getattr(res, "nfev", maxiter)


def vqe_test_extended(
    entangler_qc: QuantumCircuit,
    n_qubits: int,
    H_matrix: np.ndarray,
    method: str = "COBYLA",
    n_restarts: int = 40,
    maxiter: int = 800,
    seed: int = 42,
) -> dict:
    """Run VQE with specified optimizer. Returns detailed results."""
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
    best_params = None
    all_energies = []
    convergence_trace = []
    total_evals = 0

    for _ in range(n_restarts):
        x0 = rng.uniform(-np.pi, np.pi, n_params)
        try:
            res_fun, res_x, res_nfev = _run_optimizer(
                energy, x0, method, n_params, maxiter, rng)
            total_evals += res_nfev
            all_energies.append(float(res_fun))
            if res_fun < best_energy:
                best_energy = float(res_fun)
                best_params = res_x.tolist()
            convergence_trace.append(float(best_energy))
        except Exception:
            convergence_trace.append(float(best_energy))

    return {
        "best_energy": best_energy,
        "best_params": best_params,
        "convergence_trace": convergence_trace,
        "all_energies": all_energies,
        "n_evals": total_evals,
    }


def vqe_test_ansatz(
    build_fn,
    n_qubits: int,
    n_params: int,
    H_matrix: np.ndarray,
    method: str = "COBYLA",
    n_restarts: int = 40,
    maxiter: int = 800,
    seed: int = 42,
) -> dict:
    """Run VQE with a fully-parameterized ansatz.

    build_fn(n_qubits, params) -> QuantumCircuit
    All rotation angles are variational — no fixed angles in the ansatz.
    """
    def energy(params):
        qc = build_fn(n_qubits, params)
        sv = Statevector.from_instruction(qc)
        return float(np.real(np.array(sv.data).conj() @ H_matrix @ np.array(sv.data)))

    rng = np.random.default_rng(seed)
    best_energy = float("inf")
    best_params = None
    all_energies = []
    convergence_trace = []
    total_evals = 0

    for _ in range(n_restarts):
        x0 = rng.uniform(-np.pi, np.pi, n_params)
        try:
            res_fun, res_x, res_nfev = _run_optimizer(
                energy, x0, method, n_params, maxiter, rng)
            total_evals += res_nfev
            all_energies.append(float(res_fun))
            if res_fun < best_energy:
                best_energy = float(res_fun)
                best_params = res_x.tolist()
            convergence_trace.append(float(best_energy))
        except Exception:
            convergence_trace.append(float(best_energy))

    return {
        "best_energy": best_energy,
        "best_params": best_params,
        "convergence_trace": convergence_trace,
        "all_energies": all_energies,
        "n_evals": total_evals,
    }


def _vqe_energy_at_params(
    entangler_qc: QuantumCircuit, n_qubits: int,
    H_matrix: np.ndarray, params: list[float],
) -> float:
    """Compute VQE energy at specific parameters."""
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


def _vqe_statevector_at_params(
    entangler_qc: QuantumCircuit, n_qubits: int, params: list[float],
) -> np.ndarray:
    """Get statevector at specific VQE parameters."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(params[2 * i], i)
        qc.rz(params[2 * i + 1], i)
    qc.compose(entangler_qc, inplace=True)
    for i in range(n_qubits):
        qc.ry(params[2 * n_qubits + 2 * i], i)
        qc.rz(params[2 * n_qubits + 2 * i + 1], i)
    return np.array(Statevector.from_instruction(qc).data)


# ═══════════════════════════════════════════════════════════════════════
# Hamiltonian builders for Phase 3
# ═══════════════════════════════════════════════════════════════════════


def _build_xy_hamiltonian(n: int) -> np.ndarray:
    """XY model: sum XX + YY."""
    d = 2**n
    H = np.zeros((d, d), dtype=complex)
    for i in range(n - 1):
        for p in "XY":
            label = ["I"] * n
            label[i] = p
            label[i + 1] = p
            H += _pauli_matrix("".join(label))
    return H


def _build_xxz_hamiltonian(n: int, delta: float = 0.5) -> np.ndarray:
    """Anisotropic Heisenberg (XXZ): XX + YY + delta*ZZ."""
    d = 2**n
    H = np.zeros((d, d), dtype=complex)
    for i in range(n - 1):
        for p in "XY":
            label = ["I"] * n
            label[i] = p
            label[i + 1] = p
            H += _pauli_matrix("".join(label))
        label = ["I"] * n
        label[i] = "Z"
        label[i + 1] = "Z"
        H += delta * _pauli_matrix("".join(label))
    return H


def _build_random_2local_hamiltonian(n: int, seed: int = 42) -> np.ndarray:
    """Random 2-local Hamiltonian with nearest-neighbor couplings."""
    rng = np.random.default_rng(seed)
    d = 2**n
    H = np.zeros((d, d), dtype=complex)
    for i in range(n - 1):
        for p1 in "XYZ":
            for p2 in "XYZ":
                coeff = rng.normal(0, 1)
                if abs(coeff) < 0.3:
                    continue
                label = ["I"] * n
                label[i] = p1
                label[i + 1] = p2
                H += coeff * _pauli_matrix("".join(label))
    # Single-qubit terms
    for i in range(n):
        for p in "XYZ":
            coeff = rng.normal(0, 0.5)
            if abs(coeff) < 0.2:
                continue
            label = ["I"] * n
            label[i] = p
            H += coeff * _pauli_matrix("".join(label))
    H = (H + H.conj().T) / 2
    return H


# ═══════════════════════════════════════════════════════════════════════
# Parameterized ansatz builders (all rotations are variational)
# ═══════════════════════════════════════════════════════════════════════


def _full_hea_n_params(n: int, n_layers: int = 2) -> int:
    """Number of variational parameters for full HEA ansatz."""
    return 2 * n * (n_layers + 1)  # (n_layers + 1 final) rotation layers


def _build_full_hea_ansatz(n: int, params, n_layers: int = 2) -> QuantumCircuit:
    """Full HEA: L layers of (RY(θ),RZ(θ) per qubit → CZ brick-layer) + final rotations.

    All rotation angles are independently variational.
    n_params = 2 * n * (n_layers + 1).
    """
    qc = QuantumCircuit(n)
    idx = 0
    for layer in range(n_layers):
        for i in range(n):
            qc.ry(float(params[idx]), i); idx += 1
            qc.rz(float(params[idx]), i); idx += 1
        start = layer % 2
        for i in range(start, n - 1, 2):
            qc.cz(i, i + 1)
    # Final rotation layer
    for i in range(n):
        qc.ry(float(params[idx]), i); idx += 1
        qc.rz(float(params[idx]), i); idx += 1
    return qc


def _qaoa_flex_n_params(n: int) -> int:
    """Number of variational parameters for QAOA-flex ansatz."""
    return 2 * n + (n - 1) + n + 2 * n  # init RY/RZ + gammas + betas + final RY/RZ


def _build_qaoa_flex_ansatz(n: int, params) -> QuantumCircuit:
    """QAOA-flex: RY/RZ → variational ZZ interactions → variational RX mixer → RY/RZ.

    All angles are independently variational.
    n_params = 2*n + (n-1) + n + 2*n = 6*n - 1.
    """
    qc = QuantumCircuit(n)
    idx = 0
    # Initial rotations
    for i in range(n):
        qc.ry(float(params[idx]), i); idx += 1
        qc.rz(float(params[idx]), i); idx += 1
    # ZZ interactions with variational angles
    for i in range(n - 1):
        gamma = float(params[idx]); idx += 1
        qc.cx(i, i + 1)
        qc.rz(gamma, i + 1)
        qc.cx(i, i + 1)
    # RX mixer with variational angles
    for i in range(n):
        beta = float(params[idx]); idx += 1
        qc.rx(beta, i)
    # Final rotations
    for i in range(n):
        qc.ry(float(params[idx]), i); idx += 1
        qc.rz(float(params[idx]), i); idx += 1
    return qc


def _hva_heisenberg_n_params(n: int) -> int:
    """Number of variational parameters for HVA Heisenberg ansatz."""
    return 2 * n + 3 * (n - 1) + 2 * n  # init RY/RZ + θ_XX/YY/ZZ per pair + final RY/RZ


def _build_hva_heisenberg_ansatz(n: int, params) -> QuantumCircuit:
    """HVA Heisenberg: RY/RZ → variational exp(-iθ XX)·exp(-iθ YY)·exp(-iθ ZZ) per pair → RY/RZ.

    All interaction angles and rotations are independently variational.
    n_params = 2*n + 3*(n-1) + 2*n = 7*n - 3.
    """
    qc = QuantumCircuit(n)
    idx = 0
    # Initial rotations
    for i in range(n):
        qc.ry(float(params[idx]), i); idx += 1
        qc.rz(float(params[idx]), i); idx += 1
    # Variational XX/YY/ZZ interactions per pair
    for i in range(n - 1):
        theta_xx = float(params[idx]); idx += 1
        theta_yy = float(params[idx]); idx += 1
        theta_zz = float(params[idx]); idx += 1
        # XX interaction: H⊗H → CX → RZ(θ) → CX → H⊗H
        qc.h(i); qc.h(i + 1)
        qc.cx(i, i + 1); qc.rz(theta_xx, i + 1); qc.cx(i, i + 1)
        qc.h(i); qc.h(i + 1)
        # YY interaction: Sdg⊗Sdg → H⊗H → CX → RZ(θ) → CX → H⊗H → S⊗S
        qc.sdg(i); qc.sdg(i + 1)
        qc.h(i); qc.h(i + 1)
        qc.cx(i, i + 1); qc.rz(theta_yy, i + 1); qc.cx(i, i + 1)
        qc.h(i); qc.h(i + 1)
        qc.s(i); qc.s(i + 1)
        # ZZ interaction: CX → RZ(θ) → CX
        qc.cx(i, i + 1); qc.rz(theta_zz, i + 1); qc.cx(i, i + 1)
    # Final rotations
    for i in range(n):
        qc.ry(float(params[idx]), i); idx += 1
        qc.rz(float(params[idx]), i); idx += 1
    return qc


def _build_random_entangler(
    n: int, n_2q: int, seed: int = 0,
    allowed_edges: set | None = None,
) -> QuantumCircuit:
    """Random circuit with specified gate count, respecting connectivity constraints.

    If allowed_edges is provided, 2Q gates are placed only on those edges.
    Otherwise falls back to nearest-neighbor connectivity.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    gates_1q = [("h", 0), ("s", 0), ("t", 0),
                ("rx", np.pi / 4), ("ry", np.pi / 4), ("rz", np.pi / 4)]

    if allowed_edges is None:
        # Default: nearest-neighbor (not all-to-all)
        allowed_edges = {(i, i + 1) for i in range(n - 1)}

    edge_list = list(allowed_edges)
    if not edge_list:
        return qc

    for _ in range(n_2q):
        q0, q1 = edge_list[rng.integers(len(edge_list))]
        if rng.random() < 0.5:
            qc.cx(int(q0), int(q1))
        else:
            qc.cz(int(q0), int(q1))
        for q in [q0, q1]:
            gate_name, angle = gates_1q[rng.integers(len(gates_1q))]
            if angle == 0:
                getattr(qc, gate_name)(int(q))
            else:
                getattr(qc, gate_name)(float(angle), int(q))
    return qc


# ═══════════════════════════════════════════════════════════════════════
# Concurrence helper
# ═══════════════════════════════════════════════════════════════════════


def _pairwise_concurrences(sv: np.ndarray, n: int) -> np.ndarray:
    """Compute pairwise concurrences for all qubit pairs."""
    from qiskit.quantum_info import Statevector as SV, concurrence, partial_trace as pt

    conc_matrix = np.zeros((n, n))
    sv_obj = SV(sv)
    for i in range(n):
        for j in range(i + 1, n):
            others = [q for q in range(n) if q != i and q != j]
            if others:
                rho2 = pt(sv_obj, others)
            else:
                rho2 = sv_obj.to_operator()
            try:
                c = float(concurrence(rho2))
            except Exception:
                c = 0.0
            conc_matrix[i, j] = c
            conc_matrix[j, i] = c
    return conc_matrix


# ═══════════════════════════════════════════════════════════════════════
# Phase 0: Auto-select targets & setup
# ═══════════════════════════════════════════════════════════════════════


def phase0_setup() -> tuple[list[dict], dict, dict]:
    """Load results, auto-select significant targets, rebuild templates."""
    print("Phase 0: Auto-selecting targets & rebuilding templates...")
    t0 = time.time()

    # Load comparison results
    comp_results = json.loads((COMPARISON_DIR / "results.json").read_text())
    deep_vqe = comp_results.get("deep_vqe", {})

    # Filter for significant results
    targets = []
    for key, dv in deep_vqe.items():
        sig = dv.get("significant")
        if sig is True or sig == "True" or sig == True:
            targets.append({
                "key": key,
                "candidate": dv["candidate"],
                "n_qubits": dv["n_qubits"],
                "model": dv["model"],
                "candidate_error": dv["candidate_error"],
                "cx_chain_error": dv["cx_chain_error"],
                "hea_error": dv["hea_error"],
                "improvement": dv["improvement_vs_best_baseline"],
                "n_2q_gates": dv["n_2q_gates"],
            })

    # Rank by improvement
    targets.sort(key=lambda x: -x["improvement"])
    print(f"  Found {len(targets)} significant deep VQE results")
    for t in targets:
        print(f"    {t['key']}: improvement={t['improvement']:.4f}, "
              f"error={t['candidate_error']:.4f}")

    if not targets:
        print("  WARNING: No significant results found. Using top by improvement instead.")
        ranked = sorted(deep_vqe.items(),
                        key=lambda x: x[1].get("improvement_vs_best_baseline", 0),
                        reverse=True)
        for key, dv in ranked[:3]:
            targets.append({
                "key": key,
                "candidate": dv["candidate"],
                "n_qubits": dv["n_qubits"],
                "model": dv["model"],
                "candidate_error": dv["candidate_error"],
                "cx_chain_error": dv["cx_chain_error"],
                "hea_error": dv["hea_error"],
                "improvement": dv.get("improvement_vs_best_baseline", 0),
                "n_2q_gates": dv["n_2q_gates"],
            })

    # Load discovery results for scored entries
    disc_results = json.loads(
        (SCRIPT_DIR / "output" / "discovery" / "results.json").read_text()
    )
    scored = disc_results.get("scored", {})

    # Rebuild templates
    print("  Rebuilding corpus & templates...")
    phylo_results, freq_df = load_phylogeny_results()
    survivors = phylo_results["cross_level_survival"]["survivors_at_full_reduce"]
    corpus = build_corpus()
    motifs = discover_motifs(corpus)
    templates = build_template_registry(motifs, freq_df, corpus, survivors)
    print(f"  Templates: {len(templates)}, elapsed: {time.time() - t0:.1f}s")

    return targets, scored, templates


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Verify the Result Is Real
# ═══════════════════════════════════════════════════════════════════════


def verify_circuit_identity(
    target: dict, scored: dict, templates: dict,
) -> dict:
    """1.1: Reconstruct circuit at multiple sizes, verify identity."""
    cand_name = target["candidate"]
    score_entry = scored.get(cand_name, {})
    if not score_entry:
        base = cand_name.rsplit("_q", 1)[0] if "_q" in cand_name else cand_name
        score_entry = scored.get(base, {})
    score_entry["name"] = cand_name

    results = {}
    for nq in [4, 6, 8]:
        qc = _scale_candidate_spec(score_entry, nq, templates)
        if qc is None:
            results[f"{nq}q"] = {"status": "failed_to_build"}
            continue

        gates = []
        for inst in qc.data:
            gates.append({
                "gate": inst.operation.name,
                "qubits": [qc.find_bit(q).index for q in inst.qubits],
                "params": [float(p) for p in inst.operation.params],
            })

        # Compute unitary for small circuits
        unitary_fidelity = None
        if nq <= 6:
            try:
                U = Operator(qc).data
                # Check unitarity
                I = np.eye(2**nq)
                unitarity_err = float(np.linalg.norm(U @ U.conj().T - I))
                unitary_fidelity = 1.0 - unitarity_err
            except Exception:
                unitary_fidelity = None

        results[f"{nq}q"] = {
            "n_gates": qc.size(),
            "n_2q_gates": _count_2q(qc),
            "depth": qc.depth(),
            "gate_sequence": gates[:20],  # first 20 for comparison
            "unitary_fidelity": unitary_fidelity,
        }

    return results


def test_optimizer_sensitivity(
    target: dict, scored: dict, templates: dict,
) -> dict:
    """1.2: Test with 5 optimizers × 3 circuits."""
    nq = target["n_qubits"]
    model = target["model"]
    H = _build_hamiltonian_matrix(nq, model)
    evals = np.linalg.eigvalsh(H)
    exact_gs = float(evals[0])

    cand_name = target["candidate"]
    score_entry = scored.get(cand_name, {})
    if not score_entry:
        base = cand_name.rsplit("_q", 1)[0] if "_q" in cand_name else cand_name
        score_entry = scored.get(base, {})
    score_entry["name"] = cand_name

    cand_qc = _scale_candidate_spec(score_entry, nq, templates)
    if cand_qc is None:
        return {"error": "failed to build candidate"}

    n_2q = _count_2q(cand_qc)
    cx_qc = _cx_chain_entangler(nq, max(1, n_2q))
    hea_qc = _hea_entangler(nq, max(1, n_2q))

    circuits = {"candidate": cand_qc, "cx_chain": cx_qc, "hea": hea_qc}
    optimizers = ["COBYLA", "L-BFGS-B", "Nelder-Mead", "Powell", "SPSA"]

    results = {}
    total = len(circuits) * len(optimizers)
    done = 0
    for opt in optimizers:
        for cname, qc in circuits.items():
            done += 1
            print(f"    [{done}/{total}] {opt} × {cname}")
            vqe = vqe_test_extended(qc, nq, H, method=opt, n_restarts=40, maxiter=800)
            err = abs(vqe["best_energy"] - exact_gs) / abs(exact_gs) if exact_gs != 0 else 0
            results[f"{opt}_{cname}"] = {
                "optimizer": opt,
                "circuit": cname,
                "best_energy": vqe["best_energy"],
                "error": err,
                "n_evals": vqe["n_evals"],
            }

    # Check if candidate wins across all optimizers
    candidate_wins = {}
    for opt in optimizers:
        cand_err = results[f"{opt}_candidate"]["error"]
        best_base = min(results[f"{opt}_cx_chain"]["error"], results[f"{opt}_hea"]["error"])
        candidate_wins[opt] = cand_err < best_base

    results["candidate_wins_all"] = all(candidate_wins.values())
    results["wins_by_optimizer"] = candidate_wins
    return results


def test_statistical_significance(
    target: dict, scored: dict, templates: dict,
) -> dict:
    """1.3: 20 independent trials with different seeds."""
    nq = target["n_qubits"]
    model = target["model"]
    H = _build_hamiltonian_matrix(nq, model)
    evals = np.linalg.eigvalsh(H)
    exact_gs = float(evals[0])

    cand_name = target["candidate"]
    score_entry = scored.get(cand_name, {})
    if not score_entry:
        base = cand_name.rsplit("_q", 1)[0] if "_q" in cand_name else cand_name
        score_entry = scored.get(base, {})
    score_entry["name"] = cand_name

    cand_qc = _scale_candidate_spec(score_entry, nq, templates)
    if cand_qc is None:
        return {"error": "failed to build candidate"}

    n_2q = _count_2q(cand_qc)
    cx_qc = _cx_chain_entangler(nq, max(1, n_2q))

    cand_errors = []
    base_errors = []
    n_trials = 20
    for trial in range(n_trials):
        seed = 1000 + trial * 137
        print(f"    trial {trial + 1}/{n_trials}")
        cand_res = vqe_test_extended(cand_qc, nq, H, n_restarts=10, maxiter=400, seed=seed)
        base_res = vqe_test_extended(cx_qc, nq, H, n_restarts=10, maxiter=400, seed=seed)
        cand_errors.append(abs(cand_res["best_energy"] - exact_gs) / abs(exact_gs))
        base_errors.append(abs(base_res["best_energy"] - exact_gs) / abs(exact_gs))

    cand_arr = np.array(cand_errors)
    base_arr = np.array(base_errors)

    # Welch's t-test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(cand_arr, base_arr, equal_var=False)

    # Cohen's d
    pooled_std = np.sqrt((np.var(cand_arr) + np.var(base_arr)) / 2)
    cohens_d = (np.mean(base_arr) - np.mean(cand_arr)) / pooled_std if pooled_std > 0 else 0

    # 95% CI for the difference
    diff = base_arr - cand_arr
    ci_low = float(np.percentile(diff, 2.5))
    ci_high = float(np.percentile(diff, 97.5))

    return {
        "n_trials": n_trials,
        "candidate_errors": cand_arr.tolist(),
        "baseline_errors": base_arr.tolist(),
        "candidate_mean": float(np.mean(cand_arr)),
        "candidate_std": float(np.std(cand_arr)),
        "baseline_mean": float(np.mean(base_arr)),
        "baseline_std": float(np.std(base_arr)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "significant_at_005": p_value < 0.05,
    }


def verify_reproducibility(
    target: dict, scored: dict, templates: dict,
    best_params_from_1_2: list[float] | None = None,
) -> dict:
    """1.4: Verify best params reproduce, test seed consistency."""
    nq = target["n_qubits"]
    model = target["model"]
    H = _build_hamiltonian_matrix(nq, model)
    evals = np.linalg.eigvalsh(H)
    exact_gs = float(evals[0])

    cand_name = target["candidate"]
    score_entry = scored.get(cand_name, {})
    if not score_entry:
        base = cand_name.rsplit("_q", 1)[0] if "_q" in cand_name else cand_name
        score_entry = scored.get(base, {})
    score_entry["name"] = cand_name

    cand_qc = _scale_candidate_spec(score_entry, nq, templates)
    if cand_qc is None:
        return {"error": "failed to build candidate"}

    # Verify best params if available
    param_verification = None
    if best_params_from_1_2 is not None:
        e_direct = _vqe_energy_at_params(cand_qc, nq, H, best_params_from_1_2)
        # Also verify via statevector
        sv = _vqe_statevector_at_params(cand_qc, nq, best_params_from_1_2)
        e_sv = float(np.real(sv.conj() @ H @ sv))
        param_verification = {
            "energy_direct": e_direct,
            "energy_statevector": e_sv,
            "match": abs(e_direct - e_sv) < 1e-10,
            "error": abs(e_direct - exact_gs) / abs(exact_gs),
        }

    # Seed consistency: 5 seeds × 40 restarts
    seed_results = []
    for seed in [42, 123, 456, 789, 2024]:
        print(f"    seed={seed}")
        res = vqe_test_extended(cand_qc, nq, H, n_restarts=40, maxiter=800, seed=seed)
        err = abs(res["best_energy"] - exact_gs) / abs(exact_gs)
        seed_results.append({"seed": seed, "error": err, "best_energy": res["best_energy"]})

    errors = [r["error"] for r in seed_results]
    return {
        "param_verification": param_verification,
        "seed_results": seed_results,
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "max_error": float(np.max(errors)),
        "min_error": float(np.min(errors)),
        "consistent": float(np.std(errors)) < 0.05,
    }


def test_full_hea_baseline(
    target: dict, scored: dict, templates: dict,
) -> dict:
    """1.5: Test with proper full HEA baseline (all rotations variational)."""
    nq = target["n_qubits"]
    model = target["model"]
    H = _build_hamiltonian_matrix(nq, model)
    evals = np.linalg.eigvalsh(H)
    exact_gs = float(evals[0])

    cand_name = target["candidate"]
    score_entry = scored.get(cand_name, {})
    if not score_entry:
        base = cand_name.rsplit("_q", 1)[0] if "_q" in cand_name else cand_name
        score_entry = scored.get(base, {})
    score_entry["name"] = cand_name

    cand_qc = _scale_candidate_spec(score_entry, nq, templates)
    if cand_qc is None:
        return {"error": "failed to build candidate"}

    n_hea_params = _full_hea_n_params(nq, n_layers=2)

    print("    candidate...")
    cand_res = vqe_test_extended(cand_qc, nq, H, n_restarts=40, maxiter=800)
    cand_err = abs(cand_res["best_energy"] - exact_gs) / abs(exact_gs)

    print(f"    full HEA ({n_hea_params} params)...")
    hea_res = vqe_test_ansatz(
        lambda n, p: _build_full_hea_ansatz(n, p, n_layers=2),
        nq, n_hea_params, H, n_restarts=40, maxiter=800)
    hea_err = abs(hea_res["best_energy"] - exact_gs) / abs(exact_gs)

    return {
        "candidate_error": cand_err,
        "full_hea_error": hea_err,
        "candidate_beats_full_hea": cand_err < hea_err,
        "improvement": hea_err - cand_err,
        "candidate_n_params": 4 * nq,
        "hea_n_params": n_hea_params,
        "candidate_2q_gates": _count_2q(cand_qc),
    }


def run_phase1(target: dict, scored: dict, templates: dict, ckpt: dict) -> dict:
    """Run all Phase 1 sub-tests."""
    tk = target["key"]
    results = {}

    if not _phase_done(ckpt, tk, "1.1_circuit_identity"):
        print("  1.1 Verify circuit identity...")
        r = verify_circuit_identity(target, scored, templates)
        _save_phase(ckpt, tk, "1.1_circuit_identity", r)
        results["circuit_identity"] = r
    else:
        results["circuit_identity"] = ckpt[tk]["1.1_circuit_identity"]
        print("  1.1 (cached)")

    if not _phase_done(ckpt, tk, "1.2_optimizer_sensitivity"):
        print("  1.2 Optimizer sensitivity...")
        r = test_optimizer_sensitivity(target, scored, templates)
        _save_phase(ckpt, tk, "1.2_optimizer_sensitivity", r)
        results["optimizer_sensitivity"] = r
    else:
        results["optimizer_sensitivity"] = ckpt[tk]["1.2_optimizer_sensitivity"]
        print("  1.2 (cached)")

    if not _phase_done(ckpt, tk, "1.3_statistical_significance"):
        print("  1.3 Statistical significance...")
        r = test_statistical_significance(target, scored, templates)
        _save_phase(ckpt, tk, "1.3_statistical_significance", r)
        results["statistical_significance"] = r
    else:
        results["statistical_significance"] = ckpt[tk]["1.3_statistical_significance"]
        print("  1.3 (cached)")

    # Get best params from 1.2 for reproducibility check
    opt_data = results["optimizer_sensitivity"]
    best_params = None
    if "COBYLA_candidate" in opt_data:
        # Find the optimizer result that has stored best_params
        # We need to re-run a quick VQE to get params since 1.2 only stored errors
        pass  # params not saved in 1.2 summary — reproducibility will run fresh

    if not _phase_done(ckpt, tk, "1.4_reproducibility"):
        print("  1.4 Reproducibility...")
        r = verify_reproducibility(target, scored, templates, best_params)
        _save_phase(ckpt, tk, "1.4_reproducibility", r)
        results["reproducibility"] = r
    else:
        results["reproducibility"] = ckpt[tk]["1.4_reproducibility"]
        print("  1.4 (cached)")

    if not _phase_done(ckpt, tk, "1.5_full_hea_baseline"):
        print("  1.5 Full HEA baseline...")
        r = test_full_hea_baseline(target, scored, templates)
        _save_phase(ckpt, tk, "1.5_full_hea_baseline", r)
        results["full_hea_baseline"] = r
    else:
        results["full_hea_baseline"] = ckpt[tk]["1.5_full_hea_baseline"]
        print("  1.5 (cached)")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Understand Why It Works
# ═══════════════════════════════════════════════════════════════════════


def _get_candidate_qc(target, scored, templates):
    """Helper to reconstruct candidate circuit."""
    cand_name = target["candidate"]
    score_entry = scored.get(cand_name, {})
    if not score_entry:
        base = cand_name.rsplit("_q", 1)[0] if "_q" in cand_name else cand_name
        score_entry = scored.get(base, {})
    score_entry["name"] = cand_name
    nq = target["n_qubits"]
    return _scale_candidate_spec(score_entry, nq, templates), score_entry


def extract_circuit_structure(qc: QuantumCircuit) -> dict:
    """2.1: Parse circuit, build connectivity graph, identify gate pattern."""
    n = qc.num_qubits
    connectivity = np.zeros((n, n), dtype=int)
    gate_list = []
    for inst in qc.data:
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        gate_list.append({
            "gate": inst.operation.name,
            "qubits": qubits,
            "params": [float(p) for p in inst.operation.params],
        })
        if len(qubits) == 2:
            connectivity[qubits[0], qubits[1]] += 1
            connectivity[qubits[1], qubits[0]] += 1

    n_edges = np.sum(connectivity > 0) // 2
    max_possible = n * (n - 1) // 2
    if n_edges == 0:
        pattern = "none"
    elif n_edges == n - 1:
        degrees = np.sum(connectivity > 0, axis=1)
        if max(degrees) <= 2:
            pattern = "linear"
        elif max(degrees) == n - 1:
            pattern = "star"
        else:
            pattern = "tree"
    elif n_edges == max_possible:
        pattern = "all-to-all"
    elif n_edges >= max_possible * 0.7:
        pattern = "dense"
    else:
        pattern = "sparse"

    return {
        "n_qubits": n,
        "n_gates": qc.size(),
        "n_1q_gates": _count_1q(qc),
        "n_2q_gates": _count_2q(qc),
        "depth": qc.depth(),
        "connectivity": connectivity.tolist(),
        "pattern": pattern,
        "n_edges": int(n_edges),
        "gates": gate_list,
    }


def analyze_entangler_hamiltonian(qc: QuantumCircuit, n: int) -> dict:
    """2.2: Unitary -> eigendecomp -> H -> Pauli decompose."""
    if n > 6:
        return {"error": "too many qubits for unitary"}
    try:
        U = Operator(qc).data
    except Exception:
        return {"error": "unitary computation failed"}
    try:
        eigenvalues, V = np.linalg.eig(U)
        eigenvalues = eigenvalues / np.abs(eigenvalues)
        phases = np.angle(eigenvalues)
        H = V @ np.diag(-phases) @ np.linalg.inv(V)
        H = (H + H.conj().T) / 2
    except Exception:
        return {"error": "eigendecomposition failed"}

    coeffs = pauli_decompose(H, n)
    if not coeffs:
        return {"error": "trivial Hamiltonian"}
    coeffs.pop("I" * n, None)

    sorted_terms = sorted(coeffs.items(), key=lambda x: -abs(x[1]))
    top_20 = sorted_terms[:20]

    n_body = {}
    for label, c in coeffs.items():
        weight = sum(1 for ch in label if ch != "I")
        n_body[weight] = n_body.get(weight, 0) + abs(c)

    # Cosine similarity with Heisenberg
    def _label(n, pairs):
        l = ["I"] * n
        for q, p in pairs:
            l[q] = p
        return "".join(l)

    heisenberg_coeffs = {}
    for i in range(n - 1):
        for p in "XYZ":
            heisenberg_coeffs[_label(n, [(i, p), (i + 1, p)])] = 1.0

    all_keys = set(coeffs) | set(heisenberg_coeffs)
    va = np.array([coeffs.get(k, 0) for k in all_keys])
    vb = np.array([heisenberg_coeffs.get(k, 0) for k in all_keys])
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    cosine_sim = float(np.dot(va, vb) / (na * nb)) if na > 1e-10 and nb > 1e-10 else 0

    xyz_weight = sum(abs(coeffs.get(_label(n, [(i, p), (i + 1, p)]), 0))
                     for i in range(n - 1) for p in "XYZ")
    total_weight = sum(abs(c) for c in coeffs.values())
    xyz_ratio = xyz_weight / total_weight if total_weight > 0 else 0

    return {
        "top_20_terms": [(l, float(c)) for l, c in top_20],
        "n_terms": len(coeffs),
        "n_body_distribution": {int(k): float(v) for k, v in n_body.items()},
        "heisenberg_cosine_sim": cosine_sim,
        "xyz_ratio": xyz_ratio,
    }


def analyze_entanglement_structure(
    target: dict, scored: dict, templates: dict,
) -> dict:
    """2.3: Compare entanglement patterns of VQE state vs ground state."""
    nq = target["n_qubits"]
    model = target["model"]
    H = _build_hamiltonian_matrix(nq, model)
    evals_arr, evecs = np.linalg.eigh(H)
    gs_state = evecs[:, 0]

    cand_qc, _ = _get_candidate_qc(target, scored, templates)
    if cand_qc is None:
        return {"error": "failed to build candidate"}

    res = vqe_test_extended(cand_qc, nq, H, n_restarts=40, maxiter=800)
    best_params = res["best_params"]
    if best_params is None:
        return {"error": "no best params available"}

    vqe_state = _vqe_statevector_at_params(cand_qc, nq, best_params)

    gs_conc = _pairwise_concurrences(gs_state, nq)
    vqe_conc = _pairwise_concurrences(vqe_state, nq)
    gs_entropy = _half_cut_entropy(gs_state, nq)
    vqe_entropy = _half_cut_entropy(vqe_state, nq)

    # Random input entropies
    rng = np.random.default_rng(42)
    random_entropies = []
    if nq <= 6:
        try:
            U = Operator(cand_qc).data
            for _ in range(50):
                psi = _random_product_state(nq, rng)
                out = U @ psi
                random_entropies.append(_half_cut_entropy(out, nq))
        except Exception:
            pass

    # Correlation of concurrence patterns
    triu_idx = np.triu_indices(nq, 1)
    gs_vals = gs_conc[triu_idx]
    vqe_vals = vqe_conc[triu_idx]
    corr = float(np.corrcoef(gs_vals, vqe_vals)[0, 1]) if len(gs_vals) > 1 else 0

    return {
        "gs_concurrences": gs_conc.tolist(),
        "vqe_concurrences": vqe_conc.tolist(),
        "gs_half_cut_entropy": gs_entropy,
        "vqe_half_cut_entropy": vqe_entropy,
        "concurrence_correlation": corr,
        "random_entropy_mean": float(np.mean(random_entropies)) if random_entropies else 0,
        "random_entropy_std": float(np.std(random_entropies)) if random_entropies else 0,
    }


def analyze_parameter_landscape(
    target: dict, scored: dict, templates: dict,
) -> dict:
    """2.4: 1D slices through the energy landscape."""
    nq = target["n_qubits"]
    model = target["model"]
    H = _build_hamiltonian_matrix(nq, model)
    evals_arr = np.linalg.eigvalsh(H)
    exact_gs = float(evals_arr[0])

    cand_qc, _ = _get_candidate_qc(target, scored, templates)
    if cand_qc is None:
        return {"error": "failed to build candidate"}

    res = vqe_test_extended(cand_qc, nq, H, n_restarts=40, maxiter=800)
    opt_params = res["best_params"]
    if opt_params is None:
        return {"error": "no optimal params"}

    n_params = len(opt_params)
    n_sweep = 50
    thetas = np.linspace(-np.pi, np.pi, n_sweep)

    slices = {}
    for pidx in range(min(n_params, 24)):
        energies = []
        for theta in thetas:
            p = list(opt_params)
            p[pidx] = theta
            e = _vqe_energy_at_params(cand_qc, nq, H, p)
            energies.append(e)
        energies = np.array(energies)
        n_minima = 0
        for i in range(1, len(energies) - 1):
            if energies[i] < energies[i - 1] and energies[i] < energies[i + 1]:
                n_minima += 1
        opt_e = min(energies)
        threshold = opt_e + 0.1 * (max(energies) - opt_e)
        basin_pts = np.sum(energies < threshold)
        basin_width = basin_pts / n_sweep * 2 * np.pi

        slices[f"param_{pidx}"] = {
            "n_minima": n_minima,
            "basin_width": float(basin_width),
            "energy_range": float(max(energies) - min(energies)),
            "energies": energies.tolist(),
        }

    return {
        "n_params": n_params,
        "slices": slices,
        "optimal_energy": res["best_energy"],
        "optimal_error": abs(res["best_energy"] - exact_gs) / abs(exact_gs),
    }


def compute_gradient_variance(
    target: dict, scored: dict, templates: dict,
) -> dict:
    """2.5: Gradient variance across random parameter points."""
    nq = target["n_qubits"]
    model = target["model"]
    H = _build_hamiltonian_matrix(nq, model)

    cand_qc, _ = _get_candidate_qc(target, scored, templates)
    if cand_qc is None:
        return {"error": "failed to build candidate"}

    n_2q = _count_2q(cand_qc)
    circuits = {
        "candidate": cand_qc,
        "cx_chain": _cx_chain_entangler(nq, max(1, n_2q)),
        "hea": _hea_entangler(nq, max(1, n_2q)),
    }

    results = {}
    n_points = 100
    n_params = 4 * nq
    eps = 1e-4

    for cname, qc in circuits.items():
        rng = np.random.default_rng(42)
        all_grads = []
        for _ in range(n_points):
            params = rng.uniform(-np.pi, np.pi, n_params).tolist()
            grads = []
            for pidx in range(n_params):
                p_plus = list(params)
                p_plus[pidx] += eps
                p_minus = list(params)
                p_minus[pidx] -= eps
                grad = (_vqe_energy_at_params(qc, nq, H, p_plus)
                        - _vqe_energy_at_params(qc, nq, H, p_minus)) / (2 * eps)
                grads.append(grad)
            all_grads.append(grads)
        grads_arr = np.array(all_grads)
        var_per_param = np.var(grads_arr, axis=0)
        results[cname] = {
            "mean_variance": float(np.mean(var_per_param)),
            "min_variance": float(np.min(var_per_param)),
            "max_variance": float(np.max(var_per_param)),
            "median_variance": float(np.median(var_per_param)),
        }

    return results


def compute_expressibility(
    target: dict, scored: dict, templates: dict,
) -> dict:
    """2.6: Expressibility via fidelity distribution vs Haar-random."""
    nq = target["n_qubits"]
    cand_qc, _ = _get_candidate_qc(target, scored, templates)
    if cand_qc is None:
        return {"error": "failed to build candidate"}

    n_samples = 500
    n_params = 4 * nq
    rng = np.random.default_rng(42)

    states = []
    for _ in range(n_samples):
        params = rng.uniform(-np.pi, np.pi, n_params).tolist()
        sv = _vqe_statevector_at_params(cand_qc, nq, params)
        states.append(sv)

    fidelities = []
    n_pairs = min(2000, n_samples * (n_samples - 1) // 2)
    for _ in range(n_pairs):
        i, j = rng.choice(n_samples, 2, replace=False)
        fid = abs(np.dot(states[i].conj(), states[j])) ** 2
        fidelities.append(fid)

    fid_arr = np.array(fidelities)
    d = 2**nq

    n_bins = 50
    hist_circuit, bin_edges = np.histogram(fid_arr, bins=n_bins, range=(0, 1), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    haar_pdf = np.array([(d - 1) * (1 - f) ** (d - 2) if f < 1 else 0 for f in bin_centers])

    hist_circuit = hist_circuit / (np.sum(hist_circuit) + 1e-12)
    haar_pdf = haar_pdf / (np.sum(haar_pdf) + 1e-12)

    kl = 0.0
    for p, q in zip(hist_circuit, haar_pdf):
        if p > 1e-12 and q > 1e-12:
            kl += p * np.log(p / q)

    return {
        "kl_divergence": float(kl),
        "mean_fidelity": float(np.mean(fid_arr)),
        "std_fidelity": float(np.std(fid_arr)),
        "n_samples": n_samples,
        "n_pairs": n_pairs,
    }


def run_phase2(target: dict, scored: dict, templates: dict, ckpt: dict) -> dict:
    """Run all Phase 2 sub-tests."""
    tk = target["key"]
    results = {}

    cand_qc, _ = _get_candidate_qc(target, scored, templates)
    nq = target["n_qubits"]

    phases = [
        ("2.1_circuit_structure", "Circuit structure",
         lambda: extract_circuit_structure(cand_qc) if cand_qc else {"error": "no circuit"}),
        ("2.2_entangler_hamiltonian", "Entangler Hamiltonian",
         lambda: analyze_entangler_hamiltonian(cand_qc, nq) if cand_qc else {"error": "no circuit"}),
        ("2.3_entanglement_structure", "Entanglement structure",
         lambda: analyze_entanglement_structure(target, scored, templates)),
        ("2.4_parameter_landscape", "Parameter landscape",
         lambda: analyze_parameter_landscape(target, scored, templates)),
        ("2.5_gradient_variance", "Gradient variance",
         lambda: compute_gradient_variance(target, scored, templates)),
        ("2.6_expressibility", "Expressibility",
         lambda: compute_expressibility(target, scored, templates)),
    ]

    for phase_key, label, func in phases:
        short_key = phase_key.split("_", 1)[1]
        if not _phase_done(ckpt, tk, phase_key):
            print(f"  {phase_key[:3]} {label}...")
            r = func()
            _save_phase(ckpt, tk, phase_key, r)
            results[short_key] = r
        else:
            results[short_key] = ckpt[tk][phase_key]
            print(f"  {phase_key[:3]} (cached)")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Characterize the Limits
# ═══════════════════════════════════════════════════════════════════════


def qubit_scaling_study(target: dict, scored: dict, templates: dict) -> dict:
    """3.1: Test at 4-8 qubits on Heisenberg and TFIM."""
    _, score_entry = _get_candidate_qc(target, scored, templates)

    results = {}
    for nq in [4, 5, 6, 7, 8]:
        cand_qc = _scale_candidate_spec(score_entry, nq, templates)
        if cand_qc is None or _count_2q(cand_qc) == 0:
            results[f"{nq}q"] = {"error": "failed to build or no 2q gates"}
            continue

        n_2q = _count_2q(cand_qc)
        n_hea_params = _full_hea_n_params(nq, n_layers=2)

        for model in ["heisenberg", "tfim"]:
            H = _build_hamiltonian_matrix(nq, model)
            exact_gs = float(np.linalg.eigvalsh(H)[0])
            if exact_gs == 0:
                continue
            print(f"    {nq}q {model}...")
            cand_res = vqe_test_extended(cand_qc, nq, H, n_restarts=20, maxiter=400)
            cx_res = vqe_test_extended(
                _cx_chain_entangler(nq, max(1, n_2q)), nq, H, n_restarts=20, maxiter=400)
            hea_res = vqe_test_ansatz(
                lambda n, p: _build_full_hea_ansatz(n, p, n_layers=2),
                nq, n_hea_params, H, n_restarts=20, maxiter=400)

            cand_err = abs(cand_res["best_energy"] - exact_gs) / abs(exact_gs)
            cx_err = abs(cx_res["best_energy"] - exact_gs) / abs(exact_gs)
            hea_err = abs(hea_res["best_energy"] - exact_gs) / abs(exact_gs)
            results[f"{nq}q_{model}"] = {
                "candidate_error": cand_err,
                "cx_chain_error": cx_err,
                "hea_error": hea_err,
                "candidate_beats_both": cand_err < cx_err and cand_err < hea_err,
            }

    return results


def hamiltonian_generality_study(target: dict, scored: dict, templates: dict) -> dict:
    """3.2: Test across different Hamiltonians at 6q."""
    nq = 6
    _, score_entry = _get_candidate_qc(target, scored, templates)
    cand_qc = _scale_candidate_spec(score_entry, nq, templates)
    if cand_qc is None:
        return {"error": "failed to build candidate"}

    n_2q = _count_2q(cand_qc)
    cx_qc = _cx_chain_entangler(nq, max(1, n_2q))
    n_hea_params = _full_hea_n_params(nq, n_layers=2)

    hamiltonians = {
        "heisenberg": _build_hamiltonian_matrix(nq, "heisenberg"),
        "tfim": _build_hamiltonian_matrix(nq, "tfim"),
        "xy": _build_xy_hamiltonian(nq),
        "xxz": _build_xxz_hamiltonian(nq, delta=0.5),
        "random_2local": _build_random_2local_hamiltonian(nq),
    }

    results = {}
    for hname, H in hamiltonians.items():
        exact_gs = float(np.linalg.eigvalsh(H)[0])
        if exact_gs == 0:
            continue
        print(f"    {hname}...")
        cand_res = vqe_test_extended(cand_qc, nq, H, n_restarts=20, maxiter=400)
        cx_res = vqe_test_extended(cx_qc, nq, H, n_restarts=20, maxiter=400)
        hea_res = vqe_test_ansatz(
            lambda n, p: _build_full_hea_ansatz(n, p, n_layers=2),
            nq, n_hea_params, H, n_restarts=20, maxiter=400)
        results[hname] = {
            "exact_gs": exact_gs,
            "candidate_error": abs(cand_res["best_energy"] - exact_gs) / abs(exact_gs),
            "cx_chain_error": abs(cx_res["best_energy"] - exact_gs) / abs(exact_gs),
            "hea_error": abs(hea_res["best_energy"] - exact_gs) / abs(exact_gs),
        }

    return results


def layer_depth_study(target: dict, scored: dict, templates: dict) -> dict:
    """3.3: Modified ansatz with 1-4 rotation layers."""
    nq = target["n_qubits"]
    model = target["model"]
    H = _build_hamiltonian_matrix(nq, model)
    exact_gs = float(np.linalg.eigvalsh(H)[0])

    cand_qc, _ = _get_candidate_qc(target, scored, templates)
    if cand_qc is None:
        return {"error": "failed to build candidate"}

    from scipy.optimize import minimize as sp_minimize

    results = {}
    for n_layers in [1, 2, 3, 4]:
        n_params = 2 * nq * (n_layers + 1)

        def energy_ml(params, _qc=cand_qc, _nq=nq, _H=H, _nl=n_layers):
            qc = QuantumCircuit(_nq)
            idx = 0
            for layer in range(_nl):
                for i in range(_nq):
                    qc.ry(params[idx], i); idx += 1
                    qc.rz(params[idx], i); idx += 1
                qc.compose(_qc, inplace=True)
            for i in range(_nq):
                qc.ry(params[idx], i); idx += 1
                qc.rz(params[idx], i); idx += 1
            sv = Statevector.from_instruction(qc)
            return float(np.real(np.array(sv.data).conj() @ _H @ np.array(sv.data)))

        rng = np.random.default_rng(42)
        best_e = float("inf")
        print(f"    {n_layers} layers...")
        for _ in range(20):
            x0 = rng.uniform(-np.pi, np.pi, n_params)
            try:
                res = sp_minimize(energy_ml, x0, method="COBYLA",
                                  options={"maxiter": 400, "rhobeg": 0.5})
                if res.fun < best_e:
                    best_e = res.fun
            except Exception:
                pass
        results[f"{n_layers}_layers"] = {
            "n_params": n_params,
            "best_energy": best_e,
            "error": abs(best_e - exact_gs) / abs(exact_gs) if exact_gs != 0 else 0,
        }

    return results


def noise_resilience_study(target: dict, scored: dict, templates: dict) -> dict:
    """3.4: Analytical noise resilience at VQE-optimal params."""
    nq = target["n_qubits"]
    model = target["model"]
    H = _build_hamiltonian_matrix(nq, model)
    exact_gs = float(np.linalg.eigvalsh(H)[0])
    d = 2**nq
    trace_H = float(np.real(np.trace(H))) / d

    cand_qc, _ = _get_candidate_qc(target, scored, templates)
    if cand_qc is None:
        return {"error": "failed to build candidate"}

    n_2q = _count_2q(cand_qc)
    n_1q = _count_1q(cand_qc)

    res = vqe_test_extended(cand_qc, nq, H, n_restarts=40, maxiter=800)
    e_ideal = res["best_energy"]

    noise_rates = [0, 0.001, 0.005, 0.01, 0.02, 0.05]
    noise_results = {}
    for p in noise_rates:
        fid = ((1 - p) ** n_2q) * ((1 - p / 10) ** n_1q) if p > 0 else 1.0
        e_noisy = fid * e_ideal + (1 - fid) * trace_H
        err = abs(e_noisy - exact_gs) / abs(exact_gs) if exact_gs != 0 else 0
        noise_results[str(p)] = {"fidelity": fid, "noisy_energy": e_noisy, "error": err}

    noise_results["n_1q_gates"] = n_1q
    noise_results["n_2q_gates"] = n_2q
    noise_results["ideal_energy"] = e_ideal
    return noise_results


def topology_analysis(target: dict, scored: dict, templates: dict) -> dict:
    """3.5: SWAP count for different hardware topologies."""
    nq = target["n_qubits"]
    cand_qc, _ = _get_candidate_qc(target, scored, templates)
    if cand_qc is None:
        return {"error": "failed to build candidate"}

    edges = set()
    for inst in cand_qc.data:
        if inst.operation.num_qubits >= 2:
            qubits = tuple(cand_qc.find_bit(q).index for q in inst.qubits)
            edges.add((min(qubits), max(qubits)))

    linear = {(i, i + 1) for i in range(nq - 1)}
    ring = linear | {(0, nq - 1)}
    heavy_hex = set(linear)
    for i in range(0, nq - 2, 2):
        if i + 2 < nq:
            heavy_hex.add((i, i + 2))

    def _swaps(edges, topo):
        return sum(1 for e in edges if e not in topo)

    return {
        "circuit_edges": [list(e) for e in edges],
        "n_unique_edges": len(edges),
        "swaps_linear": _swaps(edges, linear),
        "swaps_ring": _swaps(edges, ring),
        "swaps_heavy_hex": _swaps(edges, heavy_hex),
        "note": "SWAP counts are approximate lower bounds (1 per non-native edge). "
                "Real routing may require more SWAPs depending on graph distance.",
    }


def run_phase3(target: dict, scored: dict, templates: dict, ckpt: dict) -> dict:
    """Run all Phase 3 sub-tests."""
    tk = target["key"]
    results = {}
    phases = [
        ("3.1_qubit_scaling", "qubit_scaling",
         lambda: qubit_scaling_study(target, scored, templates)),
        ("3.2_hamiltonian_generality", "hamiltonian_generality",
         lambda: hamiltonian_generality_study(target, scored, templates)),
        ("3.3_layer_depth", "layer_depth",
         lambda: layer_depth_study(target, scored, templates)),
        ("3.4_noise_resilience", "noise_resilience",
         lambda: noise_resilience_study(target, scored, templates)),
        ("3.5_topology", "topology",
         lambda: topology_analysis(target, scored, templates)),
    ]
    for phase_key, result_key, func in phases:
        if not _phase_done(ckpt, tk, phase_key):
            print(f"  {phase_key[:3]} {result_key.replace('_', ' ').title()}...")
            r = func()
            _save_phase(ckpt, tk, phase_key, r)
            results[result_key] = r
        else:
            results[result_key] = ckpt[tk][phase_key]
            print(f"  {phase_key[:3]} (cached)")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Stronger Baselines
# ═══════════════════════════════════════════════════════════════════════


def _extract_candidate_edges(qc: QuantumCircuit) -> set:
    """Extract the set of qubit-pair edges used by 2Q gates in the circuit."""
    edges = set()
    for inst in qc.data:
        if inst.operation.num_qubits >= 2:
            qubits = tuple(qc.find_bit(q).index for q in inst.qubits)
            edges.add((min(qubits), max(qubits)))
    return edges


def run_phase4(target: dict, scored: dict, templates: dict, ckpt: dict) -> dict:
    """Test against stronger baselines (all with variational parameters)."""
    tk = target["key"]
    if _phase_done(ckpt, tk, "phase4"):
        print("  Phase 4 (cached)")
        return ckpt[tk]["phase4"]

    nq = target["n_qubits"]
    model = target["model"]
    H = _build_hamiltonian_matrix(nq, model)
    exact_gs = float(np.linalg.eigvalsh(H)[0])

    cand_qc, _ = _get_candidate_qc(target, scored, templates)
    if cand_qc is None:
        r = {"error": "failed to build candidate"}
        _save_phase(ckpt, tk, "phase4", r)
        return r

    n_2q = _count_2q(cand_qc)
    cand_edges = _extract_candidate_edges(cand_qc)

    # Fixed-structure baselines (used with vqe_test_extended wrapper)
    fixed_baselines = {
        "candidate": cand_qc,
        "cx_chain": _cx_chain_entangler(nq, max(1, n_2q)),
        "hea_cz_brick": _hea_entangler(nq, max(1, n_2q)),
    }

    # Shuffled candidate: same gates, random qubit permutation
    rng = np.random.default_rng(42)
    perm = rng.permutation(nq).tolist()
    shuffled_qc = QuantumCircuit(nq)
    for inst in cand_qc.data:
        qubits = [perm[cand_qc.find_bit(q).index] for q in inst.qubits]
        if inst.operation.num_qubits == 1:
            shuffled_qc.append(inst.operation, [qubits[0]])
        elif inst.operation.num_qubits == 2:
            shuffled_qc.append(inst.operation, qubits[:2])
    fixed_baselines["shuffled_candidate"] = shuffled_qc

    # Fully-parameterized ansatze (used with vqe_test_ansatz)
    param_baselines = {
        "full_hea_2L": (
            lambda n, p: _build_full_hea_ansatz(n, p, n_layers=2),
            _full_hea_n_params(nq, n_layers=2),
        ),
        "qaoa_flex": (
            _build_qaoa_flex_ansatz,
            _qaoa_flex_n_params(nq),
        ),
        "hva_heisenberg": (
            _build_hva_heisenberg_ansatz,
            _hva_heisenberg_n_params(nq),
        ),
    }

    results = {}
    total = len(fixed_baselines) + len(param_baselines) + 20
    done = 0

    # Run fixed-structure baselines (with standard 2-layer RY/RZ wrapper)
    for bname, bqc in fixed_baselines.items():
        done += 1
        print(f"    [{done}/{total}] {bname}")
        res = vqe_test_extended(bqc, nq, H, n_restarts=40, maxiter=800)
        err = abs(res["best_energy"] - exact_gs) / abs(exact_gs) if exact_gs != 0 else 0
        results[bname] = {
            "error": err, "best_energy": res["best_energy"],
            "n_params": 4 * nq, "convergence": res["convergence_trace"],
        }

    # Run fully-parameterized baselines
    for bname, (build_fn, n_params) in param_baselines.items():
        done += 1
        print(f"    [{done}/{total}] {bname} ({n_params} params)")
        res = vqe_test_ansatz(build_fn, nq, n_params, H, n_restarts=40, maxiter=800)
        err = abs(res["best_energy"] - exact_gs) / abs(exact_gs) if exact_gs != 0 else 0
        results[bname] = {
            "error": err, "best_energy": res["best_energy"],
            "n_params": n_params, "convergence": res["convergence_trace"],
        }

    # 20 random circuits (constrained to candidate's connectivity)
    random_errors = []
    for i in range(20):
        done += 1
        print(f"    [{done}/{total}] random_{i}")
        rqc = _build_random_entangler(nq, max(1, n_2q), seed=i,
                                       allowed_edges=cand_edges)
        res = vqe_test_extended(rqc, nq, H, n_restarts=20, maxiter=400)
        err = abs(res["best_energy"] - exact_gs) / abs(exact_gs) if exact_gs != 0 else 0
        random_errors.append(err)

    results["random_mean_error"] = float(np.mean(random_errors))
    results["random_std_error"] = float(np.std(random_errors))
    results["random_min_error"] = float(np.min(random_errors))
    results["random_max_error"] = float(np.max(random_errors))
    results["random_errors"] = random_errors

    # Candidate rank
    named_errors = [results[k]["error"] for k in
                    list(fixed_baselines) + list(param_baselines) if k != "candidate"]
    all_errors = named_errors + random_errors
    cand_err = results["candidate"]["error"]
    rank = 1 + sum(1 for e in all_errors if e < cand_err)
    results["candidate_rank"] = rank
    results["total_tested"] = len(all_errors) + 1

    _save_phase(ckpt, tk, "phase4", results)
    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Structural Origin
# ═══════════════════════════════════════════════════════════════════════


def _identify_motif_gate_ranges(
    spec_motif_ids: list[str], n_qubits: int, templates: dict,
) -> list[tuple[str, int, int]]:
    """Build the circuit motif-by-motif and track which gate indices belong to each motif.

    Returns list of (motif_id, start_gate_idx, end_gate_idx).
    Mirrors the round-robin logic of _build_template_circuit.
    """
    ranges = []
    offset = 0
    gate_idx = 0
    for mid in spec_motif_ids:
        if mid not in templates:
            continue
        tpl_qubits, apply_fn = templates[mid]
        if offset + tpl_qubits > n_qubits:
            offset = 0
        probe = QuantumCircuit(n_qubits)
        try:
            apply_fn(probe, offset)
        except Exception:
            continue
        n_gates = probe.size()
        ranges.append((mid, gate_idx, gate_idx + n_gates))
        gate_idx += n_gates
        offset = (offset + 1) % max(1, n_qubits - tpl_qubits + 1)
    return ranges


def motif_ablation_study(target: dict, scored: dict, templates: dict) -> dict:
    """5.1: Remove each motif's gates (replace with identity) from the existing circuit."""
    nq = target["n_qubits"]
    model = target["model"]
    H = _build_hamiltonian_matrix(nq, model)
    exact_gs = float(np.linalg.eigvalsh(H)[0])

    cand_name = target["candidate"]
    score_entry = scored.get(cand_name, {})
    if not score_entry:
        base = cand_name.rsplit("_q", 1)[0] if "_q" in cand_name else cand_name
        score_entry = scored.get(base, {})

    top_motifs = score_entry.get("top_motifs", [])
    motif_ids = [mid for mid, _ in top_motifs[:6]]

    full_qc, _ = _get_candidate_qc(target, scored, templates)
    if full_qc is None:
        return {"error": "failed to build candidate"}

    full_res = vqe_test_extended(full_qc, nq, H, n_restarts=20, maxiter=400)
    full_err = abs(full_res["best_energy"] - exact_gs) / abs(exact_gs)

    # Identify which gates belong to each motif
    gate_ranges = _identify_motif_gate_ranges(motif_ids, nq, templates)

    results = {"full_error": full_err, "ablations": {}}

    for mid, g_start, g_end in gate_ranges:
        # Build ablated circuit: skip gates in [g_start, g_end)
        ablated = QuantumCircuit(nq)
        for idx, inst in enumerate(full_qc.data):
            if g_start <= idx < g_end:
                continue  # replace with identity (omit)
            qubits = [full_qc.find_bit(q).index for q in inst.qubits]
            ablated.append(inst.operation, qubits)

        if ablated.size() == 0:
            results["ablations"][mid] = {"error": "all gates removed"}
            continue

        print(f"    remove {mid} (gates {g_start}:{g_end})...")
        res = vqe_test_extended(ablated, nq, H, n_restarts=20, maxiter=400)
        err = abs(res["best_energy"] - exact_gs) / abs(exact_gs)
        results["ablations"][mid] = {
            "error": err,
            "error_increase": err - full_err,
            "gates_removed": g_end - g_start,
            "critical": (err - full_err) > 0.05,
        }

    return results


def gate_ablation_study(target: dict, scored: dict, templates: dict) -> dict:
    """5.2: Remove each 2Q gate individually, test VQE."""
    nq = target["n_qubits"]
    model = target["model"]
    H = _build_hamiltonian_matrix(nq, model)
    exact_gs = float(np.linalg.eigvalsh(H)[0])

    cand_qc, _ = _get_candidate_qc(target, scored, templates)
    if cand_qc is None:
        return {"error": "failed to build candidate"}

    full_res = vqe_test_extended(cand_qc, nq, H, n_restarts=20, maxiter=400)
    full_err = abs(full_res["best_energy"] - exact_gs) / abs(exact_gs)

    # Find 2Q gate indices
    gate_indices = []
    for idx, inst in enumerate(cand_qc.data):
        if inst.operation.num_qubits >= 2:
            gate_indices.append(idx)

    results = {"full_error": full_err, "gate_ablations": []}
    for gi in gate_indices:
        ablated = QuantumCircuit(nq)
        for idx, inst in enumerate(cand_qc.data):
            if idx == gi:
                continue
            qubits = [cand_qc.find_bit(q).index for q in inst.qubits]
            ablated.append(inst.operation, qubits)

        print(f"    remove gate {gi}...")
        res = vqe_test_extended(ablated, nq, H, n_restarts=20, maxiter=400)
        err = abs(res["best_energy"] - exact_gs) / abs(exact_gs)
        orig_inst = cand_qc.data[gi]
        results["gate_ablations"].append({
            "gate_index": gi,
            "gate_name": orig_inst.operation.name,
            "qubits": [cand_qc.find_bit(q).index for q in orig_inst.qubits],
            "error": err,
            "error_increase": err - full_err,
            "critical": (err - full_err) > 0.05,
        })

    return results


def zx_structural_analysis(target: dict, scored: dict, templates: dict) -> dict:
    """5.3: Full ZX simplification trajectory."""
    nq = target["n_qubits"]
    cand_qc, _ = _get_candidate_qc(target, scored, templates)
    if cand_qc is None:
        return {"error": "failed to build candidate"}

    # Also compute for baselines
    n_2q = _count_2q(cand_qc)
    circuits = {
        "candidate": cand_qc,
        "cx_chain": _cx_chain_entangler(nq, max(1, n_2q)),
        "hea": _hea_entangler(nq, max(1, n_2q)),
    }

    results = {}
    for cname, qc in circuits.items():
        zx_data = compute_zx_compression(cname, qc)
        results[cname] = zx_data

    return results


def entanglement_pathway_analysis(target: dict, scored: dict, templates: dict) -> dict:
    """5.4: Track entropy gate-by-gate."""
    nq = target["n_qubits"]
    cand_qc, _ = _get_candidate_qc(target, scored, templates)
    if cand_qc is None:
        return {"error": "failed to build candidate"}

    # Simulate gate by gate, tracking entropy
    partial_qc = QuantumCircuit(nq)
    entropies = [0.0]  # initial |0...0> has 0 entropy
    concurrences = []

    for inst in cand_qc.data:
        qubits = [cand_qc.find_bit(q).index for q in inst.qubits]
        partial_qc.append(inst.operation, qubits)
        try:
            sv = np.array(Statevector.from_instruction(partial_qc).data)
            ent = _half_cut_entropy(sv, nq)
            entropies.append(ent)
            if nq <= 8:
                conc = _pairwise_concurrences(sv, nq)
                concurrences.append(float(np.mean(conc[np.triu_indices(nq, 1)])))
        except Exception:
            entropies.append(entropies[-1])
            concurrences.append(0.0)

    return {
        "entropy_trajectory": entropies,
        "avg_concurrence_trajectory": concurrences,
        "n_gates": len(cand_qc.data),
    }


def run_phase5(target: dict, scored: dict, templates: dict, ckpt: dict) -> dict:
    """Run all Phase 5 sub-tests."""
    tk = target["key"]
    results = {}
    phases = [
        ("5.1_motif_ablation", "motif_ablation",
         lambda: motif_ablation_study(target, scored, templates)),
        ("5.2_gate_ablation", "gate_ablation",
         lambda: gate_ablation_study(target, scored, templates)),
        ("5.3_zx_structure", "zx_structure",
         lambda: zx_structural_analysis(target, scored, templates)),
        ("5.4_entanglement_pathway", "entanglement_pathway",
         lambda: entanglement_pathway_analysis(target, scored, templates)),
    ]
    for phase_key, result_key, func in phases:
        if not _phase_done(ckpt, tk, phase_key):
            print(f"  {phase_key[:3]} {result_key.replace('_', ' ').title()}...")
            r = func()
            _save_phase(ckpt, tk, phase_key, r)
            results[result_key] = r
        else:
            results[result_key] = ckpt[tk][phase_key]
            print(f"  {phase_key[:3]} (cached)")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 6: Publication Prep — Visualizations
# ═══════════════════════════════════════════════════════════════════════


def plot_optimizer_comparison(phase1: dict, target: dict):
    """Plot 1: Grouped bars for candidate vs baselines × 5 optimizers."""
    opt_data = phase1.get("optimizer_sensitivity", {})
    if "error" in opt_data:
        return

    optimizers = ["COBYLA", "L-BFGS-B", "Nelder-Mead", "Powell", "SPSA"]
    circuit_types = ["candidate", "cx_chain", "hea"]
    colors = {"candidate": "#4daf4a", "cx_chain": "#2166ac", "hea": "#984ea3"}

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(optimizers))
    w = 0.25

    for i, ct in enumerate(circuit_types):
        errors = []
        for opt in optimizers:
            key = f"{opt}_{ct}"
            errors.append(opt_data.get(key, {}).get("error", 0))
        ax.bar(x + i * w, errors, w, label=ct.replace("_", " ").title(),
               color=colors[ct], alpha=0.8, edgecolor="k", linewidth=0.3)

    ax.set_xticks(x + w)
    ax.set_xticklabels(optimizers)
    ax.set_ylabel("Relative Error vs Exact GS")
    ax.set_title(f"Optimizer Sensitivity: {target['candidate']} "
                 f"{target['n_qubits']}q {target['model']}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "optimizer_comparison.png", dpi=150)
    plt.close(fig)


def plot_statistical_significance(phase1: dict, target: dict):
    """Plot 2: Box plots of bootstrap trials."""
    stat_data = phase1.get("statistical_significance", {})
    if "error" in stat_data:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    data = [stat_data.get("candidate_errors", []), stat_data.get("baseline_errors", [])]
    bp = ax.boxplot(data, tick_labels=["Candidate", "CX-Chain Baseline"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#4daf4a")
    bp["boxes"][1].set_facecolor("#2166ac")
    for box in bp["boxes"]:
        box.set_alpha(0.7)

    ax.set_ylabel("Relative Error vs Exact GS")
    p_val = stat_data.get("p_value", 1)
    d = stat_data.get("cohens_d", 0)
    ax.set_title(f"Statistical Significance (p={p_val:.4f}, Cohen's d={d:.2f})\n"
                 f"{target['candidate']} {target['n_qubits']}q {target['model']}")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "statistical_significance.png", dpi=150)
    plt.close(fig)


def plot_entanglement_structure(phase2: dict, target: dict):
    """Plot 3: Pairwise concurrence heatmaps."""
    ent_data = phase2.get("entanglement_structure", {})
    if "error" in ent_data:
        return

    gs_conc = np.array(ent_data.get("gs_concurrences", []))
    vqe_conc = np.array(ent_data.get("vqe_concurrences", []))
    if gs_conc.size == 0 or vqe_conc.size == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    vmax = max(np.max(gs_conc), np.max(vqe_conc))

    im1 = ax1.imshow(gs_conc, cmap="Reds", vmin=0, vmax=vmax)
    ax1.set_title("Ground State Concurrences")
    ax1.set_xlabel("Qubit")
    ax1.set_ylabel("Qubit")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(vqe_conc, cmap="Reds", vmin=0, vmax=vmax)
    ax2.set_title("VQE State Concurrences")
    ax2.set_xlabel("Qubit")
    plt.colorbar(im2, ax=ax2)

    corr = ent_data.get("concurrence_correlation", 0)
    fig.suptitle(f"Entanglement Structure (correlation={corr:.3f})\n"
                 f"{target['candidate']} {target['n_qubits']}q {target['model']}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "entanglement_structure.png", dpi=150)
    plt.close(fig)


def plot_parameter_landscape(phase2: dict, target: dict):
    """Plot 4: 1D landscape slices for top parameters."""
    land_data = phase2.get("parameter_landscape", {})
    if "error" in land_data:
        return

    slices = land_data.get("slices", {})
    if not slices:
        return

    # Pick top 6 by energy range
    sorted_slices = sorted(slices.items(), key=lambda x: -x[1].get("energy_range", 0))[:6]

    n_plots = len(sorted_slices)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    thetas = np.linspace(-np.pi, np.pi, 50)

    for idx, (pname, sdata) in enumerate(sorted_slices):
        ax = axes[idx // 3][idx % 3]
        energies = sdata.get("energies", [])
        if energies:
            ax.plot(thetas[:len(energies)], energies, "b-", linewidth=1.5)
        ax.set_title(f"{pname} ({sdata.get('n_minima', 0)} minima)", fontsize=9)
        ax.set_xlabel("θ")
        ax.set_ylabel("Energy")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_plots, 6):
        axes[idx // 3][idx % 3].set_visible(False)

    fig.suptitle(f"Parameter Landscape Slices\n{target['candidate']} "
                 f"{target['n_qubits']}q {target['model']}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "parameter_landscape.png", dpi=150)
    plt.close(fig)


def plot_scaling_analysis(phase3: dict, target: dict):
    """Plot 5: Error vs qubit count."""
    scaling = phase3.get("qubit_scaling", {})
    if "error" in scaling:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for model, ax in [("heisenberg", ax1), ("tfim", ax2)]:
        nqs, cand_errs, cx_errs, hea_errs = [], [], [], []
        for nq in [4, 5, 6, 7, 8]:
            key = f"{nq}q_{model}"
            if key in scaling and "error" not in scaling[key]:
                nqs.append(nq)
                cand_errs.append(scaling[key]["candidate_error"])
                cx_errs.append(scaling[key]["cx_chain_error"])
                hea_errs.append(scaling[key]["hea_error"])

        if nqs:
            ax.plot(nqs, cand_errs, "o-", color="#4daf4a", label="Candidate", linewidth=2)
            ax.plot(nqs, cx_errs, "s--", color="#2166ac", label="CX-chain", linewidth=1.5)
            ax.plot(nqs, hea_errs, "^--", color="#984ea3", label="Full HEA", linewidth=1.5)
            ax.set_xlabel("Qubits")
            ax.set_ylabel("Relative Error")
            ax.set_title(f"{model.upper()}")
            ax.legend()
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"Qubit Scaling: {target['candidate']}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scaling_analysis.png", dpi=150)
    plt.close(fig)


def plot_hamiltonian_generality(phase3: dict, target: dict):
    """Plot 6: Error across different Hamiltonians."""
    ham_data = phase3.get("hamiltonian_generality", {})
    if "error" in ham_data:
        return

    models = [k for k in ham_data if isinstance(ham_data[k], dict) and "candidate_error" in ham_data[k]]
    if not models:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    w = 0.25

    cand_errs = [ham_data[m]["candidate_error"] for m in models]
    cx_errs = [ham_data[m]["cx_chain_error"] for m in models]
    hea_errs = [ham_data[m]["hea_error"] for m in models]

    ax.bar(x - w, cand_errs, w, label="Candidate", color="#4daf4a", alpha=0.8,
           edgecolor="k", linewidth=0.3)
    ax.bar(x, cx_errs, w, label="CX-chain", color="#2166ac", alpha=0.8,
           edgecolor="k", linewidth=0.3)
    ax.bar(x + w, hea_errs, w, label="Full HEA", color="#984ea3", alpha=0.8,
           edgecolor="k", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models], rotation=30)
    ax.set_ylabel("Relative Error")
    ax.set_title(f"Hamiltonian Generality (6q): {target['candidate']}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hamiltonian_generality.png", dpi=150)
    plt.close(fig)


def plot_baseline_comparison(phase4: dict, target: dict):
    """Plot 7: All baselines sorted by VQE error."""
    if "error" in phase4:
        return

    items = []
    for k, v in phase4.items():
        if isinstance(v, dict) and "error" in v:
            items.append((k, v["error"]))
    items.sort(key=lambda x: x[1])

    if not items:
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(items) * 0.3)))
    y = np.arange(len(items))
    colors = ["#4daf4a" if i[0] == "candidate" else
              "#e41a1c" if "random" in i[0] else "#2166ac" for i in items]

    ax.barh(y, [i[1] for i in items], color=colors, alpha=0.8,
            edgecolor="k", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels([i[0].replace("_", " ").title() for i in items], fontsize=8)
    ax.set_xlabel("Relative Error")
    ax.set_title(f"Baseline Comparison: {target['candidate']} "
                 f"{target['n_qubits']}q {target['model']}")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # Add random circuit range
    rmean = phase4.get("random_mean_error", 0)
    rstd = phase4.get("random_std_error", 0)
    if rmean > 0:
        ax.axvspan(rmean - rstd, rmean + rstd, alpha=0.1, color="red",
                   label=f"Random range ({rmean:.3f}±{rstd:.3f})")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "baseline_comparison.png", dpi=150)
    plt.close(fig)


def plot_ablation_study(phase5: dict, target: dict):
    """Plot 8: Energy change when each motif/gate removed."""
    motif_abl = phase5.get("motif_ablation", {})
    gate_abl = phase5.get("gate_ablation", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Motif ablation
    if "ablations" in motif_abl:
        ablations = motif_abl["ablations"]
        full_err = motif_abl.get("full_error", 0)
        names, increases = [], []
        for mid, data in ablations.items():
            if isinstance(data, dict) and "error_increase" in data:
                names.append(mid[:15])
                increases.append(data["error_increase"])
        if names:
            y = np.arange(len(names))
            colors = ["#e41a1c" if inc > 0.05 else "#4daf4a" for inc in increases]
            ax1.barh(y, increases, color=colors, alpha=0.8, edgecolor="k", linewidth=0.3)
            ax1.set_yticks(y)
            ax1.set_yticklabels(names, fontsize=7)
            ax1.axvline(0, color="k", linewidth=0.5)
            ax1.set_xlabel("Error Increase When Removed")
            ax1.set_title("Motif Ablation")
            ax1.invert_yaxis()

    # Gate ablation
    if "gate_ablations" in gate_abl:
        abl_list = gate_abl["gate_ablations"]
        names, increases = [], []
        for g in abl_list:
            names.append(f"{g['gate_name']} q{g['qubits']}")
            increases.append(g.get("error_increase", 0))
        if names:
            y = np.arange(len(names))
            colors = ["#e41a1c" if inc > 0.05 else "#4daf4a" for inc in increases]
            ax2.barh(y, increases, color=colors, alpha=0.8, edgecolor="k", linewidth=0.3)
            ax2.set_yticks(y)
            ax2.set_yticklabels(names, fontsize=7)
            ax2.axvline(0, color="k", linewidth=0.5)
            ax2.set_xlabel("Error Increase When Removed")
            ax2.set_title("Gate Ablation")
            ax2.invert_yaxis()

    fig.suptitle(f"Ablation Study: {target['candidate']} "
                 f"{target['n_qubits']}q {target['model']}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ablation_study.png", dpi=150)
    plt.close(fig)


def plot_convergence(phase4: dict, target: dict):
    """Plot 9: Convergence traces — best energy vs restart for candidate and baselines."""
    if "error" in phase4:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "candidate": "#4daf4a", "cx_chain": "#2166ac", "hea_cz_brick": "#984ea3",
        "full_hea_2L": "#e41a1c", "qaoa_flex": "#ff7f00", "hva_heisenberg": "#a65628",
        "shuffled_candidate": "#999999",
    }
    for name, color in colors.items():
        data = phase4.get(name, {})
        trace = data.get("convergence")
        if trace:
            ax.plot(range(1, len(trace) + 1), trace, "-", color=color,
                    label=f"{name} ({data.get('n_params', '?')}p)", linewidth=1.5,
                    alpha=0.8)

    ax.set_xlabel("Restart")
    ax.set_ylabel("Best Energy Found")
    ax.set_title(f"Convergence: {target['candidate']} "
                 f"{target['n_qubits']}q {target['model']}")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "convergence.png", dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Phase 6: Report & QASM export
# ═══════════════════════════════════════════════════════════════════════


def export_qasm(target: dict, scored: dict, templates: dict):
    """Export candidate circuit as QASM."""
    cand_qc, _ = _get_candidate_qc(target, scored, templates)
    if cand_qc is None:
        return
    import qiskit.qasm2
    qasm_str = qiskit.qasm2.dumps(cand_qc)
    (OUTPUT_DIR / "candidate_circuit.qasm").write_text(qasm_str)
    return qasm_str


def compute_error_budget(all_results: dict, target: dict) -> dict:
    """Compute comprehensive error budget."""
    p1 = all_results.get("phase1", {})
    stat = p1.get("statistical_significance", {})

    budget = {
        "candidate": target["candidate"],
        "n_qubits": target["n_qubits"],
        "model": target["model"],
        "original_error": target["candidate_error"],
        "original_improvement": target["improvement"],
    }

    if stat and "error" not in stat:
        budget["statistical_mean_error"] = stat.get("candidate_mean", 0)
        budget["statistical_std"] = stat.get("candidate_std", 0)
        budget["p_value"] = stat.get("p_value", 1)
        budget["significant"] = stat.get("significant_at_005", False)
        budget["ci_95"] = [stat.get("ci_95_low", 0), stat.get("ci_95_high", 0)]

    repro = p1.get("reproducibility", {})
    if repro and "error" not in repro:
        budget["reproducibility_std"] = repro.get("std_error", 0)

    return budget


def generate_verification_report(all_results: dict, target: dict, elapsed: float) -> str:
    """Generate comprehensive text report."""
    L = []
    L.append("=" * 80)
    L.append("VQE RESULT VERIFICATION REPORT")
    L.append("=" * 80)
    L.append("")
    L.append(f"Target: {target['candidate']} {target['n_qubits']}q {target['model']}")
    L.append(f"Original finding: {target['candidate_error']:.4f} error, "
             f"+{target['improvement']:.4f} improvement over baselines")
    L.append(f"Elapsed: {elapsed:.1f}s")
    L.append("")

    # Phase 1: Verification
    L.append("=" * 80)
    L.append("PHASE 1: IS THE RESULT REAL?")
    L.append("=" * 80)
    p1 = all_results.get("phase1", {})

    # 1.1 Circuit identity
    ci = p1.get("circuit_identity", {})
    L.append("")
    L.append("1.1 Circuit Identity:")
    for nq_key, data in ci.items():
        if isinstance(data, dict):
            L.append(f"  {nq_key}: {data.get('n_gates', '?')} gates, "
                     f"{data.get('n_2q_gates', '?')} 2Q, depth={data.get('depth', '?')}")

    # 1.2 Optimizer sensitivity
    opt = p1.get("optimizer_sensitivity", {})
    L.append("")
    L.append("1.2 Optimizer Sensitivity:")
    wins = opt.get("wins_by_optimizer", {})
    if wins:
        for o, w in wins.items():
            L.append(f"  {o}: candidate {'WINS' if w else 'loses'}")
        L.append(f"  Wins all optimizers: {opt.get('candidate_wins_all', False)}")

    # 1.3 Statistical significance
    stat = p1.get("statistical_significance", {})
    L.append("")
    L.append("1.3 Statistical Significance:")
    if stat and "error" not in stat:
        L.append(f"  Candidate mean error: {stat.get('candidate_mean', 0):.4f} "
                 f"± {stat.get('candidate_std', 0):.4f}")
        L.append(f"  Baseline mean error:  {stat.get('baseline_mean', 0):.4f} "
                 f"± {stat.get('baseline_std', 0):.4f}")
        L.append(f"  p-value: {stat.get('p_value', 1):.6f}")
        L.append(f"  Cohen's d: {stat.get('cohens_d', 0):.3f}")
        L.append(f"  95% CI for improvement: [{stat.get('ci_95_low', 0):.4f}, "
                 f"{stat.get('ci_95_high', 0):.4f}]")
        L.append(f"  Significant at 0.05: {stat.get('significant_at_005', False)}")
        L.append(f"  Note: Each trial uses 10 restarts / 400 iter (reduced from 40/800 deep benchmark).")
        L.append(f"  Absolute errors are higher but comparison is valid (same settings for both).")

    # 1.4 Reproducibility
    repro = p1.get("reproducibility", {})
    L.append("")
    L.append("1.4 Reproducibility:")
    if repro and "error" not in repro:
        L.append(f"  Mean error across seeds: {repro.get('mean_error', 0):.4f} "
                 f"± {repro.get('std_error', 0):.4f}")
        L.append(f"  Consistent: {repro.get('consistent', False)}")

    # 1.5 Full HEA
    hea = p1.get("full_hea_baseline", {})
    L.append("")
    L.append("1.5 Full HEA Baseline:")
    if hea and "error" not in hea:
        L.append(f"  Candidate error: {hea.get('candidate_error', 0):.4f}")
        L.append(f"  Full HEA error:  {hea.get('full_hea_error', 0):.4f}")
        L.append(f"  Beats full HEA: {hea.get('candidate_beats_full_hea', False)}")

    # Phase 2
    L.append("")
    L.append("=" * 80)
    L.append("PHASE 2: WHY DOES IT WORK?")
    L.append("=" * 80)
    p2 = all_results.get("phase2", {})

    cs = p2.get("circuit_structure", {})
    L.append("")
    L.append("2.1 Circuit Structure:")
    L.append(f"  Pattern: {cs.get('pattern', '?')}, {cs.get('n_edges', 0)} unique edges")
    L.append(f"  Gates: {cs.get('n_gates', 0)} ({cs.get('n_1q_gates', 0)} 1Q, "
             f"{cs.get('n_2q_gates', 0)} 2Q), depth={cs.get('depth', 0)}")

    eh = p2.get("entangler_hamiltonian", {})
    L.append("")
    L.append("2.2 Entangler Hamiltonian:")
    if "error" not in eh:
        L.append(f"  Heisenberg cosine similarity: {eh.get('heisenberg_cosine_sim', 0):.4f}")
        L.append(f"  XX+YY+ZZ ratio: {eh.get('xyz_ratio', 0):.4f}")
        L.append(f"  N-body distribution: {eh.get('n_body_distribution', {})}")

    ent = p2.get("entanglement_structure", {})
    L.append("")
    L.append("2.3 Entanglement Structure:")
    if "error" not in ent:
        L.append(f"  Concurrence correlation with GS: {ent.get('concurrence_correlation', 0):.4f}")
        L.append(f"  GS half-cut entropy: {ent.get('gs_half_cut_entropy', 0):.4f}")
        L.append(f"  VQE half-cut entropy: {ent.get('vqe_half_cut_entropy', 0):.4f}")

    gv = p2.get("gradient_variance", {})
    L.append("")
    L.append("2.5 Gradient Variance (barren plateau indicator):")
    if "error" not in gv:
        for cname, gdata in gv.items():
            if isinstance(gdata, dict):
                L.append(f"  {cname}: mean={gdata.get('mean_variance', 0):.6f}, "
                         f"min={gdata.get('min_variance', 0):.6f}")

    expr = p2.get("expressibility", {})
    L.append("")
    L.append("2.6 Expressibility:")
    if "error" not in expr:
        L.append(f"  KL divergence from Haar: {expr.get('kl_divergence', 0):.4f}")

    # Phase 3
    L.append("")
    L.append("=" * 80)
    L.append("PHASE 3: LIMITS")
    L.append("=" * 80)
    p3 = all_results.get("phase3", {})

    sc = p3.get("qubit_scaling", {})
    L.append("")
    L.append("3.1 Qubit Scaling:")
    for key in sorted(sc):
        if isinstance(sc[key], dict) and "candidate_error" in sc[key]:
            d = sc[key]
            wins = "WINS" if d.get("candidate_beats_both") else "loses"
            L.append(f"  {key}: cand={d['candidate_error']:.4f} "
                     f"cx={d['cx_chain_error']:.4f} hea={d['hea_error']:.4f} [{wins}]")

    hg = p3.get("hamiltonian_generality", {})
    L.append("")
    L.append("3.2 Hamiltonian Generality (6q):")
    for hname, hdata in hg.items():
        if isinstance(hdata, dict) and "candidate_error" in hdata:
            L.append(f"  {hname}: cand={hdata['candidate_error']:.4f} "
                     f"cx={hdata['cx_chain_error']:.4f} hea={hdata['hea_error']:.4f}")

    ld = p3.get("layer_depth", {})
    L.append("")
    L.append("3.3 Layer Depth:")
    for key in sorted(ld):
        if isinstance(ld[key], dict) and "error" in ld[key]:
            L.append(f"  {key}: error={ld[key]['error']:.4f} "
                     f"({ld[key].get('n_params', 0)} params)")

    # Phase 4
    L.append("")
    L.append("=" * 80)
    L.append("PHASE 4: STRONGER BASELINES")
    L.append("=" * 80)
    L.append("Full HEA, QAOA-flex, HVA-Heisenberg use fully variational parameters.")
    L.append("Random circuits are constrained to the candidate's qubit connectivity.")
    p4 = all_results.get("phase4", {})
    L.append("")
    if "error" not in p4:
        L.append(f"Candidate rank: {p4.get('candidate_rank', '?')} / "
                 f"{p4.get('total_tested', '?')}")
        L.append(f"Random circuits: mean={p4.get('random_mean_error', 0):.4f} "
                 f"± {p4.get('random_std_error', 0):.4f}")
        L.append("")
        # Show all baselines sorted
        items = [(k, v["error"], v.get("n_params", "?")) for k, v in p4.items()
                 if isinstance(v, dict) and "error" in v]
        items.sort(key=lambda x: x[1])
        L.append(f"  {'Name':30s} {'Error':>8s} {'Params':>8s}")
        L.append(f"  {'-'*48}")
        for name, err, np_ in items:
            marker = " <--" if name == "candidate" else ""
            L.append(f"  {name:30s} {err:8.4f} {str(np_):>8s}{marker}")

    # Phase 5
    L.append("")
    L.append("=" * 80)
    L.append("PHASE 5: STRUCTURAL ORIGIN")
    L.append("=" * 80)
    p5 = all_results.get("phase5", {})

    ma = p5.get("motif_ablation", {})
    L.append("")
    L.append("5.1 Motif Ablation:")
    if "ablations" in ma:
        L.append(f"  Full circuit error: {ma.get('full_error', 0):.4f}")
        for mid, data in ma["ablations"].items():
            if isinstance(data, dict) and "error_increase" in data:
                critical = " CRITICAL" if data.get("critical") else ""
                L.append(f"  Remove {mid}: +{data['error_increase']:.4f}{critical}")

    ga = p5.get("gate_ablation", {})
    L.append("")
    L.append("5.2 Gate Ablation:")
    if "gate_ablations" in ga:
        L.append(f"  Full circuit error: {ga.get('full_error', 0):.4f}")
        for g in ga["gate_ablations"]:
            critical = " CRITICAL" if g.get("critical") else ""
            L.append(f"  Remove {g['gate_name']} q{g['qubits']}: "
                     f"+{g.get('error_increase', 0):.4f}{critical}")

    ep = p5.get("entanglement_pathway", {})
    L.append("")
    L.append("5.4 Entanglement Pathway:")
    if "entropy_trajectory" in ep:
        traj = ep["entropy_trajectory"]
        L.append(f"  Entropy: 0 -> {traj[-1]:.4f} over {ep.get('n_gates', 0)} gates")

    # Verdict
    L.append("")
    L.append("=" * 80)
    L.append("VERDICT")
    L.append("=" * 80)
    L.append("")

    # Collect evidence
    confirmed = []
    refuted = []

    stat = p1.get("statistical_significance", {})
    if stat.get("significant_at_005"):
        confirmed.append(f"Statistically significant (p={stat['p_value']:.4f})")
    elif stat and "p_value" in stat:
        refuted.append(f"Not statistically significant (p={stat['p_value']:.4f})")

    opt_wins = p1.get("optimizer_sensitivity", {}).get("wins_by_optimizer", {})
    if p1.get("optimizer_sensitivity", {}).get("candidate_wins_all"):
        confirmed.append("Advantage holds across all 5 optimizers")
    elif opt_wins:
        n_opt_wins = sum(1 for w in opt_wins.values() if w)
        refuted.append(f"Only wins with {n_opt_wins}/5 optimizers")

    if p1.get("reproducibility", {}).get("consistent"):
        confirmed.append("Reproducible across seeds")

    if p1.get("full_hea_baseline", {}).get("candidate_beats_full_hea"):
        confirmed.append("Beats proper full HEA baseline")
    elif p1.get("full_hea_baseline", {}) and "error" not in p1["full_hea_baseline"]:
        refuted.append("Does NOT beat proper full HEA baseline")

    rank = p4.get("candidate_rank", 999)
    total = p4.get("total_tested", 0)
    if rank == 1:
        confirmed.append(f"Ranks #1 among {total} circuits")
    elif rank <= 3:
        confirmed.append(f"Ranks #{rank} among {total} circuits")
    else:
        refuted.append(f"Ranks only #{rank} among {total} circuits")

    if confirmed:
        L.append("EVIDENCE FOR genuine advantage:")
        for c in confirmed:
            L.append(f"  + {c}")
    if refuted:
        L.append("EVIDENCE AGAINST:")
        for r in refuted:
            L.append(f"  - {r}")

    if len(confirmed) >= 3 and not refuted:
        L.append("")
        L.append("CONCLUSION: Result appears GENUINE. Proceed with further investigation.")
    elif len(refuted) > len(confirmed):
        L.append("")
        L.append("CONCLUSION: Result is likely an ARTIFACT. "
                 "Advantage does not survive rigorous testing.")
    else:
        L.append("")
        L.append("CONCLUSION: MIXED evidence. Some aspects genuine, "
                 "others suggest the advantage is partial or conditional.")

    L.append("")
    L.append("=" * 80)
    L.append(f"Outputs: {OUTPUT_DIR}")
    L.append("=" * 80)
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()

    # Phase 0: Setup
    targets, scored, templates = phase0_setup()
    if not targets:
        print("No targets to verify. Exiting.")
        return

    ckpt = _load_checkpoint()
    all_target_results = {}

    for tidx, target in enumerate(targets):
        tk = target["key"]
        print(f"\n{'='*70}")
        print(f"Target {tidx + 1}/{len(targets)}: {tk}")
        print(f"{'='*70}")

        target_results = {}

        # Phase 1
        print("\nPhase 1: Verify the Result Is Real")
        target_results["phase1"] = run_phase1(target, scored, templates, ckpt)

        # Phase 2
        print("\nPhase 2: Understand Why It Works")
        target_results["phase2"] = run_phase2(target, scored, templates, ckpt)

        # Phase 3
        print("\nPhase 3: Characterize the Limits")
        target_results["phase3"] = run_phase3(target, scored, templates, ckpt)

        # Phase 4
        print("\nPhase 4: Stronger Baselines")
        target_results["phase4"] = run_phase4(target, scored, templates, ckpt)

        # Phase 5
        print("\nPhase 5: Structural Origin")
        target_results["phase5"] = run_phase5(target, scored, templates, ckpt)

        # Phase 6: Plots & report
        print("\nPhase 6: Publication Prep")
        plot_optimizer_comparison(target_results["phase1"], target)
        print("  optimizer_comparison.png")
        plot_statistical_significance(target_results["phase1"], target)
        print("  statistical_significance.png")
        plot_entanglement_structure(target_results["phase2"], target)
        print("  entanglement_structure.png")
        plot_parameter_landscape(target_results["phase2"], target)
        print("  parameter_landscape.png")
        plot_scaling_analysis(target_results["phase3"], target)
        print("  scaling_analysis.png")
        plot_hamiltonian_generality(target_results["phase3"], target)
        print("  hamiltonian_generality.png")
        plot_baseline_comparison(target_results["phase4"], target)
        print("  baseline_comparison.png")
        plot_ablation_study(target_results["phase5"], target)
        print("  ablation_study.png")
        plot_convergence(target_results["phase4"], target)
        print("  convergence.png")

        export_qasm(target, scored, templates)
        print("  candidate_circuit.qasm")

        all_target_results[tk] = target_results

    # Final report (for first/primary target)
    primary_target = targets[0]
    primary_results = all_target_results[primary_target["key"]]
    elapsed = time.time() - t0

    error_budget = compute_error_budget(primary_results, primary_target)
    primary_results["error_budget"] = error_budget

    report = generate_verification_report(primary_results, primary_target, elapsed)
    print("\n" + report)
    (OUTPUT_DIR / "verification_report.txt").write_text(report)

    # JSON results
    json_out = {
        "elapsed": elapsed,
        "targets": [t["key"] for t in targets],
        "results": {},
    }
    for tk, tr in all_target_results.items():
        json_out["results"][tk] = tr

    (OUTPUT_DIR / "verification_results.json").write_text(
        json.dumps(json_out, indent=2, default=str)
    )

    print(f"\nDone in {elapsed:.1f}s. Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
