#!/usr/bin/env python3
"""
Honest & Thorough Comparison: Discovery Candidates vs Industry Standards
========================================================================

Evaluates candidates on every axis where genuine advantage could appear:

  1. Output state analysis   — What does U|0...0> produce? Gate overhead?
  2. Entangling power & EPD  — Average entanglement over random inputs
  3. Best input search       — Find the input that maximises entanglement
  4. Hamiltonian matching    — What physical system does U = exp(-iH) simulate?
  5. Symmetry detection      — Which Pauli strings commute with U?
  6. Operator entanglement   — How entangling is U as a bipartite operator?
  7. Conditional states      — Measure spare qubits, check conditional properties
  8. VQE probe               — Test as entangling layer in variational ansatz
  9. ZX compression          — Legitimate structural metric

Outputs (scripts/output/comparison/):
  comparison_report.txt, results.json, overhead_analysis.png,
  state_structure.png, entangling_power.png, vqe_comparison.png,
  zx_compression.png
"""

from __future__ import annotations

import itertools
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from qiskit import QuantumCircuit

# ── Project imports ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zx_motifs.algorithms.registry import ALGORITHM_FAMILY_MAP, REGISTRY
from zx_motifs.pipeline.converter import (
    SimplificationLevel,
    convert_at_all_levels,
    count_t_gates,
    qiskit_to_zx,
)

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DISCOVERY_DIR = SCRIPT_DIR / "output" / "discovery"
OUTPUT_DIR = SCRIPT_DIR / "output" / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from discover_algorithm import (
    CandidateSpec,
    _HANDCRAFTED_TEMPLATES,
    build_circuit_from_spec,
    build_corpus,
    build_template_registry,
    discover_motifs,
    load_phylogeny_results,
    make_distillation_entanglement_bridge,
    make_irreducible_core_circuit,
    make_trotter_variational_hybrid,
)

REGISTRY_MAP = {entry.name: entry for entry in REGISTRY}

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

MAX_QUBITS_STATEVECTOR = 10  # statevector analysis
MAX_QUBITS_UNITARY = 6       # full unitary matrix (2^n x 2^n)
MAX_QUBITS_HAMILTONIAN = 5   # Pauli decomposition (4^n terms)
EP_SAMPLES = 100              # random inputs for entangling power
BEST_INPUT_SAMPLES = 100      # random inputs for best-case search
VQE_TOP_K = 15                # how many candidates to VQE-test
VQE_RESTARTS = 5
VQE_MAXITER = 150
NOISE_RATES = [0.001, 0.005, 0.01, 0.05]
DEEP_VQE_RESTARTS = 40
DEEP_VQE_MAXITER = 800
DEEP_VQE_TOP_K = 3                # auto-select top K from quick probe
DEEP_VQE_QUBIT_SIZES = [4, 6, 8]
DEEP_VQE_MODELS = ["heisenberg", "tfim"]
SIGNIFICANT_IMPROVEMENT_PP = 0.05  # 5 percentage points

# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Build Circuits (with key mismatch fix)
# ═══════════════════════════════════════════════════════════════════════

STANDARD_SPECS = [
    ("bell_state", 2),
    ("ghz_state", 4),
    ("cluster_state", 4),
    ("w_state", 4),
    ("teleportation", 3),
    ("grover", 4),
    ("qft", 4),
    ("phase_estimation", 4),
    ("qaoa_maxcut", 4),
    ("vqe_uccsd", 4),
    ("trotter_ising", 4),
    ("bit_flip_code", 5),
    ("bbpssw_distillation", 4),
    ("bb84_encode", 4),
    ("iqp_sampling", 4),
]


def build_standard_circuits() -> list[tuple[str, QuantumCircuit, str]]:
    circuits = []
    for name, nq in STANDARD_SPECS:
        entry = REGISTRY_MAP.get(name)
        if entry is None:
            continue
        try:
            qc = entry.generator(nq)
            circuits.append((f"std_{name}", qc, entry.family))
        except Exception as exc:
            print(f"  WARNING: {name}({nq}q) failed: {exc}")
    return circuits


def rebuild_candidates(
    discovery_results: dict, templates: dict
) -> list[tuple[str, QuantumCircuit, str]]:
    validations = discovery_results.get("validations", {})
    scored = discovery_results.get("scored", {})
    circuits = []
    for cname, score_info in scored.items():
        strategy = score_info.get("strategy", "unknown")

        # Fix key mismatch: scored keys have _q* suffix, validation keys don't
        val = validations.get(cname, {})
        if not val and "_q" in cname:
            base = cname.rsplit("_q", 1)[0]
            val = validations.get(base, {})

        n_qubits = val.get("n_qubits", 4)
        top_motifs = score_info.get("top_motifs", [])
        motif_ids = [mid for mid, _ in top_motifs[:6]]

        spec = CandidateSpec(
            name=cname,
            strategy=strategy,
            motif_ids=motif_ids,
            n_qubits=n_qubits,
            source_algo_a=score_info.get("nearest_algorithm"),
            source_algo_b=None,
        )
        qc = build_circuit_from_spec(spec, templates)
        if qc is not None and qc.size() > 0:
            circuits.append((cname, qc, strategy))
    return circuits


def _scale_candidate_spec(
    scored_entry: dict, target_nq: int, templates: dict,
) -> QuantumCircuit | None:
    """Create a CandidateSpec at a target qubit count from a scored dict entry."""
    strategy = scored_entry.get("strategy", "unknown")
    top_motifs = scored_entry.get("top_motifs", [])
    motif_ids = [mid for mid, _ in top_motifs[:6]]
    spec = CandidateSpec(
        name=scored_entry.get("name", "scaled"),
        strategy=strategy,
        motif_ids=motif_ids,
        n_qubits=target_nq,
        source_algo_a=scored_entry.get("nearest_algorithm"),
        source_algo_b=None,
    )
    qc = build_circuit_from_spec(spec, templates)
    if qc is not None and qc.size() > 0:
        return qc
    return None


def build_legacy_circuits() -> list[tuple[str, QuantumCircuit, str]]:
    return [
        ("legacy_tvh", make_trotter_variational_hybrid(n_qubits=4), "legacy"),
        ("legacy_deb", make_distillation_entanglement_bridge(n_qubits=4), "legacy"),
        ("legacy_icc", make_irreducible_core_circuit(n_qubits=4), "legacy"),
    ]


def _count_2q(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.num_qubits >= 2)


def _count_1q(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 1)


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Compute Unitaries (once, reuse everywhere)
# ═══════════════════════════════════════════════════════════════════════


def compute_unitaries(
    all_circuits: list[tuple[str, QuantumCircuit, str]],
) -> dict[str, np.ndarray]:
    from qiskit.quantum_info import Operator

    unitaries = {}
    for name, qc, _ in all_circuits:
        if qc.num_qubits <= MAX_QUBITS_UNITARY:
            try:
                unitaries[name] = Operator(qc).data
            except Exception:
                pass
    return unitaries


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Output State Analysis (SVD factorisation)
# ═══════════════════════════════════════════════════════════════════════


def _svd_bipartite(sv: np.ndarray, n: int, part_a: list[int]) -> tuple[np.ndarray, bool]:
    """SVD at bipartition. Returns (singular_values, is_product)."""
    n_a = len(part_a)
    n_b = n - n_a
    part_b = [i for i in range(n) if i not in part_a]
    tensor = sv.reshape([2] * n)
    # Qiskit little-endian: qubit q → tensor axis (n-1-q)
    axes_a = [n - 1 - q for q in part_a]
    axes_b = [n - 1 - q for q in part_b]
    tensor = np.transpose(tensor, axes_a + axes_b)
    mat = tensor.reshape(2**n_a, 2**n_b)
    s = np.linalg.svd(mat, compute_uv=False)
    return s, np.sum(s > 1e-10) <= 1


def _identify_1q(amp: np.ndarray) -> tuple[str, int]:
    """Identify single-qubit state. Returns (description, min_gates)."""
    amp = amp / np.linalg.norm(amp)
    a, b = amp[0], amp[1]
    if abs(abs(a) - 1.0) < 1e-6:
        return "|0>", 0
    if abs(abs(b) - 1.0) < 1e-6:
        return "|1>", 1
    if abs(abs(a) - abs(b)) < 1e-6:
        phase = np.angle(b / a) if abs(a) > 1e-10 else 0
        if abs(phase) < 0.1:
            return "|+>", 1
        if abs(abs(phase) - np.pi) < 0.1:
            return "|->", 2
        if abs(phase - np.pi / 2) < 0.1:
            return "|Y+>", 1
        if abs(phase + np.pi / 2) < 0.1:
            return "|Y->", 1
    return "rotation", 2


def _identify_2q(sv2: np.ndarray) -> tuple[str, str | None, float, int, int]:
    """Identify 2-qubit entangled state → (type, match, fidelity, min_g, min_2q)."""
    bell = {
        "Phi+": np.array([1, 0, 0, 1]) / np.sqrt(2),
        "Phi-": np.array([1, 0, 0, -1]) / np.sqrt(2),
        "Psi+": np.array([0, 1, 1, 0]) / np.sqrt(2),
        "Psi-": np.array([0, 1, -1, 0]) / np.sqrt(2),
    }
    best_name, best_fid = None, 0.0
    for nm, target in bell.items():
        fid = abs(np.dot(sv2.conj(), target)) ** 2
        if fid > best_fid:
            best_fid, best_name = fid, nm

    if best_fid > 0.99:
        return "bell_pair", best_name, best_fid, 2, 1

    from qiskit.quantum_info import Statevector, concurrence
    try:
        conc = concurrence(Statevector(sv2).to_operator())
    except Exception:
        conc = 0.0

    if conc > 0.99:
        return "rotated_bell", best_name, best_fid, 6, 1
    if conc > 0.5:
        return "partial_entangled_2q", None, conc, 5, 1
    return "weak_entangled_2q", None, conc, 4, 1


def _identify_nq(sv_nq: np.ndarray, nq: int) -> tuple[str, str | None, float, int, int]:
    """Identify n≥3 qubit entangled state → (type, match, score, min_g, min_2q)."""
    ghz = np.zeros(2**nq)
    ghz[0] = ghz[-1] = 1.0 / np.sqrt(2)
    fid_ghz = abs(np.dot(sv_nq.conj(), ghz)) ** 2
    if fid_ghz > 0.99:
        return "ghz", f"ghz_{nq}q", fid_ghz, nq, nq - 1

    w = np.zeros(2**nq)
    for i in range(nq):
        w[1 << i] = 1.0 / np.sqrt(nq)
    fid_w = abs(np.dot(sv_nq.conj(), w)) ** 2
    if fid_w > 0.99:
        return "w_state", f"w_{nq}q", fid_w, 2 * (nq - 1), nq - 1

    from qiskit.quantum_info import Statevector, entropy, partial_trace
    sv = Statevector(sv_nq)
    half = nq // 2
    rho_a = partial_trace(sv, list(range(half, nq)))
    ent = float(entropy(rho_a, base=2))

    if ent > 0.9 * half:
        return "highly_entangled", None, ent, 2 * nq, nq - 1
    if ent > 0.1:
        return "partially_entangled", None, ent, nq + 2, max(1, nq // 2)
    return "low_entanglement", None, ent, nq, 1


def _find_blocks(sv: np.ndarray, n: int, qubits: list[int]) -> list[list[int]]:
    """Find entangled sub-blocks via bipartite SVD."""
    if len(qubits) <= 1:
        return [qubits]
    if len(qubits) <= 6:
        from itertools import combinations
        for sz in range(1, len(qubits)):
            for ga in combinations(qubits, sz):
                _, prod = _svd_bipartite(sv, n, list(ga))
                if prod:
                    gb = [q for q in qubits if q not in ga]
                    return _find_blocks(sv, n, list(ga)) + _find_blocks(sv, n, gb)
    return [qubits]


def _extract_sub_state(sv_obj, n: int, qubits: list[int]) -> np.ndarray:
    from qiskit.quantum_info import partial_trace
    others = [q for q in range(n) if q not in qubits]
    if not others:
        return np.array(sv_obj.data)
    rho = partial_trace(sv_obj, others)
    evals, evecs = np.linalg.eigh(np.array(rho.data))
    return evecs[:, np.argmax(evals)]


def analyze_output_state(name: str, qc: QuantumCircuit) -> dict:
    """Factor U|0...0> into subsystems. Returns analysis dict."""
    from qiskit.quantum_info import Statevector, partial_trace

    n = qc.num_qubits
    n_g = qc.size()
    n_2q = _count_2q(qc)
    result = {
        "name": name, "n_qubits": n, "n_gates": n_g,
        "n_2q_gates": n_2q, "depth": qc.depth(),
        "n_idle": 0, "n_product": 0, "n_entangled": 0,
        "max_entangled_size": 0, "total_min_gates": 0,
        "total_min_2q": 0, "gate_overhead": 1.0,
        "subsystems": [], "description": "",
    }

    if n > MAX_QUBITS_STATEVECTOR:
        result["description"] = f">{MAX_QUBITS_STATEVECTOR}q, skipped"
        result["total_min_gates"] = n_g
        return result

    try:
        sv = Statevector.from_instruction(qc)
    except Exception:
        result["description"] = "statevector failed"
        result["total_min_gates"] = n_g
        return result

    sv_arr = np.array(sv.data)
    subsystems = []
    factored = set()

    # Peel off product qubits
    for q in range(n):
        others = [qq for qq in range(n) if qq != q and qq not in factored]
        if not others:
            break
        remaining = [qq for qq in range(n) if qq not in factored]
        if q not in remaining:
            continue
        _, is_prod = _svd_bipartite(sv_arr, n, [q])
        if is_prod:
            rho_q = partial_trace(sv, [qq for qq in range(n) if qq != q])
            evals, evecs = np.linalg.eigh(np.array(rho_q.data))
            q_state = evecs[:, np.argmax(evals)]
            desc, mg = _identify_1q(q_state)
            stype = "idle" if desc == "|0>" else "product_1q"
            if stype == "idle":
                result["n_idle"] += 1
            else:
                result["n_product"] += 1
            subsystems.append({
                "qubits": [q], "type": stype, "match": desc,
                "fidelity": 1.0, "min_gates": mg, "min_2q": 0,
            })
            factored.add(q)

    # Remaining qubits
    entangled_qs = [q for q in range(n) if q not in factored]
    if len(entangled_qs) == 1:
        q = entangled_qs[0]
        rho_q = partial_trace(sv, [qq for qq in range(n) if qq != q])
        evals, evecs = np.linalg.eigh(np.array(rho_q.data))
        q_state = evecs[:, np.argmax(evals)]
        desc, mg = _identify_1q(q_state)
        result["n_product"] += 1
        subsystems.append({
            "qubits": [q], "type": "product_1q", "match": desc,
            "fidelity": 1.0, "min_gates": mg, "min_2q": 0,
        })
    elif len(entangled_qs) >= 2:
        blocks = _find_blocks(sv_arr, n, entangled_qs)
        for blk in blocks:
            nb = len(blk)
            result["n_entangled"] += nb
            result["max_entangled_size"] = max(result["max_entangled_size"], nb)
            sub_state = _extract_sub_state(sv, n, blk)
            if nb == 1:
                desc, mg = _identify_1q(sub_state)
                subsystems.append({
                    "qubits": blk, "type": "product_1q", "match": desc,
                    "fidelity": 1.0, "min_gates": mg, "min_2q": 0,
                })
                result["n_product"] += 1
                result["n_entangled"] -= 1
            elif nb == 2:
                stype, match, fid, mg, m2q = _identify_2q(sub_state)
                subsystems.append({
                    "qubits": blk, "type": stype, "match": match,
                    "fidelity": fid, "min_gates": mg, "min_2q": m2q,
                })
            else:
                stype, match, fid, mg, m2q = _identify_nq(sub_state, nb)
                subsystems.append({
                    "qubits": blk, "type": stype, "match": match,
                    "fidelity": fid, "min_gates": mg, "min_2q": m2q,
                })

    result["subsystems"] = subsystems
    result["total_min_gates"] = sum(s["min_gates"] for s in subsystems)
    result["total_min_2q"] = sum(s["min_2q"] for s in subsystems)
    tmg = result["total_min_gates"]
    if tmg > 0:
        result["gate_overhead"] = n_g / tmg
    elif n_g > 0:
        result["gate_overhead"] = float("inf")

    # Description
    parts = []
    for s in subsystems:
        if s["type"] == "idle":
            continue
        qs = ",".join(str(q) for q in s["qubits"])
        if s["type"] == "product_1q":
            parts.append(f"q{qs}:{s['match']}")
        else:
            parts.append(f"q[{qs}]:{s['type']}")
    if result["n_idle"] > 0:
        parts.append(f"{result['n_idle']} idle")
    result["description"] = " | ".join(parts) if parts else "identity"
    return result


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Entangling Power & EPD
# ═══════════════════════════════════════════════════════════════════════


def _random_product_state(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random product state."""
    state = np.array([1.0 + 0j])
    for _ in range(n):
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(0, 2 * np.pi)
        q = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
        state = np.kron(state, q)
    return state


def _half_cut_entropy(sv: np.ndarray, n: int) -> float:
    """Von Neumann entropy across half-cut."""
    half = n // 2
    mat = sv.reshape(2**half, 2**(n - half))
    s = np.linalg.svd(mat, compute_uv=False)
    s = s[s > 1e-12]
    probs = s**2
    return float(-np.sum(probs * np.log2(probs)))


def compute_entangling_power(U: np.ndarray, n: int, n_samples: int = EP_SAMPLES) -> dict:
    """Entangling power (avg entropy) and EPD (std) over random product inputs."""
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


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Best Input Search
# ═══════════════════════════════════════════════════════════════════════


def scan_best_input(U: np.ndarray, n: int, n_samples: int = BEST_INPUT_SAMPLES) -> dict:
    """Find product-state input that maximises output entanglement."""
    rng = np.random.default_rng(123)
    best_ent = 0.0
    best_conc = 0.0

    for _ in range(n_samples):
        psi_in = _random_product_state(n, rng)
        psi_out = U @ psi_in
        ent = _half_cut_entropy(psi_out, n)
        if ent > best_ent:
            best_ent = ent

        # Concurrence for first 2 qubits
        if n >= 2:
            try:
                from qiskit.quantum_info import Statevector, concurrence, partial_trace
                sv = Statevector(psi_out)
                others = [q for q in range(n) if q >= 2]
                rho2 = partial_trace(sv, others) if others else sv.to_operator()
                conc = float(concurrence(rho2))
                if conc > best_conc:
                    best_conc = conc
            except Exception:
                pass

    return {
        "best_entropy": best_ent,
        "best_concurrence": best_conc,
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 6: Hamiltonian Reverse-Engineering
# ═══════════════════════════════════════════════════════════════════════

_PAULI_1Q = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_matrix(label: str) -> np.ndarray:
    """Build n-qubit Pauli matrix from label like 'XIZZ'."""
    result = np.array([[1.0 + 0j]])
    for ch in label:
        result = np.kron(result, _PAULI_1Q[ch])
    return result


def pauli_decompose(H: np.ndarray, n: int) -> dict[str, float]:
    """Decompose Hermitian H into Pauli coefficients. O(4^n)."""
    d = 2**n
    coeffs = {}
    for labels in itertools.product("IXYZ", repeat=n):
        label = "".join(labels)
        P = _pauli_matrix(label)
        c = np.real(np.trace(P @ H)) / d
        if abs(c) > 1e-6:
            coeffs[label] = float(c)
    return coeffs


def _reference_hamiltonians(n: int) -> dict[str, dict[str, float]]:
    """Build Pauli decompositions of reference Hamiltonians."""
    refs = {}

    def _label(n, pairs):
        l = ['I'] * n
        for q, p in pairs:
            l[q] = p
        return ''.join(l)

    # Heisenberg: sum_{<i,j>} XX + YY + ZZ
    h_terms = {}
    for i in range(n - 1):
        for p in 'XYZ':
            h_terms[_label(n, [(i, p), (i + 1, p)])] = 1.0
    refs["heisenberg"] = h_terms

    # Ising: sum Z_i Z_{i+1}
    ising = {}
    for i in range(n - 1):
        ising[_label(n, [(i, 'Z'), (i + 1, 'Z')])] = 1.0
    refs["ising_zz"] = ising

    # TFIM: -sum Z_i Z_{i+1} - sum X_i
    tfim = {}
    for i in range(n - 1):
        tfim[_label(n, [(i, 'Z'), (i + 1, 'Z')])] = -1.0
    for i in range(n):
        tfim[_label(n, [(i, 'X')])] = -1.0
    refs["tfim"] = tfim

    # XY model: sum X_i X_{i+1} + Y_i Y_{i+1}
    xy = {}
    for i in range(n - 1):
        for p in 'XY':
            xy[_label(n, [(i, p), (i + 1, p)])] = 1.0
    refs["xy_model"] = xy

    return refs


def _pauli_cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two Pauli decompositions."""
    all_keys = set(a) | set(b)
    # Remove identity
    all_keys.discard('I' * len(next(iter(a), 'I')))
    if not all_keys:
        return 0.0
    va = np.array([a.get(k, 0) for k in all_keys])
    vb = np.array([b.get(k, 0) for k in all_keys])
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def analyze_hamiltonian(U: np.ndarray, n: int) -> dict:
    """Extract H from U = exp(-iH), Pauli-decompose, match references."""
    try:
        # Eigendecomposition-based log avoids logm branch-cut issues
        eigenvalues, V = np.linalg.eig(U)
        # Force eigenvalues onto unit circle (remove numerical noise)
        eigenvalues = eigenvalues / np.abs(eigenvalues)
        # Extract phases bounded to [-pi, pi]
        phases = np.angle(eigenvalues)
        # H = V @ diag(-phases) @ V^-1  (since U = exp(-iH))
        H = V @ np.diag(-phases) @ np.linalg.inv(V)
        # Enforce Hermiticity
        H = (H + H.conj().T) / 2
        # Sanity check: reject garbage coefficients
        if np.max(np.abs(H)) > 100:
            return {"error": "H coefficients too large (>100), likely degenerate spectrum"}
    except Exception:
        return {"error": "eigendecomposition failed"}

    coeffs = pauli_decompose(H, n)
    if not coeffs:
        return {"error": "trivial Hamiltonian"}

    # Remove identity component
    id_label = 'I' * n
    coeffs.pop(id_label, None)

    # Sort by magnitude
    sorted_terms = sorted(coeffs.items(), key=lambda x: -abs(x[1]))
    top_terms = sorted_terms[:10]

    # Classify interaction types
    n_body_counts = {}
    for label, c in coeffs.items():
        weight = sum(1 for ch in label if ch != 'I')
        n_body_counts[weight] = n_body_counts.get(weight, 0) + abs(c)

    # Match against references
    refs = _reference_hamiltonians(n)
    matches = {}
    for ref_name, ref_coeffs in refs.items():
        sim = _pauli_cosine_sim(coeffs, ref_coeffs)
        matches[ref_name] = sim

    best_match = max(matches, key=matches.get) if matches else "none"

    return {
        "top_terms": [(l, c) for l, c in top_terms],
        "n_terms": len(coeffs),
        "n_body_distribution": n_body_counts,
        "matches": matches,
        "best_match": best_match,
        "best_match_score": matches.get(best_match, 0),
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 7: Symmetry Detection
# ═══════════════════════════════════════════════════════════════════════


def detect_symmetries(U: np.ndarray, n: int) -> list[str]:
    """Find Pauli strings that commute with U (weight 1 and 2)."""
    symmetries = []

    # Weight-1 Pauli strings
    for q in range(n):
        for p in 'XYZ':
            label = ['I'] * n
            label[q] = p
            P = _pauli_matrix(''.join(label))
            comm = U @ P - P @ U
            if np.linalg.norm(comm) < 1e-8:
                symmetries.append(''.join(label))

    # Weight-2 Pauli strings
    for q1 in range(n):
        for q2 in range(q1 + 1, n):
            for p1 in 'XYZ':
                for p2 in 'XYZ':
                    label = ['I'] * n
                    label[q1] = p1
                    label[q2] = p2
                    P = _pauli_matrix(''.join(label))
                    comm = U @ P - P @ U
                    if np.linalg.norm(comm) < 1e-8:
                        symmetries.append(''.join(label))

    # Weight-n (check full Pauli strings up to weight n for small n)
    if n <= 5:
        for labels in itertools.product("XYZ", repeat=n):
            label = ''.join(labels)
            if label in symmetries:
                continue
            weight = sum(1 for ch in label if ch != 'I')
            if weight <= 2:
                continue  # already checked
            P = _pauli_matrix(label)
            comm = U @ P - P @ U
            if np.linalg.norm(comm) < 1e-8:
                symmetries.append(label)

    return symmetries


# ═══════════════════════════════════════════════════════════════════════
# Phase 8: Operator Entanglement
# ═══════════════════════════════════════════════════════════════════════


def compute_operator_entanglement(U: np.ndarray, n: int) -> float:
    """Operator entanglement across half-cut (Schmidt decomp of U as bipartite op)."""
    half_a = n // 2
    half_b = n - half_a
    d_a = 2**half_a
    d_b = 2**half_b

    # Reshape U as (d_a, d_b, d_a, d_b) then to (d_a^2, d_b^2)
    U_reshaped = U.reshape(d_a, d_b, d_a, d_b)
    U_reshaped = U_reshaped.transpose(0, 2, 1, 3).reshape(d_a**2, d_b**2)
    s = np.linalg.svd(U_reshaped, compute_uv=False)
    s = s / np.linalg.norm(s)
    s_sq = s**2
    s_sq = s_sq[s_sq > 1e-12]
    return float(-np.sum(s_sq * np.log2(s_sq)))


# ═══════════════════════════════════════════════════════════════════════
# Phase 9: Conditional State Analysis
# ═══════════════════════════════════════════════════════════════════════


def analyze_conditional_states(
    name: str, qc: QuantumCircuit, output_analysis: dict,
) -> dict | None:
    """For circuits with spare qubits, measure them and check conditional states."""
    from qiskit.quantum_info import Statevector, concurrence, partial_trace

    n = qc.num_qubits
    if n > MAX_QUBITS_STATEVECTOR or n < 3:
        return None

    # Identify "spare" qubits: idle or product qubits that could be ancillae
    spare = []
    data = []
    for s in output_analysis.get("subsystems", []):
        if s["type"] in ("idle", "product_1q"):
            spare.extend(s["qubits"])
        else:
            data.extend(s["qubits"])

    if not spare or len(data) < 2:
        return None

    try:
        sv = Statevector.from_instruction(qc)
    except Exception:
        return None

    # For each measurement outcome on spare qubits, compute conditional state on data qubits
    n_spare = len(spare)
    outcomes = []
    for outcome_bits in range(2**n_spare):
        # Probability and conditional state
        # Project spare qubits onto |outcome_bits>
        proj_state = np.array(sv.data).copy()
        prob = 0.0

        # Build projector for this outcome on spare qubits
        # Use Qiskit little-endian: for each spare qubit, keep only amplitudes
        # where that qubit matches the outcome bit
        mask = proj_state.copy()
        for idx_s, sq in enumerate(spare):
            bit_val = (outcome_bits >> idx_s) & 1
            for amp_idx in range(len(mask)):
                # Qubit sq has value = bit sq of amp_idx (little-endian)
                qubit_val = (amp_idx >> sq) & 1
                if qubit_val != bit_val:
                    mask[amp_idx] = 0.0

        prob = float(np.sum(np.abs(mask) ** 2))
        if prob < 1e-12:
            continue

        # Normalise conditional state
        cond = mask / np.sqrt(prob)

        # Trace out spare qubits to get data-qubit state
        cond_sv = Statevector(cond)
        rho_data = partial_trace(cond_sv, spare)

        # Concurrence of data qubits (if exactly 2)
        conc = 0.0
        if len(data) == 2:
            try:
                conc = float(concurrence(rho_data))
            except Exception:
                pass
        else:
            # Purity as proxy for how mixed the conditional state is
            rho_arr = np.array(rho_data.data)
            conc = float(np.real(np.trace(rho_arr @ rho_arr)))  # purity

        outcomes.append({
            "outcome": format(outcome_bits, f"0{n_spare}b"),
            "probability": prob,
            "concurrence_or_purity": conc,
        })

    if not outcomes:
        return None

    # Key finding: does EVERY outcome yield the same high entanglement?
    concs = [o["concurrence_or_purity"] for o in outcomes if o["probability"] > 0.01]
    all_high = all(c > 0.9 for c in concs) if concs else False
    any_high = any(c > 0.9 for c in concs) if concs else False

    return {
        "spare_qubits": spare,
        "data_qubits": data,
        "n_outcomes": len(outcomes),
        "outcomes": outcomes,
        "all_high_entanglement": all_high,
        "any_high_entanglement": any_high,
        "avg_concurrence": float(np.mean(concs)) if concs else 0.0,
        "std_concurrence": float(np.std(concs)) if concs else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 10: VQE Probe
# ═══════════════════════════════════════════════════════════════════════


def _build_hamiltonian_matrix(n: int, model: str = "heisenberg") -> np.ndarray:
    """Build Hamiltonian matrix for VQE test."""
    d = 2**n
    H = np.zeros((d, d), dtype=complex)

    def _label(pairs):
        l = ['I'] * n
        for q, p in pairs:
            l[q] = p
        return ''.join(l)

    if model == "heisenberg":
        for i in range(n - 1):
            for p in 'XYZ':
                H += _pauli_matrix(_label([(i, p), (i + 1, p)]))
    elif model == "tfim":
        for i in range(n - 1):
            H -= _pauli_matrix(_label([(i, 'Z'), (i + 1, 'Z')]))
        for i in range(n):
            H -= _pauli_matrix(_label([(i, 'X')]))

    return H


def _cx_chain_entangler(n: int, n_2q: int) -> QuantumCircuit:
    """Build CX-chain entangling layer with given 2Q gate count."""
    qc = QuantumCircuit(n)
    placed = 0
    while placed < n_2q:
        for i in range(min(n - 1, n_2q - placed)):
            qc.cx(i, i + 1)
            placed += 1
            if placed >= n_2q:
                break
    return qc


def _hea_entangler(n: int, n_2q_target: int) -> QuantumCircuit:
    """CZ brick-layer entangler (alternating even/odd connectivity)."""
    qc = QuantumCircuit(n)
    placed = 0
    layer = 0
    while placed < n_2q_target:
        start = layer % 2  # alternate even/odd layers
        for i in range(start, n - 1, 2):
            qc.cz(i, i + 1)
            placed += 1
            if placed >= n_2q_target:
                break
        layer += 1
    return qc


def vqe_test(
    entangler_qc: QuantumCircuit,
    n_qubits: int,
    H_matrix: np.ndarray,
    n_restarts: int = VQE_RESTARTS,
    maxiter: int = VQE_MAXITER,
) -> float:
    """Run VQE with given entangling layer, return best energy found."""
    from scipy.optimize import minimize
    from qiskit.quantum_info import Statevector

    n_params = 4 * n_qubits  # 2 rotation params per qubit, 2 layers

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

    rng = np.random.default_rng(42)
    best_energy = float("inf")
    for _ in range(n_restarts):
        x0 = rng.uniform(-np.pi, np.pi, n_params)
        try:
            res = minimize(energy, x0, method="COBYLA",
                           options={"maxiter": maxiter, "rhobeg": 0.5})
            if res.fun < best_energy:
                best_energy = res.fun
        except Exception:
            pass

    return best_energy


def run_vqe_probe(
    candidates: list[tuple[str, QuantumCircuit, str]],
    ep_results: dict,
) -> dict:
    """Run VQE on top candidates by entangling power. Returns results dict."""
    # Group candidates by qubit count, pick top by EP
    by_nq: dict[int, list] = {}
    for name, qc, cat in candidates:
        nq = qc.num_qubits
        ep = ep_results.get(name, {}).get("entangling_power", 0)
        by_nq.setdefault(nq, []).append((name, qc, cat, ep))

    results = {}
    for nq in sorted(by_nq):
        if nq < 2 or nq > 6:
            continue

        # Sort by EP descending, take top
        group = sorted(by_nq[nq], key=lambda x: -x[3])[:VQE_TOP_K]
        if not group:
            continue

        # Build Hamiltonians and exact ground state
        for model in ["heisenberg"]:
            H = _build_hamiltonian_matrix(nq, model)
            evals = np.linalg.eigvalsh(H)
            exact_gs = float(evals[0])

            for name, qc, cat, ep in group:
                n_2q = _count_2q(qc)
                # Candidate VQE
                cand_energy = vqe_test(qc, nq, H)
                # Baseline: CX-chain with same 2Q gates
                baseline_qc = _cx_chain_entangler(nq, max(1, n_2q))
                base_energy = vqe_test(baseline_qc, nq, H)
                # No-entangler baseline
                no_ent_qc = QuantumCircuit(nq)
                no_ent_energy = vqe_test(no_ent_qc, nq, H)

                cand_err = abs(cand_energy - exact_gs) / abs(exact_gs) if exact_gs != 0 else 0
                base_err = abs(base_energy - exact_gs) / abs(exact_gs) if exact_gs != 0 else 0
                no_ent_err = abs(no_ent_energy - exact_gs) / abs(exact_gs) if exact_gs != 0 else 0

                results[name] = {
                    "model": model,
                    "n_qubits": nq,
                    "n_2q_gates": n_2q,
                    "exact_gs": exact_gs,
                    "candidate_energy": cand_energy,
                    "candidate_error": cand_err,
                    "baseline_energy": base_energy,
                    "baseline_error": base_err,
                    "no_ent_energy": no_ent_energy,
                    "no_ent_error": no_ent_err,
                    "beats_baseline": cand_err < base_err,
                    "improvement_vs_baseline": base_err - cand_err,
                }

    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 10b: Deep VQE Benchmark
# ═══════════════════════════════════════════════════════════════════════


def run_deep_vqe_benchmark(
    vqe_results: dict,
    scored: dict,
    templates: dict,
) -> dict:
    """Rigorous VQE benchmark for auto-selected top candidates at multiple qubit sizes.

    Selects the top DEEP_VQE_TOP_K candidates from the quick probe (by improvement
    over baseline), then tests each at multiple qubit sizes and Hamiltonians with
    higher restarts/iterations. Compares against CX-chain and HEA baselines.
    """
    # Auto-select top candidates by improvement over baseline
    ranked = sorted(
        vqe_results.items(),
        key=lambda x: x[1].get("improvement_vs_baseline", 0),
        reverse=True,
    )
    targets = [name for name, v in ranked if v.get("beats_baseline")][:DEEP_VQE_TOP_K]
    if not targets:
        # Fall back to top-K by lowest candidate error
        targets = [name for name, _ in sorted(
            ranked, key=lambda x: x[1].get("candidate_error", 1.0),
        )][:DEEP_VQE_TOP_K]

    if not targets:
        return {}

    # Count total runs for progress
    total_runs = len(targets) * len(DEEP_VQE_QUBIT_SIZES) * len(DEEP_VQE_MODELS) * 3
    done = 0
    results = {}

    for cand_name in targets:
        # Look up scored entry for scaling
        score_entry = scored.get(cand_name, {})
        if not score_entry:
            # Try without _q* suffix
            base = cand_name.rsplit("_q", 1)[0] if "_q" in cand_name else cand_name
            score_entry = scored.get(base, {})
        score_entry["name"] = cand_name

        for nq in DEEP_VQE_QUBIT_SIZES:
            # Scale candidate to target qubit count
            cand_qc = _scale_candidate_spec(score_entry, nq, templates)
            if cand_qc is None or cand_qc.num_qubits != nq:
                continue

            n_2q = _count_2q(cand_qc)
            if n_2q == 0:
                continue

            for model in DEEP_VQE_MODELS:
                H = _build_hamiltonian_matrix(nq, model)
                evals = np.linalg.eigvalsh(H)
                exact_gs = float(evals[0])
                if exact_gs == 0:
                    continue

                # Candidate
                done += 1
                print(f"    [{done}/{total_runs}] {cand_name} {nq}q {model} — candidate")
                cand_energy = vqe_test(
                    cand_qc, nq, H,
                    n_restarts=DEEP_VQE_RESTARTS, maxiter=DEEP_VQE_MAXITER,
                )
                cand_err = abs(cand_energy - exact_gs) / abs(exact_gs)

                # CX-chain baseline
                done += 1
                print(f"    [{done}/{total_runs}] {cand_name} {nq}q {model} — CX-chain")
                cx_qc = _cx_chain_entangler(nq, max(1, n_2q))
                cx_energy = vqe_test(
                    cx_qc, nq, H,
                    n_restarts=DEEP_VQE_RESTARTS, maxiter=DEEP_VQE_MAXITER,
                )
                cx_err = abs(cx_energy - exact_gs) / abs(exact_gs)

                # HEA baseline
                done += 1
                print(f"    [{done}/{total_runs}] {cand_name} {nq}q {model} — HEA")
                hea_qc = _hea_entangler(nq, max(1, n_2q))
                hea_energy = vqe_test(
                    hea_qc, nq, H,
                    n_restarts=DEEP_VQE_RESTARTS, maxiter=DEEP_VQE_MAXITER,
                )
                hea_err = abs(hea_energy - exact_gs) / abs(exact_gs)

                # Best baseline
                best_base_err = min(cx_err, hea_err)
                improvement_cx = cx_err - cand_err
                improvement_hea = hea_err - cand_err
                improvement_best = best_base_err - cand_err
                significant = improvement_best > SIGNIFICANT_IMPROVEMENT_PP

                key = f"{cand_name}_{nq}q_{model}"
                results[key] = {
                    "candidate": cand_name,
                    "n_qubits": nq,
                    "model": model,
                    "n_2q_gates": n_2q,
                    "exact_gs": exact_gs,
                    "candidate_error": cand_err,
                    "cx_chain_error": cx_err,
                    "hea_error": hea_err,
                    "improvement_vs_cx": improvement_cx,
                    "improvement_vs_hea": improvement_hea,
                    "improvement_vs_best_baseline": improvement_best,
                    "significant": significant,
                    "beats_cx": cand_err < cx_err,
                    "beats_hea": cand_err < hea_err,
                }

    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 11: ZX Compression
# ═══════════════════════════════════════════════════════════════════════


def compute_zx_compression(name: str, qc: QuantumCircuit) -> dict:
    try:
        snapshots = convert_at_all_levels(qc, name)
    except Exception:
        return {"raw_vertices": 0, "reduced_vertices": 0, "compression_ratio": 0.0, "t_count": 0}

    raw_v = red_v = t_count = 0
    for snap in snapshots:
        if snap.level == SimplificationLevel.RAW:
            raw_v = snap.num_vertices
            t_count = snap.num_t_gates
        if snap.level == SimplificationLevel.FULL_REDUCE:
            red_v = snap.num_vertices

    return {
        "raw_vertices": raw_v, "reduced_vertices": red_v,
        "compression_ratio": 1.0 - (red_v / max(1, raw_v)), "t_count": t_count,
    }


# ═══════════════════════════════════════════════════════════════════════
# Noise Resilience (analytical)
# ═══════════════════════════════════════════════════════════════════════


def noise_fidelity(n_1q: int, n_2q: int) -> dict[str, float]:
    return {str(p): ((1 - p)**n_2q) * ((1 - p / 10)**n_1q) for p in NOISE_RATES}


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

CAT_COLORS = {
    "standard": "#2166ac", "coverage_gap": "#e41a1c", "cross_family": "#377eb8",
    "irreducible": "#4daf4a", "pca_void": "#984ea3", "legacy": "#999999",
}
CAT_MARKERS = {
    "standard": "s", "coverage_gap": "o", "cross_family": "D",
    "irreducible": "^", "pca_void": "v", "legacy": "p",
}


def _c(cat): return CAT_COLORS.get(cat, "#333")
def _m(cat): return CAT_MARKERS.get(cat, "o")


def _legend_handles():
    return [
        Line2D([0], [0], marker=_m(c), color="w", markerfacecolor=_c(c),
               markersize=8, label=c.replace("_", " ").title())
        for c in ["standard", "coverage_gap", "cross_family", "irreducible", "pca_void", "legacy"]
    ]


def plot_overhead(analyses: dict, categories: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for name, a in analyses.items():
        cat = categories.get(name, "unknown")
        tmg = a.get("total_min_gates", 0)
        if tmg == 0:
            continue
        ax1.scatter(tmg, a["n_gates"], c=_c(cat), marker=_m(cat),
                    s=80, alpha=0.7, edgecolors="k", linewidths=0.5)
        if cat == "standard" or a.get("gate_overhead", 1) > 3:
            ax1.annotate(name.replace("std_", ""), (tmg, a["n_gates"]),
                         fontsize=6, alpha=0.7, xytext=(4, 4), textcoords="offset points")

    mx = max((a["n_gates"] for a in analyses.values()), default=10)
    ax1.plot([0, mx], [0, mx], "k--", alpha=0.3)
    ax1.set_xlabel("Minimal Equivalent Gates")
    ax1.set_ylabel("Actual Gates")
    ax1.set_title("Gate Overhead: Actual vs Minimal")
    ax1.legend(handles=_legend_handles(), fontsize=7, loc="upper left")
    ax1.grid(True, alpha=0.3)

    cands = [(n, a["gate_overhead"], categories.get(n, "?"))
             for n, a in analyses.items()
             if not n.startswith("std_") and 0 < a.get("gate_overhead", 0) < float("inf")
             and a.get("total_min_gates", 0) > 0]
    cands.sort(key=lambda x: x[1])
    if cands:
        y = np.arange(len(cands))
        ax2.barh(y, [x[1] for x in cands], color=[_c(x[2]) for x in cands],
                 alpha=0.8, edgecolor="k", linewidth=0.3)
        ax2.axvline(1.0, color="k", linestyle="--", alpha=0.3)
        ax2.set_yticks(y)
        ax2.set_yticklabels([x[0] for x in cands], fontsize=5)
        ax2.set_xlabel("Gate Overhead (actual / minimal)")
        ax2.set_title("Candidate Gate Overhead")
        ax2.invert_yaxis()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "overhead_analysis.png", dpi=150)
    plt.close(fig)


def plot_state_structure(analyses: dict, categories: dict):
    cands = [(n, a) for n, a in sorted(analyses.items()) if not n.startswith("std_")]
    if not cands:
        return
    fig, ax = plt.subplots(figsize=(12, max(6, len(cands) * 0.25)))
    names = [c[0] for c in cands]
    idle = [c[1]["n_idle"] for c in cands]
    prod = [c[1]["n_product"] for c in cands]
    ent = [c[1]["n_entangled"] for c in cands]
    y = np.arange(len(names))
    ax.barh(y, idle, 0.6, label="Idle (|0>)", color="#d9d9d9", edgecolor="k", linewidth=0.3)
    ax.barh(y, prod, 0.6, left=idle, label="Product (1Q)", color="#fc8d62", edgecolor="k", linewidth=0.3)
    left2 = [i + p for i, p in zip(idle, prod)]
    ax.barh(y, ent, 0.6, left=left2, label="Entangled", color="#66c2a5", edgecolor="k", linewidth=0.3)
    for idx, (nm, a) in enumerate(cands):
        ax.text(a["n_qubits"] + 0.2, idx, a["description"], fontsize=5, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Qubits")
    ax.set_title("Output State: What Each Candidate Produces")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "state_structure.png", dpi=150)
    plt.close(fig)


def plot_entangling_power(ep_results: dict, categories: dict, analyses: dict):
    """Scatter: entangling power vs gate count, sized by EPD."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, ep in ep_results.items():
        cat = categories.get(name, "unknown")
        n_g = analyses.get(name, {}).get("n_gates", 0)
        if n_g == 0:
            continue
        ax.scatter(n_g, ep["entangling_power"], c=_c(cat), marker=_m(cat),
                   s=40 + 200 * ep["epd"], alpha=0.7, edgecolors="k", linewidths=0.5)
        if cat == "standard" or ep["entangling_power"] > 0.5:
            ax.annotate(name.replace("std_", ""), (n_g, ep["entangling_power"]),
                        fontsize=6, alpha=0.7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Total Gates")
    ax.set_ylabel("Entangling Power (avg half-cut entropy)")
    ax.set_title("Entangling Power vs Gate Count (size = EPD)")
    ax.legend(handles=_legend_handles(), fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "entangling_power.png", dpi=150)
    plt.close(fig)


def plot_vqe_comparison(vqe_results: dict, categories: dict):
    if not vqe_results:
        return
    fig, ax = plt.subplots(figsize=(12, max(6, len(vqe_results) * 0.3)))
    items = sorted(vqe_results.items(), key=lambda x: x[1]["candidate_error"])
    names = [x[0] for x in items]
    cand_err = [x[1]["candidate_error"] for x in items]
    base_err = [x[1]["baseline_error"] for x in items]
    y = np.arange(len(names))
    ax.barh(y - 0.15, cand_err, 0.3, label="Candidate", color="#4daf4a", alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.barh(y + 0.15, base_err, 0.3, label="CX-chain baseline", color="#2166ac", alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Relative Error vs Exact Ground State")
    ax.set_title("VQE Probe: Candidate vs CX-Chain Baseline (Heisenberg)")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    # Mark winners
    for idx, (nm, vqe) in enumerate(items):
        if vqe["beats_baseline"]:
            ax.annotate("*", (max(cand_err[idx], base_err[idx]) + 0.001, idx),
                        fontsize=12, color="red", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "vqe_comparison.png", dpi=150)
    plt.close(fig)


def plot_zx_compression(zx: dict, categories: dict):
    data = [(n, z["compression_ratio"], categories.get(n, "?"))
            for n, z in zx.items() if z.get("compression_ratio", 0) > 0]
    if not data:
        return
    data.sort(key=lambda x: -x[1])
    data = data[:40]
    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(data))
    ax.barh(y, [d[1] for d in data], color=[_c(d[2]) for d in data],
            alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in data], fontsize=6)
    ax.set_xlabel("Compression Ratio")
    ax.set_title("ZX Compression: Vertex Reduction via full_reduce")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.legend(handles=_legend_handles(), fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "zx_compression.png", dpi=150)
    plt.close(fig)


def plot_deep_vqe(deep_results: dict):
    """Grouped horizontal bar chart: candidate vs CX-chain vs HEA error."""
    if not deep_results:
        return
    items = sorted(deep_results.items(), key=lambda x: x[1]["candidate_error"])
    n = len(items)
    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.5)))
    y = np.arange(n)
    h = 0.25

    cand_err = [v["candidate_error"] for _, v in items]
    cx_err = [v["cx_chain_error"] for _, v in items]
    hea_err = [v["hea_error"] for _, v in items]

    ax.barh(y - h, cand_err, h, label="Candidate", color="#4daf4a", alpha=0.8,
            edgecolor="k", linewidth=0.3)
    ax.barh(y, cx_err, h, label="CX-chain", color="#2166ac", alpha=0.8,
            edgecolor="k", linewidth=0.3)
    ax.barh(y + h, hea_err, h, label="HEA brick", color="#984ea3", alpha=0.8,
            edgecolor="k", linewidth=0.3)

    labels = []
    for key, v in items:
        label = f"{v['candidate']} {v['n_qubits']}q {v['model']}"
        if v["significant"]:
            label += " *"
        labels.append(label)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Relative Error vs Exact Ground State")
    ax.set_title("Deep VQE Benchmark: Candidate vs Baselines "
                 f"({DEEP_VQE_RESTARTS} restarts, {DEEP_VQE_MAXITER} iter)")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "deep_vqe_benchmark.png", dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════


def generate_report(
    analyses: dict, ep_results: dict, ham_results: dict,
    sym_results: dict, op_ent_results: dict, cond_results: dict,
    vqe_results: dict, best_input_results: dict, zx_metrics: dict,
    novelty: dict, categories: dict, elapsed: float,
    deep_vqe_results: dict | None = None,
) -> str:
    L = []
    std_names = [n for n in analyses if n.startswith("std_")]
    cand_names = [n for n in analyses if not n.startswith("std_")]

    L.append("=" * 80)
    L.append("COMPREHENSIVE HONEST COMPARISON: Candidates vs Industry Standards")
    L.append("=" * 80)
    L.append("")
    L.append(f"Circuits: {len(analyses)} ({len(std_names)} standards, {len(cand_names)} candidates)")
    L.append(f"Elapsed: {elapsed:.1f}s")
    L.append("")

    # ── Section 1: Executive Summary ──────────────────────────────────
    L.append("=" * 80)
    L.append("1. EXECUTIVE SUMMARY")
    L.append("=" * 80)
    L.append("")

    n_with_ent = sum(1 for n in cand_names if analyses[n]["n_entangled"] > 0)
    n_all_idle = sum(1 for n in cand_names if analyses[n]["n_idle"] == analyses[n]["n_qubits"])
    overheads = [analyses[n]["gate_overhead"] for n in cand_names
                 if 0 < analyses[n].get("gate_overhead", 0) < float("inf")]
    avg_oh = np.mean(overheads) if overheads else 0

    # EP stats
    cand_eps = [ep_results[n]["entangling_power"] for n in cand_names if n in ep_results]
    std_eps = [ep_results[n]["entangling_power"] for n in std_names if n in ep_results]
    n_ep_above_median = 0
    if std_eps and cand_eps:
        med_std_ep = np.median(std_eps)
        n_ep_above_median = sum(1 for e in cand_eps if e > med_std_ep)

    # VQE stats — split significant vs marginal
    vqe_significant = [n for n, v in vqe_results.items()
                       if v.get("beats_baseline")
                       and v.get("improvement_vs_baseline", 0) > SIGNIFICANT_IMPROVEMENT_PP]
    vqe_marginal = [n for n, v in vqe_results.items()
                    if v.get("beats_baseline")
                    and v.get("improvement_vs_baseline", 0) <= SIGNIFICANT_IMPROVEMENT_PP]
    n_vqe_wins = len(vqe_significant)
    n_vqe_marginal = len(vqe_marginal)

    # Conditional stats
    n_cond_interesting = sum(1 for c in cond_results.values()
                             if c and c.get("all_high_entanglement"))

    L.append("Key findings:")
    L.append(f"  - {n_with_ent}/{len(cand_names)} candidates produce entangled states")
    L.append(f"  - {n_all_idle}/{len(cand_names)} produce only |0...0> (circuit = identity on zero state)")
    L.append(f"  - Average gate overhead: {avg_oh:.2f}x (actual gates / minimal equivalent)")
    L.append(f"  - {n_ep_above_median}/{len(cand_eps)} candidates have entangling power above "
             f"median standard ({np.median(std_eps):.3f})" if std_eps else
             "  - No entangling power comparison possible")
    L.append(f"  - {n_vqe_wins}/{len(vqe_results)} candidates significantly beat CX-chain "
             f"baseline in VQE (>{SIGNIFICANT_IMPROVEMENT_PP*100:.0f}pp), "
             f"{n_vqe_marginal} marginal")
    L.append(f"  - {n_cond_interesting} candidates show interesting conditional-state properties")
    L.append("")

    # ── Section 2: Entangling Power Rankings ──────────────────────────
    L.append("-" * 80)
    L.append("2. ENTANGLING POWER (average entanglement over random inputs)")
    L.append("-" * 80)
    L.append("")
    L.append("EP = avg half-cut entropy over Haar-random product inputs.")
    L.append("EPD = std dev (low = input-agnostic entangler; high = input-sensitive).")
    L.append("")
    L.append(f"{'Rank':>4s}  {'Name':35s} {'EP':>7s} {'EPD':>7s} {'BestEnt':>8s} "
             f"{'Gates':>6s} {'2Q':>4s} {'Cat':>12s}")
    L.append("-" * 90)

    ep_ranked = sorted(ep_results.items(), key=lambda x: -x[1]["entangling_power"])
    for rank, (name, ep) in enumerate(ep_ranked[:30], 1):
        cat = categories.get(name, "?")
        a = analyses.get(name, {})
        bi = best_input_results.get(name, {})
        L.append(
            f"{rank:4d}  {name:35s} {ep['entangling_power']:7.4f} {ep['epd']:7.4f} "
            f"{bi.get('best_entropy', 0):8.4f} "
            f"{a.get('n_gates', 0):6d} {a.get('n_2q_gates', 0):4d} {cat:>12s}"
        )
    L.append("")

    # Any candidate above median standard?
    if std_eps:
        med = np.median(std_eps)
        above = [(n, ep_results[n]) for n in cand_names
                 if n in ep_results and ep_results[n]["entangling_power"] > med]
        if above:
            L.append(f"  Candidates above median standard EP ({med:.4f}):")
            for name, ep in sorted(above, key=lambda x: -x[1]["entangling_power"]):
                ng = analyses[name]["n_gates"]
                L.append(f"    {name}: EP={ep['entangling_power']:.4f} ({ng} gates)")
        else:
            L.append(f"  No candidate exceeds median standard EP ({med:.4f})")
    L.append("")

    # ── Section 3: Hamiltonian Matching ───────────────────────────────
    L.append("-" * 80)
    L.append("3. HAMILTONIAN REVERSE-ENGINEERING (what does U = exp(-iH) simulate?)")
    L.append("-" * 80)
    L.append("")
    L.append("For each circuit, extract H via matrix log, Pauli-decompose,")
    L.append("and match against reference Hamiltonians (Heisenberg, Ising, TFIM, XY).")
    L.append("")

    significant_matches = []
    for name in sorted(ham_results):
        hr = ham_results[name]
        if "error" in hr:
            continue
        bm = hr.get("best_match", "none")
        bms = hr.get("best_match_score", 0)
        if bms > 0.5:  # significant match
            significant_matches.append((name, bm, bms, hr))

    if significant_matches:
        L.append(f"{'Name':35s} {'Best Match':15s} {'Score':>7s} {'Terms':>6s} {'Dominant Body':>14s}")
        L.append("-" * 82)
        for name, bm, bms, hr in sorted(significant_matches, key=lambda x: -x[2]):
            cat = categories.get(name, "?")
            n_body = hr.get("n_body_distribution", {})
            dom = max(n_body, key=n_body.get) if n_body else 0
            L.append(
                f"{name:35s} {bm:15s} {bms:7.3f} {hr.get('n_terms', 0):6d} "
                f"{dom:>14d}-body"
            )
            # Show top 3 terms
            for label, coeff in hr.get("top_terms", [])[:3]:
                L.append(f"    {label}: {coeff:+.4f}")
        L.append("")
    else:
        L.append("  No significant matches found (all similarity < 0.5).")
    L.append("")

    # ── Section 4: VQE Comparison ─────────────────────────────────────
    L.append("-" * 80)
    L.append("4. VQE PROBE: Can any candidate serve as a useful entangling layer?")
    L.append("-" * 80)
    L.append("")
    L.append(f"Test: RY/RZ params → candidate → RY/RZ params, optimised for Heisenberg GS.")
    L.append(f"Baseline: same structure but CX-chain entangler with same 2Q gate count.")
    L.append(f"Settings: {VQE_RESTARTS} restarts, {VQE_MAXITER} iterations, COBYLA.")
    L.append("")

    if vqe_results:
        L.append(f"{'Name':35s} {'CandErr':>8s} {'BaseErr':>8s} {'Improv':>8s} "
                 f"{'2Q':>4s} {'Wins?':>6s}")
        L.append("-" * 78)
        for name, vqe in sorted(vqe_results.items(), key=lambda x: x[1]["candidate_error"]):
            wins = "YES" if vqe["beats_baseline"] else "no"
            L.append(
                f"{name:35s} {vqe['candidate_error']:8.4f} {vqe['baseline_error']:8.4f} "
                f"{vqe['improvement_vs_baseline']:+8.4f} {vqe['n_2q_gates']:4d} {wins:>6s}"
            )
        L.append("")
        if vqe_significant:
            L.append(f"  SIGNIFICANT VQE WINNERS ({len(vqe_significant)} candidates, "
                     f">{SIGNIFICANT_IMPROVEMENT_PP*100:.0f}pp improvement):")
            for w in vqe_significant:
                v = vqe_results[w]
                L.append(f"    {w}: error {v['candidate_error']:.4f} vs "
                         f"baseline {v['baseline_error']:.4f} "
                         f"(improvement: {v['improvement_vs_baseline']:+.4f})")
        if vqe_marginal:
            L.append(f"  MARGINAL (likely noise, <={SIGNIFICANT_IMPROVEMENT_PP*100:.0f}pp):")
            for w in vqe_marginal:
                v = vqe_results[w]
                L.append(f"    {w}: error {v['candidate_error']:.4f} vs "
                         f"baseline {v['baseline_error']:.4f} "
                         f"(improvement: {v['improvement_vs_baseline']:+.4f})")
        if not vqe_significant and not vqe_marginal:
            L.append("  No candidate beats the CX-chain baseline.")
    else:
        L.append("  VQE probe not run.")
    L.append("")

    # ── Section 4b: Deep VQE Benchmark ───────────────────────────────
    if deep_vqe_results:
        L.append("-" * 80)
        L.append("4b. DEEP VQE BENCHMARK (rigorous validation of top candidates)")
        L.append("-" * 80)
        L.append("")
        L.append(f"Settings: {DEEP_VQE_RESTARTS} restarts, {DEEP_VQE_MAXITER} iterations, "
                 f"COBYLA. Baselines: CX-chain + HEA brick-layer (same 2Q gate count).")
        L.append(f"Significance threshold: >{SIGNIFICANT_IMPROVEMENT_PP*100:.0f} "
                 f"percentage points improvement over best baseline.")
        L.append("")
        L.append(f"{'Config':45s} {'CandErr':>8s} {'CX Err':>8s} {'HEA Err':>8s} "
                 f"{'Improv':>8s} {'Sig?':>5s}")
        L.append("-" * 80)
        for key, dv in sorted(deep_vqe_results.items(),
                               key=lambda x: x[1]["candidate_error"]):
            label = f"{dv['candidate']} {dv['n_qubits']}q {dv['model']}"
            sig = "YES" if dv["significant"] else "no"
            L.append(
                f"{label:45s} {dv['candidate_error']:8.4f} {dv['cx_chain_error']:8.4f} "
                f"{dv['hea_error']:8.4f} {dv['improvement_vs_best_baseline']:+8.4f} "
                f"{sig:>5s}"
            )
        L.append("")
        n_sig = sum(1 for v in deep_vqe_results.values() if v["significant"])
        if n_sig > 0:
            L.append(f"  {n_sig} configurations show SIGNIFICANT improvement "
                     f"(confirmed at higher restarts/iterations).")
        else:
            L.append("  No configuration shows significant improvement in deep benchmark.")
        L.append("")

    # ── Section 5: Symmetry Detection ─────────────────────────────────
    L.append("-" * 80)
    L.append("5. SYMMETRY DETECTION (Pauli strings commuting with U)")
    L.append("-" * 80)
    L.append("")

    circuits_with_sym = [(n, s) for n, s in sym_results.items() if s]
    if circuits_with_sym:
        L.append(f"{'Name':35s} {'#Sym':>5s} {'Notable':40s}")
        L.append("-" * 85)
        for name, syms in sorted(circuits_with_sym, key=lambda x: -len(x[1])):
            # Highlight high-weight symmetries
            notable = [s for s in syms if sum(1 for c in s if c != 'I') >= 2]
            notable_str = ", ".join(notable[:3]) if notable else "(weight-1 only)"
            L.append(f"{name:35s} {len(syms):5d} {notable_str:40s}")
    else:
        L.append("  No symmetries found.")
    L.append("")

    # ── Section 6: Conditional State Analysis ─────────────────────────
    L.append("-" * 80)
    L.append("6. CONDITIONAL STATE ANALYSIS (measure spare qubits)")
    L.append("-" * 80)
    L.append("")

    interesting_cond = [(n, c) for n, c in cond_results.items()
                        if c and c.get("any_high_entanglement")]
    if interesting_cond:
        for name, cr in interesting_cond:
            L.append(f"  {name}:")
            L.append(f"    Spare qubits: {cr['spare_qubits']}, Data qubits: {cr['data_qubits']}")
            L.append(f"    Avg concurrence/purity: {cr['avg_concurrence']:.4f} "
                     f"(std: {cr['std_concurrence']:.4f})")
            all_high = "YES" if cr["all_high_entanglement"] else "no"
            L.append(f"    All outcomes high entanglement: {all_high}")
            for o in cr["outcomes"][:4]:
                L.append(f"      |{o['outcome']}>: prob={o['probability']:.3f}, "
                         f"C/P={o['concurrence_or_purity']:.4f}")
            L.append("")
    else:
        L.append("  No interesting conditional-state properties found.")
        L.append("  (No circuit produces high entanglement on data qubits")
        L.append("   conditional on measurement of spare qubits.)")
    L.append("")

    # ── Section 7: Operator Entanglement ──────────────────────────────
    L.append("-" * 80)
    L.append("7. OPERATOR ENTANGLEMENT (how entangling is U as a bipartite operator?)")
    L.append("-" * 80)
    L.append("")
    oe_ranked = sorted(op_ent_results.items(), key=lambda x: -x[1])
    if oe_ranked:
        L.append(f"{'Rank':>4s}  {'Name':35s} {'OpEnt':>8s} {'Gates':>6s} {'2Q':>4s} {'Cat':>12s}")
        for rank, (name, oe) in enumerate(oe_ranked[:20], 1):
            a = analyses.get(name, {})
            cat = categories.get(name, "?")
            L.append(f"{rank:4d}  {name:35s} {oe:8.4f} "
                     f"{a.get('n_gates', 0):6d} {a.get('n_2q_gates', 0):4d} {cat:>12s}")
    L.append("")

    # ── Section 8: Output State & Overhead ────────────────────────────
    L.append("-" * 80)
    L.append("8. OUTPUT STATE ANALYSIS & GATE OVERHEAD")
    L.append("-" * 80)
    L.append("")
    L.append(f"{'Name':35s} {'Q':>3s} {'Gates':>6s} {'Idle':>5s} {'Ent':>4s} "
             f"{'MinG':>5s} {'OH':>6s} {'Output':40s}")
    L.append("-" * 110)
    for name in sorted(cand_names):
        a = analyses[name]
        oh = f"{a['gate_overhead']:.1f}x" if 0 < a.get("gate_overhead", 0) < float("inf") else "n/a"
        L.append(
            f"{name:35s} {a['n_qubits']:3d} {a['n_gates']:6d} "
            f"{a['n_idle']:5d} {a['n_entangled']:4d} "
            f"{a['total_min_gates']:5d} {oh:>6s} {a['description']:40s}"
        )
    L.append("")

    # ── Section 9: Per-Candidate Profiles (top 15 by EP) ─────────────
    L.append("-" * 80)
    L.append("9. CANDIDATE PROFILES (top 15 by entangling power)")
    L.append("-" * 80)
    L.append("")

    top_by_ep = sorted(
        [(n, ep_results.get(n, {}).get("entangling_power", 0)) for n in cand_names],
        key=lambda x: -x[1]
    )[:15]

    for name, ep_val in top_by_ep:
        a = analyses[name]
        ep = ep_results.get(name, {})
        bi = best_input_results.get(name, {})
        hr = ham_results.get(name, {})
        sym = sym_results.get(name, [])
        oe = op_ent_results.get(name, 0)
        cr = cond_results.get(name)
        vqe = vqe_results.get(name)
        nov = novelty.get(name, {})

        L.append(f"  {name}")
        L.append(f"    Output:  {a['description']}")
        oh = f"{a['gate_overhead']:.1f}x ({a['n_gates']}g vs {a['total_min_gates']} min)" \
            if 0 < a.get("gate_overhead", 0) < float("inf") else "n/a"
        L.append(f"    Overhead: {oh}")
        L.append(f"    EP: {ep.get('entangling_power', 0):.4f} "
                 f"(EPD: {ep.get('epd', 0):.4f}, best input: {bi.get('best_entropy', 0):.4f})")
        L.append(f"    OpEnt: {oe:.4f}")

        if hr and "error" not in hr:
            bm = hr.get("best_match", "none")
            bms = hr.get("best_match_score", 0)
            L.append(f"    Hamiltonian: best match = {bm} ({bms:.3f})")
            for label, c in hr.get("top_terms", [])[:3]:
                L.append(f"      {label}: {c:+.4f}")

        if sym:
            hw = [s for s in sym if sum(1 for c in s if c != 'I') >= 2]
            L.append(f"    Symmetries: {len(sym)} total"
                     + (f", notable: {', '.join(hw[:3])}" if hw else ""))

        if cr and cr.get("any_high_entanglement"):
            L.append(f"    Conditional: avg C/P={cr['avg_concurrence']:.4f}, "
                     f"all_high={cr['all_high_entanglement']}")

        if vqe:
            L.append(f"    VQE: error={vqe['candidate_error']:.4f} vs "
                     f"baseline={vqe['baseline_error']:.4f} "
                     f"({'WINS' if vqe['beats_baseline'] else 'loses'})")

        if nov:
            L.append(f"    Novelty: composite={nov.get('composite_novelty', 0):.3f}")

        # Best property
        best_props = []
        if ep.get("entangling_power", 0) > 0.3:
            best_props.append(f"EP={ep['entangling_power']:.3f}")
        if oe > 2.0:
            best_props.append(f"OpEnt={oe:.3f}")
        if hr and hr.get("best_match_score", 0) > 0.7:
            best_props.append(f"Hamiltonian~{hr['best_match']}")
        if vqe and vqe["beats_baseline"]:
            best_props.append("VQE winner")
        if cr and cr.get("all_high_entanglement"):
            best_props.append("deterministic conditional entanglement")
        L.append(f"    Best property: {', '.join(best_props) if best_props else 'none identified'}")
        L.append("")

    # ── Section 10: Honest Verdict ────────────────────────────────────
    L.append("=" * 80)
    L.append("10. HONEST VERDICT")
    L.append("=" * 80)
    L.append("")

    # Collect all genuine advantages found
    advantages = []

    # Only count significant quick-probe wins (>5pp)
    if vqe_significant:
        for w in vqe_significant:
            v = vqe_results[w]
            advantages.append(
                f"VQE (quick): {w} significantly beats CX-chain by "
                f"{v['improvement_vs_baseline']:+.4f} relative error on Heisenberg"
            )

    # Deep VQE confirmed wins
    if deep_vqe_results:
        for key, dv in deep_vqe_results.items():
            if dv["significant"]:
                advantages.append(
                    f"VQE (deep): {dv['candidate']} {dv['n_qubits']}q {dv['model']} "
                    f"beats best baseline by {dv['improvement_vs_best_baseline']:+.4f} "
                    f"({DEEP_VQE_RESTARTS} restarts, {DEEP_VQE_MAXITER} iter)"
                )

    if n_cond_interesting > 0:
        for name, cr in cond_results.items():
            if cr and cr.get("all_high_entanglement"):
                advantages.append(
                    f"Conditional: {name} produces high entanglement for ALL "
                    f"measurement outcomes (DEB-like property)"
                )

    # EP advantage
    if std_eps:
        med = np.median(std_eps)
        for name in cand_names:
            ep = ep_results.get(name, {})
            if ep.get("entangling_power", 0) > med:
                a = analyses[name]
                # But is it efficient?
                std_at_similar_gates = [
                    ep_results[sn]["entangling_power"] for sn in std_names
                    if sn in ep_results and analyses.get(sn, {}).get("n_gates", 999) >= a["n_gates"] * 0.8
                ]
                if std_at_similar_gates and ep["entangling_power"] > max(std_at_similar_gates):
                    advantages.append(
                        f"EP: {name} has higher entangling power ({ep['entangling_power']:.4f}) "
                        f"than any standard at similar gate count"
                    )

    # Hamiltonian matching
    for name, hr in ham_results.items():
        if name in std_names:
            continue
        if hr.get("best_match_score", 0) > 0.8:
            advantages.append(
                f"Hamiltonian: {name} closely matches {hr['best_match']} "
                f"(similarity {hr['best_match_score']:.3f})"
            )

    if advantages:
        L.append(f"GENUINE ADVANTAGES FOUND ({len(advantages)}):")
        L.append("")
        for adv in advantages:
            L.append(f"  * {adv}")
        L.append("")
        L.append("These are provisional — each needs further investigation:")
        L.append("  - VQE wins: test with more restarts, larger systems, different Hamiltonians")
        L.append("  - Conditional properties: verify robustness under noise")
        L.append("  - EP advantages: compare against optimised circuits at same gate budget")
        L.append("  - Hamiltonian matches: verify Trotter step efficiency vs standard decomposition")
    else:
        L.append("NO genuine advantages found on any evaluation axis.")
        L.append("")
        L.append("Axes tested:")
        L.append("  1. Output state analysis:     All candidates have gate overhead >= 1.5x")
        L.append("  2. Entangling power:          No candidate exceeds standards at same gate count")
        L.append("  3. VQE probe:                 No candidate beats CX-chain entangling layer")
        L.append("  4. Hamiltonian matching:       No close match to known physical Hamiltonians")
        L.append("  5. Conditional states:         No DEB-like deterministic properties")
        L.append("  6. Symmetry:                   No useful symmetry groups detected")
        L.append("")
        L.append("What the pipeline demonstrates:")
        L.append("  + Valid circuit construction from motif building blocks")
        L.append("  + Structural novelty detection in fingerprint space")
        L.append("  + ZX simplification trajectory and motif survival analysis")
        L.append("  - Gap between structural novelty and operational utility remains open")

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

    # ── Phase 1: Build circuits ───────────────────────────────────────
    print("Phase 1: Building circuits...")
    discovery_results = json.loads((DISCOVERY_DIR / "results.json").read_text())
    print(f"  Discovery: {discovery_results['n_candidates']} candidates")

    print("  Rebuilding corpus & templates...")
    t1 = time.time()
    phylo_results, existing_freq_df = load_phylogeny_results()
    survivors = phylo_results["cross_level_survival"]["survivors_at_full_reduce"]
    corpus = build_corpus()
    motifs = discover_motifs(corpus)
    templates = build_template_registry(motifs, existing_freq_df, corpus, survivors)
    print(f"  Done ({time.time() - t1:.1f}s)")

    standards = build_standard_circuits()
    candidates = rebuild_candidates(discovery_results, templates)
    legacy = build_legacy_circuits()
    print(f"  Standards: {len(standards)}, Candidates: {len(candidates)}, Legacy: {len(legacy)}")

    all_circuits = standards + candidates + legacy
    categories = {}
    for name, _, cat in all_circuits:
        categories[name] = "standard" if name.startswith("std_") else cat

    print(f"  Total: {len(all_circuits)} circuits")

    # ── Phase 2: Compute unitaries ────────────────────────────────────
    print("\nPhase 2: Computing unitary matrices...")
    t2 = time.time()
    unitaries = compute_unitaries(all_circuits)
    print(f"  {len(unitaries)} unitaries computed ({time.time() - t2:.1f}s)")

    # ── Phase 3: Output state analysis ────────────────────────────────
    print("\nPhase 3: Output state analysis (SVD factorisation)...")
    t3 = time.time()
    analyses = {}
    for name, qc, _ in all_circuits:
        analyses[name] = analyze_output_state(name, qc)

    cand_names = [n for n in analyses if not n.startswith("std_")]
    n_ent = sum(1 for n in cand_names if analyses[n]["n_entangled"] > 0)
    n_idle = sum(1 for n in cand_names if analyses[n]["n_idle"] == analyses[n]["n_qubits"])
    print(f"  {n_ent} with entanglement, {n_idle} all-idle ({time.time() - t3:.1f}s)")

    # ── Phase 4: Entangling power ─────────────────────────────────────
    print("\nPhase 4: Entangling power & EPD...")
    t4 = time.time()
    ep_results = {}
    for name in unitaries:
        n = analyses[name]["n_qubits"]
        ep_results[name] = compute_entangling_power(unitaries[name], n)
    print(f"  {len(ep_results)} circuits ({time.time() - t4:.1f}s)")

    # ── Phase 5: Best input search ────────────────────────────────────
    print("\nPhase 5: Scanning best input states...")
    t5 = time.time()
    best_input_results = {}
    for name in unitaries:
        n = analyses[name]["n_qubits"]
        best_input_results[name] = scan_best_input(unitaries[name], n)
    print(f"  Done ({time.time() - t5:.1f}s)")

    # ── Phase 6: Hamiltonian reverse-engineering ──────────────────────
    print("\nPhase 6: Hamiltonian reverse-engineering...")
    t6 = time.time()
    ham_results = {}
    for name, U in unitaries.items():
        n = analyses[name]["n_qubits"]
        if n <= MAX_QUBITS_HAMILTONIAN:
            ham_results[name] = analyze_hamiltonian(U, n)
    print(f"  {len(ham_results)} circuits ({time.time() - t6:.1f}s)")

    # ── Phase 7: Symmetry detection ───────────────────────────────────
    print("\nPhase 7: Symmetry detection...")
    t7 = time.time()
    sym_results = {}
    for name, U in unitaries.items():
        n = analyses[name]["n_qubits"]
        sym_results[name] = detect_symmetries(U, n)
    n_with_sym = sum(1 for s in sym_results.values() if s)
    print(f"  {n_with_sym} circuits have symmetries ({time.time() - t7:.1f}s)")

    # ── Phase 8: Operator entanglement ────────────────────────────────
    print("\nPhase 8: Operator entanglement...")
    t8 = time.time()
    op_ent_results = {}
    for name, U in unitaries.items():
        n = analyses[name]["n_qubits"]
        if n >= 2:
            op_ent_results[name] = compute_operator_entanglement(U, n)
    print(f"  Done ({time.time() - t8:.1f}s)")

    # ── Phase 9: Conditional state analysis ───────────────────────────
    print("\nPhase 9: Conditional state analysis...")
    t9 = time.time()
    cond_results = {}
    for name, qc, _ in all_circuits:
        if name.startswith("std_"):
            continue
        cond_results[name] = analyze_conditional_states(name, qc, analyses[name])
    n_interesting = sum(1 for c in cond_results.values() if c and c.get("any_high_entanglement"))
    print(f"  {n_interesting} interesting ({time.time() - t9:.1f}s)")

    # ── Phase 10: VQE probe ───────────────────────────────────────────
    print(f"\nPhase 10: VQE probe (top {VQE_TOP_K} candidates by EP)...")
    t10 = time.time()
    cand_circuits = [(n, qc, cat) for n, qc, cat in all_circuits if not n.startswith("std_")]
    vqe_results = run_vqe_probe(cand_circuits, ep_results)
    n_wins = sum(1 for v in vqe_results.values() if v["beats_baseline"])
    print(f"  {len(vqe_results)} tested, {n_wins} beat baseline ({time.time() - t10:.1f}s)")

    # ── Phase 10b: Deep VQE benchmark ────────────────────────────────
    print(f"\nPhase 10b: Deep VQE benchmark (auto-selecting top {DEEP_VQE_TOP_K} "
          f"from quick probe)...")
    t10b = time.time()
    scored = discovery_results.get("scored", {})
    deep_vqe_results = run_deep_vqe_benchmark(vqe_results, scored, templates)
    n_deep_sig = sum(1 for v in deep_vqe_results.values() if v["significant"])
    print(f"  {len(deep_vqe_results)} configs tested, {n_deep_sig} significant "
          f"({time.time() - t10b:.1f}s)")

    # ── Phase 11: ZX compression ──────────────────────────────────────
    print("\nPhase 11: ZX compression...")
    t11 = time.time()
    zx_metrics = {}
    for name, qc, _ in all_circuits:
        zx_metrics[name] = compute_zx_compression(name, qc)
    print(f"  Done ({time.time() - t11:.1f}s)")

    # ── Load novelty scores ───────────────────────────────────────────
    novelty = {}
    for section in ["scored", "legacy_scored"]:
        for cname, info in discovery_results.get(section, {}).items():
            novelty[cname] = {
                "cosine_novelty": info.get("cosine_novelty", 0.0),
                "pca_isolation": info.get("pca_isolation", 0.0),
                "motif_diversity": info.get("motif_diversity", 0.0),
                "composite_novelty": info.get("composite_novelty", 0.0),
            }

    # ── Phase 12: Report ──────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\nPhase 12: Report (total elapsed: {elapsed:.1f}s)...")
    report = generate_report(
        analyses, ep_results, ham_results, sym_results,
        op_ent_results, cond_results, vqe_results, best_input_results,
        zx_metrics, novelty, categories, elapsed,
        deep_vqe_results=deep_vqe_results,
    )
    print(report)
    (OUTPUT_DIR / "comparison_report.txt").write_text(report)

    # ── Phase 13: Visualization ───────────────────────────────────────
    print("\nPhase 13: Visualization...")
    plot_overhead(analyses, categories)
    print("  overhead_analysis.png")
    plot_state_structure(analyses, categories)
    print("  state_structure.png")
    plot_entangling_power(ep_results, categories, analyses)
    print("  entangling_power.png")
    plot_vqe_comparison(vqe_results, categories)
    print("  vqe_comparison.png")
    plot_zx_compression(zx_metrics, categories)
    print("  zx_compression.png")
    plot_deep_vqe(deep_vqe_results)
    if deep_vqe_results:
        print("  deep_vqe_benchmark.png")

    # ── JSON results ──────────────────────────────────────────────────
    json_out = {
        "n_circuits": len(analyses),
        "elapsed": elapsed,
        "analyses": {},
        "entangling_power": {},
        "best_input": best_input_results,
        "hamiltonian": {},
        "symmetries": {},
        "operator_entanglement": op_ent_results,
        "conditional": {},
        "vqe": vqe_results,
        "deep_vqe": deep_vqe_results,
        "zx": zx_metrics,
    }
    for name, a in analyses.items():
        json_out["analyses"][name] = {
            k: v for k, v in a.items()
            if k != "subsystems" or isinstance(v, (int, float, str, bool, list, dict, type(None)))
        }
        # Serialize subsystems separately
        json_out["analyses"][name]["subsystems"] = a.get("subsystems", [])

    for name, ep in ep_results.items():
        json_out["entangling_power"][name] = ep
    for name, hr in ham_results.items():
        if "error" not in hr:
            json_out["hamiltonian"][name] = {
                "best_match": hr.get("best_match"),
                "best_match_score": hr.get("best_match_score"),
                "n_terms": hr.get("n_terms"),
                "top_terms": hr.get("top_terms", [])[:5],
            }
    for name, s in sym_results.items():
        if s:
            json_out["symmetries"][name] = s
    for name, cr in cond_results.items():
        if cr:
            json_out["conditional"][name] = {
                "spare_qubits": cr["spare_qubits"],
                "data_qubits": cr["data_qubits"],
                "all_high": cr["all_high_entanglement"],
                "avg_concurrence": cr["avg_concurrence"],
            }

    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(json_out, indent=2, default=str)
    )

    print(f"\nDone in {elapsed:.1f}s. Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
