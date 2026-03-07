#!/usr/bin/env python3
"""Theory-informed ansatz search via motif-derived circuit fragments.

Uses ZX-calculus structural principles to guide systematic composition:

1. **Phase gadget theory** -- Z-spiders with Hadamard-connected targets
   implement multi-qubit phase rotations; T-phases give non-Clifford
   expressibility (PQAS-ZX, Staudacher et al. 2025).

2. **Spider connectivity** -- Star (hub) topologies create GHZ-like
   entanglement; chain topologies create nearest-neighbor correlations;
   mixed topologies balance both.

3. **Color alternation** -- Alternating Z/X spiders with SIMPLE edges
   maximizes entanglement generation per gate (ZX-calculus duality).

4. **T-count as complexity** -- Non-Clifford gate density controls
   expressibility; too few = limited, too many = hard to optimize.

The search:
  - Defines circuit fragments corresponding to discovered motifs
  - Enumerates fragment combinations (order + selection)
  - Screens via entangling power (fast: ~0.1s per candidate)
  - Validates top candidates via VQE on small Hamiltonians (~30s each)
  - Selects the best composition as the final ansatz
"""
from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.pipeline.ansatz import (
    build_hamiltonian,
    cx_chain_entangler,
    hea_entangler,
    irr_pair11_entangler,
)
from zx_motifs.pipeline.evaluation import (
    compute_entangling_power,
    count_2q,
    vqe_test,
)


# ── Motif-derived circuit fragments ──────────────────────────────────
#
# Each fragment is a function(qc, qubits) that appends gates to qc
# acting on the given qubit indices. These correspond directly to
# ZX-calculus motifs discovered in the library:
#
#   ZX motif structure          →  Circuit translation
#   Z(arb) hub, 3×X(zero)      →  Phase gadget: T + CX fan-out
#   Z(zero)-2X(arb) fork       →  Dual rotation: CX-T-T-CX sandwich
#   X(zero)-2Z(cliff) fan      →  Clifford bridge: S-CX-Sdg
#   Z(t_like)-Z(zero) chain    →  T-chain: CX + T on target
#   Z-Z-Z Hadamard triangle    →  IQP clique: CZ triangle + T phases
#   Z(zero)-Z(arb)-Z(zero)     →  ZZ interaction: CX-Rz-CX


def frag_phase_gadget(qc: QuantumCircuit, qubits: list[int]):
    """Phase gadget: T-phase hub fans out to targets via CX.

    ZX origin: phase_hub_3x motif -- Z(arbitrary) connected to 3 X(zero)
    spiders. In circuit form: the hub qubit gets a T gate, then CX to
    each target creates correlated phase kickback.
    """
    hub = qubits[0]
    qc.t(hub)
    for target in qubits[1:]:
        qc.cx(hub, target)


def frag_dual_rotation(qc: QuantumCircuit, qubits: list[int]):
    """Dual rotation fork: CX pair with T on both sides.

    ZX origin: dual_rotation_fork -- Z(zero) hub with 2×X(arbitrary).
    The sandwich CX-T-T-CX creates entangled non-Clifford rotations
    that cannot be simplified by spider fusion.
    """
    for i in range(0, len(qubits) - 1, 2):
        a, b = qubits[i], qubits[i + 1]
        qc.cx(a, b)
        qc.t(a)
        qc.t(b)
        qc.cx(a, b)


def frag_clifford_bridge(qc: QuantumCircuit, qubits: list[int]):
    """Clifford bridge: S-conjugated CX connections.

    ZX origin: clifford_fan_out -- X(zero) with 2×Z(clifford).
    The S-CX-Sdg pattern creates CZ-like entanglement with a phase
    offset, adding Clifford-group structure.
    """
    for i in range(0, len(qubits) - 1, 2):
        a, b = qubits[i], qubits[i + 1]
        qc.s(a)
        qc.cx(a, b)
        qc.sdg(a)


def frag_t_chain(qc: QuantumCircuit, qubits: list[int]):
    """T-chain: nearest-neighbor CX with T gates on targets.

    ZX origin: t_gate_fan_out -- X(zero) hub with 2×Z(t_like).
    Linear chain of CX gates with T-phase injection creates a cascade
    of non-Clifford correlations along the chain.
    """
    for i in range(len(qubits) - 1):
        qc.cx(qubits[i], qubits[i + 1])
        qc.t(qubits[i + 1])


def frag_iqp_clique(qc: QuantumCircuit, qubits: list[int]):
    """IQP triangle: all-to-all CZ with T phases.

    ZX origin: iqp_triangle_2t -- Z-Z-Z Hadamard triangle with t_like
    phases. CZ gates implement Hadamard edges between Z-spiders;
    T gates add non-Clifford phases. Creates dense 3-body correlations.
    """
    # Apply CZ between all pairs
    for i in range(len(qubits)):
        for j in range(i + 1, len(qubits)):
            qc.cz(qubits[i], qubits[j])
    # T on first two (matching motif's 2 t_like phases)
    for q in qubits[:2]:
        qc.t(q)


def frag_zz_interaction(qc: QuantumCircuit, qubits: list[int]):
    """ZZ interaction: CX-Rz-CX for ZZ rotation.

    ZX origin: zz_interaction_param -- Z(zero)-Z(any_nonzero)-Z(zero).
    This is the standard ZZ(theta) gate used in QAOA and Hamiltonian
    simulation. The Rz angle controls interaction strength.
    """
    for i in range(len(qubits) - 1):
        qc.cx(qubits[i], qubits[i + 1])
        qc.rz(np.pi / 4, qubits[i + 1])  # T-equivalent rotation
        qc.cx(qubits[i], qubits[i + 1])


def frag_star_fanin(qc: QuantumCircuit, qubits: list[int]):
    """Star fan-in: multiple controls into one target.

    ZX origin: z_hub_3x / x_hub_2z -- hub spider with multiple
    neighbors. Fan-in CX creates GHZ-like entanglement centered on
    the hub qubit. This is the key structure in irr_pair11.
    """
    hub = qubits[0]
    for ctrl in qubits[1:]:
        qc.cx(ctrl, hub)


def frag_mixed_edge(qc: QuantumCircuit, qubits: list[int]):
    """Mixed-edge triple: CX + CZ combining SIMPLE and HADAMARD edges.

    ZX origin: mixed_edge_triple -- Z-X via SIMPLE + Z-Z via HADAMARD.
    Mixing CX (SIMPLE) and CZ (HADAMARD) edge types in the same
    fragment creates richer entanglement structure than either alone.
    """
    if len(qubits) >= 3:
        qc.cx(qubits[0], qubits[1])
        qc.cz(qubits[1], qubits[2])
        qc.t(qubits[0])
    elif len(qubits) == 2:
        qc.cx(qubits[0], qubits[1])
        qc.t(qubits[0])


# ── Fragment catalog ─────────────────────────────────────────────────

FRAGMENTS = {
    "phase_gadget": frag_phase_gadget,
    "dual_rotation": frag_dual_rotation,
    "clifford_bridge": frag_clifford_bridge,
    "t_chain": frag_t_chain,
    "iqp_clique": frag_iqp_clique,
    "zz_interaction": frag_zz_interaction,
    "star_fanin": frag_star_fanin,
    "mixed_edge": frag_mixed_edge,
}


# ── Qubit allocation strategies ─────────────────────────────────────
#
# How to distribute n qubits across fragments. This is the "parallel
# vs sequential" composition from ZX box theory:
#   - Sequential: each fragment uses all qubits (deep circuit)
#   - Partitioned: split qubits into groups (wide circuit)
#   - Overlapping: sliding window (balanced)


def allocate_all(n: int) -> list[int]:
    """All qubits for a single fragment."""
    return list(range(n))


def allocate_halves(n: int) -> tuple[list[int], list[int]]:
    """Split into two halves."""
    mid = n // 2
    return list(range(mid)), list(range(mid, n))


def allocate_sliding(n: int, window: int = 3) -> list[list[int]]:
    """Sliding windows of given size."""
    groups = []
    for start in range(0, n - window + 1, max(1, window - 1)):
        groups.append(list(range(start, min(start + window, n))))
    return groups


# ── Candidate builder ───────────────────────────────────────────────


def build_candidate(
    n: int,
    fragment_sequence: list[str],
    strategy: str = "sequential",
) -> QuantumCircuit:
    """Build an entangler circuit from a sequence of fragment names.

    strategy:
      - "sequential": apply each fragment to all qubits in order
      - "partitioned": split qubits, apply fragments to halves
      - "sliding": apply fragments via sliding window
    """
    qc = QuantumCircuit(n)

    if strategy == "sequential":
        for fname in fragment_sequence:
            fn = FRAGMENTS[fname]
            fn(qc, list(range(n)))

    elif strategy == "partitioned":
        left, right = allocate_halves(n)
        for i, fname in enumerate(fragment_sequence):
            fn = FRAGMENTS[fname]
            # Alternate between halves and full
            if i % 3 == 2:
                fn(qc, list(range(n)))  # bridge layer
            elif i % 2 == 0:
                fn(qc, left)
            else:
                fn(qc, right)

    elif strategy == "sliding":
        windows = allocate_sliding(n, window=3)
        for i, fname in enumerate(fragment_sequence):
            fn = FRAGMENTS[fname]
            win = windows[i % len(windows)]
            fn(qc, win)

    return qc


# ── Search ──────────────────────────────────────────────────────────


def search_best_ansatz(n_qubits: int = 6, top_k: int = 5):
    """Search over fragment combinations, rank by entangling power.

    The search space is:
      - 2-3 fragments from the catalog (order matters)
      - 3 composition strategies
      - Scored by entangling power (average bipartite entropy)

    Theory justification:
      - Entangling power correlates with expressibility (Sim et al. 2019)
      - Higher entangling power → circuit can reach more of Hilbert space
      - But we also penalize excessive gate count (parsimony)
    """
    n = n_qubits
    frag_names = list(FRAGMENTS.keys())

    # Reference gate counts
    ref_2q = count_2q(irr_pair11_entangler(n))
    print(f"Reference 2q gate count (irr_pair11, n={n}): {ref_2q}")

    candidates = []

    # Generate 2-fragment and 3-fragment combinations
    for r in [2, 3]:
        for combo in itertools.permutations(frag_names, r):
            for strategy in ["sequential", "partitioned", "sliding"]:
                try:
                    qc = build_candidate(n, list(combo), strategy)
                    n_2q = count_2q(qc)

                    # Gate budget filter: skip if >2x or <0.3x reference
                    if n_2q > 2 * ref_2q or n_2q < max(1, ref_2q * 0.3):
                        continue

                    candidates.append({
                        "fragments": combo,
                        "strategy": strategy,
                        "circuit": qc,
                        "n_2q": n_2q,
                    })
                except Exception:
                    continue

    print(f"Generated {len(candidates)} valid candidates (gate budget filtered)")

    # Phase 1: Screen by entangling power (fast, ~0.05s each)
    print("\nPhase 1: Screening by entangling power...")
    for i, cand in enumerate(candidates):
        ep = compute_entangling_power(cand["circuit"], n_samples=30)
        cand["ent_power"] = ep["entangling_power"]
        cand["ent_std"] = ep["epd"]
        if (i + 1) % 50 == 0:
            print(f"  Screened {i + 1}/{len(candidates)}")

    # Rank by entangling power, weighted by gate efficiency
    for cand in candidates:
        # Score: entangling power / sqrt(gate count)
        # This rewards high entanglement with fewer gates
        cand["efficiency"] = cand["ent_power"] / max(1.0, np.sqrt(cand["n_2q"]))

    candidates.sort(key=lambda c: -c["efficiency"])

    # Print top candidates from screening
    print(f"\nTop {min(top_k * 2, len(candidates))} by efficiency (ent_power / sqrt(gates)):")
    for i, cand in enumerate(candidates[:top_k * 2]):
        print(f"  {i + 1}. {cand['fragments']} [{cand['strategy']}] "
              f"ent={cand['ent_power']:.4f} gates={cand['n_2q']} "
              f"eff={cand['efficiency']:.4f}")

    # Phase 2: Validate top candidates with VQE (slower, ~30s each)
    print(f"\nPhase 2: Validating top {top_k} with VQE (Heisenberg, n={n})...")
    H = build_hamiltonian(n, "heisenberg")
    exact_gs = float(np.linalg.eigvalsh(H)[0])
    print(f"  Exact ground state: {exact_gs:.6f}")

    finalists = candidates[:top_k]
    for cand in finalists:
        result = vqe_test(cand["circuit"], n, H, n_restarts=5, maxiter=200, seed=42)
        rel_err = abs(result["best_energy"] - exact_gs) / abs(exact_gs)
        cand["vqe_rel_err"] = rel_err
        cand["vqe_energy"] = result["best_energy"]
        print(f"  {cand['fragments']} [{cand['strategy']}]: "
              f"E={result['best_energy']:.6f} rel_err={rel_err:.6f}")

    # Also test baselines
    print("\n  Baselines:")
    for name, ent_fn in [
        ("irr_pair11", lambda: irr_pair11_entangler(n)),
        ("cx_chain", lambda: cx_chain_entangler(n, ref_2q)),
        ("hea", lambda: hea_entangler(n, ref_2q)),
    ]:
        qc = ent_fn()
        result = vqe_test(qc, n, H, n_restarts=5, maxiter=200, seed=42)
        rel_err = abs(result["best_energy"] - exact_gs) / abs(exact_gs)
        ep = compute_entangling_power(qc, n_samples=30)
        n2q = count_2q(qc)
        print(f"  {name}: E={result['best_energy']:.6f} rel_err={rel_err:.6f} "
              f"ent={ep['entangling_power']:.4f} gates={n2q}")

    # Select winner
    finalists.sort(key=lambda c: c["vqe_rel_err"])
    winner = finalists[0]

    print(f"\n{'=' * 70}")
    print(f"WINNER: {winner['fragments']} [{winner['strategy']}]")
    print(f"  VQE rel_err: {winner['vqe_rel_err']:.6f}")
    print(f"  Ent power:   {winner['ent_power']:.4f}")
    print(f"  2q gates:    {winner['n_2q']}")
    print(f"  Efficiency:  {winner['efficiency']:.4f}")
    print(f"{'=' * 70}")

    return winner, finalists


def validate_winner(winner: dict, n_qubits: int = 6):
    """Run extended validation across multiple Hamiltonians."""
    n = n_qubits
    ref_2q = count_2q(irr_pair11_entangler(n))

    print(f"\nExtended validation (n={n}):")
    print(f"{'model':>14} | {'winner':>12} | {'irr_pair11':>12} | {'cx_chain':>12} | {'hea':>12}")
    print("-" * 72)

    for model in ["heisenberg", "tfim", "xxz", "random_2local"]:
        H = build_hamiltonian(n, model)
        exact_gs = float(np.linalg.eigvalsh(H)[0])
        if exact_gs == 0:
            continue

        results = {}
        for name, qc in [
            ("winner", winner["circuit"]),
            ("irr_pair11", irr_pair11_entangler(n)),
            ("cx_chain", cx_chain_entangler(n, ref_2q)),
            ("hea", hea_entangler(n, ref_2q)),
        ]:
            r = vqe_test(qc, n, H, n_restarts=5, maxiter=200, seed=42)
            results[name] = abs(r["best_energy"] - exact_gs) / abs(exact_gs)

        print(f"{model:>14} | {results['winner']:>12.6f} | "
              f"{results['irr_pair11']:>12.6f} | "
              f"{results['cx_chain']:>12.6f} | "
              f"{results['hea']:>12.6f}")

    return results


def print_circuit_code(winner: dict, n_qubits: int = 6):
    """Print the Python code for the winning entangler function."""
    frags = winner["fragments"]
    strategy = winner["strategy"]
    n = n_qubits

    print(f"\n{'=' * 70}")
    print("GENERATED ENTANGLER CODE")
    print(f"Fragments: {frags}")
    print(f"Strategy: {strategy}")
    print(f"{'=' * 70}")

    # Show the actual circuit
    qc = winner["circuit"]
    print(f"\nCircuit ({qc.num_qubits} qubits, {count_2q(qc)} 2q gates):")
    print(qc.draw(output="text"))


def main():
    print("=" * 70)
    print("THEORY-INFORMED ANSATZ SEARCH")
    print("=" * 70)
    print()
    print("Using ZX-calculus motif fragments as building blocks.")
    print("Screening by entangling power, validating with VQE.")
    print()

    winner, finalists = search_best_ansatz(n_qubits=6, top_k=5)
    validate_winner(winner, n_qubits=6)
    print_circuit_code(winner, n_qubits=6)

    # Also test at 4 qubits
    print(f"\n{'=' * 70}")
    print("CROSS-VALIDATION AT 4 QUBITS")
    print(f"{'=' * 70}")
    winner_4q, _ = search_best_ansatz(n_qubits=4, top_k=3)
    validate_winner(winner_4q, n_qubits=4)


if __name__ == "__main__":
    main()
