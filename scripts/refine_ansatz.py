#!/usr/bin/env python3
"""Refine the ansatz search with strict gate-budget matching.

The initial search found star_fanin + phase_gadget as the best core
pattern. This script refines it to:
1. Match the exact 2q gate budget of irr_pair11
2. Test more targeted variations of the winning pattern
3. Run full multi-seed VQE validation
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from qiskit import QuantumCircuit

from zx_motifs.pipeline.ansatz import build_hamiltonian, irr_pair11_entangler
from zx_motifs.pipeline.evaluation import (
    compute_entangling_power,
    count_2q,
    vqe_test,
)


def motif_entangler_v1(n: int) -> QuantumCircuit:
    """Star fan-in then phase gadget fan-out (search winner).

    ZX theory: combines the z_hub_3x motif (fan-in creates GHZ-like
    correlations on hub) with phase_hub_3x motif (T-phase + fan-out
    distributes non-Clifford phase). The fan-in/fan-out sandwich
    creates bidirectional entanglement that cannot be simplified
    by ZX spider fusion rules.
    """
    qc = QuantumCircuit(n)
    hub = 0
    reach = min(n - 1, 3)

    # Fan-in: CX targets into hub (z_hub motif)
    for i in range(1, reach + 1):
        qc.cx(i, hub)

    # Phase gadget fan-out: T on hub, CX out (phase_hub motif)
    qc.t(hub)
    for i in range(1, reach + 1):
        qc.cx(hub, i)

    return qc


def motif_entangler_v2(n: int) -> QuantumCircuit:
    """Star fan-in + T-chain tail (matched to irr_pair11 gate count).

    Like v1 but extends to full qubit range using t_chain fragment
    for remaining qubits not reached by the hub.
    """
    qc = QuantumCircuit(n)
    hub = 0
    reach = min(n - 1, 3)

    # Fan-in
    for i in range(1, reach + 1):
        qc.cx(i, hub)

    # Phase gadget fan-out with T
    qc.t(hub)
    for i in range(1, min(reach + 1, n)):
        qc.cx(hub, i)

    # T-chain for remaining qubits (if n > reach+1)
    for i in range(reach, n - 1):
        if count_2q(qc) >= count_2q(irr_pair11_entangler(n)):
            break
        qc.cx(i, i + 1)
        qc.t(i + 1)

    return qc


def motif_entangler_v3(n: int) -> QuantumCircuit:
    """Bidirectional hub: fan-in on q0, fan-out from q0, phase on all.

    Strictly gate-matched to irr_pair11. Uses the theoretical insight
    that bidirectional CX (control→target then target→control) creates
    maximal 2-qubit entanglement per gate pair.
    """
    qc = QuantumCircuit(n)
    target_2q = count_2q(irr_pair11_entangler(n))

    hub = 0
    placed = 0

    # Phase 1: Fan-in to hub (half the budget)
    fan_budget = target_2q // 2
    for i in range(1, n):
        if placed >= fan_budget:
            break
        qc.cx(i, hub)
        placed += 1

    # T on hub
    qc.t(hub)

    # Phase 2: Fan-out from hub (remaining budget)
    for i in range(1, n):
        if placed >= target_2q:
            break
        qc.cx(hub, i)
        placed += 1

    return qc


def motif_entangler_v4(n: int) -> QuantumCircuit:
    """Two-hub design: fan-in on q0, fan-out from q_{n//2}.

    Distributes entanglement across two hub qubits to avoid
    bottleneck on a single qubit. T phases on both hubs.
    """
    qc = QuantumCircuit(n)
    target_2q = count_2q(irr_pair11_entangler(n))

    hub1 = 0
    hub2 = n // 2
    placed = 0

    # Fan-in to hub1 from lower half
    for i in range(1, hub2):
        if placed >= target_2q // 2:
            break
        qc.cx(i, hub1)
        placed += 1

    qc.t(hub1)

    # Fan-in to hub2 from upper half, then bridge
    for i in range(hub2 + 1, n):
        if placed >= (3 * target_2q) // 4:
            break
        qc.cx(i, hub2)
        placed += 1

    qc.t(hub2)

    # Bridge between hubs
    if placed < target_2q:
        qc.cx(hub1, hub2)
        placed += 1

    # Fill remaining with fan-out
    for i in range(1, n):
        if placed >= target_2q:
            break
        if i != hub2:
            qc.cx(hub2, i)
            placed += 1

    return qc


def motif_entangler_v5(n: int) -> QuantumCircuit:
    """ZZ-interaction chain + phase gadget.

    Combines the zz_interaction motif (CX-Rz-CX pairs) with a
    star phase gadget. This mirrors QAOA/Trotter circuit structure.
    """
    qc = QuantumCircuit(n)
    target_2q = count_2q(irr_pair11_entangler(n))
    placed = 0

    # ZZ interactions on adjacent pairs
    for i in range(0, n - 1, 2):
        if placed + 2 > target_2q:
            break
        qc.cx(i, i + 1)
        qc.rz(np.pi / 4, i + 1)
        qc.cx(i, i + 1)
        placed += 2

    # Phase gadget on remaining budget
    if placed < target_2q:
        qc.t(0)
        for i in range(1, n):
            if placed >= target_2q:
                break
            qc.cx(0, i)
            placed += 1

    return qc


VARIANTS = {
    "v1_fanin_gadget": motif_entangler_v1,
    "v2_fanin_tchain": motif_entangler_v2,
    "v3_bidirectional": motif_entangler_v3,
    "v4_two_hub": motif_entangler_v4,
    "v5_zz_gadget": motif_entangler_v5,
}


def main():
    for n in [4, 6]:
        ref_2q = count_2q(irr_pair11_entangler(n))
        print(f"\n{'=' * 70}")
        print(f"n={n} qubits (irr_pair11 budget: {ref_2q} 2q gates)")
        print(f"{'=' * 70}")

        # Gate counts
        print("\n--- Gate Counts ---")
        for name, fn in VARIANTS.items():
            qc = fn(n)
            print(f"  {name}: {count_2q(qc)} 2q gates")

        # Entangling power
        print("\n--- Entangling Power ---")
        for name, fn in VARIANTS.items():
            qc = fn(n)
            ep = compute_entangling_power(qc, n_samples=50)
            print(f"  {name}: {ep['entangling_power']:.4f} "
                  f"(std={ep['epd']:.4f})")

        ep = compute_entangling_power(irr_pair11_entangler(n), n_samples=50)
        print(f"  irr_pair11:  {ep['entangling_power']:.4f} "
              f"(std={ep['epd']:.4f})")

        # VQE across models
        print(f"\n--- VQE Relative Error (n_restarts=8, maxiter=300) ---")
        models = ["heisenberg", "tfim", "xxz", "random_2local"]

        header = f"{'name':>20} |" + "".join(f" {m:>14}" for m in models)
        print(header)
        print("-" * len(header))

        all_fns = dict(VARIANTS)
        all_fns["irr_pair11"] = irr_pair11_entangler

        for name, fn in all_fns.items():
            errs = []
            for model in models:
                H = build_hamiltonian(n, model)
                exact = float(np.linalg.eigvalsh(H)[0])
                if exact == 0:
                    errs.append(float("nan"))
                    continue
                qc = fn(n)
                r = vqe_test(qc, n, H, n_restarts=8, maxiter=300, seed=42)
                errs.append(abs(r["best_energy"] - exact) / abs(exact))
            vals = "".join(f" {e:>14.6f}" for e in errs)
            print(f"{name:>20} |{vals}")

    print(f"\n{'=' * 70}")
    print("REFINEMENT COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
