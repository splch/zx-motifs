#!/usr/bin/env python3
"""Benchmark the motif-composed ansatz against baselines.

Compares motif_composed, irr_pair11, cx_chain, and hea entanglers
across multiple Hamiltonians and qubit sizes using VQE.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from zx_motifs.pipeline.ansatz import (
    build_hamiltonian,
    cx_chain_entangler,
    hea_entangler,
    irr_pair11_entangler,
    motif_composed_entangler,
)
from zx_motifs.pipeline.evaluation import (
    compute_entangling_power,
    count_2q,
    run_benchmark,
)


def main():
    print("=" * 70)
    print("MOTIF-COMPOSED ANSATZ BENCHMARK")
    print("=" * 70)

    # Gate counts for reference
    print("\n--- 2-Qubit Gate Counts ---")
    for n in [4, 6, 8]:
        mc = motif_composed_entangler(n)
        ip = irr_pair11_entangler(n)
        n_2q_mc = count_2q(mc)
        n_2q_ip = count_2q(ip)
        print(f"  n={n}: motif_composed={n_2q_mc}, irr_pair11={n_2q_ip}")

    # Build configs
    configs = []
    for n_qubits in [4, 6]:
        n_2q_ref = count_2q(motif_composed_entangler(n_qubits))
        for model in ["heisenberg", "tfim", "xxz", "random_2local"]:
            H = build_hamiltonian(n_qubits, model)
            configs.extend([
                {
                    "name": "motif_composed",
                    "entangler_fn": motif_composed_entangler,
                    "n_qubits": n_qubits,
                    "model": model,
                    "hamiltonian": H,
                },
                {
                    "name": "irr_pair11",
                    "entangler_fn": irr_pair11_entangler,
                    "n_qubits": n_qubits,
                    "model": model,
                    "hamiltonian": H,
                },
                {
                    "name": "cx_chain",
                    "entangler_fn": lambda n, _n2q=n_2q_ref: cx_chain_entangler(n, _n2q),
                    "n_qubits": n_qubits,
                    "model": model,
                    "hamiltonian": H,
                },
                {
                    "name": "hea",
                    "entangler_fn": lambda n, _n2q=n_2q_ref: hea_entangler(n, _n2q),
                    "n_qubits": n_qubits,
                    "model": model,
                    "hamiltonian": H,
                },
            ])

    print(f"\nRunning {len(configs)} benchmark configurations...")
    print("(n_seeds=5, n_restarts=8, maxiter=300)")

    df = run_benchmark(configs, n_seeds=5, n_restarts=8, maxiter=300)

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS: Mean Relative Error by Ansatz, Qubits, Model")
    print("=" * 70)

    summary = df.groupby(["name", "n_qubits", "model"])["relative_error"].mean()
    summary = summary.unstack("name")

    # Print header
    names = sorted(df["name"].unique())
    header = f"{'qubits':>6} {'model':>14} | " + " | ".join(f"{n:>14}" for n in names)
    print(header)
    print("-" * len(header))

    for (n_q, model), row in summary.iterrows():
        vals = " | ".join(f"{row.get(n, float('nan')):>14.4f}" for n in names)
        print(f"{n_q:>6} {model:>14} | {vals}")

    # Win counts
    print("\n--- Win/Loss Summary ---")
    wins = {n: 0 for n in names}
    for (n_q, model), row in summary.iterrows():
        best = row.idxmin()
        if not isinstance(best, float):
            wins[best] = wins.get(best, 0) + 1
    for name, count in sorted(wins.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count} wins out of {len(summary)}")

    # Entangling power
    print("\n--- Entangling Power ---")
    for n in [4, 6]:
        print(f"\n  n={n} qubits:")
        for label, fn in [
            ("motif_composed", motif_composed_entangler),
            ("irr_pair11", irr_pair11_entangler),
        ]:
            qc = fn(n)
            ep = compute_entangling_power(qc, n_samples=50)
            print(f"    {label}: power={ep['entangling_power']:.4f} "
                  f"(std={ep['epd']:.4f}, max={ep['max_entropy']:.4f})")

    # Gate budget comparison
    n_2q_mc = count_2q(motif_composed_entangler(6))
    n_2q_ip = count_2q(irr_pair11_entangler(6))
    print(f"\n    Note: At 6 qubits, motif_composed uses {n_2q_mc} 2q gates "
          f"vs irr_pair11's {n_2q_ip}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
