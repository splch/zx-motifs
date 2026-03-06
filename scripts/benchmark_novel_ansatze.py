#!/usr/bin/env python3
"""Benchmark novel motif-composed ansätze against irr_pair11 and baselines.

Evaluates 5 new candidates + irr_pair11 + 2 baselines across:
- Hamiltonians: heisenberg, tfim, xxz, xy
- Qubit counts: 4, 5, 6
- Multiple random seeds for statistical significance

Outputs: CSV results + summary table + statistical tests.
"""

from __future__ import annotations

import sys
import time
from functools import partial

import numpy as np
import pandas as pd
from scipy import stats

from zx_motifs.pipeline.ansatz import (
    build_hamiltonian,
    cx_chain_entangler,
    hea_entangler,
    irr_double_star,
    irr_gadget_ladder,
    irr_iqp_dense,
    irr_pair11_entangler,
    irr_sandwich_star,
    irr_triangle_t,
)
from zx_motifs.pipeline.evaluation import (
    compute_entangling_power,
    count_2q,
    vqe_test,
)


def main():
    # ── Configuration ─────────────────────────────────────────────────
    qubit_counts = [4, 5, 6]
    models = ["heisenberg", "tfim", "xxz", "xy"]
    n_seeds = 5
    n_restarts = 10
    maxiter = 400

    # All ansätze to test (each takes n_qubits as first arg)
    ansatze = {
        "irr_pair11": irr_pair11_entangler,
        "irr_triangle_t": irr_triangle_t,
        "irr_double_star": irr_double_star,
        "irr_gadget_ladder": irr_gadget_ladder,
        "irr_iqp_dense": irr_iqp_dense,
        "irr_sandwich_star": irr_sandwich_star,
    }

    total_configs = len(qubit_counts) * len(models) * len(ansatze) * n_seeds
    print(f"Running {total_configs} VQE configurations...")
    print(f"  Ansätze: {list(ansatze.keys())} + 2 baselines")
    print(f"  Qubits: {qubit_counts}, Models: {models}")
    print(f"  Seeds: {n_seeds}, Restarts: {n_restarts}, Maxiter: {maxiter}")
    print()

    rows = []
    done = 0

    for n in qubit_counts:
        for model in models:
            H = build_hamiltonian(n, model)
            exact_gs = float(np.linalg.eigvalsh(H)[0])
            if exact_gs == 0:
                print(f"  SKIP {n}q {model} (zero ground state)")
                continue

            # Build baseline entanglers matched to irr_pair11 2q-gate count
            irr_qc = irr_pair11_entangler(n)
            irr_2q = count_2q(irr_qc)

            baselines = {
                "cx_chain": partial(cx_chain_entangler, n_2q=irr_2q),
                "hea": partial(hea_entangler, n_2q=irr_2q),
            }

            all_entanglers = {}
            for name, fn in ansatze.items():
                all_entanglers[name] = fn(n)
            for name, fn in baselines.items():
                all_entanglers[name] = fn(n)

            for name, entangler_qc in all_entanglers.items():
                n2q = count_2q(entangler_qc)
                for seed in range(42, 42 + n_seeds):
                    t0 = time.time()
                    result = vqe_test(entangler_qc, n, H, n_restarts, maxiter, seed=seed)
                    elapsed = time.time() - t0
                    rel_err = abs(result["best_energy"] - exact_gs) / abs(exact_gs)
                    rows.append({
                        "name": name,
                        "n_qubits": n,
                        "model": model,
                        "seed": seed,
                        "best_energy": result["best_energy"],
                        "exact_gs": exact_gs,
                        "relative_error": rel_err,
                        "n_params": result["n_params"],
                        "n_2q_gates": n2q,
                        "total_gates": len(entangler_qc.data),
                        "elapsed_s": elapsed,
                    })
                    done += 1
                    if done % 10 == 0:
                        print(f"  [{done}/{total_configs}] {name} {n}q {model} "
                              f"seed={seed}: rel_err={rel_err:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv("scripts/benchmark_results.csv", index=False)
    print(f"\nResults saved to scripts/benchmark_results.csv")

    # ── Summary statistics ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY: Mean Relative Error by Ansatz / Config")
    print("=" * 80)

    pivot = df.groupby(["name", "n_qubits", "model"])["relative_error"].agg(
        ["mean", "std", "min"]
    ).round(4)
    print(pivot.to_string())

    # ── Head-to-head vs irr_pair11 ────────────────────────────────────
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD vs irr_pair11 (lower relative error = better)")
    print("=" * 80)

    irr_results = df[df["name"] == "irr_pair11"]
    novel_names = [n for n in ansatze.keys() if n != "irr_pair11"]
    novel_names += ["cx_chain", "hea"]

    comparison_rows = []
    for name in novel_names:
        other = df[df["name"] == name]
        for n in qubit_counts:
            for model in models:
                irr_sub = irr_results[
                    (irr_results["n_qubits"] == n) & (irr_results["model"] == model)
                ]["relative_error"]
                other_sub = other[
                    (other["n_qubits"] == n) & (other["model"] == model)
                ]["relative_error"]

                if len(irr_sub) == 0 or len(other_sub) == 0:
                    continue

                irr_mean = irr_sub.mean()
                other_mean = other_sub.mean()
                improvement = (irr_mean - other_mean) / irr_mean * 100

                # Welch's t-test
                if len(irr_sub) > 1 and len(other_sub) > 1:
                    t_stat, p_val = stats.ttest_ind(other_sub, irr_sub, equal_var=False)
                else:
                    t_stat, p_val = 0.0, 1.0

                winner = name if other_mean < irr_mean else "irr_pair11"
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

                comparison_rows.append({
                    "config": f"{n}q_{model}",
                    "challenger": name,
                    "irr_pair11_err": round(irr_mean, 4),
                    "challenger_err": round(other_mean, 4),
                    "improvement_%": round(improvement, 1),
                    "p_value": round(p_val, 4),
                    "sig": sig,
                    "winner": winner,
                })

    comp_df = pd.DataFrame(comparison_rows)
    print(comp_df.to_string(index=False))

    # ── Win/Loss summary ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("WIN/LOSS RECORD vs irr_pair11")
    print("=" * 80)

    for name in novel_names:
        sub = comp_df[comp_df["challenger"] == name]
        wins = (sub["winner"] == name).sum()
        losses = (sub["winner"] == "irr_pair11").sum()
        sig_wins = ((sub["winner"] == name) & (sub["sig"] != "")).sum()
        avg_imp = sub["improvement_%"].mean()
        n2q = df[df["name"] == name]["n_2q_gates"].iloc[0] if len(df[df["name"] == name]) > 0 else "?"
        print(f"  {name:20s}: {wins}W-{losses}L "
              f"({sig_wins} sig wins), avg improvement: {avg_imp:+.1f}%, "
              f"2q gates (4q): {n2q}")

    # ── Entangling power ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ENTANGLING POWER (6 qubits, 100 samples)")
    print("=" * 80)

    for name, fn in ansatze.items():
        qc = fn(6)
        ep = compute_entangling_power(qc, n_samples=100)
        n2q = count_2q(qc)
        print(f"  {name:20s}: EP={ep['entangling_power']:.3f} ± {ep['epd']:.3f}  "
              f"({len(qc.data)} gates, {n2q} 2q)")

    # ── Gate efficiency ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("GATE COUNTS BY QUBIT SIZE")
    print("=" * 80)

    for name, fn in ansatze.items():
        counts = []
        for n in [4, 5, 6, 8]:
            qc = fn(n)
            counts.append(f"{n}q: {len(qc.data)}g/{count_2q(qc)}cx")
        print(f"  {name:20s}: {', '.join(counts)}")

    # ── Best overall candidates ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("OVERALL RANKING (by mean relative error across ALL configs)")
    print("=" * 80)

    overall = df.groupby("name")["relative_error"].mean().sort_values()
    for rank, (name, err) in enumerate(overall.items(), 1):
        marker = " <-- CHAMPION" if rank == 1 else ""
        print(f"  {rank}. {name:20s}: {err:.4f}{marker}")


if __name__ == "__main__":
    main()
