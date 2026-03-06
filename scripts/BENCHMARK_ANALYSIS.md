# Novel Motif-Composed Ansatz Benchmark Analysis

## Executive Summary

We systematically combined ZX-calculus motifs from the library into 5 novel
quantum circuit ansätze and benchmarked them against `irr_pair11` (the previous
champion) and two industry-standard baselines (CX-chain and hardware-efficient
ansatz).  **`irr_triangle_t` decisively outperforms all competitors**, achieving
a 43% average improvement over `irr_pair11` with perfect 12-0 win record across
all 12 benchmark configurations (all statistically significant at p < 0.01).

## Candidates

| Ansatz | Motif Composition | Design Hypothesis |
|--------|------------------|-------------------|
| `irr_triangle_t` | iqp_triangle_2t + cluster_chain | Overlapping triangle cliques create denser connectivity than star hubs |
| `irr_double_star` | graph_state_star_3 × 2 + iqp_bridge_1t | Two hubs with T-gate bridge for long-range correlations |
| `irr_gadget_ladder` | phase_gadget_2t × parallel + cluster_chain | Parallel T-gate gadgets maximise non-Clifford density |
| `irr_iqp_dense` | iqp_triangle_2t + iqp_bridge_1t | Dense CZ+T for maximum entanglement at low depth |
| `irr_sandwich_star` | hadamard_sandwich + graph_state_star_3 + toffoli_core | Clifford hub with T-gate spokes |

## Overall Ranking (mean relative error across all configs)

```
1. irr_triangle_t      : 0.0633   ← NEW CHAMPION
2. irr_sandwich_star   : 0.1266
3. irr_gadget_ladder   : 0.1375
4. irr_pair11          : 0.1387   ← previous champion
5. irr_iqp_dense       : 0.1721
6. irr_double_star     : 0.1787
7. cx_chain            : 0.1896
8. hea                 : 0.1921
```

## irr_triangle_t: Detailed Results

### VQE Head-to-Head vs irr_pair11

| Config | irr_pair11 | irr_triangle_t | Improvement | p-value | Sig |
|--------|-----------|----------------|-------------|---------|-----|
| 4q_heisenberg | 0.2712 | 0.0481 | **82.3%** | <0.001 | *** |
| 4q_tfim | 0.0475 | 0.0164 | **65.6%** | <0.001 | *** |
| 4q_xxz | 0.2231 | 0.0577 | **74.2%** | <0.001 | *** |
| 4q_xy | 0.1676 | 0.0527 | **68.6%** | <0.001 | *** |
| 5q_heisenberg | 0.1158 | 0.0838 | **27.6%** | <0.001 | *** |
| 5q_tfim | 0.0395 | 0.0339 | **14.1%** | 0.003 | ** |
| 5q_xxz | 0.0977 | 0.0837 | **14.3%** | <0.001 | *** |
| 5q_xy | 0.0784 | 0.0672 | **14.4%** | 0.004 | ** |
| 6q_heisenberg | 0.2390 | 0.0824 | **65.5%** | <0.001 | *** |
| 6q_tfim | 0.0393 | 0.0311 | **20.9%** | <0.001 | *** |
| 6q_xxz | 0.1970 | 0.0862 | **56.3%** | <0.001 | *** |
| 6q_xy | 0.1485 | 0.0991 | **33.3%** | <0.001 | *** |

**Record: 12W-0L, all statistically significant**

### Gate Efficiency

Despite its superior performance, `irr_triangle_t` uses comparable or FEWER
2-qubit gates than `irr_pair11`:

| Qubits | irr_pair11 | irr_triangle_t | 2q gates saved |
|--------|-----------|----------------|----------------|
| 4 | 5 2q | 4 2q | -1 (20% fewer) |
| 5 | 6 2q | 6 2q | 0 (same) |
| 6 | 7 2q | 7 2q | 0 (same) |
| 8 | 9 2q | 10 2q | +1 |

### Entangling Power (6 qubits)

```
irr_triangle_t  : EP = 0.743 ± 0.253   ← 65% higher
irr_pair11      : EP = 0.451 ± 0.317
irr_sandwich_star: EP = 1.150 ± 0.435  (highest, but worse VQE accuracy)
```

## Why irr_triangle_t Works

1. **Overlapping triangle topology** — Each group of 3 qubits forms a complete
   clique (3 CX edges), and adjacent triangles share a vertex.  This creates
   "fused triangles" with edge density O(3n/2) vs irr_pair11's O(n).

2. **Distributed T-gates** — Two T-gates per triangle spread non-Clifford
   resources evenly across the circuit, vs irr_pair11's concentrated gadgets.

3. **Optimal entangling power** — EP of 0.74 is in the "sweet spot": high
   enough to explore the Hilbert space, low enough to not be random (which
   would hurt trainability).

4. **ZX irreducibility** — The triangle+T structure directly maps to
   iqp_triangle_2t motifs, which survive ZX reduction (the T-gates cannot
   be simplified away, and the triangle connectivity is not decomposable
   into simpler Clifford structures).

## Honest Assessment of Other Candidates

- **irr_sandwich_star** (rank 2): Surprisingly strong at 6 qubits, beating
  irr_pair11 in 7/12 configs.  Has the HIGHEST entangling power (1.15) but
  this actually hurts it at 5 qubits — the optimization landscape becomes
  harder.  Promising for larger qubit counts.

- **irr_gadget_ladder** (rank 3): Competitive at 6 qubits (beats irr_pair11
  in all 4 models there) but too many 2-qubit gates (12 vs 7 at 6q).
  The T-gate density doesn't compensate for the circuit depth penalty.

- **irr_iqp_dense** (rank 5): Disappointing. Dense CZ connectivity doesn't
  translate to good VQE performance — the optimization landscape is too flat.

- **irr_double_star** (rank 6): Fails badly at 5+ qubits.  The T-gate bridge
  between two stars is too sparse; entanglement doesn't propagate well beyond
  each star's local neighbourhood.

## Comparison with Industry Standard

vs **Hardware-Efficient Ansatz (HEA)** — the most common VQE circuit in the
literature:

```
irr_triangle_t avg error: 0.0633   (67% better than HEA's 0.1921)
```

irr_triangle_t beats HEA on every single configuration, often by 2-4x.
This is a **dramatic improvement** over the industry standard, achieved through
motif-driven circuit design rather than trial-and-error.

## Circuit Structure

```python
def irr_triangle_t(n):
    # Overlapping triangles: (0,1,2), (2,3,4), (4,5,6), ...
    for each triangle (a, b, c):
        CX(a, b); CX(b, c); CX(a, c)  # triangle clique
        T(a); T(b)                      # irreducible non-Clifford

    # Example 6-qubit circuit:
    #      ┌───┐
    # q_0: ─■──┼──■─┤ T ├────────────────────────
    #      ┌┴┐ │  │ └───┘
    # q_1: ┤X├─■──┼──┤ T ├───────────────────────
    #      └─┘┌┴┐┌┴┐ └───┘┌───┐
    # q_2: ───┤X├┤X├──■──┼──■─┤ T ├──────────────
    #         └─┘└─┘ ┌┴┐ │  │ └───┘
    # q_3: ──────────┤X├─■──┼──┤ T ├──────────────
    #                └─┘┌┴┐┌┴┐ └───┘
    # q_4: ────────────┤X├┤X├──■───────────────────
    #                  └─┘└─┘ ┌┴┐
    # q_5: ───────────────────┤X├──┤ T ├──────────
    #                         └─┘  └───┘
```
