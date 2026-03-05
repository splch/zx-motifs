# ZX-Motifs: Data-Driven Quantum Algorithm Taxonomy and Circuit Design via ZX-Calculus Subgraph Analysis

---

**Abstract.**
We present ZX-Motifs, a computational pipeline that converts quantum circuits into ZX-calculus diagrams, discovers recurring structural subgraph patterns (motifs) across a corpus of 32 quantum algorithms (88 instances spanning 10 families), and uses motif fingerprints to construct a quantitative phylogeny of quantum algorithms. The pipeline identifies 367 motifs---8 universal, 29 common, and 323 family-specific---and reveals previously unknown cross-family structural relationships: BBPSSW entanglement distillation and GHZ state preparation share 99% motif similarity, while QAOA MaxCut and Trotter-Ising simulation share 93%. We identify four *ZX-irreducible motifs* that survive full ZX-calculus simplification, representing the computational atoms that no rewrite rule can eliminate. Guided by motif coverage gaps and structural phylogeny, we construct three novel quantum circuits (DEB, TVH, ICC) and evaluate them against industry-standard algorithms. While none outperforms existing methods at their hypothesised tasks, the analysis yields an actionable design principle: ZX-irreducible entangling layers improve Hilbert space coverage per parameter. We validate this principle with ZXEA, a variational ansatz that uses cluster-chain entangling and achieves 2.1% VQE error on the 4-qubit Heisenberg model versus 3.5% for the hardware-efficient ansatz at equal parameter count. The methodology demonstrates that ZX-calculus motif analysis provides a viable, quantitative framework for algorithm taxonomy and structure-guided circuit design.

---

## 1. Introduction

The growing zoo of quantum algorithms---from Grover's search to variational eigensolvers to error-correcting codes---lacks a principled structural taxonomy. Algorithms are typically classified by their *purpose* (optimisation, simulation, error correction) rather than their *structure*. Yet two algorithms solving entirely different problems may share deep structural features when viewed through an appropriate lens. Identifying such shared structure could accelerate algorithm design by enabling principled transfer of circuit primitives across problem domains.

The ZX-calculus [1,2] provides a natural framework for structural analysis of quantum circuits. Unlike the gate-level representation, ZX diagrams absorb commutation relations and circuit identities into their graph topology, so that structurally equivalent circuits yield isomorphic graphs. This makes subgraph isomorphism a meaningful proxy for structural similarity---a property unavailable at the gate level, where syntactically different circuits may implement identical unitaries.

In this work, we develop a complete pipeline for ZX-calculus-based structural analysis of quantum algorithms. Our contributions are:

1. **A motif discovery pipeline** that converts quantum circuits to ZX diagrams, extracts recurring subgraph patterns via VF2 isomorphism, and produces a 367-motif library with occurrence statistics across 88 algorithm instances.

2. **A quantitative algorithm phylogeny** based on motif fingerprints (88 x 367 occurrence matrices), revealing cross-family structural relationships invisible to gate-level analysis.

3. **Identification of ZX-irreducible motifs**---the four subgraph patterns that survive full ZX simplification---and experimental characterisation of their role in circuit expressibility and compressibility.

4. **Structure-guided circuit design**, yielding three novel circuits (DEB, TVH, ICC) and, subsequently, the ZXEA ansatz, which translates ZX-irreducible design principles into measurable VQE performance improvements.

The paper is structured as follows. Section 2 describes the pipeline architecture. Section 3 presents the motif library and phylogeny results. Section 4 analyses the ZX-irreducible motifs. Section 5 reports on data-driven circuit construction and benchmarks. Section 6 introduces the ZXEA ansatz. Section 7 discusses limitations and future work.

---

## 2. Pipeline Architecture

### 2.1 Conversion

The pipeline ingests quantum circuits as Qiskit `QuantumCircuit` objects from a registry of 32 algorithms spanning 10 families (Table 1). Each circuit is exported to OpenQASM 2.0 via `qiskit.qasm2.dumps()` and imported into PyZX [3], producing a ZX-calculus graph. We snapshot each graph at six simplification levels:

- **RAW**: No simplification; preserves compiler artifacts.
- **SPIDER_FUSED**: Spider fusion absorbs trivial identity wires (default for motif detection).
- **INTERIOR_CLIFFORD**: Interior Clifford simplification.
- **CLIFFORD_SIMP**: Full Clifford simplification.
- **FULL_REDUCE**: Complete ZX reduction.
- **TELEPORT_REDUCE**: Teleportation-based reduction.

Each snapshot is stored as a `ZXSnapshot` dataclass recording the graph, simplification level, qubit count, vertex/edge counts, T-gate count, and a heuristic for circuit-extractability.

### 2.2 Featurisation

ZX graphs are converted to labelled NetworkX graphs for subgraph analysis. Node attributes include vertex type (Z, X, H_BOX, BOUNDARY), exact phase (as a rational multiple of pi), and a *coarsened phase class*---one of {zero, pauli, clifford, t_like, arbitrary}---that enables structural matching modulo exact phase values. Edge attributes carry the edge type (SIMPLE or HADAMARD). A fixed 12-element feature vector per graph enables fast pre-filtering: [n_nodes, n_edges, n_Z, n_X, n_H_BOX, n_zero, n_clifford, n_t_like, n_arbitrary, n_hadamard_edges, n_simple_edges, density].

### 2.3 Motif Detection

Three complementary strategies generate candidate motifs:

**Top-down (hand-crafted).** Nine motifs encode known quantum primitives: phase gadgets (2-target and 3-target), CX pairs, ZZ interactions, syndrome extraction units, Toffoli cores, cluster chains, and Trotter layers. Six parametric variants use phase wildcards (PHASE_ANY, PHASE_ANY_NONZERO, PHASE_ANY_NONCLIFFORD), yielding 15 extended motifs.

**Bottom-up enumeration.** Starting from every interior vertex, we recursively expand connected subgraphs of size 3--6. Deduplication uses Weisfeiler-Leman (WL) graph hashing with VF2 isomorphism confirmation as a collision guard. Motifs appearing in at least 2 distinct algorithms are retained.

**Hybrid neighbourhood extraction.** BFS neighbourhoods of radius 2 are extracted around vertices flagged as "interesting" by three criteria: non-Clifford phases, high degree (>4), or colour boundary (adjacent to both Z and X spiders). Recurring neighbourhoods become candidate motifs.

The VF2 subgraph isomorphism engine uses semantic node matching (vertex type + phase class, with wildcard support) and edge matching (SIMPLE vs HADAMARD). A fast pre-filter on type counts, edge counts, and maximum degree prunes impossible matches before invoking VF2.

### 2.4 Composition

Confirmed motifs are wrapped as `ZXBox` objects with explicit left and right boundary vertex lists and a `BoundarySpec` (wire count, types, phases). Two composition modes are supported: sequential (connecting outputs to inputs) and parallel (tensor product). Simplification of composed boxes validates boundary survival; `spider_fusion` and `interior_clifford` are guaranteed safe, while `clifford_simp` and `full_reduce` may destroy boundaries (raising `BoundaryDestroyedError`).

### 2.5 Cataloguing and Decomposition

A persistent JSON catalogue stores each motif's graph, per-algorithm occurrence statistics, feature vector, and co-occurrence-based similarity scores. A greedy set-cover decomposer covers arbitrary ZX graphs with non-overlapping motif placements and reports a coverage ratio. An iterative inducer fills coverage gaps by extracting new motifs from uncovered regions.

**Table 1.** Algorithm corpus.

| Family | Algorithms | Instances |
|---|---|---|
| Entanglement | Bell, GHZ, W, Cluster state | 13 |
| Protocol | Teleportation, Superdense, Ent. swapping | 5 |
| Transform | QFT, Phase estimation | 8 |
| Oracle | Grover, Bernstein-Vazirani, Deutsch-Jozsa, Simon, Quantum counting | 18 |
| Variational | QAOA MaxCut, VQE UCCSD fragment, HEA | 10 |
| Error correction | Bit-flip, Phase-flip, Steane, Shor code | 4 |
| Simulation | Trotter Ising, Trotter Heisenberg, Quantum walk | 12 |
| Arithmetic | Ripple-carry adder, QFT adder | 6 |
| Distillation | BBPSSW, DEJMPS, Recurrence, Pumping | 4 |
| Machine learning | Quantum kernel, Data re-uploading | 8 |
| **Total** | **32 algorithms** | **88 instances** |

---

## 3. Motif Library and Algorithm Phylogeny

### 3.1 Library Composition

The pipeline discovers 367 motifs: 15 hand-crafted (extended) + 345 bottom-up + 7 neighbourhood-extracted. Motifs are categorised by universality:

- **Universal** (present in >= 80% of families): 8 motifs
- **Common** (40--80%): 29 motifs
- **Specific** (<40%): 323 motifs
- **Unused** (no matches): 7 motifs

The 8 universal motifs include `cx_pair` (present in all 10 families), `syndrome_extraction` (10/10), and their parametric variants. These represent the structural atoms of quantum computation: the CNOT interaction (Z-X spider pair) and the parity-check unit (Z hub with X spider leaves).

### 3.2 Fingerprint Matrix and PCA

The 88 x 367 fingerprint matrix records L1-normalised motif occurrence frequencies for each algorithm instance. Principal component analysis reveals meaningful structure: PC1 explains 33.2% and PC2 explains 20.1% of the variance, with the two components together capturing over 53% of structural variation (Figure 1).

The PCA embedding groups algorithm instances by family while revealing cross-family proximity. Notably, the variational and simulation families partially overlap along PC1, reflecting their shared ZZ-interaction structure.

### 3.3 Cross-Family Structural Relatives

Computing pairwise cosine similarity over the 367-dimensional fingerprint space, we identify 30 cross-family pairs with similarity above 0.70. The most striking findings:

**BBPSSW distillation and GHZ state preparation share 99% similarity.** BBPSSW distillation (4 qubits) and GHZ state preparation (5--6 qubits) achieve cosine similarities of 0.982--0.990. Both build on Bell pair creation (H + CNOT) and fan-out CNOT patterns. Despite solving entirely different problems---entanglement purification versus multi-party entanglement generation---their ZX diagrams are near-identical at the spider-fused level.

**QAOA MaxCut and Trotter-Ising simulation share 93% similarity.** All tested qubit counts of QAOA MaxCut and Trotter-Ising achieve pairwise cosine similarities of 0.906--0.936. Both encode ZZ interactions via CX-RZ-CX blocks; they differ only in interpretation (variational cost function versus Hamiltonian time evolution).

**GHZ, teleportation, and entanglement swapping form a cluster.** GHZ state preparation, quantum teleportation, and entanglement swapping achieve mutual similarities of 0.923--0.978, consistent with their shared reliance on Bell pair creation and CNOT fan-out.

### 3.4 Motif Coverage

Greedy set-cover decomposition achieves a mean coverage of 81.8% across all algorithm instances, with per-family breakdown:

| Family | Coverage |
|---|---|
| Simulation | 86.2% |
| Transform | 85.6% |
| Error correction | 85.5% |
| Entanglement | 85.0% |
| Machine learning | 83.4% |
| Oracle | 82.0% |
| Arithmetic | 79.4% |
| Variational | 73.6% |
| Protocol | 70.5% |
| **Distillation** | **61.1%** |

Distillation's anomalously low coverage (11 percentage points below the next-lowest family) signals that its ZX structure is poorly captured by the current motif library---a finding that directly motivates the DEB circuit construction in Section 5.

---

## 4. ZX-Irreducible Motifs

### 4.1 Cross-Level Survival Analysis

We track each of the 367 motifs through all six simplification levels, classifying their fate as *survives* (found at the target level), *transforms* (absent but a related motif appears), or *vanishes* (no trace remains). At the FULL_REDUCE level, only **four motifs survive**:

1. **phase_gadget_2t**: A T-like Z-spider hub connected to 2 phaseless targets via Hadamard edges. Implements exp(-i pi/8 Z_a Z_b).
2. **phase_gadget_3t**: Same structure with 3 targets.
3. **hadamard_sandwich**: A Clifford Z-spider flanked by two phaseless Z-spiders via Hadamard edges. Implements H-S-H = R_x(pi/2).
4. **cluster_chain**: Three phaseless Z-spiders in a Hadamard-edge chain. Encodes a 3-qubit linear cluster state fragment.

These four motifs are *ZX-irreducible*: no sequence of ZX-calculus rewrite rules (spider fusion, local complementation, pivoting, Hadamard reduction) can simplify them further. They represent the structural atoms---the minimal subgraph patterns that encode non-trivial quantum computation which the ZX calculus cannot absorb.

### 4.2 Interpretation

The phase gadgets survive because they contain T-gates (pi/4 phases), which are not in the Clifford group and therefore cannot be eliminated by Clifford-complete ZX rewrite rules. The hadamard sandwich survives because S-gates (pi/2 phases) sandwiched between Hadamard edges create a non-trivial Clifford structure that resists fusion. The cluster chain survives because Hadamard-edge chains encode a measurement-based computation primitive that is irreducible in the ZX framework.

Together, these four motifs span two key computational resources: **non-Clifford magic** (phase gadgets) and **measurement-based entanglement** (hadamard sandwich, cluster chain). Any circuit that performs non-trivial computation must contain instances of these irreducible atoms in its fully reduced ZX diagram.

---

## 5. Data-Driven Circuit Construction

Guided by phylogeny findings---specifically, cross-family similarities and motif coverage gaps---we construct three novel quantum circuits by composing motifs from the library.

### 5.1 Distillation-Entanglement Bridge (DEB)

**Motivation.** The 99% BBPSSW-GHZ similarity and distillation's 61% coverage gap suggest a structural void between the distillation and entanglement families.

**Construction.** DEB is a 4-qubit, 12-gate, depth-6 circuit with one continuous parameter (theta). It combines GHZ-style fan-out (H + CNOT chain on the first 2 qubits), RY(theta) rotations on all qubits, bilateral CNOTs from the distillation template, and Hadamard + reverse-CNOT syndrome decoding.

**Properties.** DEB is a deterministic conditional Bell pair factory: measuring the syndrome register (qubits 2--3) *always* collapses the data register (qubits 0--1) into a maximally entangled Bell state with concurrence C = 1.0, for every measurement outcome at every value of theta. The parameter theta controls only the probability distribution over which Bell state is produced.

**Benchmarks.** Compared against H+CX Bell pair generation and BBPSSW/DEJMPS distillation, DEB shows no advantage: under channel noise, DEB's weighted-average concurrence equals H+CX at every noise level (difference = 0.000); under gate noise, DEB's 12 gates accumulate proportionally more error than H+CX's 2 gates. The comparison with BBPSSW/DEJMPS is category-level mismatched: DEB generates entanglement from a product state, while distillation protocols purify existing noisy pairs.

**Novelty score: 0.654** (highest of the three circuits; nearest existing algorithm: W state at 0.346 similarity). DEB occupies a genuinely novel structural region but addresses no demonstrated operational need.

### 5.2 Trotter-Variational Hybrid (TVH)

**Motivation.** The 93% QAOA-Trotter similarity motivates a circuit bridging the variational and simulation families.

**Construction.** TVH is a 4-qubit, 60-gate, depth-31 circuit with 2 continuous parameters (gamma, beta). Per layer: (1) ZZ-interaction backbone (CX-RZ(gamma)-CX), (2) cluster_chain entangling (H-CZ-H), (3) hadamard_sandwich (H-RZ(beta)-H) on alternating qubits, (4) RX(2*beta) mixer. Layers 2--3 incorporate ZX-irreducible motifs not present in standard QAOA or Trotter circuits.

**VQE benchmarks.** On the 4-qubit Heisenberg model (E_0 = -6.464), TVH achieves only -3.356 (48.1% error), far behind HEA's -6.235 (3.5% error). The core problem: 60 gates over-determined by only 2 parameters, creating flat energy landscapes with gradient variance 6x below QAOA.

**Expressibility.** TVH's one genuinely strong metric: with only 2 parameters, it achieves mean pairwise fidelity of 0.115 (1.9x Haar random), compared to QAOA's 0.278 (4.5x Haar) at the same parameter count. Ablation confirms the ZX-irreducible motifs are responsible: removing cluster_chain and hadamard_sandwich degrades expressibility by 2x (to 0.238, or 3.8x Haar).

**Design insight.** ZX-irreducible motifs improve Hilbert space coverage per parameter, but high expressibility with few parameters implies extreme parameter sensitivity---exactly the barren plateau phenomenon. The extractable principle: use ZX-irreducible motifs as *fixed entangling layers* combined with *per-qubit variational parameters*.

### 5.3 Irreducible Core Circuit (ICC)

**Motivation.** Construct a circuit exclusively from the four ZX-irreducible motifs to explore the properties of ZX-irreducibility-first design.

**Construction.** ICC is a 4-qubit, 39-gate, depth-20 Clifford+T circuit with zero continuous parameters. Structure: cluster_chain encoding -> phase_gadget_2t on pairs (0,1) and (2,3) -> hadamard_sandwich on all qubits -> phase_gadget_3t on all qubits -> cluster_chain decoding.

**Properties.** T-count = 3, provably optimal via ZX verification. ZX reduction ratio = 70% (63 vertices / 75 edges reduced to 19 / 28), significantly above the typical 40--60% for random Clifford+T circuits. XXXX parity symmetry. Genuine 4-party entanglement: zero pairwise concurrence but bipartite entropy S(01:23) = 1.20 bits with Schmidt rank 4.

**Scaling.** T-count scales sublinearly as floor(n/2) + 1; depth scales approximately as n + 16.

**Verdict.** ICC is a proof-of-concept for ZX-irreducibility-first design, not a practical algorithm. Its primary utility is as a benchmark for ZX-based quantum compilers: a 39-gate circuit with a known-optimal 19-vertex ZX representation and provably minimal T-count.

### 5.4 Summary of Discovered Circuits

| Criterion | DEB | TVH | ICC |
|---|---|---|---|
| Novelty score | 0.654 (highest) | 0.257 (lowest) | 0.359 |
| Clean mathematical property | Deterministic Bell pairs | Near-Haar expressibility | T-count optimal |
| Beats industry standard | No | No | N/A |
| ZX reduction ratio | 39% | 69% | 70% |
| Primary value | Structural exploration | Design insight (Sec. 6) | Compiler benchmark |

None of the three circuits outperforms the simplest standard solution for its hypothesised task. However, the TVH analysis yields an actionable design principle that we validate in Section 6.

---

## 6. ZXEA: ZX-Irreducible Entangling Ansatz

### 6.1 Design Rationale

The TVH analysis (Section 5.2) established two facts: (1) ZX-irreducible motifs (cluster_chain, hadamard_sandwich) improve expressibility per parameter, and (2) sharing parameters globally across 60 gates produces barren plateaus. The natural hypothesis: combine ZX-irreducible entangling layers with per-qubit variational parameters to achieve both expressibility and trainability.

### 6.2 Ansatz Definitions

**ZXEA.** Per layer: RY(theta_i) + RZ(theta_j) per qubit (variational), then a fixed cluster_chain layer (H-CZ-H on nearest-neighbour pairs), then RY(theta_k) per qubit. For 4 qubits and 2 layers: 24 parameters, 46 total gates, 6 two-qubit gates, depth 15.

**ZXEA-H.** ZXEA plus a fixed hadamard_sandwich layer (H-S-H) per layer. Same 24 parameters, 70 gates, depth 21.

**HEA baseline.** Standard hardware-efficient ansatz: RY + RZ per qubit, CX chain, RY per qubit. 24 parameters, 30 gates, 6 two-qubit gates, depth 11.

**ZXEA topology variants.** We also test ZXEA-grid (2D grid CZ pattern, 8 two-qubit gates) and ZXEA-alt (alternating brick-layer CZ, 3 two-qubit gates).

### 6.3 VQE Performance

All ansatze are optimised with COBYLA (40 random restarts, 800 max iterations) on noiseless statevector simulation.

**4-qubit Heisenberg chain** (E_0 = -6.4641):

| Ansatz | Best Energy | Error (%) | Mean +/- Std |
|---|---|---|---|
| **ZXEA** | **-6.327** | **2.1** | -6.206 +/- 0.338 |
| ZXEA-H | -6.327 | 2.1 | -6.206 +/- 0.338 |
| HEA | -6.235 | 3.5 | -6.134 +/- 0.228 |
| QAOA-flex | -3.477 | 46.2 | -3.148 +/- 0.657 |
| TVH | -3.356 | 48.1 | -2.157 +/- 0.959 |

ZXEA achieves 2.1% error versus HEA's 3.5%---a 40% relative improvement---with the same parameter count (24) and the same number of two-qubit gates (6).

**4-qubit transverse-field Ising model** (E_0 = -4.7588):

| Ansatz | Best Energy | Error (%) |
|---|---|---|
| ZXEA | -4.743 | 0.3 |
| ZXEA-H | -4.609 | 3.1 |
| HEA | -4.757 | 0.0 |

Near parity, with HEA slightly better on TFIM.

**Scaling (Heisenberg):**

| Qubits | ZXEA Error | HEA Error |
|---|---|---|
| 4 | 2.1% | 3.5% |
| 6 | 15.0% | 6.2% |
| 8 | 13.6% | 6.8% |

ZXEA's advantage at 4 qubits does not persist at 6--8 qubits, where HEA's CX-chain entangling appears better suited to the nearest-neighbour Heisenberg correlation structure. This suggests the cluster-chain entangling pattern, while effective at small scales, may not match the entanglement structure needed for larger Heisenberg ground states.

### 6.4 Expressibility and Trainability

| Ansatz | Mean Fidelity | Ratio to Haar | Mean Var(dE/dtheta) |
|---|---|---|---|
| ZXEA | 0.0624 | 1.06x | 0.286 |
| ZXEA-H | 0.0599 | 1.02x | 0.258 |
| HEA | 0.0619 | 1.05x | 0.243 |
| TVH | 0.1147 | 1.95x | 9.373 |
| QAOA-flex | 0.2311 | 3.93x | 4.757 |

All three per-qubit-parameter ansatze (ZXEA, ZXEA-H, HEA) achieve near-Haar expressibility and comparable gradient variance---well above the barren plateau regime. ZXEA-H is closest to the Haar random limit (1.02x). The per-qubit parameterisation successfully resolves TVH's barren plateau problem while retaining near-Haar state space coverage.

### 6.5 Noise Resilience and Gate Efficiency

Under per-gate depolarising noise (p = 0.001): HEA achieves -6.050, ZXEA achieves -6.043, ZXEA-H achieves -5.899. HEA's advantage grows with noise due to its lower gate count (30 vs 46 vs 70).

Gate efficiency (error per two-qubit gate): ZXEA achieves 0.023% per two-qubit gate versus HEA's 0.038%---a 1.7x improvement, indicating more effective use of entangling resources.

### 6.6 Convergence

| Ansatz | @50 evals | @200 evals | @800 evals |
|---|---|---|---|
| ZXEA | -4.938 | -6.267 | -6.327 |
| HEA | -5.192 | -6.220 | -6.235 |

ZXEA converges to a better final energy despite slightly slower initial progress. By 200 evaluations, ZXEA has already surpassed HEA's final result.

### 6.7 Topology Comparison

At 8 qubits on Heisenberg:

| Topology | Error (%) | 2-Qubit Gates |
|---|---|---|
| ZXEA (chain) | 13.6 | 6 per layer |
| ZXEA-alt (brick) | 17.0 | 3 per layer |
| ZXEA-grid (2D) | 26.4 | 8 per layer |

The 12.8 percentage-point spread between topologies indicates that entangling connectivity significantly impacts performance, warranting further topology exploration for larger systems.

---

## 7. Discussion

### 7.1 What the Methodology Produces

The ZX motif fingerprinting methodology successfully accomplishes three goals:

**Quantitative taxonomy.** The 367-dimensional fingerprint space provides a principled distance metric between quantum algorithms. The cross-family similarities it reveals (BBPSSW-GHZ at 99%, QAOA-Trotter at 93%) are not heuristic claims but measured cosine similarities that are stable across qubit counts and reproducible.

**Structural gap identification.** Coverage analysis pinpoints families whose ZX structure is poorly captured by the motif library. Distillation's anomalous 61% coverage (11 points below the next family) correctly identifies it as structurally under-represented.

**Actionable design principles.** The TVH-to-ZXEA progression demonstrates that motif analysis can yield extractable insights: ZX-irreducible motifs improve expressibility per parameter, and this principle translates to measurable VQE improvement when combined with per-qubit parameterisation.

### 7.2 What It Does Not Produce

**Operationally superior algorithms.** None of the three directly constructed circuits (DEB, TVH, ICC) outperforms the simplest standard solution for its hypothesised task. The methodology finds structurally interesting circuits, not necessarily operationally superior ones. The most operationally valuable output (ZXEA) came from extracting a design principle from the failed TVH, not from direct gap-filling.

**Scalability guarantees.** ZXEA's advantage at 4 qubits does not persist at 6--8 qubits. The cluster-chain entangling pattern may be best suited to specific correlation structures rather than serving as a universal replacement for CX-chain entangling.

### 7.3 Limitations

**VF2 complexity.** Subgraph isomorphism is NP-complete in the worst case. Our pre-filtering heuristics and the moderate size of ZX graphs (typically <500 vertices after spider fusion) keep runtime practical, but scaling to circuits with thousands of qubits would require approximate matching methods.

**Rewrite equivalence.** The matcher finds literal structural matches only. Structurally different subgraphs that are ZX-equivalent (related by rewrite rules not applied at the current simplification level) are treated as distinct. A rewrite-aware matching framework could uncover additional structural relationships.

**Clifford+T bias.** PyZX is strongest on Clifford+T circuits. Variational circuits with arbitrary rotation angles produce "arbitrary" phase classes that match too broadly, potentially inflating similarity scores for variational family members.

**Phase coarsening.** The five-level phase classification (zero, pauli, clifford, t_like, arbitrary) sacrifices precision for matching generality. Two circuits with different non-Clifford phases would appear structurally identical.

### 7.4 Future Work

Several directions emerge from this work:

1. **Hamiltonian-aware topology search.** ZXEA's topology dependence (Section 6.7) suggests that matching the entangling topology to the target Hamiltonian's interaction graph could restore the small-scale advantage at larger qubit counts.

2. **Rewrite-aware motif matching.** Extending the matcher to recognise ZX-equivalent but structurally distinct subgraphs would provide a finer-grained taxonomy.

3. **Larger corpora.** Extending the algorithm registry beyond 32 algorithms---particularly to include quantum chemistry, quantum machine learning, and fault-tolerant constructions---would test the generality of the discovered motifs and phylogenetic structure.

4. **Hardware-native motifs.** Defining motifs that respect hardware connectivity constraints (e.g., linear, grid, heavy-hex topologies) could bridge the gap between abstract ZX analysis and physical circuit optimisation.

5. **Automated ansatz generation.** The TVH-to-ZXEA progression was manual. Automating the extraction of design principles from failed candidates---and their translation into improved architectures---is an open challenge for data-driven quantum circuit design.

---

## 8. Conclusion

We have presented ZX-Motifs, a pipeline for structural analysis of quantum algorithms via ZX-calculus subgraph patterns. Applied to a corpus of 32 algorithms (88 instances, 10 families), the pipeline discovers 367 motifs, constructs a quantitative phylogeny revealing deep cross-family structural relationships, and identifies four ZX-irreducible motifs that survive full simplification. Data-driven circuit construction guided by the phylogeny produces three novel circuits that, while not outperforming standard alternatives, yield an actionable design principle: ZX-irreducible entangling layers improve expressibility per parameter. The ZXEA ansatz validates this principle, achieving 2.1% VQE error on the 4-qubit Heisenberg model versus 3.5% for the hardware-efficient ansatz at equal parameter count and two-qubit gate budget.

The honest assessment: ZX motif analysis is a viable tool for algorithm taxonomy and structure-guided design, but it is not yet a reliable route to operationally superior quantum algorithms. The methodology's strength lies in revealing structural relationships and generating design hypotheses. Translating those hypotheses into practical advantage remains an iterative, partly manual process. We view this work as a proof of concept for a research programme in which structural analysis of the ZX calculus informs---but does not replace---physics-guided quantum algorithm design.

---

## References

[1] B. Coecke and R. Duncan, "Interacting quantum observables: categorical algebra and diagrammatics," *New J. Phys.*, vol. 13, 043016, 2011.

[2] J. van de Wetering, "ZX-calculus for the working quantum computer scientist," arXiv:2012.13966, 2020.

[3] A. Kissinger and J. van de Wetering, "PyZX: Large scale automated diagrammatic reasoning," *EPTCS*, vol. 318, pp. 229--241, 2020.

[4] L. P. Cordella, P. Foggia, C. Sansone, and M. Vento, "A (sub)graph isomorphism algorithm for matching large graphs," *IEEE Trans. PAMI*, vol. 26, no. 10, pp. 1367--1372, 2004.

[5] A. Kandala et al., "Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets," *Nature*, vol. 549, pp. 242--246, 2017.

[6] E. Farhi, J. Goldstone, and S. Gutmann, "A quantum approximate optimization algorithm," arXiv:1411.4028, 2014.

[7] S. Sim, P. D. Johnson, and A. Aspuru-Guzik, "Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms," *Adv. Quantum Technol.*, vol. 2, 1900070, 2019.

[8] M. Cerezo et al., "Cost function dependent barren plateaus in shallow parametrized quantum circuits," *Nat. Commun.*, vol. 12, 1791, 2021.

[9] C. H. Bennett, G. Brassard, S. Popescu, B. Schumacher, J. A. Smolin, and W. K. Wootters, "Purification of noisy entanglement and faithful teleportation via noisy channels," *Phys. Rev. Lett.*, vol. 76, pp. 722--725, 1996.

[10] D. Deutsch, A. Ekert, R. Jozsa, C. Macchiavello, S. Popescu, and A. Sanpera, "Quantum privacy amplification and the security of quantum cryptography over noisy channels," *Phys. Rev. Lett.*, vol. 77, pp. 2818--2821, 1996.

---

## Appendix A: Reproduction

```bash
# From repository root:
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run phylogeny analysis (~15 min):
python scripts/discover_phylogeny.py

# Generate candidate circuits (~10 sec):
python scripts/discover_algorithm.py

# Run ZXEA benchmarks (~30 min):
python scripts/benchmark_zxea.py

# Run test suite:
.venv/bin/python -m pytest tests/ -v
```

All numerical results were computed on noiseless statevector simulation using Qiskit and verified via ZX tensor comparison in PyZX. VQE optimisation used COBYLA with 40 random restarts and 800 maximum iterations per restart.

## Appendix B: ZX-Irreducible Motif Diagrams

The four ZX-irreducible motifs, shown in ZX-calculus notation:

**phase_gadget_2t:** A Z-spider with phase pi/4 (T-gate) connected via Hadamard edges to two phaseless Z-spider targets. Implements the two-body phase rotation exp(-i pi/8 Z_a Z_b).

**phase_gadget_3t:** Same as above with three targets: exp(-i pi/8 Z_a Z_b Z_c).

**hadamard_sandwich:** A Z-spider with phase pi/2 (S-gate) flanked by two phaseless Z-spiders via Hadamard edges. Equivalent to H-S-H = R_x(pi/2).

**cluster_chain:** Three phaseless Z-spiders connected in a linear chain by Hadamard edges. Encodes the adjacency structure of a 3-qubit linear cluster state.

## Appendix C: ZXEA Circuit Construction

For n qubits and L layers, the ZXEA circuit is:

```
for layer in 1..L:
    for qubit i in 0..n-1:
        RY(theta_{layer,i,0})  on qubit i
        RZ(theta_{layer,i,1})  on qubit i
    for pair (i, i+1) in 0..n-2:        # cluster_chain entangling
        H on qubit i
        CZ(i, i+1)
        H on qubit i
    for qubit i in 0..n-1:
        RY(theta_{layer,i,2})  on qubit i
```

Total parameters: 3nL. Total two-qubit gates: (n-1)L. The H-CZ-H sequence implements the cluster_chain motif, creating graph-state entanglement between adjacent qubits.
