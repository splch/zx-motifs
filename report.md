**Programmatic Quantum Algorithm Discovery**

via ZX-Calculus Sub-Diagram Mining and Composition

Research Pipeline Design and Feasibility Analysis

A Seven-Stage Framework: From Qiskit Implementation through PyZX

Transformation to Novel Algorithm Benchmarking

March 2026


# 1. Executive Summary

This report presents a detailed feasibility analysis and implementation guide for a seven-stage pipeline designed to programmatically discover new quantum algorithms. The approach combines IBM’s Qiskit framework for implementing a diverse corpus of known quantum algorithms, the PyZX library for translating those circuits into ZX-calculus diagrams, a novel sub-diagram mining phase to identify reusable structural motifs (termed “ZX-Webs”), a combinatorial composition engine that reassembles those motifs into candidate algorithms, a circuit-extraction filter that discards candidates lacking the flow properties needed for conversion back to executable circuits, a benchmarking stage that evaluates surviving candidates against established application suites, and a reporting mechanism for surfacing algorithms that outperform existing solutions.

Each stage rests on a distinct body of prior work in quantum computing, graphical languages, and combinatorial optimization. Some stages are well-supported by existing open-source tooling, while others—particularly the sub-diagram mining and composition phases—represent genuinely novel research contributions that have no off-the-shelf solution and require careful theoretical grounding.


# 2. Stage 1: Implementing Quantum Algorithms in Qiskit

## **2.1 Current State of Qiskit**

Qiskit remains the most widely adopted open-source quantum SDK. As of early 2026, the Qiskit SDK provides modules for building quantum circuits, a library of predefined gates and parametrized circuits, a quantum information module for working with states and operators, and a transpiler that optimizes circuits for specific hardware backends. The separate qiskit-algorithms library (now community-maintained under Apache 2.0 license following IBM’s shift toward the Qiskit Functions catalog) organizes algorithms by function: eigensolvers, amplitude estimators, phase estimators, time evolvers, and variational routines.

## **2.2 Recommended Algorithm Corpus**

To maximize the diversity of ZX-diagram structures encountered in later stages, the corpus should span multiple algorithm families. The following table outlines a recommended starting corpus:

| **Category** | **Algorithms** | **Qubit Range** | **ZX Diversity** |
| - | - | - | - |
| Search/Amplitude | Grover, Amplitude Estimation, Amplitude Amplification | 3–15 | Oracle structures, diffusion operators |
| Fourier/Phase | QFT, QPE, HHL, Shor (toy) | 4–20 | Phase-rotation ladders, controlled-rotation fans |
| Variational | VQE, QAOA, VarQITE | 4–30 | Parametrized ansatz blocks, entanglers |
| Simulation | Trotter, qDRIFT, LCU | 6–40 | Hamiltonian decomposition gadgets |
| Error Correction | Steane, Surface Code prep | 7–17 | Syndrome extraction patterns |
| Arithmetic | Adders, Multipliers, Comparators | 4–30 | Carry-chain patterns, Toffoli decompositions |


## **2.3 Implementation Guidance**

Each algorithm should be implemented as a Python function returning a Qiskit QuantumCircuit with parametrized sizes. The unitary portion of each circuit (excluding measurements and classical control) should be exportable to OpenQASM 2.0, since this is the interchange format consumed by PyZX. Qiskit’s qc.qasm() method produces this representation. Circuits should be transpiled to a universal gate set (Clifford+T or \{CX, Rz, H\}) before export, since PyZX is most effective when operating on circuits in these gate sets.


# 3. Stage 2: Conversion to ZX-Diagrams with PyZX

## **3.1 The ZX-Calculus Background**

The ZX-calculus is a graphical language for reasoning about quantum computations. A ZX-diagram consists of Z-spiders (green nodes) and X-spiders (red nodes) connected by wires, with Hadamard edges represented as blue lines or yellow boxes. The key insight is that ZX-diagrams are strictly more expressive than circuit notation: they can represent any linear map between qubits and come equipped with a set of rewrite rules that are complete for the Clifford+T fragment—meaning any equation between Clifford+T linear maps that is true can be derived using the rewrite rules.

## **3.2 Qiskit-to-PyZX Conversion**

The interoperability pathway between Qiskit and PyZX is well-established. There are two primary approaches:

1. **QASM interchange: **Export the Qiskit circuit via qc.qasm(), then import into PyZX using zx.Circuit.from\_qasm(qasm\_string). PyZX will ignore measurement and classical control instructions, retaining only the unitary portion.

2. **Direct transpiler pass: **The qiskit-zx-transpiler package (available on PyPI) provides a ZXPass that can be inserted into Qiskit’s transpilation pipeline. It converts a Qiskit DAGCircuit to a PyZX Circuit, runs optimization, and converts back.

## **3.3 Simplification to Graph-Like Form**

Once a circuit is loaded into PyZX, the simplification pipeline is:

1. **Graph-like conversion: **zx.to\_graph\_like(g) converts all spiders to Z-spiders connected by Hadamard edges. Every circuit can be brought into this normal form.

2. **Interior Clifford simplification: **zx.simplify.interior\_clifford\_simp(g) applies local complementation and pivoting rules to remove interior proper-Clifford spiders.

3. **Full reduction: **zx.full\_reduce(g) produces the “reduced gadget form” where every internal spider is non-Clifford or part of a non-Clifford phase gadget.

For the purpose of sub-diagram mining, it is advisable to store diagrams at multiple levels of simplification (raw graph-like, after Clifford simplification, and fully reduced) since patterns may be visible at different granularities.

## **3.4 Key Code Pattern**

import pyzx as zx

c = zx.Circuit.from\_qasm(qasm\_str)

g = c.to\_graph()

zx.simplify.full\_reduce(g)

\# g is now a simplified ZX-diagram (Graph object)


# 4. Stage 3: Mining Common Sub-Diagrams (ZX-Webs)

## **4.1 Conceptual Framework**

This is the most novel and research-intensive stage of the pipeline. The concept of “ZX-Webs”—reusable structural motifs extracted from ZX-diagrams—draws direct inspiration from recent work on automated gadget discovery in quantum circuits. A September 2025 paper by researchers studying reinforcement learning for quantum error correction demonstrated that representing quantum circuits as directed graphs and searching for repeated subgraphs successfully identifies composite gate structures (“gadgets”) that enhance algorithmic performance. The approach identified two new gadget families that were previously unknown.

Extending this idea to ZX-diagrams offers a potentially richer space: because ZX-diagrams are a more flexible representation than circuits, the sub-structures found may include patterns that have no clean circuit-level equivalent—multi-leg phase gadgets, star graphs, or chains of fused spiders that encode entanglement patterns not naturally visible in gate notation.

## **4.2 Sub-Graph Isomorphism Approach**

The core algorithmic challenge is frequent subgraph mining over a corpus of labeled graphs. Each ZX-diagram is a graph with node labels (spider type: Z or X, plus phase value) and edge labels (simple or Hadamard). The task is to find subgraphs that appear across multiple diagrams or repeatedly within a single diagram. Established algorithms for this task include:

- **VF2/VF3: **Exact subgraph isomorphism algorithms available in NetworkX. Suitable for small patterns (up to ~20 nodes) against medium graphs.

- **gSpan: **Frequent subgraph mining that enumerates all subgraphs exceeding a support threshold. Available in Python via the gspan-mining package.

- **GNN-based approaches: **Graph neural networks trained to embed subgraph motifs into a latent space, enabling approximate matching at scale. Relevant recent work includes AltGraph’s use of DAG Variational Autoencoders for quantum circuit representation.

## **4.3 Practical Considerations**

Several practical decisions shape the design of this stage:

- **Phase abstraction: **Phase values should be grouped into equivalence classes (Pauli phases, Clifford phases, T-phases, arbitrary) rather than treated as exact values, to allow matching of structurally similar motifs with different rotations.

- **Boundary handling: **Each extracted sub-diagram must track its open wires (boundary spiders connected to the rest of the diagram) so that it can be composed with other sub-diagrams in Stage 4.

- **Minimum support threshold: **A sub-diagram should appear in at least 2–3 distinct algorithms to be considered a “web.” Cross-algorithm occurrences are more valuable than repeated occurrences within a single large diagram.

- **Annotation: **Each ZX-Web should be annotated with metadata: source algorithms, qubit count, spider count, phase distribution, boundary structure (number and type of open legs).

## **4.4 Expected Motif Categories**

Based on the structure of known quantum algorithms, the following categories of ZX-Webs are likely to emerge:

- **Phase fans: **Central spider connected to multiple boundary spiders via Hadamard edges, encoding controlled phase rotations (common in QFT, QPE).

- **Entanglement stars: **Fully connected clusters of spiders representing GHZ-like state preparation or multi-qubit entanglement.

- **Gadget pairs: **Phase gadgets consisting of a phase spider connected to a target set via Hadamard edges (common in Toffoli decompositions and arithmetic circuits).

- **Clifford skeletons: **Residual graph structure after full reduction, representing the purely Clifford portion of an algorithm.


# 5. Stage 4: Composing Candidate Algorithms from ZX-Webs

## **5.1 Composition Mechanics**

The composition of ZX-Webs into candidate algorithms is the creative engine of the pipeline. Because ZX-diagrams compose by connecting open wires (boundary spiders), the composition operation is straightforward at the graph level: match compatible boundary interfaces and merge. The challenge lies in generating compositions that are semantically meaningful rather than random graphs.

## **5.2 Composition Strategies**

1. **Template-guided composition: **Define high-level algorithm templates (e.g., “oracle + amplification + measurement” or “state preparation + evolution + readout”) and fill template slots with compatible ZX-Webs. This constrains the search space to algorithmically plausible structures.

2. **Evolutionary composition: **Use genetic algorithms where individuals are compositions of ZX-Webs, crossover swaps sub-diagrams between candidates, and fitness is measured by extractability and circuit metrics (depth, gate count, T-count).

3. **Type-constrained random walks: **Treat the library of ZX-Webs as a graph grammar and perform random walks through the composition space, constrained by boundary compatibility (matching number and type of open legs).

## **5.3 Maintaining Soundness**

A critical concern is ensuring that composed diagrams represent valid quantum operations (i.e., linear maps between qubits). Several safeguards apply:

- **Unitarity checks: **For small diagrams (up to ~12 qubits), compute the matrix representation and verify unitarity via PyZX’s compare\_tensors. For larger diagrams, use statistical tests on random input states.

- **Boundary balance: **The composed diagram must have equal numbers of inputs and outputs to represent a unitary operation on a fixed number of qubits.

- **Semantic labeling: **Annotate ZX-Webs with the computational role they play (state preparation, entangling, phase accumulation, readout) and enforce role-consistency during composition.


# 6. Stage 5: Circuit Extraction Filtering

## **6.1 The Circuit Extraction Problem**

This is the single most critical bottleneck in the pipeline. Converting an arbitrary ZX-diagram back into an executable quantum circuit is provably hard: the general circuit extraction problem for ZX-diagrams is \#P-hard, meaning it is at least as hard as strong simulation of quantum circuits. This result, proven by de Beaudrap and colleagues, means that no efficient algorithm can extract circuits from arbitrary ZX-diagrams unless fundamental complexity-theoretic assumptions are violated.

## **6.2 Tractable Extraction via Flow Properties**

Fortunately, efficient extraction is possible for diagrams possessing certain structural properties imported from measurement-based quantum computing (MBQC). The key hierarchy of flow conditions is:

| **Flow Type** | **Description** | **Extraction Status** |
| - | - | - |
| Causal flow | Strict one-to-one mapping of measured qubits to corrections | Polynomial extraction, most restrictive |
| gFlow | Generalized flow allowing set-valued correction functions | Polynomial extraction, standard in PyZX |
| Pauli flow | Most general known condition; plane-dependent corrections | Polynomial extraction, most permissive known |
| No flow | Arbitrary ZX-diagram without flow structure | \#P-hard in general |


PyZX’s extract\_circuit function implements extraction for diagrams with gFlow. The Quantinuum tket library’s ZX module supports extraction from both gFlow and extended gFlow. Recent 2025 PhD work by Mitosek at the University of Birmingham has improved the algorithmic complexity of Pauli flow detection from O(n⁴) to O(n³), making it increasingly practical to check for the most general flow condition.

## **6.3 Flow Preservation Strategy**

Rather than generating arbitrary compositions and then hoping they have flow, a more effective strategy is to ensure flow is preserved throughout the composition process. Recent research on flow-preserving ZX-calculus rewrite rules (McElvanney and Kissinger, 2023) has identified rules that increase qubit count while preserving Pauli flow. This work proved that the “neighbour unfusion” rule always preserves Pauli flow, even when it may break gFlow. The implication for the pipeline is significant: if all ZX-Webs originate from circuits (which always have causal flow), and all composition operations use flow-preserving rules, then the resulting candidate diagrams are guaranteed to be extractable.

## **6.4 Extraction Pipeline**

For each candidate diagram that passes the flow check, extraction proceeds as follows:

3. Check for gFlow using PyZX or tket’s built-in algorithm.

4. If gFlow exists, extract using zx.extract\_circuit(g).

5. If gFlow is absent, check for Pauli flow using Mitosek’s O(n³) algorithm.

6. If Pauli flow exists, extract using tket’s ZXDiagram.to\_circuit().

7. Apply post-extraction optimization: zx.basic\_optimization and transpilation to the target gate set.

8. Discard candidates where no efficient extraction is possible.

For circuits that survive extraction but have been inflated with CNOT gates (a known artifact of the gFlow extraction process), the teleport\_reduce method in PyZX provides an alternative that preserves circuit structure while reducing T-count.


# 7. Stage 6: Benchmarking Against Application Suites

## **7.1 Available Benchmark Suites**

Two established benchmark suites are well-suited for evaluating candidate quantum algorithms:

**QASMBench** is a low-level OpenQASM benchmark suite developed at Pacific Northwest National Laboratory. It consolidates quantum routines from chemistry, simulation, linear algebra, searching, optimization, arithmetic, machine learning, fault tolerance, and cryptography. QASMBench partitions benchmarks into three categories by qubit count (small: 2–5, medium: 6–27, large: 28+) and defines six evaluation metrics including gate density, retention lifespan, measurement density, and entanglement variance.

**SupermarQ** is a scalable, hardware-agnostic benchmark suite that applies classical benchmarking methodology to quantum computing. It defines hardware-agnostic feature vectors (Program Communication, Entanglement Ratio, Critical Depth, Liveness) and includes benchmarks from diverse domains: QAOA for combinatorial optimization, VQE for chemistry, GHZ state preparation, Mermin–Bell inequality tests, and error correction subroutines.

## **7.2 Benchmarking Methodology**

Each surviving candidate algorithm should be benchmarked along four axes:

- **Circuit metrics: **Total gate count, two-qubit gate count, T-count, circuit depth, circuit width. Compare against the best-known circuit for the same computational task.

- **Simulation fidelity: **Run on Qiskit Aer simulators with and without noise models. Compare output distributions against ideal distributions via total variation distance (TVD) or fidelity.

- **Hardware execution: **Where possible, execute on IBM Quantum hardware and compare fidelity across backends. The Qiskit Functions catalog provides pre-built circuit functions with error mitigation.

- **Computational advantage: **For optimization problems, compare the solution quality achieved by the candidate algorithm against classical baselines and existing quantum algorithms on the same problem instance.

## **7.3 Comparison Baselines**

For each candidate algorithm, the relevant baseline depends on the computational task it performs. If a candidate produces a circuit that solves MaxCut, compare against standard QAOA at the same depth. If it produces a Hamiltonian simulation circuit, compare against Trotter-based methods. The ZX-optimized circuit should also be compared against the best circuit obtainable by Qiskit’s default transpiler at optimization level 3, to isolate the contribution of the ZX-based discovery process from standard compilation optimization.


# 8. Stage 7: Reporting Novel Algorithms

## **8.1 Criteria for Novelty**

A candidate algorithm qualifies as “novel and outperforming” if it satisfies all of the following criteria:

- It implements a well-defined computational task (not merely a random unitary).

- Its extracted circuit achieves lower depth, gate count, or T-count than the best known circuit for the same task.

- The improvement persists after applying standard optimization passes to both the candidate and the baseline.

- For variational algorithms, the improvement in solution quality exceeds the statistical noise of the optimization.

## **8.2 Interpretability**

One of the strongest arguments for using ZX-calculus as the intermediate representation is interpretability. The ZX-Web composition that produced a novel algorithm carries semantic information: which motifs were combined, from which source algorithms they originated, and what computational role each plays. This provenance trail can guide human researchers in understanding why the new algorithm works, potentially leading to theoretical insights beyond the purely empirical discovery.

## **8.3 Reporting Artifacts**

For each novel algorithm discovered, the pipeline should produce: the ZX-diagram of the discovered algorithm (exportable to TikZ for publication), the extracted quantum circuit in OpenQASM format, benchmarking results in tabular and graphical form, the ZX-Web composition recipe (which sub-diagrams were combined and how), and a provenance log linking each component to its source algorithm.


# 9. Key Risks, Open Challenges, and Mitigations

## **9.1 Circuit Extraction Bottleneck**

The \#P-hardness of general circuit extraction is the pipeline’s most significant theoretical risk. The mitigation strategy is to constrain the composition process to only produce diagrams with guaranteed flow properties, but this necessarily limits the creative search space. An alternative approach is to accept that some fraction of candidates will be discarded at the extraction stage and design the composition engine to produce a sufficiently large pool.

## **9.2 Semantic Meaningfulness**

Random compositions of structural motifs are overwhelmingly unlikely to produce algorithms that solve meaningful computational problems. The template-guided composition strategy (Stage 4) is the primary mitigation, but it requires careful design of templates that are neither too rigid (precluding discovery) nor too loose (producing noise).

## **9.3 Scalability of Sub-Graph Mining**

Frequent subgraph mining is NP-hard in general, and exact subgraph isomorphism on ZX-diagrams with dozens to hundreds of nodes will be computationally intensive. Practical mitigations include working with simplified (reduced) diagrams, using approximate matching via graph neural networks, and pre-filtering by coarse graph statistics (node count, degree distribution, phase histogram).

## **9.4 Qiskit-PyZX Gate Set Mismatches**

Not all Qiskit gates are natively supported by PyZX. The QASM interchange path handles standard gates well, but custom or composite gates may require decomposition before import. The qiskit-zx-transpiler package addresses some of these issues but is not yet feature-complete for all gate types.


# 10. Tool Ecosystem Summary

| **Stage** | **Primary Tool** | **Version / Status** | **Notes** |
| - | - | - | - |
| 1. Implementation | Qiskit SDK | Actively maintained | Use qiskit-algorithms (community) for algorithm templates |
| 2. ZX Conversion | PyZX | v0.8.0, active development | QASM interchange; also qiskit-zx-transpiler for direct pass |
| 3. Sub-Diagram Mining | Custom (gSpan, VF2, GNNs) | Research prototype required | NetworkX for graph ops; gspan-mining for frequent subgraph mining |
| 4. Composition | Custom engine | Research prototype required | Evolutionary / template-guided approach |
| 5. Extraction Filter | PyZX + tket ZX module | Mature | gFlow extraction in PyZX; Pauli flow in tket |
| 6. Benchmarking | QASMBench / SupermarQ | Open source | OpenQASM-level evaluation; cross-platform |
| 7. Reporting | PyZX visualization + custom | Mature | TikZ export for publications; matplotlib for plots |



# 11. Conclusion and Recommended Next Steps

The proposed seven-stage pipeline is technically feasible but requires significant original research in Stages 3 and 4. Stages 1, 2, 5, 6, and 7 are well-supported by existing open-source tooling and documented techniques. The pipeline’s most novel contribution—mining ZX-calculus sub-diagrams and composing them into new algorithms—has strong theoretical foundations in the ZX-calculus literature and recent empirical precedent in the gadget-discovery work, but has not been attempted at this level of generality.

Recommended next steps for implementation:

9. Build the Stage 1 corpus with 30–50 algorithm instances spanning all six categories, parametrized for scalability.

10. Implement the Stage 2 pipeline and store diagrams at three simplification levels (graph-like, Clifford-simplified, fully reduced).

11. Prototype Stage 3 using gSpan on the fully reduced diagrams, starting with a low support threshold (2) to discover the initial ZX-Web library.

12. Build a minimal Stage 4 composition engine using template-guided composition with flow-preservation constraints.

13. Integrate Stages 5–7 using PyZX extraction, QASMBench evaluation, and automated comparison against baselines.

14. Iterate: expand the corpus, refine the mining parameters, and explore evolutionary composition strategies.

The probability of discovering a genuinely novel algorithm that outperforms all existing solutions on a standard benchmark is low on any single run. However, the pipeline’s value lies not only in end-to-end algorithm discovery but also in the intermediate outputs: the ZX-Web library itself provides insight into the structural building blocks of quantum computing, the composition engine reveals which combinations of motifs produce valid circuits, and the benchmarking infrastructure provides a systematic framework for evaluating quantum algorithm variants.


# 12. Key References

*ZX-Calculus Foundations*

- Coecke, B. and Duncan, R. (2008). Interacting Quantum Observables: Categorical Algebra and Diagrammatics.

- van de Wetering, J. (2020). ZX-calculus for the working quantum computer scientist. arXiv:2012.13966.

- Kissinger, A. and van de Wetering, J. (2020). PyZX: Large Scale Automated Diagrammatic Reasoning. EPTCS 318.

*Circuit Extraction and Flow*

- Duncan, R. et al. (2020). Graph-theoretic Simplification of Quantum Circuits with the ZX-calculus. Quantum 4, 279.

- Backens, M. et al. (2021). There and back again: A circuit extraction tale. Quantum 5, 421.

- de Beaudrap, N. et al. (2022). Circuit Extraction for ZX-diagrams can be \#P-hard. arXiv:2202.09194.

- McElvanney, T. and Simmons, W. (2023). Flow-preserving ZX-calculus Rewrite Rules. arXiv:2304.08166.

- Mitosek, P. (2025). Computational complexity perspective on graphical calculi. PhD Thesis, University of Birmingham.

*Quantum Algorithm Discovery*

- Automated Discovery of Gadgets in Quantum Circuits (2025). arXiv:2509.24666.

- AltGraph: Redesigning Quantum Circuits Using Generative Graph Models (2024). GLSVLSI.

*Benchmarking*

- Li, A. et al. (2023). QASMBench: A Low-Level Quantum Benchmark Suite. ACM Trans. Quantum Comput. 4(2).

- Tomesh, T. et al. (2022). SupermarQ: A Scalable Quantum Benchmark Suite. arXiv:2202.11045.

*Tools and Frameworks*

- Qiskit SDK: https://github.com/Qiskit/qiskit

- PyZX: https://github.com/zxcalc/pyzx

- qiskit-zx-transpiler: https://pypi.org/project/qiskit-zx-transpiler/

- tket ZX module: https://docs.quantinuum.com/tket/user-guide/manual/manual\_zx.html

- Fischbach et al. (2025). Exhaustive Search for Quantum Circuit Optimization using ZX Calculus. OLA 2025.
