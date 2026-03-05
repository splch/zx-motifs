#!/usr/bin/env python3
"""
Quantum Algorithm Discovery via ZX Motif Phylogeny (v2)
=======================================================

Data-driven pipeline that uses structural insights from motif phylogeny
analysis to construct novel quantum circuits. Four discovery strategies:

  1. Coverage Gap Targeting   — fills under-represented families
  2. Cross-Family Bridge      — hybridises high-similarity cross-family pairs
  3. Irreducible Composition  — composes motifs surviving full ZX reduction
  4. PCA Void Filling         — targets empty regions in fingerprint PCA space

Outputs (scripts/output/discovery/):
  - pca_candidates.png           PCA scatter with new algorithms overlaid
  - dendrogram_candidates.png    Dendrogram with candidates highlighted
  - candidate_profiles.png       Top-motif bar charts per candidate
  - strategy_comparison.png      Grouped bar comparison across strategies
  - candidate_fingerprints.csv   Fingerprint vectors
  - results.json                 Machine-readable results
  - report.txt                   Human-readable discovery report
"""

from __future__ import annotations

import copy
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
import pandas as pd
import pyzx as zx
from matplotlib.patches import Patch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# ── Project imports ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zx_motifs.algorithms.registry import ALGORITHM_FAMILY_MAP, REGISTRY
from zx_motifs.pipeline.converter import (
    SimplificationLevel,
    convert_at_all_levels,
    qiskit_to_zx,
)
from zx_motifs.pipeline.featurizer import pyzx_to_networkx
from zx_motifs.pipeline.matcher import MotifPattern, find_motif_in_graph
from zx_motifs.pipeline.motif_generators import (
    EXTENDED_MOTIFS,
    _is_isomorphic,
    find_neighborhood_motifs,
    find_recurring_subgraphs,
    wl_hash,
)

warnings.filterwarnings("ignore")

PHYLOGENY_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR = PHYLOGENY_DIR / "discovery"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FAMILY_COLOURS = {
    "oracle": "#e41a1c",
    "entanglement": "#377eb8",
    "error_correction": "#4daf4a",
    "distillation": "#984ea3",
    "protocol": "#ff7f00",
    "variational": "#a65628",
    "simulation": "#f781bf",
    "transform": "#999999",
    "arithmetic": "#66c2a5",
    "machine_learning": "#e6ab02",
    "linear_algebra": "#1b9e77",
    "cryptography": "#d95f02",
    "sampling": "#7570b3",
    "error_mitigation": "#e7298a",
    "topological": "#66a61e",
    "metrology": "#e6ab02",
    "differential_equations": "#a6761d",
    "tda": "#666666",
    "communication": "#1f78b4",
}

STRATEGY_COLOURS = {
    "coverage_gap": "#e41a1c",
    "cross_family": "#377eb8",
    "irreducible": "#4daf4a",
    "pca_void": "#984ea3",
    "legacy": "#999999",
}

STRATEGY_MARKERS = {
    "coverage_gap": "*",
    "cross_family": "D",
    "irreducible": "p",
    "pca_void": "^",
    "legacy": "s",
}


# ── Data types ────────────────────────────────────────────────────────


@dataclass
class CandidateSpec:
    """Specification for a candidate algorithm to be constructed."""

    name: str
    strategy: str
    motif_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    n_qubits: int = 4
    # For cross-family bridge
    source_algo_a: str | None = None
    source_algo_b: str | None = None
    shared_motifs: list[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────


def _get_family(instance_name: str) -> str:
    base = instance_name.rsplit("_q", 1)[0]
    return ALGORITHM_FAMILY_MAP.get(base, "unknown")


# ═══════════════════════════════════════════════════════════════════════
# Phase 0: Load Phylogeny Results + Rebuild Corpus/Motifs
# ═══════════════════════════════════════════════════════════════════════


def load_phylogeny_results() -> tuple[dict, pd.DataFrame]:
    """Load results.json and fingerprint_frequencies.csv from prior run."""
    results = json.loads((PHYLOGENY_DIR / "results.json").read_text())
    freq_df = pd.read_csv(PHYLOGENY_DIR / "fingerprint_frequencies.csv", index_col=0)
    return results, freq_df


def build_corpus():
    """Build the full algorithm corpus (same as discover_phylogeny.py)."""
    corpus = {}
    for entry in REGISTRY:
        lo, hi = entry.qubit_range
        for n in range(lo, min(hi, 5) + 1):
            if entry.name == "grover" and n >= 6:
                continue
            instance = f"{entry.name}_q{n}"
            try:
                qc = entry.generator(n)
                snapshots = convert_at_all_levels(qc, instance)
                for snap in snapshots:
                    nxg = pyzx_to_networkx(snap.graph, coarsen_phases=True)
                    corpus[(instance, snap.level.value)] = nxg
            except Exception:
                pass
    return corpus


def discover_motifs(corpus: dict) -> list[MotifPattern]:
    """Discover motifs from corpus (same as discover_phylogeny.py)."""
    all_motifs = list(EXTENDED_MOTIFS)
    seen_hashes = {}
    for i, mp in enumerate(all_motifs):
        seen_hashes[wl_hash(mp.graph)] = i

    def _add_novel(candidates):
        for mp in candidates:
            h = wl_hash(mp.graph)
            if h in seen_hashes:
                existing = all_motifs[seen_hashes[h]]
                if _is_isomorphic(mp.graph, existing.graph):
                    continue
            seen_hashes[h] = len(all_motifs)
            all_motifs.append(mp)

    try:
        bu = find_recurring_subgraphs(
            corpus, target_level="spider_fused", min_size=3, max_size=5
        )
        _add_novel(bu)
    except Exception:
        pass
    try:
        nb = find_neighborhood_motifs(corpus, target_level="spider_fused", radius=2)
        _add_novel(nb)
    except Exception:
        pass

    return all_motifs


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Motif Template Registry
# ═══════════════════════════════════════════════════════════════════════

# Maps motif_id -> function(qc, offset) that appends gates to qc.
# offset is the starting qubit index for this motif in the circuit.
# Returns the number of qubits consumed by this motif.

MOTIF_TEMPLATES: dict[str, tuple[int, object]] = {}  # populated by _register_templates


def _tpl_phase_gadget_2t(qc, off):
    qc.cx(off, off + 1)
    qc.t(off + 1)
    qc.cx(off, off + 1)
    return 2


def _tpl_phase_gadget_3t(qc, off):
    qc.cx(off, off + 1)
    qc.cx(off, off + 2)
    qc.t(off + 2)
    qc.cx(off, off + 2)
    qc.cx(off, off + 1)
    return 3


def _tpl_cx_pair(qc, off):
    qc.cx(off, off + 1)
    return 2


def _tpl_hadamard_sandwich(qc, off):
    qc.h(off)
    qc.s(off)
    qc.h(off)
    return 1


def _tpl_zz_interaction(qc, off):
    qc.cx(off, off + 1)
    qc.rz(0.7, off + 1)
    qc.cx(off, off + 1)
    return 2


def _tpl_zz_interaction_param(qc, off):
    qc.cx(off, off + 1)
    qc.rz(np.pi / 5, off + 1)
    qc.cx(off, off + 1)
    return 2


def _tpl_syndrome_extraction(qc, off):
    qc.cx(off, off + 1)
    qc.cx(off, off + 2)
    return 3


def _tpl_syndrome_extraction_param(qc, off):
    qc.cx(off, off + 1)
    qc.cx(off, off + 2)
    return 3


def _tpl_toffoli_core(qc, off):
    qc.t(off)
    qc.cx(off, off + 1)
    qc.t(off + 1)
    return 2


def _tpl_toffoli_core_param(qc, off):
    qc.t(off)
    qc.cx(off, off + 1)
    qc.rz(np.pi / 4, off + 1)
    return 2


def _tpl_cluster_chain(qc, off):
    qc.h(off)
    qc.cz(off, off + 1)
    qc.h(off + 1)
    return 2


def _tpl_trotter_layer(qc, off):
    qc.cx(off, off + 1)
    qc.rz(0.5, off + 1)
    qc.cx(off + 1, off + 2)
    qc.rz(0.3, off + 2)
    return 3


def _tpl_x_hub_3z_param(qc, off):
    qc.cx(off + 1, off)
    qc.cx(off + 2, off)
    qc.cx(off + 3, off)
    return 4


def _tpl_hadamard_pauli_pair(qc, off):
    qc.h(off)
    qc.x(off)
    return 1


def _tpl_pauli_x_hub_3z(qc, off):
    qc.x(off)
    qc.cx(off + 1, off)
    qc.cx(off + 2, off)
    qc.cx(off + 3, off)
    return 4


# Template info: (n_qubits_needed, apply_fn)
_HANDCRAFTED_TEMPLATES = {
    "phase_gadget_2t": (2, _tpl_phase_gadget_2t),
    "phase_gadget_3t": (3, _tpl_phase_gadget_3t),
    "cx_pair": (2, _tpl_cx_pair),
    "hadamard_sandwich": (1, _tpl_hadamard_sandwich),
    "zz_interaction": (2, _tpl_zz_interaction),
    "zz_interaction_param": (2, _tpl_zz_interaction_param),
    "syndrome_extraction": (3, _tpl_syndrome_extraction),
    "syndrome_extraction_param": (3, _tpl_syndrome_extraction_param),
    "toffoli_core": (2, _tpl_toffoli_core),
    "toffoli_core_param": (2, _tpl_toffoli_core_param),
    "cluster_chain": (2, _tpl_cluster_chain),
    "trotter_layer": (3, _tpl_trotter_layer),
    "x_hub_3z_param": (4, _tpl_x_hub_3z_param),
    "hadamard_pauli_pair": (1, _tpl_hadamard_pauli_pair),
    "pauli_x_hub_3z": (4, _tpl_pauli_x_hub_3z),
}


def _try_extract_auto_template(
    motif_id: str,
    motif: MotifPattern,
    freq_df: pd.DataFrame,
    corpus: dict,
) -> tuple[int, object] | None:
    """Try to auto-extract a gate template for an auto-discovered motif.

    Find the first algorithm containing this motif, match it, and extract
    the corresponding gate slice from the source QuantumCircuit.
    Returns (n_qubits, apply_fn) or None if extraction fails.
    """
    if motif_id not in freq_df.columns:
        return None

    # Find an algorithm that has this motif (non-zero frequency)
    col = freq_df[motif_id]
    algo_instances = col[col > 0].index.tolist()
    if not algo_instances:
        return None

    # Try each until we succeed
    for inst in algo_instances[:3]:
        key = (inst, "spider_fused")
        if key not in corpus:
            continue

        host = corpus[key]
        matches = find_motif_in_graph(motif.graph, host, max_matches=1)
        if not matches:
            continue

        mapping = matches[0]
        matched_host_nodes = set(mapping.values())
        n_matched = len(matched_host_nodes)

        # Use the motif size as qubit count (capped at 4)
        n_qubits = min(n_matched, 4)
        if n_qubits < 1:
            continue

        # Build a simple template that approximates the motif structure
        # by inspecting the motif graph's node/edge types
        node_types = []
        for node in motif.graph.nodes():
            vt = motif.graph.nodes[node].get("vertex_type", "Z")
            pc = motif.graph.nodes[node].get("phase_class", "zero")
            node_types.append((vt, pc))

        has_hadamard_edge = any(
            motif.graph.edges[e].get("edge_type") == "HADAMARD"
            for e in motif.graph.edges()
        )

        # Build a generic template from the structure
        def _make_auto_fn(nq, node_info, has_h):
            def _fn(qc, off):
                # Apply phase gates based on node types
                for i, (vt, pc) in enumerate(node_info[:nq]):
                    q = off + i
                    if pc == "t_like":
                        qc.t(q)
                    elif pc == "clifford":
                        qc.s(q)
                    elif pc == "pauli":
                        qc.z(q)
                    elif pc == "arbitrary":
                        qc.rz(0.3, q)
                # Entangle via CX/CZ chain
                for i in range(nq - 1):
                    if has_h:
                        qc.h(off + i)
                        qc.cz(off + i, off + i + 1)
                    else:
                        qc.cx(off + i, off + i + 1)
                return nq

            return _fn

        return (n_qubits, _make_auto_fn(n_qubits, node_types, has_hadamard_edge))

    return None


def build_template_registry(
    motifs: list[MotifPattern],
    freq_df: pd.DataFrame,
    corpus: dict,
    survivors: list[str],
) -> dict[str, tuple[int, object]]:
    """Build the motif_id -> (n_qubits, apply_fn) template registry.

    Starts with 15 handcrafted templates, then attempts auto-extraction
    for survivor motifs and other frequently occurring auto-discovered motifs.
    """
    templates = dict(_HANDCRAFTED_TEMPLATES)

    # Build motif lookup
    motif_by_id = {mp.motif_id: mp for mp in motifs}

    # Try auto-extraction for survivor motifs first
    for mid in survivors:
        if mid in templates:
            continue
        mp = motif_by_id.get(mid)
        if mp is None:
            continue
        result = _try_extract_auto_template(mid, mp, freq_df, corpus)
        if result is not None:
            templates[mid] = result

    # Also try for frequent auto-discovered motifs (top 20 by family count)
    motif_stats = []
    for mp in motifs:
        if mp.motif_id.startswith("auto_") and mp.motif_id not in templates:
            if mp.motif_id in freq_df.columns:
                n_nonzero = (freq_df[mp.motif_id] > 0).sum()
                motif_stats.append((mp.motif_id, n_nonzero))
    motif_stats.sort(key=lambda x: -x[1])

    for mid, _ in motif_stats[:20]:
        if mid in templates:
            continue
        mp = motif_by_id.get(mid)
        if mp is None:
            continue
        result = _try_extract_auto_template(mid, mp, freq_df, corpus)
        if result is not None:
            templates[mid] = result

    return templates


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Four Discovery Strategies
# ═══════════════════════════════════════════════════════════════════════


def strategy_coverage_gap(
    phylo_results: dict,
    freq_df: pd.DataFrame,
    templates: dict[str, tuple[int, object]],
) -> list[CandidateSpec]:
    """Strategy 1: Target families with coverage < 70%.

    For each low-coverage family, find common/universal motifs absent
    from that family, combine with native family motifs.
    """
    coverage = phylo_results["coverage"]["average_coverage_by_family"]
    motif_stats = phylo_results["universality"]["motif_stats"]

    # Build motif -> families and category lookup
    motif_families = {}
    motif_category = {}
    for ms in motif_stats:
        motif_families[ms["motif_id"]] = set(ms["families"])
        motif_category[ms["motif_id"]] = ms["category"]

    candidates = []
    gap_families = {
        fam: cov for fam, cov in coverage.items() if cov < 0.70
    }

    for fam, cov in sorted(gap_families.items(), key=lambda x: x[1]):
        # Find motifs native to this family
        native_motifs = [
            mid
            for mid, fams in motif_families.items()
            if fam in fams and mid in templates
        ]
        # Find common/universal motifs absent from this family
        missing_motifs = [
            mid
            for mid, fams in motif_families.items()
            if fam not in fams
            and motif_category.get(mid) in ("common", "universal")
            and mid in templates
        ]

        if not missing_motifs:
            continue

        # Pick 2-3 native + 2-3 missing
        selected_native = native_motifs[:3]
        selected_missing = missing_motifs[:3]
        selected = selected_native + selected_missing

        if len(selected) < 2:
            continue

        name = f"covgap_{fam}"
        candidates.append(
            CandidateSpec(
                name=name,
                strategy="coverage_gap",
                motif_ids=selected,
                rationale=(
                    f"Coverage gap: {fam} at {cov:.1%}. "
                    f"Native motifs: {selected_native}, "
                    f"missing common: {selected_missing}"
                ),
                n_qubits=min(6, max(4, max(templates[m][0] for m in selected) + 1)),
            )
        )

    return candidates


def strategy_cross_family_bridge(
    phylo_results: dict,
    templates: dict[str, tuple[int, object]],
) -> list[CandidateSpec]:
    """Strategy 2: Build hybrid circuits from cross-family surprise pairs.

    Group surprises by unique (family_a, family_b) pair, pick the
    highest-similarity representative from each pair.
    """
    surprises = phylo_results["surprises"]

    # Group by family pair (sorted to deduplicate)
    pair_groups: dict[tuple[str, str], list] = {}
    for s in surprises:
        pair = tuple(sorted([s["family_a"], s["family_b"]]))
        pair_groups.setdefault(pair, []).append(s)

    candidates = []
    for pair, entries in sorted(pair_groups.items()):
        if len(entries) < 2:
            continue

        # Pick the highest-similarity entry
        best = max(entries, key=lambda e: e["cosine_similarity"])
        sim = best["cosine_similarity"]
        algo_a = best["algo_a"]
        algo_b = best["algo_b"]
        fam_a = best["family_a"]
        fam_b = best["family_b"]

        # Find shared motifs (motifs present in both families) that have templates
        motif_stats = phylo_results["universality"]["motif_stats"]
        shared = []
        for ms in motif_stats:
            fams = set(ms["families"])
            if fam_a in fams and fam_b in fams and ms["motif_id"] in templates:
                shared.append(ms["motif_id"])

        if not shared:
            continue

        name = f"bridge_{pair[0][:4]}_{pair[1][:4]}"
        candidates.append(
            CandidateSpec(
                name=name,
                strategy="cross_family",
                motif_ids=shared[:4],
                rationale=(
                    f"Cross-family bridge: {fam_a} <-> {fam_b} (sim={sim:.3f}). "
                    f"Representatives: {algo_a}, {algo_b}"
                ),
                n_qubits=4,
                source_algo_a=algo_a,
                source_algo_b=algo_b,
                shared_motifs=shared[:3],
            )
        )

    return candidates


def strategy_irreducible_composition(
    phylo_results: dict,
    templates: dict[str, tuple[int, object]],
) -> list[CandidateSpec]:
    """Strategy 3: Compose subsets of motifs surviving full ZX reduction.

    - Pairwise sequential combos
    - Full-stack: chain all survivors
    - Layered: 2-3 survivors in parallel, repeated
    """
    survivors = phylo_results["cross_level_survival"]["survivors_at_full_reduce"]
    # Filter to those with templates
    available = [s for s in survivors if s in templates]

    if not available:
        return []

    candidates = []

    # Pairwise combos (filtered to compatible qubit counts, cap at 6 qubits)
    seen_pairs = set()
    pair_counter = 0
    for i, m1 in enumerate(available):
        for m2 in available[i + 1 :]:
            total_q = templates[m1][0] + templates[m2][0]
            if total_q > 6:
                continue
            pair_key = tuple(sorted([m1, m2]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            pair_counter += 1
            name = f"irr_pair{pair_counter:02d}"
            candidates.append(
                CandidateSpec(
                    name=name,
                    strategy="irreducible",
                    motif_ids=[m1, m2],
                    rationale=(
                        f"Irreducible pair: {m1} + {m2} "
                        f"(both survive full ZX reduction)"
                    ),
                    n_qubits=min(6, total_q + 1),
                )
            )

    # Full-stack: chain all available survivors
    total_q_all = sum(templates[m][0] for m in available[:6])
    if total_q_all <= 6 and len(available) >= 3:
        candidates.append(
            CandidateSpec(
                name="irr_full_stack",
                strategy="irreducible",
                motif_ids=available[:6],
                rationale=(
                    f"Full irreducible stack: all {len(available[:6])} survivors chained"
                ),
                n_qubits=min(6, max(4, max(templates[m][0] for m in available[:6]) + 1)),
            )
        )

    # Layered: 2-3 survivors in parallel, 2 layers
    parallel_group = [m for m in available if templates[m][0] <= 2][:3]
    if len(parallel_group) >= 2:
        candidates.append(
            CandidateSpec(
                name="irr_layered",
                strategy="irreducible",
                motif_ids=parallel_group * 2,  # repeat for 2 layers
                rationale=(
                    f"Layered irreducible: {parallel_group} in parallel, 2 layers"
                ),
                n_qubits=min(6, sum(templates[m][0] for m in parallel_group)),
            )
        )

    return candidates


def strategy_pca_void(
    freq_df: pd.DataFrame,
    templates: dict[str, tuple[int, object]],
) -> list[CandidateSpec]:
    """Strategy 4: Fill empty regions in PCA space.

    PCA the frequency matrix, find void cells on a grid, invert PCA
    to get target motif vectors, select top motifs as templates.
    """
    mat = freq_df.fillna(0).values
    mean = mat.mean(axis=0)
    mat_c = mat - mean

    U, S, Vt = np.linalg.svd(mat_c, full_matrices=False)
    coords = U[:, :2] * S[:2]  # (n_instances, 2)

    # Build 20x20 grid over PCA extent
    margin = 0.1
    x_min, x_max = coords[:, 0].min() - margin, coords[:, 0].max() + margin
    y_min, y_max = coords[:, 1].min() - margin, coords[:, 1].max() + margin

    grid_x = np.linspace(x_min, x_max, 20)
    grid_y = np.linspace(y_min, y_max, 20)

    # For each grid cell, compute min distance to any existing point
    best_voids = []
    for gx in grid_x:
        for gy in grid_y:
            dists = np.sqrt((coords[:, 0] - gx) ** 2 + (coords[:, 1] - gy) ** 2)
            min_dist = dists.min()
            best_voids.append((min_dist, gx, gy))

    best_voids.sort(key=lambda x: -x[0])

    candidates = []
    for rank, (dist, gx, gy) in enumerate(best_voids[:6]):
        # Invert PCA: target = target_pca @ Vt[:2] + mean
        target_pca = np.array([gx, gy])
        target = target_pca @ Vt[:2] + mean
        target = np.maximum(target, 0)  # clamp negatives

        # Select top motifs from target vector that have templates
        motif_ids = list(freq_df.columns)
        motif_scores = [(motif_ids[j], target[j]) for j in range(len(motif_ids))]
        motif_scores.sort(key=lambda x: -x[1])

        selected = [
            mid for mid, score in motif_scores if score > 0 and mid in templates
        ][:5]

        if len(selected) < 3:
            continue

        name = f"void_{rank}"
        candidates.append(
            CandidateSpec(
                name=name,
                strategy="pca_void",
                motif_ids=selected,
                rationale=(
                    f"PCA void at ({gx:.2f}, {gy:.2f}), "
                    f"isolation={dist:.3f}. Target motifs: {selected}"
                ),
                n_qubits=min(6, max(4, max(templates[m][0] for m in selected) + 1)),
            )
        )

    return candidates[:4]  # Cap at 4 candidates


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Build QuantumCircuits from Specs
# ═══════════════════════════════════════════════════════════════════════


def build_circuit_from_spec(
    spec: CandidateSpec,
    templates: dict[str, tuple[int, object]],
) -> QuantumCircuit | None:
    """Build a QuantumCircuit from a CandidateSpec.

    Two modes:
    - Template composition (strategies 1, 3, 4): sequentially append motif
      gate templates, shifting qubit allocation by 1 each motif (round-robin)
    - Algorithm splicing (strategy 2): load two source algorithms, take gate
      slices, interleave with shared motif bridging layer
    """
    n_qubits = spec.n_qubits

    if spec.strategy == "cross_family" and spec.source_algo_a and spec.source_algo_b:
        return _build_splice_circuit(spec, templates)

    return _build_template_circuit(spec, templates)


def _build_template_circuit(
    spec: CandidateSpec,
    templates: dict[str, tuple[int, object]],
) -> QuantumCircuit | None:
    """Build circuit by sequentially applying motif templates."""
    n_qubits = spec.n_qubits
    qc = QuantumCircuit(n_qubits)

    offset = 0
    applied = 0
    for mid in spec.motif_ids:
        if mid not in templates:
            continue
        tpl_qubits, apply_fn = templates[mid]

        # Ensure we don't exceed circuit size
        if offset + tpl_qubits > n_qubits:
            offset = 0  # wrap around

        try:
            apply_fn(qc, offset)
            applied += 1
        except Exception:
            continue

        # Shift by 1 for inter-motif entanglement (round-robin)
        offset = (offset + 1) % max(1, n_qubits - tpl_qubits + 1)

    if applied == 0:
        return None

    return qc


def _build_splice_circuit(
    spec: CandidateSpec,
    templates: dict[str, tuple[int, object]],
) -> QuantumCircuit | None:
    """Build hybrid circuit by splicing two source algorithms with shared motifs."""
    # Find source algorithm generators
    reg_map = {entry.name: entry for entry in REGISTRY}

    base_a = spec.source_algo_a.rsplit("_q", 1)[0] if spec.source_algo_a else None
    base_b = spec.source_algo_b.rsplit("_q", 1)[0] if spec.source_algo_b else None

    entry_a = reg_map.get(base_a)
    entry_b = reg_map.get(base_b)

    if not entry_a or not entry_b:
        # Fall back to template composition
        return _build_template_circuit(spec, templates)

    n_qubits = spec.n_qubits
    try:
        qc_a = entry_a.generator(n_qubits)
        qc_b = entry_b.generator(n_qubits)
    except Exception:
        return _build_template_circuit(spec, templates)

    # Build hybrid: first half of A + shared motif bridge + second half of B
    qc = QuantumCircuit(n_qubits)

    # Extract gates from source circuits
    gates_a = list(qc_a.data)
    gates_b = list(qc_b.data)

    # First half of A
    half_a = len(gates_a) // 2
    for instruction in gates_a[:half_a]:
        try:
            qc.append(instruction)
        except Exception:
            pass

    # Shared motif bridging layer
    offset = 0
    for mid in spec.shared_motifs:
        if mid not in templates:
            continue
        tpl_qubits, apply_fn = templates[mid]
        if offset + tpl_qubits > n_qubits:
            offset = 0
        try:
            apply_fn(qc, offset)
        except Exception:
            pass
        offset = (offset + 1) % max(1, n_qubits - tpl_qubits + 1)

    # Second half of B
    half_b = len(gates_b) // 2
    for instruction in gates_b[half_b:]:
        try:
            qc.append(instruction)
        except Exception:
            pass

    if qc.size() == 0:
        return _build_template_circuit(spec, templates)

    return qc


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Validate Candidates
# ═══════════════════════════════════════════════════════════════════════


def validate_candidate(name: str, qc: QuantumCircuit) -> dict:
    """Validate unitarity and ZX tensor preservation."""
    op = Operator(qc)
    mat = op.data
    n = mat.shape[0]
    product = mat @ mat.conj().T
    identity_err = float(np.linalg.norm(product - np.eye(n)) / n)
    is_unitary = identity_err < 1e-6

    zx_circ = qiskit_to_zx(qc)
    g = zx_circ.to_graph()
    n_inputs = len(list(g.inputs()))
    n_outputs = len(list(g.outputs()))
    has_valid_io = (n_inputs == qc.num_qubits) and (n_outputs == qc.num_qubits)

    g_copy = copy.deepcopy(g)
    zx.simplify.full_reduce(g_copy)
    try:
        tensors_match = zx.compare_tensors(g, g_copy)
    except Exception:
        tensors_match = None

    return {
        "name": name,
        "n_qubits": qc.num_qubits,
        "n_gates": qc.size(),
        "depth": qc.depth(),
        "is_unitary": is_unitary,
        "unitarity_error": identity_err,
        "has_valid_io": has_valid_io,
        "zx_vertices_raw": g.num_vertices(),
        "zx_edges_raw": g.num_edges(),
        "zx_vertices_reduced": g_copy.num_vertices(),
        "zx_edges_reduced": g_copy.num_edges(),
        "tensors_match": tensors_match,
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Fingerprint Candidates
# ═══════════════════════════════════════════════════════════════════════


def fingerprint_candidates(
    candidates: list[tuple[str, QuantumCircuit]],
    corpus: dict,
    motifs: list[MotifPattern],
    existing_columns: list[str],
) -> pd.DataFrame:
    """Fingerprint candidate circuits against the motif library."""
    for name, qc in candidates:
        instance = f"{name}_q{qc.num_qubits}"
        try:
            snapshots = convert_at_all_levels(qc, instance)
            for snap in snapshots:
                nxg = pyzx_to_networkx(snap.graph, coarsen_phases=True)
                corpus[(instance, snap.level.value)] = nxg
        except Exception as e:
            print(f"    Warning: failed to convert {instance}: {e}")

    instances = [f"{name}_q{qc.num_qubits}" for name, qc in candidates]

    counts = np.zeros((len(instances), len(motifs)), dtype=int)
    for i, inst in enumerate(instances):
        key = (inst, "spider_fused")
        if key not in corpus:
            continue
        host = corpus[key]
        for j, mp in enumerate(motifs):
            matches = find_motif_in_graph(mp.graph, host, max_matches=100)
            counts[i, j] = len(matches)

    motif_ids = [mp.motif_id for mp in motifs]
    counts_df = pd.DataFrame(counts, index=instances, columns=motif_ids)
    row_sums = counts_df.sum(axis=1).replace(0, 1)
    freq_df = counts_df.div(row_sums, axis=0)

    freq_aligned = freq_df.reindex(columns=existing_columns, fill_value=0.0)
    return freq_aligned


# ═══════════════════════════════════════════════════════════════════════
# Phase 6: Scoring
# ═══════════════════════════════════════════════════════════════════════


def compute_pca_info(freq_df: pd.DataFrame) -> dict:
    """Compute PCA from existing fingerprint matrix."""
    mat = freq_df.fillna(0).values
    mean = mat.mean(axis=0)
    mat_c = mat - mean

    U, S, Vt = np.linalg.svd(mat_c, full_matrices=False)
    coords = U[:, :2] * S[:2]

    instances = list(freq_df.index)
    families = [_get_family(inst) for inst in instances]

    centroids = {}
    for fam in sorted(set(families)):
        idx = [i for i, f in enumerate(families) if f == fam]
        centroids[fam] = coords[idx].mean(axis=0)

    return {
        "mean": mean,
        "S": S,
        "Vt": Vt,
        "coords": coords,
        "instances": instances,
        "families": families,
        "centroids": centroids,
    }


def score_candidates(
    existing_freq_df: pd.DataFrame,
    candidate_freq_df: pd.DataFrame,
    pca_info: dict,
    validations: dict,
    candidate_strategies: dict[str, str],
) -> dict:
    """Score each candidate with composite novelty metric.

    composite = 0.35 * cosine_novelty      # 1 - max_sim to any existing
              + 0.25 * pca_isolation        # normalized min-distance in PCA
              + 0.20 * motif_diversity      # fraction of non-zero motifs
              + 0.20 * tensor_bonus         # 1.0 if tensors match, 0.5 otherwise
    """
    existing_mat = existing_freq_df.fillna(0).values
    cand_mat = candidate_freq_df.fillna(0).values
    instances = list(existing_freq_df.index)
    cand_names = list(candidate_freq_df.index)

    mean = pca_info["mean"]
    Vt = pca_info["Vt"]
    coords = pca_info["coords"]
    centroids = pca_info["centroids"]

    results = {}
    all_pca_dists = []

    for ci, cname in enumerate(cand_names):
        cv = cand_mat[ci]
        cv_norm = np.linalg.norm(cv)

        # Cosine similarity to every existing algorithm
        sims = []
        for ei in range(len(instances)):
            ev = existing_mat[ei]
            ev_norm = np.linalg.norm(ev)
            if cv_norm < 1e-10 or ev_norm < 1e-10:
                sims.append(0.0)
            else:
                sims.append(float(np.dot(cv, ev) / (cv_norm * ev_norm)))

        best_idx = int(np.argmax(sims))
        nearest = instances[best_idx]
        nearest_sim = sims[best_idx]
        cosine_novelty = 1.0 - nearest_sim

        # PCA coords and isolation
        cand_centered = cv - mean
        cand_pca = cand_centered @ Vt[:2].T
        pca_dists = np.sqrt(((coords - cand_pca) ** 2).sum(axis=1))
        pca_min_dist = float(pca_dists.min())
        all_pca_dists.append(pca_min_dist)

        # Family affinity
        family_dists = {}
        for fam, centroid in centroids.items():
            family_dists[fam] = float(np.linalg.norm(cand_pca - centroid))
        closest_family = min(family_dists, key=family_dists.get)

        # Motif diversity: fraction of non-zero motifs
        motif_diversity = float((cv > 0).sum()) / max(len(cv), 1)

        # Tensor bonus
        base_name = cname.rsplit("_q", 1)[0]
        val = validations.get(base_name, {})
        tm = val.get("tensors_match")
        tensor_bonus = 1.0 if tm else 0.5

        # Top motifs
        top_idx = np.argsort(cv)[::-1][:10]
        top_motifs = [
            (candidate_freq_df.columns[ti], float(cv[ti]))
            for ti in top_idx
            if cv[ti] > 0
        ]

        results[cname] = {
            "nearest_algorithm": nearest,
            "nearest_family": _get_family(nearest),
            "nearest_similarity": round(nearest_sim, 4),
            "closest_family": closest_family,
            "family_distances": {k: round(v, 4) for k, v in family_dists.items()},
            "cosine_novelty": round(cosine_novelty, 4),
            "pca_min_dist": round(pca_min_dist, 4),
            "motif_diversity": round(motif_diversity, 4),
            "tensor_bonus": tensor_bonus,
            "pca_coords": [float(cand_pca[0]), float(cand_pca[1])],
            "top_motifs": top_motifs[:10],
            "strategy": candidate_strategies.get(base_name, "unknown"),
        }

    # Normalize PCA isolation across candidates
    max_pca_dist = max(all_pca_dists) if all_pca_dists else 1.0
    if max_pca_dist < 1e-10:
        max_pca_dist = 1.0

    for ci, cname in enumerate(cand_names):
        r = results[cname]
        pca_isolation = all_pca_dists[ci] / max_pca_dist

        composite = (
            0.35 * r["cosine_novelty"]
            + 0.25 * pca_isolation
            + 0.20 * r["motif_diversity"]
            + 0.20 * r["tensor_bonus"]
        )
        r["pca_isolation"] = round(pca_isolation, 4)
        r["composite_novelty"] = round(composite, 4)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 7: Visualize
# ═══════════════════════════════════════════════════════════════════════


def plot_pca_with_candidates(
    existing_freq_df: pd.DataFrame,
    candidate_freq_df: pd.DataFrame,
    pca_info: dict,
    scored: dict,
) -> None:
    """PCA scatter with candidates overlaid, marker per strategy."""
    coords = pca_info["coords"]
    instances = pca_info["instances"]
    families = pca_info["families"]
    S = pca_info["S"]

    var_explained = (S[:2] ** 2) / max((S**2).sum(), 1e-10)

    fig, ax = plt.subplots(figsize=(12, 9))

    # Existing algorithms
    for fam in sorted(set(families)):
        idx = [i for i, f in enumerate(families) if f == fam]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            c=FAMILY_COLOURS.get(fam, "#333"),
            label=fam,
            alpha=0.35,
            s=25,
            zorder=2,
        )

    # Candidate algorithms
    for cname, info in scored.items():
        strat = info.get("strategy", "unknown")
        marker = STRATEGY_MARKERS.get(strat, "o")
        color = STRATEGY_COLOURS.get(strat, "#000")
        pca_c = info["pca_coords"]
        label_short = cname.rsplit("_q", 1)[0][:15]
        ax.scatter(
            pca_c[0],
            pca_c[1],
            marker=marker,
            s=250,
            c=color,
            edgecolors="black",
            linewidths=1,
            zorder=5,
            label=f"{strat}: {label_short}",
        )
        ax.annotate(
            label_short,
            (pca_c[0], pca_c[1]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=7,
            fontweight="bold",
            color=color,
        )

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title("Algorithm Phylogeny PCA — Data-Driven Candidate Placement")

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_h.append(h)
            unique_l.append(l)
    ax.legend(unique_h, unique_l, fontsize=6, ncol=2, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_candidates.png", dpi=150)
    plt.close(fig)


def plot_dendrogram_with_candidates(
    existing_freq_df: pd.DataFrame,
    candidate_freq_df: pd.DataFrame,
    scored: dict,
) -> None:
    """Dendrogram with candidates highlighted by strategy colour."""
    combined = pd.concat([existing_freq_df, candidate_freq_df]).fillna(0)
    mat = combined.values
    row_norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = np.where(row_norms == 0, 1e-10, mat)

    dist = pdist(mat, metric="cosine")
    dist = np.nan_to_num(dist, nan=1.0)
    Z = linkage(dist, method="average")

    labels = list(combined.index)
    cand_set = set(candidate_freq_df.index)

    # Build cname -> strategy colour
    cand_colours = {}
    for cname, info in scored.items():
        strat = info.get("strategy", "unknown")
        cand_colours[cname] = STRATEGY_COLOURS.get(strat, "#000")

    fig, ax = plt.subplots(figsize=(16, max(10, len(labels) * 0.22)))
    dendrogram(Z, labels=labels, orientation="left", leaf_font_size=5, ax=ax)

    for lbl in ax.get_yticklabels():
        inst = lbl.get_text()
        if inst in cand_set:
            lbl.set_color(cand_colours.get(inst, "#000"))
            lbl.set_fontweight("bold")
            lbl.set_fontsize(7)
        else:
            fam = _get_family(inst)
            lbl.set_color(FAMILY_COLOURS.get(fam, "#333"))

    ax.set_title("Algorithm Phylogeny — Data-Driven Discovery Candidates", fontsize=14)
    ax.set_xlabel("Cosine Distance")

    handles = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLOURS.items()]
    for strat, color in STRATEGY_COLOURS.items():
        handles.append(Patch(facecolor=color, label=f"Strategy: {strat}"))
    ax.legend(handles=handles, loc="upper right", fontsize=5, ncol=2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dendrogram_candidates.png", dpi=150)
    plt.close(fig)


def plot_candidate_profiles(
    candidate_freq_df: pd.DataFrame,
    scored: dict,
) -> None:
    """Grid of top-motif bar charts, adapts to N candidates."""
    cand_names = list(candidate_freq_df.index)
    n = len(cand_names)
    if n == 0:
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, cname in enumerate(cand_names):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        info = scored.get(cname, {})
        top = info.get("top_motifs", [])[:8]

        if not top:
            ax.set_title(cname, fontsize=7)
            continue

        motif_ids = [t[0] for t in top]
        freqs = [t[1] for t in top]
        short_ids = [m[:18] + ".." if len(m) > 18 else m for m in motif_ids]

        strat = info.get("strategy", "unknown")
        color = STRATEGY_COLOURS.get(strat, "#1f77b4")
        ax.barh(range(len(freqs)), freqs, color=color, alpha=0.8)
        ax.set_yticks(range(len(short_ids)))
        ax.set_yticklabels(short_ids, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency", fontsize=7)

        label_short = cname.rsplit("_q", 1)[0][:20]
        comp = info.get("composite_novelty", 0)
        ax.set_title(f"{label_short}\nnovelty={comp:.3f}", fontsize=8)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Candidate Motif Profiles", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "candidate_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_comparison(scored: dict) -> None:
    """Grouped bars: candidates per strategy, avg novelty, pass rate."""
    strategy_stats = {}
    for cname, info in scored.items():
        strat = info.get("strategy", "unknown")
        if strat not in strategy_stats:
            strategy_stats[strat] = {"count": 0, "novelties": [], "passed": 0, "total": 0}
        strategy_stats[strat]["count"] += 1
        strategy_stats[strat]["novelties"].append(info.get("composite_novelty", 0))
        strategy_stats[strat]["total"] += 1
        if info.get("tensor_bonus", 0) == 1.0:
            strategy_stats[strat]["passed"] += 1

    if not strategy_stats:
        return

    strats = sorted(strategy_stats.keys())
    counts = [strategy_stats[s]["count"] for s in strats]
    avg_novelties = [
        np.mean(strategy_stats[s]["novelties"]) if strategy_stats[s]["novelties"] else 0
        for s in strats
    ]
    pass_rates = [
        strategy_stats[s]["passed"] / max(strategy_stats[s]["total"], 1)
        for s in strats
    ]

    x = np.arange(len(strats))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, counts, width, label="# Candidates", color="#377eb8", alpha=0.8)
    ax.bar(x, [n * 10 for n in avg_novelties], width, label="Avg Novelty (×10)", color="#4daf4a", alpha=0.8)
    ax.bar(x + width, [r * 10 for r in pass_rates], width, label="Pass Rate (×10)", color="#e41a1c", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(strats, fontsize=9)
    ax.set_ylabel("Value")
    ax.set_title("Strategy Comparison")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "strategy_comparison.png", dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Legacy Comparison (old TVH/DEB/ICC)
# ═══════════════════════════════════════════════════════════════════════


def make_trotter_variational_hybrid(
    n_qubits: int = 4,
    n_layers: int = 2,
    gamma: float = 0.7,
    beta: float = 0.4,
    **kwargs,
) -> QuantumCircuit:
    """Legacy TVH candidate."""
    n_qubits = max(3, n_qubits)
    qc = QuantumCircuit(n_qubits)
    for _layer in range(n_layers):
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(gamma, i + 1)
            qc.cx(i, i + 1)
        qc.h(range(n_qubits))
        for i in range(n_qubits - 1):
            qc.cz(i, i + 1)
        qc.h(range(n_qubits))
        for i in range(0, n_qubits - 1, 2):
            qc.h(i)
            qc.rz(beta, i)
            qc.h(i)
        for i in range(n_qubits):
            qc.rx(2 * beta, i)
    return qc


def make_distillation_entanglement_bridge(
    n_qubits: int = 4, theta: float = np.pi / 4, **kwargs
) -> QuantumCircuit:
    """Legacy DEB candidate."""
    n_qubits = max(4, n_qubits)
    if n_qubits % 2 != 0:
        n_qubits += 1
    half = n_qubits // 2
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(half - 1):
        qc.cx(i, i + 1)
    for i in range(n_qubits):
        qc.ry(theta, i)
    for i in range(half):
        qc.cx(i, half + i)
    for i in range(half):
        qc.h(half + i)
        qc.cx(half + i, i)
    return qc


def make_irreducible_core_circuit(n_qubits: int = 4, **kwargs) -> QuantumCircuit:
    """Legacy ICC candidate."""
    n_qubits = max(4, n_qubits)
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)
    for i in range(0, n_qubits - 1, 2):
        qc.cx(i, i + 1)
        qc.t(i + 1)
        qc.cx(i, i + 1)
    for i in range(n_qubits):
        qc.h(i)
        qc.s(i)
        qc.h(i)
    if n_qubits >= 4:
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.t(3)
        qc.cx(0, 3)
        qc.cx(0, 2)
        qc.cx(0, 1)
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)
    qc.h(range(n_qubits))
    return qc


# ═══════════════════════════════════════════════════════════════════════
# Phase 8: Report
# ═══════════════════════════════════════════════════════════════════════


def generate_report(
    validations: dict,
    scored: dict,
    specs: dict[str, CandidateSpec],
    legacy_scored: dict | None = None,
) -> str:
    """Generate structured discovery report grouped by strategy."""
    lines = [
        "=" * 70,
        "QUANTUM ALGORITHM DISCOVERY VIA ZX MOTIF PHYLOGENY (v2)",
        "=" * 70,
        "",
        "Data-driven: 4 strategies, {} candidates ({} passed validation)".format(
            len(scored),
            sum(1 for v in validations.values() if v.get("is_unitary")),
        ),
        "",
    ]

    # Group by strategy
    by_strategy: dict[str, list[str]] = {}
    for cname, info in scored.items():
        strat = info.get("strategy", "unknown")
        by_strategy.setdefault(strat, []).append(cname)

    strategy_names = {
        "coverage_gap": "Coverage Gap Targeting",
        "cross_family": "Cross-Family Bridge",
        "irreducible": "Irreducible Composition",
        "pca_void": "PCA Void Filling",
    }

    for strat in ["coverage_gap", "cross_family", "irreducible", "pca_void"]:
        cnames = by_strategy.get(strat, [])
        title = strategy_names.get(strat, strat)
        lines.append(f"--- Strategy: {title} ({len(cnames)} candidates) ---")

        for cname in cnames:
            info = scored[cname]
            base = cname.rsplit("_q", 1)[0]
            val = validations.get(base, {})
            spec = specs.get(base)

            status = "PASS" if val.get("is_unitary") else "FAIL"
            tensor_status = (
                "YES"
                if val.get("tensors_match")
                else ("N/A" if val.get("tensors_match") is None else "NO")
            )

            lines.append(f"  [{base}]")
            if spec:
                lines.append(f"    Rationale: {spec.rationale[:100]}")
            lines.append(
                f"    Qubits: {val.get('n_qubits', '?')}  "
                f"Gates: {val.get('n_gates', '?')}  "
                f"Depth: {val.get('depth', '?')}"
            )
            lines.append(f"    Unitarity: {status} (err={val.get('unitarity_error', 0):.2e})")
            lines.append(f"    ZX tensor preserved: {tensor_status}")
            lines.append(
                f"    ZX size: {val.get('zx_vertices_raw', '?')}V/"
                f"{val.get('zx_edges_raw', '?')}E raw -> "
                f"{val.get('zx_vertices_reduced', '?')}V/"
                f"{val.get('zx_edges_reduced', '?')}E reduced"
            )
            lines.append(
                f"    Nearest: {info.get('nearest_algorithm', '?')} "
                f"(sim={info.get('nearest_similarity', 0):.3f})"
            )
            lines.append(
                f"    Composite novelty: {info.get('composite_novelty', 0):.3f}"
            )

            top_motifs = info.get("top_motifs", [])[:5]
            if top_motifs:
                lines.append("    Top motifs:")
                for mid, freq in top_motifs:
                    lines.append(f"      {mid:40s} {freq:.3f}")
            lines.append("")

        lines.append("")

    # Top 10 by composite novelty
    ranked = sorted(scored.items(), key=lambda x: -x[1].get("composite_novelty", 0))
    lines.append("--- Top 10 by Composite Novelty ---")
    for rank, (cname, info) in enumerate(ranked[:10], 1):
        base = cname.rsplit("_q", 1)[0]
        lines.append(
            f"  {rank:2d}. {base:35s} "
            f"novelty={info.get('composite_novelty', 0):.3f}  "
            f"strategy={info.get('strategy', '?')}"
        )
    lines.append("")

    # Legacy comparison
    if legacy_scored:
        lines.append("--- Legacy Comparison (old TVH/DEB/ICC) ---")
        # Rank legacy among new candidates
        all_scored = list(scored.items()) + list(legacy_scored.items())
        all_ranked = sorted(all_scored, key=lambda x: -x[1].get("composite_novelty", 0))
        for rank, (cname, info) in enumerate(all_ranked, 1):
            base = cname.rsplit("_q", 1)[0]
            is_legacy = cname in legacy_scored
            tag = " [LEGACY]" if is_legacy else ""
            lines.append(
                f"  {rank:2d}. {base:35s} "
                f"novelty={info.get('composite_novelty', 0):.3f}{tag}"
            )
        lines.append("")

    lines.append("=" * 70)
    lines.append(f"Outputs saved to: {OUTPUT_DIR}")
    lines.append("=" * 70)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()

    # ── Phase 0: Load phylogeny data + rebuild corpus/motifs ──────────
    print("Phase 0: Loading phylogeny results...")
    phylo_results, existing_freq_df = load_phylogeny_results()
    print(f"  Loaded fingerprint matrix: {existing_freq_df.shape}")
    survivors = phylo_results["cross_level_survival"]["survivors_at_full_reduce"]
    print(f"  Survivors at full_reduce: {len(survivors)}")
    print(f"  Surprise pairs: {len(phylo_results['surprises'])}")

    print("  Rebuilding corpus & discovering motifs...")
    t_corpus = time.time()
    corpus = build_corpus()
    motifs = discover_motifs(corpus)
    print(f"  Corpus: {len(corpus)} graphs, {len(motifs)} motifs ({time.time() - t_corpus:.1f}s)")

    # ── Phase 1: Build motif template registry ────────────────────────
    print("\nPhase 1: Building motif template registry...")
    templates = build_template_registry(motifs, existing_freq_df, corpus, survivors)
    n_hand = sum(1 for k in templates if k in _HANDCRAFTED_TEMPLATES)
    n_auto = len(templates) - n_hand
    print(f"  Templates: {n_hand} handcrafted + {n_auto} auto-extracted = {len(templates)} total")

    # ── Phase 2: Run 4 strategies ─────────────────────────────────────
    print("\nPhase 2: Running discovery strategies...")

    specs_coverage = strategy_coverage_gap(phylo_results, existing_freq_df, templates)
    print(f"  Strategy 1 (Coverage Gap):     {len(specs_coverage)} candidates")

    specs_bridge = strategy_cross_family_bridge(phylo_results, templates)
    print(f"  Strategy 2 (Cross-Family):     {len(specs_bridge)} candidates")

    specs_irreducible = strategy_irreducible_composition(phylo_results, templates)
    print(f"  Strategy 3 (Irreducible):      {len(specs_irreducible)} candidates")

    specs_void = strategy_pca_void(existing_freq_df, templates)
    print(f"  Strategy 4 (PCA Void):         {len(specs_void)} candidates")

    all_specs = specs_coverage + specs_bridge + specs_irreducible + specs_void
    print(f"  Total candidate specs: {len(all_specs)}")

    # ── Phase 3: Build circuits ───────────────────────────────────────
    print("\nPhase 3: Building circuits from specs...")
    built_candidates = []  # (name, qc, spec)
    for spec in all_specs:
        qc = build_circuit_from_spec(spec, templates)
        if qc is not None and qc.size() > 0:
            built_candidates.append((spec.name, qc, spec))
            print(f"  {spec.name}: {qc.num_qubits}q, {qc.size()} gates, depth {qc.depth()}")
        else:
            print(f"  {spec.name}: FAILED to build")

    print(f"  Built {len(built_candidates)}/{len(all_specs)} circuits")

    # ── Phase 4: Validate ─────────────────────────────────────────────
    print("\nPhase 4: Validating candidates...")
    validations = {}
    candidate_strategies = {}
    for name, qc, spec in built_candidates:
        val = validate_candidate(name, qc)
        validations[name] = val
        candidate_strategies[name] = spec.strategy
        status = "PASS" if val["is_unitary"] else "FAIL"
        tensor = (
            "yes"
            if val.get("tensors_match")
            else ("n/a" if val["tensors_match"] is None else "no")
        )
        print(f"  {name}: {status} (err={val['unitarity_error']:.2e}, tensor={tensor})")

    n_passed = sum(1 for v in validations.values() if v["is_unitary"])
    print(f"  Passed: {n_passed}/{len(validations)}")

    # ── Phase 5: Fingerprint new candidates ───────────────────────────
    print("\nPhase 5: Fingerprinting candidates...")
    t5 = time.time()
    cand_pairs = [(name, qc) for name, qc, _ in built_candidates]
    candidate_freq_df = fingerprint_candidates(
        cand_pairs, corpus, motifs, list(existing_freq_df.columns)
    )
    candidate_freq_df.to_csv(OUTPUT_DIR / "candidate_fingerprints.csv")
    print(f"  Done ({time.time() - t5:.1f}s)")

    # ── Phase 6: Score & rank ─────────────────────────────────────────
    print("\nPhase 6: Scoring candidates...")
    pca_info = compute_pca_info(existing_freq_df)
    scored = score_candidates(
        existing_freq_df, candidate_freq_df, pca_info, validations, candidate_strategies
    )

    for cname in sorted(scored, key=lambda c: -scored[c].get("composite_novelty", 0)):
        info = scored[cname]
        print(
            f"  {cname}: composite={info['composite_novelty']:.3f} "
            f"(cos={info['cosine_novelty']:.3f}, "
            f"pca={info['pca_isolation']:.3f}, "
            f"div={info['motif_diversity']:.3f})"
        )

    # Legacy comparison: score old TVH/DEB/ICC
    print("\n  Scoring legacy candidates for comparison...")
    legacy_candidates = [
        ("legacy_tvh", make_trotter_variational_hybrid(n_qubits=4)),
        ("legacy_deb", make_distillation_entanglement_bridge(n_qubits=4)),
        ("legacy_icc", make_irreducible_core_circuit(n_qubits=4)),
    ]
    legacy_validations = {}
    for name, qc in legacy_candidates:
        legacy_validations[name] = validate_candidate(name, qc)

    legacy_freq_df = fingerprint_candidates(
        legacy_candidates, corpus, motifs, list(existing_freq_df.columns)
    )
    legacy_strategies = {n: "legacy" for n, _ in legacy_candidates}
    legacy_scored = score_candidates(
        existing_freq_df, legacy_freq_df, pca_info, legacy_validations, legacy_strategies
    )
    for cname, info in legacy_scored.items():
        print(
            f"  {cname}: composite={info['composite_novelty']:.3f} [LEGACY]"
        )

    # ── Phase 7: Visualize ────────────────────────────────────────────
    print("\nPhase 7: Generating visualizations...")
    plot_pca_with_candidates(existing_freq_df, candidate_freq_df, pca_info, scored)
    print("  Saved pca_candidates.png")
    plot_dendrogram_with_candidates(existing_freq_df, candidate_freq_df, scored)
    print("  Saved dendrogram_candidates.png")
    plot_candidate_profiles(candidate_freq_df, scored)
    print("  Saved candidate_profiles.png")
    plot_strategy_comparison(scored)
    print("  Saved strategy_comparison.png")

    # ── Phase 8: Report ───────────────────────────────────────────────
    print("\nPhase 8: Generating report...")
    specs_by_name = {spec.name: spec for spec in all_specs}
    report = generate_report(validations, scored, specs_by_name, legacy_scored)
    print(report)

    (OUTPUT_DIR / "report.txt").write_text(report)

    # JSON results
    json_results = {
        "n_candidates": len(scored),
        "n_passed": n_passed,
        "strategies": {
            "coverage_gap": len(specs_coverage),
            "cross_family": len(specs_bridge),
            "irreducible": len(specs_irreducible),
            "pca_void": len(specs_void),
        },
        "validations": validations,
        "scored": {},
        "legacy_scored": {},
    }
    for k, v in scored.items():
        json_results["scored"][k] = {kk: vv for kk, vv in v.items()}
    for k, v in legacy_scored.items():
        json_results["legacy_scored"][k] = {kk: vv for kk, vv in v.items()}

    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(json_results, indent=2, default=str)
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. All outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
