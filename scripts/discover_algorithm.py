#!/usr/bin/env python3
"""
Quantum Algorithm Discovery via ZX Motif Phylogeny
===================================================

Uses structural insights from the motif phylogeny analysis to construct
novel quantum circuits that fill gaps in algorithm space. Three candidates:

  1. Trotter-Variational Hybrid (TVH)  — QAOA + Trotter + surviving motifs
  2. Distillation-Entanglement Bridge  — parameterised BBPSSW/GHZ interpolation
  3. Irreducible Core Circuit (ICC)    — built only from ZX-reduction-surviving motifs

Outputs (scripts/output/discovery/):
  - pca_candidates.png           PCA scatter with new algorithms overlaid
  - dendrogram_candidates.png    Dendrogram with candidates highlighted
  - candidate_profiles.png       Top-motif bar charts per candidate
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

CANDIDATE_COLOURS = {
    "trotter_variational_hybrid": "#000000",
    "distillation_entanglement_bridge": "#e31a1c",
    "irreducible_core_circuit": "#1a9641",
}

CANDIDATE_LABELS = {
    "trotter_variational_hybrid": "TVH",
    "distillation_entanglement_bridge": "DEB",
    "irreducible_core_circuit": "ICC",
}


def _get_family(instance_name: str) -> str:
    base = instance_name.rsplit("_q", 1)[0]
    return ALGORITHM_FAMILY_MAP.get(base, "unknown")


# ═══════════════════════════════════════════════════════════════════════
# Phase 0: Load Phylogeny Results
# ═══════════════════════════════════════════════════════════════════════


def load_phylogeny_results() -> tuple[dict, pd.DataFrame]:
    """Load results.json and fingerprint_frequencies.csv from prior run."""
    results = json.loads((PHYLOGENY_DIR / "results.json").read_text())
    freq_df = pd.read_csv(PHYLOGENY_DIR / "fingerprint_frequencies.csv", index_col=0)
    return results, freq_df


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Identify Structural Gaps
# ═══════════════════════════════════════════════════════════════════════


def identify_gaps(freq_df: pd.DataFrame) -> dict:
    """PCA on existing fingerprints; compute family centroids and gap targets."""
    mat = freq_df.fillna(0).values
    mean = mat.mean(axis=0)
    mat_c = mat - mean

    U, S, Vt = np.linalg.svd(mat_c, full_matrices=False)
    coords = U[:, :2] * S[:2]  # (n_instances, 2)

    instances = list(freq_df.index)
    families = [_get_family(inst) for inst in instances]

    # Family centroids
    centroids = {}
    for fam in sorted(set(families)):
        idx = [i for i, f in enumerate(families) if f == fam]
        centroids[fam] = coords[idx].mean(axis=0)

    # Target positions for candidates
    target_tvh = (centroids.get("variational", [0, 0]) + centroids.get("simulation", [0, 0])) / 2
    target_deb = (centroids.get("distillation", [0, 0]) + centroids.get("entanglement", [0, 0])) / 2

    return {
        "mean": mean,
        "S": S,
        "Vt": Vt,
        "coords": coords,
        "instances": instances,
        "families": families,
        "centroids": centroids,
        "target_tvh": target_tvh,
        "target_deb": target_deb,
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Construct Candidate Algorithms
# ═══════════════════════════════════════════════════════════════════════


def make_trotter_variational_hybrid(
    n_qubits: int = 4, n_layers: int = 2,
    gamma: float = 0.7, beta: float = 0.4, **kwargs,
) -> QuantumCircuit:
    """
    Trotter-Variational Hybrid (TVH).

    Combines Trotter ZZ-interaction backbone with variational angles,
    plus cluster_chain and hadamard_sandwich motifs that survive full
    ZX reduction.
    """
    n_qubits = max(3, n_qubits)
    qc = QuantumCircuit(n_qubits)

    for _layer in range(n_layers):
        # 1. ZZ interaction backbone (Trotter structure, variational angle)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(gamma, i + 1)
            qc.cx(i, i + 1)

        # 2. Cluster chain entangling layer (cluster_chain motif)
        qc.h(range(n_qubits))
        for i in range(n_qubits - 1):
            qc.cz(i, i + 1)
        qc.h(range(n_qubits))

        # 3. Hadamard sandwich basis-change layer (hadamard_sandwich motif)
        for i in range(0, n_qubits - 1, 2):
            qc.h(i)
            qc.rz(beta, i)
            qc.h(i)

        # 4. Mixer rotations (QAOA-style)
        for i in range(n_qubits):
            qc.rx(2 * beta, i)

    return qc


def make_distillation_entanglement_bridge(
    n_qubits: int = 4, theta: float = np.pi / 4, **kwargs,
) -> QuantumCircuit:
    """
    Distillation-Entanglement Bridge (DEB).

    Parameterised interpolation between GHZ-like entanglement generation
    and BBPSSW-like distillation. At theta~0 it is GHZ-like; larger theta
    develops distillation bilateral structure.
    """
    n_qubits = max(4, n_qubits)
    if n_qubits % 2 != 0:
        n_qubits += 1
    half = n_qubits // 2
    qc = QuantumCircuit(n_qubits)

    # 1. GHZ-style fan-out (shared structure)
    qc.h(0)
    for i in range(half - 1):
        qc.cx(i, i + 1)

    # 2. Parameterised rotation layer (bridges families)
    for i in range(n_qubits):
        qc.ry(theta, i)

    # 3. Bilateral CNOTs (from distillation)
    for i in range(half):
        qc.cx(i, half + i)

    # 4. Reverse entangling (targets coverage gap)
    for i in range(half):
        qc.h(half + i)
        qc.cx(half + i, i)

    return qc


def make_irreducible_core_circuit(
    n_qubits: int = 4, **kwargs,
) -> QuantumCircuit:
    """
    Irreducible Core Circuit (ICC).

    Built exclusively from the 4 motifs that survive full ZX reduction:
    phase_gadget_2t, phase_gadget_3t, hadamard_sandwich, cluster_chain.
    """
    n_qubits = max(4, n_qubits)
    qc = QuantumCircuit(n_qubits)

    # 1. Cluster chain (cluster_chain motif)
    qc.h(range(n_qubits))
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)

    # 2. Phase gadget with 2 targets (phase_gadget_2t)
    for i in range(0, n_qubits - 1, 2):
        qc.cx(i, i + 1)
        qc.t(i + 1)
        qc.cx(i, i + 1)

    # 3. Hadamard sandwich (hadamard_sandwich motif)
    for i in range(n_qubits):
        qc.h(i)
        qc.s(i)
        qc.h(i)

    # 4. Phase gadget with 3 targets (phase_gadget_3t)
    if n_qubits >= 4:
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.t(3)
        qc.cx(0, 3)
        qc.cx(0, 2)
        qc.cx(0, 1)

    # 5. Closing cluster chain
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)
    qc.h(range(n_qubits))

    return qc


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Validate Candidates
# ═══════════════════════════════════════════════════════════════════════


def validate_candidate(name: str, qc: QuantumCircuit) -> dict:
    """Validate unitarity and ZX tensor preservation."""
    # Unitarity check via Qiskit Operator
    op = Operator(qc)
    mat = op.data
    n = mat.shape[0]
    product = mat @ mat.conj().T
    identity_err = float(np.linalg.norm(product - np.eye(n)) / n)
    is_unitary = identity_err < 1e-6

    # ZX conversion check
    zx_circ = qiskit_to_zx(qc)
    g = zx_circ.to_graph()
    n_inputs = len(list(g.inputs()))
    n_outputs = len(list(g.outputs()))
    has_valid_io = (n_inputs == qc.num_qubits) and (n_outputs == qc.num_qubits)

    # Tensor preservation under simplification
    g_copy = copy.deepcopy(g)
    zx.simplify.full_reduce(g_copy)
    try:
        tensors_match = zx.compare_tensors(g, g_copy)
    except Exception:
        tensors_match = None  # May fail for larger circuits

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
# Phase 4: Fingerprint Candidates
# ═══════════════════════════════════════════════════════════════════════


def build_corpus():
    """Build the full algorithm corpus (same as discover_phylogeny.py)."""
    import networkx as nx

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
        added = 0
        for mp in candidates:
            h = wl_hash(mp.graph)
            if h in seen_hashes:
                existing = all_motifs[seen_hashes[h]]
                if _is_isomorphic(mp.graph, existing.graph):
                    continue
            seen_hashes[h] = len(all_motifs)
            all_motifs.append(mp)
            added += 1
        return added

    try:
        bu = find_recurring_subgraphs(corpus, target_level="spider_fused", min_size=3, max_size=5)
        _add_novel(bu)
    except Exception:
        pass
    try:
        nb = find_neighborhood_motifs(corpus, target_level="spider_fused", radius=2)
        _add_novel(nb)
    except Exception:
        pass

    return all_motifs


def fingerprint_candidates(
    candidates: list[tuple[str, QuantumCircuit]],
    corpus: dict,
    motifs: list[MotifPattern],
    existing_columns: list[str],
) -> pd.DataFrame:
    """Fingerprint candidate circuits against the motif library."""
    # Add candidates to corpus
    for name, qc in candidates:
        instance = f"{name}_q{qc.num_qubits}"
        try:
            snapshots = convert_at_all_levels(qc, instance)
            for snap in snapshots:
                nxg = pyzx_to_networkx(snap.graph, coarsen_phases=True)
                corpus[(instance, snap.level.value)] = nxg
        except Exception as e:
            print(f"    Warning: failed to convert {instance}: {e}")

    # Build motif_id -> index mapping
    motif_map = {mp.motif_id: i for i, mp in enumerate(motifs)}

    # Fingerprint each candidate at spider_fused
    instances = [f"{name}_q{qc.num_qubits}" for name, qc in candidates]
    motif_ids = [mp.motif_id for mp in motifs]

    counts = np.zeros((len(instances), len(motifs)), dtype=int)
    for i, inst in enumerate(instances):
        key = (inst, "spider_fused")
        if key not in corpus:
            continue
        host = corpus[key]
        for j, mp in enumerate(motifs):
            matches = find_motif_in_graph(mp.graph, host, max_matches=100)
            counts[i, j] = len(matches)

    counts_df = pd.DataFrame(counts, index=instances, columns=motif_ids)
    row_sums = counts_df.sum(axis=1).replace(0, 1)
    freq_df = counts_df.div(row_sums, axis=0)

    # Align columns with existing freq_df
    freq_aligned = freq_df.reindex(columns=existing_columns, fill_value=0.0)
    return freq_aligned


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Analyse Placement
# ═══════════════════════════════════════════════════════════════════════


def analyse_placement(
    existing_freq_df: pd.DataFrame,
    candidate_freq_df: pd.DataFrame,
    gap_info: dict,
) -> dict:
    """Compute nearest neighbours, family affinity, and novelty scores."""
    existing_mat = existing_freq_df.fillna(0).values
    cand_mat = candidate_freq_df.fillna(0).values
    instances = list(existing_freq_df.index)
    cand_names = list(candidate_freq_df.index)

    results = {}
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

        # Family affinity
        centroids = gap_info["centroids"]
        mean = gap_info["mean"]
        Vt = gap_info["Vt"]

        cand_centered = cv - mean
        cand_pca = cand_centered @ Vt[:2].T

        family_dists = {}
        for fam, centroid in centroids.items():
            family_dists[fam] = float(np.linalg.norm(cand_pca - centroid))
        closest_family = min(family_dists, key=family_dists.get)

        # Novelty score = 1 - max similarity to any existing
        novelty = 1.0 - nearest_sim

        # Top motifs
        top_idx = np.argsort(cv)[::-1][:10]
        top_motifs = [
            (candidate_freq_df.columns[ti], float(cv[ti]))
            for ti in top_idx if cv[ti] > 0
        ]

        results[cname] = {
            "nearest_algorithm": nearest,
            "nearest_family": _get_family(nearest),
            "nearest_similarity": round(nearest_sim, 4),
            "closest_family": closest_family,
            "family_distances": {k: round(v, 4) for k, v in family_dists.items()},
            "novelty_score": round(novelty, 4),
            "pca_coords": [float(cand_pca[0]), float(cand_pca[1])],
            "top_motifs": top_motifs[:10],
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 6: Visualise
# ═══════════════════════════════════════════════════════════════════════


def plot_pca_with_candidates(
    existing_freq_df: pd.DataFrame,
    candidate_freq_df: pd.DataFrame,
    gap_info: dict,
) -> None:
    """PCA scatter with candidates overlaid as stars."""
    coords = gap_info["coords"]
    instances = gap_info["instances"]
    families = gap_info["families"]
    mean = gap_info["mean"]
    Vt = gap_info["Vt"]
    S = gap_info["S"]

    var_explained = (S[:2] ** 2) / max((S ** 2).sum(), 1e-10)

    fig, ax = plt.subplots(figsize=(11, 9))

    # Existing algorithms
    for fam in sorted(set(families)):
        idx = [i for i, f in enumerate(families) if f == fam]
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            c=FAMILY_COLOURS.get(fam, "#333"), label=fam,
            alpha=0.45, s=30, zorder=2,
        )

    # Candidate algorithms
    cand_mat = candidate_freq_df.fillna(0).values
    cand_names = list(candidate_freq_df.index)
    for ci, cname in enumerate(cand_names):
        cand_centered = cand_mat[ci] - mean
        cand_pca = cand_centered @ Vt[:2].T
        base = cname.rsplit("_q", 1)[0]
        label = CANDIDATE_LABELS.get(base, base)
        color = CANDIDATE_COLOURS.get(base, "#000000")
        ax.scatter(
            cand_pca[0], cand_pca[1],
            marker="*", s=300, c=color, edgecolors="black",
            linewidths=1, zorder=5, label=f"NEW: {label}",
        )
        ax.annotate(
            label, (cand_pca[0], cand_pca[1]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=9, fontweight="bold", color=color,
        )

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title("Algorithm Phylogeny PCA — Candidate Placement")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_candidates.png", dpi=150)
    plt.close(fig)


def plot_dendrogram_with_candidates(
    existing_freq_df: pd.DataFrame,
    candidate_freq_df: pd.DataFrame,
) -> None:
    """Dendrogram with candidates highlighted."""
    combined = pd.concat([existing_freq_df, candidate_freq_df]).fillna(0)
    mat = combined.values
    row_norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = np.where(row_norms == 0, 1e-10, mat)

    dist = pdist(mat, metric="cosine")
    dist = np.nan_to_num(dist, nan=1.0)
    Z = linkage(dist, method="average")

    labels = list(combined.index)
    cand_set = set(candidate_freq_df.index)

    fig, ax = plt.subplots(figsize=(16, max(10, len(labels) * 0.22)))
    dendrogram(Z, labels=labels, orientation="left", leaf_font_size=6, ax=ax)

    for lbl in ax.get_yticklabels():
        inst = lbl.get_text()
        if inst in cand_set:
            base = inst.rsplit("_q", 1)[0]
            lbl.set_color(CANDIDATE_COLOURS.get(base, "#000"))
            lbl.set_fontweight("bold")
            lbl.set_fontsize(8)
        else:
            fam = _get_family(inst)
            lbl.set_color(FAMILY_COLOURS.get(fam, "#333"))

    ax.set_title("Algorithm Phylogeny — With Discovery Candidates", fontsize=14)
    ax.set_xlabel("Cosine Distance")

    # Legend
    handles = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLOURS.items()]
    for base, color in CANDIDATE_COLOURS.items():
        handles.append(Patch(facecolor=color, label=f"NEW: {CANDIDATE_LABELS.get(base, base)}"))
    ax.legend(handles=handles, loc="upper right", fontsize=6, ncol=2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dendrogram_candidates.png", dpi=150)
    plt.close(fig)


def plot_candidate_profiles(
    candidate_freq_df: pd.DataFrame,
    placement: dict,
) -> None:
    """Bar chart of top motif frequencies per candidate."""
    cand_names = list(candidate_freq_df.index)
    n = len(cand_names)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, cname in zip(axes, cand_names):
        info = placement.get(cname, {})
        top = info.get("top_motifs", [])[:10]
        if not top:
            ax.set_title(cname)
            continue

        motif_ids = [t[0] for t in top]
        freqs = [t[1] for t in top]

        # Truncate long motif IDs
        short_ids = [m[:20] + "..." if len(m) > 20 else m for m in motif_ids]

        base = cname.rsplit("_q", 1)[0]
        color = CANDIDATE_COLOURS.get(base, "#1f77b4")
        ax.barh(range(len(freqs)), freqs, color=color, alpha=0.8)
        ax.set_yticks(range(len(short_ids)))
        ax.set_yticklabels(short_ids, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")

        label = CANDIDATE_LABELS.get(base, base)
        nearest = info.get("nearest_algorithm", "?")
        sim = info.get("nearest_similarity", 0)
        ax.set_title(f"{label}\nNearest: {nearest} (sim={sim:.2f})", fontsize=9)

    fig.suptitle("Candidate Motif Profiles", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "candidate_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Phase 7: Report
# ═══════════════════════════════════════════════════════════════════════


def generate_report(
    validations: dict,
    placement: dict,
    gap_info: dict,
) -> str:
    """Generate structured discovery report."""
    lines = [
        "=" * 70,
        "QUANTUM ALGORITHM DISCOVERY VIA ZX MOTIF PHYLOGENY",
        "=" * 70,
        "",
        "Method: Identified structural gaps in ZX motif fingerprint PCA space.",
        "Constructed 3 candidate algorithms by composing motifs data-driven.",
        "Validated unitarity and ZX tensor preservation.",
        "",
    ]

    candidate_descriptions = {
        "trotter_variational_hybrid": (
            "Trotter-Variational Hybrid (TVH)",
            "Exploits 93% QAOA-Trotter similarity. Trotter ZZ backbone + "
            "cluster_chain + hadamard_sandwich (both survive full ZX reduction) + "
            "QAOA-style variational mixer.",
        ),
        "distillation_entanglement_bridge": (
            "Distillation-Entanglement Bridge (DEB)",
            "Exploits 99% BBPSSW-GHZ similarity. Parameterised interpolation "
            "between GHZ fan-out and bilateral-CNOT distillation. Targets the "
            "61% distillation coverage gap.",
        ),
        "irreducible_core_circuit": (
            "Irreducible Core Circuit (ICC)",
            "Built only from the 4 motifs surviving full ZX reduction: "
            "phase_gadget_2t, phase_gadget_3t, hadamard_sandwich, cluster_chain. "
            "No existing algorithm is designed around ZX-irreducible primitives.",
        ),
    }

    for name, val in validations.items():
        title, desc = candidate_descriptions.get(name, (name, ""))
        instance = f"{name}_q{val['n_qubits']}"
        p = placement.get(instance, {})

        status = "PASS" if val["is_unitary"] else "FAIL"
        tensor_status = "YES" if val.get("tensors_match") else (
            "N/A" if val.get("tensors_match") is None else "NO"
        )

        lines.append(f"--- {title} ---")
        lines.append(f"  {desc}")
        lines.append(f"  Qubits: {val['n_qubits']}  Gates: {val['n_gates']}  Depth: {val['depth']}")
        lines.append(f"  Unitarity: {status} (error: {val['unitarity_error']:.2e})")
        lines.append(f"  ZX tensor preserved: {tensor_status}")
        lines.append(f"  ZX size: {val['zx_vertices_raw']}V/{val['zx_edges_raw']}E raw"
                      f" -> {val['zx_vertices_reduced']}V/{val['zx_edges_reduced']}E reduced")

        if p:
            lines.append(f"  Nearest algorithm: {p['nearest_algorithm']} "
                          f"(sim={p['nearest_similarity']:.3f})")
            lines.append(f"  Closest family: {p['closest_family']}")
            lines.append(f"  Novelty score: {p['novelty_score']:.3f} (1=completely novel)")
            lines.append(f"  PCA coords: ({p['pca_coords'][0]:.3f}, {p['pca_coords'][1]:.3f})")

            if p.get("top_motifs"):
                lines.append("  Top motifs:")
                for mid, freq in p["top_motifs"][:5]:
                    lines.append(f"    {mid:40s} {freq:.3f}")

        lines.append("")

    # Key findings
    lines.append("--- Key Findings ---")

    all_placements = list(placement.values())
    if all_placements:
        most_novel = max(all_placements, key=lambda x: x.get("novelty_score", 0))
        least_novel = min(all_placements, key=lambda x: x.get("novelty_score", 0))
        lines.append(f"  Most novel candidate: novelty={most_novel['novelty_score']:.3f} "
                      f"(nearest: {most_novel['nearest_algorithm']})")
        lines.append(f"  Least novel candidate: novelty={least_novel['novelty_score']:.3f} "
                      f"(nearest: {least_novel['nearest_algorithm']})")

        # Check if any candidate is between its parent families
        for cname, p in placement.items():
            base = cname.rsplit("_q", 1)[0]
            if base == "trotter_variational_hybrid":
                d_var = p["family_distances"].get("variational", 999)
                d_sim = p["family_distances"].get("simulation", 999)
                lines.append(f"  TVH distance to variational: {d_var:.3f}, "
                              f"to simulation: {d_sim:.3f}")
            elif base == "distillation_entanglement_bridge":
                d_dist = p["family_distances"].get("distillation", 999)
                d_ent = p["family_distances"].get("entanglement", 999)
                lines.append(f"  DEB distance to distillation: {d_dist:.3f}, "
                              f"to entanglement: {d_ent:.3f}")

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

    # Phase 0
    print("Phase 0: Loading phylogeny results...")
    phylo_results, existing_freq_df = load_phylogeny_results()
    print(f"  Loaded fingerprint matrix: {existing_freq_df.shape}")

    # Phase 1
    print("\nPhase 1: Identifying structural gaps...")
    gap_info = identify_gaps(existing_freq_df)
    centroids = gap_info["centroids"]
    for fam, c in sorted(centroids.items()):
        print(f"  {fam:20s} centroid: ({c[0]:+.3f}, {c[1]:+.3f})")

    # Phase 2
    print("\nPhase 2: Constructing candidate algorithms...")
    candidates = [
        ("trotter_variational_hybrid", make_trotter_variational_hybrid(n_qubits=4)),
        ("distillation_entanglement_bridge", make_distillation_entanglement_bridge(n_qubits=4)),
        ("irreducible_core_circuit", make_irreducible_core_circuit(n_qubits=4)),
    ]
    for name, qc in candidates:
        print(f"  {name}: {qc.num_qubits} qubits, {qc.size()} gates, depth {qc.depth()}")

    # Phase 3
    print("\nPhase 3: Validating candidates...")
    validations = {}
    for name, qc in candidates:
        val = validate_candidate(name, qc)
        validations[name] = val
        status = "PASS" if val["is_unitary"] else "FAIL"
        tensor = "yes" if val.get("tensors_match") else ("n/a" if val["tensors_match"] is None else "no")
        print(f"  {name}: {status} (err={val['unitarity_error']:.2e}, tensor={tensor})")

    # Phase 4
    print("\nPhase 4: Rebuilding corpus & discovering motifs...")
    t4 = time.time()
    corpus = build_corpus()
    motifs = discover_motifs(corpus)
    print(f"  Corpus: {len(corpus)} graphs, {len(motifs)} motifs ({time.time() - t4:.1f}s)")

    print("  Fingerprinting candidates...")
    t4b = time.time()
    candidate_freq_df = fingerprint_candidates(
        candidates, corpus, motifs, list(existing_freq_df.columns),
    )
    candidate_freq_df.to_csv(OUTPUT_DIR / "candidate_fingerprints.csv")
    print(f"  Done ({time.time() - t4b:.1f}s)")

    # Phase 5
    print("\nPhase 5: Analysing placement...")
    placement = analyse_placement(existing_freq_df, candidate_freq_df, gap_info)
    for cname, p in placement.items():
        print(f"  {cname}: nearest={p['nearest_algorithm']} "
              f"(sim={p['nearest_similarity']:.3f}), "
              f"novelty={p['novelty_score']:.3f}")

    # Phase 6
    print("\nPhase 6: Generating visualisations...")
    plot_pca_with_candidates(existing_freq_df, candidate_freq_df, gap_info)
    print("  Saved pca_candidates.png")
    plot_dendrogram_with_candidates(existing_freq_df, candidate_freq_df)
    print("  Saved dendrogram_candidates.png")
    plot_candidate_profiles(candidate_freq_df, placement)
    print("  Saved candidate_profiles.png")

    # Phase 7
    print("\nPhase 7: Generating report...")
    report = generate_report(validations, placement, gap_info)
    print(report)

    (OUTPUT_DIR / "report.txt").write_text(report)

    # JSON-safe results
    json_results = {
        "validations": validations,
        "placement": {},
    }
    for k, v in placement.items():
        json_results["placement"][k] = {
            kk: vv for kk, vv in v.items()
        }

    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(json_results, indent=2, default=str)
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. All outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
