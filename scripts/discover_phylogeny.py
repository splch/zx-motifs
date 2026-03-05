#!/usr/bin/env python3
"""
Quantum Algorithm Phylogeny via ZX Motif Fingerprints
=====================================================

Fingerprints all registered quantum algorithms by their ZX motif profiles,
clusters them into a phylogenetic tree, and uncovers structural relationships
across algorithm families.

Outputs (scripts/output/):
  - phylogeny_dendrogram.png   Hierarchical clustering dendrogram
  - pca_scatter.png            2D PCA scatter coloured by family
  - universality_spectrum.png  Motif universality bar chart
  - cross_level_survival.png   Motif survival heatmap across levels
  - coverage_landscape.png     Decomposition coverage by family
  - fingerprint_counts.csv     Raw match counts
  - fingerprint_frequencies.csv  L1-normalised frequencies
  - results.json               Machine-readable results
  - report.txt                 Human-readable findings
"""

from __future__ import annotations

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
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cosine, pdist, squareform

# ── Project imports ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zx_motifs.algorithms.registry import ALGORITHM_FAMILY_MAP, REGISTRY
from zx_motifs.pipeline.converter import SimplificationLevel, convert_at_all_levels
from zx_motifs.pipeline.decomposer import decompose_across_corpus
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

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette for the 10 families
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
}


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Build Corpus
# ═══════════════════════════════════════════════════════════════════════


def build_corpus() -> dict[tuple[str, str], "nx.Graph"]:
    """Convert every algorithm instance into NetworkX graphs at all levels."""
    import networkx as nx

    corpus: dict[tuple[str, str], nx.Graph] = {}
    errors: list[str] = []

    for entry in REGISTRY:
        lo, hi = entry.qubit_range
        qubit_sizes = list(range(lo, min(hi, 6) + 1))

        for n in qubit_sizes:
            # Known failure: Grover QASM at n>=6
            if entry.name == "grover" and n >= 6:
                continue

            instance = f"{entry.name}_q{n}"
            try:
                qc = entry.generator(n)
                snapshots = convert_at_all_levels(qc, instance)
                for snap in snapshots:
                    nxg = pyzx_to_networkx(snap.graph, coarsen_phases=True)
                    corpus[(instance, snap.level.value)] = nxg
            except Exception as e:
                errors.append(f"{instance}: {e}")

    if errors:
        print(f"  Skipped {len(errors)} instances due to errors:")
        for err in errors[:10]:
            print(f"    {err}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")

    return corpus


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Discover Motifs
# ═══════════════════════════════════════════════════════════════════════


def discover_motifs(
    corpus: dict,
) -> list[MotifPattern]:
    """Combine hand-crafted, bottom-up, and neighbourhood motifs; deduplicate."""
    # Hand-crafted (already MotifPattern objects)
    all_motifs: list[MotifPattern] = list(EXTENDED_MOTIFS)
    seen_hashes: dict[str, int] = {}

    # Record hashes of hand-crafted motifs
    for i, mp in enumerate(all_motifs):
        h = wl_hash(mp.graph)
        seen_hashes[h] = i

    def _add_if_novel(candidates: list[MotifPattern]) -> int:
        added = 0
        for mp in candidates:
            h = wl_hash(mp.graph)
            if h in seen_hashes:
                # Check for hash collision via VF2
                existing = all_motifs[seen_hashes[h]]
                if _is_isomorphic(mp.graph, existing.graph):
                    continue
            seen_hashes[h] = len(all_motifs)
            all_motifs.append(mp)
            added += 1
        return added

    # Bottom-up enumeration
    try:
        bottom_up = find_recurring_subgraphs(
            corpus,
            target_level="spider_fused",
            min_size=3,
            max_size=5,
        )
        n_bu = _add_if_novel(bottom_up)
        print(f"  Bottom-up: {len(bottom_up)} found, {n_bu} novel")
    except Exception as e:
        print(f"  Bottom-up failed: {e}")

    # Neighbourhood extraction
    try:
        neighbourhood = find_neighborhood_motifs(
            corpus,
            target_level="spider_fused",
            radius=2,
        )
        n_nb = _add_if_novel(neighbourhood)
        print(f"  Neighbourhood: {len(neighbourhood)} found, {n_nb} novel")
    except Exception as e:
        print(f"  Neighbourhood failed: {e}")

    return all_motifs


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Build Fingerprint Matrix
# ═══════════════════════════════════════════════════════════════════════


def build_fingerprint_matrix(
    corpus: dict,
    motifs: list[MotifPattern],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count motif occurrences per algorithm instance at spider_fused level."""
    # Collect algorithm instances at spider_fused
    instances = sorted(
        {name for (name, level) in corpus if level == "spider_fused"}
    )
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

    # L1-normalise each row
    row_sums = counts_df.sum(axis=1).replace(0, 1)
    freq_df = counts_df.div(row_sums, axis=0)

    return counts_df, freq_df


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Analyses
# ═══════════════════════════════════════════════════════════════════════


def _get_family(instance_name: str) -> str:
    """Strip _q{n} to get base algorithm name, then look up family."""
    base = instance_name.rsplit("_q", 1)[0]
    return ALGORITHM_FAMILY_MAP.get(base, "unknown")


def _family_color_list(instances: list[str]) -> list[str]:
    return [FAMILY_COLOURS.get(_get_family(inst), "#333333") for inst in instances]


# ── 4a. Phylogenetic Dendrogram ──────────────────────────────────────


def analysis_phylogeny(freq_df: pd.DataFrame) -> dict:
    """Hierarchical clustering on cosine distance."""
    if freq_df.shape[0] < 2:
        print("  Skipping phylogeny: fewer than 2 instances")
        return {}

    # Replace NaN with 0 for distance computation
    mat = freq_df.fillna(0).values

    # Some rows may be all-zero; add tiny epsilon to avoid nan distance
    row_norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = np.where(row_norms == 0, 1e-10, mat)

    dist = pdist(mat, metric="cosine")
    dist = np.nan_to_num(dist, nan=1.0)
    Z = linkage(dist, method="average")

    fig, ax = plt.subplots(figsize=(16, max(8, len(freq_df) * 0.25)))
    labels = list(freq_df.index)
    colours = _family_color_list(labels)

    dendro = dendrogram(
        Z,
        labels=labels,
        orientation="left",
        leaf_font_size=7,
        ax=ax,
    )

    # Colour the labels by family
    yticklabels = ax.get_yticklabels()
    for lbl in yticklabels:
        inst = lbl.get_text()
        fam = _get_family(inst)
        lbl.set_color(FAMILY_COLOURS.get(fam, "#333333"))

    ax.set_title("Quantum Algorithm Phylogeny (ZX Motif Fingerprints)", fontsize=14)
    ax.set_xlabel("Cosine Distance")

    # Legend
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=c, label=f) for f, c in FAMILY_COLOURS.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "phylogeny_dendrogram.png", dpi=150)
    plt.close(fig)
    print("  Saved phylogeny_dendrogram.png")

    return {"linkage_shape": list(Z.shape)}


# ── 4b. PCA Scatter ─────────────────────────────────────────────────


def analysis_pca(freq_df: pd.DataFrame) -> dict:
    """SVD-based PCA on frequency vectors."""
    mat = freq_df.fillna(0).values
    # Centre
    mat_c = mat - mat.mean(axis=0)
    U, S, Vt = np.linalg.svd(mat_c, full_matrices=False)
    coords = U[:, :2] * S[:2]

    var_explained = (S[:2] ** 2) / max((S**2).sum(), 1e-10)

    fig, ax = plt.subplots(figsize=(10, 8))
    instances = list(freq_df.index)
    families = [_get_family(inst) for inst in instances]

    for fam in sorted(set(families)):
        idx = [i for i, f in enumerate(families) if f == fam]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            c=FAMILY_COLOURS.get(fam, "#333333"),
            label=fam,
            alpha=0.7,
            s=40,
        )

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)")
    ax.set_title("PCA of ZX Motif Fingerprints")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pca_scatter.png", dpi=150)
    plt.close(fig)
    print("  Saved pca_scatter.png")

    return {
        "variance_explained_pc1": float(var_explained[0]),
        "variance_explained_pc2": float(var_explained[1]),
    }


# ── 4c. Universality Spectrum ───────────────────────────────────────


def analysis_universality(counts_df: pd.DataFrame) -> dict:
    """Classify motifs as universal, common, or family-specific."""
    instances = list(counts_df.index)
    family_of = {inst: _get_family(inst) for inst in instances}
    all_families = sorted(set(family_of.values()))
    n_families = len(all_families)

    motif_stats = []
    for motif_id in counts_df.columns:
        col = counts_df[motif_id]
        present_instances = col[col > 0].index.tolist()
        present_families = set(family_of[inst] for inst in present_instances)
        n_fam = len(present_families)

        if n_fam == 0:
            category = "unused"
        elif n_fam >= n_families * 0.8:
            category = "universal"
        elif n_fam >= n_families * 0.4:
            category = "common"
        else:
            category = "specific"

        motif_stats.append(
            {
                "motif_id": motif_id,
                "n_families": n_fam,
                "category": category,
                "families": sorted(present_families),
            }
        )

    # Bar chart
    df_stats = pd.DataFrame(motif_stats).sort_values("n_families", ascending=False)

    cat_colours = {
        "universal": "#2ca02c",
        "common": "#1f77b4",
        "specific": "#ff7f0e",
        "unused": "#cccccc",
    }

    fig, ax = plt.subplots(figsize=(max(10, len(df_stats) * 0.4), 6))
    bar_colors = [cat_colours.get(c, "#999") for c in df_stats["category"]]
    ax.bar(range(len(df_stats)), df_stats["n_families"], color=bar_colors)
    ax.set_xticks(range(len(df_stats)))
    ax.set_xticklabels(df_stats["motif_id"], rotation=90, fontsize=6)
    ax.set_ylabel("Number of Families with Motif")
    ax.set_title("Motif Universality Spectrum")
    ax.axhline(
        y=n_families * 0.8, color="#2ca02c", linestyle="--", alpha=0.5, label="universal threshold"
    )
    ax.axhline(
        y=n_families * 0.4, color="#1f77b4", linestyle="--", alpha=0.5, label="common threshold"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "universality_spectrum.png", dpi=150)
    plt.close(fig)
    print("  Saved universality_spectrum.png")

    category_counts = {}
    for ms in motif_stats:
        category_counts[ms["category"]] = category_counts.get(ms["category"], 0) + 1

    return {
        "motif_stats": motif_stats,
        "category_counts": category_counts,
        "n_families": n_families,
    }


# ── 4d. Cross-level Survival ────────────────────────────────────────


def analysis_cross_level_survival(
    corpus: dict,
    motifs: list[MotifPattern],
) -> dict:
    """Check which motifs survive at each simplification level."""
    levels = [lvl.value for lvl in SimplificationLevel]

    # Pick one representative per algorithm (smallest qubit count)
    seen_bases: dict[str, str] = {}
    for name, level in corpus:
        base = name.rsplit("_q", 1)[0]
        if base not in seen_bases or name < seen_bases[base]:
            seen_bases[base] = name
    representatives = sorted(set(seen_bases.values()))

    motif_ids = [mp.motif_id for mp in motifs]
    survival = np.zeros((len(motifs), len(levels)), dtype=float)

    for j, level in enumerate(levels):
        for rep in representatives:
            key = (rep, level)
            if key not in corpus:
                continue
            host = corpus[key]
            for i, mp in enumerate(motifs):
                matches = find_motif_in_graph(mp.graph, host, max_matches=5)
                if len(matches) > 0:
                    survival[i, j] += 1

    # Normalise by number of representatives checked
    n_reps = len(representatives)
    if n_reps > 0:
        survival /= n_reps

    fig, ax = plt.subplots(figsize=(10, max(6, len(motifs) * 0.35)))
    sns.heatmap(
        survival,
        xticklabels=levels,
        yticklabels=motif_ids,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Fraction of algorithms with motif"},
    )
    ax.set_title("Motif Survival Across Simplification Levels")
    ax.set_xlabel("Simplification Level")
    ax.set_ylabel("Motif")
    plt.yticks(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cross_level_survival.png", dpi=150)
    plt.close(fig)
    print("  Saved cross_level_survival.png")

    # Find motifs that survive to full_reduce or teleport_reduce
    survivors = []
    fr_idx = levels.index("full_reduce") if "full_reduce" in levels else -1
    for i, mid in enumerate(motif_ids):
        if fr_idx >= 0 and survival[i, fr_idx] > 0:
            survivors.append(mid)

    return {"survivors_at_full_reduce": survivors, "n_representatives": n_reps}


# ── 4e. Surprise Cross-family Relatives ──────────────────────────────


def analysis_surprises(freq_df: pd.DataFrame) -> list[dict]:
    """Find unexpectedly similar algorithms from different families."""
    instances = list(freq_df.index)
    mat = freq_df.fillna(0).values

    surprises = []
    for i in range(len(instances)):
        for j in range(i + 1, len(instances)):
            fam_i = _get_family(instances[i])
            fam_j = _get_family(instances[j])
            if fam_i == fam_j:
                continue  # Only cross-family

            vi, vj = mat[i], mat[j]
            norm_i = np.linalg.norm(vi)
            norm_j = np.linalg.norm(vj)
            if norm_i < 1e-10 or norm_j < 1e-10:
                continue
            sim = float(np.dot(vi, vj) / (norm_i * norm_j))
            if sim > 0.7:
                surprises.append(
                    {
                        "algo_a": instances[i],
                        "family_a": fam_i,
                        "algo_b": instances[j],
                        "family_b": fam_j,
                        "cosine_similarity": round(sim, 4),
                    }
                )

    surprises.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    return surprises[:30]  # Top 30


# ── 4f. Coverage Landscape ──────────────────────────────────────────


def analysis_coverage(
    corpus: dict,
    motifs: list[MotifPattern],
) -> dict:
    """Greedy set-cover decomposition; box-plot by family."""
    results = decompose_across_corpus(
        corpus, motifs, target_level="spider_fused"
    )

    coverage_data = []
    for algo_name, dr in results.items():
        base = algo_name.rsplit("_q", 1)[0]
        fam = ALGORITHM_FAMILY_MAP.get(base, "unknown")
        coverage_data.append(
            {"algorithm": algo_name, "family": fam, "coverage": dr.coverage_ratio}
        )

    cov_df = pd.DataFrame(coverage_data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot by family
    if not cov_df.empty:
        families_present = sorted(cov_df["family"].unique())
        box_data = [
            cov_df[cov_df["family"] == f]["coverage"].values for f in families_present
        ]
        bp = axes[0].boxplot(
            box_data,
            labels=families_present,
            patch_artist=True,
        )
        for patch, fam in zip(bp["boxes"], families_present):
            patch.set_facecolor(FAMILY_COLOURS.get(fam, "#999999"))
        axes[0].set_ylabel("Coverage Ratio")
        axes[0].set_title("Motif Coverage by Algorithm Family")
        axes[0].tick_params(axis="x", rotation=45)

        # Histogram
        axes[1].hist(cov_df["coverage"], bins=20, color="#1f77b4", edgecolor="white")
        axes[1].set_xlabel("Coverage Ratio")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Coverage Distribution")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "coverage_landscape.png", dpi=150)
    plt.close(fig)
    print("  Saved coverage_landscape.png")

    avg_by_family = {}
    if not cov_df.empty:
        for fam in sorted(cov_df["family"].unique()):
            avg_by_family[fam] = round(
                float(cov_df[cov_df["family"] == fam]["coverage"].mean()), 4
            )

    return {
        "average_coverage_by_family": avg_by_family,
        "overall_mean_coverage": round(float(cov_df["coverage"].mean()), 4)
        if not cov_df.empty
        else 0,
        "n_algorithms": len(cov_df),
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Report
# ═══════════════════════════════════════════════════════════════════════


def generate_report(
    results: dict,
    counts_df: pd.DataFrame,
    freq_df: pd.DataFrame,
    motifs: list[MotifPattern],
) -> str:
    """Format structured findings as plain text."""
    lines = [
        "=" * 70,
        "QUANTUM ALGORITHM PHYLOGENY — ZX MOTIF FINGERPRINT ANALYSIS",
        "=" * 70,
        "",
        f"Corpus: {counts_df.shape[0]} algorithm instances",
        f"Motif library: {counts_df.shape[1]} motifs ({len(EXTENDED_MOTIFS)} hand-crafted + discovered)",
        "",
    ]

    # Universality
    uni = results.get("universality", {})
    cats = uni.get("category_counts", {})
    lines.append("─── Motif Universality ───")
    for cat in ["universal", "common", "specific", "unused"]:
        lines.append(f"  {cat:12s}: {cats.get(cat, 0)}")
    lines.append("")

    # Universal motifs detail
    if uni.get("motif_stats"):
        universal_motifs = [
            m for m in uni["motif_stats"] if m["category"] == "universal"
        ]
        if universal_motifs:
            lines.append("Universal motifs (present in ≥80% of families):")
            for m in universal_motifs:
                lines.append(
                    f"  {m['motif_id']:30s}  ({m['n_families']}/{uni['n_families']} families)"
                )
            lines.append("")

    # PCA
    pca = results.get("pca", {})
    if pca:
        lines.append("─── PCA Variance Explained ───")
        lines.append(
            f"  PC1: {pca.get('variance_explained_pc1', 0):.1%}  "
            f"PC2: {pca.get('variance_explained_pc2', 0):.1%}"
        )
        lines.append("")

    # Cross-level survival
    surv = results.get("cross_level_survival", {})
    if surv.get("survivors_at_full_reduce"):
        lines.append("─── Motifs Surviving Full Reduction ───")
        for mid in surv["survivors_at_full_reduce"]:
            lines.append(f"  {mid}")
        lines.append("")

    # Surprises
    surprises = results.get("surprises", [])
    lines.append(f"─── Cross-family Structural Relatives ({len(surprises)} found) ───")
    if surprises:
        for s in surprises[:15]:
            lines.append(
                f"  {s['algo_a']:30s} ({s['family_a']:18s}) <-> "
                f"{s['algo_b']:30s} ({s['family_b']:18s})  "
                f"sim={s['cosine_similarity']:.3f}"
            )
    else:
        lines.append("  None found above threshold (0.7)")
    lines.append("")

    # Coverage
    cov = results.get("coverage", {})
    lines.append("─── Motif Coverage ───")
    lines.append(f"  Overall mean coverage: {cov.get('overall_mean_coverage', 0):.1%}")
    if cov.get("average_coverage_by_family"):
        for fam, avg in sorted(
            cov["average_coverage_by_family"].items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {fam:20s}: {avg:.1%}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("Outputs saved to: scripts/output/")
    lines.append("=" * 70)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()

    # Phase 1
    print("Phase 1: Building corpus...")
    corpus = build_corpus()
    n_instances = len({name for name, _ in corpus})
    n_graphs = len(corpus)
    print(f"  {n_instances} instances, {n_graphs} graphs ({time.time() - t0:.1f}s)")

    # Phase 2
    print("\nPhase 2: Discovering motifs...")
    t1 = time.time()
    motifs = discover_motifs(corpus)
    print(f"  {len(motifs)} total motifs after dedup ({time.time() - t1:.1f}s)")

    # Phase 3
    print("\nPhase 3: Building fingerprint matrix...")
    t2 = time.time()
    counts_df, freq_df = build_fingerprint_matrix(corpus, motifs)
    counts_df.to_csv(OUTPUT_DIR / "fingerprint_counts.csv")
    freq_df.to_csv(OUTPUT_DIR / "fingerprint_frequencies.csv")
    print(
        f"  Matrix shape: {counts_df.shape} ({time.time() - t2:.1f}s)"
    )

    # Phase 4
    print("\nPhase 4: Running analyses...")
    results: dict = {}

    print("  4a. Phylogenetic clustering...")
    results["phylogeny"] = analysis_phylogeny(freq_df)

    print("  4b. PCA...")
    results["pca"] = analysis_pca(freq_df)

    print("  4c. Universality spectrum...")
    results["universality"] = analysis_universality(counts_df)

    print("  4d. Cross-level survival...")
    t4d = time.time()
    results["cross_level_survival"] = analysis_cross_level_survival(corpus, motifs)
    print(f"      ({time.time() - t4d:.1f}s)")

    print("  4e. Surprise relatives...")
    results["surprises"] = analysis_surprises(freq_df)
    print(f"      {len(results['surprises'])} cross-family surprises found")

    print("  4f. Coverage landscape...")
    t4f = time.time()
    results["coverage"] = analysis_coverage(corpus, motifs)
    print(f"      ({time.time() - t4f:.1f}s)")

    # Phase 5
    print("\nPhase 5: Generating report...")
    report = generate_report(results, counts_df, freq_df, motifs)
    print(report)

    # Save outputs
    (OUTPUT_DIR / "report.txt").write_text(report)

    # Make results JSON-serialisable
    json_results = {}
    for k, v in results.items():
        if isinstance(v, (dict, list)):
            json_results[k] = v
        else:
            json_results[k] = str(v)

    # Strip non-serialisable items from universality motif_stats
    if "universality" in json_results and "motif_stats" in json_results["universality"]:
        for ms in json_results["universality"]["motif_stats"]:
            ms["families"] = list(ms.get("families", []))

    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(json_results, indent=2, default=str)
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. All outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
