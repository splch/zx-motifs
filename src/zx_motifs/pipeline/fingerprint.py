"""Corpus building, motif discovery, and fingerprint matrix construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from zx_motifs.algorithms.registry import REGISTRY
from zx_motifs.pipeline.converter import convert_at_all_levels
from zx_motifs.pipeline.featurizer import pyzx_to_networkx
from zx_motifs.pipeline.matcher import MotifPattern, find_motif_in_graph
from zx_motifs.pipeline.motif_generators import (
    EXTENDED_MOTIFS,
    _is_isomorphic,
    find_neighborhood_motifs,
    find_recurring_subgraphs,
    wl_hash,
)


def build_corpus(max_qubits: int = 5) -> dict[tuple[str, str], "nx.Graph"]:
    """Convert all REGISTRY algorithms to NetworkX at all simplification levels.

    Parameters
    ----------
    max_qubits : int
        Cap qubit count to keep VF2 matching tractable.

    Returns
    -------
    dict mapping (instance_name, level_value) to NetworkX graph.
    """
    import networkx as nx

    corpus: dict[tuple[str, str], nx.Graph] = {}
    errors: list[str] = []

    for entry in tqdm(REGISTRY, desc="Building corpus", unit="algo"):
        lo, hi = entry.qubit_range
        qubit_sizes = list(range(lo, min(hi, max_qubits) + 1))

        for n in qubit_sizes:
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


def discover_motifs(corpus: dict) -> list[MotifPattern]:
    """Combine hand-crafted, bottom-up, and neighbourhood motifs; deduplicate.

    Parameters
    ----------
    corpus : dict
        Mapping from (instance_name, level) to NetworkX graph.

    Returns
    -------
    list of deduplicated MotifPattern objects.
    """
    all_motifs: list[MotifPattern] = list(EXTENDED_MOTIFS)
    seen_hashes: dict[str, int] = {}

    for i, mp in enumerate(all_motifs):
        h = wl_hash(mp.graph)
        seen_hashes[h] = i

    def _add_if_novel(candidates: list[MotifPattern]) -> int:
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


def build_fingerprint_matrix(
    corpus: dict,
    motifs: list[MotifPattern],
    target_level: str = "spider_fused",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count motif occurrences per algorithm instance.

    Parameters
    ----------
    corpus : dict
        Mapping from (instance_name, level) to NetworkX graph.
    motifs : list[MotifPattern]
        Motif library to match against.
    target_level : str
        Simplification level to fingerprint at.

    Returns
    -------
    (counts_df, freq_df) where freq_df is L1-normalised per row.
    """
    instances = sorted(
        {name for (name, level) in corpus if level == target_level}
    )
    motif_ids = [mp.motif_id for mp in motifs]

    counts = np.zeros((len(instances), len(motifs)), dtype=int)

    for i, inst in enumerate(tqdm(instances, desc="Fingerprinting", unit="inst")):
        key = (inst, target_level)
        if key not in corpus:
            continue
        host = corpus[key]
        for j, mp in enumerate(motifs):
            matches = find_motif_in_graph(mp.graph, host, max_matches=20)
            counts[i, j] = len(matches)

    counts_df = pd.DataFrame(counts, index=instances, columns=motif_ids)

    row_sums = counts_df.sum(axis=1).replace(0, 1)
    freq_df = counts_df.div(row_sums, axis=0)

    return counts_df, freq_df
