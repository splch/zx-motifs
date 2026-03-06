"""Corpus building, motif discovery, and fingerprint matrix construction."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from zx_motifs.algorithms.registry import REGISTRY
from zx_motifs.config import CONFIG
from zx_motifs.pipeline.converter import convert_at_all_levels
from zx_motifs.pipeline.featurizer import pyzx_to_networkx
from zx_motifs.pipeline.matcher import MotifPattern, find_motif_in_graph
from zx_motifs.pipeline.motif_generators import (
    EXTENDED_MOTIFS,
    is_isomorphic,
    find_neighborhood_motifs,
    find_recurring_subgraphs,
    wl_hash,
)


def _convert_instance(
    generator,
    name: str,
    n_qubits: int,
) -> tuple[str, dict[str, nx.Graph]] | tuple[str, str]:
    """Convert a single algorithm instance to NetworkX graphs at all levels.

    Top-level function so it is picklable by ProcessPoolExecutor.

    Returns (instance_name, {level: graph}) on success,
    or (instance_name, error_message) on failure.
    """
    instance = f"{name}_q{n_qubits}"
    try:
        qc = generator(n_qubits)
        snapshots = convert_at_all_levels(qc, instance)
        graphs = {}
        for snap in snapshots:
            nxg = pyzx_to_networkx(snap.graph, coarsen_phases=True)
            graphs[snap.level.value] = nxg
        return (instance, graphs)
    except Exception as e:
        return (instance, str(e))


def build_corpus(
    max_qubits: int = CONFIG.max_qubits,
    max_workers: int | None = None,
) -> dict[tuple[str, str], nx.Graph]:
    """Convert all REGISTRY algorithms to NetworkX at all simplification levels.

    Parameters
    ----------
    max_qubits : int
        Cap qubit count to keep VF2 matching tractable.
    max_workers : int or None
        Number of parallel workers. None uses all available CPUs.
        Use 1 for sequential execution.

    Returns
    -------
    dict mapping (instance_name, level_value) to NetworkX graph.
    """
    # Build list of (generator, name, n_qubits) tasks
    tasks = []
    for entry in REGISTRY:
        lo, hi = entry.qubit_range
        effective_hi = max_qubits if hi is None else min(hi, max_qubits)
        for n in range(lo, effective_hi + 1):
            tasks.append((entry.generator, entry.name, n))

    corpus: dict[tuple[str, str], nx.Graph] = {}
    errors: list[str] = []

    if max_workers == 1:
        for generator, name, n in tqdm(tasks, desc="Building corpus", unit="inst"):
            result = _convert_instance(generator, name, n)
            instance, payload = result
            if isinstance(payload, str):
                errors.append(f"{instance}: {payload}")
            else:
                for level, nxg in payload.items():
                    corpus[(instance, level)] = nxg
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_convert_instance, gen, name, n): (name, n)
                for gen, name, n in tasks
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Building corpus",
                unit="inst",
            ):
                instance, payload = future.result()
                if isinstance(payload, str):
                    errors.append(f"{instance}: {payload}")
                else:
                    for level, nxg in payload.items():
                        corpus[(instance, level)] = nxg

    if errors:
        print(f"  Skipped {len(errors)} instances due to errors:")
        for err in errors:
            print(f"    {err}")

    return corpus


def discover_motifs(
    corpus: dict,
    max_workers: int | None = None,
) -> list[MotifPattern]:
    """Combine hand-crafted, bottom-up, and neighbourhood motifs; deduplicate.

    Parameters
    ----------
    corpus : dict
        Mapping from (instance_name, level) to NetworkX graph.
    max_workers : int or None
        Number of parallel workers for discovery. None uses all available
        CPUs. Use 1 for sequential execution.

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
                if is_isomorphic(mp.graph, existing.graph):
                    continue
            seen_hashes[h] = len(all_motifs)
            all_motifs.append(mp)
            added += 1
        return added

    try:
        bottom_up = find_recurring_subgraphs(
            corpus,
            target_level="spider_fused",
            min_size=CONFIG.min_motif_size,
            max_size=CONFIG.max_motif_size,
            max_workers=max_workers,
        )
        n_bu = _add_if_novel(bottom_up)
        print(f"  Bottom-up: {len(bottom_up)} found, {n_bu} novel")
    except Exception as e:
        print(f"  Bottom-up failed: {e}")

    try:
        neighbourhood = find_neighborhood_motifs(
            corpus,
            target_level="spider_fused",
            radius=CONFIG.neighbourhood_radius,
            max_workers=max_workers,
        )
        n_nb = _add_if_novel(neighbourhood)
        print(f"  Neighbourhood: {len(neighbourhood)} found, {n_nb} novel")
    except Exception as e:
        print(f"  Neighbourhood failed: {e}")

    return all_motifs


def _fingerprint_instance(
    host: nx.Graph,
    motif_graphs: list[nx.Graph],
    max_matches: int,
) -> list[int]:
    """Count motif occurrences in a single host graph.

    Top-level function so it is picklable by ProcessPoolExecutor.

    Returns list of match counts, one per motif.
    """
    return [
        len(find_motif_in_graph(mg, host, max_matches=max_matches))
        for mg in motif_graphs
    ]


def build_fingerprint_matrix(
    corpus: dict,
    motifs: list[MotifPattern],
    target_level: str = "spider_fused",
    max_workers: int | None = None,
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
    max_workers : int or None
        Number of parallel workers. None uses all available CPUs.
        Use 1 for sequential execution.

    Returns
    -------
    (counts_df, freq_df) where freq_df is L1-normalised per row.
    """
    instances = sorted(
        {name for (name, level) in corpus if level == target_level}
    )
    motif_ids = [mp.motif_id for mp in motifs]
    motif_graphs = [mp.graph for mp in motifs]

    counts = np.zeros((len(instances), len(motifs)), dtype=int)

    if max_workers == 1:
        for i, inst in enumerate(tqdm(instances, desc="Fingerprinting", unit="inst")):
            key = (inst, target_level)
            if key not in corpus:
                continue
            counts[i, :] = _fingerprint_instance(
                corpus[key], motif_graphs, CONFIG.fingerprint_max_matches,
            )
    else:
        inst_to_idx = {inst: i for i, inst in enumerate(instances)}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for inst in instances:
                key = (inst, target_level)
                if key not in corpus:
                    continue
                futures[executor.submit(
                    _fingerprint_instance,
                    corpus[key], motif_graphs,
                    CONFIG.fingerprint_max_matches,
                )] = inst
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Fingerprinting",
                unit="inst",
            ):
                inst = futures[future]
                counts[inst_to_idx[inst], :] = future.result()

    counts_df = pd.DataFrame(counts, index=instances, columns=motif_ids)

    # L1-normalise each row.  Rows with no motif matches remain all-zero
    # rather than being converted to a uniform distribution.
    row_sums = counts_df.sum(axis=1)
    nonzero = row_sums != 0
    freq_df = pd.DataFrame(0.0, index=counts_df.index, columns=counts_df.columns)
    freq_df.loc[nonzero] = counts_df.loc[nonzero].div(row_sums[nonzero], axis=0)

    return counts_df, freq_df
