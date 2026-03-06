"""
Persistent catalog of discovered motifs with metadata,
occurrence statistics, and relationships.
"""
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np
import networkx as nx
from networkx.readwrite import json_graph as _json_graph

from .featurizer import compute_motif_feature_vector, motif_similarity
from .matcher import MotifPattern


@dataclass
class CatalogEntry:
    motif_id: str
    graph_json: dict
    source: str
    family: str
    description: str
    n_nodes: int
    n_edges: int
    node_type_signature: str  # e.g. "Z3_X1"
    edge_type_signature: str  # e.g. "H2_S1"
    total_occurrences: int
    algorithms_found_in: list[str]
    occurrence_by_family: dict[str, int]  # {"oracle": 5, "variational": 3}
    tags: list[str] = field(default_factory=list)
    related_motifs: list[str] = field(default_factory=list)
    feature_vector: list[float] = field(default_factory=list)
    cross_level_info: dict = field(default_factory=dict)


class MotifCatalog:
    def __init__(self, path: str = "motif_library/catalog.json"):
        self.path = path
        self.entries: dict[str, CatalogEntry] = {}
        if os.path.exists(path):
            self.load()

    def add_motif(
        self, motif: MotifPattern, algorithm_family_map: dict[str, str]
    ) -> None:
        """
        Add a MotifPattern (with populated occurrences) to the catalog.
        algorithm_family_map: {algo_base_name: family_string}
        """
        g = motif.graph

        type_counts = Counter(
            g.nodes[n].get("vertex_type", "?") for n in g.nodes()
        )
        node_sig = "_".join(f"{t}{c}" for t, c in sorted(type_counts.items()))

        edge_counts = Counter(
            g.edges[e].get("edge_type", "?") for e in g.edges()
        )
        edge_sig = "_".join(f"{t[0]}{c}" for t, c in sorted(edge_counts.items()))

        # Count occurrences by algorithm family
        algo_counts = Counter(m.host_algorithm for m in motif.occurrences)
        family_counts: Counter[str] = Counter()
        algos_list: list[str] = []
        for algo, _count in algo_counts.items():
            base_algo = algo.rsplit("_q", 1)[0]
            family = algorithm_family_map.get(base_algo, "unknown")
            family_counts[family] += _count
            if algo not in algos_list:
                algos_list.append(algo)

        dominant_family = (
            family_counts.most_common(1)[0][0] if family_counts else "universal"
        )

        fvec = compute_motif_feature_vector(g).tolist()

        entry = CatalogEntry(
            motif_id=motif.motif_id,
            graph_json=_json_graph.node_link_data(g, edges="links"),
            source=motif.source,
            family=dominant_family,
            description=motif.description,
            n_nodes=g.number_of_nodes(),
            n_edges=g.number_of_edges(),
            node_type_signature=node_sig,
            edge_type_signature=edge_sig,
            total_occurrences=len(motif.occurrences),
            algorithms_found_in=algos_list,
            occurrence_by_family=dict(family_counts),
            feature_vector=fvec,
        )

        self.entries[motif.motif_id] = entry

    def find_related(
        self,
        motif_id: str,
        similarity_threshold: float = 0.5,
        structural_weight: float = 0.6,
        cooccurrence_weight: float = 0.4,
    ) -> list[tuple[str, float]]:
        """
        Find related motifs using combined structural + co-occurrence similarity.

        Score = structural_weight * cosine_sim(features) +
                cooccurrence_weight * jaccard(algorithms)
        """
        if motif_id not in self.entries:
            return []

        target = self.entries[motif_id]
        target_vec = np.array(target.feature_vector) if target.feature_vector else None
        target_algos = set(target.algorithms_found_in)
        related = []

        for mid, entry in self.entries.items():
            if mid == motif_id:
                continue

            # Structural similarity
            structural_sim = 0.0
            if target_vec is not None and entry.feature_vector:
                entry_vec = np.array(entry.feature_vector)
                structural_sim = motif_similarity(target_vec, entry_vec)

            # Co-occurrence similarity (Jaccard)
            entry_algos = set(entry.algorithms_found_in)
            cooc_sim = 0.0
            if target_algos and entry_algos:
                cooc_sim = len(target_algos & entry_algos) / len(
                    target_algos | entry_algos
                )

            score = (
                structural_weight * structural_sim
                + cooccurrence_weight * cooc_sim
            )
            if score >= similarity_threshold:
                related.append((mid, score))

        return sorted(related, key=lambda x: -x[1])

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        data = {mid: asdict(entry) for mid, entry in self.entries.items()}
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self) -> None:
        with open(self.path) as f:
            data = json.load(f)
        for mid, edict in data.items():
            self.entries[mid] = CatalogEntry(**edict)

    def summary(self) -> str:
        lines = [f"Motif Catalog: {len(self.entries)} entries\n"]
        by_family = Counter(e.family for e in self.entries.values())
        for fam, count in by_family.most_common():
            lines.append(f"  {fam}: {count} motifs")
        lines.append("")
        for mid, entry in sorted(
            self.entries.items(), key=lambda x: -x[1].total_occurrences
        ):
            lines.append(
                f"  {mid}: {entry.n_nodes}N/{entry.n_edges}E, "
                f"{entry.total_occurrences} occ, "
                f"family={entry.family}, sig={entry.node_type_signature}"
            )
        return "\n".join(lines)
