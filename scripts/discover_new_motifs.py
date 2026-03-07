#!/usr/bin/env python3
"""End-to-end motif discovery pipeline.

Builds the full algorithm corpus, runs bottom-up + neighbourhood discovery,
selects the best novel motifs by algorithm spread, and outputs them as JSON.
"""
from __future__ import annotations

import json
import os
import re
import sys

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from zx_motifs.motifs import MOTIF_REGISTRY
from zx_motifs.pipeline.fingerprint import build_corpus, discover_motifs
from zx_motifs.pipeline.matcher import find_motif_in_graph
from zx_motifs.pipeline.motif_generators import is_isomorphic, wl_hash


def motif_to_json(motif, motif_id: str | None = None) -> dict:
    """Convert a MotifPattern's graph to a library-ready JSON dict."""
    g = motif.graph
    old_to_new = {old: new for new, old in enumerate(sorted(g.nodes()))}
    nodes = []
    for old_id in sorted(g.nodes()):
        d = g.nodes[old_id]
        nodes.append({
            "id": old_to_new[old_id],
            "vertex_type": d.get("vertex_type", "Z"),
            "phase_class": d.get("phase_class", "zero"),
            "is_boundary": False,
        })
    links = []
    for u, v, d in sorted(g.edges(data=True), key=lambda e: (e[0], e[1])):
        links.append({
            "source": old_to_new[u],
            "target": old_to_new[v],
            "edge_type": d.get("edge_type", "SIMPLE"),
        })
    return {
        "motif_id": motif_id or motif.motif_id,
        "description": motif.description,
        "source": motif.source or "data_driven",
        "tags": [],
        "graph": {"nodes": nodes, "links": links},
    }


def is_novel(motif, existing_motifs) -> bool:
    """Check whether motif is not isomorphic to any existing library motif."""
    for existing in existing_motifs:
        if existing.graph.number_of_nodes() != motif.graph.number_of_nodes():
            continue
        if existing.graph.number_of_edges() != motif.graph.number_of_edges():
            continue
        if is_isomorphic(motif.graph, existing.graph):
            return False
    return True


def extract_algo_count(description: str) -> int:
    """Extract the number of algorithms from a motif's description."""
    m = re.search(r"found in (\d+) algorithms", description)
    return int(m.group(1)) if m else 0


def main():
    print("=" * 60)
    print("ZX-MOTIFS: End-to-End Motif Discovery Pipeline")
    print("=" * 60)

    # Step 1: Build corpus
    print("\n[1/4] Building corpus from all registered algorithms...")
    corpus = build_corpus(max_qubits=10)
    n_instances = len({name for name, _ in corpus})
    n_graphs = len(corpus)
    print(f"  Corpus: {n_instances} instances, {n_graphs} graphs across all levels")

    # Step 2: Run combined discovery
    print("\n[2/4] Running combined motif discovery...")
    discovered = discover_motifs(corpus)
    print(f"  Total after discovery: {len(discovered)} motifs")

    # Step 3: Filter to novel motifs with high algorithm spread
    print("\n[3/4] Filtering to novel motifs...")
    existing = list(MOTIF_REGISTRY)
    print(f"  Existing library: {len(existing)} motifs")

    novel = [m for m in discovered if is_novel(m, existing)]
    print(f"  Novel motifs (not in library): {len(novel)}")

    # Sort by algorithm count, prefer larger motifs as tiebreaker
    novel.sort(
        key=lambda m: (extract_algo_count(m.description), m.graph.number_of_nodes()),
        reverse=True,
    )

    # Take top candidates
    candidates = novel[:20]
    print(f"  Top candidates: {len(candidates)}")

    # Step 4: Quick validation - match against a sample of spider_fused graphs
    print("\n[4/4] Validating matches on spider_fused corpus...")
    spider_fused = {
        name: g for (name, level), g in corpus.items()
        if level == "spider_fused"
    }
    # Sample up to 100 graphs for validation speed
    sample_names = sorted(spider_fused.keys())[:100]

    validated = []
    for motif in candidates:
        match_count = 0
        algo_set = set()
        for name in sample_names:
            host = spider_fused[name]
            matches = find_motif_in_graph(motif.graph, host, max_matches=5)
            if matches:
                match_count += len(matches)
                algo_set.add(name)
        if match_count >= 2 and len(algo_set) >= 2:
            motif._match_count = match_count
            motif._algo_count = len(algo_set)
            validated.append(motif)
            print(f"  {motif.motif_id}: {match_count} matches in {len(algo_set)} algos")
        else:
            print(f"  {motif.motif_id}: DROPPED ({match_count} matches, {len(algo_set)} algos)")

    # Output
    out_dir = os.path.join(os.path.dirname(__file__), "..", "discovered")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"DISCOVERED {len(validated)} NOVEL MOTIFS")
    print("=" * 60)

    for i, motif in enumerate(validated):
        g = motif.graph
        node_types = {}
        phase_classes = {}
        edge_types = {}
        for _, d in g.nodes(data=True):
            vt = d.get("vertex_type", "?")
            pc = d.get("phase_class", "?")
            node_types[vt] = node_types.get(vt, 0) + 1
            phase_classes[pc] = phase_classes.get(pc, 0) + 1
        for _, _, d in g.edges(data=True):
            et = d.get("edge_type", "?")
            edge_types[et] = edge_types.get(et, 0) + 1

        print(f"\n--- Motif {i+1}: {motif.motif_id} ---")
        print(f"  Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")
        print(f"  Node types: {node_types}")
        print(f"  Phase classes: {phase_classes}")
        print(f"  Edge types: {edge_types}")
        print(f"  Validated: {motif._match_count} matches in {motif._algo_count} algos")
        print(f"  Source: {motif.source}")
        print(f"  Description: {motif.description}")
        print(f"  Graph adjacency:")
        old_to_new = {old: new for new, old in enumerate(sorted(g.nodes()))}
        for n in sorted(g.nodes()):
            nd = g.nodes[n]
            nbrs = sorted(g.neighbors(n))
            edges_str = ", ".join(
                f"{old_to_new[nb]}({g.edges[n,nb].get('edge_type','?')})"
                for nb in nbrs
            )
            print(f"    {old_to_new[n]}: {nd.get('vertex_type','?')}({nd.get('phase_class','?')}) -> [{edges_str}]")

        data = motif_to_json(motif)
        fname = f"{motif.motif_id}.json"
        path = os.path.join(out_dir, fname)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {path}")

    # Save summary
    summary_path = os.path.join(out_dir, "_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Novel motifs discovered: {len(validated)}\n\n")
        for motif in validated:
            g = motif.graph
            old_to_new = {old: new for new, old in enumerate(sorted(g.nodes()))}
            f.write(f"{motif.motif_id}:\n")
            f.write(f"  nodes={g.number_of_nodes()}, edges={g.number_of_edges()}\n")
            f.write(f"  matches={motif._match_count}, algos={motif._algo_count}\n")
            for n in sorted(g.nodes()):
                d = g.nodes[n]
                f.write(f"  node {old_to_new[n]}: vertex_type={d.get('vertex_type')}, phase_class={d.get('phase_class')}\n")
            for u, v, d in sorted(g.edges(data=True)):
                f.write(f"  edge {old_to_new[u]}-{old_to_new[v]}: edge_type={d.get('edge_type')}\n")
            f.write("\n")

    print(f"\nSummary saved: {summary_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
