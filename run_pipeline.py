#!/usr/bin/env python3
"""
ZX-Web Discovery Pipeline — Main Entry Point
==============================================
Run the full 7-step pipeline for discovering novel quantum algorithms
by mining and recombining ZX-diagram sub-structures.

Usage:
    python run_pipeline.py
    python run_pipeline.py --max-candidates 500 --seed 123
"""

import argparse
import json
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zx_discovery.pipeline import run_pipeline
from zx_discovery.reporter import report_to_json, report_to_text


def main():
    parser = argparse.ArgumentParser(
        description="ZX-Web Discovery Pipeline for Quantum Algorithm Innovation"
    )
    parser.add_argument("--max-subgraph-size", type=int, default=5,
                        help="Largest ZX-Web motif size (default: 5)")
    parser.add_argument("--min-frequency", type=int, default=2,
                        help="Min distinct-algorithm frequency (default: 2)")
    parser.add_argument("--max-candidates", type=int, default=150,
                        help="Number of candidates to compose (default: 150)")
    parser.add_argument("--web-combo-size", type=int, default=3,
                        help="Webs per candidate (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--min-gates", type=int, default=3,
                        help="Min gate count filter (default: 3)")
    parser.add_argument("--max-qubits", type=int, default=8,
                        help="Max qubit width (default: 8)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Path to save JSON report")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    report = run_pipeline(
        max_subgraph_size=args.max_subgraph_size,
        min_frequency=args.min_frequency,
        max_candidates=args.max_candidates,
        web_combo_size=args.web_combo_size,
        seed=args.seed,
        min_gates=args.min_gates,
        max_qubits=args.max_qubits,
        verbose=not args.quiet,
    )

    # Save JSON report
    json_path = args.output_json or "pipeline_report.json"
    with open(json_path, "w") as f:
        f.write(report_to_json(report))
    print(f"\nJSON report saved to: {json_path}")

    return 0 if report.n_outperformers >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
