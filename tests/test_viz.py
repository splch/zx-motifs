"""Tests for Phase 5: visualization utilities."""
import numpy as np
import pytest

from zx_motifs.pipeline.viz import (
    plot_ablation,
    plot_energy_comparison,
    plot_evolution_history,
    plot_scaling,
    plot_scorecard_comparison,
)


@pytest.fixture(autouse=True)
def _use_agg_backend():
    """Use non-interactive backend for tests."""
    import matplotlib
    matplotlib.use("Agg")


class TestViz:
    def test_evolution_history(self):
        history = [
            {"generation": 0, "best_fitness": -0.5, "mean_fitness": -1.0,
             "best_energy": -1.0, "best_rel_error": 0.1, "alive": 10},
            {"generation": 1, "best_fitness": -0.3, "mean_fitness": -0.7,
             "best_energy": -1.1, "best_rel_error": 0.05, "alive": 12},
        ]
        fig, ax = plot_evolution_history(history)
        assert fig is not None

    def test_scorecard_comparison(self):
        scorecards = [
            {"ansatz": "MAGIC", "layers_passed": 5, "classification": "publishable",
             "layers_killed": 0, "layer_details": {}},
            {"ansatz": "HEA", "layers_passed": 2, "classification": "insufficient",
             "layers_killed": 1, "layer_details": {}},
        ]
        fig, ax = plot_scorecard_comparison(scorecards)
        assert fig is not None

    def test_ablation_plot(self):
        summary = {
            "baseline": {"fitness": -0.3, "relative_error": 0.05, "gate_count": 10},
            "ablations": {
                "pattern_brick": {"fitness": -0.4, "relative_error": 0.1, "delta_fitness": -0.1},
                "layers_3": {"fitness": -0.2, "relative_error": 0.03, "delta_fitness": 0.1},
            },
        }
        fig, ax = plot_ablation(summary)
        assert fig is not None

    def test_scaling_plot(self):
        ns = np.array([4, 6, 8, 10, 12])
        values = 2.0 * ns**1.5
        fig, ax = plot_scaling(ns, values, ylabel="Gate count")
        assert fig is not None

    def test_scaling_with_fits(self):
        from zx_motifs.pipeline.statistics import fit_scaling

        ns = np.array([4, 6, 8, 10, 12])
        values = 2.0 * ns**1.5
        fits = fit_scaling(ns, values)
        fig, ax = plot_scaling(ns, values, fits=fits)
        assert fig is not None

    def test_energy_comparison(self):
        results = {
            "MAGIC": np.array([-1.1, -1.05, -1.12, -1.08]),
            "HEA": np.array([-0.9, -0.85, -0.95, -0.88]),
        }
        fig, ax = plot_energy_comparison(results, exact_energy=-1.137)
        assert fig is not None
