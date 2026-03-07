"""
Visualization utilities for MAGIC results.

Generates matplotlib plots for evolution history, scorecard comparisons,
and scaling analysis. All functions return (fig, ax) tuples for composability.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure
    import matplotlib.axes


def _get_plt():
    """Lazy import matplotlib."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization: pip install matplotlib")


# ── Evolution History ─────────────────────────────────────────────


def plot_evolution_history(
    history: list[dict],
    metric: str = "best_fitness",
    title: str = "Evolution Progress",
    ax=None,
) -> tuple:
    """Plot a metric over generations from evolution history.

    Args:
        history: List of per-generation dicts from EvolutionResult.history.
        metric: Key to plot (e.g. "best_fitness", "best_energy", "best_rel_error").
        title: Plot title.
        ax: Optional matplotlib Axes. Created if None.

    Returns:
        (fig, ax) tuple.
    """
    plt = _get_plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    generations = [h["generation"] for h in history]
    values = [h.get(metric, float("nan")) for h in history]

    ax.plot(generations, values, "o-", linewidth=2, markersize=6)
    ax.set_xlabel("Generation")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if "mean_fitness" in history[0] and metric == "best_fitness":
        means = [h["mean_fitness"] for h in history]
        ax.fill_between(generations, means, values, alpha=0.2, label="best - mean gap")
        ax.plot(generations, means, "--", alpha=0.6, label="mean fitness")
        ax.legend()

    return fig, ax


# ── Scorecard Comparison ──────────────────────────────────────────


def plot_scorecard_comparison(
    scorecards: list[dict],
    title: str = "Scorecard Comparison",
    ax=None,
) -> tuple:
    """Bar chart comparing scorecard layers across ansatze.

    Args:
        scorecards: List of Scorecard.summary() dicts.
        title: Plot title.
        ax: Optional matplotlib Axes.

    Returns:
        (fig, ax) tuple.
    """
    plt = _get_plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    names = [s["ansatz"] for s in scorecards]
    passed = [s["layers_passed"] for s in scorecards]
    classifications = [s["classification"] for s in scorecards]

    colors = {
        "credible_advantage_candidate": "#2ecc71",
        "publishable": "#3498db",
        "interesting": "#f1c40f",
        "insufficient": "#e74c3c",
    }
    bar_colors = [colors.get(c, "#95a5a6") for c in classifications]

    bars = ax.barh(names, passed, color=bar_colors)
    ax.set_xlabel("Layers Passed")
    ax.set_title(title)
    ax.set_xlim(0, 8)

    for bar, cls in zip(bars, classifications):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                cls.replace("_", " "), va="center", fontsize=8)

    return fig, ax


# ── Ablation Impact ───────────────────────────────────────────────


def plot_ablation(
    ablation_summary: dict,
    title: str = "Ablation Study",
    ax=None,
) -> tuple:
    """Bar chart of fitness deltas from ablation study.

    Args:
        ablation_summary: AblationResult.summary() dict.
        title: Plot title.
        ax: Optional matplotlib Axes.

    Returns:
        (fig, ax) tuple.
    """
    plt = _get_plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    ablations = ablation_summary.get("ablations", {})
    names = list(ablations.keys())
    deltas = [ablations[n]["delta_fitness"] for n in names]

    colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]
    ax.barh(names, deltas, color=colors)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("Fitness Change (positive = improvement)")
    ax.set_title(title)

    return fig, ax


# ── Scaling Plot ──────────────────────────────────────────────────


def plot_scaling(
    ns: np.ndarray,
    values: np.ndarray,
    fits: list | None = None,
    ylabel: str = "Value",
    title: str = "Scaling Analysis",
    ax=None,
) -> tuple:
    """Log-log scaling plot with optional fitted curves.

    Args:
        ns: Array of system sizes.
        values: Corresponding measurements.
        fits: Optional list of ScalingFit objects.
        ylabel: Y-axis label.
        title: Plot title.
        ax: Optional matplotlib Axes.

    Returns:
        (fig, ax) tuple.
    """
    plt = _get_plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    ax.plot(ns, values, "ko", markersize=8, label="data")

    if fits:
        ns_smooth = np.linspace(ns.min(), ns.max(), 100)
        for fit in fits:
            if fit.model == "power_law":
                a, b = fit.params["a"], fit.params["b"]
                predicted = a * ns_smooth**b
                label = f"power law: {a:.2g}·n^{b:.2f} (R²={fit.r_squared:.3f})"
            elif fit.model == "exponential":
                a, b = fit.params["a"], fit.params["b"]
                predicted = a * np.exp(b * ns_smooth)
                label = f"exp: {a:.2g}·e^({b:.2f}n) (R²={fit.r_squared:.3f})"
            else:
                continue
            ax.plot(ns_smooth, predicted, "--", label=label, linewidth=1.5)

    ax.set_xlabel("System size (n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    return fig, ax


# ── Energy Comparison ─────────────────────────────────────────────


def plot_energy_comparison(
    results: dict[str, np.ndarray],
    exact_energy: float | None = None,
    title: str = "Energy Distribution",
    ax=None,
) -> tuple:
    """Box plot comparing energy distributions across ansatze.

    Args:
        results: Dict mapping ansatz name to array of energies.
        exact_energy: Optional exact ground state energy (horizontal line).
        title: Plot title.
        ax: Optional matplotlib Axes.

    Returns:
        (fig, ax) tuple.
    """
    plt = _get_plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    names = list(results.keys())
    data = [results[n] for n in names]

    bp = ax.boxplot(data, tick_labels=names, orientation="vertical", patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.6)

    if exact_energy is not None:
        ax.axhline(y=exact_energy, color="red", linestyle="--",
                    label=f"exact = {exact_energy:.4f}")
        ax.legend()

    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)

    return fig, ax
