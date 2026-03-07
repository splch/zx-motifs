"""
Statistical analysis pipeline for rigorous ansatz comparison.

Multi-seed runner, multi-optimizer wrapper, effect size computation,
hypothesis testing with corrections, and scaling model fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as scipy_stats


# ── Effect Sizes ───────────────────────────────────────────────────


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size between two samples."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled_std < 1e-15:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta (non-parametric effect size)."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0
    more = sum(1 for xi in x for yj in y if xi > yj)
    less = sum(1 for xi in x for yj in y if xi < yj)
    return float((more - less) / (nx * ny))


def bootstrap_ci(
    data: np.ndarray,
    statistic=np.median,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a statistic."""
    rng = np.random.default_rng(seed)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(float(statistic(sample)))
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(boot_stats, 100 * alpha))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return (lo, hi)


# ── Hypothesis Testing ─────────────────────────────────────────────


@dataclass
class ComparisonResult:
    """Result of comparing two ansatze on a single problem."""

    ansatz_a: str
    ansatz_b: str
    problem: str
    n_samples: int
    mean_a: float
    mean_b: float
    median_a: float
    median_b: float
    std_a: float
    std_b: float
    wilcoxon_p: float
    wilcoxon_stat: float
    cohens_d: float
    cliffs_delta: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    significant: bool  # after correction


def compare_results(
    energies_a: np.ndarray,
    energies_b: np.ndarray,
    ansatz_a: str = "A",
    ansatz_b: str = "B",
    problem: str = "",
    alpha: float = 0.05,
) -> ComparisonResult:
    """Compare two sets of VQE energies with full statistical analysis."""
    # Wilcoxon signed-rank test (paired)
    try:
        stat, p = scipy_stats.wilcoxon(energies_a, energies_b, alternative="two-sided")
    except ValueError:
        stat, p = 0.0, 1.0

    return ComparisonResult(
        ansatz_a=ansatz_a,
        ansatz_b=ansatz_b,
        problem=problem,
        n_samples=len(energies_a),
        mean_a=float(np.mean(energies_a)),
        mean_b=float(np.mean(energies_b)),
        median_a=float(np.median(energies_a)),
        median_b=float(np.median(energies_b)),
        std_a=float(np.std(energies_a, ddof=1)) if len(energies_a) > 1 else 0.0,
        std_b=float(np.std(energies_b, ddof=1)) if len(energies_b) > 1 else 0.0,
        wilcoxon_p=float(p),
        wilcoxon_stat=float(stat),
        cohens_d=cohens_d(energies_a, energies_b),
        cliffs_delta=cliffs_delta(energies_a, energies_b),
        ci_a=bootstrap_ci(energies_a),
        ci_b=bootstrap_ci(energies_b),
        significant=p < alpha,
    )


def bonferroni_correct(
    results: list[ComparisonResult],
    alpha: float = 0.05,
) -> list[ComparisonResult]:
    """Apply Bonferroni correction to multiple comparisons."""
    n = len(results)
    if n == 0:
        return results
    corrected_alpha = alpha / n
    for r in results:
        r.significant = r.wilcoxon_p < corrected_alpha
    return results


# ── Scaling Analysis ───────────────────────────────────────────────


@dataclass
class ScalingFit:
    """Result of fitting a scaling model."""

    model: str  # "power_law" or "exponential"
    params: dict
    r_squared: float
    aic: float


def fit_scaling(
    ns: np.ndarray,
    values: np.ndarray,
) -> list[ScalingFit]:
    """Fit power-law and exponential models to scaling data.

    Returns both fits sorted by AIC (lower = better).
    """
    fits = []
    log_ns = np.log(ns)
    log_vals = np.log(np.abs(values) + 1e-30)

    # Power law: y = a * n^b -> log y = log a + b * log n
    try:
        coeffs = np.polyfit(log_ns, log_vals, 1)
        b, log_a = coeffs
        predicted = np.exp(log_a) * ns**b
        ss_res = np.sum((values - predicted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        k = 2  # two parameters
        n = len(values)
        aic = n * np.log(ss_res / n + 1e-30) + 2 * k
        fits.append(ScalingFit("power_law", {"a": float(np.exp(log_a)), "b": float(b)}, float(r2), float(aic)))
    except Exception:
        pass

    # Exponential: y = a * exp(b * n) -> log y = log a + b * n
    try:
        coeffs = np.polyfit(ns, log_vals, 1)
        b, log_a = coeffs
        predicted = np.exp(log_a) * np.exp(b * ns)
        ss_res = np.sum((values - predicted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        k = 2
        n = len(values)
        aic = n * np.log(ss_res / n + 1e-30) + 2 * k
        fits.append(ScalingFit("exponential", {"a": float(np.exp(log_a)), "b": float(b)}, float(r2), float(aic)))
    except Exception:
        pass

    return sorted(fits, key=lambda f: f.aic)


# ── Multi-Seed Runner ──────────────────────────────────────────────


def run_multi_seed(
    ansatz_fn,
    n_qubits: int,
    hamiltonian: np.ndarray,
    n_seeds: int = 30,
    n_restarts: int = 10,
    maxiter: int = 400,
    optimizers: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Run VQE with multiple seeds and optimizers.

    Returns dict mapping optimizer name to array of best energies.
    """
    from .evaluation import vqe_test

    if optimizers is None:
        optimizers = ["COBYLA"]

    results: dict[str, list[float]] = {opt: [] for opt in optimizers}

    for opt in optimizers:
        for seed in range(42, 42 + n_seeds):
            qc = ansatz_fn(n_qubits)
            result = vqe_test(qc, n_qubits, hamiltonian, n_restarts=n_restarts,
                              maxiter=maxiter, seed=seed)
            results[opt].append(result["best_energy"])

    return {opt: np.array(vals) for opt, vals in results.items()}
