"""Parallel execution utilities for the pipeline."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def resolve_workers(workers: int | None) -> int:
    """Resolve worker count: None -> os.cpu_count(), 0 or 1 -> sequential."""
    if workers is None:
        return os.cpu_count() or 1
    return max(workers, 1)


def parallel_map(
    fn: Callable[..., T],
    items: list[tuple],
    workers: int | None,
    desc: str = "tasks",
) -> list[T]:
    """Execute fn(*item) for each item in items, using ProcessPoolExecutor.

    If workers <= 1, executes sequentially (useful for debugging).
    Returns results in completion order.  Exceptions in individual tasks
    are logged and the task result is skipped.
    """
    n_workers = resolve_workers(workers)
    results: list[T] = []

    if n_workers <= 1:
        for item_args in items:
            try:
                results.append(fn(*item_args))
            except Exception:
                logger.warning("Failed %s task", desc, exc_info=True)
        return results

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(fn, *item_args): i
            for i, item_args in enumerate(items)
        }
        for future in as_completed(future_to_idx):
            try:
                results.append(future.result())
            except Exception:
                logger.warning("Failed %s task", desc, exc_info=True)

    return results
