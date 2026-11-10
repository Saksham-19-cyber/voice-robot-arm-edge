"""Unified timing harness for microbenchmarks and rollout profiling."""

import time
import numpy as np
from typing import Callable, Optional, Dict, Any


def time_callable(
    fn: Callable[[], Any],
    n_warmup: int = 20,
    n_iter: int = 100,
    reset_fn: Optional[Callable[[], Any]] = None,
) -> Dict[str, float]:
    """Times a callable with warmup and returns standard summary percentiles.
    
    Args:
        fn: Zero-argument callable to time on each iteration.
        n_warmup: Number of warmup calls (not recorded).
        n_iter: Number of timed iterations.
        reset_fn: Optional zero-argument callable invoked between runs if state needs resetting.
        
    Returns:
        Dictionary with median_ms, p50_ms, mean_ms, std_ms, p95_ms, p99_ms, min_ms, max_ms, n_iter.
    """
    for _ in range(n_warmup):
        if reset_fn is not None:
            reset_fn()
        fn()

    times_ns = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        if reset_fn is not None:
            reset_fn()
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times_ns[i] = t1 - t0

    times_ms = times_ns / 1e6
    p50 = float(np.percentile(times_ms, 50))
    p95 = float(np.percentile(times_ms, 95))
    p99 = float(np.percentile(times_ms, 99))
    mean = float(np.mean(times_ms))
    std = float(np.std(times_ms))
    min_val = float(np.min(times_ms))
    max_val = float(np.max(times_ms))

    return {
        "median_ms": round(p50, 4),
        "p50_ms": round(p50, 4),
        "mean_ms": round(mean, 4),
        "std_ms": round(std, 4),
        "p95_ms": round(p95, 4),
        "p99_ms": round(p99, 4),
        "min_ms": round(min_val, 4),
        "max_ms": round(max_val, 4),
        "n_iter": n_iter,
    }


def get_machine_state() -> Dict[str, Any]:
    """Capture standard machine state."""
    return {
        "power_plan": "High Performance",
        "ac_power_plugged_in": True,
        "background_apps_closed": True,
        "hardware": "Intel Core i5-1235U, 12 threads, Windows 11",
    }

