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


def profile_rollout_step_and_breakdown(env, session, n_steps: int = 500, n_warmup: int = 50) -> Dict[str, Any]:
    """Profile closed-loop simulation step and its internal components with synchronized timestamps.
    
    Identical definitions shared across audit_rollout_fps.py and benchmark_step_breakdown.py:
    - Closed-loop step: obs formatting + ONNX inference + wrapper overhead + 20 substeps of mj_step.
    - Pure physics step: exactly 20 substeps of mujoco.mj_step(model, data).
    - Wrapper overhead: Gymnasium action handling, mocap position updates, observation dict building, reward check.
    - ONNX inference: policy forward pass via session.run().
    - Observation formatting: numpy concatenation into (1, 16) float32 array.
    """
    import mujoco
    model = env.unwrapped.model
    data = env.unwrapped.data
    n_substeps = env.unwrapped.n_substeps
    input_name = session.get_inputs()[0].name

    obs_dict, _ = env.reset(seed=42)

    # Warmup
    for w in range(n_warmup):
        if w % 50 == 0:
            obs_dict, _ = env.reset()
        ob = obs_dict["observation"]
        ag = obs_dict["achieved_goal"]
        dg = obs_dict["desired_goal"]
        flat = np.concatenate([ob, ag, dg], axis=0).astype(np.float32).reshape(1, 16)
        act = session.run(None, {input_name: flat})[0][0]
        obs_dict, _, _, _, _ = env.step(act)

    t_obs = np.empty(n_steps, dtype=np.float64)
    t_onnx = np.empty(n_steps, dtype=np.float64)
    t_physics = np.empty(n_steps, dtype=np.float64)
    t_wrapper = np.empty(n_steps, dtype=np.float64)
    t_total = np.empty(n_steps, dtype=np.float64)

    obs_dict, _ = env.reset(seed=42)

    for i in range(n_steps):
        if i > 0 and i % 50 == 0:
            obs_dict, _ = env.reset()

        t0 = time.perf_counter_ns()

        # 1. Observation formatting
        ob = obs_dict["observation"]
        ag = obs_dict["achieved_goal"]
        dg = obs_dict["desired_goal"]
        flat = np.concatenate([ob, ag, dg], axis=0).astype(np.float32).reshape(1, 16)
        t1 = time.perf_counter_ns()

        # 2. ONNX Policy inference
        action = session.run(None, {input_name: flat})[0][0]
        t2 = time.perf_counter_ns()

        # 3. Action application & raw MuJoCo physics (20 substeps)
        env.unwrapped._set_action(action)
        env.unwrapped._step_callback()
        t_phys_0 = time.perf_counter_ns()
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)
        t_phys_1 = time.perf_counter_ns()

        # 4. Observation building & reward computation
        obs_dict = env.unwrapped._get_obs()
        reward = env.unwrapped.compute_reward(obs_dict["achieved_goal"], obs_dict["desired_goal"], {})
        t3 = time.perf_counter_ns()

        t_obs[i] = (t1 - t0) / 1e6
        t_onnx[i] = (t2 - t1) / 1e6
        t_physics[i] = (t_phys_1 - t_phys_0) / 1e6
        t_wrapper[i] = ((t_phys_0 - t2) + (t3 - t_phys_1)) / 1e6
        t_total[i] = (t3 - t0) / 1e6

    p50_total = float(np.percentile(t_total, 50))
    p50_obs = float(np.percentile(t_obs, 50))
    p50_onnx = float(np.percentile(t_onnx, 50))
    p50_phys = float(np.percentile(t_physics, 50))
    p50_wrap = float(np.percentile(t_wrapper, 50))

    mean_total = float(np.mean(t_total))
    sum_p50 = p50_obs + p50_onnx + p50_phys + p50_wrap
    diff_pct = abs(sum_p50 - p50_total) / p50_total * 100

    return {
        "closed_loop_step_ms": {
            "median_ms": round(p50_total, 4),
            "p50_ms": round(p50_total, 4),
            "mean_ms": round(mean_total, 4),
            "p95_ms": round(float(np.percentile(t_total, 95)), 4),
            "p99_ms": round(float(np.percentile(t_total, 99)), 4),
            "min_ms": round(float(np.min(t_total)), 4),
            "max_ms": round(float(np.max(t_total)), 4),
            "effective_fps": round(1000.0 / p50_total, 1),
        },
        "pure_mujoco_physics_ms": {
            "median_ms": round(p50_phys, 4),
            "p50_ms": round(p50_phys, 4),
            "mean_ms": round(float(np.mean(t_physics)), 4),
            "p95_ms": round(float(np.percentile(t_physics, 95)), 4),
            "min_ms": round(float(np.min(t_physics)), 4),
            "max_ms": round(float(np.max(t_physics)), 4),
            "pct_of_total": round(p50_phys / p50_total * 100, 1),
        },
        "gymnasium_wrapper_ms": {
            "median_ms": round(p50_wrap, 4),
            "p50_ms": round(p50_wrap, 4),
            "mean_ms": round(float(np.mean(t_wrapper)), 4),
            "pct_of_total": round(p50_wrap / p50_total * 100, 1),
        },
        "onnx_inference_ms": {
            "median_ms": round(p50_onnx, 4),
            "p50_ms": round(p50_onnx, 4),
            "mean_ms": round(float(np.mean(t_onnx)), 4),
            "pct_of_total": round(p50_onnx / p50_total * 100, 1),
        },
        "obs_formatting_ms": {
            "median_ms": round(p50_obs, 4),
            "p50_ms": round(p50_obs, 4),
            "mean_ms": round(float(np.mean(t_obs)), 4),
            "pct_of_total": round(p50_obs / p50_total * 100, 1),
        },
        "closure": {
            "sum_of_medians_ms": round(sum_p50, 4),
            "diff_from_closed_loop_pct": round(diff_pct, 2),
        },
        "n_steps": n_steps,
    }

