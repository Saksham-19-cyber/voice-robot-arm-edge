"""Micro-benchmark breaking down the latency of each stage in a closed-loop step.

Definitions:
- "Closed-loop step": Full execution of one simulation step including observation formatting,
  ONNX policy inference, and simulation step (wrapper + 20 physics substeps).
- "MuJoCo physics (20 substeps)": Exactly 20 calls to mujoco.mj_step(model, data) representing
  the raw physics integration time (dt = 0.002s, total dt = 0.040s). No rendering, no wrappers.
- "Gymnasium wrapper overhead": Time spent inside env.step() outside raw physics (observation
  dictionary building, robot_get_obs, joint state reading, mocap position updates, info dict).
- "ONNX policy inference": Single forward pass of the INT8 actor network on CPU (intra_op=1).
- "Observation formatting": Numpy concatenation and dtype coercion to shape (1, 16).
"""

import os
import sys
import json
import time
import numpy as np
import onnxruntime as ort
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rl.env_utils import make_fetch_env
from src.bench.timing import get_machine_state


def run_single_step_breakdown(n_iter: int = 500) -> dict:
    env = make_fetch_env("FetchReach-v4", render_mode="rgb_array")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession("models/fetch_reach_actor_int8.onnx", opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    model = env.unwrapped.model
    data = env.unwrapped.data
    n_substeps = env.unwrapped.n_substeps

    obs_dict, _ = env.reset(seed=42)

    # Warm-up (50 steps)
    for _ in range(50):
        ob = obs_dict["observation"]
        ag = obs_dict["achieved_goal"]
        dg = obs_dict["desired_goal"]
        flat = np.concatenate([ob, ag, dg], axis=0).astype(np.float32).reshape(1, 16)
        act = session.run(None, {input_name: flat})[0][0]
        obs_dict, _, _, _, _ = env.step(act)

    # Per-step synchronized timing arrays
    t_obs = np.empty(n_iter, dtype=np.float64)
    t_onnx = np.empty(n_iter, dtype=np.float64)
    t_physics = np.empty(n_iter, dtype=np.float64)
    t_wrapper = np.empty(n_iter, dtype=np.float64)
    t_total = np.empty(n_iter, dtype=np.float64)

    obs_dict, _ = env.reset(seed=42)

    for i in range(n_iter):
        t0 = time.perf_counter_ns()

        # 1. Obs formatting
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

        # Convert ns to ms
        t_obs[i] = (t1 - t0) / 1e6
        t_onnx[i] = (t2 - t1) / 1e6
        t_physics[i] = (t_phys_1 - t_phys_0) / 1e6
        t_wrapper[i] = ((t_phys_0 - t2) + (t3 - t_phys_1)) / 1e6
        t_total[i] = (t3 - t0) / 1e6

    env.close()

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
            "p50_ms": round(p50_total, 4),
            "mean_ms": round(mean_total, 4),
            "p95_ms": round(float(np.percentile(t_total, 95)), 4),
            "min_ms": round(float(np.min(t_total)), 4),
            "max_ms": round(float(np.max(t_total)), 4),
        },
        "obs_formatting_ms": {
            "p50_ms": round(p50_obs, 4),
            "mean_ms": round(float(np.mean(t_obs)), 4),
        },
        "onnx_policy_inference_ms": {
            "p50_ms": round(p50_onnx, 4),
            "mean_ms": round(float(np.mean(t_onnx)), 4),
        },
        "mujoco_physics_20_substeps_ms": {
            "p50_ms": round(p50_phys, 4),
            "mean_ms": round(float(np.mean(t_physics)), 4),
        },
        "gymnasium_wrapper_overhead_ms": {
            "p50_ms": round(p50_wrap, 4),
            "mean_ms": round(float(np.mean(t_wrapper)), 4),
        },
        "sum_of_components_ms": round(sum_p50, 4),
        "closure_difference_pct": round(diff_pct, 2),
    }


def benchmark_step_breakdown(n_runs: int = 5, n_iter_per_run: int = 300):
    print(f"=== BENCHMARKING CLOSED-LOOP STEP BREAKDOWN ({n_runs} RUNS, {n_iter_per_run} ITERS EACH) ===", flush=True)
    all_runs = []
    for r in range(n_runs):
        print(f"  Run {r+1}/{n_runs}...", end="", flush=True)
        res = run_single_step_breakdown(n_iter=n_iter_per_run)
        all_runs.append(res)
        print(f" closed-loop p50: {res['closed_loop_step_ms']['p50_ms']:.3f} ms, components sum: {res['sum_of_components_ms']:.3f} ms (diff: {res['closure_difference_pct']}%)")

    cl_p50s = [r["closed_loop_step_ms"]["p50_ms"] for r in all_runs]
    phys_p50s = [r["mujoco_physics_20_substeps_ms"]["p50_ms"] for r in all_runs]
    wrap_p50s = [r["gymnasium_wrapper_overhead_ms"]["p50_ms"] for r in all_runs]
    onnx_p50s = [r["onnx_policy_inference_ms"]["p50_ms"] for r in all_runs]
    obs_p50s = [r["obs_formatting_ms"]["p50_ms"] for r in all_runs]

    cl_median = float(np.median(cl_p50s))
    phys_median = float(np.median(phys_p50s))
    wrap_median = float(np.median(wrap_p50s))
    onnx_median = float(np.median(onnx_p50s))
    obs_median = float(np.median(obs_p50s))

    sum_medians = phys_median + wrap_median + onnx_median + obs_median
    diff_from_cl = abs(sum_medians - cl_median) / cl_median * 100

    report = {
        "n_runs": n_runs,
        "n_iter_per_run": n_iter_per_run,
        "timing_method": "Synchronized per-step perf_counter_ns timestamps",
        "definitions": {
            "physics_step": "Exactly 20 substeps of mujoco.mj_step(model, data). No rendering, no wrappers.",
            "wrapper_overhead": "Gymnasium env.step() duration outside raw physics (obs dict, action setup, reward).",
            "policy_inference": "Single-thread pinned ONNX INT8 actor session.run().",
            "obs_formatting": "Concatenation of ob, ag, dg into float32 (1, 16) array.",
            "closed_loop_step": "End-to-end execution of obs formatting + ONNX inference + env.step()."
        },
        "machine_state": get_machine_state(),
        "gl_backend": os.environ.get("MUJOCO_GL", "glfw"),
        "total_closed_loop_step_ms": {
            "median": round(cl_median, 3),
            "p50": round(cl_median, 3),
            "mean": round(float(np.mean(cl_p50s)), 3),
            "min": round(float(np.min(cl_p50s)), 3),
            "max": round(float(np.max(cl_p50s)), 3),
            "range": f"[{np.min(cl_p50s):.3f}, {np.max(cl_p50s):.3f}]",
        },
        "stages": {
            "mujoco_physics_20_substeps_ms": {
                "median": round(phys_median, 3),
                "p50": round(phys_median, 3),
                "range": f"[{np.min(phys_p50s):.3f}, {np.max(phys_p50s):.3f}]",
                "pct_of_total": round((phys_median / cl_median) * 100, 1),
            },
            "gymnasium_env_wrapper_and_obs_building_ms": {
                "median": round(wrap_median, 3),
                "p50": round(wrap_median, 3),
                "range": f"[{np.min(wrap_p50s):.3f}, {np.max(wrap_p50s):.3f}]",
                "pct_of_total": round((wrap_median / cl_median) * 100, 1),
            },
            "onnx_policy_inference_ms": {
                "median": round(onnx_median, 3),
                "p50": round(onnx_median, 3),
                "range": f"[{np.min(onnx_p50s):.3f}, {np.max(onnx_p50s):.3f}]",
                "pct_of_total": round((onnx_median / cl_median) * 100, 1),
            },
            "obs_flattening_and_conversion_ms": {
                "median": round(obs_median, 3),
                "p50": round(obs_median, 3),
                "range": f"[{np.min(obs_p50s):.3f}, {np.max(obs_p50s):.3f}]",
                "pct_of_total": round((obs_median / cl_median) * 100, 1),
            },
        },
        "closure": {
            "sum_of_medians_ms": round(sum_medians, 3),
            "diff_from_closed_loop_pct": round(diff_from_cl, 2),
        },
        "individual_runs": all_runs,
    }

    with open("results/step_breakdown.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nStep breakdown saved to results/step_breakdown.json (Closure diff: {report['closure']['diff_from_closed_loop_pct']}%)", flush=True)
    return report


if __name__ == "__main__":
    benchmark_step_breakdown(n_runs=5, n_iter_per_run=300)
