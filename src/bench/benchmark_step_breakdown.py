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
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rl.env_utils import make_fetch_env
from src.bench.timing import profile_rollout_step_and_breakdown, get_machine_state


def run_single_step_breakdown(n_iter: int = 500) -> dict:
    env = make_fetch_env("FetchReach-v4", render_mode="rgb_array")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession("models/fetch_reach_actor_int8.onnx", opts, providers=["CPUExecutionProvider"])

    res = profile_rollout_step_and_breakdown(env, session, n_steps=n_iter, n_warmup=50)
    env.close()
    return res


def benchmark_step_breakdown_5x():
    print("=== Running Step Breakdown Benchmark 5x ===", flush=True)
    runs = []
    total_steps_ms = []
    physics_ms = []
    wrapper_ms = []
    onnx_ms = []
    obs_ms = []

    for run_idx in range(1, 6):
        print(f"  Step breakdown run {run_idx}/5...", flush=True)
        res = run_single_step_breakdown(n_iter=500)
        runs.append(res)

        total_steps_ms.append(res["closed_loop_step_ms"]["median_ms"])
        physics_ms.append(res["pure_mujoco_physics_ms"]["median_ms"])
        wrapper_ms.append(res["gymnasium_wrapper_ms"]["median_ms"])
        onnx_ms.append(res["onnx_inference_ms"]["median_ms"])
        obs_ms.append(res["obs_formatting_ms"]["median_ms"])

        with open(f"results/runs/step_breakdown_run_{run_idx}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

    med_total = round(float(np.median(total_steps_ms)), 3)
    med_phys = round(float(np.median(physics_ms)), 3)
    med_wrap = round(float(np.median(wrapper_ms)), 3)
    med_onnx = round(float(np.median(onnx_ms)), 3)
    med_obs = round(float(np.median(obs_ms)), 3)

    sum_meds = round(med_phys + med_wrap + med_onnx + med_obs, 3)
    diff_pct = round(abs(sum_meds - med_total) / med_total * 100, 1)

    pct_phys = round(med_phys / med_total * 100, 1)
    pct_wrap = round(med_wrap / med_total * 100, 1)
    pct_onnx = round(med_onnx / med_total * 100, 1)
    pct_obs = round(med_obs / med_total * 100, 1)

    result = {
        "n_runs": 5,
        "n_iter_per_run": 500,
        "timing_method": "src/bench/timing.py: profile_rollout_step_and_breakdown",
        "definitions": {
            "physics_step": "Exactly 20 substeps of mujoco.mj_step(model, data). No rendering, no wrappers.",
            "wrapper_overhead": "Gymnasium env.step() duration outside raw physics (obs dict, action setup, reward).",
            "policy_inference": "Single-thread pinned ONNX INT8 actor session.run().",
            "obs_formatting": "Concatenation of ob, ag, dg into float32 (1, 16) array.",
            "closed_loop_step": "End-to-end execution of obs formatting + ONNX inference + env.step().",
        },
        "machine_state": get_machine_state(),
        "gl_backend": os.environ.get("MUJOCO_GL", "glfw"),
        "total_closed_loop_step_ms": {
            "median": med_total,
            "p50": med_total,
            "mean": round(float(np.mean(total_steps_ms)), 3),
            "min": round(float(np.min(total_steps_ms)), 3),
            "max": round(float(np.max(total_steps_ms)), 3),
            "range": f"[{np.min(total_steps_ms):.3f}, {np.max(total_steps_ms):.3f}]",
        },
        "stages": {
            "mujoco_physics_20_substeps_ms": {
                "median": med_phys,
                "p50": med_phys,
                "range": f"[{np.min(physics_ms):.3f}, {np.max(physics_ms):.3f}]",
                "pct_of_total": pct_phys,
            },
            "gymnasium_env_wrapper_and_obs_building_ms": {
                "median": med_wrap,
                "p50": med_wrap,
                "range": f"[{np.min(wrapper_ms):.3f}, {np.max(wrapper_ms):.3f}]",
                "pct_of_total": pct_wrap,
            },
            "onnx_policy_inference_ms": {
                "median": med_onnx,
                "p50": med_onnx,
                "range": f"[{np.min(onnx_ms):.3f}, {np.max(onnx_ms):.3f}]",
                "pct_of_total": pct_onnx,
            },
            "obs_flattening_and_conversion_ms": {
                "median": med_obs,
                "p50": med_obs,
                "range": f"[{np.min(obs_ms):.3f}, {np.max(obs_ms):.3f}]",
                "pct_of_total": pct_obs,
            },
        },
        "closure": {
            "sum_of_medians_ms": sum_meds,
            "diff_from_closed_loop_pct": diff_pct,
        },
        "individual_runs": runs,
    }

    with open("results/step_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    benchmark_step_breakdown_5x()
