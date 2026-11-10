"""Audit closed-loop rollout throughput and raw MuJoCo physics step latency using unified timing harness."""

import os
import sys
import json
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rl.env_utils import make_fetch_env
from src.bench.timing import profile_rollout_step_and_breakdown, get_machine_state


def audit_rollout_fps(n_steps: int = 500) -> dict:
    print("=== AUDITING CLOSED-LOOP ROLLOUT FPS AND MUJOCO PHYSICS LATENCY ===", flush=True)
    env = make_fetch_env("FetchReach-v4", render_mode="rgb_array")

    # 1. Single thread ONNX session
    opts1 = ort.SessionOptions()
    opts1.intra_op_num_threads = 1
    opts1.inter_op_num_threads = 1
    session_st = ort.InferenceSession("models/fetch_reach_actor_int8.onnx", sess_options=opts1, providers=["CPUExecutionProvider"])

    # 2. Default multi-threaded ONNX session
    session_mt = ort.InferenceSession("models/fetch_reach_actor_int8.onnx", providers=["CPUExecutionProvider"])

    # Profile single-thread closed-loop and breakdown
    res_st = profile_rollout_step_and_breakdown(env, session_st, n_steps=n_steps, n_warmup=50)

    # Profile default multi-threaded closed-loop
    res_mt = profile_rollout_step_and_breakdown(env, session_mt, n_steps=n_steps, n_warmup=50)

    env.close()

    report = {
        "timing_method": "src/bench/timing.py: profile_rollout_step_and_breakdown with synchronized perf_counter_ns",
        "definitions": {
            "pure_mujoco_physics_20_substeps": "Exactly 20 calls to mj_step(model, data). No rendering, no wrappers.",
            "wrapper_overhead": "Gymnasium env.step() duration outside raw physics (obs dict, action setup, reward).",
            "closed_loop_rollout": "Sequential rollout of obs prep + ONNX inference + env.step() without reset overhead.",
        },
        "machine_state": get_machine_state(),
        "gl_backend": os.environ.get("MUJOCO_GL", "glfw"),
        "raw_mujoco_physics_20_substeps_ms": res_st["pure_mujoco_physics_ms"],
        "pure_mujoco_step_alone_ms": res_st["pure_mujoco_physics_ms"]["median_ms"],
        "closed_loop_single_thread_ort": {
            "total_steps": n_steps,
            "effective_fps": res_st["closed_loop_step_ms"]["effective_fps"],
            "ms_per_step": res_st["closed_loop_step_ms"]["median_ms"],
        },
        "closed_loop_default_thread_ort": {
            "total_steps": n_steps,
            "effective_fps": res_mt["closed_loop_step_ms"]["effective_fps"],
            "ms_per_step": res_mt["closed_loop_step_ms"]["median_ms"],
        },
        "breakdown": res_st,
    }

    with open("results/rollout_fps_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    audit_rollout_fps()
