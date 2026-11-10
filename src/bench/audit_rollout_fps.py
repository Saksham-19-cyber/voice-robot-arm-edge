"""Audit closed-loop rollout throughput and raw MuJoCo physics step latency."""

import os
import sys
import json
import time
import numpy as np
import onnxruntime as ort
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rl.env_utils import make_fetch_env
from src.bench.timing import time_callable, get_machine_state


def audit_rollout_fps():
    print("=== AUDITING CLOSED-LOOP ROLLOUT FPS AND MUJOCO PHYSICS LATENCY ===", flush=True)
    env = make_fetch_env("FetchReach-v4", render_mode="rgb_array")
    
    # 1. Single thread ONNX session
    opts1 = ort.SessionOptions()
    opts1.intra_op_num_threads = 1
    opts1.inter_op_num_threads = 1
    session_st = ort.InferenceSession("models/fetch_reach_actor_int8.onnx", sess_options=opts1, providers=["CPUExecutionProvider"])
    
    # 2. Default multi-threaded ONNX session
    session_mt = ort.InferenceSession("models/fetch_reach_actor_int8.onnx", providers=["CPUExecutionProvider"])
    
    obs_dict, _ = env.reset(seed=42)
    zero_action = np.zeros(4, dtype=np.float32)
    model = env.unwrapped.model
    data = env.unwrapped.data
    n_substeps = env.unwrapped.n_substeps

    # A. Raw MuJoCo physics: exactly 20 substeps of mj_step (no env wrappers, no rendering)
    def time_raw_physics():
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)

    res_physics = time_callable(time_raw_physics, n_warmup=50, n_iter=300)
    
    # B. Isolated env.step() (wrapper + 20 substeps)
    def time_env_step():
        env.step(zero_action)

    res_env_step = time_callable(time_env_step, n_warmup=50, n_iter=300)

    # C. Observation prep alone
    def time_obs_prep():
        ob = obs_dict["observation"]
        ag = obs_dict["achieved_goal"]
        dg = obs_dict["desired_goal"]
        return np.concatenate([ob, ag, dg], axis=0).astype(np.float32).reshape(1, 16)

    res_obs_prep = time_callable(time_obs_prep, n_warmup=50, n_iter=300)
    
    # D. Closed loop rollout with 1-thread ORT (50 episodes x 50 steps = 2500 steps)
    t0 = time.perf_counter()
    total_steps_st = 0
    for ep in range(50):
        obs_dict, _ = env.reset(seed=42 + ep)
        for _ in range(50):
            ob = obs_dict["observation"]
            ag = obs_dict["achieved_goal"]
            dg = obs_dict["desired_goal"]
            obs_flat = np.concatenate([ob, ag, dg], axis=0).astype(np.float32).reshape(1, 16)
            action = session_st.run(None, {"observation_flat": obs_flat})[0][0]
            obs_dict, _, _, _, _ = env.step(action)
            total_steps_st += 1
    total_time_st = time.perf_counter() - t0
    fps_st = total_steps_st / total_time_st
    ms_per_step_st = (total_time_st / total_steps_st) * 1000
    
    # E. Closed loop rollout with default threads ORT (50 episodes x 50 steps = 2500 steps)
    t0 = time.perf_counter()
    total_steps_mt = 0
    for ep in range(50):
        obs_dict, _ = env.reset(seed=42 + ep)
        for _ in range(50):
            ob = obs_dict["observation"]
            ag = obs_dict["achieved_goal"]
            dg = obs_dict["desired_goal"]
            obs_flat = np.concatenate([ob, ag, dg], axis=0).astype(np.float32).reshape(1, 16)
            action = session_mt.run(None, {"observation_flat": obs_flat})[0][0]
            obs_dict, _, _, _, _ = env.step(action)
            total_steps_mt += 1
    total_time_mt = time.perf_counter() - t0
    fps_mt = total_steps_mt / total_time_mt
    ms_per_step_mt = (total_time_mt / total_steps_mt) * 1000
    
    env.close()
    
    report = {
        "timing_method": "src/bench/timing.py: time_callable with time.perf_counter_ns",
        "definitions": {
            "pure_mujoco_physics_20_substeps": "Exactly 20 calls to mj_step(model, data). No rendering, no wrappers.",
            "isolated_env_step": "Full gymnasium env.step() call including observation dict and reward computation.",
            "closed_loop_rollout": "Sequential rollout of obs prep + ONNX inference + env.step() over 2,500 total steps."
        },
        "machine_state": get_machine_state(),
        "gl_backend": os.environ.get("MUJOCO_GL", "glfw"),
        "raw_mujoco_physics_20_substeps_ms": res_physics,
        "pure_mujoco_step_alone_ms": res_physics["median_ms"],
        "isolated_env_step_ms": res_env_step,
        "pure_obs_formatting_alone_ms": res_obs_prep["median_ms"],
        "closed_loop_single_thread_ort": {
            "total_steps": total_steps_st,
            "total_time_sec": round(total_time_st, 3),
            "effective_fps": round(fps_st, 1),
            "ms_per_step": round(ms_per_step_st, 3)
        },
        "closed_loop_default_thread_ort": {
            "total_steps": total_steps_mt,
            "total_time_sec": round(total_time_mt, 3),
            "effective_fps": round(fps_mt, 1),
            "ms_per_step": round(ms_per_step_mt, 3)
        },
    }
    
    with open("results/rollout_fps_audit.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Rollout FPS Audit Report:", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    audit_rollout_fps()
