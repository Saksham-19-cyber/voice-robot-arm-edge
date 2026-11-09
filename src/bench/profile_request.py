import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.abspath("."))
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
else:
    os.environ.setdefault("MUJOCO_GL", "glfw")

import gymnasium as gym
import gymnasium_robotics
import onnxruntime as ort
import imageio
from src.rl.env_utils import make_fetch_env, clip_to_workspace, WORKSPACE_CENTER

def profile_single_request():
    print("Profiling request stages in detail...")
    target = WORKSPACE_CENTER + np.array([-0.05, 0.10, 0.05], dtype=np.float32)
    onnx_path = "models/fetch_reach_actor_int8.onnx"
    max_steps = 25
    render_width = 320
    render_height = 320

    timings = {}

    # 1. ORT Session init
    t0 = time.perf_counter()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    timings["ort_session_init_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 2. make_env
    t0 = time.perf_counter()
    env = make_fetch_env("FetchReach-v4", render_mode="rgb_array", fixed_goal=target)
    timings["env_construction_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 3. env.reset
    t0 = time.perf_counter()
    obs_dict, _ = env.reset(seed=42)
    timings["env_reset_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 4. First render (renderer initialization & GL context binding)
    t0 = time.perf_counter()
    f0 = env.render()
    timings["first_render_init_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # 5. Stepping & Subsequent renders
    step_times = []
    policy_times = []
    physics_times = []
    subsequent_render_times = []
    frames = [f0] if f0 is not None else []

    for step in range(max_steps):
        ob = obs_dict["observation"]
        ag = obs_dict["achieved_goal"]
        dg = obs_dict["desired_goal"]
        obs_flat = np.concatenate([ob, ag, dg], axis=0).astype(np.float32).reshape(1, 16)

        t_pol = time.perf_counter()
        action = session.run(None, {"observation_flat": obs_flat})[0][0]
        policy_times.append((time.perf_counter() - t_pol) * 1000)

        t_phys = time.perf_counter()
        obs_dict, reward, terminated, truncated, info = env.step(action)
        physics_times.append((time.perf_counter() - t_phys) * 1000)
        step_times.append(policy_times[-1] + physics_times[-1])

        t_rend = time.perf_counter()
        fr = env.render()
        subsequent_render_times.append((time.perf_counter() - t_rend) * 1000)
        if fr is not None:
            frames.append(fr)

        if info.get("is_success", 0.0) == 1.0 and step >= 10:
            break

    timings["steps_executed"] = len(step_times)
    timings["policy_inference_per_step_ms"] = {
        "mean": round(float(np.mean(policy_times)), 3),
        "total": round(float(np.sum(policy_times)), 2),
    }
    timings["physics_step_per_step_ms"] = {
        "mean": round(float(np.mean(physics_times)), 3),
        "total": round(float(np.sum(physics_times)), 2),
    }
    timings["render_per_frame_ms"] = {
        "mean": round(float(np.mean(subsequent_render_times)), 3),
        "total": round(float(np.sum(subsequent_render_times)), 2),
    }

    # 6. MP4 encoding
    import tempfile
    temp_mp4 = os.path.join(tempfile.gettempdir(), "test_prof.mp4")
    t0 = time.perf_counter()
    imageio.mimwrite(
        temp_mp4,
        frames,
        fps=15,
        codec="libx264",
        pixelformat="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    timings["mp4_encode_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    timings["mp4_size_bytes"] = os.path.getsize(temp_mp4)

    # 7. GIF encoding
    temp_gif = os.path.join(tempfile.gettempdir(), "test_prof.gif")
    t0 = time.perf_counter()
    imageio.mimwrite(temp_gif, frames, fps=12)
    timings["gif_encode_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    timings["gif_size_bytes"] = os.path.getsize(temp_gif)

    env.close()

    total_pipeline_ms = (
        timings["env_construction_ms"]
        + timings["env_reset_ms"]
        + timings["first_render_init_ms"]
        + timings["policy_inference_per_step_ms"]["total"]
        + timings["physics_step_per_step_ms"]["total"]
        + timings["render_per_frame_ms"]["total"]
        + timings["mp4_encode_ms"]
    )
    timings["total_request_latency_ms"] = round(total_pipeline_ms, 2)

    os.makedirs("results", exist_ok=True)
    out_path = "results/request_profile.json"
    with open(out_path, "w") as f:
        json.dump(timings, f, indent=2)

    print("Request profile results:")
    print(json.dumps(timings, indent=2))
    return timings

if __name__ == "__main__":
    profile_single_request()

