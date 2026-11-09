"""
Inference-only rollout generator for Streamlit app.
Runs ONNX policy (FP32 or INT8) in MuJoCo, renders frames, and encodes to MP4/GIF.
Uses zero PyTorch dependencies during inference.
"""

import os
import sys
import time
import tempfile
import numpy as np
import imageio
import onnxruntime as ort
import gymnasium as gym
import gymnasium_robotics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rl.env_utils import make_fetch_env, clip_to_workspace, setup_headless_rendering


def run_onnx_rollout(
    goal_xyz: np.ndarray,
    onnx_path: str = "models/fetch_reach_actor_int8.onnx",
    max_steps: int = 25,
    render_width: int = 320,
    render_height: int = 240,
    seed: int = 42,
    output_format: str = "mp4",
) -> dict:
    """
    Executes closed-loop rollout using ONNX policy in MuJoCo.
    Renders frames and encodes to video (MP4) or GIF.
    Returns rollout stats and filepath to rendered video.
    """
    setup_headless_rendering()

    clipped_goal = clip_to_workspace(goal_xyz)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Create FetchReach environment with goal override and rgb_array rendering
    env = make_fetch_env("FetchReach-v4", render_mode="rgb_array", fixed_goal=clipped_goal)
    obs_dict, _ = env.reset(seed=seed)

    frames = []
    t0 = time.perf_counter()
    step = 0
    succeeded = False
    first_success_step = None
    distances = []

    # Capture initial frame
    frame = env.render()
    if frame is not None:
        # Resize or crop if needed, or take native
        frames.append(frame)

    while step < max_steps:
        # Format obs as [observation (10), achieved_goal (3), desired_goal (3)]
        ob = obs_dict["observation"]
        ag = obs_dict["achieved_goal"]
        dg = obs_dict["desired_goal"]
        obs_flat = np.concatenate([ob, ag, dg], axis=0).astype(np.float32).reshape(1, 16)

        # ONNX inference
        action = session.run(None, {"observation_flat": obs_flat})[0][0]

        obs_dict, reward, terminated, truncated, info = env.step(action)
        step += 1

        dist = float(np.linalg.norm(obs_dict["achieved_goal"] - obs_dict["desired_goal"]))
        distances.append(round(dist, 4))

        if info.get("is_success", 0.0) == 1.0 and not succeeded:
            succeeded = True
            first_success_step = step

        # Render frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        # If achieved goal and held for 3 frames, terminate early to keep video snappy
        if succeeded and (step - first_success_step) >= 3:
            break

    rollout_time_ms = (time.perf_counter() - t0) * 1000
    env.close()

    # Encode frames
    temp_dir = tempfile.gettempdir()
    ext = ".mp4" if output_format == "mp4" else ".gif"
    video_path = os.path.join(temp_dir, f"rollout_{int(time.time()*1000)}{ext}")

    t_enc_0 = time.perf_counter()
    if output_format == "mp4":
        try:
            imageio.mimwrite(video_path, frames, fps=15, format="FFMPEG", quality=8)
        except Exception:
            # Fallback to GIF if ffmpeg plugin fails
            video_path = os.path.join(temp_dir, f"rollout_{int(time.time()*1000)}.gif")
            imageio.mimwrite(video_path, frames, fps=12)
    else:
        imageio.mimwrite(video_path, frames, fps=12)
    encoding_time_ms = (time.perf_counter() - t_enc_0) * 1000

    return {
        "video_path": video_path,
        "n_frames": len(frames),
        "steps_taken": step,
        "succeeded": succeeded,
        "first_success_step": first_success_step,
        "final_distance_m": distances[-1] if distances else 0.0,
        "rollout_time_ms": round(rollout_time_ms, 2),
        "encoding_time_ms": round(encoding_time_ms, 2),
    }
