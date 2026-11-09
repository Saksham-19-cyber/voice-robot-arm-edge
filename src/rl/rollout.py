"""
Inference-only rollout generator for Streamlit app and edge deployment.
Runs ONNX policy (FP32 or INT8) in MuJoCo FetchReach-v4, renders frames,
and encodes to browser-compatible MP4 (H.264 yuv420p + faststart) and GIF.
Supports environment reuse to eliminate ~1.2s cold start on warm requests.
Uses zero PyTorch dependencies during inference.
"""

import os
import sys
import time
import tempfile
from typing import Optional, List, Dict, Any
import numpy as np
import imageio
import onnxruntime as ort
import gymnasium as gym

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rl.env_utils import make_fetch_env, clip_to_workspace, setup_headless_rendering


HOLD_STEPS = 10  # Constant: number of steps to hold position after first is_success


def run_onnx_rollout(
    goal_xyz: np.ndarray,
    onnx_path: str = "models/fetch_reach_actor_int8.onnx",
    max_steps: int = 50,
    hold_steps: int = HOLD_STEPS,
    early_termination: bool = True,
    render_width: int = 320,
    render_height: int = 320,
    seed: int = 42,
    output_format: str = "mp4",
    env: Optional[gym.Env] = None,
    session: Optional[ort.InferenceSession] = None,
) -> Dict[str, Any]:
    """
    Executes closed-loop rollout using ONNX policy in MuJoCo.
    - If env is passed in, reuses it without re-constructing renderer context.
    - If early_termination is True, stops at first_success_step + hold_steps.
    - Encodes MP4 with H.264 yuv420p and -movflags +faststart for browser playback.
    - Returns full trajectory metrics, video bytes, gif bytes, and frames for UI tabs.
    """
    setup_headless_rendering()

    # Ensure even dimensions for yuv420p compatibility
    rw = render_width if render_width % 2 == 0 else render_width + 1
    rh = render_height if render_height % 2 == 0 else render_height + 1

    goal_arr = np.asarray(goal_xyz, dtype=np.float32).flatten()
    clipped_goal = clip_to_workspace(goal_arr)
    is_clipped = bool(not np.allclose(goal_arr, clipped_goal, atol=1e-4))

    # Reuse or create ONNX session
    if session is None:
        opts = ort.SessionOptions()
        session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])

    # Reuse or create environment
    owns_env = False
    if env is None:
        env = make_fetch_env("FetchReach-v4", render_mode="rgb_array", width=rw, height=rh, fixed_goal=clipped_goal)
        owns_env = True
    else:
        if hasattr(env, "set_goal"):
            env.set_goal(clipped_goal)

    t0 = time.perf_counter()
    obs_dict, _ = env.reset(seed=seed)

    start_pos = [round(float(x), 4) for x in obs_dict["achieved_goal"]]
    frames: List[np.ndarray] = []
    render_times_ms: List[float] = []

    # Initial frame capture
    t_r0 = time.perf_counter()
    f0 = env.render()
    render_times_ms.append((time.perf_counter() - t_r0) * 1000)
    if f0 is not None:
        frames.append(f0)

    step = 0
    succeeded = False
    first_success_step = None
    distances: List[float] = []
    ee_positions: List[List[float]] = [start_pos]
    actions_taken: List[List[float]] = []

    while step < max_steps:
        # Obs layout: [observation (10), achieved_goal (3), desired_goal (3)]
        ob = obs_dict["observation"]
        ag = obs_dict["achieved_goal"]
        dg = obs_dict["desired_goal"]
        obs_flat = np.concatenate([ob, ag, dg], axis=0).astype(np.float32).reshape(1, 16)

        # Policy inference
        action = session.run(None, {"observation_flat": obs_flat})[0][0]
        actions_taken.append([round(float(a), 4) for a in action])

        # Step physics
        obs_dict, reward, terminated, truncated, info = env.step(action)
        step += 1

        curr_ag = [round(float(x), 4) for x in obs_dict["achieved_goal"]]
        ee_positions.append(curr_ag)

        dist = float(np.linalg.norm(obs_dict["achieved_goal"] - obs_dict["desired_goal"]))
        distances.append(round(dist, 4))

        if info.get("is_success", 0.0) == 1.0 and not succeeded:
            succeeded = True
            first_success_step = step

        # Render frame
        t_r = time.perf_counter()
        frame = env.render()
        render_times_ms.append((time.perf_counter() - t_r) * 1000)
        if frame is not None:
            frames.append(frame)

        # Early termination
        if early_termination and succeeded and (step - first_success_step) >= hold_steps:
            break

    rollout_time_ms = (time.perf_counter() - t0) * 1000

    if owns_env:
        env.close()

    # Encode MP4 with H.264 yuv420p and +faststart
    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time() * 1000)
    mp4_path = os.path.join(temp_dir, f"rollout_{timestamp}.mp4")
    gif_path = os.path.join(temp_dir, f"rollout_{timestamp}.gif")

    t_enc_0 = time.perf_counter()
    video_bytes = None
    gif_bytes = None

    try:
        imageio.mimwrite(
            mp4_path,
            frames,
            fps=15,
            codec="libx264",
            pixelformat="yuv420p",
            output_params=["-movflags", "+faststart"],
        )
        with open(mp4_path, "rb") as fp:
            video_bytes = fp.read()
    except Exception as e:
        print(f"Warning: MP4 encoding fallback: {e}")

    try:
        # Generate GIF fallback (useful for browser fallback & frames tab)
        imageio.mimwrite(gif_path, frames, fps=12)
        with open(gif_path, "rb") as fp:
            gif_bytes = fp.read()
    except Exception as e:
        print(f"Warning: GIF encoding failed: {e}")

    encoding_time_ms = (time.perf_counter() - t_enc_0) * 1000
    final_dist_m = distances[-1] if distances else float(np.linalg.norm(start_pos - clipped_goal))

    return {
        "video_path": mp4_path,
        "video_bytes": video_bytes,
        "gif_path": gif_path,
        "gif_bytes": gif_bytes,
        "frames": frames,
        "n_frames": len(frames),
        "steps_taken": step,
        "succeeded": succeeded,
        "first_success_step": first_success_step,
        "final_distance_m": round(final_dist_m, 4),
        "final_distance_cm": round(final_dist_m * 100, 2),
        "start_pos_xyz": start_pos,
        "goal_xyz": [round(float(x), 4) for x in clipped_goal],
        "raw_goal_xyz": [round(float(x), 4) for x in goal_arr],
        "clipped": is_clipped,
        "distances_m": distances,
        "ee_positions": ee_positions,
        "actions": actions_taken,
        "rollout_time_ms": round(rollout_time_ms, 2),
        "encoding_time_ms": round(encoding_time_ms, 2),
        "render_time_per_frame_ms": round(float(np.mean(render_times_ms)), 2) if render_times_ms else 0.0,
    }
