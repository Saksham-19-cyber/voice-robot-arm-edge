import os
import sys

# Platform-aware GL context: set osmesa if on Linux (cloud), native on Windows
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
else:
    os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import gymnasium as gym
import gymnasium_robotics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rl.env_utils import make_fetch_env, WORKSPACE_CENTER


def test_fetch_reach_smoke_lifecycle():
    """Verifies creation, reset, step, and rendering of FetchReach-v4 without assertion errors."""
    env = make_fetch_env("FetchReach-v4", render_mode="rgb_array", fixed_goal=WORKSPACE_CENTER)
    try:
        obs, info = env.reset(seed=42)
        assert "observation" in obs
        assert "achieved_goal" in obs
        assert "desired_goal" in obs
        assert obs["desired_goal"].shape == (3,)

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        assert next_obs["observation"].shape == (10,)

        frame = env.render()
        assert frame is not None
        assert len(frame.shape) == 3
        assert frame.shape[2] == 3  # RGB channels
        assert frame.shape[0] > 0 and frame.shape[1] > 0
    finally:
        env.close()

