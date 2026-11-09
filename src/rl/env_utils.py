"""
Environment utilities for Gymnasium-Robotics Fetch environments.
Provides platform-aware headless rendering, workspace bounds clipping,
goal-override wrapping for voice commands, and observation flattening.
"""

import os
import sys
import numpy as np
import gymnasium as gym

# Safe import and registration of gymnasium_robotics
try:
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
except ImportError:
    pass

# Fetch workspace bounds (tabletop reachable volume)
WORKSPACE_BOUNDS = {
    "x": (1.15, 1.50),  # Depth (front/back relative to base)
    "y": (0.60, 0.90),  # Lateral (left/right)
    "z": (0.40, 0.70),  # Vertical (table surface to upper reach)
}
WORKSPACE_CENTER = np.array([1.34, 0.75, 0.53], dtype=np.float32)


def setup_headless_rendering():
    """
    Configures MuJoCo GL backend depending on OS.
    On Linux (Streamlit Cloud), uses osmesa or egl.
    On Windows, uses native desktop context / wgl.
    """
    if sys.platform.startswith("linux"):
        # Prefer osmesa for headless containers without GPU/EGL drivers
        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "osmesa"
        return os.environ.get("MUJOCO_GL", "osmesa")
    else:
        # On Windows, leave unset or default for native WGL/software
        return os.environ.get("MUJOCO_GL", "native")


def clip_to_workspace(goal_xyz: np.ndarray) -> np.ndarray:
    """Clips target goal to physical reachable workspace limits."""
    clipped = np.array(goal_xyz, dtype=np.float32).copy()
    clipped[0] = np.clip(clipped[0], WORKSPACE_BOUNDS["x"][0], WORKSPACE_BOUNDS["x"][1])
    clipped[1] = np.clip(clipped[1], WORKSPACE_BOUNDS["y"][0], WORKSPACE_BOUNDS["y"][1])
    clipped[2] = np.clip(clipped[2], WORKSPACE_BOUNDS["z"][0], WORKSPACE_BOUNDS["z"][1])
    return clipped


def flatten_obs(obs_dict: dict) -> np.ndarray:
    """
    Flattens a Fetch observation dictionary into a 1D vector (16,)
    matching SB3 MultiInputPolicy extractor order:
    [achieved_goal (3), desired_goal (3), observation (10)]
    """
    ag = np.asarray(obs_dict["achieved_goal"], dtype=np.float32).flatten()
    dg = np.asarray(obs_dict["desired_goal"], dtype=np.float32).flatten()
    ob = np.asarray(obs_dict["observation"], dtype=np.float32).flatten()
    return np.concatenate([ag, dg, ob], axis=0)


class GoalOverrideWrapper(gym.Wrapper):
    """
    Gym wrapper enabling dynamic goal overrides from voice/NLU input.
    """
    def __init__(self, env: gym.Env, fixed_goal: np.ndarray = None):
        super().__init__(env)
        self.fixed_goal = fixed_goal

    def set_goal(self, goal: np.ndarray):
        self.fixed_goal = clip_to_workspace(goal)
        if hasattr(self.env.unwrapped, "goal"):
            self.env.unwrapped.goal = self.fixed_goal.copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.fixed_goal is not None:
            self.set_goal(self.fixed_goal)
            obs = self.env.unwrapped._get_obs()
        return obs, info


def make_fetch_env(env_id: str = "FetchReach-v4", render_mode: str = None, fixed_goal: np.ndarray = None):
    """Factory creating Fetch env with headless rendering and optional goal override."""
    setup_headless_rendering()
    env = gym.make(env_id, render_mode=render_mode)
    if fixed_goal is not None:
        env = GoalOverrideWrapper(env, fixed_goal=fixed_goal)
    return env

