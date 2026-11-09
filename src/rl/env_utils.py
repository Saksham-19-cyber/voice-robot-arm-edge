"""
Environment utilities for Gymnasium-Robotics Fetch environments.
Provides platform-aware headless rendering, workspace bounds clipping,
goal-override wrapping for voice commands, and observation flattening.
Includes comprehensive pybind11 enum compatibility patching for Linux/OSMesa.
"""

import os
import sys
import numpy as np
import gymnasium as gym

# Safe import and registration of gymnasium_robotics
try:
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)

    # Strategy B Hotfix:
    # Under Linux GCC with pybind11 and NumPy >= 2.0, pybind11's enum equality checks
    # evaluate `mjtJoint.__eq__(numpy.int32)` which returns False / NotImplemented.
    # Gymnasium-Robotics 1.4.2 mujoco_utils functions assert:
    #   assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    #   if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
    # where joint_type and eq_type are numpy.int32 scalars directly from model arrays.
    # Coercing both the array values and enum variants to int() resolves this on all Linux distros.
    import gymnasium_robotics.utils.mujoco_utils as _mu
    import mujoco as _mj

    def _safe_get_joint_qpos(model, data, name):
        joint_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, name)
        assert joint_id != -1, f"Joint with name '{name}' is not part of the model!"
        joint_type = int(model.jnt_type[joint_id])
        joint_addr = model.jnt_qposadr[joint_id]

        if joint_type == int(_mj.mjtJoint.mjJNT_FREE):
            ndim = 7
        elif joint_type == int(_mj.mjtJoint.mjJNT_BALL):
            ndim = 4
        else:
            assert joint_type in (int(_mj.mjtJoint.mjJNT_HINGE), int(_mj.mjtJoint.mjJNT_SLIDE))
            ndim = 1

        start_idx = joint_addr
        end_idx = joint_addr + ndim
        return data.qpos[start_idx:end_idx].copy()

    def _safe_get_joint_qvel(model, data, name):
        joint_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, name)
        assert joint_id != -1, f"Joint with name '{name}' is not part of the model!"
        joint_type = int(model.jnt_type[joint_id])
        joint_addr = model.jnt_dofadr[joint_id]

        if joint_type == int(_mj.mjtJoint.mjJNT_FREE):
            ndim = 6
        elif joint_type == int(_mj.mjtJoint.mjJNT_BALL):
            ndim = 3
        else:
            assert joint_type in (int(_mj.mjtJoint.mjJNT_HINGE), int(_mj.mjtJoint.mjJNT_SLIDE))
            ndim = 1

        start_idx = joint_addr
        end_idx = joint_addr + ndim
        return data.qvel[start_idx:end_idx].copy()

    def _safe_set_joint_qpos(model, data, name, value):
        joint_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, name)
        assert joint_id != -1, f"Joint with name '{name}' is not part of the model!"
        joint_type = int(model.jnt_type[joint_id])
        joint_addr = model.jnt_qposadr[joint_id]

        if joint_type == int(_mj.mjtJoint.mjJNT_FREE):
            ndim = 7
        elif joint_type == int(_mj.mjtJoint.mjJNT_BALL):
            ndim = 4
        else:
            assert joint_type in (int(_mj.mjtJoint.mjJNT_HINGE), int(_mj.mjtJoint.mjJNT_SLIDE))
            ndim = 1

        start_idx = joint_addr
        end_idx = joint_addr + ndim
        val_arr = np.array(value)
        if ndim > 1:
            assert val_arr.shape == (end_idx - start_idx), f"Value has incorrect shape {name}: {value}"
        data.qpos[start_idx:end_idx] = val_arr

    def _safe_set_joint_qvel(model, data, name, value):
        joint_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, name)
        assert joint_id != -1, f"Joint with name '{name}' is not part of the model!"
        joint_type = int(model.jnt_type[joint_id])
        joint_addr = model.jnt_dofadr[joint_id]

        if joint_type == int(_mj.mjtJoint.mjJNT_FREE):
            ndim = 6
        elif joint_type == int(_mj.mjtJoint.mjJNT_BALL):
            ndim = 3
        else:
            assert joint_type in (int(_mj.mjtJoint.mjJNT_HINGE), int(_mj.mjtJoint.mjJNT_SLIDE))
            ndim = 1

        start_idx = joint_addr
        end_idx = joint_addr + ndim
        val_arr = np.array(value)
        if ndim > 1:
            assert val_arr.shape == (end_idx - start_idx), f"Value has incorrect shape {name}: {value}"
        data.qvel[start_idx:end_idx] = val_arr

    def _safe_reset_mocap_welds(model, data):
        if model.nmocap > 0 and model.eq_data is not None:
            for i in range(model.eq_data.shape[0]):
                if int(model.eq_type[i]) == int(_mj.mjtEq.mjEQ_WELD):
                    model.eq_data[i, :7] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        _mj.mj_forward(model, data)

    def _safe_reset_mocap2body_xpos(model, data):
        if model.eq_type is None or model.eq_obj1id is None or model.eq_obj2id is None:
            return
        for eq_type, obj1_id, obj2_id in zip(
            model.eq_type, model.eq_obj1id, model.eq_obj2id
        ):
            if int(eq_type) != int(_mj.mjtEq.mjEQ_WELD):
                continue

            mocap_id = model.body_mocapid[obj1_id]
            if mocap_id != -1:
                body_idx = obj2_id
            else:
                mocap_id = model.body_mocapid[obj2_id]
                body_idx = obj1_id

            assert mocap_id != -1
            data.mocap_pos[mocap_id][:] = data.xpos[body_idx]
            data.mocap_quat[mocap_id][:] = data.xquat[body_idx]

    def apply_mujoco_utils_patch():
        if os.environ.get("DISABLE_MUJOCO_UTILS_PATCH") == "1":
            print("[env_utils] MuJoCo pybind11 enum compatibility patch: DISABLED by env var", flush=True)
            return False

        _mu.set_joint_qpos = _safe_set_joint_qpos
        _mu.set_joint_qvel = _safe_set_joint_qvel
        _mu.get_joint_qpos = _safe_get_joint_qpos
        _mu.get_joint_qvel = _safe_get_joint_qvel
        _mu.reset_mocap_welds = _safe_reset_mocap_welds
        _mu.reset_mocap2body_xpos = _safe_reset_mocap2body_xpos
        print("[env_utils] MuJoCo pybind11 enum compatibility patch: ACTIVE (coercing jnt_type & eq_type to int)", flush=True)
        return True

    apply_mujoco_utils_patch()

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


def clip_to_workspace(coords: np.ndarray) -> np.ndarray:
    """Clips 3D Cartesian coordinates strictly to the robot's physical workspace bounds."""
    x = np.clip(coords[0], WORKSPACE_BOUNDS["x"][0], WORKSPACE_BOUNDS["x"][1])
    y = np.clip(coords[1], WORKSPACE_BOUNDS["y"][0], WORKSPACE_BOUNDS["y"][1])
    z = np.clip(coords[2], WORKSPACE_BOUNDS["z"][0], WORKSPACE_BOUNDS["z"][1])
    return np.array([x, y, z], dtype=np.float32)


class GoalOverrideWrapper(gym.Wrapper):
    """
    Wraps FetchReach environment to override the randomly sampled goal
    with a user-specified 3D target coordinate.
    """

    def __init__(self, env: gym.Env, fixed_goal: np.ndarray = WORKSPACE_CENTER):
        super().__init__(env)
        self.fixed_goal = np.array(fixed_goal, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.fixed_goal is not None:
            self.env.unwrapped.goal = self.fixed_goal.copy()
            obs["desired_goal"] = self.fixed_goal.copy()
        return obs, info


def make_fetch_env(
    env_id: str = "FetchReach-v4",
    render_mode: str = "rgb_array",
    fixed_goal: np.ndarray = None,
    width: int = 320,
    height: int = 320,
) -> gym.Env:
    """
    Factory function to instantiate FetchReach environment with headless rendering
    and goal override capabilities.
    """
    setup_headless_rendering()

    try:
        env = gym.make(
            env_id,
            render_mode=render_mode,
            width=width,
            height=height,
        )
    except TypeError:
        # Fallback if gymnasium-robotics version does not accept width/height in make()
        env = gym.make(env_id, render_mode=render_mode)

    if fixed_goal is not None:
        env = GoalOverrideWrapper(env, fixed_goal=fixed_goal)

    return env


def flatten_observation(obs: dict) -> np.ndarray:
    """
    Flattens Gymnasium-Robotics GoalEnv dictionary into a 16-element 1D vector
    expected by the SAC policy actor:
    [observation (10), achieved_goal (3), desired_goal (3)].
    """
    return np.concatenate([
        obs["observation"].flatten(),
        obs["achieved_goal"].flatten(),
        obs["desired_goal"].flatten(),
    ], dtype=np.float32)
