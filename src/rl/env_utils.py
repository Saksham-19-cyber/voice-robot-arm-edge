"""FetchReach environment utilities and compatibility patch."""






import os
import sys
import numpy as np
import gymnasium as gym
from typing import Optional


try:
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)









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


WORKSPACE_BOUNDS = {
    "x": (1.15, 1.50),
    "y": (0.60, 0.90),
    "z": (0.40, 0.70),
}
WORKSPACE_CENTER = np.array([1.34, 0.75, 0.53], dtype=np.float32)


def setup_headless_rendering():





    if sys.platform.startswith("linux"):

        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "osmesa"
        return os.environ.get("MUJOCO_GL", "osmesa")
    else:

        return os.environ.get("MUJOCO_GL", "native")


def clip_to_workspace(coords: np.ndarray) -> np.ndarray:

    x = np.clip(coords[0], WORKSPACE_BOUNDS["x"][0], WORKSPACE_BOUNDS["x"][1])
    y = np.clip(coords[1], WORKSPACE_BOUNDS["y"][0], WORKSPACE_BOUNDS["y"][1])
    z = np.clip(coords[2], WORKSPACE_BOUNDS["z"][0], WORKSPACE_BOUNDS["z"][1])
    return np.array([x, y, z], dtype=np.float32)


class GoalOverrideWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, fixed_goal: Optional[np.ndarray] = None):
        super().__init__(env)
        self.fixed_goal = np.array(fixed_goal, dtype=np.float32) if fixed_goal is not None else None

    def set_goal(self, goal: np.ndarray):
        self.fixed_goal = np.array(goal, dtype=np.float32)
        if hasattr(self.env.unwrapped, "goal"):
            self.env.unwrapped.goal = self.fixed_goal.copy()
        if hasattr(self.env.unwrapped, "_render_callback"):
            self.env.unwrapped._render_callback()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.fixed_goal is not None:
            self.env.unwrapped.goal = self.fixed_goal.copy()
            if hasattr(self.env.unwrapped, "_render_callback"):
                self.env.unwrapped._render_callback()
            obs["desired_goal"] = self.fixed_goal.copy()
        return obs, info


def make_fetch_env(
    env_id: str = "FetchReach-v4",
    render_mode: str = "rgb_array",
    fixed_goal: np.ndarray = None,
    width: int = 320,
    height: int = 320,
) -> gym.Env:
    setup_headless_rendering()

    try:
        env = gym.make(
            env_id,
            render_mode=render_mode,
            width=width,
            height=height,
        )
    except TypeError:
        env = gym.make(env_id, render_mode=render_mode)

    return GoalOverrideWrapper(env, fixed_goal=fixed_goal)


def flatten_observation(obs: dict) -> np.ndarray:





    return np.concatenate([
        obs["observation"].flatten(),
        obs["achieved_goal"].flatten(),
        obs["desired_goal"].flatten(),
    ], dtype=np.float32)
