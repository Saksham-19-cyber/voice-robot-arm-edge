"""Proportional baseline controller for FetchReach-v4."""

import numpy as np


class ProportionalReachController:
    """Proportional controller for 3D Cartesian end-effector positioning in FetchReach.

    action = clip(k * (goal - gripper_pos) / max_step, -1.0, 1.0)
    For FetchReach-v4, each step moves the gripper by at most ~0.05m per axis.
    The action vector is 4D: [dx, dy, dz, gripper_ctrl].
    """

    def __init__(self, k: float = 5.0, max_step: float = 0.05):
        self.k = float(k)
        self.max_step = float(max_step)

    def predict(self, observation: dict) -> np.ndarray:
        """Compute action from Gymnasium GoalEnv observation dictionary or flat vector.

        If observation is dict:
          achieved_goal: gripper end-effector position (3D)
          desired_goal: target position (3D)
        If observation is 16D array:
          achieved_goal is obs[10:13]
          desired_goal is obs[13:16]
        """
        if isinstance(observation, dict):
            gripper_pos = np.array(observation["achieved_goal"], dtype=np.float32)
            goal_pos = np.array(observation["desired_goal"], dtype=np.float32)
        else:
            obs = np.asarray(observation).flatten()
            gripper_pos = obs[10:13]
            goal_pos = obs[13:16]

        delta = goal_pos - gripper_pos
        action_xyz = np.clip(self.k * delta / self.max_step, -1.0, 1.0)
        action = np.zeros(4, dtype=np.float32)
        action[:3] = action_xyz
        action[3] = 0.0  # gripper inactive for reach
        return action

