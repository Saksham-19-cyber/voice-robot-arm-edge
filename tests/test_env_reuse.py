"""
Test environment reuse correctness.
Verifies that:
1. Reusing an existing simulation environment for consecutive different commands
   (e.g., Reach Left then Reach Right) properly updates the target goal,
   end-effector trajectory, and rendered visual frames.
2. The results on the reused environment match a fresh environment instantiated
   with the same random seed.
"""

import numpy as np
from src.rl.env_utils import make_fetch_env, WORKSPACE_CENTER
from src.rl.rollout import run_onnx_rollout


def test_env_reuse_goal_update_and_frame_divergence():
    goal_left = WORKSPACE_CENTER + np.array([-0.1, 0.0, 0.0], dtype=np.float32)
    goal_right = WORKSPACE_CENTER + np.array([0.1, 0.0, 0.0], dtype=np.float32)

    # 1. Reuse single environment across both rollouts
    shared_env = make_fetch_env("FetchReach-v4", render_mode="rgb_array", fixed_goal=goal_left)
    try:
        res_reused_left = run_onnx_rollout(goal_xyz=goal_left, env=shared_env, seed=42, max_steps=20)
        res_reused_right = run_onnx_rollout(goal_xyz=goal_right, env=shared_env, seed=42, max_steps=20)

        # Assert goals differ
        assert not np.allclose(res_reused_left["goal_xyz"], res_reused_right["goal_xyz"]), (
            "Goals must differ between Left and Right rollouts on reused env"
        )
        assert np.isclose(res_reused_left["goal_xyz"][0], float(goal_left[0]), atol=1e-3)
        assert np.isclose(res_reused_right["goal_xyz"][0], float(goal_right[0]), atol=1e-3)

        # Assert final end-effector positions differ
        final_ee_left = res_reused_left["ee_positions"][-1]
        final_ee_right = res_reused_right["ee_positions"][-1]
        assert not np.allclose(final_ee_left, final_ee_right, atol=1e-2), (
            f"Final EE positions must differ (left={final_ee_left}, right={final_ee_right})"
        )

        # Assert final visual frames differ
        frame_diff = np.abs(
            res_reused_left["frames"][-1].astype(np.float32)
            - res_reused_right["frames"][-1].astype(np.float32)
        )
        assert np.max(frame_diff) > 20.0, "Visual frames must visibly differ between Left and Right goals"

        # 2. Compare against fresh environment runs on the exact same seed
        res_fresh_left = run_onnx_rollout(goal_xyz=goal_left, seed=42, max_steps=20)
        res_fresh_right = run_onnx_rollout(goal_xyz=goal_right, seed=42, max_steps=20)

        # Reused Left must match Fresh Left
        assert np.allclose(final_ee_left, res_fresh_left["ee_positions"][-1], atol=1e-3), (
            "Reused Left final EE position does not match fresh env run"
        )
        # Reused Right must match Fresh Right
        assert np.allclose(final_ee_right, res_fresh_right["ee_positions"][-1], atol=1e-3), (
            "Reused Right final EE position does not match fresh env run"
        )
    finally:
        try:
            shared_env.close()
        except Exception:
            pass

