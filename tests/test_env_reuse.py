"""Unit tests for simulation environment reuse."""









import numpy as np
from src.rl.env_utils import make_fetch_env, WORKSPACE_CENTER
from src.rl.rollout import run_onnx_rollout


def test_env_reuse_goal_update_and_frame_divergence():
    goal_left = WORKSPACE_CENTER + np.array([-0.1, 0.0, 0.0], dtype=np.float32)
    goal_right = WORKSPACE_CENTER + np.array([0.1, 0.0, 0.0], dtype=np.float32)


    shared_env = make_fetch_env("FetchReach-v4", render_mode="rgb_array", fixed_goal=goal_left)
    try:
        res_reused_left = run_onnx_rollout(goal_xyz=goal_left, env=shared_env, seed=42, max_steps=20)
        res_reused_right = run_onnx_rollout(goal_xyz=goal_right, env=shared_env, seed=42, max_steps=20)


        assert not np.allclose(res_reused_left["goal_xyz"], res_reused_right["goal_xyz"]), (
            "Goals must differ between Left and Right rollouts on reused env"
        )
        assert np.isclose(res_reused_left["goal_xyz"][0], float(goal_left[0]), atol=1e-3)
        assert np.isclose(res_reused_right["goal_xyz"][0], float(goal_right[0]), atol=1e-3)


        final_ee_left = res_reused_left["ee_positions"][-1]
        final_ee_right = res_reused_right["ee_positions"][-1]
        assert not np.allclose(final_ee_left, final_ee_right, atol=1e-2), (
            f"Final EE positions must differ (left={final_ee_left}, right={final_ee_right})"
        )


        frame_diff = np.abs(
            res_reused_left["frames"][-1].astype(np.float32)
            - res_reused_right["frames"][-1].astype(np.float32)
        )
        assert np.max(frame_diff) > 20.0, "Visual frames must visibly differ between Left and Right goals"


        res_fresh_left = run_onnx_rollout(goal_xyz=goal_left, seed=42, max_steps=20)
        res_fresh_right = run_onnx_rollout(goal_xyz=goal_right, seed=42, max_steps=20)


        assert np.allclose(final_ee_left, res_fresh_left["ee_positions"][-1], atol=1e-3), (
            "Reused Left final EE position does not match fresh env run"
        )

        assert np.allclose(final_ee_right, res_fresh_right["ee_positions"][-1], atol=1e-3), (
            "Reused Right final EE position does not match fresh env run"
        )
    finally:
        try:
            shared_env.close()
        except Exception:
            pass


def test_env_unwrapped_goal_matches_requested_goal_on_reused_env():
    """Verify that env.unwrapped.goal matches requested goal after reset on reused env."""
    shared_env = make_fetch_env("FetchReach-v4", render_mode="rgb_array")
    test_targets = [
        np.array([1.25, 0.70, 0.60], dtype=np.float32),
        np.array([1.45, 0.85, 0.45], dtype=np.float32),
        np.array([1.30, 0.65, 0.50], dtype=np.float32),
    ]
    try:
        for target in test_targets:
            shared_env.set_goal(target)
            obs, _ = shared_env.reset(seed=123)
            assert np.allclose(shared_env.unwrapped.goal, target, atol=1e-4), (
                f"env.unwrapped.goal {shared_env.unwrapped.goal} != target {target}"
            )
            assert np.allclose(obs["desired_goal"], target, atol=1e-4), (
                f"obs['desired_goal'] {obs['desired_goal']} != target {target}"
            )
            res = run_onnx_rollout(goal_xyz=target, env=shared_env, max_steps=5, seed=123)
            assert np.allclose(shared_env.unwrapped.goal, target, atol=1e-4), (
                f"After rollout: env.unwrapped.goal {shared_env.unwrapped.goal} != target {target}"
            )
    finally:
        shared_env.close()
