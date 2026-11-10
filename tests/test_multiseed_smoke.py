"""Smoke test for multi-seed training and evaluation harness."""

import os
import tempfile
import pytest
from src.rl.train_multiseed import train_multiseed, bootstrap_ci


def test_bootstrap_ci_computation():
    """Verify bootstrap confidence interval calculation on known values."""
    values = [1.0, 1.0, 1.0, 1.0, 1.0]
    ci = bootstrap_ci(values, n_boot=200)
    assert ci["mean"] == 1.0
    assert ci["ci_low"] == 1.0
    assert ci["ci_high"] == 1.0
    assert ci["std"] == 0.0

    varied = [2.0, 3.0, 4.0, 5.0, 6.0]
    ci_varied = bootstrap_ci(varied, n_boot=200)
    assert ci_varied["mean"] == 4.0
    assert ci_varied["ci_low"] <= 4.0
    assert ci_varied["ci_high"] >= 4.0


def test_train_multiseed_smoke():
    """Smoke test running 2 seeds for 50 steps and 2 eval episodes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_multiseed.json")
        summary = train_multiseed(
            seeds=[101, 102],
            total_timesteps=50,
            n_eval_episodes=2,
            output_path=out_path,
        )

        assert os.path.exists(out_path)
        assert summary["n_seeds"] == 2
        assert len(summary["seed_breakdown"]) == 2
        assert "success_rate" in summary
        assert "mean_steps_to_goal" in summary

