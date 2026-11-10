"""Multi-seed training and evaluation harness for FetchReach-v4 SAC policies."""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Any, List

import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rl.env_utils import setup_headless_rendering
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv


def bootstrap_ci(values: List[float], n_boot: int = 1000, ci: float = 0.95) -> Dict[str, float]:
    """Calculate mean and bootstrap percentile confidence interval."""
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 1:
        return {"mean": float(arr[0]), "ci_low": float(arr[0]), "ci_high": float(arr[0])}

    rng = np.random.default_rng(42)
    boot_means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0,
        "ci_low": round(float(np.percentile(boot_means, alpha * 100)), 4),
        "ci_high": round(float(np.percentile(boot_means, (1.0 - alpha) * 100)), 4),
    }


def evaluate_single_seed(model, n_episodes: int = 100, seed: int = 42) -> Dict[str, Any]:
    """Evaluate a trained model over n_episodes."""
    eval_env = gym.make("FetchReach-v4")
    successes = []
    steps_to_goal = []

    for ep in range(n_episodes):
        obs_dict, _ = eval_env.reset(seed=seed + ep)
        done = False
        step = 0
        first_success = None

        while not done and step < 50:
            action, _ = model.predict(obs_dict, deterministic=True)
            obs_dict, reward, terminated, truncated, info = eval_env.step(action)
            dist = np.linalg.norm(obs_dict["achieved_goal"] - obs_dict["desired_goal"])
            if dist <= 0.05 and first_success is None:
                first_success = step + 1
            step += 1
            done = terminated or truncated

        successes.append(1.0 if first_success is not None else 0.0)
        steps_to_goal.append(first_success if first_success is not None else 50)

    eval_env.close()
    return {
        "success_rate": float(np.mean(successes)),
        "mean_steps_to_goal": float(np.mean(steps_to_goal)),
        "n_episodes": n_episodes,
    }


def train_multiseed(
    seeds: List[int] = (42, 43, 44, 45, 46),
    total_timesteps: int = 4000,
    n_eval_episodes: int = 100,
    output_path: str = "results/multiseed_reach_eval.json",
    save_models: bool = False,
) -> Dict[str, Any]:
    """Train SAC on FetchReach-v4 across multiple seeds and aggregate metrics."""
    setup_headless_rendering()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    seed_results = []
    start_time = time.time()

    for idx, seed in enumerate(seeds):
        print(f"\n--- Training Seed {seed} ({idx + 1}/{len(seeds)}) for {total_timesteps} steps ---")
        train_env = gym.make("FetchReach-v4")

        # Set PyTorch & NumPy seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = SAC(
            "MultiInputPolicy",
            train_env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            ),
            learning_rate=1e-3,
            buffer_size=100000,
            batch_size=256,
            gamma=0.98,
            tau=0.05,
            seed=seed,
            verbose=0,
        )

        t0 = time.time()
        model.learn(total_timesteps=total_timesteps)
        train_duration = time.time() - t0
        train_env.close()

        eval_res = evaluate_single_seed(model, n_episodes=n_eval_episodes, seed=seed * 100)
        eval_res["seed"] = seed
        eval_res["train_duration_sec"] = round(train_duration, 2)
        seed_results.append(eval_res)
        print(f"Seed {seed} complete in {train_duration:.1f}s: success_rate={eval_res['success_rate']*100:.1f}%, mean_steps={eval_res['mean_steps_to_goal']:.2f}")

        if save_models:
            model.save(f"models/checkpoint_fetch_reach_seed_{seed}")

    success_rates = [r["success_rate"] for r in seed_results]
    mean_steps = [r["mean_steps_to_goal"] for r in seed_results]

    summary = {
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "total_timesteps_per_seed": total_timesteps,
        "n_eval_episodes_per_seed": n_eval_episodes,
        "total_wallclock_sec": round(time.time() - start_time, 2),
        "success_rate": bootstrap_ci(success_rates),
        "mean_steps_to_goal": bootstrap_ci(mean_steps),
        "seed_breakdown": seed_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMulti-seed results saved to {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-seed SAC training on FetchReach-v4")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46], help="List of random seeds")
    parser.add_argument("--total-steps", type=int, default=4000, help="Timesteps per seed")
    parser.add_argument("--n-eval", type=int, default=100, help="Eval episodes per seed")
    parser.add_argument("--output", type=str, default="results/multiseed_reach_eval.json", help="Output path")
    args = parser.parse_args()

    train_multiseed(
        seeds=args.seeds,
        total_timesteps=args.total_steps,
        n_eval_episodes=args.n_eval,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

