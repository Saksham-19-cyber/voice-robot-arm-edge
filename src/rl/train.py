"""
Training script for Fetch environments using SAC + HerReplayBuffer.
Logs success rate vs. timesteps, saves VecNormalize statistics,
generates training curves, and evaluates closed-loop performance over >=100 episodes.
"""

import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import gymnasium_robotics

# Optimize CPU thread allocation on hybrid P/E core architectures
torch.set_num_threads(min(4, torch.get_num_threads()))

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rl.env_utils import setup_headless_rendering, WORKSPACE_CENTER


class SuccessLoggerCallback(BaseCallback):
    """
    Logs evaluation success rate and episode lengths periodically during training.
    Also saves periodic checkpoints.
    """
    def __init__(self, eval_env, eval_freq=500, n_eval_episodes=10, checkpoint_dir="models", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.checkpoint_dir = checkpoint_dir
        self.history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            successes = []
            ep_lengths = []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                steps = 0
                ep_success = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, dones, infos = self.eval_env.step(action)
                    done = dones[0]
                    steps += 1
                    if "is_success" in infos[0]:
                        ep_success = max(ep_success, float(infos[0]["is_success"]))
                successes.append(ep_success)
                ep_lengths.append(steps)

            mean_success = float(np.mean(successes))
            mean_length = float(np.mean(ep_lengths))
            record = {
                "step": int(self.num_timesteps),
                "success_rate": round(mean_success, 4),
                "mean_ep_length": round(mean_length, 2),
                "timestamp": round(time.time(), 2),
            }
            self.history.append(record)
            if self.verbose > 0:
                print(f"[Step {self.num_timesteps:6d}] Eval Success Rate: {mean_success * 100:5.1f}% | Mean Ep Length: {mean_length:4.1f}", flush=True)

            # Checkpoint model and stats periodically
            ckpt_path = os.path.join(self.checkpoint_dir, "checkpoint_fetch_reach_sac")
            self.model.save(ckpt_path)
            eval_env_norm = self.eval_env
            if hasattr(eval_env_norm, "save"):
                eval_env_norm.save(os.path.join(self.checkpoint_dir, "checkpoint_vecnormalize.pkl"))
        return True


def evaluate_policy(model, env, n_episodes=100, seed=42):
    """
    Rigorous closed-loop evaluation over n_episodes with fixed seed.
    Returns success rate, steps-to-goal stats, and per-episode metrics.
    """
    successes = []
    steps_to_goal = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        step = 0
        succeeded = False
        first_success_step = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = dones[0]
            step += 1
            if infos[0].get("is_success", 0.0) == 1.0 and not succeeded:
                succeeded = True
                first_success_step = step

        successes.append(1.0 if succeeded else 0.0)
        steps_to_goal.append(first_success_step if succeeded else step)

    success_rate = float(np.mean(successes))
    mean_steps = float(np.mean(steps_to_goal))
    std_steps = float(np.std(steps_to_goal))

    return {
        "n_episodes": n_episodes,
        "seed": seed,
        "success_rate": round(success_rate, 4),
        "mean_steps_to_goal": round(mean_steps, 2),
        "std_steps_to_goal": round(std_steps, 2),
        "min_steps": int(np.min(steps_to_goal)),
        "max_steps": int(np.max(steps_to_goal)),
    }


def train_fetch_reach(total_timesteps=4000, seed=42, output_dir="models", results_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    setup_headless_rendering()

    env_id = "FetchReach-v4"
    print(f"=== Training SAC + HER on {env_id} ===", flush=True)
    print(f"Total timesteps budget: {total_timesteps} | Seed: {seed} | CPU Threads: {torch.get_num_threads()}", flush=True)

    def make_train_env():
        return gym.make(env_id)

    train_vec = DummyVecEnv([make_train_env])
    train_env = VecNormalize(train_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    def make_eval_env():
        return gym.make(env_id)

    eval_vec = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    eval_env.obs_rms = train_env.obs_rms

    callback = SuccessLoggerCallback(eval_env, eval_freq=500, n_eval_episodes=10, checkpoint_dir=output_dir)

    policy_kwargs = dict(net_arch=[256, 256])
    replay_buffer_kwargs = dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    )

    t0 = time.time()
    model = SAC(
        "MultiInputPolicy",
        train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=replay_buffer_kwargs,
        learning_rate=1e-3,
        buffer_size=50000,
        batch_size=256,
        gamma=0.95,
        tau=0.05,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)
    train_duration = time.time() - t0
    print(f"Training completed in {train_duration:.2f} seconds.", flush=True)

    # Sync latest normalization stats to eval env
    eval_env.obs_rms = train_env.obs_rms

    # Save final model and VecNormalize statistics
    model_path = os.path.join(output_dir, "fetch_reach_sac")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip", flush=True)

    vec_norm_path = os.path.join(output_dir, "fetch_reach_vecnormalize.pkl")
    train_env.save(vec_norm_path)
    print(f"VecNormalize stats saved to {vec_norm_path}", flush=True)

    # Save raw observation RMS stats as JSON for standalone ONNX inference
    rms_dict = {}
    for key in train_env.obs_rms.keys():
        rms_dict[f"{key}_mean"] = train_env.obs_rms[key].mean.tolist()
        rms_dict[f"{key}_var"] = train_env.obs_rms[key].var.tolist()
    rms_json_path = os.path.join(output_dir, "fetch_reach_obs_rms.json")
    with open(rms_json_path, "w") as f:
        json.dump(rms_dict, f, indent=2)
    print(f"Raw obs RMS exported to {rms_json_path}", flush=True)

    # Rigorous final evaluation over 100 episodes
    print("\n--- Running Final Evaluation (100 episodes, fixed seed=42) ---", flush=True)
    eval_metrics = evaluate_policy(model, eval_env, n_episodes=100, seed=42)
    eval_metrics["train_duration_sec"] = round(train_duration, 2)
    eval_metrics["total_timesteps"] = total_timesteps
    print(f"Evaluation Success Rate: {eval_metrics['success_rate'] * 100:.2f}%", flush=True)
    print(f"Mean Steps to Goal: {eval_metrics['mean_steps_to_goal']} +/- {eval_metrics['std_steps_to_goal']}", flush=True)

    eval_json_path = os.path.join(results_dir, "reach_eval.json")
    with open(eval_json_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    history_json_path = os.path.join(results_dir, "training_reach.json")
    with open(history_json_path, "w") as f:
        json.dump(callback.history, f, indent=2)

    # Generate and save plot
    steps = [h["step"] for h in callback.history]
    success_rates = [h["success_rate"] * 100 for h in callback.history]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(steps, success_rates, marker="o", color="#1f77b4", linewidth=2, label="Success Rate (%)")
    ax1.set_xlabel("Environment Steps", fontsize=12)
    ax1.set_ylabel("Success Rate (%)", color="#1f77b4", fontsize=12)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle="--", alpha=0.6)

    mean_lengths = [h["mean_ep_length"] for h in callback.history]
    ax2 = ax1.twinx()
    ax2.plot(steps, mean_lengths, marker="s", color="#ff7f0e", linestyle=":", linewidth=1.5, label="Mean Ep Length")
    ax2.set_ylabel("Mean Episode Steps", color="#ff7f0e", fontsize=12)
    ax2.set_ylim(0, 55)

    plt.title(f"FetchReach-v4 SAC+HER Training Curve (Final: {eval_metrics['success_rate']*100:.1f}%)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    curve_path = os.path.join(results_dir, "reach_training_curve.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Training curve plot saved to {curve_path}", flush=True)

    train_env.close()
    eval_env.close()
    return eval_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train FetchReach SAC+HER")
    parser.add_argument("--timesteps", type=int, default=4000, help="Total timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_fetch_reach(total_timesteps=args.timesteps, seed=args.seed)
