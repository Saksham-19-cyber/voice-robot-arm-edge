"""Exploratory SAC training on FetchPickAndPlace-v4."""




import os
import sys
import time
import json
import torch
import numpy as np
import gymnasium as gym
import gymnasium_robotics

torch.set_num_threads(min(4, torch.get_num_threads()))

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rl.env_utils import setup_headless_rendering


class StretchLoggerCallback(BaseCallback):
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
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                ep_success = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, dones, infos = self.eval_env.step(action)
                    done = dones[0]
                    if "is_success" in infos[0]:
                        ep_success = max(ep_success, float(infos[0]["is_success"]))
                successes.append(ep_success)

            mean_success = float(np.mean(successes))
            record = {
                "step": int(self.num_timesteps),
                "success_rate": round(mean_success, 4),
                "timestamp": round(time.time(), 2),
            }
            self.history.append(record)
            if self.verbose > 0:
                print(f"[Stretch Step {self.num_timesteps:5d}] Eval Success Rate: {mean_success * 100:5.1f}%", flush=True)

            ckpt_path = os.path.join(self.checkpoint_dir, "checkpoint_fetch_pick_sac")
            self.model.save(ckpt_path)
        return True


def run_stretch(env_id="FetchPickAndPlace-v4", total_timesteps=1000, seed=42):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    setup_headless_rendering()

    print(f"=== Running Stretch RL Exploration: {env_id} ===", flush=True)
    print(f"Budget: {total_timesteps} steps on CPU. Will record real numbers honestly.", flush=True)

    def make_train_env():
        return gym.make(env_id)

    train_vec = DummyVecEnv([make_train_env])
    train_env = VecNormalize(train_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    def make_eval_env():
        return gym.make(env_id)

    eval_vec = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    eval_env.obs_rms = train_env.obs_rms

    callback = StretchLoggerCallback(eval_env, eval_freq=500, n_eval_episodes=10)

    policy_kwargs = dict(net_arch=[256, 256])
    replay_buffer_kwargs = dict(n_sampled_goal=4, goal_selection_strategy="future")

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
    duration = time.time() - t0


    print("Evaluating stretch policy over 50 episodes...", flush=True)
    eval_env.obs_rms = train_env.obs_rms
    successes = []
    for _ in range(50):
        obs = eval_env.reset()
        done = False
        ep_success = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = eval_env.step(action)
            done = dones[0]
            if "is_success" in infos[0]:
                ep_success = max(ep_success, float(infos[0]["is_success"]))
        successes.append(ep_success)

    final_sr = float(np.mean(successes))
    result = {
        "env_id": env_id,
        "timesteps_budget": total_timesteps,
        "duration_sec": round(duration, 2),
        "final_success_rate": round(final_sr, 4),
        "note": "PickAndPlace requires 100k-500k steps for full grasping convergence. On a CPU time budget, low success is expected and reported honestly.",
        "history": callback.history,
    }

    with open("results/stretch_eval.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Stretch evaluation complete: {final_sr * 100:.1f}% success rate. Saved to results/stretch_eval.json", flush=True)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    run_stretch()
