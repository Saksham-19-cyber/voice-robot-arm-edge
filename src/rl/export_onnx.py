"""
Export and Quantization pipeline for FetchReach SAC policy.
Bakes observation normalization directly into the PyTorch actor computation graph,
exports to ONNX FP32, verifies numerical tolerance over 1,000 random inputs,
quantizes to INT8 via onnxruntime, and benchmarks closed-loop evaluation over >=100 episodes.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
import gymnasium_robotics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rl.env_utils import setup_headless_rendering


class NormalizedActor(nn.Module):
    """
    Standalone deterministic SAC actor with baked-in observation normalization.
    Input: flattened observation tensor (batch_size, 16) arranged as:
           [observation (10), achieved_goal (3), desired_goal (3)]
    Output: deterministic continuous action tensor (batch_size, 4) in [-1.0, 1.0]
    """
    def __init__(self, actor_net, obs_mean, obs_var, clip_obs=10.0, epsilon=1e-8):
        super().__init__()
        # SB3 actor components
        self.latent_pi = actor_net.latent_pi
        self.mu = actor_net.mu
        self.clip_obs = float(clip_obs)
        self.epsilon = float(epsilon)

        # Buffers for normalization in extractor order: [achieved_goal (3), desired_goal (3), observation (10)]
        self.register_buffer("mean", torch.tensor(obs_mean, dtype=torch.float32).unsqueeze(0))
        self.register_buffer("var", torch.tensor(obs_var, dtype=torch.float32).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x has shape (batch_size, 16) arranged as:
        # x[:, 0:10]  -> observation (10)
        # x[:, 10:13] -> achieved_goal (3)
        # x[:, 13:16] -> desired_goal (3)
        # Reorder to match SB3 CombinedExtractor keys order: [achieved_goal, desired_goal, observation]
        obs = x[:, 0:10]
        ag = x[:, 10:13]
        dg = x[:, 13:16]
        reordered = torch.cat([ag, dg, obs], dim=1)

        # Observation normalization with clipping
        norm_x = torch.clamp((reordered - self.mean) / torch.sqrt(self.var + self.epsilon), -self.clip_obs, self.clip_obs)

        # Actor MLP forward pass
        latent = self.latent_pi(norm_x)
        mean_actions = self.mu(latent)
        actions = torch.tanh(mean_actions)
        return actions


def export_and_quantize(
    model_path="models/fetch_reach_sac.zip",
    rms_path="models/fetch_reach_obs_rms.json",
    fp32_onnx_path="models/fetch_reach_actor_fp32.onnx",
    int8_onnx_path="models/fetch_reach_actor_int8.onnx",
    results_path="results/quantization_eval.json",
):
    os.makedirs(os.path.dirname(fp32_onnx_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    print("=== Step 1: Loading PyTorch SAC Actor & Normalization Stats ===")
    dummy_env = gym.make("FetchReach-v4")
    sac_model = SAC.load(model_path, env=dummy_env, device="cpu")
    actor = sac_model.policy.actor
    dummy_env.close()

    with open(rms_path, "r") as f:
        rms_data = json.load(f)

    # Order expected by CombinedExtractor: achieved_goal (3), desired_goal (3), observation (10)
    full_mean = (
        rms_data["achieved_goal_mean"]
        + rms_data["desired_goal_mean"]
        + rms_data["observation_mean"]
    )
    full_var = (
        rms_data["achieved_goal_var"]
        + rms_data["desired_goal_var"]
        + rms_data["observation_var"]
    )

    norm_actor = NormalizedActor(actor, full_mean, full_var)
    norm_actor.eval()

    print("=== Step 2: Exporting to ONNX FP32 ===")
    dummy_input = torch.randn(1, 16, dtype=torch.float32)

    torch.onnx.export(
        norm_actor,
        dummy_input,
        fp32_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
        input_names=["observation_flat"],
        output_names=["action"],
        dynamic_axes={"observation_flat": {0: "batch_size"}, "action": {0: "batch_size"}},
    )
    print(f"Exported ONNX FP32 model to {fp32_onnx_path} (Size: {os.path.getsize(fp32_onnx_path) / 1024:.1f} KB)")

    # Verify ONNX model integrity
    onnx_model = onnx.load(fp32_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure verified successfully.")

    print("\n=== Step 3: Verifying Numerical Consistency (1,000 random samples) ===")
    ort_session_fp32 = ort.InferenceSession(fp32_onnx_path, providers=["CPUExecutionProvider"])

    np.random.seed(42)
    random_inputs = np.random.randn(1000, 16).astype(np.float32)

    with torch.no_grad():
        torch_outputs = norm_actor(torch.from_numpy(random_inputs)).numpy()

    onnx_outputs = ort_session_fp32.run(None, {"observation_flat": random_inputs})[0]

    max_abs_diff = float(np.max(np.abs(torch_outputs - onnx_outputs)))
    mean_abs_diff = float(np.mean(np.abs(torch_outputs - onnx_outputs)))
    print(f"PyTorch vs ONNX FP32 Max Absolute Diff:  {max_abs_diff:.6e}")
    print(f"PyTorch vs ONNX FP32 Mean Absolute Diff: {mean_abs_diff:.6e}")
    assert max_abs_diff < 1e-4, f"Tolerance exceeded: {max_abs_diff}"
    print("Numerical consistency verified within tolerance (< 1e-4)!")

    print("\n=== Step 4: INT8 Dynamic Quantization ===")
    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QInt8,
    )
    fp32_size = os.path.getsize(fp32_onnx_path)
    int8_size = os.path.getsize(int8_onnx_path)
    print(f"Exported ONNX INT8 model to {int8_onnx_path}")
    print(f"FP32 Size: {fp32_size / 1024:.1f} KB | INT8 Size: {int8_size / 1024:.1f} KB (Compression: {fp32_size / int8_size:.2f}x)")

    ort_session_int8 = ort.InferenceSession(int8_onnx_path, providers=["CPUExecutionProvider"])
    onnx_int8_outputs = ort_session_int8.run(None, {"observation_flat": random_inputs})[0]
    int8_fp32_diff = float(np.max(np.abs(onnx_outputs - onnx_int8_outputs)))
    print(f"FP32 vs INT8 Output Max Absolute Diff on random inputs: {int8_fp32_diff:.4f}")

    print("\n=== Step 5: Closed-Loop Task Evaluation (100 episodes, fixed seed=42) ===")
    eval_results = run_closed_loop_comparison(
        norm_actor,
        ort_session_fp32,
        ort_session_int8,
        n_episodes=100,
        seed=42,
    )

    eval_results["fp32_size_bytes"] = fp32_size
    eval_results["int8_size_bytes"] = int8_size
    eval_results["max_abs_error_torch_fp32"] = max_abs_diff
    eval_results["max_abs_error_fp32_int8"] = int8_fp32_diff

    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved quantization evaluation report to {results_path}")
    return eval_results


def run_closed_loop_comparison(torch_actor, ort_fp32, ort_int8, n_episodes=100, seed=42):
    setup_headless_rendering()

    def make_eval_env():
        return gym.make("FetchReach-v4")

    # Closed-loop evaluation runners for PyTorch, ONNX FP32, and ONNX INT8
    # Input flattening order: [observation (10), achieved_goal (3), desired_goal (3)]
    policies = {
        "PyTorch_FP32": lambda obs: torch_actor(torch.from_numpy(obs).float()).detach().numpy(),
        "ONNX_FP32": lambda obs: ort_fp32.run(None, {"observation_flat": obs})[0],
        "ONNX_INT8": lambda obs: ort_int8.run(None, {"observation_flat": obs})[0],
    }

    results = {}

    for name, policy_fn in policies.items():
        env = make_eval_env()
        successes = []
        steps_list = []
        t0 = time.time()

        for ep in range(n_episodes):
            obs_dict, _ = env.reset(seed=seed + ep)
            done = False
            step = 0
            succeeded = False
            first_success_step = None

            while not done:
                # Format raw un-normalized obs as [observation (10), achieved_goal (3), desired_goal (3)]
                ob = obs_dict["observation"]
                ag = obs_dict["achieved_goal"]
                dg = obs_dict["desired_goal"]
                obs_flat = np.concatenate([ob, ag, dg], axis=0).astype(np.float32).reshape(1, 16)

                action = policy_fn(obs_flat)[0]

                obs_dict, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

                if info.get("is_success", 0.0) == 1.0 and not succeeded:
                    succeeded = True
                    first_success_step = step

            successes.append(1.0 if succeeded else 0.0)
            steps_list.append(first_success_step if succeeded else step)

        dt = time.time() - t0
        sr = float(np.mean(successes))
        mean_steps = float(np.mean(steps_list))
        std_steps = float(np.std(steps_list))

        results[name] = {
            "success_rate": round(sr, 4),
            "mean_steps_to_goal": round(mean_steps, 2),
            "std_steps_to_goal": round(std_steps, 2),
            "total_eval_time_sec": round(dt, 2),
            "fps": round((n_episodes * 50) / dt, 1),
        }
        print(f"[{name:12s}] Success Rate: {sr * 100:5.1f}% | Steps to Goal: {mean_steps:.2f} +/- {std_steps:.2f} | Time: {dt:.2f}s")
        env.close()

    return results


if __name__ == "__main__":
    export_and_quantize()
