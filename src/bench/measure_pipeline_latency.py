import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.abspath("."))
os.environ["MUJOCO_GL"] = "glfw"

from src.rl.rollout import run_onnx_rollout, HOLD_STEPS
from src.rl.env_utils import WORKSPACE_CENTER
from src.nlu.parser import VoiceCommandParser

def benchmark_pipeline_latency(n_runs: int = 20):
    print(f"Benchmarking pipeline latency over {n_runs} runs...")
    parser = VoiceCommandParser()
    target_text = "reach to the left"

    results = {}
    for mode, early_term, max_steps in [("early_termination", True, 25), ("full_50_steps", False, 50)]:
        print(f"\nEvaluating mode: {mode} (early_termination={early_term}, max_steps={max_steps}, HOLD_STEPS={HOLD_STEPS})...")
        rollout_times = []
        encode_times = []
        total_times = []
        steps_taken = []
        first_success_steps = []

        # Warmup
        run_onnx_rollout(WORKSPACE_CENTER, "models/fetch_reach_actor_int8.onnx", max_steps=max_steps, early_termination=early_term)

        for i in range(n_runs):
            # Target slightly offset to simulate varied goals
            offset = np.array([0.02 * ((i % 3) - 1), 0.03 * (((i + 1) % 3) - 1), 0.02 * (((i + 2) % 3) - 1)], dtype=np.float32)
            goal = WORKSPACE_CENTER + offset

            t0 = time.perf_counter()
            res = run_onnx_rollout(goal, "models/fetch_reach_actor_int8.onnx", max_steps=max_steps, early_termination=early_term)
            total_duration_ms = (time.perf_counter() - t0) * 1000

            rollout_times.append(res["rollout_time_ms"])
            encode_times.append(res["encoding_time_ms"])
            total_times.append(total_duration_ms)
            steps_taken.append(res["steps_taken"])
            first_success_steps.append(res["first_success_step"])

        results[mode] = {
            "n_runs": n_runs,
            "hold_steps_constant": HOLD_STEPS,
            "mean_steps_taken": round(float(np.mean(steps_taken)), 1),
            "mean_first_success_step": round(float(np.mean([s for s in first_success_steps if s is not None])), 1),
            "rollout_ms": {
                "mean": round(float(np.mean(rollout_times)), 1),
                "p50": round(float(np.median(rollout_times)), 1),
                "std": round(float(np.std(rollout_times)), 1),
            },
            "encode_ms": {
                "mean": round(float(np.mean(encode_times)), 1),
                "p50": round(float(np.median(encode_times)), 1),
                "std": round(float(np.std(encode_times)), 1),
            },
            "total_ms": {
                "mean": round(float(np.mean(total_times)), 1),
                "p50": round(float(np.median(total_times)), 1),
                "std": round(float(np.std(total_times)), 1),
            },
        }

    os.makedirs("results", exist_ok=True)
    out_path = "results/pipeline_latency.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved pipeline latency comparison to {out_path}")
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    benchmark_pipeline_latency()

