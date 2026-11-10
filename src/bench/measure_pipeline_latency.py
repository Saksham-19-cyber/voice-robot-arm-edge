"""Pipeline latency and early termination benchmark."""












import os
import sys
import time
import json
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.abspath("."))
os.environ["MUJOCO_GL"] = "glfw"

from src.asr.transcriber import WhisperASR
from src.nlu.parser import VoiceCommandParser
from src.rl.env_utils import make_fetch_env, WORKSPACE_CENTER
from src.rl.rollout import run_onnx_rollout, HOLD_STEPS

def benchmark_pipeline_latency(n_runs: int = 20):
    audio_path = "results/sample_2s.wav"
    if not os.path.exists(audio_path):
        audio_path = "results/sample_speech.wav"
    
    print(f"Benchmarking End-to-End Pipeline Latency over {n_runs} runs...")
    

    print("\nMeasuring Cold Start (first request from scratch)...")
    t_cold_start = time.perf_counter()
    
    t0 = time.perf_counter()
    cold_asr = WhisperASR(model_size="tiny.en", compute_type="int8", cpu_threads=1)
    cold_asr_load_ms = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    cold_asr_res = cold_asr.transcribe(audio_path)
    cold_asr_inf_ms = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    cold_parser = VoiceCommandParser()
    cold_nlu_cmd = cold_parser.parse(cold_asr_res["text"])
    cold_nlu_ms = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    cold_env = make_fetch_env(render_mode="rgb_array", width=320, height=320)
    cold_env_init_ms = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    cold_session = ort.InferenceSession("models/fetch_reach_actor_int8.onnx", providers=["CPUExecutionProvider"])
    cold_ort_init_ms = (time.perf_counter() - t0) * 1000
    
    cold_rollout_res = run_onnx_rollout(
        goal_xyz=cold_nlu_cmd.goal_xyz if cold_nlu_cmd.goal_xyz is not None else WORKSPACE_CENTER,
        onnx_path="models/fetch_reach_actor_int8.onnx",
        early_termination=True,
        env=cold_env,
        session=cold_session,
    )
    cold_total_user_wait_ms = (time.perf_counter() - t_cold_start) * 1000
    
    cold_start_metrics = {
        "asr_model_load_ms": round(cold_asr_load_ms, 1),
        "asr_inference_ms": round(cold_asr_inf_ms, 1),
        "nlu_parse_ms": round(cold_nlu_ms, 2),
        "env_construction_ms": round(cold_env_init_ms, 1),
        "ort_session_init_ms": round(cold_ort_init_ms, 1),
        "rollout_and_render_ms": round(cold_rollout_res["rollout_time_ms"], 1),
        "video_encode_ms": round(cold_rollout_res["encoding_time_ms"], 1),
        "total_cold_start_ms": round(cold_total_user_wait_ms, 1),
    }
    print(f"Cold Start Total: {cold_total_user_wait_ms:.1f} ms")
    

    recreate_times = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        tmp_env = make_fetch_env(render_mode="rgb_array", width=320, height=320)
        tmp_env.reset()
        _ = tmp_env.render()
        recreate_times.append((time.perf_counter_ns() - t0) / 1e6)
        tmp_env.close()

    warm_reset_times = []
    # Warmup runs to stabilize JIT / caches
    for _ in range(10):
        cold_env.reset()
        _ = cold_env.render()

    # 100 warm resets back-to-back using high-precision nanosecond timer
    for _ in range(100):
        t0 = time.perf_counter_ns()
        cold_env.reset()
        _ = cold_env.render()
        warm_reset_times.append((time.perf_counter_ns() - t0) / 1e6)

    median_cold = float(np.median(recreate_times))
    median_warm = float(np.median(warm_reset_times))
    speedup_ratio_of_medians = median_cold / median_warm if median_warm > 0 else 0.0

    # Also benchmark pure reset without rendering for baseline comparison
    pure_recreate_times = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        p_env = make_fetch_env()
        p_env.reset()
        pure_recreate_times.append((time.perf_counter_ns() - t0) / 1e6)
        p_env.close()

    pure_warm_times = []
    p_env = make_fetch_env()
    for _ in range(10):
        p_env.reset()
    for _ in range(100):
        t0 = time.perf_counter_ns()
        p_env.reset()
        pure_warm_times.append((time.perf_counter_ns() - t0) / 1e6)
    p_env.close()

    pure_median_cold = float(np.median(pure_recreate_times))
    pure_median_warm = float(np.median(pure_warm_times))
    pure_speedup = pure_median_cold / pure_median_warm if pure_median_warm > 0 else 0.0

    env_reuse_metrics = {
        "env_recreate_with_first_render_mean_ms": float(np.mean(recreate_times)),
        "env_recreate_with_first_render_median_ms": median_cold,
        "env_recreate_with_first_render_min_ms": float(np.min(recreate_times)),
        "env_recreate_with_first_render_max_ms": float(np.max(recreate_times)),
        "env_warm_reset_with_render_mean_ms": float(np.mean(warm_reset_times)),
        "env_warm_reset_with_render_median_ms": median_warm,
        "env_warm_reset_with_render_min_ms": float(np.min(warm_reset_times)),
        "env_warm_reset_with_render_max_ms": float(np.max(warm_reset_times)),
        "speedup_factor": round(speedup_ratio_of_medians, 1),
        "speedup_ratio_of_medians": speedup_ratio_of_medians,
        "n_warm_resets": len(warm_reset_times),
        "n_recreate_runs": len(recreate_times),
        "pure_env_reset_without_render": {
            "cold_recreate_median_ms": pure_median_cold,
            "cold_recreate_min_ms": float(np.min(pure_recreate_times)),
            "cold_recreate_max_ms": float(np.max(pure_recreate_times)),
            "warm_reset_median_ms": pure_median_warm,
            "warm_reset_min_ms": float(np.min(pure_warm_times)),
            "warm_reset_max_ms": float(np.max(pure_warm_times)),
            "speedup_ratio_of_medians": pure_speedup,
            "n_warm_resets": len(pure_warm_times),
        },
        "note": "Measured with high-precision time.perf_counter_ns() over 100 warm resets. Speedup reported as ratio of medians (median_cold / median_warm); ratio varies by run.",
    }
    print(f"Env Recreate vs Warm Reset (with render): {median_cold:.1f} ms vs {median_warm:.1f} ms ({speedup_ratio_of_medians:.1f}x speedup)")
    print(f"Pure Env Recreate vs Warm Reset (no render): {pure_median_cold:.1f} ms vs {pure_median_warm:.1f} ms ({pure_speedup:.1f}x speedup)")


    shared_env = cold_env
    shared_session = cold_session
    shared_asr = cold_asr
    shared_parser = cold_parser

    results = {
        "cold_start": cold_start_metrics,
        "env_reuse_comparison": env_reuse_metrics,
    }

    for mode, early_term, max_steps in [("early_termination", True, 25), ("full_50_steps", False, 50)]:
        print(f"\nBenchmarking Warm Pipeline: {mode} ({n_runs} runs)...")
        asr_times = []
        nlu_times = []
        rollout_times = []
        encode_times = []
        total_user_wait_times = []
        steps_taken = []
        first_success_steps = []

        for i in range(n_runs):
            t_req_start = time.perf_counter()


            t0 = time.perf_counter()
            asr_out = shared_asr.transcribe(audio_path)
            t_asr = (time.perf_counter() - t0) * 1000


            t0 = time.perf_counter()
            nlu_out = shared_parser.parse(asr_out["text"])
            t_nlu = (time.perf_counter() - t0) * 1000

            offset = np.array([0.02 * ((i % 3) - 1), 0.03 * (((i + 1) % 3) - 1), 0.02 * (((i + 2) % 3) - 1)], dtype=np.float32)
            goal = (nlu_out.goal_xyz if nlu_out.goal_xyz is not None else WORKSPACE_CENTER) + offset


            res = run_onnx_rollout(
                goal_xyz=goal,
                onnx_path="models/fetch_reach_actor_int8.onnx",
                max_steps=max_steps,
                early_termination=early_term,
                env=shared_env,
                session=shared_session,
            )
            t_total_req = (time.perf_counter() - t_req_start) * 1000

            asr_times.append(t_asr)
            nlu_times.append(t_nlu)
            rollout_times.append(res["rollout_time_ms"])
            encode_times.append(res["encoding_time_ms"])
            total_user_wait_times.append(t_total_req)
            steps_taken.append(res["steps_taken"])
            first_success_steps.append(res["first_success_step"])

        results[mode] = {
            "n_runs": n_runs,
            "hold_steps_constant": HOLD_STEPS,
            "mean_steps_taken": round(float(np.mean(steps_taken)), 1),
            "mean_first_success_step": round(float(np.mean([s for s in first_success_steps if s is not None])), 1),
            "asr_inference_ms": {
                "mean": round(float(np.mean(asr_times)), 1),
                "p50": round(float(np.median(asr_times)), 1),
                "p95": round(float(np.percentile(asr_times, 95)), 1),
            },
            "nlu_parse_ms": {
                "mean": round(float(np.mean(nlu_times)), 2),
                "p50": round(float(np.median(nlu_times)), 2),
                "p95": round(float(np.percentile(nlu_times, 95)), 2),
            },
            "rollout_and_render_ms": {
                "mean": round(float(np.mean(rollout_times)), 1),
                "p50": round(float(np.median(rollout_times)), 1),
                "p95": round(float(np.percentile(rollout_times, 95)), 1),
            },
            "video_encode_ms": {
                "mean": round(float(np.mean(encode_times)), 1),
                "p50": round(float(np.median(encode_times)), 1),
                "p95": round(float(np.percentile(encode_times, 95)), 1),
            },
            "total_user_wait_ms": {
                "mean": round(float(np.mean(total_user_wait_times)), 1),
                "p50": round(float(np.median(total_user_wait_times)), 1),
                "p95": round(float(np.percentile(total_user_wait_times, 95)), 1),
            },
        }

    out_path = "results/pipeline_latency.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved updated pipeline latency comparison to {out_path}")
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    benchmark_pipeline_latency()
