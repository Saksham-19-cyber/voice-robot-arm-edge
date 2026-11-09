"""
End-to-End Pipeline Latency Benchmark:
Measures end-to-end user wait time:
1. Cold start (first request without cached env/models)
2. Warm start with env reuse (session_state cached):
   - ASR transcription (faster-whisper tiny.en, int8)
   - NLU intent & coordinate extraction
   - ONNX policy rollout + MuJoCo physics step + 320x320 RGB rendering
   - MP4 video encoding (H.264 yuv420p + faststart)
3. Compares early termination (first success + 10 hold steps) vs full 50 steps.
4. Quantifies env recreate vs warm reset latency.
"""

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
    
    # --- 1. Cold Start Measurement ---
    print("\nMeasuring Cold Start (first request from scratch)...")
    t_cold_start = time.perf_counter()
    
    t0 = time.perf_counter()
    cold_asr = WhisperASR(model_size="tiny.en", compute_type="int8")
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
    
    # --- 2. Measure Env Recreate vs Warm Reset ---
    recreate_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        tmp_env = make_fetch_env(render_mode="rgb_array", width=320, height=320)
        tmp_env.reset()
        _ = tmp_env.render()
        recreate_times.append((time.perf_counter() - t0) * 1000)
        tmp_env.close()
        
    warm_reset_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        cold_env.reset()
        _ = cold_env.render()
        warm_reset_times.append((time.perf_counter() - t0) * 1000)
        
    env_reuse_metrics = {
        "env_recreate_with_first_render_mean_ms": round(float(np.mean(recreate_times)), 1),
        "env_warm_reset_with_render_mean_ms": round(float(np.mean(warm_reset_times)), 1),
        "speedup_factor": round(float(np.mean(recreate_times)) / float(np.mean(warm_reset_times)), 1),
    }
    print(f"Env Recreate vs Warm Reset: {env_reuse_metrics['env_recreate_with_first_render_mean_ms']} ms vs {env_reuse_metrics['env_warm_reset_with_render_mean_ms']} ms ({env_reuse_metrics['speedup_factor']}x faster)")

    # --- 3. Warm Request Pipeline Benchmarks (>=20 runs) ---
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

            # ASR
            t0 = time.perf_counter()
            asr_out = shared_asr.transcribe(audio_path)
            t_asr = (time.perf_counter() - t0) * 1000

            # NLU
            t0 = time.perf_counter()
            nlu_out = shared_parser.parse(asr_out["text"])
            t_nlu = (time.perf_counter() - t0) * 1000

            offset = np.array([0.02 * ((i % 3) - 1), 0.03 * (((i + 1) % 3) - 1), 0.02 * (((i + 2) % 3) - 1)], dtype=np.float32)
            goal = (nlu_out.goal_xyz if nlu_out.goal_xyz is not None else WORKSPACE_CENTER) + offset

            # Rollout + Render + Encode (warm env & session reused)
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
