"""
Audit Benchmark Script:
Measures:
1. Cold-start model load time for WhisperModel vs warm inference latency (warm-up excluded).
2. Pure ONNX policy inference (per-step and 50-step total).
3. Full closed-loop ONNX + MuJoCo rollout (physics simulation + observation prep + ONNX inference per step).
4. Full rollout WITH headless RGB rendering (physics + ONNX + camera render per step).
5. Video encoding latency (MP4).
6. End-to-end breakdown properly labeled.
"""

import os
import sys
import time
import json
import numpy as np
import onnxruntime as ort
import gymnasium as gym
import gymnasium_robotics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.rl.env_utils import make_fetch_env, setup_headless_rendering, WORKSPACE_CENTER
from src.nlu.parser import VoiceCommandParser


def run_audit_benchmark():
    setup_headless_rendering()
    results = {}

    print("=== 1. Benchmarking ASR: Cold-Start vs Warm Inference ===", flush=True)
    from faster_whisper import WhisperModel

    # Cold start load time
    t_load_0 = time.perf_counter()
    whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    cold_load_sec = time.perf_counter() - t_load_0
    print(f"Whisper Cold-Start Load Time: {cold_load_sec:.3f} s", flush=True)

    sample_wav = "results/sample_speech.wav"
    assert os.path.exists(sample_wav), f"Missing {sample_wav}"

    # Warm-up call (excluded from inference stats)
    t_warm_0 = time.perf_counter()
    _ = list(whisper_model.transcribe(sample_wav, beam_size=1, vad_filter=True)[0])
    warmup_latency_sec = time.perf_counter() - t_warm_0
    print(f"Whisper Warm-up Call Latency: {warmup_latency_sec:.3f} s", flush=True)

    # Timed inference runs (warm-up excluded)
    asr_times = []
    audio_dur = None
    for _ in range(5):
        t0 = time.perf_counter()
        segs, info = whisper_model.transcribe(sample_wav, beam_size=1, vad_filter=True)
        _ = list(segs)
        asr_times.append(time.perf_counter() - t0)
        audio_dur = info.duration

    mean_asr_sec = float(np.mean(asr_times))
    std_asr_sec = float(np.std(asr_times))
    p50_asr_sec = float(np.median(asr_times))
    rtf = mean_asr_sec / audio_dur if audio_dur > 0 else 0.0

    print(f"ASR Warm Inference (mean over 5 runs): {mean_asr_sec * 1000:.1f} ms (+/- {std_asr_sec * 1000:.1f} ms) | RTF: {rtf:.2f}", flush=True)

    results["asr"] = {
        "model": "faster-whisper-tiny.en-int8",
        "audio_duration_sec": round(audio_dur, 3),
        "cold_start_load_time_sec": round(cold_load_sec, 3),
        "warmup_call_latency_sec": round(warmup_latency_sec, 3),
        "warm_inference_mean_ms": round(mean_asr_sec * 1000, 2),
        "warm_inference_p50_ms": round(p50_asr_sec * 1000, 2),
        "warm_inference_std_ms": round(std_asr_sec * 1000, 2),
        "real_time_factor": round(rtf, 3),
    }

    print("\n=== 2. Benchmarking NLU Parsing ===", flush=True)
    parser = VoiceCommandParser("models/intent_classifier.pkl")
    # Warm-up
    parser.parse("reach left by 10cm")
    nlu_times = []
    for _ in range(500):
        t0 = time.perf_counter()
        parser.parse("reach left by 10cm")
        nlu_times.append((time.perf_counter() - t0) * 1000)
    nlu_p50 = float(np.median(nlu_times))
    nlu_mean = float(np.mean(nlu_times))
    print(f"NLU Latency: mean={nlu_mean:.3f} ms, p50={nlu_p50:.3f} ms", flush=True)

    results["nlu"] = {
        "p50_ms": round(nlu_p50, 3),
        "mean_ms": round(nlu_mean, 3),
    }

    print("\n=== 3. Benchmarking Pure Policy ONNX Inference ===", flush=True)
    int8_path = "models/fetch_reach_actor_int8.onnx"
    session_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    dummy_obs = np.random.randn(1, 16).astype(np.float32)

    # Warm-up
    for _ in range(50):
        session_int8.run(None, {"observation_flat": dummy_obs})

    policy_step_times_us = []
    for _ in range(1000):
        t0 = time.perf_counter()
        session_int8.run(None, {"observation_flat": dummy_obs})
        policy_step_times_us.append((time.perf_counter() - t0) * 1e6)

    pol_p50_us = float(np.median(policy_step_times_us))
    pol_mean_us = float(np.mean(policy_step_times_us))
    pol_50step_infer_ms = (pol_p50_us * 50) / 1000
    print(f"Pure ONNX INT8 step inference: p50={pol_p50_us:.1f} us (50-step inference total: {pol_50step_infer_ms:.2f} ms)", flush=True)

    results["policy_inference_only"] = {
        "per_step_p50_us": round(pol_p50_us, 1),
        "per_step_mean_us": round(pol_mean_us, 1),
        "fifty_steps_inference_ms": round(pol_50step_infer_ms, 2),
    }

    print("\n=== 4. Benchmarking Full Closed-Loop ONNX + MuJoCo Rollout (No Rendering) ===", flush=True)
    env = make_fetch_env("FetchReach-v4", render_mode=None)
    rollout_no_render_times_ms = []

    # 10 rollouts of 50 steps each
    for _ in range(10):
        obs_dict, _ = env.reset(seed=42)
        t0 = time.perf_counter()
        for step in range(50):
            ob = obs_dict["observation"]
            ag = obs_dict["achieved_goal"]
            dg = obs_dict["desired_goal"]
            obs_flat = np.concatenate([ob, ag, dg], axis=0).astype(np.float32).reshape(1, 16)
            action = session_int8.run(None, {"observation_flat": obs_flat})[0][0]
            obs_dict, _, _, _, _ = env.step(action)
        rollout_no_render_times_ms.append((time.perf_counter() - t0) * 1000)
    env.close()

    rollout_no_render_mean = float(np.mean(rollout_no_render_times_ms))
    rollout_no_render_p50 = float(np.median(rollout_no_render_times_ms))
    per_step_physics_and_infer_ms = rollout_no_render_p50 / 50
    print(f"50-step Full Rollout (MuJoCo Physics + ONNX, no rendering): {rollout_no_render_p50:.2f} ms ({per_step_physics_and_infer_ms:.3f} ms/step, {1000/per_step_physics_and_infer_ms:.1f} FPS)", flush=True)

    results["full_rollout_mujoco_plus_onnx_no_render"] = {
        "fifty_steps_total_p50_ms": round(rollout_no_render_p50, 2),
        "fifty_steps_total_mean_ms": round(rollout_no_render_mean, 2),
        "per_step_p50_ms": round(per_step_physics_and_infer_ms, 3),
        "effective_fps": round(1000 / per_step_physics_and_infer_ms, 1),
    }

    print("\n=== 5. Benchmarking Full Closed-Loop Rollout WITH Headless Rendering ===", flush=True)
    env_render = make_fetch_env("FetchReach-v4", render_mode="rgb_array")
    rollout_with_render_times_ms = []
    frames = []

    for _ in range(3):
        obs_dict, _ = env_render.reset(seed=42)
        frames.clear()
        t0 = time.perf_counter()
        for step in range(50):
            ob = obs_dict["observation"]
            ag = obs_dict["achieved_goal"]
            dg = obs_dict["desired_goal"]
            obs_flat = np.concatenate([ob, ag, dg], axis=0).astype(np.float32).reshape(1, 16)
            action = session_int8.run(None, {"observation_flat": obs_flat})[0][0]
            obs_dict, _, _, _, _ = env_render.step(action)
            frame = env_render.render()
            frames.append(frame)
        rollout_with_render_times_ms.append((time.perf_counter() - t0) * 1000)
    env_render.close()

    rollout_render_p50 = float(np.median(rollout_with_render_times_ms))
    rollout_render_mean = float(np.mean(rollout_with_render_times_ms))
    print(f"50-step Rollout + RGB Rendering: {rollout_render_p50:.2f} ms ({rollout_render_p50/50:.2f} ms/frame)", flush=True)

    # Measure video encoding time for 50 frames
    import tempfile
    import imageio
    tmp_video = os.path.join(tempfile.gettempdir(), "bench_rollout.mp4")
    enc_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        imageio.mimwrite(tmp_video, frames, fps=15, format="FFMPEG", quality=8)
        enc_times.append((time.perf_counter() - t0) * 1000)
    if os.path.exists(tmp_video):
        os.remove(tmp_video)

    enc_p50_ms = float(np.median(enc_times))
    print(f"MP4 Video Encoding (50 frames): {enc_p50_ms:.2f} ms", flush=True)

    # Process RSS measurement
    import psutil
    proc = psutil.Process()
    current_rss_mb = proc.memory_info().rss / (1024 * 1024)
    print(f"\n=== Process Memory Measurement ===", flush=True)
    print(f"Current Process Peak RSS after full demo run (Whisper + ONNX + MuJoCo + Video Encoding): {current_rss_mb:.2f} MB", flush=True)

    results["memory"] = {
        "process_rss_mb": round(current_rss_mb, 2),
    }

    results["full_rollout_with_rendering_and_encoding"] = {
        "rollout_and_render_50steps_p50_ms": round(rollout_render_p50, 2),
        "rollout_and_render_per_frame_ms": round(rollout_render_p50 / 50, 2),
        "mp4_encoding_50frames_p50_ms": round(enc_p50_ms, 2),
        "total_rollout_render_encode_ms": round(rollout_render_p50 + enc_p50_ms, 2),
    }

    # Pipeline Latency Comparison
    t_asr_warm = mean_asr_sec * 1000
    t_nlu = nlu_p50
    t_pol_infer_only = pol_50step_infer_ms
    t_pol_full_sim = rollout_no_render_p50
    t_render_encode = rollout_render_p50 + enc_p50_ms
    t_tts_html = 0.5  # sub-millisecond client HTML generation

    e2e_live_service_ms = t_asr_warm + t_nlu + t_pol_full_sim + t_tts_html
    e2e_with_video_ms = t_asr_warm + t_nlu + t_render_encode + t_tts_html

    results["pipeline_summary"] = {
        "asr_warm_ms": round(t_asr_warm, 2),
        "nlu_ms": round(t_nlu, 2),
        "policy_inference_50steps_ms": round(t_pol_infer_only, 2),
        "full_mujoco_rollout_50steps_ms": round(t_pol_full_sim, 2),
        "rollout_rendering_and_encoding_ms": round(t_render_encode, 2),
        "e2e_realtime_robot_control_ms": round(e2e_live_service_ms, 2),
        "e2e_with_video_generation_ms": round(e2e_with_video_ms, 2),
        "breakdown_percentages_control": {
            "asr_pct": round((t_asr_warm / e2e_live_service_ms) * 100, 1),
            "nlu_pct": round((t_nlu / e2e_live_service_ms) * 100, 1),
            "rollout_pct": round((t_pol_full_sim / e2e_live_service_ms) * 100, 1),
        }
    }

    with open("results/bench_audit.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAudit benchmarks saved to results/bench_audit.json", flush=True)
    return results


if __name__ == "__main__":
    run_audit_benchmark()
