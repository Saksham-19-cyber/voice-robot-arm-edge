"""ASR latency ablation benchmark."""










import os
import sys
import time
import json
import psutil
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

def prepare_2s_clip(source_path: str = "results/sample_speech.wav", target_path: str = "results/sample_2s.wav") -> str:

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source audio {source_path} not found.")
    data, sr = sf.read(source_path)
    target_samples = int(2.0 * sr)
    if len(data) < target_samples:
        padded = np.zeros(target_samples, dtype=data.dtype)
        padded[:len(data)] = data
    else:
        padded = data[:target_samples]
    sf.write(target_path, padded, sr)
    return target_path

def run_asr_ablation(n_warmup: int = 5, n_runs: int = 20):
    audio_path = prepare_2s_clip()
    print(f"Prepared 2.0s audio clip at {audio_path}")
    
    configs = [
        {"model": "tiny.en", "compute_type": "int8", "cpu_threads": 1},
        {"model": "tiny.en", "compute_type": "int8", "cpu_threads": 4},
        {"model": "tiny.en", "compute_type": "float32", "cpu_threads": 1},
        {"model": "tiny.en", "compute_type": "float32", "cpu_threads": 4},
        {"model": "base.en", "compute_type": "int8", "cpu_threads": 1},
        {"model": "base.en", "compute_type": "int8", "cpu_threads": 4},
        {"model": "base.en", "compute_type": "float32", "cpu_threads": 1},
        {"model": "base.en", "compute_type": "float32", "cpu_threads": 4},
    ]

    results = []
    proc = psutil.Process()

    print(f"Running ASR ablation ({len(configs)} configs, {n_warmup} warmup, {n_runs} timed runs each)...")

    for cfg in configs:
        name = f"{cfg['model']}_{cfg['compute_type']}_threads{cfg['cpu_threads']}"
        print(f"\n--- Testing config: {name} ---", flush=True)

        t_load_start = time.perf_counter()
        model = WhisperModel(
            cfg["model"],
            device="cpu",
            compute_type=cfg["compute_type"],
            cpu_threads=cfg["cpu_threads"],
        )
        load_time_ms = (time.perf_counter() - t_load_start) * 1000


        # Warm-up runs matching production VAD settings
        for _ in range(n_warmup):
            _ = list(model.transcribe(
                audio_path, beam_size=1, language="en", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=400)
            )[0])

        # Timed evaluation runs
        latencies = []
        transcript = ""
        for i in range(n_runs):
            t0 = time.perf_counter()
            segs, _ = model.transcribe(
                audio_path, beam_size=1, language="en", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=400)
            )
            text = " ".join([s.text.strip() for s in segs]).strip()
            lat_ms = (time.perf_counter() - t0) * 1000
            latencies.append(lat_ms)
            transcript = text

        rss_mb = proc.memory_info().rss / (1024 * 1024)

        entry = {
            "config": cfg,
            "config_name": name,
            "n_runs": n_runs,
            "audio_duration_s": 2.0,
            "load_time_ms": round(load_time_ms, 1),
            "latency_p50_ms": round(float(np.median(latencies)), 1),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 1),
            "latency_mean_ms": round(float(np.mean(latencies)), 1),
            "latency_std_ms": round(float(np.std(latencies)), 1),
            "peak_rss_mb": round(rss_mb, 1),
            "transcript": transcript,
        }
        print(f"[{name}] p50: {entry['latency_p50_ms']} ms | p95: {entry['latency_p95_ms']} ms | RSS: {entry['peak_rss_mb']} MB")
        results.append(entry)


        del model

    out_file = "results/asr_ablation.json"
    os.makedirs("results", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved ASR ablation results to {out_file}")
    return results

if __name__ == "__main__":
    run_asr_ablation()
